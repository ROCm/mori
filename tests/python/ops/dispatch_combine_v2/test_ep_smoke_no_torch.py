# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""EPv2 intranode dispatch + combine without torch.

For GPU targets torch is not built for (gfx1250-strict). Plans come from
mori.jit.v2.plan_api, memory from HIP via ctypes, references from numpy; the
buffers mirror hip_backend's intranode setup. Identity expert throughout.

  bf16  dispatch exact; gather combine == bf16(U * x), weights == U * w
  fp8   dispatch + scale rows exact; bf16 combine of the dequantized rows exact
  fp4   dispatch exact; fp4_blockwise push combine == the mx reference (gfx125x)

    python3 test_ep_smoke_no_torch.py [--world N] [--cases bf16,fp8,fp4]
"""

import argparse
import ctypes
import hashlib
import multiprocessing as mp
import os
import queue
import traceback

import numpy as np

M = 256  # max tokens per rank
H = 7168
K = 8
EPR = 32
SWEEP = (256, 77, 1)
CASES = ("bf16", "fp8", "fp4")
SCALE_I32 = H // 128  # one fp32 scale per 128 elements
TIMEOUT_S = 1800  # without progress
POLL_S = 5

_H2D, _D2H = 1, 2
_c = ctypes


class _Hip:
    """The few HIP runtime calls this test needs."""

    def __init__(self):
        from mori.jit import hip_driver

        self._lib = hip_driver._get_hip_lib()
        self._check = hip_driver._check
        self._ptrs = []

    def call(self, name, *args):
        self._check(getattr(self._lib, name)(*args), name)

    def device_count(self):
        n = _c.c_int()
        self.call("hipGetDeviceCount", _c.byref(n))
        return n.value

    def set_device(self, dev):
        self.call("hipSetDevice", _c.c_int(dev))

    def malloc(self, nbytes):
        p = _c.c_void_p()
        self.call("hipMalloc", _c.byref(p), _c.c_size_t(max(nbytes, 1)))
        self._ptrs.append(p.value)
        return p.value

    def upload(self, arr):
        arr = np.ascontiguousarray(arr)
        ptr = self.malloc(arr.nbytes)
        self.h2d(ptr, arr)
        return ptr

    def h2d(self, dst, arr):
        arr = np.ascontiguousarray(arr)
        self.call(
            "hipMemcpy",
            _c.c_void_p(dst),
            arr.ctypes.data_as(_c.c_void_p),
            _c.c_size_t(arr.nbytes),
            _c.c_int(_H2D),
        )

    def d2h(self, src, dtype, shape):
        out = np.empty(shape, dtype)
        self.call(
            "hipMemcpy",
            out.ctypes.data_as(_c.c_void_p),
            _c.c_void_p(src),
            _c.c_size_t(out.nbytes),
            _c.c_int(_D2H),
        )
        return out

    def memset(self, ptr, value, nbytes):
        self.call("hipMemset", _c.c_void_p(ptr), _c.c_int(value), _c.c_size_t(nbytes))

    def sync(self):
        self.call("hipDeviceSynchronize")

    def free_all(self):
        for p in self._ptrs:
            self._lib.hipFree(_c.c_void_p(p))
        self._ptrs.clear()


class _Arena:
    """SymmArena's layout (256 B-aligned regions in one cco window), no torch."""

    _ALIGN = 256

    def __init__(self, comm, regions):
        self._off, self._size = {}, {}
        off = 0
        for name, nbytes in regions:
            off = -(-off // self._ALIGN) * self._ALIGN
            self._off[name], self._size[name] = off, nbytes
            off += nbytes
        self.total = max(-(-off // self._ALIGN) * self._ALIGN, self._ALIGN)
        self._mem = comm.alloc_mem(self.total)
        self._win = comm.register_window(self._mem.ptr, self.total)

    @property
    def handle(self):
        return self._win.handle

    def has(self, name):
        return name in self._off

    def offset(self, name):
        return self._off[name]

    def size(self, name):
        return self._size[name]

    def ptr(self, name=None):
        return self._win.local_ptr + (self._off[name] if name else 0)

    def close(self):
        self._win.close()
        self._mem.close()


# -- reference data -----------------------------------------------------------


def _bf16(x):
    """f32 -> bf16 bits, round to nearest even."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _e4m3(b):
    """OCP e4m3fn bytes -> f32 (NaN never generated)."""
    b = b.astype(np.int32)
    e, m = (b >> 3) & 0xF, (b & 7).astype(np.float64)
    v = np.where(e == 0, m / 8 * 2.0**-6, (1 + m / 8) * np.exp2(e - 7.0))
    return np.where(b & 0x80, -v, v).astype(np.float32)


_FP4_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)


def _fp4_rne(v):
    v = np.minimum(v, 6.0)
    i = np.clip(np.searchsorted(_FP4_GRID, v), 1, 7)
    lo, hi = _FP4_GRID[i - 1], _FP4_GRID[i]
    up = (hi - v < v - lo) | ((hi - v == v - lo) & (i % 2 == 0))
    return np.where(up, hi, lo)


def _push_fp4_ref(rows, u):
    """bf16(U * mx-dequant(x)) per 32-element group, as the fp4_blockwise combine
    quantizes (bench_ep.push_quant_deq). rows: bf16 bits [n, H]."""
    n = rows.shape[0]
    g = rows.reshape(n, H // 32, 32)
    x = _f32(g)
    m15 = (g & 0x7FFF).astype(np.int32).max(axis=-1, keepdims=True)
    sc = np.exp2(np.maximum(((m15 + 0x3F) >> 7) - 2, 1) - 127.0).astype(np.float32)
    deq = np.sign(x) * _fp4_rne(np.abs(x) / sc) * sc
    return _bf16(u.reshape(n, 1, 1) * deq).reshape(n, H)


def _inputs(case, src, world):
    """Rank src's routing and payload; any rank can rebuild any other's."""
    rng = np.random.default_rng(1234 + src)
    idx = np.argsort(rng.random((M, world * EPR)), axis=1)[:, :K].astype(np.int32)
    # k/1024: sums of up to K copies stay exact in f32.
    wts = (rng.integers(1, 1024, (M, K)) / 1024).astype(np.float32)
    scales = None
    if case == "bf16":
        tok = _bf16(rng.standard_normal((M, H), dtype=np.float32))
    elif case == "fp8":
        tok = rng.integers(0, 256, (M, H), dtype=np.uint8)
        tok[(tok & 0x7F) == 0x7F] -= 1  # no NaN
        scales = (
            src * 1000003 + np.arange(M)[:, None] * 64 + np.arange(SCALE_I32)
        ).astype(np.int32)
    else:
        tok = rng.integers(0, 256, (M, H // 2), dtype=np.uint8)
    return idx, wts, tok, scales


def _expert_rows(case, src, tok):
    """The bf16 rows the identity expert hands to combine for rank src's tokens."""
    if case == "bf16":
        return tok
    if case == "fp8":
        return _bf16(_e4m3(tok))  # exact: e4m3 fits in bf16
    rng = np.random.default_rng(4321 + src)
    return _bf16(rng.standard_normal((M, H), dtype=np.float32))


# -- one rank -----------------------------------------------------------------


def _run_case(case, rank, world, comm, hip, plans_api, log):
    plan_api, Disp, Comb = plans_api
    data = [_inputs(case, s, world) for s in range(world)]
    idx_all = np.stack([d[0] for d in data])
    wts_all = np.stack([d[1] for d in data])
    tok_all = np.stack([d[2] for d in data])
    rows_all = np.stack([_expert_rows(case, s, data[s][2]) for s in range(world)])
    scales_all = np.stack([d[3] for d in data]) if case == "fp8" else None
    idx, wts, tok, scales = data[rank]

    max_recv = world * M
    common = dict(
        world_size=world,
        max_tok_per_rank=M,
        num_expert_per_rank=EPR,
        num_expert_per_token=K,
        max_recv=max_recv,
        use_weights=True,
    )
    disp_req = dict(
        common,
        hidden_dim=H // 2 if case == "fp4" else H,
        dtype={"bf16": "bf16", "fp8": "fp8", "fp4": "fp4x2"}[case],
        scale_bytes=SCALE_I32 * 4 if case == "fp8" else 0,
        tok_off_ext=0,
    )
    comb_req = dict(common, hidden_dim=H, dtype="bf16")
    push = case == "fp4"
    if push:
        comb_req["combine_fp4"] = True

    disp_info = Disp.describe(**disp_req)
    is1250 = disp_info["stagingBytes"] > 0
    if push and not is1250:
        return {"skip": "fp4_blockwise push combine is gfx125x only"}
    comb_info = Comb.describe(**comb_req)
    arena = _Arena(comm, plan_api.arena_regions(disp_info, comb_info))
    hip.memset(arena.ptr(), 0, arena.total)
    hip.sync()
    comm.barrier()

    staging = hip.malloc(disp_info["stagingBytes"]) if is1250 else 0
    if staging:
        hip.memset(staging, 0, disp_info["stagingBytes"])
    disp = Disp(**disp_req, arena=arena)
    disp.bind(rank=rank)
    if staging:
        disp.bind(staging_base=staging)
    comb = Comb(**comb_req, arena=arena)
    comb.bind(rank=rank)
    if rank == 0:
        # cacheDir is <root>/<arch>_<nic>/<entry>: the arch the JIT built for.
        jit = os.path.basename(os.path.dirname(disp.info.get("cacheDir", "?/?")))
        log(f"[{case}] jit={jit} waveSize={disp.info.get('waveSize')}")

    i32 = np.int32
    d_idx, d_wts, d_tok = hip.upload(idx), hip.upload(wts), hip.upload(tok)
    d_scales = hip.upload(scales) if scales is not None else 0
    dest_map = hip.upload(np.full(M * K, world * max_recv, i32))
    dest_pe_counter = hip.upload(np.zeros(2 * world, i32))
    total_recv = hip.upload(np.zeros(1, i32))
    disp_barrier = hip.upload(np.zeros(1, np.uint32))
    comb_barrier = hip.upload(np.zeros(1, np.uint32))
    xdb_flag = hip.upload(np.ones(256 if is1250 else 1, np.int64))
    fan = hip.upload(np.zeros(comb.info["blockNum"] * 16, i32)) if is1250 else 0
    comb_out = hip.malloc(M * H * 2)
    comb_out_w = hip.malloc(M * K * 4)

    row_bytes = tok.shape[1] * tok.dtype.itemsize
    # 0xFF in every output a step must write, so one that writes nothing fails.
    # Peers write this rank's regions, hence the barrier before dispatch.
    disp_out = ["dispOut", "outIdx", "outWts", "recvToSrc"]
    disp_out += ["outScales"] if scales_all is not None else []
    results = {}
    for ct in SWEEP:
        for name in disp_out:
            hip.memset(arena.ptr(name), 0xFF, arena.size(name))
        hip.memset(comb_out, 0xFF, M * H * 2)  # bf16 NaN
        hip.memset(comb_out_w, 0xFF, M * K * 4)  # f32 NaN
        hip.sync()
        comm.barrier()
        disp.launch(
            stream=0,
            token_indices=d_idx,
            inp_token_buf=d_tok,
            weights_buf=d_wts,
            scales_buf=d_scales,
            disp_dest_tok_id_map=dest_map,
            dest_pe_token_counter=dest_pe_counter,
            tok_off_peers=0,
            total_recv_token_num=total_recv,
            grid_barrier=disp_barrier,
            num_tokens=ct,
        )
        hip.sync()
        comm.barrier()

        # Dispatch: rank s's token t lands here once iff one of its experts does.
        total = int(hip.d2h(total_recv, i32, 1)[0])
        exp_keys = np.sort(
            np.concatenate(
                [
                    s * M + np.nonzero((idx_all[s, :ct] // EPR == rank).any(axis=1))[0]
                    for s in range(world)
                ]
            )
        )
        rev = hip.d2h(arena.ptr("recvToSrc"), i32, total)
        ok_map = total == len(exp_keys) and np.array_equal(np.sort(rev), exp_keys)
        ok_disp = ok_map
        if ok_map:
            s, t = rev // M, rev % M
            got = hip.d2h(arena.ptr("dispOut"), np.uint8, (total, row_bytes))
            ok_disp &= np.array_equal(got, tok_all[s, t].view(np.uint8))
            got_i = hip.d2h(arena.ptr("outIdx"), i32, (total, K))
            ok_disp &= np.array_equal(got_i, idx_all[s, t])
            got_w = hip.d2h(arena.ptr("outWts"), np.float32, (total, K))
            ok_disp &= np.array_equal(got_w, wts_all[s, t])
            if scales_all is not None:
                stride = disp.info["scaleStride"]
                got_s = hip.d2h(arena.ptr("outScales"), np.uint8, (total, stride))
                got_s = np.ascontiguousarray(got_s[:, : SCALE_I32 * 4]).view(i32)
                ok_disp &= np.array_equal(got_s, scales_all[s, t])

        # Combine: the expert rows go where the kernel stages them (out_tok), except
        # bf16, whose recv rows are its input -- that also runs the in-kernel copy.
        if case == "bf16":
            comb_in = arena.ptr("dispOut")
        else:
            comb_in = arena.ptr("outTok")
            if ok_map:
                hip.h2d(comb_in, rows_all[rev // M, rev % M])
        hip.sync()
        comm.barrier()
        comb.launch(
            stream=0,
            inp_token_buf=comb_in,
            out_token_buf=comb_out,
            out_weights_buf=0 if push else comb_out_w,
            disp_dest_tok_id_map=dest_map,
            dest_pe_token_counter=dest_pe_counter,
            total_recv_token_num=total_recv,
            grid_barrier=comb_barrier,
            xdb_flag=xdb_flag,
            combine_barrier_fan=fan,
            num_tokens=ct,
        )
        hip.sync()
        comm.barrier()

        u = np.array(
            [len(set((idx[t] // EPR).tolist())) for t in range(ct)], np.float32
        )
        out = hip.d2h(comb_out, np.uint16, (ct, H))
        if push:
            ref = _push_fp4_ref(rows_all[rank, :ct], u)
            ok_w = True
        else:
            ref = _bf16(u[:, None] * _f32(rows_all[rank, :ct]))
            out_w = hip.d2h(comb_out_w, np.float32, (ct, K))
            ok_w = np.array_equal(out_w, u[:, None] * wts[:ct])
        # By value: the kernel's sum turns a -0 input into +0.
        bad_rows = int((_f32(out) != _f32(ref)).any(axis=1).sum())
        results[ct] = dict(
            recv=total,
            dispatch=bool(ok_disp),
            combine=bad_rows == 0,
            weights=bool(ok_w),
            bad_rows=bad_rows,
            digest=hashlib.sha256(out.tobytes()).hexdigest()[:16],
        )

    disp.close()
    comb.close()
    hip.free_all()
    comm.barrier()
    arena.close()
    return results


def _worker(rank, world, uid_bytes, cases, out_q):
    comm = None
    try:
        from mori.cco import Communicator, UniqueId

        hip = _Hip()
        hip.set_device(rank)
        from mori.jit.v2 import plan_api

        lib_dir = os.environ.get("MORI_V2_LIB_DIR")
        plan_api.load_library(
            "libmori_ops_v2.so", extra_dirs=[lib_dir] if lib_dir else []
        )
        plans = (
            plan_api,
            plan_api.make_plan("ep_dispatch"),
            plan_api.make_plan("ep_combine"),
        )

        def log(msg):
            out_q.put(("log", rank, msg))

        comm = Communicator.init(
            world, rank, UniqueId.from_bytes(uid_bytes), per_rank_vmm=1 << 30
        )
        for case in cases:
            res = _run_case(case, rank, world, comm, hip, plans, log)
            out_q.put(("result", rank, case, res))
        out_q.put(("done", rank))
    except BaseException:
        # Out before teardown: a failed destroy can exit the process.
        out_q.put(("error", rank, traceback.format_exc()))
        out_q.close()
        out_q.join_thread()
    finally:
        if comm is not None:
            comm.destroy()


def run(world=None, cases=CASES):
    """Spawn one rank per GPU; returns {case: {ct: [per-rank result]}}."""
    from mori.cco import Communicator

    os.environ.setdefault("MORI_SOCKET_IFNAME", "lo")
    world = world or _Hip().device_count()
    print(f"# EPv2 smoke (no torch): world={world}", flush=True)
    uid = bytes(Communicator.get_unique_id())
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(r, world, uid, tuple(cases), out_q))
        for r in range(world)
    ]
    for p in procs:
        p.start()

    results, errors, finished, idle = {}, [], set(), 0
    try:
        while len(finished) < world:
            try:
                msg = out_q.get(timeout=POLL_S)
            except queue.Empty:
                idle += POLL_S
                gone = [
                    r
                    for r, p in enumerate(procs)
                    if r not in finished and not p.is_alive()
                ]
                if gone:  # died without a word: a crash, not a slow rank
                    codes = [procs[r].exitcode for r in gone]
                    errors.append(
                        f"ranks {gone} exited with {codes} and reported nothing"
                    )
                    break
                if idle >= TIMEOUT_S:
                    errors.append(
                        f"no progress for {TIMEOUT_S}s (ranks unfinished: "
                        f"{sorted(set(range(world)) - finished)})"
                    )
                    break
                continue
            idle = 0
            kind, rank = msg[0], msg[1]
            if kind == "log":
                print(f"# {msg[2]}", flush=True)
            elif kind == "result":
                results.setdefault(msg[2], {})[rank] = msg[3]
            else:
                finished.add(rank)
                if kind == "error":
                    errors.append(f"rank {rank}:\n{msg[2]}")
    finally:
        for p in procs:
            p.join(timeout=5 if errors else 60)  # peers of a dead rank wait forever
            if p.is_alive():
                p.kill()
    if errors:
        raise RuntimeError("\n".join(errors))

    failed = []
    for case in cases:
        per_rank = results.get(case, {})
        if any("skip" in r for r in per_rank.values()):
            print(
                f"# {case}: SKIP ({next(iter(per_rank.values()))['skip']})", flush=True
            )
            continue
        for ct in SWEEP:
            rs = [per_rank[r][ct] for r in range(world)]
            ok = all(r["dispatch"] and r["combine"] and r["weights"] for r in rs)
            digest = hashlib.sha256(
                "".join(r["digest"] for r in rs).encode()
            ).hexdigest()[:16]
            print(
                f"# {case} ct={ct}: {'PASS' if ok else 'FAIL'} "
                f"recv={[r['recv'] for r in rs]} "
                f"dispatch={'ok' if all(r['dispatch'] for r in rs) else 'BAD'} "
                f"combine={'ok' if all(r['combine'] for r in rs) else 'BAD'} "
                f"(bad rows {[r['bad_rows'] for r in rs]}) "
                f"weights={'ok' if all(r['weights'] for r in rs) else 'BAD'} "
                f"digest={digest}",
                flush=True,
            )
            if not ok:
                failed.append(f"{case} ct={ct}")
    return failed


def test_ep_smoke_no_torch():
    import pytest

    n = _Hip().device_count()
    if n < 2:
        pytest.skip(f"needs >= 2 GPUs, have {n}")
    assert not run(n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--world", type=int, default=None, help="ranks (default: all GPUs)")
    ap.add_argument("--cases", default=",".join(CASES))
    a = ap.parse_args()
    failed = run(a.world, [c for c in a.cases.split(",") if c])
    print(f"# {'FAIL: ' + ', '.join(failed) if failed else 'ALL PASS'}", flush=True)
    raise SystemExit(1 if failed else 0)
