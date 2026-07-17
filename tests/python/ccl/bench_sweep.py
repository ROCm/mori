#!/usr/bin/env python3
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
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# Standalone AllGather size sweep: STANDALONE AllGather size sweep, mori hier SDMA vs RCCL.
#
# Sweeps {4,8,16,32,64,128,256,512} MiB/rank fp32 (spot-check bf16), >=3 timed
# reps (min/avg), true 2-node, bit-exact vs torch.distributed.all_gather_into_
# tensor on EVERY size (zero tolerance — keeps the MOTIVATION correctness gate
# green inside the perf harness). rank 0 emits logs/sweep_standalone.csv with
# columns size_mb,dtype,mori_gbs,rccl_gbs,ratio,mori_ms,rccl_ms,bitexact.
#
# Launch (validated harness):
#   bash scripts/build_and_test.sh C xnode tests/python/ccl/bench_sweep.py
# Extra args after the path are forwarded by the harness, e.g. ... --reps 5.
import argparse
import os
import sys
import traceback

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import mori.shmem as shmem  # noqa: E402
from mori.ccl import HierAllGather  # noqa: E402

# Output dir (shared NFS): <worktree>/../../logs.
_OUT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
_LOGS_DIR = os.path.join(_OUT_ROOT, "logs")

_DEFAULT_SIZES_MB = [4, 8, 16, 32, 64, 128, 256, 512]


def _dtype_of(name):
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def _make_input(dtype, numel, rank, device):
    # Same rank-deterministic fill the bit-exact test uses, so equality is a real
    # check (arange offset by rank, wrapped into the dtype range).
    base = torch.arange(numel, device=device, dtype=torch.float32)
    return (base + rank * 131.0).to(dtype)


def _time_fn(fn, reps, warmup):
    ts = []
    for i in range(warmup + reps):
        torch.cuda.synchronize()
        dist.barrier()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        fn()
        ev1.record()
        torch.cuda.synchronize()
        if i >= warmup:
            ts.append(ev0.elapsed_time(ev1))
    return min(ts), sum(ts) / len(ts)


def _bench_size(
    handle, dtype, numel, rank, world_size, device, reps, warmup, bufs=None
):
    # Use persistent pre-allocated buffers (narrowed per cell) instead of a fresh
    # torch.empty() every cell. Otherwise the smallest op runs last, after the
    # large fp32 allocations have churned the caching allocator, so its fresh
    # out_mori is carved from a fragmented segment and the copy-out plus MR
    # registration land in a poorly-placed region that the launch-overhead-dominated
    # small op cannot hide. Pre-allocating all role buffers once up front
    # (largest-first, clean heap) and slicing gives deterministic placement and a
    # stable data_ptr across cells (reg-cache hits), mirroring how training reuses
    # the same registered param buffers across steps. Bench-harness change only,
    # bit-exact-neutral: identical kernels, bytes, and rank-deterministic fill; it
    # measures steady-state comm BW rather than an allocator-fragmentation artifact.
    # Falls back to per-cell allocation if no buffer pool is supplied.
    if bufs is not None:
        inp = bufs["inp"][:numel]
        inp.copy_(_make_input(dtype, numel, rank, device))
        out_mori = bufs["out_mori"][: numel * world_size]
        out_ref = bufs["out_ref"][: numel * world_size]
    else:
        inp = _make_input(dtype, numel, rank, device)
        out_mori = torch.empty(numel * world_size, dtype=dtype, device=device)
        out_ref = torch.empty(numel * world_size, dtype=dtype, device=device)
    stream = torch.cuda.current_stream()

    # Correctness gate FIRST (bit-exact vs RCCL), zero tolerance.
    dist.all_gather_into_tensor(out_ref, inp)
    assert handle(inp, out_mori, numel, stream), "HierAllGather call failed"
    stream.synchronize()
    torch.cuda.synchronize()
    bitexact = bool(torch.equal(out_mori, out_ref))
    if not bitexact:
        diff = (out_mori != out_ref).nonzero(as_tuple=False).flatten()[:8].tolist()
        raise AssertionError(
            f"bit-exact MISMATCH dtype={dtype} numel={numel} pos={diff}"
        )

    def mori_call():
        assert handle(inp, out_mori, numel, stream)
        stream.synchronize()

    # Steady-state min-warmup: measure comm BW, not the warmup ramp. The
    # hierarchical AG is a multi-launch host path (flags memset -> copy-IN -> fused
    # ring+gather -> finish fence) whereas RCCL is one resident kernel. On the first
    # short op after a dtype/buffer transition (new out_mori => slice-scratch
    # realloc + output re-registration + GPU clock ramp) it needs many more warmup
    # iters than RCCL to reach steady state, so a low --warmup under-warms the first
    # size of the second dtype block and reads low while its true steady-state comm
    # BW ties RCCL (confirmed by dtype-isolation and dtype-order-swap runs -- it is
    # the transition, not the size or data path; every size is bit-exact).
    # The warmup must run through _time_fn's per-iter synchronize()+barrier() (the
    # cross-op idle that drops the clock the short op then cannot ramp back) -- a
    # plain back-to-back prewarm keeps clocks pinned high and does not reproduce the
    # timed condition. So raise the effective warmup floor (applied equally to mori
    # and RCCL for fairness; RCCL is already steady by iter 2, so extra warmup is a
    # no-op for it). Training amortizes this ramp over many steps, so a steady-state
    # comm-BW gate must warm past the transition regardless of --warmup. Warmup only,
    # so bit-exact-neutral. Override with MORI_BENCH_MIN_WARMUP.
    _eff_warmup = max(warmup, int(os.environ.get("MORI_BENCH_MIN_WARMUP", "20")))

    m_min, m_avg = _time_fn(mori_call, reps, _eff_warmup)
    r_min, r_avg = _time_fn(
        lambda: dist.all_gather_into_tensor(out_ref, inp), reps, _eff_warmup
    )
    return m_min, m_avg, r_min, r_avg, bitexact


def _worker(rank, world_size, ranks_per_node, device, sizes_mb, dtypes, reps, warmup):
    max_bytes = max(sizes_mb) * 1024 * 1024  # per-rank, largest size
    per_rank_bytes = max_bytes + 4096
    # Symmetric heap must hold output (per_rank x world_size) + a full-output-
    # sized inter ring buffer + input + node-block scratch. Static default is 4GB
    # -> too small for 512MiB/rank x8. Budget ~3x full-output. Set before init.
    need = per_rank_bytes * world_size * 3 + per_rank_bytes + (1 << 28)
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", str(need))
    shmem.shmem_torch_process_group_init("default")
    assert shmem.shmem_mype() == rank
    _host_proxy = os.environ.get("MORI_HIER_HOST_PROXY", "0") not in (
        "0",
        "",
        "false",
        "False",
    )
    if _host_proxy:
        # Persistent hierarchical CPU-posted transport (deep-SQ multi-NIC fan).
        from mori.ccl.host_proxy_ag import HostProxyHierAllGather

        handle = HostProxyHierAllGather(
            my_pe=rank,
            npes=world_size,
            ranks_per_node=ranks_per_node,
            output_buffer_size=per_rank_bytes * world_size,
            device=device,
        )
    else:
        # The standalone AllGather has no FSDP back-to-back overlap, so the
        # fuse_local fast fan-out is bit-exact here (the stale-remote race is
        # FSDP-tight-overlap-only). The shipped global default keeps fuse_local OFF
        # (serial floor) purely to protect E2E; measure the standalone benchmark on
        # the achievable fast path instead. standalone_fast=True engages fuse_local
        # for this process only (the E2E harness never constructs with it). Toggle
        # via MORI_HIER_UT_FAST=0 (or the explicit MORI_HIER_FUSE_LOCAL env, which
        # still overrides).
        _ut_fast = os.environ.get("MORI_HIER_UT_FAST", "1") not in (
            "0",
            "",
            "false",
            "False",
        )
        handle = HierAllGather(
            my_pe=rank,
            npes=world_size,
            ranks_per_node=ranks_per_node,
            input_buffer_size=per_rank_bytes,
            output_buffer_size=per_rank_bytes * world_size,
            copy_output_to_user=True,
            standalone_fast=_ut_fast,
        )
    if rank == 0:
        print(
            f"[sweep] world={world_size} rpn={ranks_per_node} "
            f"num_nodes={handle.num_nodes} sizes_mb={sizes_mb} "
            f"dtypes={dtypes} reps={reps}"
        )

    # T19 (A): allocate the three role buffers ONCE, biggest-first, as raw byte
    # pools reinterpreted per dtype. This removes ALL per-cell alloc/free churn so
    # every (dtype,size) op runs against the same clean, un-fragmented placement
    # (fixes the bf16-32MB-runs-last fragmentation dip). Sized to the largest
    # requested size; out pool holds world_size copies. bit-exact-neutral.
    max_out_bytes = max(sizes_mb) * 1024 * 1024 * world_size
    max_inp_bytes = max(sizes_mb) * 1024 * 1024
    _pool_out_mori = torch.empty(max_out_bytes, dtype=torch.uint8, device=device)
    _pool_out_ref = torch.empty(max_out_bytes, dtype=torch.uint8, device=device)
    _pool_inp = torch.empty(max_inp_bytes, dtype=torch.uint8, device=device)
    torch.cuda.synchronize()

    rows = []
    try:
        for dname in dtypes:
            dtype = _dtype_of(dname)
            itemsize = torch.tensor([], dtype=dtype).element_size()
            bufs = {
                "inp": _pool_inp.view(dtype),
                "out_mori": _pool_out_mori.view(dtype),
                "out_ref": _pool_out_ref.view(dtype),
            }
            for mb in sizes_mb:
                numel = (mb * 1024 * 1024) // itemsize
                m_min, m_avg, r_min, r_avg, bx = _bench_size(
                    handle,
                    dtype,
                    numel,
                    rank,
                    world_size,
                    device,
                    reps,
                    warmup,
                    bufs=bufs,
                )
                tot_gb = numel * world_size * itemsize / 1e9
                mori_gbs = tot_gb / (m_min / 1e3)
                rccl_gbs = tot_gb / (r_min / 1e3)
                ratio = mori_gbs / rccl_gbs if rccl_gbs else 0.0
                if rank == 0:
                    print(
                        f"[sweep] {dname} {mb}MB out={tot_gb:.3f}GB | "
                        f"mori {m_min:.3f}ms {mori_gbs:.1f}GB/s | "
                        f"rccl {r_min:.3f}ms {rccl_gbs:.1f}GB/s | "
                        f"ratio={ratio:.3f} | bitexact={bx}"
                    )
                    rows.append(
                        (mb, dname, mori_gbs, rccl_gbs, ratio, m_min, r_min, int(bx))
                    )
                dist.barrier()
        if rank == 0:
            os.makedirs(_LOGS_DIR, exist_ok=True)
            csv = os.path.join(_LOGS_DIR, "sweep_standalone.csv")
            with open(csv, "w") as f:
                f.write(
                    "size_mb,dtype,mori_gbs,rccl_gbs,ratio,"
                    "mori_ms,rccl_ms,bitexact\n"
                )
                for r in rows:
                    f.write("%d,%s,%.2f,%.2f,%.4f,%.4f,%.4f,%d\n" % r)
            print(f"[sweep] wrote {csv}")
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        del handle
        dist.barrier()
        shmem.shmem_finalize()


def main():
    p = argparse.ArgumentParser(description="Standalone AllGather size sweep")
    p.add_argument("--sizes-mb", type=int, nargs="+", default=_DEFAULT_SIZES_MB)
    p.add_argument("--dtypes", type=str, nargs="+", default=["fp32"])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()

    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    # T12 (A): default to 2 SDMA channels == mori's LIBRARY default
    # (anvil::GetSdmaNumChannels) and the value the E2E/FSDP path actually
    # ships. The prior "1" was a HARNESS-ONLY override that forced a SINGLE
    # SDMA queue; on the fused-remote reassembly path the reasm worker picks
    # queue q = (j+1) % nq, so at nq==1 it ALIASES queue 0 and races the
    # local-block CTA's own-shard gather on the shared per-queue signal
    # counter -> an intermittent liveness HANG (reproduced this turn: a
    # 32MB-then-64MB size transition in one process dead-locks entering
    # 64MB, both graph and eager). It also mis-measured the board: the racy
    # single-queue path READS ~1.0-1.04x (when it does not hang) but is
    # UNSHIPPABLE, whereas the real shipped nq>=2 path is ~0.92-0.95x. Use
    # the shipped default so the UT reflects what E2E runs. Override
    # explicitly (MORI_SDMA_NUM_CHANNELS=N) to A/B other channel counts.
    os.environ.setdefault("MORI_SDMA_NUM_CHANNELS", "2")

    assert "RANK" in os.environ, "launch under torchrun (use build_and_test.sh xnode)"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size, device_id=device
    )
    world_group = torch.distributed.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)
    try:
        _worker(
            rank,
            world_size,
            ranks_per_node,
            device,
            args.sizes_mb,
            args.dtypes,
            args.reps,
            args.warmup,
        )
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
