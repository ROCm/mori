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
"""Worker for ``test_gemm_ar_v2.py``: one torchrun spawn on 8 ranks.

Prints one ``RESULT_JSON {"case": ...}`` line per case from rank 0, each case
merged over ranks. Operands depend only on (seed, rank, salt, M), so every rank
rebuilds every rank's operands and checks against the fp32 sum locally.
"""

import functools
import hashlib
import json
import os
import statistics
import traceback

import torch
import torch.distributed as dist

N, K = 7168, 2048
M_MAX = 16384
TOL = 6e-3
WRAP = 2**32 - 3


def _gen(*parts):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(hashlib.sha256(repr(parts).encode()).hexdigest()[:15], 16))
    return g


def act(rank, salt, m):
    return (torch.randn(m, K, generator=_gen("a", rank, salt, m)) / 8).to(torch.bfloat16).cuda()


@functools.lru_cache(maxsize=None)
def wgt(rank):
    return (torch.randn(N, K, generator=_gen("b", rank)) / 8).to(torch.bfloat16).cuda()


def reference(world, salt, m):
    ref = torch.zeros(m, N, dtype=torch.float32, device="cuda")
    for r in range(world):
        ref += act(r, salt, m).float() @ wgt(r).float().t()
    return ref


def rel_l2(got, ref):
    return float((got.float() - ref).norm() / ref.norm().clamp_min(1e-30))


def wait():
    """Busy-poll the stream: a sleeping wait skews the ranks (G950-MEAS-07)."""
    ev = torch.cuda.Event()
    ev.record()
    while not ev.query():
        pass


class Ctx:
    def __init__(self, rank, world, comm):
        self.rank, self.world, self.comm = rank, world, comm
        self.out = {}

    def emit(self, case, mine):
        every = [None] * self.world
        dist.all_gather_object(every, mine)
        if self.rank == 0:
            rec = {"case": case, "per_rank": every}
            print("RESULT_JSON " + json.dumps(rec), flush=True)


def check_calls(op, world, m, salts, graph=False):
    rows = []
    rank = dist.get_rank()
    if not graph:
        for salt in salts:
            out = op(act(rank, salt, m))
            wait()
            rows.append(
                {"m": m, "salt": salt, "kind": "eager", "rel_l2": rel_l2(out, reference(world, salt, m)),
                 "fused": op.fused_path_ran}
            )
        return rows
    a = act(rank, salts[0], m)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(a)
    torch.cuda.current_stream().wait_stream(side)
    wait()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = op(a)
    for salt in salts:
        a.copy_(act(rank, salt, m))
        g.replay()
        wait()
        rows.append(
            {"m": m, "salt": salt, "kind": "graph", "rel_l2": rel_l2(out, reference(world, salt, m)),
             "fused": op.fused_path_ran}
        )
    del g
    return rows


def time_graph(op, x, warmup=10, iters=30):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(x)
    torch.cuda.current_stream().wait_stream(side)
    wait()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        op(x)
    for _ in range(warmup):
        g.replay()
    wait()
    s, e, samples = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), []
    for _ in range(iters):
        s.record()
        g.replay()
        e.record()
        wait()
        samples.append(s.elapsed_time(e) * 1000.0)
    del g
    return statistics.median(samples)


def main():
    from mori.cco import Communicator, UniqueId
    from mori.ops.gemm_ar_v2.op import GemmArV2, GemmArV2Split

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    uid = UniqueId.from_bytes(payload[0])
    win = GemmArV2.window_bytes_for(world, m_max=M_MAX, n=N, k=K)
    # two instances live at once in the ordering case
    vmm = 3 * win + (256 << 20)
    with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
        ctx = Ctx(rank, world, comm)
        op = GemmArV2(comm, n=N, k=K, m_max=M_MAX)
        try:
            op.set_weight(wgt(rank))
            try:
                op.self_test()
                ctx.emit("sdma", {"available": True})
            except RuntimeError as e:
                ctx.emit("sdma", {"available": False, "error": str(e)[:500]})
                return
            # correctness: M=1, a non-multiple of 256, a bucket, an odd M
            for i, m in enumerate((1, 1000, 4096, 6309)):
                try:
                    rows = check_calls(op, world, m, [10 * i + 1, 10 * i + 2])
                    rows += check_calls(op, world, m, [10 * i + 3, 10 * i + 4, 10 * i + 5], graph=True)
                    ctx.emit(f"correct_m{m}", {"rows": rows})
                except Exception:  # noqa: BLE001
                    ctx.emit(f"correct_m{m}", {"error": traceback.format_exc()[-2000:]})
            # counter / epoch wrap
            try:
                torch.cuda.synchronize()
                op.debug_preset_counters(WRAP)
                rows = check_calls(op, world, 4096, [91, 92, 93, 94])
                rows += check_calls(op, world, 4096, [95, 96, 97], graph=True)
                ctx.emit("wrap", {"rows": rows, "dump": op.debug_dump()})
            except Exception:  # noqa: BLE001
                ctx.emit("wrap", {"error": traceback.format_exc()[-2000:]})
            # same-process ordering: fused (stream) against split, interleaved
            try:
                split = GemmArV2Split(comm, n=N, k=K, m_max=M_MAX)
                try:
                    split.set_weight(wgt(rank))
                    x = act(rank, 0, M_MAX)
                    t = {"fused": [], "split": []}
                    for _ in range(3):
                        t["fused"].append(time_graph(op, x))
                        t["split"].append(time_graph(split, x))
                    ctx.emit("ordering", {"m": M_MAX, "us": t})
                finally:
                    split.close()
            except Exception:  # noqa: BLE001
                ctx.emit("ordering", {"error": traceback.format_exc()[-2000:]})
        finally:
            torch.cuda.synchronize()
            op.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
