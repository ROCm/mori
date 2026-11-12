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
"""Ranks for ``test_a2a_gemm_op.py``: exercises ``A2aGemmOp`` itself.

Every check compares this rank's ``C`` against an fp32 reference rebuilt from
every source's seed. Each source owns a K segment, so a band that never
arrived, or arrived stale, moves relL2 far from fp8's floor.

Prints one ``RESULT_JSON`` line from rank 0 with every rank's verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from mori.cco import Communicator, UniqueId
from mori.ops.a2a_gemm import A2aGemmOp
from mori.ops.gemm_ar import preshuffle_b

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "benchmark/cco/flydsl/a2a_gemm")
)
import bench_a2a_gemm as B  # noqa: E402  (operands and reference only)

#: fp8's floor at these operands is ~1.7e-3; a missing segment lands near 1/P.
TOL = 3e-3
#: Spans a single band (P=8 at 1024 and 2048), several bands with ``wait=tile``
#: (P=8 at 4096), and a change of chunk count between consecutive calls.
M_PLAN = (4096, 1024, 2048, 4096)


def _setup():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return rank, world, UniqueId.from_bytes(payload[0])


class _Inputs:
    def __init__(self, rank, world, n, k):
        self.rank, self.world, self.k = rank, world, k
        self.b, self.sb = B.weight(n, k)
        self.b_pre = preshuffle_b(self.b)

    def a(self, m, salt):
        return B.head_shard(self.rank, self.world * m, self.k // self.world, salt)

    def sa(self, m, salt):
        r = self.rank
        return B.token_scale(self.world * m, salt)[r * m : (r + 1) * m].contiguous()

    def rel(self, c, m, salt):
        ref = B.reference(self.world, self.rank, m, self.k, self.b, self.sb, salt)
        return ((c.float() - ref).norm() / ref.norm()).item()


def case_correct(op, x: _Inputs):
    """Several M on one instance (and so several chunk counts), one call that
    quantizes straight into ``send_buffer``, then graph replays."""
    out = []
    for i, m in enumerate(M_PLAN):
        salt = 1 + i
        c = op(x.a(m, salt), x.b_pre, x.sa(m, salt), x.sb)
        torch.cuda.synchronize()
        out.append(("eager", m, op.chunks_for(m), x.rel(c, m, salt)))
    m, salt = 2048, 9
    send = op.send_buffer(m)
    send.copy_(x.a(m, salt))
    c = op(send, x.b_pre, x.sa(m, salt), x.sb)
    torch.cuda.synchronize()
    out.append(("in-place", m, op.chunks_for(m), x.rel(c, m, salt)))
    # Graph: capture on static inputs, then refill them and replay. The epoch
    # and the counter snapshot live in the window, so a replay advances them
    # exactly as an eager call does.
    m = 4096
    a_s, sa_s = x.a(m, 10).clone(), x.sa(m, 10).clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        op(a_s, x.b_pre, sa_s, x.sb)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        c = op(a_s, x.b_pre, sa_s, x.sb)
    for salt in (11, 12):
        a_s.copy_(x.a(m, salt))
        sa_s.copy_(x.sa(m, salt))
        g.replay()
        torch.cuda.synchronize()
        out.append(("graph", m, op.chunks_for(m), x.rel(c, m, salt)))
    return {"checks": out, "ok": all(r < TOL for *_, r in out)}


def _delay(ms: float):
    """Keep this rank's stream busy for about ``ms`` before the next work."""
    if hasattr(torch.cuda, "_sleep"):
        torch.cuda._sleep(int(ms * 1e6))  # cycles at ~1 GHz; only "long" matters
    else:  # pragma: no cover - a GPU without the sleep kernel
        y = torch.randn(4096, 4096, device="cuda")
        for _ in range(int(ms)):
            y = y @ y / 64


def case_skew(op, x: _Inputs, delay_ms):
    """Back to back with no host sync, one rank late per call, as a slow
    attention would be. Bands from a peer that is ahead must not be counted
    toward the late rank's previous call, nor land in a recv it still reads."""
    m, calls = 4096, x.world
    outs = []
    for i in range(calls):
        if x.rank == i % x.world:
            _delay(delay_ms)
        outs.append(op(x.a(m, 20 + i), x.b_pre, x.sa(m, 20 + i), x.sb).clone())
    torch.cuda.synchronize()
    rels = [x.rel(c, m, 20 + i) for i, c in enumerate(outs)]
    return {"rels": rels, "ok": all(r < TOL for r in rels)}


def case_reject(comm, n, k):
    def verdict(**kw):
        kw = {"n": n, "k": k, "m_max": 256, **kw}
        try:
            with A2aGemmOp(comm, **kw) as op:
                op.self_test()
                return "accepted"
        except (ValueError, RuntimeError) as e:
            return f"rejected: {type(e).__name__}"

    return {
        "ok_args": verdict(),
        "k_not_whole_steps": verdict(k=k + 64),
        "n_not_tiled": verdict(n=n + 128),
        "bad_mode": verdict(mode="pull"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--case",
        choices=("correct", "skew", "self_test", "reject", "all"),
        required=True,
    )
    p.add_argument("--mode", default="fused")
    p.add_argument("-n", type=int, default=2048)
    p.add_argument("--k-shard", type=int, default=512)
    p.add_argument("--delay-ms", type=float, default=20.0)
    args = p.parse_args()

    rank, world, uid = _setup()
    k = world * args.k_shard
    m_max = max(M_PLAN)
    # "all": every case but reject, an op each, keyed case/mode. reject stays a
    # launch of its own: it needs the communicator's first dev-comm.
    if args.case == "all":
        runs = [("self_test", "fused")] + [
            (case, mode) for case in ("correct", "skew") for mode in ("fused", "split")
        ]
    else:
        runs = [(args.case, args.mode)]
    wb = A2aGemmOp.window_bytes_for(world, k=k, m_max=m_max)
    vmm = (len(runs) + 1) * wb + (256 << 20)
    results = []
    with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
        for case, mode in runs:
            if case == "reject":
                res = case_reject(comm, args.n, k)
            else:
                x = _Inputs(rank, world, args.n, k)
                with A2aGemmOp(comm, n=args.n, k=k, m_max=m_max, mode=mode) as op:
                    if case == "self_test":
                        op.self_test()
                        res = {"ok": True}
                    elif case == "correct":
                        res = case_correct(op, x)
                    else:
                        res = case_skew(op, x, args.delay_ms)
                    torch.cuda.synchronize()
            key = (
                case if args.case != "all" or case == "self_test" else f"{case}/{mode}"
            )
            results.append((key, res))
    for key, res in results:
        gathered = [None] * world
        dist.all_gather_object(gathered, res)
        if rank == 0:
            print("RESULT_JSON " + json.dumps({"case": key, "ranks": gathered}))
    dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
