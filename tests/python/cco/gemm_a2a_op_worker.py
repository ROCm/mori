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
"""Ranks for ``test_gemm_a2a_op.py``: exercises ``GemmA2AOp`` itself.

Every check reads what *arrived*, per source rank, against an fp32 reference
rebuilt from that source's seed -- a transport that moves nothing leaves this
rank's own block right and every other one wrong.

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
from mori.ops.gemm_a2a import GemmA2AOp
from mori.ops.gemm_ar import preshuffle_a_scale, preshuffle_b

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "benchmark/cco/flydsl/gemm_a2a")
)
import bench_gemm_a2a as B  # noqa: E402  (operands and reference only)

#: Below fp8's own floor (1.66e-3 at these operands) with headroom; a missing or
#: stale block lands near 1.0.
TOL = 3e-3


def _setup():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return rank, world, UniqueId.from_bytes(payload[0])


def _seed(src: int, salt: int) -> int:
    # make_operands seeds A from its rank argument and B from a constant, so a
    # different "rank" gives different A against the same, replicated B.
    return src + 64 * salt


def _inputs(src, salt, m, n, k, quant):
    a, b, sa, sb = B.make_operands(_seed(src, salt), m, n, k, quant)
    if quant == "mxfp8":
        sa_arg, sb_arg = preshuffle_a_scale(sa), sb.to(torch.int32).t().contiguous()
    else:
        sa_arg, sb_arg = sa, sb
    return a, preshuffle_b(b), sa_arg, sb_arg


def _worst(recv, rank, world, salt, m, n, k, quant):
    """Worst per-source relL2 of this rank's received block, and all of them."""
    shard = n // world
    cols = slice(rank * shard, (rank + 1) * shard)
    rels = []
    for src in range(world):
        a, b, sa, sb = B.make_operands(_seed(src, salt), m, n, k, quant)
        ref = B.reference_gemm(a, b, sa, sb, quant)[:, cols]
        got = recv[src * m : (src + 1) * m].float()
        rels.append(((got - ref).norm() / ref.norm()).item())
    return max(rels), rels


def case_correct(op, rank, world, n, k, quant):
    """Two M values on one instance, alternating, then a captured graph."""
    out = []
    for m, salt in ((2048, 1), (1024, 2), (2048, 3)):
        recv = op(*_inputs(rank, salt, m, n, k, quant))
        torch.cuda.synchronize()
        out.append(("eager", m, _worst(recv, rank, world, salt, m, n, k, quant)[0]))
    # Graph: capture on static inputs, then refill them and replay. The call
    # counter lives on the device, so the credit and the barrier advance per
    # replay exactly as they do per eager call.
    m = 2048
    static = [t.clone() for t in _inputs(rank, 4, m, n, k, quant)]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        op(*static)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        recv = op(*static)
    for salt in (5, 6):
        for dst, src in zip(static, _inputs(rank, salt, m, n, k, quant)):
            dst.copy_(src)
        g.replay()
        torch.cuda.synchronize()
        out.append(("graph", m, _worst(recv, rank, world, salt, m, n, k, quant)[0]))
    return {"checks": out, "ok": all(w < TOL for *_, w in out)}


def _delay(ms: float):
    """Keep this rank's stream busy for about ``ms`` before the next work."""
    if hasattr(torch.cuda, "_sleep"):
        torch.cuda._sleep(int(ms * 1e6))  # cycles at ~1 GHz; only "long" matters
    else:  # pragma: no cover - a GPU without the sleep kernel
        x = torch.randn(4096, 4096, device="cuda")
        for _ in range(int(ms)):
            x = x @ x / 64


def case_race(op, rank, world, n, k, quant, victim_delay_ms):
    """The recv-reuse race, provoked.

    Everyone runs call 1. The victim (rank 0) then keeps its stream busy before
    it reads the result, as a slow attention would; everyone else goes straight
    into call 2 with different inputs. Without the credit their pushes land in
    the victim's recv while it is still unread. With it, they wait for the
    victim to start call 2 -- which its stream only does after the read.
    """
    m = 2048
    recv = op(*_inputs(rank, 1, m, n, k, quant))
    if rank == 0:
        _delay(victim_delay_ms)
        snap = recv.clone()  # the "consumer": ordered before call 2 on this stream
    op(*_inputs(rank, 2, m, n, k, quant))
    torch.cuda.synchronize()
    if rank != 0:
        return {"victim": False}
    worst, rels = _worst(snap, rank, world, 1, m, n, k, quant)
    return {"victim": True, "worst": worst, "per_source": rels, "ok": worst < TOL}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=("correct", "race", "self_test"), required=True)
    p.add_argument("--mode", default="fused")
    p.add_argument("--quant", default="ptpc")
    p.add_argument("--no-credit", action="store_true")
    p.add_argument("-n", type=int, default=10240)
    p.add_argument("-k", type=int, default=2048)
    p.add_argument("--victim-delay-ms", type=float, default=50.0)
    args = p.parse_args()

    rank, world, uid = _setup()
    m_max = 2048
    wb = GemmA2AOp.window_bytes_for(world, n=args.n, m_max=m_max, quant=args.quant)
    with Communicator.init(world, rank, uid, per_rank_vmm=2 * wb + (256 << 20)) as comm:
        with GemmA2AOp(
            comm,
            n=args.n,
            k=args.k,
            m_max=m_max,
            quant=args.quant,
            mode=args.mode,
            war_credit=not args.no_credit,
        ) as op:
            if args.case == "self_test":
                op.self_test()
                res = {"ok": True}
            elif args.case == "correct":
                res = case_correct(op, rank, world, args.n, args.k, args.quant)
            else:
                res = case_race(
                    op, rank, world, args.n, args.k, args.quant, args.victim_delay_ms
                )
            torch.cuda.synchronize()
    gathered = [None] * world
    dist.all_gather_object(gathered, res)
    if rank == 0:
        print("RESULT_JSON " + json.dumps({"case": args.case, "ranks": gathered}))
    dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
