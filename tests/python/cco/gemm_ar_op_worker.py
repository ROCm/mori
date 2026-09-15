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
"""Ranks for ``test_gemm_ar_op.py``: exercises ``GemmAllReduceOp`` itself.

Separate from ``bench_gemm_ar.py`` on purpose. That driver calls the low-level
builders, so every number it produces is silent about the public wrapper -- the
column-major A scale, the per-M control layout and the operand checks all live
between the two, and had no coverage at all until this existed.

Prints one ``RESULT_JSON`` line per case from rank 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.distributed as dist

from mori.cco import Communicator, UniqueId
from mori.ops.gemm_ar import GemmAllReduceOp, preshuffle_b

SCALE_BK = 128


def _setup():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return rank, world, UniqueId.from_bytes(payload[0])


def _operands(rank: int, m: int, n: int, k: int, salt: int = 0):
    """fp8 operands with block scales, in the layouts the model produces.

    ``sa`` is built column-major, which is what
    ``aiter_per1x128_quant(transpose_scale=True)`` and sglang's
    ``materialize_bpreshuffle_fp8_scale`` return, and is the case the public op
    used to flatten in the wrong order.
    """
    g = torch.Generator(device="cuda").manual_seed(1234 + rank + 7919 * salt)
    a = (torch.randn(m, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    kb = k // SCALE_BK
    sa = (
        torch.rand(m, kb, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.01
    )
    sa = sa.t().contiguous().t()  # logical [M, kb], physically [kb, M]
    sb = (
        torch.rand(n // SCALE_BK, kb, generator=g, device="cuda", dtype=torch.float32)
        * 0.01
        + 0.01
    )
    return a, b, sa, sb


def _reference(world: int, m: int, n: int, k: int, salt: int = 0) -> torch.Tensor:
    """All-reduced fp32 reference, built from the same recipe on every rank."""
    acc = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    kb = k // SCALE_BK
    for r in range(world):
        a, b, sa, sb = _operands(r, m, n, k, salt)
        af, bf = a.float(), b.float()
        part = torch.zeros(m, n, device="cuda", dtype=torch.float32)
        for j in range(kb):
            sl = slice(j * SCALE_BK, (j + 1) * SCALE_BK)
            blk = af[:, sl] @ bf[:, sl].t()
            part += blk * sa[:, j : j + 1] * sb[:, j].repeat_interleave(SCALE_BK)[:n]
        acc += part
    return acc


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    return (
        torch.linalg.vector_norm(got.float() - ref) / torch.linalg.vector_norm(ref)
    ).item()


def _emit(rank: int, **record) -> None:
    if rank == 0:
        print("RESULT_JSON " + json.dumps(record, sort_keys=True), flush=True)


def case_scale_order(op, rank, world, m, n, k):
    """The documented [M, K/128] column-major scale must give the fp8 floor.

    Passing the same tensor through reshape(-1) -- the old behaviour -- walks the
    logical rows and pairs each scale with the wrong K block.
    """
    a, b, sa, sb = _operands(rank, m, n, k)
    ref = _reference(world, m, n, k)
    bs = preshuffle_b(b)

    got = op(a, bs, sa, sb).clone()
    logical = _rel_l2(got, ref)

    # The two other accepted spellings must agree with it bitwise.
    flat = op(a, bs, sa.t().reshape(-1), sb).clone()
    physical = op(a, bs, sa.t(), sb).clone()

    _emit(
        rank,
        case="scale_order",
        rel_l2=logical,
        flat_matches=bool(torch.equal(got, flat)),
        physical_matches=bool(torch.equal(got, physical)),
    )


def _peer_c(peer: int, m: int, n: int, k: int) -> torch.Tensor:
    """Peer's *unreduced* GEMM output at this shape, in bf16.

    What the scatter pushes into my recv slot for that peer, so it is also what
    a stale copy of that push would leave behind.
    """
    a, b, sa, sb = _operands(peer, m, n, k)
    af, bf = a.float(), b.float()
    out = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    kb = k // SCALE_BK
    for j in range(kb):
        sl = slice(j * SCALE_BK, (j + 1) * SCALE_BK)
        out += (
            (af[:, sl] @ bf[:, sl].t())
            * sa[:, j : j + 1]
            * sb[:, j].repeat_interleave(SCALE_BK)[:n]
        )
    return out.to(torch.bfloat16)


def _row_report(got, ref, m):
    """Which rows are wrong, as a contiguous-block summary.

    A fixed wrong value hit intermittently means the race decides *whether* a
    region is read stale, not *what* is in it -- so the row map is what points
    at the region.
    """
    err = (got.float() - ref).abs().amax(dim=1)
    scale = ref.abs().amax(dim=1).clamp_min(1e-6)
    bad = (err / scale > 0.05).nonzero().flatten()
    if bad.numel() == 0:
        return {"bad_rows": 0}
    return {
        "bad_rows": int(bad.numel()),
        "m": m,
        "first_bad": int(bad[0]),
        "last_bad": int(bad[-1]),
        "contiguous": bool(bad.numel() == int(bad[-1]) - int(bad[0]) + 1),
    }


def case_alternating_m(op, rank, world, m_small, m_large, n, k):
    """Two M values through one instance, alternating, then repeated.

    The control region moves with counter_chunks unless it is sized for the
    capacity, and the election counters carry a residue between shapes unless
    each shape owns a set. Either one hangs or corrupts the second call.
    """
    results = {}
    for label, m in (
        ("first_small", m_small),
        ("first_large", m_large),
        ("again_small", m_small),
        ("again_large", m_large),
    ):
        a, b, sa, sb = _operands(rank, m, n, k)
        got = op(a, preshuffle_b(b), sa, sb).clone()
        ref = _reference(world, m, n, k)
        results[label] = _rel_l2(got, ref)
        if results[label] > 0.01:
            rep = _row_report(got, ref, m)
            # Hypothesis: the bad range is exactly where the *previous* shape's
            # recv slot for peer 1 sat, and still holds that peer's scatter
            # payload from the previous call.
            if rep["bad_rows"] and m_small != m:
                stale = _peer_c(1, m_small, n, k)[: rep["bad_rows"]]
                seen = got[rep["first_bad"] : rep["last_bad"] + 1]
                rep["matches_previous_scatter"] = bool(torch.equal(seen, stale))
                rep["max_abs_diff_vs_stale"] = float(
                    (seen.float() - stale.float()).abs().max()
                )
            results[f"{label}_rows"] = rep
    _emit(rank, case="alternating_m", **results)


def case_close_is_idempotent(comm, rank, n, k, m_max):
    """close() releases, twice is a no-op, and a closed op refuses to run."""
    op = GemmAllReduceOp(comm, n=n, k=k, m_max=m_max)
    op.close()
    op.close()
    refused = False
    try:
        op(
            torch.zeros(m_max, k, device="cuda", dtype=torch.float8_e4m3fn),
            torch.zeros(n, k, device="cuda", dtype=torch.float8_e4m3fn),
            torch.zeros(m_max * (k // SCALE_BK), device="cuda", dtype=torch.float32),
            torch.zeros(
                (n // SCALE_BK) * (k // SCALE_BK), device="cuda", dtype=torch.float32
            ),
        )
    except RuntimeError:
        refused = True
    with GemmAllReduceOp(comm, n=n, k=k, m_max=m_max):
        pass  # context manager closes on exit
    _emit(rank, case="close", refused_after_close=refused)


def case_changing_data(op, rank, world, m, n, k, calls):
    """Same M, different operands each call.

    Every existing test and benchmark feeds identical operands on every
    iteration, so a phase that reads a peer's slice before it has landed returns
    the *previous* call's bytes -- which are bit-identical to the right answer.
    Changing the data between calls is what makes such a read visible.
    """
    out = {}
    for i in range(calls):
        a, b, sa, sb = _operands(rank, m, n, k, salt=i)
        got = op(a, preshuffle_b(b), sa, sb).clone()
        out[f"call{i}"] = _rel_l2(got, _reference(world, m, n, k, salt=i))
    _emit(rank, case="changing_data", **out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("-n", type=int, default=1024)
    p.add_argument("-k", type=int, default=512)
    p.add_argument("-m", type=int, default=0)
    p.add_argument("--m-small", type=int, default=0)
    p.add_argument("--m-large", type=int, default=0)
    p.add_argument("--calls", type=int, default=4)
    args = p.parse_args()

    rank, world, uid = _setup()
    m_max = max(args.m, args.m_large, args.m_small)
    vmm = 2 * GemmAllReduceOp.window_bytes_for(world, m_max=m_max, n=args.n) + (
        256 << 20
    )
    with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
        if args.case == "close":
            case_close_is_idempotent(comm, rank, args.n, args.k, m_max)
            return 0
        with GemmAllReduceOp(comm, n=args.n, k=args.k, m_max=m_max) as op:
            if args.case == "scale_order":
                case_scale_order(op, rank, world, args.m, args.n, args.k)
            elif args.case == "changing_data":
                case_changing_data(op, rank, world, args.m, args.n, args.k, args.calls)
            elif args.case == "alternating_m":
                case_alternating_m(
                    op, rank, world, args.m_small, args.m_large, args.n, args.k
                )
            else:
                raise SystemExit(f"unknown case {args.case}")
    dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
