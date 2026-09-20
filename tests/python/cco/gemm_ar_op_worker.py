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
from mori.ops.gemm_ar import GemmAllReduceOp, preshuffle_a_scale, preshuffle_b

SCALE_BK = 128
MXFP8_BK = 32


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
    _emit(
        rank,
        case=f"alternating_m_{m_small}_{m_large}",
        **results,
    )


def case_self_test(op, rank, world, n, k):
    """self_test must pass on a working stack, and its check must have teeth.

    The second half matters more than the first: a check that cannot fail is
    exactly what let a whole end-to-end campaign run against a mori whose SDMA
    puts were compiled out.
    """
    op.self_test()
    ok = True
    # Now break it the way a dead transport would: run `reduce -> gather` with no
    # scatter, which is what an inert put amounts to.
    #
    # The landing slots have to be zeroed first, and that is not a detail. This
    # op has already served several cases, so the slots still hold a previous
    # call's peer data -- reducing that produces a plausible sum and the check
    # passes, which is the same stale-read effect that made an inert transport
    # look correct in the first place. Without this the teeth test has none.
    from mori.tensor_utils import from_gpu_ptr

    cfg = op._make_cfg(op.m_max)
    from_gpu_ptr(op.mem.ptr + cfg.recv_off, (cfg.recv_bytes,), torch.uint8).zero_()
    plan = op._compiled(op.m_max)
    plan.input.fill_(float(rank + 1))
    plan.output.zero_()
    import flydsl.expr as _fx

    stream = _fx.Stream(torch.cuda.current_stream())
    for name in plan.parts["order"]:
        if name == "scatter":
            continue
        plan.parts[name](op.dev_comm.ptr, op.win.handle, stream=stream)
    torch.cuda.current_stream().synchronize()
    want = world * (world + 1) / 2
    detected = bool((plan.output.float() - want).abs().max().item() / want > 1e-2)
    _emit(
        rank,
        case="self_test",
        passes_on_working_stack=ok,
        detects_missing_scatter=detected,
    )


def _mxfp8_operands(rank: int, m: int, n: int, k: int, salt: int = 0):
    """ue8m0 operands, in the layouts DeepSeek-V4.1-Flash's loader produces.

    Exponent bytes rather than arbitrary floats because that is what ue8m0 is:
    every scale is exactly a power of two, which is why the scaled MFMA can take
    them as instruction operands and apply them losslessly.
    """
    g = torch.Generator(device="cuda").manual_seed(4321 + rank + 7919 * salt)
    a = (torch.randn(m, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    ea = torch.randint(
        120, 123, (m, k // MXFP8_BK), generator=g, device="cuda", dtype=torch.int32
    )
    eb = torch.randint(
        120,
        123,
        (n // MXFP8_BK, k // MXFP8_BK),
        generator=g,
        device="cuda",
        dtype=torch.int32,
    )
    return a, b, ea, eb


def _mxfp8_reference(world: int, m: int, n: int, k: int) -> torch.Tensor:
    acc = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    for r in range(world):
        a, b, ea, eb = _mxfp8_operands(r, m, n, k)
        af, bf = a.float(), b.float()
        sav, sbv = torch.exp2(ea.float() - 127.0), torch.exp2(eb.float() - 127.0)
        for j in range(k // MXFP8_BK):
            sl = slice(j * MXFP8_BK, (j + 1) * MXFP8_BK)
            acc += (
                (af[:, sl] @ bf[:, sl].t())
                * sav[:, j : j + 1]
                * sbv[:, j].repeat_interleave(MXFP8_BK)[None, :n]
            )
    return acc


def case_mxfp8(comm, rank, world, m, n, k):
    """``quant="mxfp8"`` end to end: DeepSeek-V4.1-Flash's quantisation.

    Its own op rather than a case on the shared one: mxfp8 compiles at
    BLOCK_M=256 where blockscale needs 128, so the two cannot share a window --
    the padding granule and the counter slots both come from block_m.

    The A scale goes through ``preshuffle_a_scale`` and the B scale is the
    ``[N/32, K/32]`` exponent bytes K-block major, which is exactly what
    sglang's ``prepare_mxfp8_native_weight`` leaves on the layer.
    """
    with GemmAllReduceOp(comm, n=n, k=k, m_max=m, quant="mxfp8") as op:
        op.self_test()
        a, b, ea, eb = _mxfp8_operands(rank, m, n, k)
        got = op(
            a,
            preshuffle_b(b),
            preshuffle_a_scale(ea),
            eb.t().reshape(-1).contiguous(),
        ).clone()
    _emit(
        rank,
        case="mxfp8",
        rel_l2=_rel_l2(got, _mxfp8_reference(world, m, n, k)),
        m=m,
    )


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


#: Every case this worker knows, and the operands each wants, so the whole file
#: can be served by one spawn.
#:
#: Six separate 2-rank launches is what it cost before, and the queues do not
#: recycle that fast: back-to-back 8-rank jobs fail in hsaKmtCreateQueueExt
#: (anvil.cpp:237), and the ones that fell over were as often the *neighbouring*
#: file's as this one's. One process, one communicator, every case.
ALL_CASES = [
    ("scale_order", {"m": 512}),
    ("alternating_m", {"m_small": 512, "m_large": 1024}),
    ("alternating_m", {"m_small": 2048, "m_large": 4096}),
    ("changing_data", {"m": 512, "calls": 5}),
    ("self_test", {"m": 1024}),
]


def run_all(comm, rank, world, n, k):
    """Every case in one process, sharing one communicator.

    m_max covers the largest M any case asks for, so one op serves them all --
    which is also closer to how a server uses it than a fresh op per shape.
    """
    m_max = max(max(kw.get("m", 0), kw.get("m_large", 0)) for _, kw in ALL_CASES)
    with GemmAllReduceOp(comm, n=n, k=k, m_max=m_max) as op:
        for case, kw in ALL_CASES:
            if case == "scale_order":
                case_scale_order(op, rank, world, kw["m"], n, k)
            elif case == "alternating_m":
                case_alternating_m(op, rank, world, kw["m_small"], kw["m_large"], n, k)
            elif case == "changing_data":
                case_changing_data(op, rank, world, kw["m"], n, k, kw["calls"])
            elif case == "self_test":
                case_self_test(op, rank, world, n, k)
    # Both of these need their own op, so they come after the shared one is
    # released: mxfp8 compiles at a different BLOCK_M, and close() is about the
    # lifecycle.
    case_mxfp8(comm, rank, world, world * 256, n, k)
    case_close_is_idempotent(comm, rank, n, k, 512)


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
    if args.case == "all":
        # The cases carry their own shapes; the command line has none.
        m_max = max(max(kw.get("m", 0), kw.get("m_large", 0)) for _, kw in ALL_CASES)
    else:
        m_max = max(args.m, args.m_large, args.m_small)
    vmm = 2 * GemmAllReduceOp.window_bytes_for(world, m_max=m_max, n=args.n) + (
        256 << 20
    )
    with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
        if args.case == "all":
            run_all(comm, rank, world, args.n, args.k)
            return 0
        if args.case == "close":
            case_close_is_idempotent(comm, rank, args.n, args.k, m_max)
            return 0
        with GemmAllReduceOp(comm, n=args.n, k=args.k, m_max=m_max) as op:
            if args.case == "scale_order":
                case_scale_order(op, rank, world, args.m, args.n, args.k)
            elif args.case == "self_test":
                case_self_test(op, rank, world, args.n, args.k)
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
