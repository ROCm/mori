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
"""Ranks for ``test_gemm_ag_op.py``: every multi-rank case, one process group.

SDMA queues of a finished process group take seconds to come back, and a launch
per case is what loses that race; gemm_ar's and gemm_a2a's op workers are one
spawn for the same reason. Prints one ``RESULT_JSON`` line per case from rank 0,
each carrying the worst value over all ranks.

The communicator's SDMA queue count is fixed by its first dev-comm, so the
first op built here is a fused one (``max_chunks`` queues per peer); every later
op asks for at most that many.
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist

from mori.cco import Communicator, UniqueId
from mori.ops.gemm_ag import GemmAgOp, preshuffle_b
from mori.ops.gemm_ar import preshuffle_a_scale

M, N, K = 1024, 1024, 512
#: GPU-side stall for the held rank: tens of milliseconds, against a call of
#: well under one at these shapes.
HOLD_CYCLES = 50_000_000
SCALE_BK, MXFP8_BK = 128, 32


def _setup():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return rank, world, UniqueId.from_bytes(payload[0])


def operands(src, m, prec, salt=0, valid=None):
    """Rank ``src``'s A and its scale, and the replicated B and its scale, as
    the op takes them; plus what the reference needs. B is seeded identically
    on every rank -- the gather concatenates rows that mean the same thing."""
    in_dtype, quant = prec
    ga = torch.Generator(device="cuda").manual_seed(1234 + src + 7919 * salt)
    gb = torch.Generator(device="cuda").manual_seed(9999)
    t = torch.bfloat16 if in_dtype == "bf16" else torch.float8_e4m3fn
    a = (torch.randn(m, K, generator=ga, device="cuda") / 8).to(t)
    b = (torch.randn(N, K, generator=gb, device="cuda") / 8).to(t)
    if valid is not None:
        a[valid:] = 0
    if in_dtype == "bf16":
        return dict(a=a, b=b, sa=None, sb=None, ref=(a, b, None, None))
    if quant == "ptpc":
        sa = torch.rand(m, generator=ga, device="cuda") * 0.01 + 0.01
        sb = torch.rand(N, generator=gb, device="cuda") * 0.01 + 0.01
        return dict(a=a, b=preshuffle_b(b), sa=sa, sb=sb, ref=(a, b, sa, sb))
    if quant == "mxfp8":
        ea = torch.randint(120, 123, (m, K // MXFP8_BK), generator=ga, device="cuda")
        eb = torch.randint(
            120, 123, (N // MXFP8_BK, K // MXFP8_BK), generator=gb, device="cuda"
        )
        ea, eb = ea.to(torch.uint8), eb.to(torch.uint8)
        return dict(
            a=a,
            b=preshuffle_b(b),
            sa=preshuffle_a_scale(ea),
            sb=eb.to(torch.int32).t().reshape(-1).contiguous(),
            ref=(a, b, ea, eb),
        )
    kb = K // SCALE_BK
    sa = torch.rand(m, kb, generator=ga, device="cuda") * 0.01 + 0.01
    sb = torch.rand(N // SCALE_BK, kb, generator=gb, device="cuda") * 0.01 + 0.01
    sa = sa.t().contiguous().t()  # logical [M, K/128], column-major
    return dict(a=a, b=preshuffle_b(b), sa=sa, sb=sb, ref=(a, b, sa, sb))


def partial(prec, a, b, sa, sb):
    """One source's fp32 ``[m, N]``, mirroring its quantisation."""
    af, bf = a.float(), b.float()
    quant = prec[1] if prec[0] == "fp8" else None
    if quant is None:
        return af @ bf.T
    if quant == "ptpc":
        return (af @ bf.T) * sa[:, None] * sb[None, :]
    out = torch.zeros(a.shape[0], N, device="cuda")
    group = MXFP8_BK if quant == "mxfp8" else SCALE_BK
    for i in range(K // group):
        ks = slice(i * group, (i + 1) * group)
        if quant == "mxfp8":
            s_a = torch.exp2(sa[:, i].float() - 127.0)
            s_b = torch.exp2(sb[:, i].float() - 127.0).repeat_interleave(group)
        else:
            s_a, s_b = sa[:, i], sb[:, i].repeat_interleave(group)
        out += (af[:, ks] @ bf[:, ks].T) * s_a[:, None] * s_b[None, :]
    return out


def reference(world, m, prec, salt=0, valid=None):
    return torch.cat(
        [partial(prec, *operands(s, m, prec, salt, valid)["ref"]) for s in range(world)]
    )


def call(op, o):
    return op(o["a"], o["b"], o["sa"], o["sb"])


def rel(got, ref):
    return ((got.float() - ref).norm() / ref.norm()).item()


def worst(value, op=max):
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, value)
    return op(out)


def emit(rank, **record):
    if rank == 0:
        print("RESULT_JSON " + json.dumps(record, sort_keys=True), flush=True)


PRECISIONS = {
    "fp8-ptpc": (("fp8", "ptpc"), "bf16"),
    "fp8-blockscale": (("fp8", "blockscale"), "bf16"),
    "fp8-mxfp8": (("fp8", "mxfp8"), "bf16"),
    "bf16-bf16": (("bf16", "ptpc"), "bf16"),
    "bf16-fp32": (("bf16", "ptpc"), "fp32"),
}


def op_for(comm, prec_name, **kw):
    (in_dtype, quant), out_dtype = PRECISIONS[prec_name]
    return GemmAgOp(
        comm,
        n=N,
        k=K,
        m_max=M,
        in_dtype=in_dtype,
        quant=quant,
        out_dtype=out_dtype,
        **kw,
    )


def case_self_test(comm, rank, world):
    """It passes, and it fails when nothing is pushed."""
    from mori.ops.gemm_ag import build_sdma_phases
    import flydsl.expr as fx
    from mori.tensor_utils import from_gpu_ptr

    with op_for(comm, "fp8-ptpc") as op:  # fused: fixes the queue count first
        op.self_test()
        # The teeth: the same check with the drain, which pushes nothing.
        cfg = op._cfg(op.m_max)
        phases = build_sdma_phases(cfg, rank, queues=op.sdma_queues)
        recv = from_gpu_ptr(
            op.mem.ptr + cfg.recv_off, (world, op.m_max, N), torch.bfloat16
        )
        recv.zero_()
        recv[rank].fill_(float(rank + 1))
        torch.cuda.synchronize()
        comm.barrier()
        phases["drain"](
            op.dev_comm.ptr,
            op.win.handle,
            stream=fx.Stream(torch.cuda.current_stream()),
        )
        torch.cuda.synchronize()
        missed = all(
            not bool((recv[s] == float(s + 1)).all()) for s in range(world) if s != rank
        )
    emit(rank, case="self_test", passes=True, detects_missing_push=worst(missed, all))


def case_precisions_and_modes(comm, rank, world):
    """Every precision in both modes, two operand sets each, split equal to
    fused byte for byte (same GEMM tile, same tile computation)."""
    out = {}
    for name, (prec, _) in PRECISIONS.items():
        results = {}
        for mode in ("fused", "split"):
            # One tile for both modes, so their bytes are comparable.
            tile = (256, 256) if prec == ("fp8", "mxfp8") else (128, 256)
            with op_for(comm, name, mode=mode, tile=tile) as op:
                errs = []
                for salt in (0, 1):
                    got = call(op, operands(rank, M, prec, salt)).clone()
                    errs.append(rel(got, reference(world, M, prec, salt)))
                results[mode] = (max(errs), got)
        out[f"{name}_fused"] = worst(results["fused"][0])
        out[f"{name}_split"] = worst(results["split"][0])
        out[f"{name}_equal"] = worst(
            bool(torch.equal(results["fused"][1], results["split"][1])), all
        )
    emit(rank, case="precisions_and_modes", **out)


def case_bf16_tiles(comm, rank, world):
    """Every tile pick_tile may choose, fused and split, bf16 in / fp32 out."""
    out = {}
    prec = ("bf16", "ptpc")
    for tile in ((128, 128), (128, 256), (256, 128), (256, 256)):
        for mode in ("fused", "split"):
            with op_for(comm, "bf16-fp32", mode=mode, tile=tile, chunks=2) as op:
                got = call(op, operands(rank, M, prec, salt=2))
                out[f"{tile[0]}x{tile[1]}_{mode}"] = worst(
                    rel(got, reference(world, M, prec, salt=2))
                )
    emit(rank, case="bf16_tiles", **out)


def case_chunk_counts(comm, rank, world):
    """The chunked epilogue is deterministic: every count gives the same bytes."""
    prec = ("fp8", "ptpc")
    o = operands(rank, M, prec, salt=3)
    seen = {}
    for chunks in (1, 2, 4):
        with op_for(comm, "fp8-ptpc", mode="fused", chunks=chunks) as op:
            seen[chunks] = call(op, o).clone()
            used = op.chunks_for(M)
        seen[chunks] = (seen[chunks], used)
    ref = reference(world, M, prec, salt=3)
    emit(
        rank,
        case="chunk_counts",
        used=[seen[c][1] for c in (1, 2, 4)],
        equal=worst(all(torch.equal(seen[1][0], seen[c][0]) for c in (2, 4)), all),
        rel_l2=worst(max(rel(seen[c][0], ref) for c in (1, 2, 4))),
    )


def _back_to_back(comm, rank, world, war_credit, mode="fused"):
    """Calls with no host sync between them. The reader of each result -- a
    copy out of the op's buffer -- is held on one rank, a different one each
    call, so the other ranks start the next call while it has not read."""
    prec, calls = ("bf16", "ptpc"), 8
    staged = [operands(rank, M, prec, salt=40 + i) for i in range(calls)]
    outs = [torch.empty(world * M, N, device="cuda") for _ in range(calls)]
    with op_for(comm, "bf16-fp32", mode=mode, chunks=2, war_credit=war_credit) as op:
        call(op, staged[0])  # compile outside the timed race
        torch.cuda.synchronize()
        comm.barrier()
        for i in range(calls):
            c = call(op, staged[i])
            if i % world == rank:
                torch.cuda._sleep(HOLD_CYCLES)
            outs[i].copy_(c)
        torch.cuda.synchronize()
        comm.barrier()
    return [
        worst(rel(outs[i], reference(world, M, prec, 40 + i))) for i in range(calls)
    ]


def case_back_to_back(comm, rank, world):
    emit(
        rank,
        case="back_to_back",
        fused=_back_to_back(comm, rank, world, True),
        split=_back_to_back(comm, rank, world, True, mode="split"),
        fused_no_credit=_back_to_back(comm, rank, world, False),
    )


def _alternating(comm, rank, world, war_credit):
    """Two M through one ``auto`` instance, back to back with no host sync and
    one rank's reader held per call, so a peer is certainly a call ahead.

    recv is laid out for the current M, so a push for the next call lands on
    rows the held rank has not read yet unless the credit holds it. ``cus`` is
    lowered so that ``auto`` fuses M=1024 and splits M=512 at these sizes:
    consecutive calls then also switch path, on one window and one credit.
    """
    prec, ms = ("bf16", "ptpc"), (512, 1024) * 4
    staged = [operands(rank, m, prec, salt=50 + i) for i, m in enumerate(ms)]
    outs = [torch.empty(world * m, N, device="cuda") for m in ms]
    with op_for(comm, "bf16-fp32", mode="auto", war_credit=war_credit) as op:
        op.cus = 16
        fused = [op.fuses_at(m) for m in (512, 1024)]
        for i in range(2):  # compile both plans outside the race
            call(op, staged[i])
        torch.cuda.synchronize()
        comm.barrier()
        for i in range(len(ms)):
            c = call(op, staged[i])
            if i % world == rank:
                torch.cuda._sleep(HOLD_CYCLES)
            outs[i].copy_(c)
        torch.cuda.synchronize()
        comm.barrier()
    rels = [
        worst(rel(outs[i], reference(world, m, prec, 50 + i))) for i, m in enumerate(ms)
    ]
    return rels, fused


def case_alternating_m(comm, rank, world):
    rels, fused = _alternating(comm, rank, world, True)
    no_credit, _ = _alternating(comm, rank, world, False)
    emit(rank, case="alternating_m", rels=rels, fused=fused, no_credit=no_credit)


def case_many_m(comm, rank, world):
    """One fused instance through ten distinct M, then every election counter
    back at zero. The counters used to run on and be tested modulo the chunk's
    tile count, which took a counter set per M and refused a ninth M."""
    from mori.tensor_utils import from_gpu_ptr

    prec, ms = ("bf16", "ptpc"), tuple(128 * i for i in range(1, 11))
    rels = []
    with GemmAgOp(
        comm,
        n=N,
        k=K,
        m_max=max(ms),
        in_dtype="bf16",
        out_dtype="fp32",
        mode="fused",
    ) as op:
        op.cus = 16  # so the chunk count varies with M at these sizes
        for i, m in enumerate(ms):
            got = call(op, operands(rank, m, prec, salt=80 + i))
            rels.append(worst(rel(got, reference(world, m, prec, 80 + i))))
        torch.cuda.synchronize()
        cfg = op._cfg(op.m_max)
        ctr = from_gpu_ptr(
            op.mem.ptr + cfg.counter_off, (cfg.counter_set_bytes // 4,), torch.int32
        )
        left = worst(int(ctr.count_nonzero()))
        chunks = sorted({op.chunks_for(m) for m in ms})
    emit(rank, case="many_m", rels=rels, counters_left=left, chunks=chunks)


def case_padded_rows(comm, rank, world):
    prec, valid = ("bf16", "ptpc"), 700
    with op_for(comm, "bf16-fp32") as op:
        m = op.padded_m(valid)
        got = call(op, operands(rank, m, prec, salt=60, valid=valid)).clone()
    blocks = got.view(world, m, N)
    emit(
        rank,
        case="padded_rows",
        m=m,
        rel_l2=worst(rel(got, reference(world, m, prec, 60, valid))),
        padding_is_zero=worst(bool((blocks[:, valid:] == 0).all()), all),
    )


def case_operand_checks(comm, rank):
    """What the launch would reinterpret as bytes is refused up front."""
    refused = {}
    o = operands(rank, M, ("fp8", "ptpc"))
    with op_for(comm, "fp8-ptpc") as op:
        for name, args in (
            ("bf16_a", (o["a"].to(torch.bfloat16), o["b"], o["sa"], o["sb"])),
            ("ragged_m", (o["a"][:300], o["b"], o["sa"][:300], o["sb"])),
            ("wrong_n", (o["a"], o["b"][:768], o["sa"], o["sb"][:768])),
            ("missing_scale", (o["a"], o["b"], None, o["sb"])),
            ("fp16_scale", (o["a"], o["b"], o["sa"].half(), o["sb"])),
        ):
            try:
                op(*args)
                refused[name] = False
            except ValueError:
                refused[name] = True
    for name, kw in (
        ("bf16_with_quant", dict(in_dtype="bf16", quant="mxfp8")),
        ("fp8_fp32_out", dict(in_dtype="fp8", out_dtype="fp32")),
        ("n_not_256", dict(n=1000)),
    ):
        args = dict(n=N, k=K, m_max=M)
        args.update(kw)
        try:
            GemmAgOp(comm, **args).close()
            refused[name] = False
        except ValueError:
            refused[name] = True
    emit(rank, case="operand_checks", **refused)


def case_close(comm, rank):
    op = op_for(comm, "fp8-ptpc")
    op.close()
    op.close()
    try:
        op.self_test()
        refused = False
    except RuntimeError:
        refused = True
    emit(rank, case="close", refused_after_close=refused)


def main() -> int:
    rank, world, uid = _setup()
    # Every op here allocates a window; size the reservation for all of them.
    vmm = 48 * GemmAgOp.window_bytes_for(world, n=N, m_max=M, out_dtype="fp32") + (
        1 << 30
    )
    with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
        case_self_test(comm, rank, world)
        case_precisions_and_modes(comm, rank, world)
        case_bf16_tiles(comm, rank, world)
        case_chunk_counts(comm, rank, world)
        case_back_to_back(comm, rank, world)
        case_alternating_m(comm, rank, world)
        case_many_m(comm, rank, world)
        case_padded_rows(comm, rank, world)
        case_operand_checks(comm, rank)
        case_close(comm, rank)
    dist.barrier()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
