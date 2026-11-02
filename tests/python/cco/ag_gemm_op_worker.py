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
"""Ranks for ``test_ag_gemm_op.py``: exercises ``AgGemmOp`` itself.

One spawn serves every case, as gemm_ar's and gemm_a2a's op workers do: SDMA
queues of a finished process group take seconds to come back, and a launch per
case is what loses that race. Prints one ``RESULT_JSON`` line per case from
rank 0, each carrying the worst value over all ranks.
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist

from mori.cco import Communicator, UniqueId
from mori.ops.ag_gemm import AgGemmOp
from mori.ops.gemm_ar import preshuffle_b

N, K = 1024, 1024
M_MAX = 1024
#: GPU-side stall for the held rank: tens of milliseconds, against a call of
#: well under one at this shape.
HOLD_CYCLES = 50_000_000


def _setup():
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return rank, world, UniqueId.from_bytes(payload[0])


def tokens(rank: int, m: int, salt: int, valid: int | None = None):
    """Rank ``rank``'s fp8 tokens and per-token scale; rows past ``valid`` zero."""
    g = torch.Generator(device="cuda").manual_seed(1000 + rank + 7919 * salt)
    a = (torch.randn(m, K, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    sa = torch.rand(m, generator=g, device="cuda") * 0.01 + 0.01
    if valid is not None:
        a[valid:] = 0
    return a, sa


def weight(rank: int):
    """Rank ``rank``'s column shard: a different weight on every rank."""
    g = torch.Generator(device="cuda").manual_seed(5000 + rank)
    b = (torch.randn(N, K, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    sb = torch.rand(N, generator=g, device="cuda") * 0.01 + 0.01
    return b, sb


def reference(world, rank, m, salt, valid=None):
    parts = [tokens(s, m, salt, valid) for s in range(world)]
    a = torch.cat([p[0] for p in parts]).float()
    sa = torch.cat([p[1] for p in parts])
    b, sb = weight(rank)
    return (a @ b.float().T) * sa[:, None] * sb[None, :]


def rel(got, ref):
    return ((got.float() - ref).norm() / ref.norm()).item()


def worst(value, op=max):
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, value)
    return op(out)


def emit(rank, **record):
    if rank == 0:
        print("RESULT_JSON " + json.dumps(record, sort_keys=True), flush=True)


def case_self_test(op, rank, world):
    """It passes, and it fails when the push does not happen."""
    op.self_test()
    # The teeth: the same check with only the barriers run. Peers' slots keep
    # whatever they held -- zeros here -- which is what an inert put leaves.
    from mori.ops.ag_gemm import AgGemmConfig, build_phases
    import flydsl.expr as fx

    m = op.m_max
    cfg = AgGemmConfig(world_size=world, m_local=m, k=K, block_m=256)
    phases = build_phases(cfg, rank)
    slots = op._compiled(m).rows.view(world, m, K)
    slots.zero_()
    slots[rank].fill_(rank + 1)
    torch.cuda.synchronize()
    op.comm.barrier()
    stream = fx.Stream(torch.cuda.current_stream())
    for p in ("barrier0", "barrier1"):
        phases[p](op.dev_comm.ptr, op.win.handle, stream=stream)
    torch.cuda.synchronize()
    missed = all(not bool((slots[s] == s + 1).all()) for s in range(world) if s != rank)
    emit(rank, case="self_test", passes=True, detects_missing_push=worst(missed, all))


def case_padded_rows(op, rank, world):
    """A token count that is not a whole tile, padded with pad_rows."""
    valid = 700
    m = op.padded_m(valid)
    a, sa = tokens(rank, m, salt=1, valid=valid)
    b, sb = weight(rank)
    got = op(a, preshuffle_b(b), sa, sb).clone()
    ref = reference(world, rank, m, salt=1, valid=valid)
    blocks = got.view(world, m, N)
    pad_zero = bool((blocks[:, valid:] == 0).all())
    emit(
        rank,
        case="padded_rows",
        m=m,
        rel_l2=worst(rel(got, ref)),
        padding_is_zero=worst(pad_zero, all),
    )


def case_alternating_m(op, rank, world):
    """Two M through one instance, fresh operands each call."""
    b, sb = weight(rank)
    bs = preshuffle_b(b)
    out = {}
    for i, m in enumerate((512, 1024, 512, 1024)):
        a, sa = tokens(rank, m, salt=10 + i)
        out[f"call{i}_m{m}"] = worst(
            rel(op(a, bs, sa, sb), reference(world, rank, m, 10 + i))
        )
    emit(rank, case="alternating_m", **out)


def case_local_rows_and_out(op, rank, world):
    """Quantising into the window and passing ``out`` give the same bytes."""
    m = 512
    a, sa = tokens(rank, m, salt=20)
    b, sb = weight(rank)
    bs = preshuffle_b(b)
    copied = op(a, bs, sa, sb).clone()
    rows, scales = op.local_rows(m)
    rows.copy_(a)
    scales.copy_(sa)
    mine = torch.empty(world * m, N, device="cuda", dtype=torch.bfloat16)
    zero_copy = op(rows, bs, scales, sb, out=mine)
    emit(
        rank,
        case="local_rows_and_out",
        returned_out=worst(zero_copy.data_ptr() == mine.data_ptr(), all),
        same_bytes=worst(bool(torch.equal(copied, mine)), all),
        rel_l2=worst(rel(mine, reference(world, rank, m, 20))),
    )


def case_split_matches_fused(comm, fused_op, rank, world):
    """Split and fused run the same tile computation, so the bytes agree."""
    m = 1024
    a, sa = tokens(rank, m, salt=30)
    b, sb = weight(rank)
    bs = preshuffle_b(b)
    fused = fused_op(a, bs, sa, sb).clone()
    with AgGemmOp(comm, n=N, k=K, m_max=M_MAX, mode="split") as split_op:
        split = split_op(a, bs, sa, sb).clone()
    emit(
        rank,
        case="split_matches_fused",
        equal=worst(bool(torch.equal(fused, split)), all),
        rel_l2=worst(rel(split, reference(world, rank, m, 30))),
    )


def case_back_to_back(op, rank, world):
    """Calls with no host sync between them, a different rank held each time.

    The ranks not held run a call ahead, so their posters refill slots a held
    rank may still be reading -- what the closing barrier is for.
    """
    m, calls = 1024, 8
    b, sb = weight(rank)
    bs = preshuffle_b(b)
    staged = [tokens(rank, m, salt=40 + i) for i in range(calls)]
    outs = [
        torch.empty(world * m, N, device="cuda", dtype=torch.bfloat16)
        for _ in range(calls)
    ]
    torch.cuda.synchronize()
    op.comm.barrier()
    for i in range(calls):
        if i % world == rank:
            torch.cuda._sleep(HOLD_CYCLES)
        op(staged[i][0], bs, staged[i][1], sb, out=outs[i])
    torch.cuda.synchronize()
    errs = {
        f"call{i}": worst(rel(outs[i], reference(world, rank, m, 40 + i)))
        for i in range(calls)
    }
    emit(rank, case="back_to_back", **errs)


def case_operand_checks(op, rank):
    """What the launch would reinterpret as bytes is refused up front."""
    m = 512
    a, sa = tokens(rank, m, salt=0)
    b, sb = weight(rank)
    bs = preshuffle_b(b)
    refused = {}
    for name, args in (
        ("bf16_a", (a.to(torch.bfloat16), bs, sa, sb)),
        ("ragged_m", (a[:300], bs, sa[:300], sb)),
        ("wrong_n", (a, bs[:768], sa, sb[:768])),
        ("fp16_scale", (a, bs, sa.half(), sb)),
    ):
        try:
            op(*args)
            refused[name] = False
        except ValueError:
            refused[name] = True
    emit(rank, case="operand_checks", **refused)


def case_close(comm, rank):
    op = AgGemmOp(comm, n=N, k=K, m_max=512)
    op.close()
    op.close()
    try:
        op.local_rows(512)
        refused = False
    except RuntimeError:
        refused = True
    emit(rank, case="close", refused_after_close=refused)


def main() -> int:
    rank, world, uid = _setup()
    vmm = 4 * AgGemmOp.window_bytes_for(world, k=K, m_max=M_MAX) + (256 << 20)
    with Communicator.init(world, rank, uid, per_rank_vmm=vmm) as comm:
        with AgGemmOp(comm, n=N, k=K, m_max=M_MAX) as op:
            case_self_test(op, rank, world)
            case_padded_rows(op, rank, world)
            case_alternating_m(op, rank, world)
            case_local_rows_and_out(op, rank, world)
            case_back_to_back(op, rank, world)
            case_operand_checks(op, rank)
            case_split_matches_fused(comm, op, rank, world)
        case_close(comm, rank)
    dist.barrier()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
