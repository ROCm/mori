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
"""TP8/TP4 wo_b public-op regression, run with torch.distributed.run.

Checks mixed-M graph replay, changing inputs, window guards and the actual
FP8 payload/scales on both communication legs. No performance thresholds.
"""

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from mori.cco import Communicator
from mori.ops.gemm_ar import GemmAllReduceOp, preshuffle_a_scale, preshuffle_b
from mori.tensor_utils import from_gpu_ptr

from gemm_ar_wire_reference import check_wire

sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "benchmark/cco/flydsl/gemm_ar")
)
from bench_gemm_ar import (
    _setup_distributed,
    make_operands,
    reference_partial,
)  # noqa: E402
from timing import _graph  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", choices=("blockscale", "mxfp8"), required=True)
    parser.add_argument("--scatter", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--gather", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--schedule", choices=("default", "wo_b"), default="wo_b")
    args = parser.parse_args()
    _, rank, world, uid = _setup_distributed()
    quant = args.quant
    n, k = (7168 if quant == "blockscale" else 5120), 2048
    assert world == (8 if quant == "blockscale" else 4)
    ms = [4200, 8200, 11264, 16384]
    if world == 8:
        ms += [13312, 9216]  # same physical M as tuned 8200, different counters
    settings = dict(
        n=n,
        k=k,
        m_max=16384,
        quant=quant,
        schedule=args.schedule,
        scatter_dtype=args.scatter,
        gather_dtype=args.gather,
    )
    refs, operands = {}, {}
    for m in ms:
        operands[m] = make_operands(rank, m, n, k, quant)
        ref = torch.zeros(m, n, device="cuda", dtype=torch.float32)
        for peer in range(world):
            ref += reference_partial(*make_operands(peer, m, n, k, quant), quant)
        refs[m] = ref
    with Communicator.init(world, rank, uid, per_rank_vmm=8 << 30) as comm:
        # Reserve explicit guard bytes: AllocatedMemory.size is the requested
        # size, not the allocator's physical rounding. The op still reports
        # and uses its unmodified window_bytes layout.
        alloc_mem = comm.alloc_mem
        comm.alloc_mem = lambda size: alloc_mem(size + 4096)
        with GemmAllReduceOp(comm, **settings) as op:
            comm.alloc_mem = alloc_mem
            assert op.window_bytes == GemmAllReduceOp.window_bytes_for(
                world, **settings
            )
            guard = from_gpu_ptr(
                op.mem.ptr + op.window_bytes,
                (op.mem.size - op.window_bytes,),
                torch.uint8,
            )
            guard.fill_(165)
            op.self_test(4200)
            calls, outputs, buffers, graphs, plans = {}, {}, [], {}, []
            for m in ms:
                a, b, sa, sb = operands[m]
                mp = op.padded_m(m)
                # Test the public padding helper; preserve each buffer for graphs.
                ap = op.pad_rows(a, mp).clone()
                buffers.append(ap)
                group = 32 if quant == "mxfp8" else 128
                sp = torch.full(
                    (mp, k // group),
                    127 if quant == "mxfp8" else 1,
                    device="cuda",
                    dtype=sa.dtype,
                )
                sp[:m] = sa
                asc = (
                    preshuffle_a_scale(sp) if quant == "mxfp8" else sp.t().contiguous()
                )
                bsc = (
                    sb.t().contiguous().to(torch.int32).reshape(-1)
                    if quant == "mxfp8"
                    else sb
                )
                bp = preshuffle_b(b)

                def call(ap=ap, bp=bp, asc=asc, bsc=bsc, m=m):
                    return op(ap, bp, asc, bsc, logical_m=m)

                calls[m] = call
                outputs[m] = call()[:m]
                graphs[m] = _graph(call, reps=2)
                cfg = op._make_cfg(mp, m)
                plans.append(
                    dict(
                        m=m,
                        physical_m=mp,
                        chunks=cfg.counter_chunks,
                        slot=cfg.counter_shape_index,
                    )
                )
            worst = 0.0
            for step, sign in enumerate((1, -1, 1)):
                if step:
                    for buf in buffers:
                        buf.view(torch.uint8).bitwise_xor_(128)
                for m in [*ms, ms[0]]:
                    graphs[m].replay()
                    torch.cuda.synchronize()
                    error = (
                        (outputs[m].float() - refs[m] * sign).norm() / refs[m].norm()
                    ).item()
                    limit = (
                        0.045
                        if args.scatter == "fp8" or args.gather == "fp8"
                        else 0.003
                    )
                    assert error < limit, (rank, m, sign, error)
                    worst = max(worst, error)
            wires = []
            if args.scatter == "fp8":
                for m in (4200, 8200):
                    mp = op.padded_m(m)
                    key = op._row_plan(mp, m)
                    plan = op._compiled(mp, m)
                    saved = []

                    def save_partial(dc, win, *, stream):
                        saved.append(plan.input.clone())

                    op._cache[key] = plan._replace(tail=(save_partial, *plan.tail))
                    output = calls[m]()
                    torch.cuda.synchronize()
                    op._cache[key] = plan
                    wires.append(
                        dict(
                            m=m,
                            **check_wire(
                                op._make_cfg(mp, m),
                                op.mem.ptr,
                                saved[0],
                                output,
                                rank,
                                world,
                            ),
                        )
                    )
            assert bool(torch.all(guard == 165)), ("guard", rank)
            assert guard.numel() > 0, "allocation must leave room for a tail guard"
            values = [None] * world
            dist.all_gather_object(values, worst)
            if rank == 0:
                print(
                    "RESULT_JSON "
                    + json.dumps(
                        dict(
                            quant=quant,
                            schedule=args.schedule,
                            scatter=args.scatter,
                            gather=args.gather,
                            plans=plans,
                            per_rank_max_rel_l2=values,
                            wire_checks=wires,
                            graph_replay=True,
                            changed_inputs=True,
                            guards=True,
                            window_bytes=op.window_bytes,
                        )
                    ),
                    flush=True,
                )
            torch.cuda.synchronize()
            dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
