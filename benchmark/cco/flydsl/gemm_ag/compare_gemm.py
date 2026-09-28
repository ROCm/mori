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
"""Paired native BF16-input/FP32-output GEMM comparison under CUDA graphs."""

import argparse
import json
import os
import statistics

from _bench_utils import positive_int
import bench_gemm_ag as bench
import flydsl.expr as fx
import torch
import torch.distributed as dist
from mori.ops.gemm_ag import ag_config, compile_bf16_gemm_ag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", type=positive_int, default=2048)
    parser.add_argument("--out-dim", dest="n", type=positive_int, default=2048)
    parser.add_argument("-k", type=positive_int, default=7168)
    parser.add_argument("--rounds", type=positive_int, default=5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=positive_int, default=101)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    m, n, k = args.m, args.n, args.k
    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    a = (torch.randn(m, k, device="cuda", generator=gen) / 8).bfloat16()
    b = (torch.randn(n, k, device="cuda", generator=gen) / 8).bfloat16()
    ref = a.float() @ b.float().T
    ct = torch.empty((m, n), device="cuda", dtype=torch.float32)
    a_arg = a.view(torch.int16).view(-1)
    b_arg = b.view(torch.int16).view(-1)
    cases = [
        ("torch_mm_fp32", lambda: torch.mm(a, b.T, out_dtype=torch.float32, out=ct), ct)
    ]

    def make_case(bm, bn, publish):
        cfg = ag_config(
            world_size=dist.get_world_size(),
            m=m,
            n=n,
            elem_bytes=4,
            block_m=bm,
            block_n=bn,
        )
        gemm = compile_bf16_gemm_ag(
            cfg,
            rank,
            K=k,
            BLOCK_M=bm,
            BLOCK_N=bn,
            out_dtype="fp32",
            peer_uncached=publish,
        )
        c = torch.empty((m, n), device="cuda", dtype=torch.float32)
        c_arg = c.view(-1)

        def call():
            gemm(
                a_arg,
                b_arg,
                c_arg,
                m,
                n,
                0,
                0,
                stream=fx.Stream(torch.cuda.current_stream()),
            )

        return (f"mori_{bm}x{bn}" + ("_publish" if publish else ""), call, c)

    for bm, bn, publish in [(256, 256, False), (128, 256, False), (128, 128, False)]:
        cases.append(make_case(bm, bn, publish))
    errors = {}
    for name, fn, output in cases:
        if rank == 0:
            print("VALIDATE", name, flush=True)
        output.fill_(float("nan"))
        fn()
        torch.cuda.synchronize()
        rel = ((output - ref).norm() / ref.norm()).item()
        assert rel < 1e-5, (rank, name, rel)
        errors[name] = rel
    # Compile and validate everything before any timed round. All ranks measure
    # the same case together; rotating the order reduces clock-ramp bias.
    dist.barrier()
    samples = {name: [] for name, _, _ in cases}
    for round_id in range(args.rounds):
        ordered = cases[round_id % len(cases) :] + cases[: round_id % len(cases)]
        for name, fn, _ in ordered:
            dist.barrier()
            if rank == 0:
                print("TIME", round_id, name, flush=True)
            us = bench._median_us(fn, args.warmup, args.iters, graph=True)
            if rank == 0:
                print("DONE", round_id, name, us, flush=True)
            samples[name].append(us)
        dist.barrier()
    record = dict(rank=rank, samples=samples, rel_l2=errors)
    all_records = [None] * dist.get_world_size()
    dist.all_gather_object(all_records, record)
    if rank == 0:
        summary = {}
        for name, _, _ in cases:
            per_round = [
                max(r["samples"][name][i] for r in all_records)
                for i in range(args.rounds)
            ]
            summary[name] = dict(
                us=statistics.median(per_round),
                max_rank_us_by_round=per_round,
                worst_rel_l2=max(r["rel_l2"][name] for r in all_records),
            )
        print(
            "COMPARE_JSON "
            + json.dumps(
                dict(
                    m=m,
                    n=n,
                    k=k,
                    ranks=dist.get_world_size(),
                    torch=torch.__version__,
                    method=f"median of {args.rounds} rounds; each round max across ranks of {args.iters}-event median, {args.warmup} warmups, single-GEMM CUDA graph",
                    summary=summary,
                    per_rank=all_records,
                )
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
