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
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""gfx1201 EP2/EP4: Graph Dispatch+Combine versus RCCL AllGather+ReduceScatter.

Run with torchrun --standalone --nproc_per_node=2 (or 4), MORI_RDNA4_EP=1,
MORI_EP_COMM=shmem and MORI_SHMEM_MODE=static_heap. Tokens are per rank.
Only communication is timed. MORI includes routing and optional weights;
the RCCL dense fallback transports hidden payloads. No expert compute is timed.
"""

import argparse
import gc
import itertools
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist

import mori
from tests.python.utils import TorchDistContext


def check_payload(actual, expected):
    # Bound comparison temporaries at the largest (65536 x 8192) shapes.
    for start in range(0, expected.size(0), 512):
        torch.testing.assert_close(
            actual[start : start + 512], expected[start : start + 512], rtol=0, atol=0
        )


def run_case(rank, world, dtype_name, hidden, tokens, topk, args):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    device = torch.device("cuda", rank)
    rng = torch.Generator(device=device).manual_seed(args.seed + rank)
    config = mori.ops.EpDispatchCombineConfig(
        data_type=dtype,
        rank=rank,
        world_size=world,
        gpu_per_node=world,
        hidden_dim=hidden,
        scale_dim=0,
        scale_type_size=1,
        max_token_type_size=2,
        max_num_inp_token_per_rank=tokens,
        num_experts_per_rank=args.experts // world,
        num_experts_per_token=topk,
        use_external_inp_buf=True,
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
    )
    op = mori.ops.EpDispatchCombineOp(config)
    # Dyadic values make exact checks independent of the reduction tree.
    inp = (
        torch.randint(
            -16, 16, (tokens, hidden), dtype=torch.int8, device=device, generator=rng
        ).to(dtype)
        / 16
    )
    indices = (
        torch.rand((tokens, args.experts), device=device, generator=rng)
        .topk(topk, dim=1)
        .indices.int()
    )
    weights = (
        torch.randint(1, 16, (tokens, topk), device=device, generator=rng).float() / 16
        if args.weights == "on"
        else None
    )
    membership = torch.zeros((tokens, world), device=device, dtype=torch.int32)
    membership.scatter_(1, (indices // config.num_experts_per_rank).long(), 1)
    all_membership = torch.empty(
        (world * tokens, world), device=device, dtype=torch.int32
    )
    all_indices = torch.empty((world * tokens, topk), device=device, dtype=torch.int32)
    gathered = torch.empty((world * tokens, hidden), device=device, dtype=dtype)
    dist.all_gather_into_tensor(all_membership, membership)
    dist.all_gather_into_tensor(all_indices, indices)
    dist.all_gather_into_tensor(gathered, inp)
    all_weights = None
    if weights is not None:
        all_weights = torch.empty((world * tokens, topk), device=device)
        dist.all_gather_into_tensor(all_weights, weights)
    received = int(all_membership[:, rank].sum().item())
    fanout = membership.sum(1)
    offsets = torch.arange(1, world + 1, device=device).float() / 16
    external = torch.empty((max(1, received), hidden), device=device, dtype=dtype)
    last = {}

    def dispatch():
        result = op.dispatch(inp, weights, None, indices)
        last["dispatch"] = result
        return result

    def combine(routed):
        last["combine"] = op.combine(
            external, routed[1] if weights is not None else None, indices
        )

    def check_dispatch(routed):
        assert int(routed[4].item()) == received
        flat = op.get_dispatch_src_token_pos()[:received].long()
        stride = op.max_num_tokens_to_send()
        assert bool(
            ((flat >= 0) & (flat // stride < world) & (flat % stride < tokens)).all()
        )
        ids = flat // stride * tokens + flat % stride
        assert ids.unique().numel() == received and bool(
            all_membership[ids, rank].all()
        )
        for start in range(0, received, 512):
            stop = min(start + 512, received)
            rows = ids[start:stop]
            torch.testing.assert_close(
                routed[0][start:stop], gathered[rows], rtol=0, atol=0
            )
            torch.testing.assert_close(
                routed[3][start:stop], all_indices[rows], rtol=0, atol=0
            )
            if weights is not None:
                torch.testing.assert_close(
                    routed[1][start:stop], all_weights[rows], rtol=0, atol=0
                )

    routed = dispatch()
    check_dispatch(routed)
    # First check row-dependent expert outputs, outside timing.
    torch.add(routed[0][:received], (rank + 1) / 16, out=external[:received])
    combine(routed)
    offset_sum = membership.float() @ offsets
    expected = torch.empty_like(inp)
    for start in range(0, tokens, 512):
        rows = slice(start, start + 512)
        expected[rows] = (
            inp[rows].float() * fanout[rows, None] + offset_sum[rows, None]
        ).to(dtype)
    check_payload(last["combine"][0][:tokens], expected)

    # Dispatch atomic row order can change on every call. Uniform per-rank
    # expert outputs remain valid across replays without a timed copy/GEMM.
    pattern = (torch.arange(hidden, device=device) % 17).to(dtype) / 16
    external.copy_(pattern[None, :] + (rank + 1) / 16)
    rs_input = torch.empty_like(gathered)
    torch.mul(
        all_membership[:, rank, None], pattern[None, :] + (rank + 1) / 16, out=rs_input
    )
    rs_output = torch.empty_like(inp)
    for start in range(0, tokens, 512):
        rows = slice(start, start + 512)
        expected[rows] = (
            fanout[rows, None] * pattern.float()[None, :] + offset_sum[rows, None]
        ).to(dtype)

    def check_mori():
        check_payload(last["combine"][0][:tokens], expected)
        if weights is not None:
            torch.testing.assert_close(
                last["combine"][1][:tokens], weights * fanout[:, None], rtol=0, atol=0
            )

    def all_gather():
        dist.all_gather_into_tensor(gathered, inp)

    def reduce_scatter(_):
        dist.reduce_scatter_tensor(rs_output, rs_input)

    def check_rccl():
        check_payload(gathered[rank * tokens : (rank + 1) * tokens], inp)
        check_payload(rs_output, expected)

    backends = {
        "mori": (dispatch, combine, check_mori),
        "rccl": (all_gather, reduce_scatter, check_rccl),
    }
    captures = {}
    for name, (first, second, check) in backends.items():
        for _ in range(args.warmup):
            second(first())
            torch.cuda.synchronize()
        check()
        dist.barrier()
        events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(3)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            events[0].record()
            routed = first()
            events[1].record()
            second(routed)
            events[2].record()
        for _ in range(args.warmup):
            graph.replay()
            torch.cuda.synchronize()
        check()
        captures[name] = graph, events
        dist.barrier()

    samples = {name: [] for name in backends}
    for rnd in range(args.rounds):
        for name in list(backends) if rnd % 2 == 0 else list(reversed(backends)):
            dist.barrier()
            graph, events = captures[name]
            values = []
            for _ in range(args.samples):
                graph.replay()
                torch.cuda.synchronize()
                values.append(
                    [
                        events[0].elapsed_time(events[1]),
                        events[1].elapsed_time(events[2]),
                        events[0].elapsed_time(events[2]),
                    ]
                )
            samples[name].append(values)
            backends[name][2]()
    routed = dispatch()
    check_dispatch(routed)
    combine(routed)
    check_mori()
    torch.cuda.synchronize()
    return dict(
        rank=rank,
        correctness="pass",
        samples_ms=samples,
        dispatch_kernel=op._cached_dispatch_kernel,
        combine_kernel=op._last_rdna4_combine_kernel,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topks", default="1,2,4,8,16,31,32,33,63,64")
    parser.add_argument("--hidden-sizes", default="2048,4096,8192")
    parser.add_argument("--tokens", default="64,1024,4096")
    parser.add_argument("--dtypes", default="bf16,fp16")
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--weights", choices=("on", "off"), default="on")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261225)
    parser.add_argument(
        "--output", type=Path, required=True, help="JSONL with raw per-rank samples"
    )
    args = parser.parse_args()
    topks = [int(k) for k in args.topks.split(",")]
    hiddens = [int(h) for h in args.hidden_sizes.split(",")]
    tokens = [int(t) for t in args.tokens.split(",")]
    dtypes = args.dtypes.split(",")
    world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if (
        world not in (2, 4)
        or int(os.environ["LOCAL_WORLD_SIZE"]) != world
        or args.experts <= 0
        or args.experts % world
    ):
        parser.error(
            "single-node EP2/EP4 requires a positive expert count divisible by EP"
        )
    if any(k < 1 or k > min(64, args.experts) for k in topks):
        parser.error("topk must be in 1..min(64, experts)")
    if any(h < 2048 or h > 8192 or h % 8 for h in hiddens):
        parser.error("hidden sizes must be in 2048..8192 and divisible by 8")
    if any(t < 1 for t in tokens) or min(args.warmup, args.rounds, args.samples) < 1:
        parser.error("tokens, warmup, rounds and samples must be positive")
    if any(d not in ("bf16", "fp16") for d in dtypes):
        parser.error("dtypes must be bf16 or fp16")
    if os.environ.get("MORI_RDNA4_EP") != "1":
        parser.error("set MORI_RDNA4_EP=1 before running this benchmark")
    context = TorchDistContext(
        rank,
        world,
        master_addr=os.environ["MASTER_ADDR"],
        master_port=os.environ["MASTER_PORT"],
    )
    context.__enter__()
    mori.shmem.shmem_torch_process_group_init("default")
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Refuse to overwrite an earlier benchmark's raw measurements.
        args.output.touch(exist_ok=False)
    for dtype, hidden, n, topk in itertools.product(dtypes, hiddens, tokens, topks):
        local = run_case(rank, world, dtype, hidden, n, topk, args)
        results = [None] * world if rank == 0 else None
        dist.gather_object(local, results, dst=0)
        if rank == 0:
            stats = {}
            for backend in ("mori", "rccl"):
                maxima = [
                    max(r["samples_ms"][backend][rnd][i][2] for r in results)
                    for rnd in range(args.rounds)
                    for i in range(args.samples)
                ]
                stats[backend] = dict(pair_p50_ms=statistics.median(maxima))
            reduction = 100 * (
                1 - stats["mori"]["pair_p50_ms"] / stats["rccl"]["pair_p50_ms"]
            )
            record = dict(
                world=world,
                dtype=dtype,
                hidden=hidden,
                tokens_per_rank=n,
                topk=topk,
                experts=args.experts,
                weights=args.weights,
                seed=args.seed,
                warmup=args.warmup,
                rounds=args.rounds,
                samples=args.samples,
                statistics=stats,
                latency_reduction_percent=reduction,
                raw_ranks=results,
                torch=torch.__version__,
                hip=torch.version.hip,
                rccl=list(torch.cuda.nccl.version()),
            )
            with args.output.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(
                f"EP{world} {dtype} H={hidden} T={n} K={topk}: latency reduction {reduction:.2f}%",
                flush=True,
            )
        dist.barrier()
        gc.collect()
        torch.cuda.empty_cache()
    mori.shmem.shmem_finalize()
    # On failure let torchrun terminate the peers; do not enter a cleanup barrier.
    context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
