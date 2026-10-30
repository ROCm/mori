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
"""Benchmark EpDispatchCombineKiwiOp. Requires a BUILD_OPS_KIWI build.

    torchrun --nproc_per_node=8 tests/python/ops/bench_kiwi_dispatch_combine.py \
        --num-tokens 1,32,128,4096 --rcomp-backend auto

For multi-node runs launch one torchrun per node with the usual --nnodes /
--node_rank / --master_addr arguments.

Each rank times every iteration and keeps its median. Columns report the mean,
min, and max of those per-rank medians, plus the worst per-rank p90. Bandwidth
follows bench_dispatch_combine.py: per rank, received rows x hidden elements x
element size divided by the median latency, averaged over ranks. In graph mode
dispatch and combine are captured as separate CUDA graphs and replayed back to
back.
"""

import argparse
import importlib.util
import os
import sys

import torch
import torch.distributed as dist


def load_kiwi_module():
    try:
        from mori.ops import dispatch_combine_kiwi

        return dispatch_combine_kiwi
    except ImportError:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../../python/mori/ops/dispatch_combine_kiwi.py",
        )
        spec = importlib.util.spec_from_file_location("dispatch_combine_kiwi", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def fp8_dtype():
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    if arch.startswith("gfx942"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def make_inputs(rank, num_tokens, args, dispatch_dtype, num_experts, device):
    gen = torch.Generator(device="cpu").manual_seed(4321 + rank)
    tokens = torch.randn((num_tokens, args.hidden_dim), generator=gen).to(
        dispatch_dtype
    )
    indices = torch.stack(
        [
            torch.randperm(num_experts, generator=gen)[: args.topk]
            for _ in range(num_tokens)
        ]
    ).to(torch.int32)
    weights = torch.rand((num_tokens, args.topk), generator=gen, dtype=torch.float32)
    scales = torch.rand(
        (num_tokens, args.scale_dim), generator=gen, dtype=torch.float32
    )
    return tokens.to(device), weights.to(device), scales.to(device), indices.to(device)


class Pair:
    """One dispatch+combine pair over fixed inputs and a fixed combine buffer."""

    def __init__(self, op, inputs, combine_weights):
        self.op = op
        self.inputs = inputs
        self.use_combine_weights = combine_weights
        self.combine_input = op.get_registered_combine_input_buffer(op._combine_dtype)
        self.combine_weights = None
        self.dispatch_out = None

    def dispatch(self):
        tokens, weights, scales, indices = self.inputs
        self.dispatch_out = self.op.dispatch(tokens, weights, scales, indices)

    def combine(self):
        indices = self.inputs[3]
        self.op.combine(self.combine_input, self.combine_weights, indices)

    def prepare_combine_input(self):
        """Fill the combine input from the last dispatch, like an identity expert."""
        out, out_w, _, _, total_recv = self.dispatch_out
        torch.cuda.synchronize()
        count = int(total_recv.item())
        self.combine_input[:count].copy_(out[:count].to(self.combine_input.dtype))
        if self.use_combine_weights:
            self.combine_weights = out_w.clone()
        torch.cuda.synchronize()
        return count


def time_graphs(pair, iters):
    dispatch_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(dispatch_graph):
        pair.dispatch()
    combine_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(combine_graph):
        pair.combine()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    mids = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    dist.barrier()
    for i in range(iters):
        starts[i].record()
        dispatch_graph.replay()
        mids[i].record()
        combine_graph.replay()
        ends[i].record()
    torch.cuda.synchronize()
    disp_us = [s.elapsed_time(m) * 1000.0 for s, m in zip(starts, mids)]
    comb_us = [m.elapsed_time(e) * 1000.0 for m, e in zip(mids, ends)]
    del dispatch_graph, combine_graph
    return disp_us, comb_us


def time_eager(pair, iters):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    mids = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    dist.barrier()
    for i in range(iters):
        starts[i].record()
        pair.dispatch()
        mids[i].record()
        pair.combine()
        ends[i].record()
    torch.cuda.synchronize()
    disp_us = [s.elapsed_time(m) * 1000.0 for s, m in zip(starts, mids)]
    comb_us = [m.elapsed_time(e) * 1000.0 for m, e in zip(mids, ends)]
    return disp_us, comb_us


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def bench_one(op, args, num_tokens, dispatch_dtype, rank, world_size, device):
    num_experts = args.num_experts_per_rank * world_size
    inputs = make_inputs(rank, num_tokens, args, dispatch_dtype, num_experts, device)
    pair = Pair(op, inputs, args.combine_weights)

    pair.dispatch()
    recv = pair.prepare_combine_input()
    pair.combine()
    for _ in range(args.warmup):
        pair.dispatch()
        pair.combine()
    torch.cuda.synchronize()
    dist.barrier()

    if args.timing == "graph":
        disp_samples, comb_samples = time_graphs(pair, args.iters)
    else:
        disp_samples, comb_samples = time_eager(pair, args.iters)
    disp_us, comb_us = percentile(disp_samples, 0.5), percentile(comb_samples, 0.5)

    torch.cuda.synchronize()
    dist.barrier()
    op.reset()
    dist.barrier()

    disp_bytes = (
        recv
        * op._dispatch_hidden
        * torch.empty((), dtype=dispatch_dtype).element_size()
    )
    comb_bytes = (
        recv
        * op._combine_hidden
        * torch.empty((), dtype=op._combine_dtype).element_size()
    )
    local = {
        "recv": recv,
        "disp_us": disp_us,
        "comb_us": comb_us,
        "disp_p90": percentile(disp_samples, 0.9),
        "comb_p90": percentile(comb_samples, 0.9),
        "disp_gbs": disp_bytes / (disp_us * 1e3) if disp_us > 0 else 0.0,
        "comb_gbs": comb_bytes / (comb_us * 1e3) if comb_us > 0 else 0.0,
    }
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local)
    return gathered


def summarize(num_tokens, per_rank):
    def stats(key):
        values = [r[key] for r in per_rank]
        return sum(values) / len(values), min(values), max(values)

    disp_avg, disp_min, disp_max = stats("disp_us")
    comb_avg, comb_min, comb_max = stats("comb_us")
    disp_p90 = stats("disp_p90")[2]
    comb_p90 = stats("comb_p90")[2]
    disp_bw = stats("disp_gbs")[0]
    comb_bw = stats("comb_gbs")[0]
    recv_avg = stats("recv")[0]
    print(
        f"{num_tokens:>8} {recv_avg:>9.1f} "
        f"{disp_avg:>9.2f} {disp_min:>9.2f} {disp_max:>9.2f} {disp_p90:>9.2f} {disp_bw:>9.2f} "
        f"{comb_avg:>9.2f} {comb_min:>9.2f} {comb_max:>9.2f} {comb_p90:>9.2f} {comb_bw:>9.2f} "
        f"{disp_avg + comb_avg:>9.2f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-tokens",
        default="1,32,128,1024",
        help="comma-separated tokens per rank to sweep",
    )
    parser.add_argument(
        "--max-num-inp-token-per-rank",
        type=int,
        default=0,
        help="op capacity; 0 uses the largest --num-tokens value",
    )
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--num-experts-per-rank", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument(
        "--scale-dim",
        type=int,
        default=0,
        help="float32 scales per token (e.g. 56 for fp8)",
    )
    parser.add_argument(
        "--combine-weights",
        action="store_true",
        help="also reduce per-row weights in combine (off, like MORI's and KIWI's benchmarks)",
    )
    parser.add_argument("--timing", choices=("graph", "eager"), default="graph")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--rcomp-backend", default="auto", help="auto, rdma-only, or hybrid"
    )
    parser.add_argument("--num-blocks", type=int, default=0, help="0 picks it per call")
    parser.add_argument("--num-proxy-threads", type=int, default=4)
    parser.add_argument("--queue-capacity", type=int, default=2048)
    parser.add_argument("--device-name", default="", help="RDMA device, e.g. mlx5_0")
    args = parser.parse_args()

    token_counts = [int(t) for t in args.num_tokens.split(",") if t]
    max_tokens = args.max_num_inp_token_per_rank or max(token_counts)
    if max(token_counts) > max_tokens:
        raise ValueError("--num-tokens exceeds --max-num-inp-token-per-rank")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    dispatch_dtype = torch.bfloat16 if args.dtype == "bf16" else fp8_dtype()

    kiwi = load_kiwi_module()
    config = kiwi.EpDispatchCombineKiwiConfig(
        data_type=dispatch_dtype,
        rank=rank,
        world_size=world_size,
        hidden_dim=args.hidden_dim,
        scale_dim=args.scale_dim,
        scale_type_size=4 if args.scale_dim > 0 else 0,
        max_token_type_size=2,
        max_num_inp_token_per_rank=max_tokens,
        num_experts_per_rank=args.num_experts_per_rank,
        num_experts_per_token=args.topk,
    )
    op = kiwi.EpDispatchCombineKiwiOp(
        config,
        num_blocks=args.num_blocks,
        num_proxy_threads=args.num_proxy_threads,
        queue_capacity=args.queue_capacity,
        rcomp_backend=args.rcomp_backend,
        device_name=args.device_name,
    )

    if rank == 0:
        print(
            f"KIWI EP bench: world_size={world_size} dtype={args.dtype} "
            f"combine_dtype={op._combine_dtype} hidden={args.hidden_dim} topk={args.topk} "
            f"experts={args.num_experts_per_rank * world_size} scale_dim={args.scale_dim} "
            f"max_tokens={max_tokens} timing={args.timing} iters={args.iters} "
            f"rcomp_backend={args.rcomp_backend} num_blocks={args.num_blocks or 'auto'} "
            f"proxy_threads={args.num_proxy_threads} combine_weights={args.combine_weights}",
            flush=True,
        )
        print(
            f"{'tokens':>8} {'recv':>9} "
            f"{'disp_us':>9} {'min':>9} {'max':>9} {'p90':>9} {'disp_GB/s':>9} "
            f"{'comb_us':>9} {'min':>9} {'max':>9} {'p90':>9} {'comb_GB/s':>9} {'total_us':>9}",
            flush=True,
        )
    for num_tokens in token_counts:
        per_rank = bench_one(
            op, args, num_tokens, dispatch_dtype, rank, world_size, device
        )
        if rank == 0:
            summarize(num_tokens, per_rank)

    del op
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
