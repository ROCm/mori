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
"""Example 14: a whole EP MoE layer (dispatch, GEMM1, GEMM2, SDMA sends) in one persistent kernel.

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 main.py test --bs 8192
    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 main.py bench

``test``: the layer output against a float reference (dequantised weights
and bf16 activations), relative L2 over all ranks, for two inputs called alternately; and the same
input twice must give the same bytes. ``bench``: CUDA Graph time of the layer (quant +
kernel + combine), max over ranks.
"""

import argparse
import os
import sys

import torch  # before anything that pulls in mori.cco
import torch.distributed as dist
import torch.nn.functional as F

import mori.shmem as ms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mega_moe import MegaMoE  # noqa: E402
from weights import (  # noqa: E402
    e8m0_to_f32,
    fp4x2,
    mxfp4_to_f32,
    per_1x32_f4_quant,
    shuffle_scale_a16w4,
    shuffle_weight_a16w4,
)

#: DeepSeek-V4-Pro's MoE geometry (the shape this example was tuned for).
NETWORK = {
    "model_dim": 7168,
    "inter_dim": 3072,
    "experts": 384,
    "topk": 6,
    "swiglu_limit": 10.0,
}


def _setup():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=world, device_id=device
    )
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return rank, world, device


def _make_inputs(tokens, rank, seed, device):
    """bf16 tokens, softmax routing weights over the top-k of random scores, expert ids."""
    net = NETWORK
    g = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn(
        (tokens, net["model_dim"]), dtype=torch.bfloat16, device=device, generator=g
    )
    scores = torch.randn(
        (tokens, net["experts"]), dtype=torch.float32, device=device, generator=g
    )
    values, ids = torch.topk(scores, net["topk"], dim=-1)
    return x, values.softmax(dim=-1).contiguous(), ids.to(torch.int32).contiguous()


def _make_weights(local_experts, rank, seed, device):
    """mxfp4 weights in the kernel layout, and the plain ones for the reference."""
    K, inter = NETWORK["model_dim"], NETWORK["inter_dim"]
    g = torch.Generator(device=device).manual_seed(seed + 1000 + rank)
    w1 = torch.randn(
        (local_experts, 2 * inter, K), dtype=torch.bfloat16, device=device, generator=g
    )
    w1.mul_(K**-0.25)
    w1_q, w1_s = per_1x32_f4_quant(w1, quant_dtype=fp4x2)
    del w1
    w1_q = w1_q.view(local_experts, 2 * inter, K // 2)
    w2 = torch.randn(
        (local_experts, K, inter), dtype=torch.bfloat16, device=device, generator=g
    )
    w2.mul_(inter**-0.25)
    w2_q, w2_s = per_1x32_f4_quant(w2, quant_dtype=fp4x2)
    del w2
    w2_q = w2_q.view(local_experts, K, inter // 2)
    kernel = (
        shuffle_weight_a16w4(w1_q, 16, True).contiguous(),
        shuffle_scale_a16w4(w1_s, local_experts, True).contiguous(),
        shuffle_weight_a16w4(w2_q, 16, False).contiguous(),
        shuffle_scale_a16w4(w2_s, local_experts, False).contiguous(),
    )
    ref = (
        w1_q,
        w1_s.view(local_experts, 2 * inter, K // 32),
        w2_q,
        w2_s.view(local_experts, K, inter // 32),
    )
    torch.cuda.empty_cache()
    return kernel, ref


def _dequant(weight, scale, rows, cols):
    values = mxfp4_to_f32(weight).view(rows, cols)
    return values * e8m0_to_f32(scale).view(rows, cols // 32).repeat_interleave(
        32, dim=-1
    )


@torch.no_grad()
def _reference(x, wts, ids, ref_weights, rank, world):
    """float32 MoE layer over all ranks' tokens, this rank's experts; all-reduced, this rank's rows."""
    net = NETWORK
    K, inter, limit = net["model_dim"], net["inter_dim"], net["swiglu_limit"]
    gathered = []
    for t in (x, wts, ids):
        parts = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(parts, t)
        gathered.append(torch.cat(parts))
    x_all, wts_all, ids_all = gathered
    partial = torch.zeros((x_all.shape[0], K), dtype=torch.float32, device=x.device)
    w1_q, w1_s, w2_q, w2_s = ref_weights
    epr = net["experts"] // world
    for e in range(epr):
        rows, slots = torch.nonzero(ids_all == rank * epr + e, as_tuple=True)
        if rows.numel() == 0:
            continue
        w1 = _dequant(w1_q[e], w1_s[e], 2 * inter, K)
        w2 = _dequant(w2_q[e], w2_s[e], K, inter)
        inp = x_all[rows].float()
        gate = (inp @ w1[:inter].T).clamp(max=limit)
        up = (inp @ w1[inter:].T).clamp(-limit, limit)
        out = (F.silu(gate) * up) @ w2.T
        partial.index_add_(0, rows, out * wts_all[rows, slots, None])
    dist.all_reduce(partial)
    n = x.shape[0]
    return partial[rank * n : (rank + 1) * n]


def _rel_l2(out, ref):
    err = torch.sum((out.float() - ref) ** 2)
    nrm = torch.sum(ref**2)
    dist.all_reduce(err)
    dist.all_reduce(nrm)
    return float(torch.sqrt(err / nrm))


def _all_max(v):
    t = torch.tensor([float(v)], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _build(args, rank, world, device):
    net = NETWORK
    (w1, w1_s, w2, w2_s), ref = _make_weights(
        net["experts"] // world, rank, args.seed, device
    )
    moe = MegaMoE(
        rank=rank,
        world_size=world,
        model_dim=net["model_dim"],
        inter_dim=net["inter_dim"],
        experts=net["experts"],
        topk=net["topk"],
        max_tok_per_rank=args.max_tok,
        swiglu_limit=net["swiglu_limit"],
        w1=w1,
        w1_scale=w1_s,
        w2=w2,
        w2_scale=w2_s,
    )
    return moe, ref


def run_test(args, rank, world, device):
    moe, ref_w = _build(args, rank, world, device)
    inputs = [
        _make_inputs(args.bs, rank, args.seed + i * 7919, device) for i in range(2)
    ]
    refs = [_reference(*inp, ref_w, rank, world) for inp in inputs]
    worst, outs = 0.0, {}
    for call in range(
        4
    ):  # alternating inputs: the launch parity flips, nothing is reset
        i = call % 2
        out = moe(*inputs[i]).clone()
        worst = max(worst, _rel_l2(out, refs[i]))
        if i in outs:
            same = bool(torch.equal(out, outs[i]))
            if _all_max(0.0 if same else 1.0) > 0:
                raise AssertionError(f"call {call}: same input, different output bytes")
        outs[i] = out
    ok = worst <= args.rtol
    leaked = sorted(m for m in sys.modules if m == "aiter" or m.startswith("aiter."))
    assert not leaked, f"aiter was imported: {leaked[:5]}"
    if rank == 0:
        print(
            f"bs={args.bs} relL2={worst:.6f} (limit {args.rtol}) {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    moe.close()
    if not ok:
        raise AssertionError(f"relL2 {worst} > {args.rtol}")
    if rank == 0:
        print("SUCCESS", flush=True)


def _time_graph(fn, iters):
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    fn()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return _all_max(start.elapsed_time(end) / iters)


def run_bench(args, rank, world, device):
    moe, ref_w = _build(args, rank, world, device)
    rows = []
    # Every token count takes the first rows of one max-size input, as aiter's MegaMoE test
    # does: the timings then compare one to one (at 2048 / 4096 tokens a differently drawn
    # routing alone moved the layer time by 0.06 ms, from the per-rank load).
    full = _make_inputs(args.max_tok, rank, args.seed, device)
    for bs in [int(v) for v in args.bs_list.split(",")]:
        inp = tuple(t[:bs].contiguous() for t in full)
        ms_ = [_time_graph(lambda: moe(*inp), args.iters) for _ in range(args.rounds)]
        # Graph replays must leave the protocol state valid: check the output after them.
        rel = _rel_l2(moe(*inp).clone(), _reference(*inp, ref_w, rank, world))
        rows.append((bs, min(ms_), max(ms_), rel))
        if rank == 0:
            print(
                f"bs={bs} layer={min(ms_):.4f}-{max(ms_):.4f} ms relL2={rel:.6f}",
                flush=True,
            )
    if rank == 0:
        print("\n| tokens/rank | layer (ms) | relL2 |\n|---:|---:|---:|")
        for bs, lo, hi, rel in rows:
            print(f"| {bs} | {lo:.3f}-{hi:.3f} | {rel:.4f} |")
    moe.close()
    if any(r[3] > args.rtol for r in rows):
        raise AssertionError("relL2 over the limit after the timed replays")
    if rank == 0:
        print("\nSUCCESS", flush=True)


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=123)
    common.add_argument("--max-tok", type=int, default=8192, help="max tokens per rank")
    common.add_argument(
        "--rtol", type=float, default=0.08, help="max relative L2 vs the reference"
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("test", parents=[common])
    t.add_argument("--bs", type=int, default=8192)
    b = sub.add_parser("bench", parents=[common])
    b.add_argument("--bs-list", default="2048,4096,8192")
    b.add_argument("--iters", type=int, default=20)
    b.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    rank, world, device = _setup()
    try:
        {"test": run_test, "bench": run_bench}[args.cmd](args, rank, world, device)
    finally:
        try:
            ms.shmem_finalize()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
