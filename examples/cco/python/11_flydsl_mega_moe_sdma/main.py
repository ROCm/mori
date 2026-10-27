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
"""CCO Example 11 — MoE GEMM2 fused with an SDMA combine (gfx950)

MegaMoE's GEMM2 normally scatters every output row straight into the origin
rank's ``comb_inp`` with P2P stores from the compute units. Here GEMM2 writes its
rows into a *local* send buffer grouped by origin rank, and cco's SDMA copy engine
ships each rank's run -- one contiguous ``put`` per peer -- so the transfer does
not occupy CUs:

    S = [ origin rank 0: rows | origin rank 1: rows | ... ]   (expert-major inside a run)

Three kernels on one stream (see ``sorted_combine.py``; the P2P baseline has two):

    1. rank_sort_plan   count -> scan -> assign: a stable send-buffer index per row
    2. gemm2            writes S[out_row] (``moe_gemm2_localsort.py``); extra comm waves,
                        one per peer, wait for a chunk of sort tiles to be written (a
                        per-chunk counter the GEMM workgroups bump after their stores) and
                        put that chunk for their peer on their own SDMA queue while the
                        remaining tiles are still computing
    3. combine          wait for the peers' flags, build slot -> row, top-k reduce

The example runs MegaMoEV2 (aiter) once to get the reference output and the
stage1/dispatch state, then redoes stage2 + transfer + reduce on the same state and
compares.

Needs aiter (MegaMoEV2 and the GEMM2 mainloop), FlyDSL, and mori built with SDMA:

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 main.py
"""

import argparse
import os

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "20G")
os.environ.setdefault("MORI_ENABLE_SDMA", "1")

import torch  # must be imported before anything that pulls in mori.cco
import torch.distributed as dist

import mori.shmem as ms

import aiter
from aiter import dtypes
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

from sorted_combine import SortedCombine

NETWORKS = {
    "glm52": dict(
        model_dim=6144, inter_dim=2048, experts=256, topk=8, swiglu_limit=0.0
    ),
    "v4_pro": dict(
        model_dim=7168, inter_dim=3072, experts=384, topk=6, swiglu_limit=10.0
    ),
}


def _next_pow2(v):
    return 1 << (int(v) - 1).bit_length()


def _make_inputs(tokens, model_dim, experts, topk, rank, seed, device):
    g = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn(
        (tokens, model_dim), dtype=torch.bfloat16, device=device, generator=g
    )
    scores = torch.randn(
        (tokens, experts), dtype=torch.float32, device=device, generator=g
    )
    values, ids = torch.topk(scores, topk, dim=-1)
    return (
        x.contiguous(),
        values.softmax(dim=-1).contiguous(),
        ids.to(torch.int32).contiguous(),
    )


def _quantize_weights(model_dim, inter_dim, local_experts, rank, seed, device):
    g = torch.Generator(device=device).manual_seed(seed + 1000 + rank)
    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x32)

    w1 = torch.randn(
        (local_experts, 2 * inter_dim, model_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=g,
    ).mul_(model_dim**-0.25)
    w1_q, w1_scale = quantize(w1, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(local_experts, 2 * inter_dim, model_dim // 2)
    del w1
    w2 = torch.randn(
        (local_experts, model_dim, inter_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=g,
    ).mul_(inter_dim**-0.25)
    w2_q, w2_scale = quantize(w2, quant_dtype=dtypes.fp4x2)
    w2_q = w2_q.view(local_experts, model_dim, inter_dim // 2)
    del w2
    torch.cuda.empty_cache()
    return (
        shuffle_weight_a16w4(w1_q, 16, True).contiguous(),
        shuffle_scale_a16w4(w1_scale, local_experts, True).contiguous(),
        shuffle_weight_a16w4(w2_q, 16, False).contiguous(),
        shuffle_scale_a16w4(w2_scale, local_experts, False).contiguous(),
    )


def _setup():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    # gloo only: no NCCL, so this also runs where /dev/shm is too small for it.
    dist.init_process_group("gloo")
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return rank, world, torch.device("cuda", local)


def _timeit(fn, iters):
    torch.cuda.synchronize()
    dist.barrier()
    fn()
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=NETWORKS, default="glm52")
    parser.add_argument("--bs", type=int, default=128, help="tokens per rank")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument(
        "--chunk-tiles",
        type=int,
        default=4,
        help="sort tiles per shipped chunk",
    )
    parser.add_argument("--queues", type=int, default=8)
    args = parser.parse_args()

    rank, world, device = _setup()
    try:
        net = NETWORKS[args.network]
        w1, w1s, w2, w2s = _quantize_weights(
            net["model_dim"],
            net["inter_dim"],
            net["experts"] // world,
            rank,
            args.seed,
            device,
        )
        x, wts, ids = _make_inputs(
            args.bs,
            net["model_dim"],
            net["experts"],
            net["topk"],
            rank,
            args.seed,
            device,
        )
        moe = MegaMoEV2(
            rank=rank,
            world_size=world,
            quant="a8w4",
            w1=w1,
            w1_scale=w1s,
            w2=w2,
            w2_scale=w2s,
            max_tok_per_rank=max(16, _next_pow2(args.bs)),
            fanout_masks=(),
            **net,
        )
        ref = moe(x, wts, ids)[: args.bs].clone()
        torch.cuda.synchronize()
        config = moe._active_config
        if config.stage2.aligned_pair:
            raise SystemExit(
                "this config selects aligned-pair stage2, which SortedCombine does not cover"
            )
        if rank == 0:
            print(
                f"reference: p2p_quant={config.p2p_quant} BM={moe._g2_active_block_m} "
                f"SBM={moe._s1_active_tile_m}",
                flush=True,
            )

        ref_ms = _timeit(
            lambda: moe._run_stage2(args.bs, None, True, config), args.iters
        )
        failures = []
        sc = SortedCombine(moe, queues=args.queues, chunk_tiles=args.chunk_tiles)
        worst = 0.0
        # Repeated calls with no host sync between them also exercise the epoch counter,
        # the flags and the "consumed" handshake that stops a fast rank overrunning a slow one.
        for _ in range(4):
            out = sc.run(args.bs, config).clone()
            torch.cuda.synchronize()
            diff = out.float() - ref.float()
            worst = max(worst, float(diff.norm() / ref.float().norm().clamp_min(1e-9)))
        t = torch.tensor([worst], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        worst = float(t.item())
        ms_per = _timeit(lambda: sc.run(args.bs, config), args.iters)
        ok = worst < args.rtol
        if not ok:
            failures.append("sdma")
        if rank == 0:
            print(
                f"relL2={worst:.3e} {'PASS' if ok else 'FAIL'} "
                f"time={ms_per:.4f}ms (reference stage2+combine {ref_ms:.4f}ms)",
                flush=True,
            )
        sc.close()
        if failures:
            raise AssertionError(f"relL2 above {args.rtol} for {failures}")
        if rank == 0:
            print("SUCCESS", flush=True)
    finally:
        try:
            ms.shmem_finalize()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
