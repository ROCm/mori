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
"""Latency: aiter MegaMoEV2 stage2+combine vs the rank-sorted SDMA path (gfx950)

Baseline: aiter's ``MegaMoEV2`` (mori.shmem symmetric heap). Its fused GEMM2 scatters
every output row into the origin rank's ``comb_inp`` with P2P stores from the compute
units, then a combine kernel runs a cross-rank barrier and the top-k reduce. No cco,
no LSA, no copy engine.

New path (``sorted_combine.SortedCombine``), three kernels: a plan kernel, GEMM2 writing
a rank-sorted local send buffer with comm waves inside it that put finished chunks with
cco SDMA while the tiles are still computing, and a combine kernel that waits for the
peers and does an indexed top-k reduce.

Per token count it reports, as mean/max over ranks:

    stage2+combine   baseline vs the new path (the part that is replaced)
    end to end       stage1 + that (dispatch and stage1 are shared, so e2e shows
                     how much of the layer the replaced part is)
    breakdown        per-stage time of one eager call (rank 0), to see where the
                     new path spends its time

CUDA Graph timing by default (what the production path uses); ``--no-graph`` times
eager launches.

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 bench.py \\
        --network glm52 --bs-list 16,128,1024,4096
"""

import argparse

from main import NETWORKS, _make_inputs, _next_pow2, _quantize_weights, _setup

import torch
import torch.distributed as dist

import mori.shmem as ms

from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

from sorted_combine import SortedCombine


def _over_ranks(local_ms):
    """(mean, max) of a per-rank time over all ranks."""
    t = torch.tensor([local_ms], dtype=torch.float64)
    mx = t.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    return float(t.item()) / dist.get_world_size(), float(mx.item())


def _sync():
    torch.cuda.synchronize()
    dist.barrier()


def _time(fn, iters, graph):
    """Mean/max-over-ranks ms per call. Falls back to eager if the graph cannot be captured."""
    _sync()
    fn()
    _sync()
    replay = fn
    if graph:
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=torch.cuda.Stream()):
                fn()
            replay = g.replay
        except Exception as exc:  # noqa: BLE001
            if dist.get_rank() == 0:
                print(
                    f"  (graph capture failed, timing eagerly: {type(exc).__name__}: {exc})",
                    flush=True,
                )
            torch.cuda.synchronize()
    for _ in range(5):
        replay()
    _sync()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iters):
        replay()
    end.record()
    torch.cuda.synchronize()
    return _over_ranks(start.elapsed_time(end) / iters)


def _breakdown(sc, bs, config):
    """Per-stage ms of one eager call, averaged over ranks."""
    sc.run(bs, config)  # warm
    _sync()
    marks = []
    sc.run(bs, config, profile=marks)
    torch.cuda.synchronize()
    names = [n for n, _ in marks[1:]]
    ms_ = [marks[i][1].elapsed_time(marks[i + 1][1]) for i in range(len(marks) - 1)]
    t = torch.tensor(ms_, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return dict(zip(names, t.tolist()))


def _rel_l2(out, ref):
    diff = out.float() - ref.float()
    local = torch.tensor(
        [float(diff.norm() ** 2), float(ref.float().norm() ** 2)], dtype=torch.float64
    )
    dist.all_reduce(local)
    return float((local[0] / local[1].clamp_min(1e-30)).sqrt())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", choices=NETWORKS, default="glm52")
    parser.add_argument("--bs-list", default="16,128,1024,4096", help="tokens per rank")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument(
        "--chunk-tiles", default="4", help="fused chunk sizes to sweep, e.g. 2,4,8,16"
    )
    parser.add_argument("--queues", type=int, default=8)
    parser.add_argument("--no-graph", action="store_true")
    args = parser.parse_args()

    batch_sizes = [int(v) for v in args.bs_list.split(",")]
    chunk_list = [int(v) for v in args.chunk_tiles.split(",")]
    graph = not args.no_graph

    rank, world, device = _setup()
    rows = []  # (bs, label, s2_mean, s2_max, e2e_mean, e2e_max, relL2)
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
        max_bs = max(batch_sizes)
        x_all, wts_all, ids_all = _make_inputs(
            max_bs,
            net["model_dim"],
            net["experts"],
            net["topk"],
            rank,
            args.seed,
            device,
        )
        # One operator and one SortedCombine for every token count: building more would
        # re-create symmetric heaps and SDMA queues.
        moe = MegaMoEV2(
            rank=rank,
            world_size=world,
            quant="a8w4",
            w1=w1,
            w1_scale=w1s,
            w2=w2,
            w2_scale=w2s,
            max_tok_per_rank=max(16, _next_pow2(max_bs)),
            fanout_masks=(),
            **net,
        )
        sc = SortedCombine(moe, queues=args.queues)

        for bs in batch_sizes:
            x, wts, ids = (
                x_all[:bs].contiguous(),
                wts_all[:bs].contiguous(),
                ids_all[:bs].contiguous(),
            )
            ref = moe(x, wts, ids)[:bs].clone()
            torch.cuda.synchronize()
            config = moe._active_config
            if rank == 0:
                print(
                    f"\n=== bs={bs} tokens/rank  p2p_quant={config.p2p_quant} "
                    f"BM={moe._g2_active_block_m} SBM={moe._s1_active_tile_m} "
                    f"aligned_pair={config.stage2.aligned_pair} ===",
                    flush=True,
                )
            if config.stage2.aligned_pair:
                if rank == 0:
                    print(
                        "  aligned-pair stage2 is not covered by SortedCombine; skipped",
                        flush=True,
                    )
                continue

            x_q, scale = moe.quantize(x)

            def stage1():
                moe._run_fused_stage1(x_q, wts, scale, ids)

            def base_s2():
                moe._run_stage2(bs, None, True, config)

            def base_e2e():
                moe(x, wts, ids)

            s1 = _time(stage1, args.iters, graph)
            stage1()
            _sync()
            b_s2 = _time(base_s2, args.iters, graph)
            b_e2e = _time(base_e2e, args.iters, graph)
            rows.append((bs, "aiter MegaMoEV2 (P2P scatter)", *b_s2, *b_e2e, None))
            if rank == 0:
                print(
                    f"  stage1 alone                     {s1[0]:9.4f} / {s1[1]:9.4f} ms (mean/max)",
                    flush=True,
                )
                print(
                    f"  baseline stage2+combine          {b_s2[0]:9.4f} / {b_s2[1]:9.4f} ms",
                    flush=True,
                )
                print(
                    f"  baseline end to end              {b_e2e[0]:9.4f} / {b_e2e[1]:9.4f} ms",
                    flush=True,
                )

            stage1()
            _sync()
            for ct in chunk_list:
                sc.set_chunk_tiles(ct)
                label = f"SDMA 3-kernel CT={ct}"
                out = sc.run(bs, config).clone()
                torch.cuda.synchronize()
                rel = _rel_l2(out, ref)
                n_s2 = _time(lambda: sc.run(bs, config), args.iters, graph)

                def e2e():
                    stage1()
                    sc.run(bs, config)

                n_e2e = _time(e2e, args.iters, graph)
                rows.append((bs, label, *n_s2, *n_e2e, rel))
                br = _breakdown(sc, bs, config)
                if rank == 0:
                    parts = "  ".join(f"{k}={v:.3f}" for k, v in br.items())
                    print(
                        f"  {label:<22} stage2+combine {n_s2[0]:9.4f} / {n_s2[1]:9.4f} ms  "
                        f"e2e {n_e2e[0]:9.4f} / {n_e2e[1]:9.4f} ms  relL2={rel:.2e}",
                        flush=True,
                    )
                    print(f"      breakdown (eager, ms): {parts}", flush=True)
                if rel >= args.rtol:
                    raise AssertionError(
                        f"bs={bs} {label}: relL2={rel:.3e} exceeds {args.rtol}"
                    )

        if rank == 0:
            print(
                "\n| bs | path | stage2+combine mean ms | max ms | vs baseline | "
                "e2e mean ms | vs baseline | relL2 |"
            )
            print("|---:|---|---:|---:|---:|---:|---:|---:|")
            base = {}
            for bs, label, s2m, s2x, e2m, e2x, rel in rows:
                if rel is None:
                    base[bs] = (s2m, e2m)
                b_s2, b_e2e = base.get(bs, (s2m, e2m))
                print(
                    f"| {bs} | {label} | {s2m:.4f} | {s2x:.4f} | {b_s2 / s2m:.2f}x | "
                    f"{e2m:.4f} | {b_e2e / e2m:.2f}x | {'-' if rel is None else f'{rel:.1e}'} |"
                )
            print("\nSUCCESS", flush=True)
        sc.close()
    finally:
        try:
            ms.shmem_finalize()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
