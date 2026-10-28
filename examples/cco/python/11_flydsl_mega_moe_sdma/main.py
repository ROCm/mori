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
"""CCO Example 11 -- MegaMoE stage2 + combine with the transfer done by SDMA (gfx950)

MegaMoE's GEMM2 normally scatters every output row straight into the origin
rank's ``comb_inp`` with P2P stores from the compute units. Here GEMM2 writes its
rows into a *local* send buffer grouped by origin rank, and cco's SDMA copy engine
ships each rank's run -- one contiguous ``put`` per peer -- so the transfer does
not occupy CUs:

    S = [ origin rank 0: rows | origin rank 1: rows | ... ]   (expert-major inside a run)

The SDMA path is ``mega_moe_gemm2_with_sdma.MegaMoESdma`` (the op and its three kernels -- a plan
kernel, GEMM2 with comm waves, and a combine kernel -- all in
``mega_moe_gemm2_with_sdma.py``). This file is the
harness around aiter's ``MegaMoEV2``: it runs aiter once for stage1/dispatch and the
reference output, then redoes stage2 + transfer + reduce on the same state.

    test    the SDMA output against aiter's, on one token count; bit-exact when
            both sides move fp8 rows
    bench   latency of aiter's stage2+combine vs the SDMA path, per token count

Needs aiter (MegaMoEV2 and the GEMM2 mainloop), FlyDSL, and mori built with SDMA:

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 main.py test --bs 4096
    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 main.py bench \\
        --bs-list 128,1024,4096,8192

``bench`` times eager launches, which is what prefill does; ``--graph`` times CUDA
Graph replays with the two-graph slope used by ``bench_ep.py`` instead.
"""

import argparse
from dataclasses import replace
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

from mega_moe_gemm2_with_sdma import MegaMoESdma

NETWORKS = {
    "glm52": dict(
        model_dim=6144, inter_dim=2048, experts=256, topk=8, swiglu_limit=0.0
    ),
    "v4_pro": dict(
        model_dim=7168, inter_dim=3072, experts=384, topk=6, swiglu_limit=10.0
    ),
}


# --- Inputs, weights and the aiter operator -------------------------------------------


def _next_pow2(v):
    return 1 << (int(v) - 1).bit_length()


def _make_inputs(
    tokens, model_dim, experts, topk, rank, seed, device, hot_bias=0.0, hot_experts=0
):
    """``hot_bias`` raises the router score of the first ``hot_experts`` experts, i.e.
    skews routing toward rank 0; 0 is the uniform routing everything else uses."""
    g = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn(
        (tokens, model_dim), dtype=torch.bfloat16, device=device, generator=g
    )
    scores = torch.randn(
        (tokens, experts), dtype=torch.float32, device=device, generator=g
    )
    if hot_bias:
        scores[:, :hot_experts] += hot_bias
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


def _build_moe(args, max_tokens, rank, world, device):
    """aiter's operator, sized for ``max_tokens`` per rank, with its quantised weights."""
    net = NETWORKS[args.network]
    w1, w1s, w2, w2s = _quantize_weights(
        net["model_dim"],
        net["inter_dim"],
        net["experts"] // world,
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
        max_tok_per_rank=max(16, _next_pow2(max_tokens)),
        fanout_masks=(),
        **net,
    )
    return net, moe


def _reference(moe, x, wts, ids, no_aligned_pair):
    """aiter's own output for these inputs, and the stage2 config both paths then run."""
    ref = moe(x, wts, ids)[: x.shape[0]].clone()
    torch.cuda.synchronize()
    config = moe._active_config
    if no_aligned_pair and config.stage2.aligned_pair:
        config = replace(
            config,
            stage2=replace(config.stage2, aligned_pair=False, pair_cu=0, skew_cu=96),
        )
    return ref, config


def _make_sdma(moe, args, chunk_tiles):
    return MegaMoESdma(
        moe,
        queues=args.queues,
        chunk_tiles=chunk_tiles,
        quant=args.quant,
        row_slack=args.row_slack,
        signal_mode=args.signal_mode,
    )


# --- Timing ---------------------------------------------------------------------------


def _sync():
    torch.cuda.synchronize()
    dist.barrier()


def _over_ranks(local_ms):
    """(mean, max) of a per-rank time over all ranks."""
    t = torch.tensor([local_ms], dtype=torch.float64)
    mx = t.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    return float(t.item()) / dist.get_world_size(), float(mx.item())


#: Calls captured per graph for the slope measurement, as ``bench_ep.py`` does.
GRAPH_REPS = 10


def _replay_ms(g, n):
    """Mean ms of one ``g.replay()`` over ``n`` replays."""
    _sync()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(n):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


def _graph_of(fn, reps):
    """Capture ``reps`` back-to-back calls, warm it up, lockstep the ranks."""
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=torch.cuda.Stream()):
        for _ in range(reps):
            fn()
    _sync()
    for _ in range(5):
        g.replay()
    _sync()
    return g


def _time(fn, iters, graph=False, reps=GRAPH_REPS):
    """Mean/max-over-ranks ms per call.

    With ``graph``, this is the *marginal* cost of one call: two graphs holding
    ``reps`` and ``2*reps`` calls are replayed and differenced, so the per-replay
    launch overhead -- which at bs=16 is the same order as the work -- cancels
    instead of being charged to every call. Same two-graph slope as
    ``mori/tests/python/ops/dispatch_combine_v2/bench_ep.py``. Falls back to
    eager timing if the capture fails.
    """
    _sync()
    fn()
    _sync()
    if graph:
        try:
            g1, g2 = _graph_of(fn, reps), _graph_of(fn, 2 * reps)
            t1, t2 = _replay_ms(g1, iters), _replay_ms(g2, iters)
            return _over_ranks((t2 - t1) / reps)
        except Exception as exc:  # noqa: BLE001
            if dist.get_rank() == 0:
                print(
                    f"  (graph capture failed, timing eagerly: {type(exc).__name__}: {exc})",
                    flush=True,
                )
            torch.cuda.synchronize()
    for _ in range(5):
        fn()
    _sync()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iters):
        fn()
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


# --- test -----------------------------------------------------------------------------


def run_test(args, rank, world, device):
    net, moe = _build_moe(args, args.bs, rank, world, device)
    x, wts, ids = _make_inputs(
        args.bs,
        net["model_dim"],
        net["experts"],
        net["topk"],
        rank,
        args.seed,
        device,
        hot_bias=args.hot_bias,
        hot_experts=net["experts"] // world,
    )
    ref, config = _reference(moe, x, wts, ids, args.no_aligned_pair)
    if config.stage2.aligned_pair:
        raise SystemExit(
            "this config selects aligned-pair stage2, which MegaMoESdma does not cover"
        )
    if rank == 0:
        print(
            f"reference: p2p_quant={config.p2p_quant} BM={moe._g2_active_block_m} "
            f"SBM={moe._s1_active_tile_m}",
            flush=True,
        )
    ref_ms = _time(lambda: moe._run_stage2(args.bs, None, True, config), args.iters)[0]

    sc = _make_sdma(moe, args, args.chunk_tiles)
    worst = 0.0
    mismatched = 0
    # Repeated calls with no host sync between them also exercise the epoch counter,
    # the flags and the "consumed" handshake that stops a fast rank overrunning a slow one.
    for _ in range(4):
        out = sc.run(args.bs, config).clone()
        torch.cuda.synchronize()
        diff = out.float() - ref.float()
        worst = max(worst, float(diff.norm() / ref.float().norm().clamp_min(1e-9)))
        mismatched = max(mismatched, int((out != ref).any(dim=1).sum()))
    t = torch.tensor([worst, mismatched], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    worst, mismatched = float(t[0].item()), int(t[1].item())
    ms_per = _time(lambda: sc.run(args.bs, config), args.iters)[0]

    failures = []
    ok = worst < args.rtol
    # Both sides move fp8 rows in aiter's layout and reduce in the same order, so
    # they must agree exactly; relL2 alone would let a few wrong rows through.
    exact = config.p2p_quant == args.quant == "fp8_blockwise_1x32"
    if exact and mismatched:
        ok = False
    if not ok:
        failures.append("sdma")
    try:
        sc.check_overflow()
    except RuntimeError as e:
        failures.append(f"overflow ({e})")
        ok = False
    if rank == 0:
        print(
            f"relL2={worst:.3e} mismatched_rows={mismatched} "
            f"{'PASS' if ok else 'FAIL'} "
            f"time={ms_per:.4f}ms (reference stage2+combine {ref_ms:.4f}ms)",
            flush=True,
        )
    sc.close()
    if failures:
        raise AssertionError(f"sdma path failed: {failures}")
    if rank == 0:
        print("SUCCESS", flush=True)


# --- bench ----------------------------------------------------------------------------


def run_bench(args, rank, world, device):
    batch_sizes = [int(v) for v in args.bs_list.split(",")]
    chunk_list = [int(v) for v in args.chunk_tiles.split(",")]
    graph = args.graph
    rows = []  # (bs, label, s2_mean, s2_max, relL2)

    max_bs = max(batch_sizes)
    net, moe = _build_moe(args, max_bs, rank, world, device)
    x_all, wts_all, ids_all = _make_inputs(
        max_bs,
        net["model_dim"],
        net["experts"],
        net["topk"],
        rank,
        args.seed,
        device,
    )
    # One operator and one MegaMoESdma for every token count: building more would
    # re-create symmetric heaps and SDMA queues.
    sc = _make_sdma(moe, args, chunk_list[0])

    for bs in batch_sizes:
        x, wts, ids = (
            x_all[:bs].contiguous(),
            wts_all[:bs].contiguous(),
            ids_all[:bs].contiguous(),
        )
        ref, config = _reference(moe, x, wts, ids, args.no_aligned_pair)
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
                    "  aligned-pair stage2 is not covered by MegaMoESdma; skipped",
                    flush=True,
                )
            continue

        def base_s2():
            moe._run_stage2(bs, None, True, config)

        b_s2 = _time(base_s2, args.iters, graph)
        rows.append((bs, "aiter MegaMoEV2 (P2P scatter)", *b_s2, None))
        if rank == 0:
            print(
                f"  baseline stage2+combine          {b_s2[0]:9.4f} / {b_s2[1]:9.4f} ms (mean/max)",
                flush=True,
            )

        for ct in chunk_list:
            sc.set_chunk_tiles(ct)
            label = f"SDMA 3-kernel CT={ct}"
            out = sc.run(bs, config).clone()
            torch.cuda.synchronize()
            rel = _rel_l2(out, ref)
            n_s2 = _time(lambda: sc.run(bs, config), args.iters, graph)
            rows.append((bs, label, *n_s2, rel))
            br = _breakdown(sc, bs, config)
            if rank == 0:
                parts = "  ".join(f"{k}={v:.3f}" for k, v in br.items())
                print(
                    f"  {label:<22} stage2+combine {n_s2[0]:9.4f} / {n_s2[1]:9.4f} ms  "
                    f"relL2={rel:.2e}",
                    flush=True,
                )
                print(f"      breakdown (eager, ms): {parts}", flush=True)
            if rel >= args.rtol:
                raise AssertionError(
                    f"bs={bs} {label}: relL2={rel:.3e} exceeds {args.rtol}"
                )
            sc.check_overflow()

    if rank == 0:
        print("\n| bs | path | stage2+combine mean ms | max ms | vs baseline | relL2 |")
        print("|---:|---|---:|---:|---:|---:|")
        base = {}
        for bs, label, s2m, s2x, rel in rows:
            if rel is None:
                base[bs] = s2m
            print(
                f"| {bs} | {label} | {s2m:.4f} | {s2x:.4f} | {base.get(bs, s2m) / s2m:.2f}x | "
                f"{'-' if rel is None else f'{rel:.1e}'} |"
            )
        print("\nSUCCESS", flush=True)
    sc.close()


# --- command line ---------------------------------------------------------------------


def _parse_args():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--network", choices=NETWORKS, default="glm52")
    common.add_argument("--seed", type=int, default=123)
    common.add_argument("--rtol", type=float, default=0.05)
    common.add_argument(
        "--queues",
        type=int,
        default=1,
        help="SDMA channels per (src, dst) pair. More than one measured "
        "slower: KFD recommends a single engine per peer link here, so "
        "extra channels share it and only add descriptors.",
    )
    common.add_argument("--row-slack", type=float, default=1.5)
    common.add_argument(
        "--signal-mode",
        choices=("quiet", "remote"),
        default="remote",
        help="how a sender tells a peer its rows landed",
    )
    common.add_argument(
        "--no-aligned-pair",
        action="store_true",
        help="force aiter off its common-pair stage2 so both paths run the "
        "same config. It is picked only at bucket 8192, where MegaMoESdma "
        "has no equivalent -- and measured 1.5%% slower than without it on "
        "uniform routing anyway.",
    )
    common.add_argument(
        "--quant",
        choices=("none", "fp8_blockwise_1x32"),
        default="fp8_blockwise_1x32",
        help="quantise the send buffer; halves the bytes SDMA moves",
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    test = sub.add_parser(
        "test", parents=[common], help="SDMA output vs aiter's, one token count"
    )
    test.add_argument("--bs", type=int, default=128, help="tokens per rank")
    test.add_argument("--iters", type=int, default=20)
    test.add_argument(
        "--chunk-tiles", type=int, default=16, help="sort tiles per shipped chunk"
    )
    test.add_argument(
        "--hot-bias",
        type=float,
        default=0.0,
        help="add this to the router scores of rank 0's experts: skewed routing, "
        "where one rank receives more than an even share of the rows",
    )

    bench = sub.add_parser(
        "bench", parents=[common], help="latency vs aiter, per token count"
    )
    bench.add_argument(
        "--bs-list", default="128,512,2048,4096,8192", help="tokens per rank"
    )
    bench.add_argument("--iters", type=int, default=30)
    bench.add_argument(
        "--chunk-tiles", default="16", help="fused chunk sizes to sweep, e.g. 8,16,32"
    )
    bench.add_argument(
        "--graph",
        action="store_true",
        help="time CUDA Graph replays instead of eager launches. Off by "
        "default: prefill does not capture, so eager is the shape that matters.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
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
