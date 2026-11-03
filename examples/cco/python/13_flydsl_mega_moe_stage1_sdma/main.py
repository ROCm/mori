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
"""CCO Example 13 -- MegaMoE Stage1 (dispatch + GEMM1) with the dispatch done by SDMA (gfx950)

aiter's MegaMoEV2 Stage1 moves every routed row to its destination rank with P2P
stores from the compute units, inside the GEMM1 kernel. Here a small pack kernel
lays this rank's rows out per (destination, expert) in a cco window, and one fused
kernel ships them with the SDMA copy engines (workgroup 0, one put pair per peer and
expert, every one signalled) while the other workgroups run an 8-wave GEMM1 that
starts on an expert as soon as its rows have landed:

    pack.py                  the sender-side pack kernel
    gemm1_8wave_sdma.py      the fused kernel: SDMA puts + scale shufflers + 8-wave GEMM1
    mega_moe_stage1_sdma.py  the op, ``MegaMoEStage1Sdma``: cco window, launches, and
                             ``install()`` into a MegaMoEV2 in place of its Stage1 launch

aiter's prepare, Stage2 and combine run as they are, so the layer output must match
aiter's byte for byte.

    test    one token count: aiter's layer output, then 4 back-to-back calls through the
            SDMA Stage1 with two alternating inputs; every row must be identical
    bench   per token count: CUDA Graph time of Stage1 (including prepare) and of the
            whole layer, aiter vs SDMA, max over ranks

Needs aiter (MegaMoEV2 with the compact prepare), FlyDSL, and mori built with SDMA
(BUILD_CCO_SDMA=ON), on a gfx950 node with 8 GPUs:

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 main.py test --bs 8192
    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 main.py bench --bs-list 2048,4096,8192
"""

import argparse
import os

os.environ.setdefault("MORI_ENABLE_SDMA", "1")

import torch  # must be imported before anything that pulls in mori.cco
import torch.distributed as dist

import flydsl.expr as fx
import mori.shmem as ms

import aiter
from aiter import dtypes
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

from mega_moe_stage1_sdma import MegaMoEStage1Sdma

NETWORKS = {
    "glm52": dict(
        model_dim=6144, inter_dim=2048, experts=256, topk=8, swiglu_limit=0.0
    ),
    "v4_pro": dict(
        model_dim=7168, inter_dim=3072, experts=384, topk=6, swiglu_limit=10.0
    ),
    "flash": dict(
        model_dim=5120, inter_dim=2304, experts=384, topk=6, swiglu_limit=10.0
    ),
}

#: Seed offset of the second input set: alternating two inputs catches stale landed data
#: (same addresses, different values from one call to the next).
SEED2 = 333


# --- Inputs, weights and the aiter operator -------------------------------------------


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


def _build_moe(args, rank, world, device):
    """aiter's operator, sized for ``--max-tok`` tokens per rank, with its quantised weights."""
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
        max_tok_per_rank=args.max_tok,
        **net,
    )
    return net, moe


def _make_sdma(moe, args):
    return MegaMoEStage1Sdma(moe, scale_mode=args.scale_mode, shufflers=args.shufflers)


# --- Correctness and timing -----------------------------------------------------------


def _barrier():
    torch.cuda.synchronize()
    ms.shmem_barrier_all()


def _over_ranks(local, op):
    t = torch.tensor([float(local)], dtype=torch.float64)
    dist.all_reduce(t, op=op)
    return float(t.item())


def _check(moe, s1, net, bs, args, rank, device):
    """Max over ranks of the rows that differ from aiter's output, over 4 back-to-back
    SDMA calls alternating two inputs (the parity flips and the signal targets advance
    between them, with no host sync). Leaves ``s1`` uninstalled."""
    shape = (net["model_dim"], net["experts"], net["topk"], rank)
    x, wts, ids = _make_inputs(bs, *shape, args.seed, device)
    x2, wts2, ids2 = _make_inputs(bs, *shape, args.seed + SEED2, device)
    s1.uninstall()
    ref2 = moe(x2, wts2, ids2)[:bs].clone()
    ref = moe(x, wts, ids)[:bs].clone()
    bad = 0
    with s1:
        for i in range(4):
            if i % 2:
                out = moe(x2, wts2, ids2)[:bs]
                bad = max(bad, int((out != ref2).any(dim=1).sum()))
            else:
                out = moe(x, wts, ids)[:bs]
                bad = max(bad, int((out != ref).any(dim=1).sum()))
        torch.cuda.synchronize()
    _barrier()
    return int(_over_ranks(bad, dist.ReduceOp.MAX)), (x, wts, ids, ref)


def _time_graph(fn, iters):
    """Max over ranks of the mean ms of one ``fn`` captured in a CUDA Graph."""
    _barrier()
    fn()
    _barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        fn()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return _over_ranks(start.elapsed_time(end) / iters, dist.ReduceOp.MAX)


# --- test -----------------------------------------------------------------------------


def run_test(args, rank, world, device):
    assert args.bs <= args.max_tok, "--bs exceeds --max-tok"
    net, moe = _build_moe(args, rank, world, device)
    s1 = _make_sdma(moe, args)
    bad, _ = _check(moe, s1, net, args.bs, args, rank, device)
    sbm = moe._active_config.stage1.sort_block_m
    ok = bad == 0
    if rank == 0:
        print(
            f"network={args.network} bs={args.bs} SBM={sbm} scale_mode={s1.scale_mode_for(sbm)} "
            f"mismatched_rows={bad} {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    s1.close()
    if not ok:
        raise AssertionError(f"SDMA Stage1: {bad} rows differ from aiter's")
    if rank == 0:
        print("SUCCESS", flush=True)


# --- bench ----------------------------------------------------------------------------


def run_bench(args, rank, world, device):
    batch_sizes = [int(v) for v in args.bs_list.split(",")]
    assert max(batch_sizes) <= args.max_tok, "--bs-list exceeds --max-tok"
    net, moe = _build_moe(args, rank, world, device)
    # One operator and one SDMA Stage1 for every token count: building more would
    # re-create symmetric heaps and SDMA queues.
    s1 = _make_sdma(moe, args)
    rows = []  # (bs, sbm, mode, mismatched, aiter s1, sdma s1, aiter layer, sdma layer)

    for bs in batch_sizes:
        bad, (x, wts, ids, ref) = _check(moe, s1, net, bs, args, rank, device)
        sbm = moe._active_config.stage1.sort_block_m

        def use(sdma):
            if sdma:
                s1.install()
            else:
                s1.uninstall()

        def stage1(sdma):
            def fn():
                use(sdma)
                cfg = moe._select_config(bs)
                s = fx.Stream(torch.cuda.current_stream().cuda_stream)
                moe._run_compact_prepare(ids, bs, cfg.stage1, s, quant_input=x)
                moe._run_fused_stage1(
                    moe._s1_quant_x[:bs],
                    wts,
                    moe._s1_quant_scale[:bs],
                    ids,
                    stream=s,
                    config=cfg.stage1,
                    prepared=True,
                )

            return fn

        def layer(sdma):
            def fn():
                use(sdma)
                moe(x, wts, ids)

            return fn

        # Two rounds, the four arms interleaved, so that drift cannot favour one of them.
        res = {}
        for _ in range(2):
            for name, fn in (
                ("aiter_s1", stage1(False)),
                ("sdma_s1", stage1(True)),
                ("aiter_layer", layer(False)),
                ("sdma_layer", layer(True)),
            ):
                res.setdefault(name, []).append(_time_graph(fn, args.iters))
        res = {k: sum(v) / len(v) for k, v in res.items()}
        # Graph replays must leave valid state: check once more after them.
        with s1:
            out = moe(x, wts, ids)[:bs]
            torch.cuda.synchronize()
        bad = max(
            bad, int(_over_ranks(int((out != ref).any(dim=1).sum()), dist.ReduceOp.MAX))
        )
        rows.append(
            (
                bs,
                sbm,
                s1.scale_mode_for(sbm),
                bad,
                res["aiter_s1"],
                res["sdma_s1"],
                res["aiter_layer"],
                res["sdma_layer"],
            )
        )
        if rank == 0:
            print(
                f"bs={bs} SBM={sbm} scale_mode={rows[-1][2]} mismatched_rows={bad} "
                + " ".join(f"{k}={v:.4f}" for k, v in res.items())
                + " (ms)",
                flush=True,
            )

    if rank == 0:
        print(
            f"\nnetwork={args.network}, CUDA Graph, max over ranks, ms; Stage1 includes prepare"
        )
        print(
            "| bs | SBM | scale mode | Stage1 aiter | Stage1 SDMA | diff | layer aiter | layer SDMA | diff "
            "| mismatched rows |"
        )
        print("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for bs, sbm, mode, bad, a1, s1_, al, sl in rows:
            print(
                f"| {bs} | {sbm} | {mode} | {a1:.4f} | {s1_:.4f} | {s1_ - a1:+.4f} | {al:.4f} | {sl:.4f} "
                f"| {sl - al:+.4f} | {bad} |"
            )
    s1.close()
    failed = [r[0] for r in rows if r[3]]
    if failed:
        raise AssertionError(f"SDMA Stage1 output differs from aiter's at bs {failed}")
    if rank == 0:
        print("\nSUCCESS", flush=True)


# --- command line ---------------------------------------------------------------------


def _parse_args():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--network",
        choices=NETWORKS,
        default="v4_pro",
        help="model geometry; the measurements are for v4_pro",
    )
    common.add_argument("--seed", type=int, default=123)
    common.add_argument(
        "--max-tok",
        type=int,
        default=8192,
        help="MegaMoEV2 max_tok_per_rank (a power of two, below 32768)",
    )
    common.add_argument(
        "--scale-mode",
        choices=("auto", "shuffler", "ldsw"),
        default="auto",
        help="how GEMM1 gets the A scales: shuffler blocks put them into the contiguous "
        "layout once per M-tile, or each work item stages them into LDS (ldsw); auto = "
        "shuffler when the Stage1 sort_block_m is 128 (aiter's 4096/8192 buckets), else ldsw",
    )
    common.add_argument(
        "--shufflers",
        type=int,
        default=4,
        help="shuffler workgroups in the shuffler mode",
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    test = sub.add_parser(
        "test", parents=[common], help="SDMA Stage1 output vs aiter's, one token count"
    )
    test.add_argument("--bs", type=int, default=8192, help="tokens per rank")

    bench = sub.add_parser(
        "bench",
        parents=[common],
        help="Stage1 and layer time vs aiter, per token count",
    )
    bench.add_argument("--bs-list", default="2048,4096,8192", help="tokens per rank")
    bench.add_argument(
        "--iters", type=int, default=20, help="graph replays per measurement"
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
