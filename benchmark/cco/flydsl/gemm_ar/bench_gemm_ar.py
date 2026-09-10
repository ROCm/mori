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
"""GEMM + all-reduce at the DSV4-Pro ``wo_b`` shape: fused vs split.

    MORI_SOCKET_IFNAME=lo MORI_ENABLE_SDMA=1 PYTHONPATH=/path/to/aiter \\
    torchrun --standalone --nproc_per_node=8 bench_gemm_ar.py \\
        --mode fused-sdma -m 4096 -n 7168 -k 1024

Modes, all measuring the same end state (``output`` holds the all-reduced GEMM
result on every rank):

    split-sdma   gemm -> scatter -> reduce -> gather   (4 kernels)
    fused-sdma   gemm+scatter -> drain -> reduce -> gather   (4 kernels)
    fused-lsa    gemm writing straight into the peers -> barrier -> reduce
                 -> gather. gcnasm's "Direct LSA" shape: no staging, no copy
                 engine, and nothing to publish mid-kernel
    split-lsa    gemm -> LSA 2-stage all-reduce        (2 kernels)
    gemm-only    the GEMM alone, to size the ceiling

``split-*`` uses the *same* kernel as ``fused-sdma`` with the epilogue tail
compiled out, so the two differ in one thing only. ``split-lsa`` is included
because LSA is the faster collective, and a fused SDMA path has to beat
``gemm + LSA``, not just ``gemm + SDMA``, to be worth anything.

Headline at the shape the model actually runs -- ``[16384, 7168]`` out, K=2048,
which is a ``--chunked-prefill-size 16384`` TP8 prefill chunk of DSV4-Pro
``wo_b`` (1792 tiles of 256x256 per rank). 8x MI355X, graph replay, median of
31, max over ranks, on an idle box, with the current defaults:

                          --quant ptpc   --quant blockscale
    fused-sdma  (chunks=8)      1114.5us          1146.2us
    split-sdma                  1261.7            1465.5
    split-lsa                   1262.4            1469.9
    fused-lsa   (n_stripe=2)    1365.6            1699.6
    gemm-only                    228.9             388.5

``blockscale`` is the quantisation the model actually runs -- A 1x128 and B
128x128 with fp32 scales -- and is the column to read. ``ptpc`` is the aiter
8-wave kernel's native per-token/per-channel form, kept because the two bitwise
tests and every earlier measurement are on it.

For scale, the same layer in the running model costs 1491.3us (GEMM 348.0 +
NCCL 1143.3, medians over its 61 layers), and the collective on its own is
1043us for LSA / 1047 for SDMA against 1113 for NCCL. So at the model's own
quantisation ``fused-sdma`` is **21.8% under split and 23.1% under what the
model runs today**.

The margin *grows* with blockscale (11.7% -> 21.8%) for the reason the whole
exercise was about: the GEMM goes 228.9 -> 388.5us while the 489us of link time
does not move, so there is more compute to hide the transfer behind. Our blockscale
GEMM is *not* competitive standalone: 370.5us against CK's 300.4 for its fastest
blockscale instance (64x256, Intrawave v1) measured the same way. The gap is the
per-K-block scale, not the GEMM -- unscaled we are 224.0us. See
``kernels_preshuffle4w.py``, which ports CK's 4-wave B-out-of-LDS shape and
lands at 385.9us: aligned on the GEMM, still paying for the scale.

Fusing over SDMA wins, and only because of ``--chunks``; see the table at its
definition in ``run()``. It was pinned to 1 while the aiter GEMM's
under-counted ``s_waitcnt`` made every chunked run intermittently wrong, and
with one chunk the fused path overlaps nothing at all. Fusing over LSA still
loses, though by much less since ``--n-stripe 2``: it spends 560us of its GEMM
pushing C over xGMI, where the copy engines move the same bytes in 499.

Read ``kernels_fused.py`` before trusting any fused number from this
benchmark: several of its options are intermittently wrong, and a single
passing run proves nothing. Every ``raw-wt`` fence mode is known-racy and kept
only to reproduce that.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import flydsl.expr as fx
import torch
import torch.distributed as dist

from mori.cco import (
    CCODevCommRequirements,
    Communicator,
    GDA_CONNECTION_NONE,
    UniqueId,
)
from mori.tensor_utils import from_gpu_ptr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "ar"))
from kernels_fused import compile_fused_gemm_scatter  # noqa: E402
from kernels_preshuffle4w import compile_preshuffle_gemm  # noqa: E402
from kernels_lsa import build_lsa_ar  # noqa: E402
from kernels_sdma import build_sdma_phases  # noqa: E402
from layout import ArConfig  # noqa: E402

VMM_SLACK = 512 * 1024 * 1024
MODES = ("gemm-only", "split-sdma", "fused-sdma", "fused-lsa", "split-lsa")


def _setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return local_rank, rank, world_size, UniqueId.from_bytes(payload[0])


#: fp8 block-scale group size along K. Fixed by the model's quantiser
#: (``aiter_per1x128_quant``) and by the kernel, whose BLOCK_K is already 128.
SCALE_BK = 128


def make_operands(rank: int, m: int, n: int, k: int, quant: str = "ptpc"):
    """Deterministic per-rank fp8 operands and their scales.

    Values are kept small so the fp8 rounding is the only error source and the
    all-reduce reference can be built on the host from the same recipe.

    ``quant="ptpc"`` is a8w8 per-token/per-channel: ``sa[M]``, ``sb[N]``, one
    scale for a whole row of A and a whole column of B.

    ``quant="blockscale"`` is what the model actually runs -- A quantised 1x128
    and B 128x128, so both scales vary along K: ``sa[M, K/128]`` and
    ``sb[N/128, K/128]``. The layouts match
    ``aiter.gemm_a8w8_blockscale_bpreshuffle`` so the two can be compared
    directly: **sa is column-major** (sglang's
    ``materialize_bpreshuffle_fp8_scale`` does ``t().contiguous().t()``) and sb
    is row-major. Established by sweeping the four combinations against that
    kernel: only (sa column-major, sb row-major) lands at the fp8 floor
    (1.66e-3); the other three give 0.17-0.32.
    """
    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    a = (torch.randn(m, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    if quant == "ptpc":
        sa = torch.rand(m, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.01
        sb = torch.rand(n, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.01
        return a, b, sa, sb
    if quant != "blockscale":
        raise ValueError(f"quant must be ptpc or blockscale, got {quant!r}")
    kb = k // SCALE_BK
    sa = torch.rand(m, kb, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.01
    sb = (
        torch.rand(n // SCALE_BK, kb, generator=g, device="cuda", dtype=torch.float32)
        * 0.01
        + 0.01
    )
    return a, b, sa.t().contiguous().t(), sb.contiguous()


def reference_partial(a, b, sa, sb, quant: str) -> torch.Tensor:
    """One rank's fp32 GEMM reference, matching ``make_operands``' quantisation."""
    af, bf = a.float(), b.float()
    if quant == "ptpc":
        return (af @ bf.T) * sa[:, None] * sb[None, :]
    out = torch.zeros(a.shape[0], b.shape[0], device=a.device, dtype=torch.float32)
    for i in range(a.shape[1] // SCALE_BK):
        ks = slice(i * SCALE_BK, (i + 1) * SCALE_BK)
        out += (
            (af[:, ks] @ bf[:, ks].T)
            * sa[:, i][:, None]
            * sb[:, i].repeat_interleave(SCALE_BK)[None, :]
        )
    return out


def _median_us(fn, warmup: int, iters: int, *, graph: bool = True) -> float:
    if graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return _median_us(g.replay, warmup, iters, graph=False)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def run(args) -> int:
    from aiter.ops.shuffle import shuffle_weight

    local_rank, rank, world_size, uid = _setup_distributed()
    if args.gemm_impl == "preshuffle4w" and args.mode != "gemm-only":
        raise SystemExit(
            "--gemm-impl preshuffle4w has no fused epilogue; use --mode gemm-only"
        )
    # blockscale keeps a second fp32 accumulator for the per-K-block promotion,
    # which doubles the accumulator VGPRs; 256x256 needs 256 of them and the
    # kernel already runs at ~254 with zero spill, so the tile has to halve.
    if not args.block_m:
        args.block_m = 128 if args.quant == "blockscale" else 256
    direct_lsa = args.mode == "fused-lsa"
    fused = args.mode in ("fused-sdma", "fused-lsa")
    # fused-lsa still uses the SDMA reduce/gather tail, so it wants queues too.
    needs_sdma = args.mode in ("split-sdma", "fused-sdma", "fused-lsa")
    # Pushes per destination. This is the whole overlap mechanism: with one
    # chunk the tile counter only fires when a destination's entire slice is
    # done, which under the rotated tile order is the end of the GEMM, so
    # nothing overlaps at all -- measured, the drain then costs the same as the
    # split scatter (493.8 vs 499.0us). With eight, a destination's first chunk
    # leaves while the GEMM is still computing its later ones.
    #
    # Per-kernel medians at [16384, 7168] K=2048, 8x MI355X:
    #
    #     chunks   GEMM   drain  reduce  gather   total   end-to-end
    #          1  257.5   493.8    44.0   496.4  1291.7      1299.6
    #          2  266.5   379.4    44.2   496.4  1186.6      1194.9
    #          8  265.0   303.7    43.8   496.6  1109.1      1115.7
    #
    # The drain falls 38% while the GEMM grows 7.5us for the extra counter
    # atomics and the submit lock. 8 hides 190 of the 265us of GEMM it could
    # possibly hide. More is worse: at 8 each PUT is 3.5 MiB, and 16 (via
    # --block-m 128) drops that to 1.75 MiB, below the knee in the SDMA
    # bandwidth curve -- 1130.8us, slower than 8.
    #
    # This was pinned to 1 because >1 produced wrong output 3 runs in 10. That
    # was the aiter GEMM's under-counted s_waitcnt, not the chunk protocol:
    # since that fix, chunks 1/2/4/8 are 40/40 correct and chunks=8 alone is
    # 25/25, with no hang. Two hangs were seen at 8 ranks *before* the sweep
    # settled and never reproduced in 100+ runs after; the submit lock is a
    # plain test-and-set spin, so treat a hang as possible and use `timeout`
    # when sweeping.
    DEFAULT_CHUNKS = 8
    chunks = args.chunks if args.chunks else DEFAULT_CHUNKS
    if fused:
        # Fall back rather than fail: chunks has to divide the M-tiles per
        # destination, which at small M can be fewer than 8.
        m_tiles_per_peer = max(1, (args.m + args.block_m - 1) // args.block_m // world_size)
        while chunks > 1 and m_tiles_per_peer % chunks:
            chunks //= 2
    cfg = ArConfig(
        world_size=world_size,
        m=args.m,
        n=args.n,
        recv_slots=world_size if needs_sdma else 0,
        counter_chunks=chunks,
    )
    cfg.validate()

    a, b, sa, sb = make_operands(rank, args.m, args.n, args.k, args.quant)
    b_shuf = shuffle_weight(b, layout=(16, 16))

    vmm = max(4 * cfg.window_bytes + VMM_SLACK, VMM_SLACK)
    result = None

    with Communicator.init(world_size, rank, uid, per_rank_vmm=vmm) as comm:
        mem = comm.alloc_mem(cfg.window_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        from_gpu_ptr(mem.ptr, (cfg.window_bytes,), torch.uint8).zero_()
        # C *is* the all-reduce's input region: the GEMM writes its partial
        # straight into the symmetric window, which is the whole host-side change
        # the SDMA fusion needs.
        c = from_gpu_ptr(mem.ptr + cfg.input_off, (args.m, args.n), torch.bfloat16)
        out = from_gpu_ptr(mem.ptr + cfg.output_off, (args.m, args.n), torch.bfloat16)
        torch.cuda.synchronize()

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        if needs_sdma:
            reqs.sdma_queue_count = args.sdma_queues
        dc = comm.create_dev_comm(reqs)

        if args.gemm_impl == "preshuffle4w":
            # CK's shape: 4 waves, B never staged in LDS. gemm-only, so the
            # window's input region is just an ordinary output buffer here.
            launch4w = compile_preshuffle_gemm(
                N=args.n,
                K=args.k,
                tile_m=args.tile_m,
                tile_n=args.tile_n,
                tile_k=args.tile_k,
                in_dtype="fp8",
                out_dtype="bf16",
                quant=args.quant,
                swap_ab=args.p4w_swap_ab,
                waves_per_eu=args.waves_per_eu,
                xcd_swizzle=args.xcd_swizzle,
            )
            semaphore = torch.zeros(1, device="cuda", dtype=torch.int32)
            bias_unused = torch.zeros(1, device="cuda", dtype=torch.bfloat16)

        gemm = None if args.gemm_impl == "preshuffle4w" else compile_fused_gemm_scatter(
            cfg,
            rank,
            K=args.k,
            BLOCK_M=args.block_m,
            BLOCK_N=args.block_n,
            b_preshuffled=True,
            quant=args.quant,
            sdma_queues=args.sdma_queues,
            waves_per_eu=args.waves_per_eu,
            xcd_swizzle=args.xcd_swizzle,
            fuse=fused,
            transport="lsa" if direct_lsa else "sdma",
            swap_ab=args.swap_ab,
            store_probe=args.store_probe,
            permlane=args.permlane,
            lane_transpose=args.lane_transpose,
            hoist_scales=args.hoist_scales,
            peer_uncached=args.peer_uncached,
            direct_fence=args.direct_fence,
            rotated=None if args.tile_order == "auto" else args.tile_order == "rotated",
            n_stripe=args.n_stripe or None,
            fence=args.fence,
            emit_put=not args.no_put,
            atomic_order=args.atomic,
        )
        a_i8 = a.contiguous().view(torch.int8).view(-1)
        b_i8 = b_shuf.contiguous().view(torch.int8).view(-1)
        c_flat = c.view(-1)

        # The kernel indexes both scale buffers linearly, so hand it the
        # *physical* element order as 1-D contiguous tensors. sa is logically
        # [M, K/128] but column-major (the layout aiter's blockscale GEMM
        # consumes), so its physical order is sa.t(); passing the 2-D
        # non-contiguous view through DLPack would give the kernel the wrong
        # strides.
        if args.quant == "blockscale":
            sa_arg = sa.t().reshape(-1).contiguous()
            sb_arg = sb.reshape(-1).contiguous()
        else:
            sa_arg, sb_arg = sa, sb

        def run_gemm(stream):
            if args.gemm_impl == "preshuffle4w":
                launch4w(c, c, semaphore, a, b_shuf, sa_arg, sb_arg, bias_unused,
                         args.m, args.n, stream=stream)
                return
            gemm(a_i8, b_i8, c_flat, sa_arg, sb_arg, args.m, args.n, dc.ptr,
                 win.handle, stream=stream)

        if needs_sdma:
            parts = build_sdma_phases(
                cfg,
                rank,
                queues=args.sdma_queues,
                reduce_self_from_recv=direct_lsa,
                recv_uncached=direct_lsa,
            )
        if args.mode == "split-lsa":
            lsa_ar, _ = build_lsa_ar(cfg, rank)

        def once():
            stream = fx.Stream(torch.cuda.current_stream())
            run_gemm(stream)
            if args.mode == "gemm-only":
                return
            if args.mode == "split-lsa":
                lsa_ar(dc.ptr, win.handle, stream=stream)
                return
            # fused-sdma: the puts were issued from the epilogue, so only the
            # drain runs. fused-lsa: the epilogue already wrote into the peers,
            # so this is a pure cross-rank barrier (its quiet drains nothing).
            # split: the full scatter kernel.
            parts["drain" if fused else "scatter"](dc.ptr, win.handle, stream=stream)
            if args.stop_after == "scatter":
                return
            parts["reduce"](dc.ptr, win.handle, stream=stream)
            parts["gather"](dc.ptr, win.handle, stream=stream)

        comm.barrier()
        once()
        torch.cuda.synchronize()
        comm.barrier()

        rel_l2 = float("nan")
        validated = True
        if args.mode == "gemm-only" and not args.skip_validation:
            # The GEMM on its own, against this rank's partial. Without this
            # control a GEMM bug reads as an all-reduce bug: every other mode
            # validates the *sum*, so a corrupt partial and a corrupt collective
            # are indistinguishable from the reported relL2.
            ref = reference_partial(a, b, sa, sb, args.quant)
            diff = (c.float() - ref).norm().item()
            denom = ref.norm().item()
            rel_l2 = diff / denom if denom else diff
            validated = rel_l2 < 3e-3
            if not validated:
                print(f"[rank {rank}] GEMM VALIDATION FAILED relL2={rel_l2:.3e}", flush=True)
            del ref
        elif (
            args.mode != "gemm-only"
            and args.stop_after == "all"
            and not args.skip_validation
        ):
            # Reference: every rank's partial, summed in fp32 on the host side.
            acc = torch.zeros(args.m, args.n, device="cuda", dtype=torch.float32)
            for r in range(world_size):
                ar, br, sar, sbr = make_operands(r, args.m, args.n, args.k, args.quant)
                acc += reference_partial(ar, br, sar, sbr, args.quant)
            diff = (out.float() - acc).norm().item()
            denom = acc.norm().item()
            rel_l2 = diff / denom if denom else diff
            # fp8 inputs, so this is quantization error, not a collective error;
            # the collective itself is checked bit-exactly in test_flydsl_ar.py.
            # 3e-3, not 5e-3. The fp8 quantisation floor for these inputs is
            # 2.35e-3 and the corruption this pipeline produces lands at
            # 4-9e-3, so a looser gate reports a corrupt run as validated -- one
            # did, at 3.97e-3, while the fused path was racing.
            validated = rel_l2 < 3e-3
            if not validated:
                print(f"[rank {rank}] VALIDATION FAILED relL2={rel_l2:.3e}", flush=True)
            del acc

        comm.barrier()
        elapsed_us = _median_us(once, args.warmup, args.iters, graph=not args.eager)
        comm.barrier()

        stats = torch.tensor([elapsed_us], dtype=torch.float64)
        gathered = [torch.zeros_like(stats) for _ in range(world_size)]
        dist.all_gather(gathered, stats)
        per_rank = [float(t[0]) for t in gathered]
        max_rank = max(range(world_size), key=lambda r: per_rank[r])

        ok = torch.tensor([1 if validated else 0], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        validated = bool(ok.item())

        if rank == 0:
            result = {
                "mode": args.mode,
                "world_size": world_size,
                "m": args.m,
                "n": args.n,
                "k": args.k,
                "quant": args.quant,
                "block_m": args.block_m,
                "block_n": args.block_n,
                "chunks": chunks if fused else None,
                "tile_order": args.tile_order,
                "swap_ab": args.swap_ab,
                "permlane": args.permlane,
                "lane_transpose": args.lane_transpose,
                "fence": args.fence if fused else None,
                "stop_after": args.stop_after,
                "max_rank_time_us": per_rank[max_rank],
                "critical_rank": max_rank,
                "per_rank_time_us": per_rank,
                "rel_l2": rel_l2,
                "validated": validated,
                "timing": "eager" if args.eager else "graph",
                "gemm_impl": args.gemm_impl,
                "tile": (
                    [args.tile_m, args.tile_n, args.tile_k]
                    if args.gemm_impl == "preshuffle4w"
                    else None
                ),
            }
            print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
            print(
                f"[gemm_ar] mode={args.mode} m={args.m} n={args.n} k={args.k} "
                f"max_rank_time={per_rank[max_rank]:.2f}us "
                f"relL2={rel_l2:.2e} validated={validated}",
                flush=True,
            )
            if args.json_out:
                with open(args.json_out, "w") as fh:
                    json.dump(result, fh, indent=2, sort_keys=True)

    dist.barrier()
    return 0 if (result is None or result["validated"]) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=MODES, default="fused-sdma")
    p.add_argument("-m", type=int, default=4096)
    p.add_argument("-n", type=int, default=7168)
    p.add_argument("-k", type=int, default=1024)
    p.add_argument(
        "--quant",
        choices=("ptpc", "blockscale"),
        default="ptpc",
        help="a8w8 per-token/per-channel (the aiter 8wave kernel's native form) "
        "or the model's 1x128 / 128x128 block scale. blockscale applies the "
        "scales per K-block in the mainloop, which needs a second accumulator "
        "and therefore --block-m 128",
    )
    p.add_argument("--block-m", type=int, default=0,
                   help="0 = 256 for --quant ptpc, 128 for --quant blockscale")
    p.add_argument("--block-n", type=int, default=256)
    p.add_argument("--waves-per-eu", type=int, default=2)
    p.add_argument("--xcd-swizzle", type=int, default=0)
    p.add_argument(
        "--chunks",
        type=int,
        default=0,
        help="pushes per destination (0 = the default 8, halved until it "
        "divides the M-tiles per destination). >1 is what produces the "
        "overlap, and requires the submit lock",
    )
    p.add_argument(
        "--n-stripe",
        type=int,
        default=0,
        help="with the rotated order: rotate the destination every N N-tiles "
        "instead of after a whole chunk (gcnasm opus_direct_stripe_tile). "
        "0 = per-mode default: 2 for fused-lsa, chunk-major elsewhere. 1 rotates "
        "the destination on every block",
    )
    p.add_argument(
        "--tile-order",
        choices=("auto", "rotated", "linear"),
        default="auto",
        help="auto = rotated when fusing, linear otherwise",
    )
    p.add_argument(
        "--fence",
        choices=(
            "all", "agent", "agent-leader", "nt-agent", "leader", "none",
            "writethrough", "wt-agent", "raw-wt", "raw-wt-agent",
            "raw-wt-leader",
        ),
        default="none",
        help="release before the epilogue push. Correct: 'agent' (default, "
        "cheapest) and 'all' (system scope, adds a buffer_inv that costs 20us). "
        "Incorrect, and present only to price the fence: 'leader', 'none', "
        "'writethrough'. 'wt-agent' is correct but no faster than 'agent'",
    )
    p.add_argument(
        "--stop-after",
        choices=("all", "scatter"),
        default="all",
        help="'scatter' stops after the reduce-scatter transfer has landed, "
        "which is the only phase fusion can affect; use it to attribute a "
        "fused-vs-split difference instead of inferring it from the total",
    )
    p.add_argument(
        "--no-put",
        action="store_true",
        help="keep the epilogue but drop the transfer, to price how much of its "
        "cost is the copy engine reading C while the GEMM writes it. Output is "
        "wrong by construction; use with --skip-validation",
    )
    p.add_argument(
        "--atomic",
        choices=("acq_rel", "acquire", "release", "monotonic"),
        default="acq_rel",
        help="ordering on the tile counter. The winner needs the acquire half to "
        "order the other blocks' releases before its put; 'monotonic' drops it "
        "and races",
    )
    # The three C-store stages are ON by default. They compose as a ladder --
    # each needs the one before it -- and measured together at [16384, 7168]
    # K=2048 on 8x MI355X they help every mode and change no output:
    # gemm-only -5.0%, fused-sdma -3.2%, fused-lsa -2.4%, split-lsa -0.9%,
    # split-sdma -0.7%, all still at relL2 2.350e-3. Turn one off with its
    # --no- form (and the ones above it in the ladder).
    #
    # Note where the win is: the stages coalesce the *store issue*, so they pay
    # off when C lands in local memory (fused-sdma's GEMM 297.3 -> 256.6 us)
    # and not when it goes straight to a peer (fused-lsa's GEMM 954.5 -> 952.1,
    # i.e. nothing -- that path is xGMI-bound, and ATT shows 99% of its store
    # time is stall, not issue).
    #
    # ``compile_fused_gemm_scatter`` still defaults them to False: there the
    # default has to stay "aiter's kernel verbatim", which is what
    # ``test_pinned_copy_matches_aiter_kernel_bitwise`` checks.
    p.add_argument(
        "--swap-ab",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exchange the MFMA operands so each lane owns 4 consecutive N, "
        "letting C be stored 64 bits at a time instead of 16 "
        "(gcnasm mfma_adaptor_swap_ab). On by default",
    )
    p.add_argument(
        "--store-probe",
        action="store_true",
        help="PERF PROBE, output is wrong by construction: emit the 16B-per-lane "
        "store pattern the permlane stage would produce, without the shuffle, to "
        "price it before building it. Requires --swap-ab; use --skip-validation",
    )
    p.add_argument(
        "--permlane",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with --swap-ab: two permlane16_swap per M-tile so each lane owns 16 "
        "contiguous bytes and each row gets 64. This is the real version of what "
        "--store-probe prices. On by default",
    )
    p.add_argument(
        "--lane-transpose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with --swap-ab --permlane: one ds_bpermute per dword so adjacent "
        "lanes cover one C row (gcnasm kernel_template.hpp:491-510). On by "
        "default",
    )
    p.add_argument(
        "--hoist-scales",
        action="store_true",
        help="load each A/B scale once for the four half-tile stores instead of "
        "twice (needs --permlane)",
    )
    p.add_argument(
        "--peer-uncached",
        action="store_true",
        help="fused-lsa: store to the peer with sc0|sc1 so nothing is left dirty "
        "in a local L2 the barrier cannot reach",
    )
    p.add_argument(
        "--direct-fence",
        choices=("all", "leader"),
        default="leader",
        help="fused-lsa: whether every lane or only thread 0 publishes the "
        "block's peer stores. leader is legal because the barrier pair is closed "
        "and is worth 80us (355 vs 434)",
    )
    p.add_argument("--sdma-queues", type=int, default=8)
    p.add_argument(
        "--gemm-impl",
        choices=("8wave", "preshuffle4w"),
        default="8wave",
        help="8wave: the pinned aiter kernel, the only one with a fused "
        "epilogue. preshuffle4w: CK's shape -- 4 waves, B loaded straight "
        "to registers instead of through LDS. gemm-only.",
    )
    p.add_argument("--tile-m", type=int, default=64, help="preshuffle4w only")
    p.add_argument("--tile-n", type=int, default=256, help="preshuffle4w only")
    p.add_argument("--tile-k", type=int, default=128, help="preshuffle4w only")
    p.add_argument(
        "--p4w-swap-ab",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="preshuffle4w: exchange the MFMA operands so a lane owns four "
        "consecutive N instead of four M. Off by default -- it does what it "
        "should to the instructions and still loses; see kernels_preshuffle4w.",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=51)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--json-out")
    return p


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
