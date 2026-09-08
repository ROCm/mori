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

Headline, 8x MI355X, [4096, 7168] out, K=1024, graph replay, median of 51:

    split-lsa                       324.4us
    split-sdma                      338.0
    fused-sdma  chunks=1, no fence  348.5
    fused-lsa                       341.0   RACY, see kernels_fused

Fusing loses. Read ``kernels_fused.py`` before trusting any faster fused number
from this benchmark: several of its options are intermittently wrong, and a
single passing run proves nothing. ``--chunks`` > 1 and every ``raw-wt`` fence
mode are known-racy and kept only to reproduce that.
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


def make_operands(rank: int, m: int, n: int, k: int):
    """Deterministic per-rank fp8 operands, plus the bf16 reference partial.

    Values are kept small so the fp8 rounding is the only error source and the
    all-reduce reference can be built on the host from the same recipe.
    """
    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    a = (torch.randn(m, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    sa = torch.rand(m, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.01
    sb = torch.rand(n, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.01
    return a, b, sa, sb


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
    direct_lsa = args.mode == "fused-lsa"
    fused = args.mode in ("fused-sdma", "fused-lsa")
    # fused-lsa still uses the SDMA reduce/gather tail, so it wants queues too.
    needs_sdma = args.mode in ("split-sdma", "fused-sdma", "fused-lsa")
    # 1, even though >1 is what would create the overlap. A chunk boundary means
    # the copy engine starts reading C *while the GEMM is still running*, and no
    # combination of submit lock, post-submit barrier, release fence or atomic
    # ordering made that reliable here -- see the race note in kernels_fused.
    chunks = args.chunks if args.chunks else 1
    cfg = ArConfig(
        world_size=world_size,
        m=args.m,
        n=args.n,
        recv_slots=world_size if needs_sdma else 0,
        counter_chunks=chunks,
    )
    cfg.validate()

    a, b, sa, sb = make_operands(rank, args.m, args.n, args.k)
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

        gemm = compile_fused_gemm_scatter(
            cfg,
            rank,
            K=args.k,
            BLOCK_M=args.block_m,
            BLOCK_N=args.block_n,
            b_preshuffled=True,
            waves_per_eu=args.waves_per_eu,
            xcd_swizzle=args.xcd_swizzle,
            fuse=fused,
            transport="lsa" if direct_lsa else "sdma",
            rotated=None if args.tile_order == "auto" else args.tile_order == "rotated",
            fence=args.fence,
            emit_put=not args.no_put,
            atomic_order=args.atomic,
        )
        a_i8 = a.contiguous().view(torch.int8).view(-1)
        b_i8 = b_shuf.contiguous().view(torch.int8).view(-1)
        c_flat = c.view(-1)

        def run_gemm(stream):
            gemm(a_i8, b_i8, c_flat, sa, sb, args.m, args.n, dc.ptr, win.handle,
                 stream=stream)

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
        if (
            args.mode != "gemm-only"
            and args.stop_after == "all"
            and not args.skip_validation
        ):
            # Reference: every rank's partial, summed in fp32 on the host side.
            acc = torch.zeros(args.m, args.n, device="cuda", dtype=torch.float32)
            for r in range(world_size):
                ar, br, sar, sbr = make_operands(r, args.m, args.n, args.k)
                acc += (ar.float() @ br.float().T) * sar[:, None] * sbr[None, :]
            diff = (out.float() - acc).norm().item()
            denom = acc.norm().item()
            rel_l2 = diff / denom if denom else diff
            # fp8 inputs, so this is quantization error, not a collective error;
            # the collective itself is checked bit-exactly in test_flydsl_ar.py.
            validated = rel_l2 < 5e-3
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
                "block_m": args.block_m,
                "block_n": args.block_n,
                "chunks": chunks if fused else None,
                "tile_order": args.tile_order,
                "fence": args.fence if fused else None,
                "stop_after": args.stop_after,
                "max_rank_time_us": per_rank[max_rank],
                "critical_rank": max_rank,
                "per_rank_time_us": per_rank,
                "rel_l2": rel_l2,
                "validated": validated,
                "timing": "eager" if args.eager else "graph",
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
    p.add_argument("--block-m", type=int, default=256)
    p.add_argument("--block-n", type=int, default=256)
    p.add_argument("--waves-per-eu", type=int, default=2)
    p.add_argument("--xcd-swizzle", type=int, default=0)
    p.add_argument(
        "--chunks",
        type=int,
        default=0,
        help="pushes per destination (0 = one per BLOCK_M row-band). >1 is what "
        "produces the overlap, and requires the submit lock",
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
    p.add_argument("--sdma-queues", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=51)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--json-out")
    return p


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
