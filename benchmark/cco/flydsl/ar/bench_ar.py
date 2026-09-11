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
"""Benchmark: FlyDSL cco all-reduce (LSA / SDMA) vs aiter's custom all-reduce.

Reports one ``RESULT_JSON`` line per run, using the same critical-rank
decomposition as ``benchmark/cco/triton/gemm_a2a/bench_gemm_a2a.py`` so the two
benchmarks can be read side by side.

    MORI_SOCKET_IFNAME=lo MORI_ENABLE_SDMA=1 \\
    torchrun --standalone --nproc_per_node=8 bench_ar.py \\
        --backend lsa -m 4096 -n 7168 --warmup 5 --iters 51

Measured on 8x MI355X (gfx950), ROCm 7.2, ``[M, 7168]`` bf16, CUDA-graph replay,
best of 3 runs of 101 iterations each. aiter baseline is
``tensor_model_parallel_all_reduce`` with ``set_custom_all_reduce(True)``:

    M     aiter    LSA    (/aiter)   SDMA   (/LSA)
    64    25.6us   30.7us  1.20      45.6us  1.49
    512   53.6     66.5    1.24      73.6    1.11
    2048 146.1    160.3    1.10     167.6    1.05
    4096 271.1    285.9    1.05     294.5    1.03
    8192 558.8    531.7    0.95     544.6    1.02

Read those with three caveats. First, all three have a payload-independent floor
-- LSA's is ~28us and SDMA's ~40us, neither moving between M=8 and M=64 -- so the
small-M ratios are two constants divided, not a throughput result. The 12us
between the two floors is the copy engine's ~6us dispatch cost, paid once per
transfer phase. Second, single runs vary by up to 1.7x (aiter's M=64 spanned
25.6-44.0us across three runs), which is why this uses best-of-3; do not compare
single runs. Third, SDMA's remaining deficit is almost entirely its ~40us floor:
past M=2048 it is within 5% of LSA and at M=8192 it beats aiter. It got there by
sizing the reduce kernel's grid independently -- see ``SDMA_REDUCE_BLOCK_CAP`` in
layout.py, worth 48us at M=4096 on its own.

This box is affected by mori known-issue 3 (ROCm 7.2 routes uncached VMM
allocations to the coarse-grained pool): ``CCO_UNCACHED_WINDOW=0`` changes nothing,
which is the documented tell. aiter is on IPC handles and does not pay it.
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

from mori.ops.gemm_ar import ArConfig

VMM_SLACK = 256 * 1024 * 1024


def _setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return local_rank, rank, world_size, UniqueId.from_bytes(payload[0])


def make_input(rank: int, cfg: ArConfig) -> torch.Tensor:
    """Deterministic, rank-dependent, and cheap to reproduce as a reference.

    Values stay small so the fp32 accumulation of ``world_size`` terms is exact
    in bf16 -- the test asserts bit-exactness, so the data must not be what
    introduces rounding.
    """
    idx = torch.arange(cfg.num_elems, device="cuda", dtype=torch.float32)
    vals = ((idx % 7.0) - 3.0) * 0.25 + float(rank)
    return vals.view(cfg.m, cfg.n).to(torch.bfloat16)


def reference(world_size: int, cfg: ArConfig) -> torch.Tensor:
    acc = torch.zeros(cfg.num_elems, device="cuda", dtype=torch.float32)
    for r in range(world_size):
        acc += make_input(r, cfg).view(-1).float()
    return acc.view(cfg.m, cfg.n).to(torch.bfloat16)


def _median_us(fn, warmup: int, iters: int, *, graph: bool = True) -> float:
    """Median wall time of one all-reduce.

    Graph replay is the default because the eager path pays flydsl's Python
    launch wrapper on every call, which at decode sizes is larger than the
    collective itself. Capture is safe here: the barrier flags are monotonic and
    live in device memory, so a replay advances them exactly like a fresh launch
    (the same property that lets aiter's custom AR be captured).
    """
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
    local_rank, rank, world_size, uid = _setup_distributed()
    cfg = ArConfig(
        world_size=world_size,
        m=args.m,
        n=args.n,
        max_blocks=max(args.max_blocks, args.blocks or 0),
        force_blocks=args.blocks,
        force_reduce_blocks=args.reduce_blocks,
        # LSA reduces straight out of the peers' input regions; SDMA has to be
        # given somewhere for the copy engine to land each peer's slice.
        recv_slots=world_size if args.backend == "sdma" else 0,
    )
    cfg.validate()

    vmm = max(4 * cfg.window_bytes + VMM_SLACK, VMM_SLACK)
    result = None

    with Communicator.init(world_size, rank, uid, per_rank_vmm=vmm) as comm:
        mem = comm.alloc_mem(cfg.window_bytes)
        win = comm.register_window(mem.ptr, mem.size)

        whole = from_gpu_ptr(mem.ptr, (cfg.window_bytes,), torch.uint8)
        whole.zero_()
        inp = from_gpu_ptr(mem.ptr + cfg.input_off, (cfg.m, cfg.n), torch.bfloat16)
        out = from_gpu_ptr(mem.ptr + cfg.output_off, (cfg.m, cfg.n), torch.bfloat16)
        inp.copy_(make_input(rank, cfg))
        torch.cuda.synchronize()

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        if args.backend == "sdma":
            reqs.sdma_queue_count = args.sdma_queues
        dc = comm.create_dev_comm(reqs)

        if args.backend == "lsa":
            from mori.ops.gemm_ar import build_lsa_ar

            launch, stage = build_lsa_ar(
                cfg, rank, force_stage=args.force_stage, gather=args.gather
            )
        elif args.backend == "sdma":
            from mori.ops.gemm_ar import build_sdma_phases

            parts = build_sdma_phases(
                cfg,
                rank,
                queues=args.sdma_queues,
                signal=args.sdma_signal,
                reduce_blocks=args.reduce_blocks,
            )

            def launch(dc_ptr, win_h, stream=None):
                for k in ("scatter", "reduce", "gather"):
                    parts[k](dc_ptr, win_h, stream=stream)

            stage = 2
        else:
            raise ValueError(f"unknown backend {args.backend!r}")

        def once():
            # The stream must be passed explicitly. flydsl's default
            # ``fx.Stream(None)`` resolves to the legacy default stream, which is
            # not torch's capture stream, so under ``torch.cuda.graph`` the work
            # lands outside the graph and replay measures an empty graph (a
            # constant ~4.5us whatever the size).
            launch(
                dc.ptr,
                win.handle,
                stream=fx.Stream(torch.cuda.current_stream()),
            )

        # warm up + correctness on the same path the timing loop uses
        comm.barrier()
        once()
        torch.cuda.synchronize()
        comm.barrier()

        validated = True
        rel_l2 = 0.0
        if not args.skip_validation:
            ref = reference(world_size, cfg)
            diff = (out.float() - ref.float()).norm().item()
            denom = ref.float().norm().item()
            rel_l2 = diff / denom if denom else diff
            validated = rel_l2 == 0.0
            if not validated:
                # Which owner's slice is wrong localises the bug immediately:
                # only-mine-right => the all-gather or its barrier; mine-wrong
                # too => the reduce-scatter or the index math.
                packs = cfg.packs_per_rank
                per_pack = out.view(-1).numel() // cfg.num_packs
                bad = []
                for r in range(world_size):
                    lo = r * packs * per_pack
                    hi = cfg.owner_pack_range(r)[1] * per_pack
                    d = (out.view(-1)[lo:hi].float() - ref.view(-1)[lo:hi].float())
                    if d.norm().item() != 0.0:
                        bad.append(r)
                print(
                    f"[rank {rank}] VALIDATION FAILED relL2={rel_l2:.3e} "
                    f"bad_owner_slices={bad} mine_ok={rank not in bad}",
                    flush=True,
                )

        comm.barrier()
        elapsed_us = _median_us(
            once, args.warmup, args.iters, graph=not args.eager
        )
        comm.barrier()

        stats = torch.tensor([elapsed_us], dtype=torch.float64)
        gathered = [torch.zeros_like(stats) for _ in range(world_size)]
        dist.all_gather(gathered, stats)
        per_rank = [float(t[0]) for t in gathered]
        max_rank = max(range(world_size), key=lambda r: per_rank[r])
        max_us = per_rank[max_rank]

        ok = torch.tensor([1 if validated else 0], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        validated = bool(ok.item())

        if rank == 0:
            gbps = cfg.remote_bytes_per_rank * world_size / (max_us * 1e-6) / 1e9
            result = {
                "backend": args.backend,
                "stage": stage,
                "gather": args.gather if args.backend == "lsa" else None,
                "world_size": world_size,
                "m": cfg.m,
                "n": cfg.n,
                "dtype": "bf16",
                "payload_bytes": cfg.nbytes,
                "slice_bytes": cfg.slice_bytes,
                "blocks": cfg.blocks,
                "threads": cfg.threads,
                "max_rank_time_ms": max_us / 1000.0,
                "max_rank_time_us": max_us,
                "critical_rank": max_rank,
                "per_rank_time_us": per_rank,
                "remote_bytes_per_rank": cfg.remote_bytes_per_rank,
                "aggregate_gbps": gbps,
                "rel_l2": rel_l2,
                "validated": validated,
                "timing": "eager" if args.eager else "graph",
            }
            print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
            print(
                f"[ar] backend={args.backend} stage={stage} m={cfg.m} n={cfg.n} "
                f"blocks={cfg.blocks} max_rank_time={max_us:.2f}us "
                f"aggregate={gbps:.1f}GB/s validated={validated}",
                flush=True,
            )
            if args.json_out:
                with open(args.json_out, "w") as fh:
                    json.dump(result, fh, indent=2, sort_keys=True)

    dist.barrier()
    return 0 if (result is None or result["validated"]) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("lsa", "sdma"), default="lsa")
    p.add_argument("-m", type=int, default=4096)
    p.add_argument("-n", type=int, default=7168)
    p.add_argument(
        "--force-stage",
        type=int,
        choices=(1, 2),
        default=None,
        help="override the size-based 1-stage/2-stage choice (LSA only)",
    )
    p.add_argument(
        "--gather",
        choices=("interleaved", "sequential"),
        default="interleaved",
        help="2-stage all-gather loop nesting (LSA only): 'interleaved' unrolls "
        "the peer loop inside the index loop so all links stream at once; "
        "'sequential' drains one peer at a time and is ~3x slower",
    )
    p.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="override the grid size (default: aiter's formula, capped at 80)",
    )
    p.add_argument(
        "--max-blocks",
        type=int,
        default=80,
        help="signal-array rows; must be >= --blocks",
    )
    p.add_argument(
        "--reduce-blocks",
        type=int,
        default=0,
        help="grid for the SDMA reduce (0 = layout default). Separate from "
        "--blocks because that kernel is local-HBM bound, not xGMI bound",
    )
    p.add_argument("--sdma-queues", type=int, default=8)
    p.add_argument(
        "--sdma-signal",
        action="store_true",
        help="attach put's trailing local ATOMIC. Nothing polls it and quiet is "
        "documented as signal-independent, so this only exists to measure the "
        "claim in FlyDSL's Sdma docstring that no-signal puts cannot be drained",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=51)
    p.add_argument(
        "--eager",
        action="store_true",
        help="time direct launches instead of graph replay (includes flydsl's "
        "Python launch overhead)",
    )
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--json-out")
    return p


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
