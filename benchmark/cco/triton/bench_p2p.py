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
"""Two-rank CCO Triton point-to-point latency/bandwidth benchmark."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

from mori.cco import (
    CCODevCommRequirements,
    Communicator,
    GDA_CONNECTION_FULL,
    GDA_CONNECTION_NONE,
    UniqueId,
)
from mori.ir.triton import cco
from mori.jit.hip_driver import _check, _get_hip_lib

try:
    from .kernels import (
        METRIC_BANDWIDTH,
        METRIC_LATENCY,
        OP_GET,
        OP_PUT,
        gda_p2p_kernel,
        lsa_p2p_kernel,
        sdma_p2p_kernel,
    )
except ImportError:
    from kernels import (  # type: ignore[no-redef]
        METRIC_BANDWIDTH,
        METRIC_LATENCY,
        OP_GET,
        OP_PUT,
        gda_p2p_kernel,
        lsa_p2p_kernel,
        sdma_p2p_kernel,
    )

KIB = 1024
MIB = 1024 * KIB
VMM_SLACK = 64 * MIB


def parse_size(value: str) -> int:
    text = value.strip().lower()
    multiplier = 1
    for suffix, factor in (
        ("gib", 1024**3),
        ("gb", 1000**3),
        ("g", 1024**3),
        ("mib", 1024**2),
        ("mb", 1000**2),
        ("m", 1024**2),
        ("kib", 1024),
        ("kb", 1000),
        ("k", 1024),
    ):
        if text.endswith(suffix):
            multiplier = factor
            text = text[: -len(suffix)]
            break
    return int(text) * multiplier


def size_sweep(min_size: int, max_size: int, factor: int) -> list[int]:
    if min_size <= 0 or max_size < min_size or factor < 2:
        raise ValueError("need 0 < min_size <= max_size and step_factor >= 2")
    values = []
    size = min_size
    while size <= max_size:
        values.append(size)
        size *= factor
    return values


def latency_us(elapsed_ms: float, iters: int) -> float:
    return elapsed_ms * 1000.0 / iters


def bandwidth_gbps(size_bytes: int, elapsed_ms: float, iters: int) -> float:
    return size_bytes * iters * 1000.0 / (elapsed_ms * 1e9)


def _memset(ptr: int, value: int, size: int) -> None:
    _check(
        _get_hip_lib().hipMemset(
            ctypes.c_void_p(ptr), ctypes.c_int(value), ctypes.c_size_t(size)
        ),
        "hipMemset",
    )


def _setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("CCO p2p benchmark requires exactly two ranks")
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return rank, world_size, UniqueId.from_bytes(payload[0])


def _normalize_transport(value: str) -> str:
    return "ibgda" if value == "gda" else value


def _derived_geometry(args) -> tuple[int, int]:
    if args.grid is not None:
        grid = args.grid
    elif args.metric == "latency" or args.transport == "sdma":
        grid = 1
    else:
        grid = 32
    threads = args.threads if args.threads is not None else 256
    if grid <= 0 or threads <= 0 or threads % 64 != 0:
        raise ValueError("grid must be positive and threads must be a positive multiple of 64")
    return grid, threads


def _default_sizes(metric: str) -> tuple[int, int]:
    return (8, 64 * KIB) if metric == "latency" else (64 * KIB, 8 * MIB)


def _launch(
    args,
    *,
    dev_comm: int,
    send_win: int,
    recv_win: int,
    peer_rank: int,
    peer_lsa_rank: int,
    size_bytes: int,
    iterations: int,
    sync_counter,
    grid: int,
    threads: int,
    extern_libs,
):
    op = OP_PUT if args.op == "put" else OP_GET
    metric = METRIC_LATENCY if args.metric == "latency" else METRIC_BANDWIDTH
    num_warps = threads // 64

    if args.transport == "lsa":
        lsa_p2p_kernel[(grid,)](
            dev_comm,
            send_win,
            recv_win,
            peer_lsa_rank,
            size_bytes,
            iterations,
            sync_counter,
            op=op,
            metric=metric,
            copy_block=threads,
            nprograms=grid,
            extern_libs=extern_libs,
            num_warps=num_warps,
        )
    elif args.transport == "sdma":
        sdma_p2p_kernel[(grid,)](
            dev_comm,
            send_win,
            recv_win,
            peer_lsa_rank,
            size_bytes,
            iterations,
            args.sdma_queues,
            op=op,
            metric=metric,
            aggregate=args.aggregate,
            agg_depth=args.agg_depth,
            nprograms=grid,
            extern_libs=extern_libs,
            num_warps=num_warps,
        )
    else:
        gda_p2p_kernel[(grid,)](
            dev_comm,
            send_win,
            recv_win,
            peer_rank,
            size_bytes,
            iterations,
            op=op,
            metric=metric,
            nprograms=grid,
            extern_libs=extern_libs,
            num_warps=num_warps,
        )


def run(args) -> list[dict]:
    rank, world_size, uid = _setup_distributed()
    grid, threads = _derived_geometry(args)
    min_default, max_default = _default_sizes(args.metric)
    min_size = parse_size(args.min_size) if args.min_size else min_default
    max_size = parse_size(args.max_size) if args.max_size else max_default
    sizes = size_sweep(min_size, max_size, args.step_factor)
    if any(size % 8 for size in sizes):
        raise ValueError("all message sizes must be 8-byte aligned")
    if args.metric == "bandwidth" and args.transport != "sdma":
        if any(size % (grid * 8) for size in sizes):
            raise ValueError("LSA/GDA bandwidth sizes must be divisible by grid*8")
    if args.agg_depth < 1 or min(sizes) < args.agg_depth * 8:
        raise ValueError("agg_depth requires at least one 8-byte packet per part")

    gda_enabled = args.transport == "ibgda"
    sdma_enabled = args.transport == "sdma"
    per_rank_vmm = 2 * max_size + VMM_SLACK
    extern_libs = cco.get_extern_libs()
    results: list[dict] = []

    try:
        with Communicator.init(
            world_size, rank, uid, per_rank_vmm=per_rank_vmm
        ) as comm:
            send_mem = comm.alloc_mem(max_size)
            recv_mem = comm.alloc_mem(max_size)
            send_win = comm.register_window(send_mem.ptr, send_mem.size)
            recv_win = comm.register_window(recv_mem.ptr, recv_mem.size)
            _memset(send_win.local_ptr, rank + 1, max_size)
            _memset(recv_win.local_ptr, 0, max_size)

            reqs = CCODevCommRequirements()
            reqs.gda_connection_type = (
                GDA_CONNECTION_FULL if gda_enabled else GDA_CONNECTION_NONE
            )
            reqs.gda_context_count = grid if gda_enabled else 0
            reqs.gda_signal_count = 0
            reqs.gda_counter_count = 0
            reqs.sdma_queue_count = args.sdma_queues if sdma_enabled else 0
            dc = comm.create_dev_comm(reqs)

            peer_rank = 1 - rank
            peer_lsa_rank = 1 - dc.lsa_rank
            sync_counter = torch.zeros(2, dtype=torch.int32, device="cuda")

            for size_bytes in sizes:
                comm.barrier()
                if rank == 0:
                    sync_counter.zero_()
                    _launch(
                        args,
                        dev_comm=dc.ptr,
                        send_win=send_win.handle,
                        recv_win=recv_win.handle,
                        peer_rank=peer_rank,
                        peer_lsa_rank=peer_lsa_rank,
                        size_bytes=size_bytes,
                        iterations=args.warmup,
                        sync_counter=sync_counter,
                        grid=grid,
                        threads=threads,
                        extern_libs=extern_libs,
                    )
                    torch.cuda.synchronize()

                    sync_counter.zero_()
                    start = torch.cuda.Event(enable_timing=True)
                    stop = torch.cuda.Event(enable_timing=True)
                    start.record()
                    _launch(
                        args,
                        dev_comm=dc.ptr,
                        send_win=send_win.handle,
                        recv_win=recv_win.handle,
                        peer_rank=peer_rank,
                        peer_lsa_rank=peer_lsa_rank,
                        size_bytes=size_bytes,
                        iterations=args.iters,
                        sync_counter=sync_counter,
                        grid=grid,
                        threads=threads,
                        extern_libs=extern_libs,
                    )
                    stop.record()
                    stop.synchronize()
                    elapsed_ms = start.elapsed_time(stop)
                    value = (
                        latency_us(elapsed_ms, args.iters)
                        if args.metric == "latency"
                        else bandwidth_gbps(size_bytes, elapsed_ms, args.iters)
                    )
                    result = {
                        "impl": "triton",
                        "transport": _normalize_transport(args.transport),
                        "op": args.op,
                        "metric": args.metric,
                        "scope": args.scope,
                        "size_bytes": size_bytes,
                        "value": value,
                        "unit": "us" if args.metric == "latency" else "GB/s",
                        "elapsed_ms": elapsed_ms,
                        "iters": args.iters,
                        "warmup": args.warmup,
                        "grid": grid,
                        "threads": threads,
                        "sdma_queues": args.sdma_queues if sdma_enabled else None,
                        "aggregate": args.aggregate if sdma_enabled else None,
                        "agg_depth": args.agg_depth if sdma_enabled else None,
                        "rail_mode": "single" if gda_enabled else None,
                    }
                    results.append(result)
                    print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
                comm.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    if rank == 0:
        print(
            f"# cco_triton_{args.op}_{args.metric} "
            f"transport={_normalize_transport(args.transport)} scope={args.scope} "
            f"grid={grid} block={threads} iters={args.iters} warmup={args.warmup}"
        )
        for row in results:
            print(
                f"{row['size_bytes']:>12} B  {row['value']:>12.4f} {row['unit']}",
                flush=True,
            )
        if args.json_out:
            output = Path(args.json_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in results)
            )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=("lsa", "sdma", "ibgda"), required=True)
    parser.add_argument("--op", choices=("put", "get"), required=True)
    parser.add_argument("--metric", choices=("latency", "bandwidth"), required=True)
    parser.add_argument("--scope", choices=("block",), default="block")
    parser.add_argument("--min-size")
    parser.add_argument("--max-size")
    parser.add_argument("--step-factor", type=int, default=2)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--grid", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument(
        "--sdma-queues",
        type=int,
        default=int(os.environ.get("MORI_SDMA_NUM_CHANNELS", "2")),
    )
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--agg-depth", type=int, default=1)
    parser.add_argument("--json-out")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.iters < 1 or args.warmup < 1:
        raise ValueError("iters and warmup must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
