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
"""Triton BF16 GEMM followed by a CCO LSA column-shard all-to-all."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import struct
import sys

import torch
import torch.distributed as dist

from mori.cco import (
    CCODevCommRequirements,
    Communicator,
    GDA_CONNECTION_NONE,
    UniqueId,
)
from mori.ir.triton import cco
from mori.jit.hip_driver import _check, _get_hip_lib

try:
    from .kernels import gemm_to_staging_kernel, lsa_a2a_copy_kernel
    from .layout import GemmA2AConfig
except ImportError:
    from kernels import gemm_to_staging_kernel, lsa_a2a_copy_kernel
    from layout import GemmA2AConfig

GIB = 1024**3
VMM_SLACK = 64 * 1024**2


def _memset(ptr: int, value: int, size: int) -> None:
    _check(
        _get_hip_lib().hipMemset(
            ctypes.c_void_p(ptr), ctypes.c_int(value), ctypes.c_size_t(size)
        ),
        "hipMemset",
    )


def _read_bf16(ptr: int) -> float:
    raw = ctypes.c_uint16()
    _check(
        _get_hip_lib().hipMemcpy(
            ctypes.byref(raw),
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(2),
            ctypes.c_int(2),
        ),
        "hipMemcpy D2H",
    )
    return struct.unpack("<f", struct.pack("<I", raw.value << 16))[0]


def _setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return local_rank, rank, world_size, UniqueId.from_bytes(payload[0])


def _default_shard_n(world_size: int, n: int) -> int:
    if world_size == 4 and n == 18432:
        return 2560
    if world_size == 8 and n == 18432:
        return 2304
    if n % world_size:
        raise ValueError("N must be divisible by world_size when shard_n is omitted")
    return n // world_size


def make_inputs(rank: int, config: GemmA2AConfig):
    rows = torch.arange(config.m, device="cuda", dtype=torch.int32)
    ks = torch.arange(config.k, device="cuda", dtype=torch.int32)
    cols = torch.arange(config.n, device="cuda", dtype=torch.int32)

    a = (
        0.001 * (rank + 1)
        + 0.01 * ((rows[:, None] % 17).float() - 8.0)
        + 0.002 * ((ks[None, :] % 29).float() - 14.0)
    ).to(torch.bfloat16)
    b = (
        0.003 * ((cols[:, None] % 23).float() - 11.0)
        + 0.001 * ((ks[None, :] % 31).float() - 15.0)
    ).to(torch.bfloat16)
    return a, b


def _sample_reference(
    source_rank: int,
    rows: list[int],
    global_cols: list[int],
    k: int,
) -> torch.Tensor:
    row_ids = torch.tensor(rows, dtype=torch.int32)
    col_ids = torch.tensor(global_cols, dtype=torch.int32)
    ks = torch.arange(k, dtype=torch.int32)
    a = (
        0.001 * (source_rank + 1)
        + 0.01 * ((row_ids[:, None] % 17).float() - 8.0)
        + 0.002 * ((ks[None, :] % 29).float() - 14.0)
    ).to(torch.bfloat16)
    b = (
        0.003 * ((col_ids[:, None] % 23).float() - 11.0)
        + 0.001 * ((ks[None, :] % 31).float() - 15.0)
    ).to(torch.bfloat16)
    return (a.float() @ b.float().T).to(torch.bfloat16).float()


def validate_output(
    mode: str,
    rank: int,
    config: GemmA2AConfig,
    staging_ptr: int,
    recv_ptr: int,
    tail: torch.Tensor,
    tolerance: float,
) -> None:
    rows = [value for value in (0, 17, 511, 1023, 2047) if value < config.m]
    local_cols = [
        value for value in (0, 255, 1024, config.shard_n - 1) if value < config.shard_n
    ]

    if mode == "split-lsa":
        sources = range(config.world_size)
        dst_rank = rank
        base_ptr = recv_ptr
    else:
        sources = (rank,)
        dst_rank = rank
        base_ptr = staging_ptr + dst_rank * config.slab_elements * 2

    global_cols = [dst_rank * config.shard_n + col for col in local_cols]
    for source in sources:
        reference = _sample_reference(source, rows, global_cols, config.k)
        for row_index, row in enumerate(rows):
            for col_index, local_col in enumerate(local_cols):
                if mode == "split-lsa":
                    index = config.recv_index(source, row, local_col)
                else:
                    index = row * config.shard_n + local_col
                actual = _read_bf16(base_ptr + index * 2)
                expected = reference[row_index, col_index].item()
                if abs(actual - expected) > tolerance:
                    raise AssertionError(
                        f"rank={rank} src={source} row={row} col={global_cols[col_index]} "
                        f"actual={actual} expected={expected}"
                    )

    if config.scatter_n < config.n:
        tail_cols = [config.scatter_n, config.n - 1]
        reference = _sample_reference(rank, rows, tail_cols, config.k)
        tail_cpu = tail.view(torch.uint16).cpu()
        for row_index, row in enumerate(rows):
            for col_index, col in enumerate(tail_cols):
                raw = int(tail_cpu[row, col])
                actual = struct.unpack("<f", struct.pack("<I", raw << 16))[0]
                expected = reference[row_index, col_index].item()
                if abs(actual - expected) > tolerance:
                    raise AssertionError(
                        f"tail rank={rank} row={row} col={col} "
                        f"actual={actual} expected={expected}"
                    )


def _launch_gemm(
    config,
    a,
    b,
    staging_ptr,
    tail,
    num_warps,
    num_stages,
    loop_unroll,
    tile_order,
    group_m,
):
    tile_order_id = {"linear": 0, "grouped": 1, "opus": 2}[tile_order]
    gemm_to_staging_kernel[(config.gemm_grid,)](
        a,
        b,
        staging_ptr,
        tail,
        M=config.m,
        N=config.n,
        K=config.k,
        shard_n=config.shard_n,
        scatter_n=config.scatter_n,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        PIPELINE_STAGES=num_stages,
        LOOP_UNROLL=loop_unroll,
        TILE_ORDER=tile_order_id,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_a2a(
    config,
    dc,
    staging_ptr,
    recv_win,
    blocks_per_dst,
    copy_block,
    extern_libs,
):
    grid = config.world_size * blocks_per_dst
    lsa_a2a_copy_kernel[(grid,)](
        dc.ptr,
        staging_ptr,
        recv_win.handle,
        config.m,
        config.shard_n,
        world_size=config.world_size,
        blocks_per_dst=blocks_per_dst,
        COPY_BLOCK=copy_block,
        extern_libs=extern_libs,
        num_warps=copy_block // 64,
    )


def _event_totals(
    e2e_start,
    compute_end,
    comm_start,
    e2e_end,
) -> tuple[float, float, float]:
    e2e = sum(start.elapsed_time(end) for start, end in zip(e2e_start, e2e_end))
    compute = sum(start.elapsed_time(end) for start, end in zip(e2e_start, compute_end))
    comm = sum(start.elapsed_time(end) for start, end in zip(comm_start, e2e_end))
    return e2e, compute, comm


def run(args) -> dict:
    local_rank, rank, world_size, uid = _setup_distributed()
    shard_n = args.shard_n or _default_shard_n(world_size, args.n)
    config = GemmA2AConfig(
        world_size=world_size,
        m=args.m,
        n=args.n,
        k=args.k,
        shard_n=shard_n,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
    )
    config.validate()
    if args.mode == "split-lsa" and config.staging_bytes % 8:
        raise ValueError("staging size must be 8-byte aligned")

    a, b = make_inputs(rank, config)
    tail = torch.empty((config.m, config.n), dtype=torch.bfloat16, device="cuda")
    device_props = torch.cuda.get_device_properties(local_rank)
    blocks_per_dst = args.copy_blocks_per_dst or max(
        1, device_props.multi_processor_count // world_size
    )
    per_rank_vmm = max(GIB, config.staging_bytes + config.recv_bytes + VMM_SLACK)
    extern_libs = cco.get_extern_libs()

    try:
        with Communicator.init(
            world_size, rank, uid, per_rank_vmm=per_rank_vmm
        ) as comm:
            staging_mem = comm.alloc_mem(config.staging_bytes)
            recv_mem = comm.alloc_mem(config.recv_bytes)
            staging_win = comm.register_window(staging_mem.ptr, staging_mem.size)
            recv_win = comm.register_window(recv_mem.ptr, recv_mem.size)
            _memset(staging_win.local_ptr, 0, config.staging_bytes)
            _memset(recv_win.local_ptr, 0, config.recv_bytes)
            tail.zero_()

            reqs = CCODevCommRequirements()
            reqs.gda_connection_type = GDA_CONNECTION_NONE
            reqs.gda_signal_count = 0
            reqs.gda_counter_count = 0
            reqs.sdma_queue_count = 0
            dc = comm.create_dev_comm(reqs)

            def launch_once():
                _launch_gemm(
                    config,
                    a,
                    b,
                    staging_win.local_ptr,
                    tail,
                    args.num_warps,
                    args.num_stages,
                    args.loop_unroll,
                    args.tile_order,
                    args.group_m,
                )
                if args.mode == "split-lsa":
                    _launch_a2a(
                        config,
                        dc,
                        staging_win.local_ptr,
                        recv_win,
                        blocks_per_dst,
                        args.copy_block,
                        extern_libs,
                    )

            comm.barrier()
            for _ in range(args.warmup):
                launch_once()
            torch.cuda.synchronize()
            comm.barrier()

            if not args.skip_validation:
                validate_output(
                    args.mode,
                    rank,
                    config,
                    staging_win.local_ptr,
                    recv_win.local_ptr,
                    tail,
                    args.tolerance,
                )
            comm.barrier()

            e2e_start = []
            compute_end = []
            comm_start = []
            e2e_end = []
            for _ in range(args.iters):
                start = torch.cuda.Event(enable_timing=True)
                comp_end = torch.cuda.Event(enable_timing=True)
                communication_start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _launch_gemm(
                    config,
                    a,
                    b,
                    staging_win.local_ptr,
                    tail,
                    args.num_warps,
                    args.num_stages,
                    args.loop_unroll,
                    args.tile_order,
                    args.group_m,
                )
                comp_end.record()
                communication_start.record()
                if args.mode == "split-lsa":
                    _launch_a2a(
                        config,
                        dc,
                        staging_win.local_ptr,
                        recv_win,
                        blocks_per_dst,
                        args.copy_block,
                        extern_libs,
                    )
                end.record()
                e2e_start.append(start)
                compute_end.append(comp_end)
                comm_start.append(communication_start)
                e2e_end.append(end)

            e2e_end[-1].synchronize()
            e2e_total, compute_total, comm_total = _event_totals(
                e2e_start, compute_end, comm_start, e2e_end
            )
            local = torch.tensor(
                [
                    e2e_total / args.iters,
                    compute_total / args.iters,
                    comm_total / args.iters,
                ],
                dtype=torch.float64,
            )
            gathered = [torch.empty_like(local) for _ in range(world_size)]
            dist.all_gather(gathered, local)
            stats = torch.stack(gathered)
            critical_rank = int(torch.argmax(stats[:, 0]).item())
            max_rank_ms = float(stats[critical_rank, 0])
            avg_rank_ms = float(stats[:, 0].mean())
            critical_compute_ms = float(stats[critical_rank, 1])
            critical_comm_ms = (
                float(stats[critical_rank, 2]) if args.mode == "split-lsa" else 0.0
            )
            residual_ms = max(
                0.0, max_rank_ms - critical_compute_ms - critical_comm_ms
            )
            aggregate_tflops = (
                2.0
                * config.m
                * config.n
                * config.k
                * world_size
                / (max_rank_ms * 1e9)
            )
            effective_comm_gbps = (
                config.remote_bytes_per_rank / (critical_comm_ms * 1e6)
                if critical_comm_ms > 0
                else 0.0
            )
            result = {
                "impl": "triton",
                "mode": args.mode,
                "world_size": world_size,
                "m": config.m,
                "n": config.n,
                "k": config.k,
                "shard_n": config.shard_n,
                "scatter_n": config.scatter_n,
                "warmup": args.warmup,
                "iters": args.iters,
                "block_m": config.block_m,
                "block_n": config.block_n,
                "block_k": config.block_k,
                "num_warps": args.num_warps,
                "num_stages": args.num_stages,
                "loop_unroll": args.loop_unroll,
                "tile_order": args.tile_order,
                "group_m": args.group_m,
                "copy_blocks_per_dst": blocks_per_dst,
                "max_rank_time_ms": max_rank_ms,
                "avg_rank_time_ms": avg_rank_ms,
                "critical_rank": critical_rank,
                "critical_compute_ms": critical_compute_ms,
                "critical_comm_ms": critical_comm_ms,
                "barrier_idle_residual_ms": residual_ms,
                "aggregate_tflops": aggregate_tflops,
                "effective_comm_gbps": effective_comm_gbps,
                "validated": not args.skip_validation,
            }
            if rank == 0:
                print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
                print(
                    f"triton_gemm_a2a M={config.m} N={config.n} K={config.k} "
                    f"mode={args.mode} shard_n={config.shard_n} "
                    f"max_rank_time={max_rank_ms:.4f}ms "
                    f"compute={critical_compute_ms:.4f}ms "
                    f"comm={critical_comm_ms:.4f}ms "
                    f"TFLOP/s={aggregate_tflops:.2f}",
                    flush=True,
                )
                if args.json_out:
                    output = Path(args.json_out)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(result, indent=2) + "\n")
            comm.barrier()
            return result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("local", "split-lsa"), default="split-lsa")
    parser.add_argument("-m", type=int, default=2048)
    parser.add_argument("-n", type=int, default=18432)
    parser.add_argument("-k", type=int, default=8192)
    parser.add_argument("--shard-n", type=int)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--block-m", type=int, default=256)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--loop-unroll", type=int, default=1)
    parser.add_argument(
        "--tile-order",
        choices=("linear", "grouped", "opus"),
        default="opus",
    )
    parser.add_argument("--group-m", type=int, default=8)
    parser.add_argument("--copy-blocks-per-dst", type=int, default=0)
    parser.add_argument("--copy-block", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--json-out")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.warmup < 1 or args.iters < 1:
        raise ValueError("warmup and iters must be positive")
    if args.copy_block <= 0 or args.copy_block % 64:
        raise ValueError("copy_block must be a positive multiple of 64")
    if args.num_stages < 1 or args.loop_unroll < 1:
        raise ValueError("num_stages and loop_unroll must be positive")
    if args.group_m < 1:
        raise ValueError("group_m must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
