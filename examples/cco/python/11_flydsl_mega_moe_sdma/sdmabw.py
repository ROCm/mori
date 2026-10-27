#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""What one cco SDMA channel actually moves, with no GEMM in the way.

Same shape as the fused combine's transport -- one comm wave per peer, each
putting a contiguous run from its own window into the peer's, then draining --
but nothing else in the kernel. Reports per-link and per-rank bandwidth for a
sweep of transfer sizes and channel counts, so the fused kernel's exposed
transfer time can be checked against what the hardware will actually do.

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 sdmabw.py \\
        --mb 0.36,0.72,1.44,2.89,5.78 --queues 1,2
"""

import argparse
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

import torch
import torch.distributed as dist

import mori.shmem as ms
from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements, Communicator

WAVE = 64


def build_xfer(*, npes, rank, comm_wgs, queues, split, puts):
    """One wave per peer: ``puts`` back-to-back runs, then drain every channel."""
    import mori.cco.device.flydsl as cco

    n_waves = comm_wgs * 4

    @flyc.kernel(
        name=f"sdmabw_p{npes}_w{comm_wgs}q{queues}s{split}n{puts}_r{rank}",
        known_block_size=[WAVE * 4, 1, 1],
    )
    def xfer(dev_comm: fx.Int64, win: fx.Int64, i64_bytes: fx.Int64, i64_stride: fx.Int64):
        tid = fx.thread_idx.x
        lane = tid % fx.Int32(WAVE)
        wave = rocdl.readfirstlane(T.i32, tid // fx.Int32(WAVE))
        owner = fx.block_idx.x * fx.Int32(4) + wave
        if owner < fx.Int32(npes):
            sdma = cco.DevComm(dev_comm).sdma()
            if owner != fx.Int32(rank):
                if lane == fx.Int32(0):
                    for p in range_constexpr(puts):
                        # Distinct source/destination offsets per put so the copy
                        # engine cannot coalesce the repeats into one transfer.
                        off = fx.Int64(p) * i64_stride
                        for q in range_constexpr(split):
                            part = i64_bytes // fx.Int64(split)
                            sdma.put(
                                owner,
                                win,
                                off + fx.Int64(q) * part,
                                win,
                                off + fx.Int64(q) * part,
                                part,
                                (p + q) % queues if split > 1 else p % queues,
                                coop=cco.CoopScope.THREAD,
                            )
                    sdma.quiet(owner)

    @flyc.jit
    def launch(dev_comm: fx.Int64, win: fx.Int64, i64_bytes: fx.Int64,
               i64_stride: fx.Int64, stream: fx.Stream):
        xfer(dev_comm, win, i64_bytes, i64_stride).launch(
            grid=(comm_wgs, 1, 1), block=(WAVE * 4, 1, 1), stream=stream
        )

    return launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mb", default="0.36,0.72,1.44,2.89,5.78", help="MB per put")
    parser.add_argument("--queues", default="1,2", help="channels per peer")
    parser.add_argument("--split", default="1", help="split one put across channels")
    parser.add_argument("--puts", type=int, default=8, help="back-to-back puts per peer")
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    sizes = [float(v) for v in args.mb.split(",")]
    queue_list = [int(v) for v in args.queues.split(",")]
    splits = [int(v) for v in args.split.split(",")]

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("gloo")
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")

    try:
        max_bytes = int(max(sizes) * (1 << 20))
        stride = (max_bytes + 4095) // 4096 * 4096
        total = stride * args.puts * 2  # src half and dst half never overlap
        rows = []
        for queues in queue_list:
            uid = Communicator.get_unique_id() if rank == 0 else None
            holder = [uid]
            dist.broadcast_object_list(holder, src=0)
            comm = Communicator.init(world, rank, holder[0])
            mem = comm.alloc_mem(total)
            win = comm.register_window(mem.ptr, mem.size)
            reqs = CCODevCommRequirements()
            reqs.gda_connection_type = GDA_CONNECTION_NONE
            reqs.gda_signal_count = 0
            reqs.gda_counter_count = 0
            reqs.sdma_queue_count = queues
            dev_comm = comm.create_dev_comm(reqs)
            torch.cuda.synchronize()
            comm.barrier()

            comm_wgs = max(1, (world + 3) // 4)
            for split in splits:
                if split > queues:
                    continue
                fn = build_xfer(
                    npes=world, rank=rank, comm_wgs=comm_wgs,
                    queues=queues, split=split, puts=args.puts,
                )
                for mb in sizes:
                    nbytes = int(mb * (1 << 20))
                    s_fx = fx.Stream(torch.cuda.current_stream().cuda_stream)

                    def once():
                        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

                        _run_compiled(
                            fn, fx.Int64(dev_comm.ptr), fx.Int64(win.handle),
                            fx.Int64(nbytes), fx.Int64(stride), s_fx,
                        )

                    once()
                    torch.cuda.synchronize()
                    dist.barrier()
                    for _ in range(3):
                        once()
                    torch.cuda.synchronize()
                    dist.barrier()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iters):
                        once()
                    end.record()
                    torch.cuda.synchronize()
                    ms_per = start.elapsed_time(end) / args.iters
                    # Each rank sends to world-1 peers, puts runs of nbytes each.
                    per_link = nbytes * args.puts / (ms_per * 1e-3) / 1e9
                    per_rank = per_link * (world - 1)
                    t = torch.tensor([per_link, per_rank], dtype=torch.float64)
                    dist.all_reduce(t)
                    t /= world
                    rows.append((queues, split, mb, ms_per, float(t[0]), float(t[1])))
                    if rank == 0:
                        print(
                            f"  queues={queues} split={split} put={mb:6.2f} MB "
                            f"x{args.puts}  {ms_per:8.4f} ms  "
                            f"per-link {t[0]:7.1f} GB/s  per-rank {t[1]:7.1f} GB/s",
                            flush=True,
                        )
            comm.finalize() if hasattr(comm, "finalize") else None

        if rank == 0:
            print("\n| queues | split | MB/put | ms | per-link GB/s | per-rank GB/s |")
            print("|---:|---:|---:|---:|---:|---:|")
            for q, sp, mb, t, pl, pr in rows:
                print(f"| {q} | {sp} | {mb:.2f} | {t:.4f} | {pl:.1f} | {pr:.1f} |")
            print("\nSUCCESS", flush=True)
    finally:
        try:
            ms.shmem_finalize()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
