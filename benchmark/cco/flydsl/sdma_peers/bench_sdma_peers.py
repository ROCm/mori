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
"""Benchmark: cco SDMA puts from every GPU to all of its peers at once.

Each rank posts one ``put_ptr`` (with an engine-side completion signal) to every
peer from a one-wave FlyDSL kernel, then a second kernel spins until every peer's
put to this rank has landed. Two patterns:

- ``ag``:  the same block goes to every peer (the transfer of an all-gather);
- ``a2a``: a different block goes to each peer (the transfer of a reduce-scatter).

The default size is a 12000 x 2560 bf16 tensor split across ranks, i.e. each rank
sends ``ceil(12000 / world) * 2560 * 2`` bytes to each peer. The time is push +
wait on the compute stream, median over ``--iters``, taken on the slowest rank.
Run it before and after a change to the SDMA engine assignment, with the same GPUs:

    MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \\
    torchrun --standalone --nproc_per_node=4 bench_sdma_peers.py --check

Destinations are torch CUDA IPC buffers rather than cco windows, so this also runs
on hosts where VMM windows are unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int32, Int64
from torch.multiprocessing.reductions import reduce_tensor

import mori.cco.device.flydsl as cco
from mori.cco import (
    CCODevCommRequirements,
    Communicator,
    GDA_CONNECTION_NONE,
    UniqueId,
)
from mori.cco.device.flydsl.signals import i32_type, signal_ptr

WAVE = 64


def build_kernels(rank: int, world: int):
    def load_i64(addr):
        return llvm.load(ir.IntegerType.get_signless(64), signal_ptr(addr), alignment=8)

    def spin_until(ptr, target):
        def load():
            return llvm.load(
                i32_type(),
                ptr,
                alignment=4,
                ordering=llvm.AtomicOrdering.monotonic,
                syncscope="",
            )

        loop = scf.WhileOp([i32_type()], [load()])
        cond = ir.Block.create_at_start(loop.before, [i32_type()])
        body = ir.Block.create_at_start(loop.after, [i32_type()])
        with ir.InsertionPoint(cond):
            cur = fx.Int32(cond.arguments[0])
            scf.ConditionOp(
                ((cur - fx.Int32(target)) < fx.Int32(0)).ir_value(), [cond.arguments[0]]
            )
        with ir.InsertionPoint(body):
            scf.YieldOp([load()])

    # Thread `peer` puts to that peer: its slot `rank` in the peer's receive buffer,
    # from `src` (ag) or from block `peer` of `src` (a2a, src_stride = nbytes).
    @flyc.kernel(known_block_size=[WAVE, 1, 1])
    def push(
        dev_comm: Int64, dst_table: Int64, src: Int64, nbytes: Int64, src_stride: Int64
    ):
        tid = fx.thread_idx.x
        sdma = cco.DevComm(dev_comm).sdma()
        if tid < world:
            if tid != rank:
                base = fx.Int64(
                    load_i64(fx.Int64(dst_table) + fx.Int64(tid) * fx.Int64(8))
                )
                sdma.put_ptr(
                    tid,
                    base + fx.Int64(rank) * fx.Int64(nbytes),
                    fx.Int64(src) + fx.Int64(tid) * fx.Int64(src_stride),
                    fx.Int64(nbytes),
                    0,
                    remote_signal=True,
                )
        fgpu.barrier()

    # The signal of peer `tid` on queue 0 counts its signalled puts to this rank.
    @flyc.kernel(known_block_size=[WAVE, 1, 1])
    def wait(dev_comm: Int64, expected: Int32):
        tid = fx.thread_idx.x
        sdma = cco.DevComm(dev_comm).sdma()
        if tid < world:
            if tid != rank:
                spin_until(
                    signal_ptr(fx.Int64(sdma.signal_slot(tid, 0))), fx.Int32(expected)
                )
        fgpu.barrier()

    @flyc.jit
    def push_wait(
        dev_comm: Int64,
        dst_table: Int64,
        src: Int64,
        nbytes: Int64,
        src_stride: Int64,
        expected: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        push(dev_comm, dst_table, src, nbytes, src_stride).launch(
            grid=(1, 1, 1), block=[WAVE, 1, 1], stream=stream
        )
        wait(dev_comm, expected).launch(
            grid=(1, 1, 1), block=[WAVE, 1, 1], stream=stream
        )

    return push_wait


def open_peer_buffers(local: torch.Tensor, rank: int, world: int):
    # One reduce_tensor per importer: torch's IPC refcount expects a single consumer each.
    handles = [None] * world
    mine = [None if p == rank else reduce_tensor(local) for p in range(world)]
    dist.all_gather_object(handles, mine)
    peers = []
    for r, theirs in enumerate(handles):
        if r == rank:
            peers.append(local)
        else:
            fn, args = theirs[rank]
            peers.append(fn(*args))
    table = torch.tensor(
        [p.data_ptr() for p in peers], dtype=torch.int64, device=local.device
    )
    return peers, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=12000)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--modes", default="ag,a2a")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument(
        "--check", action="store_true", help="verify the received bytes once per mode"
    )
    ap.add_argument(
        "--queues",
        type=int,
        default=1,
        help="SDMA queues created per peer (cco sdma_queue_count); the puts use the first",
    )
    args = ap.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()

    uid = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0)
    comm = Communicator.init(
        world, rank, UniqueId.from_bytes(uid[0]), per_rank_vmm=64 << 20
    )
    reqs = CCODevCommRequirements()
    reqs.gda_connection_type = GDA_CONNECTION_NONE
    reqs.gda_signal_count = 0
    reqs.gda_counter_count = 0
    reqs.sdma_queue_count = args.queues
    dev_comm = comm.create_dev_comm(reqs)

    rows = -(-args.rows // world)
    nbytes = rows * args.hidden * 2
    send = torch.empty(world * nbytes, dtype=torch.uint8, device=dev)
    recv = torch.zeros(world * nbytes, dtype=torch.uint8, device=dev)
    peers, dst_table = open_peer_buffers(recv, rank, world)

    push_wait = build_kernels(rank, world)
    compiled = None
    puts = 0  # signalled puts each peer has sent to this rank so far

    def run(src_stride: int):
        nonlocal compiled, puts
        puts += 1
        call = (
            dev_comm.ptr,
            dst_table.data_ptr(),
            send.data_ptr(),
            nbytes,
            src_stride,
            puts,
        )
        stream = fx.Stream(torch.cuda.current_stream())
        if compiled is None:
            # flyc.compile also launches once; that launch is this call.
            compiled = flyc.compile(push_wait, *call, stream)
        else:
            compiled(*call, stream)

    tick = torch.ones(1)
    results = {}
    for mode in args.modes.split(","):
        stride = 0 if mode == "ag" else nbytes
        if args.check:
            # Byte b of the block for peer p from rank r is (r * 31 + p * 7 + b) % 251.
            idx = torch.arange(nbytes, device=dev, dtype=torch.int64)
            for p in range(world):
                send[p * nbytes : (p + 1) * nbytes] = (
                    rank * 31 + (p if stride else 0) * 7 + idx
                ) % 251
            dist.all_reduce(tick)
            torch.cuda.synchronize()
            dist.all_reduce(tick)
            run(stride)
            torch.cuda.synchronize()
            ok = all(
                torch.equal(
                    recv[r * nbytes : (r + 1) * nbytes],
                    ((r * 31 + (rank if stride else 0) * 7 + idx) % 251).to(
                        torch.uint8
                    ),
                )
                for r in range(world)
                if r != rank
            )
            flag = torch.tensor([int(ok)])
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            results[f"{mode}_check"] = bool(flag.item())
        times = []
        for i in range(args.warmup + args.iters):
            dist.all_reduce(tick)
            torch.cuda.synchronize()
            dist.all_reduce(tick)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            run(stride)
            end.record()
            torch.cuda.synchronize()
            if i >= args.warmup:
                times.append(start.elapsed_time(end))
        slowest = torch.tensor([statistics.median(times)])
        dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
        results[mode] = slowest.item()

    if rank == 0:
        out_mb = nbytes * (world - 1) / 1e6
        for mode in args.modes.split(","):
            ms = results[mode]
            check = results.get(f"{mode}_check")
            print(
                f"world={world} queues={args.queues} {mode}: {ms:.3f} ms, "
                f"{out_mb:.1f} MB out per rank, {out_mb / ms:.1f} GB/s"
                + ("" if check is None else f", data ok={check}")
            )
        record = {
            "world": world,
            "queues": args.queues,
            "bytes_per_peer": nbytes,
            **results,
        }
        print("RESULT_JSON " + json.dumps(record))
    # Drop the imported IPC buffers before any rank exits and frees its exports.
    del peers
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.ipc_collect()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
