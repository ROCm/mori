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
"""All-to-all through the CCO LSA view of a torch SymmetricMemory tensor.

    torchrun --nnodes=1 --nproc_per_node=<gpus> all2all_flydsl_lsa.py

The allocation and collective rendezvous are still torch-owned. MORI returns an
owning ``CcoLsaWindow`` for the same physical allocation, and the FlyDSL kernel
uses ``cco.Window(handle).lsa_ptr(peer, offset)`` instead of loading
``hdl.buffer_ptrs_dev[peer]``.
"""

import argparse
import functools
import gc
import os
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import rocdl
from flydsl.expr import buffer_ops
from flydsl.expr.typing import Int32, Int64, T
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from mori.allocator import get_cco_window, handle_type
import mori.cco.device.flydsl as cco


THREADS = 256
_CM_UNCACHED = 3  # SC0|SC1: peer stores bypass L1 and L2.


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def _all2all_push_lsa(
    send_ptr: Int64,
    win_handle: Int64,
    chunk_elems: Int64,
    rank_id: Int32,
    total_elems: Int64,
):
    tid = fx.Int64(fx.thread_idx.x)
    linear = fx.Int64(fx.block_idx.x) * fx.Int64(THREADS) + tid
    if linear < total_elems:
        peer = fx.Int32(linear // chunk_elems)
        elem = linear % chunk_elems
        src = buffer_ops.create_buffer_resource_from_addr(
            send_ptr + linear * fx.Int64(4)
        )
        dst_base = fx.Int64(
            cco.Window(win_handle).lsa_ptr(
                peer, fx.Int64(rank_id) * chunk_elems * fx.Int64(4)
            )
        )
        dst = buffer_ops.create_buffer_resource_from_addr(dst_base + elem * fx.Int64(4))
        value = buffer_ops.buffer_load(src, 0, vec_width=1, dtype=T.i32)
        buffer_ops.buffer_store(value, dst, 0, cache_modifier=_CM_UNCACHED)
    rocdl.s_waitcnt(0)


@functools.cache
def _launcher(blocks: int):
    @flyc.jit
    def run(
        send_ptr: Int64,
        win_handle: Int64,
        chunk_elems: Int64,
        rank_id: Int32,
        total_elems: Int64,
        stream=fx.Stream(None),
    ):
        _all2all_push_lsa(
            send_ptr, win_handle, chunk_elems, rank_id, total_elems
        ).launch(grid=(blocks, 1, 1), block=[THREADS, 1, 1], stream=stream)

    return run


def all2all_push_lsa(send, window, chunk_bytes, rank_id, world_size):
    if not send.is_contiguous() or send.dtype != torch.int32:
        raise ValueError("send must be a contiguous int32 tensor")
    if window.lsa_size != world_size or window.lsa_start != 0:
        raise RuntimeError(
            "Stage 2 LSA example requires the whole process group to be one LSA team"
        )
    chunk_elems = chunk_bytes // send.element_size()
    total_elems = world_size * chunk_elems
    if send.numel() != total_elems:
        raise ValueError(f"send has {send.numel()} elements, expected {total_elems}")
    blocks = (total_elems + THREADS - 1) // THREADS
    _launcher(blocks)(
        send.data_ptr(),
        window.window_handle,
        chunk_elems,
        rank_id,
        total_elems,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-kib", type=int, default=256)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    for key, value in (
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29500"),
    ):
        os.environ.setdefault(key, value)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("gloo")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    group_name = dist.group.WORLD.group_name
    symm_mem.set_backend("MORI")
    symm_mem.enable_symm_mem_for_group(group_name)

    chunk_bytes = args.chunk_kib * 1024
    elems = chunk_bytes // 4
    recv = symm_mem.empty(world_size * elems, dtype=torch.int32, device=device)
    recv.zero_()
    send = torch.empty_like(recv)
    for peer in range(world_size):
        send[peer * elems : (peer + 1) * elems] = rank * 1000 + peer
    torch.cuda.synchronize()

    hdl = symm_mem.rendezvous(recv, group_name)
    lsa = get_cco_window(recv, group_name)
    if rank == 0:
        print(
            f"kernel=FlyDSL-LSA world={world_size} handle={handle_type(local_rank)} "
            f"chunk={args.chunk_kib} KiB stride={lsa.per_rank_size >> 30} GiB"
        )

    def run_once(window):
        all2all_push_lsa(send, window, chunk_bytes, rank, world_size)
        torch.cuda.synchronize()
        dist.barrier()

    dist.barrier()
    run_once(lsa)
    errors = 0
    for source in range(world_size):
        got = recv[source * elems].item()
        want = source * 1000 + rank
        if got != want:
            errors += 1
            print(f"[rank {rank}] chunk {source}: got {got}, want {want}", flush=True)
    failed = torch.tensor([errors], dtype=torch.int64)
    dist.all_reduce(failed)

    for _ in range(args.warmup):
        run_once(lsa)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    dist.barrier()
    start.record()
    for _ in range(args.iters):
        all2all_push_lsa(send, lsa, chunk_bytes, rank, world_size)
    end.record()
    torch.cuda.synchronize()
    milliseconds = start.elapsed_time(end) / args.iters
    bandwidth = torch.tensor(
        [(world_size - 1) * chunk_bytes / (milliseconds / 1e3) / 1e9],
        dtype=torch.float64,
    )
    dist.all_reduce(bandwidth)
    if rank == 0:
        print(
            f"{milliseconds * 1e3:7.1f} us/iter, "
            f"{bandwidth.item():8.1f} GB/s aggregate"
        )
        print("SUCCESS" if failed.item() == 0 else "FAILED")

    dist.barrier()
    del lsa, hdl, recv
    gc.collect()
    dist.destroy_process_group()
    return 0 if failed.item() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
