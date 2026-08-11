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
"""All-to-all with a custom HIP kernel over mori's torch SymmetricMemory backend.

    python3 setup.py build_ext --inplace
    torchrun --nnodes=1 --nproc_per_node=<gpus> all2all.py

The receive buffer is an ordinary symm_mem tensor; the kernel reaches peers purely as
flat_base + rank*stride. No mori shmem or cco involved.
"""

import argparse
import gc
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from mori.allocator import flat_layout, handle_type, register_symm_backend


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunk-kib", type=int, default=256, help="bytes sent to each peer")
    p.add_argument("--iters", type=int, default=20, help="timed iterations")
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


def main():
    # Imported here, not at module scope: the extension links libc10, so torch must be
    # loaded first. Module-level import sorting would put it ahead of torch.
    import all2all_kernel

    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist.init_process_group("gloo")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    group_name = dist.group.WORLD.group_name

    register_symm_backend()
    symm_mem.set_backend("MORI")
    symm_mem.enable_symm_mem_for_group(group_name)

    chunk_bytes = args.chunk_kib * 1024
    elems_per_chunk = chunk_bytes // 4

    # Receive window: one chunk per source rank. Symmetric, so peers can write into it.
    recv = symm_mem.empty(world * elems_per_chunk, dtype=torch.int32, device=device)
    recv.zero_()
    # Send buffer is ordinary local memory. Chunk p carries a value identifying (me -> p).
    send = torch.empty(world * elems_per_chunk, dtype=torch.int32, device=device)
    for p in range(world):
        send[p * elems_per_chunk : (p + 1) * elems_per_chunk] = rank * 1000 + p
    torch.cuda.synchronize()

    hdl = symm_mem.rendezvous(recv, group_name)
    base, stride = flat_layout(recv)

    if rank == 0:
        print(
            f"world={world}  handle={handle_type(local_rank)}  chunk={args.chunk_kib} KiB"
        )
        print(f"flat window: base={base:#x} stride={stride >> 20} MiB")
        strided = all(p == base + r * stride for r, p in enumerate(hdl.buffer_ptrs))
        print(f"peer(r) == base + r*stride: {strided}")

    def run_once():
        all2all_kernel.all2all_push(send, base, stride, chunk_bytes, rank, world)
        torch.cuda.synchronize()
        # The backend has no device-side barrier yet, so ranks meet on the host before
        # anyone reads what the peers wrote.
        dist.barrier()

    run_once()

    # Chunk r of our receive window must hold what rank r sent us: r*1000 + our rank.
    errors = 0
    for r in range(world):
        got = recv[r * elems_per_chunk].item()
        want = r * 1000 + rank
        if got != want:
            errors += 1
            print(f"[rank {rank}] chunk {r}: got {got}, expected {want}", flush=True)
    if rank == 0 and errors == 0:
        print(f"correctness: OK ({world}x{world} chunks)")

    for _ in range(args.warmup):
        run_once()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(args.iters):
        all2all_kernel.all2all_push(send, base, stride, chunk_bytes, rank, world)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.iters

    # Each rank pushes (world-1) chunks off-device; the self chunk stays local.
    remote_bytes = (world - 1) * chunk_bytes
    gbps = remote_bytes / (ms / 1e3) / 1e9
    totals = torch.tensor([gbps], dtype=torch.float64)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(
            f"push all-to-all: {ms * 1e3:.1f} us/iter, {totals.item():.1f} GB/s aggregate"
        )
        print("SUCCESS" if errors == 0 else "FAILED")

    dist.barrier()
    del hdl, recv
    gc.collect()
    dist.destroy_process_group()
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
