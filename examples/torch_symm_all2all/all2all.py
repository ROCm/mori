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
flat_base + rank*stride. No mori shmem or cco.
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
    p.add_argument(
        "--addressing",
        choices=("flat", "ptrs", "both"),
        default="both",
        help="flat_base + peer*stride, an N-entry peer pointer array, or both",
    )
    return p.parse_args()


def main():
    # Not at module scope: the extension links libc10, so torch must load first, and
    # import sorting would hoist it above torch.
    import all2all_kernel

    args = parse_args()
    rank_id = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
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

    # Receive window: one chunk per source rank, writable by peers.
    recv = symm_mem.empty(
        world_size * elems_per_chunk, dtype=torch.int32, device=device
    )
    recv.zero_()
    # Ordinary local memory. Chunk p carries a value identifying (me -> p).
    send = torch.empty(world_size * elems_per_chunk, dtype=torch.int32, device=device)
    for p in range(world_size):
        send[p * elems_per_chunk : (p + 1) * elems_per_chunk] = rank_id * 1000 + p
    torch.cuda.synchronize()

    hdl = symm_mem.rendezvous(recv, group_name)
    base, stride = flat_layout(hdl)
    # The same peers, as the N-entry device array torch's API always provides.
    peers_dev = hdl.buffer_ptrs_dev

    modes = ("flat", "ptrs") if args.addressing == "both" else (args.addressing,)
    launch = {
        "flat": lambda: all2all_kernel.all2all_push(
            send, base, stride, chunk_bytes, rank_id, world_size
        ),
        "ptrs": lambda: all2all_kernel.all2all_push_ptrs(
            send, peers_dev, chunk_bytes, rank_id, world_size
        ),
    }

    if rank_id == 0:
        print(
            f"world={world_size}  handle={handle_type(local_rank)}  chunk={args.chunk_kib} KiB"
        )
        print(f"flat window: base={base:#x} stride={stride >> 20} MiB")
        strided = all(p == base + r * stride for r, p in enumerate(hdl.buffer_ptrs))
        print(f"peer(r) == base + r*stride: {strided}")

    def run_once(mode):
        launch[mode]()
        torch.cuda.synchronize()
        # No device-side barrier in the backend yet, so ranks meet on the host.
        dist.barrier()

    errors = 0
    for mode in modes:
        # Zeroing is local, but peers write into this window: everyone must finish
        # clearing before anyone starts pushing, or a late zero wipes an early write.
        recv.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        run_once(mode)
        # Chunk r must hold what rank r sent us: r*1000 + our rank.
        bad = 0
        for r in range(world_size):
            got = recv[r * elems_per_chunk].item()
            want = r * 1000 + rank_id
            if got != want:
                bad += 1
                print(
                    f"[rank {rank_id}] {mode} chunk {r}: got {got}, want {want}",
                    flush=True,
                )
        errors += bad
        if rank_id == 0 and bad == 0:
            print(f"correctness ({mode}): OK ({world_size}x{world_size} chunks)")

    # Only (world_size-1) chunks leave the device; the self chunk stays local.
    remote_bytes = (world_size - 1) * chunk_bytes
    for mode in modes:
        for _ in range(args.warmup):
            run_once(mode)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        dist.barrier()
        start.record()
        for _ in range(args.iters):
            launch[mode]()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / args.iters
        gbps = remote_bytes / (ms / 1e3) / 1e9
        totals = torch.tensor([gbps], dtype=torch.float64)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if rank_id == 0:
            label = "base + rank*stride" if mode == "flat" else "peer pointer array"
            print(
                f"{label:<20}: {ms * 1e3:7.1f} us/iter, {totals.item():8.1f} GB/s aggregate"
            )

    if rank_id == 0:
        print("SUCCESS" if errors == 0 else "FAILED")

    dist.barrier()
    del hdl, recv
    gc.collect()
    dist.destroy_process_group()
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
