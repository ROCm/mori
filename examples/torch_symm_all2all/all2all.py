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
"""All-to-all over mori's torch SymmetricMemory backend, in HIP and in Triton.

    python3 setup.py build_ext --inplace          # for the HIP kernel only
    torchrun --nnodes=1 --nproc_per_node=<gpus> all2all.py

The receive buffer is an ordinary symm_mem tensor and the kernel reaches peers through
torch's peer pointer array. No mori shmem or cco. The Triton version in
``all2all_triton.py`` does the same work through the same array and needs no build step.

The HIP kernel is additionally built against a flat base + rank*stride window, so the two
addressing forms can be measured against each other -- that comparison is what motivates
the cco-window stage in ROCm/mori#557. The flat form is derived here in the example, not
offered by the allocator's API.
"""

import argparse
import gc
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from mori.allocator import handle_type  # importing registers "MORI"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunk-kib", type=int, default=256, help="bytes sent to each peer")
    p.add_argument("--iters", type=int, default=20, help="timed iterations")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--addressing",
        choices=("ptrs", "flat", "both"),
        default="both",
        help="the N-entry peer pointer array, flat base + peer*stride, or both",
    )
    p.add_argument(
        "--kernel",
        choices=("hip", "triton", "both"),
        default="both",
        help="which implementation to run; Triton is pointer-array only",
    )
    return p.parse_args()


def flat_view(handle):
    """``(base, stride)`` for the flat form, derived from the pointers torch publishes.

    Only valid because this backend maps every rank into one span; a backend handing
    back scattered per-rank pointers would fail the check, and pointing a kernel at
    ``base + rank*stride`` on one of those would silently corrupt memory.
    """
    ptrs = handle.buffer_ptrs
    base = ptrs[0]
    stride = ptrs[1] - ptrs[0] if len(ptrs) > 1 else 0
    if any(p != base + r * stride for r, p in enumerate(ptrs)):
        raise RuntimeError(
            f"window is not evenly strided; peers={[hex(p) for p in ptrs]}"
        )
    return base, stride


VARIANTS = {
    ("hip", "ptrs"): "HIP    peer pointer array",
    ("hip", "flat"): "HIP    base + rank*stride",
    ("triton", "ptrs"): "Triton peer pointer array",
}


def main():
    args = parse_args()
    # Not at module scope: the HIP extension links libc10, so torch must load first, and
    # import sorting would hoist it above torch. Triton is imported the same way just to
    # keep the two symmetrical, and so --kernel hip runs on a box without Triton.
    impls = {}
    if args.kernel in ("hip", "both"):
        import all2all_kernel

        impls["hip"] = all2all_kernel
    if args.kernel in ("triton", "both"):
        import all2all_triton

        impls["triton"] = all2all_triton

    rank_id = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist.init_process_group("gloo")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    group_name = dist.group.WORLD.group_name

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
    # The N-entry device array torch's API always provides.
    peers_dev = hdl.buffer_ptrs_dev
    # The same peers as two scalars, for the comparison only.
    base, stride = flat_view(hdl)

    modes = ("ptrs", "flat") if args.addressing == "both" else (args.addressing,)
    launch = {
        # Same call for HIP and Triton: both take the pointer array torch publishes.
        ("hip", "ptrs"): lambda: impls["hip"].all2all_push_ptrs(
            send, peers_dev, chunk_bytes, rank_id, world_size
        ),
        ("triton", "ptrs"): lambda: impls["triton"].all2all_push_ptrs(
            send, peers_dev, chunk_bytes, rank_id, world_size
        ),
        ("hip", "flat"): lambda: impls["hip"].all2all_push(
            send, base, stride, chunk_bytes, rank_id, world_size
        ),
    }
    # Triton appears under "ptrs" only: the flat form is a HIP-side comparison.
    variants = [(i, m) for (i, m) in VARIANTS if i in impls and m in modes]

    if rank_id == 0:
        print(
            f"world={world_size}  handle={handle_type(local_rank)}  chunk={args.chunk_kib} KiB"
        )
        print(f"peers: {' '.join(hex(p) for p in hdl.buffer_ptrs)}")
        print(f"same as base={base:#x} + rank*{stride / 1024:.0f} KiB")

    def run_once(variant):
        launch[variant]()
        torch.cuda.synchronize()
        # No device-side barrier in the backend yet, so ranks meet on the host.
        dist.barrier()

    errors = 0
    for variant in variants:
        # Zeroing is local, but peers write into this window: everyone must finish
        # clearing before anyone starts pushing, or a late zero wipes an early write.
        recv.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        run_once(variant)
        # Chunk r must hold what rank r sent us: r*1000 + our rank.
        bad = 0
        for r in range(world_size):
            got = recv[r * elems_per_chunk].item()
            want = r * 1000 + rank_id
            if got != want:
                bad += 1
                print(
                    f"[rank {rank_id}] {VARIANTS[variant]} chunk {r}: "
                    f"got {got}, want {want}",
                    flush=True,
                )
        errors += bad
        if rank_id == 0 and bad == 0:
            print(
                f"correctness ({VARIANTS[variant]}): OK ({world_size}x{world_size} chunks)"
            )

    # Only (world_size-1) chunks leave the device; the self chunk stays local.
    remote_bytes = (world_size - 1) * chunk_bytes
    for variant in variants:
        for _ in range(args.warmup):
            run_once(variant)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        dist.barrier()
        start.record()
        for _ in range(args.iters):
            launch[variant]()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / args.iters
        gbps = remote_bytes / (ms / 1e3) / 1e9
        totals = torch.tensor([gbps], dtype=torch.float64)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if rank_id == 0:
            print(
                f"{VARIANTS[variant]:<26}: {ms * 1e3:7.1f} us/iter, "
                f"{totals.item():8.1f} GB/s aggregate"
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
