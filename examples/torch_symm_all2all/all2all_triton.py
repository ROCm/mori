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
"""One-shot all-to-all over mori's torch SymmetricMemory backend, in Triton.

    torchrun --nnodes=1 --nproc_per_node=<gpus> all2all_triton.py --chunk-kib 256

Nothing here is mori-specific: it is the pointer-array idiom torch's own symmetric-memory
Triton kernels use. ``hdl.buffer_ptrs_dev`` goes in as a plain integer and becomes a
pointer inside the kernel, so no C++ is involved at all.

``all2all_hip.py`` is the same example in HIP.
"""

import argparse
import functools
import gc
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from mori.allocator import handle_type  # importing registers the "MORI" backend

# 256 lanes on a wave64 part, 4 int32 each -- the dwordx4 store the HIP kernel does.
BLOCK = 1024
NUM_WARPS = 4


# do_not_specialize: Triton folds an int argument that happens to equal 1 into a constexpr,
# so without it rank 1 alone gets a plain Python int where the kernel casts to tl.int64.
@triton.jit(do_not_specialize=["chunk_elems", "rank_id", "blocks_per_peer"])
def _all2all_push_ptrs(
    send_ptr,  # *i32, local, world_size*chunk_elems
    peer_ptrs,  # i64 address of the world_size-entry peer pointer array
    chunk_elems,
    rank_id,
    blocks_per_peer,
    BLOCK: tl.constexpr,
):
    # Grid flattens (peer, slice of chunk). One block per peer would idle all but
    # world_size CUs and keep too few writes in flight to cover interconnect latency.
    pid = tl.program_id(0)
    peer = pid // blocks_per_peer
    sub = pid % blocks_per_peer

    peers = peer_ptrs.to(tl.pointer_type(tl.uint64))
    dst = tl.load(peers + peer).to(tl.pointer_type(tl.int32))

    # 64-bit from here on: chunk_elems*world_size overflows i32 past 8 GiB of window.
    chunk = chunk_elems.to(tl.int64)
    dst += rank_id.to(tl.int64) * chunk  # the slot that peer reserves for us
    src = send_ptr + peer.to(tl.int64) * chunk  # what we owe that peer

    span = blocks_per_peer * BLOCK
    for start in range(sub * BLOCK, chunk_elems, span):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < chunk_elems
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


@functools.cache
def _blocks_per_peer(chunk_elems: int, world_size: int, device) -> int:
    """Two blocks per CU across peers, capped by how much there is to slice.

    Cached because it is on the launch path, where microseconds are the whole story:
    ``get_device_properties`` costs ~0.9 us against a Triton launch of ~9-13 us.
    """
    cus = torch.cuda.get_device_properties(device).multi_processor_count
    bpp = max(1, cus * 2 // max(1, world_size))
    return min(bpp, max(1, chunk_elems // BLOCK))


def all2all_push_ptrs(send, peers_dev, chunk_bytes, rank_id, world_size):
    """``peers_dev`` is hdl.buffer_ptrs_dev. Push only; the caller barriers afterwards."""
    if not send.is_contiguous():
        raise ValueError("send must be contiguous")
    if send.dtype != torch.int32:
        raise ValueError(f"send must be int32, got {send.dtype}")
    if chunk_bytes % 4:
        raise ValueError(f"chunk_bytes must be a multiple of 4, got {chunk_bytes}")
    chunk_elems = chunk_bytes // 4
    if send.numel() != world_size * chunk_elems:
        raise ValueError(
            f"send holds {send.numel()} elements, expected {world_size * chunk_elems}"
        )

    bpp = _blocks_per_peer(chunk_elems, world_size, send.device)
    _all2all_push_ptrs[(world_size * bpp,)](
        send,
        peers_dev,
        chunk_elems,
        rank_id,
        bpp,
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--chunk-kib", type=int, default=256, help="bytes pushed to each peer"
    )
    p.add_argument("--iters", type=int, default=20, help="timed iterations")
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    # Defaults so a bare `python3 all2all_*.py` is a 1-rank smoke test; torchrun sets them.
    for key, val in (
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29500"),
    ):
        os.environ.setdefault(key, val)
    rank_id = int(os.environ["RANK"])
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

    # Receive window: one chunk per source rank, writable by peers.
    recv = symm_mem.empty(world_size * elems, dtype=torch.int32, device=device)
    recv.zero_()
    # Ordinary local memory. Chunk p carries a value identifying (me -> p).
    send = torch.empty(world_size * elems, dtype=torch.int32, device=device)
    for p in range(world_size):
        send[p * elems : (p + 1) * elems] = rank_id * 1000 + p
    torch.cuda.synchronize()

    hdl = symm_mem.rendezvous(recv, group_name)
    peers_dev = hdl.buffer_ptrs_dev

    if rank_id == 0:
        print(
            f"kernel=Triton  world={world_size}  handle={handle_type(local_rank)}  "
            f"chunk={args.chunk_kib} KiB"
        )
        print(f"peers: {' '.join(hex(p) for p in hdl.buffer_ptrs)}")

    def run_once():
        all2all_push_ptrs(send, peers_dev, chunk_bytes, rank_id, world_size)
        torch.cuda.synchronize()
        dist.barrier()  # no device-side barrier in the backend yet

    # Peers write into this window, so everyone must finish clearing before anyone pushes.
    dist.barrier()
    run_once()

    errors = 0
    for r in range(world_size):
        got, want = recv[r * elems].item(), r * 1000 + rank_id
        if got != want:
            errors += 1
            print(f"[rank {rank_id}] chunk {r}: got {got}, want {want}", flush=True)
    failed = torch.tensor([errors], dtype=torch.int64)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM)
    if rank_id == 0 and failed.item() == 0:
        print(f"correctness: OK ({world_size}x{world_size} chunks)")

    for _ in range(args.warmup):
        run_once()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    dist.barrier()
    start.record()
    for _ in range(args.iters):
        all2all_push_ptrs(send, peers_dev, chunk_bytes, rank_id, world_size)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.iters

    # Only (world_size-1) chunks leave the device; the self chunk stays local.
    gbps = torch.tensor(
        [(world_size - 1) * chunk_bytes / (ms / 1e3) / 1e9], dtype=torch.float64
    )
    dist.all_reduce(gbps, op=dist.ReduceOp.SUM)
    if rank_id == 0:
        print(f"{ms * 1e3:7.1f} us/iter, {gbps.item():8.1f} GB/s aggregate")
        print("SUCCESS" if failed.item() == 0 else "FAILED")

    dist.barrier()
    del hdl, recv
    gc.collect()
    dist.destroy_process_group()
    return 0 if failed.item() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
