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
"""All-to-all on a torch symmetric tensor addressed as a CCO LSA window.

Same push as ``all2all_triton.py``, with the peer address computed rather than
loaded: ``cco.Window.lsa_ptr(win, peer, off)`` instead of an index into
``buffer_ptrs_dev``. torch still owns the allocation and its lifetime; CCO owns
the peer addressing. ``rendezvous()`` is the CCO window registration now, so the window
comes straight out of the backend::

    t   = symm_mem.empty(...)              # MORI backend, HIP VMM
    hdl = symm_mem.rendezvous(t, group)    # registers the CCO window
    win = mori.allocator.window_handle(t.data_ptr())
    ...
    addr = cco.Window.lsa_ptr(window, peer, offset)     # inside the kernel

Run it the same way as its siblings::

    torchrun --nnodes=1 --nproc_per_node=8 all2all_lsa.py --chunk-kib 256

Why bother, when the pointer array measures the same (see README): the window is
what carries the things an address array cannot express. ``ccoWindowDevice``
already holds the RDMA MR for peers with no LSA slot, so this is the addressing
mode a scale-out version has to be written against. See ROCm/mori#557.
"""

import argparse
import functools
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

import mori.allocator  # importing registers the "MORI" backend
from mori.ir.triton import cco

BLOCK = 1024
NUM_WARPS = 4


@triton.jit(do_not_specialize=["chunk_elems", "rank_id", "blocks_per_peer"])
def all2all_push_lsa(
    window,
    send_ptr,
    chunk_elems,
    rank_id,
    blocks_per_peer,
    BLOCK: tl.constexpr,
):
    """Push my chunk into every peer's receive slot, addressed through the window."""
    pid = tl.program_id(0)
    peer = pid // blocks_per_peer
    sub = pid % blocks_per_peer

    # peer's window base + my slot within it. No pointer array is dereferenced:
    # lsa_ptr is arithmetic on the flat VA, so the address is known without a load.
    # 64-bit from here on: chunk_elems*world_size overflows i32 past 8 GiB of window.
    chunk = chunk_elems.to(tl.int64)
    dst_addr = cco.Window.lsa_ptr(window, peer, rank_id.to(tl.int64) * chunk * 4)
    dst = dst_addr.to(tl.pointer_type(tl.int32), bitcast=True)
    src = (
        send_ptr.to(tl.pointer_type(tl.int32), bitcast=True) + peer.to(tl.int64) * chunk
    )

    span = blocks_per_peer * BLOCK
    for start in range(sub * BLOCK, chunk_elems, span):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < chunk_elems
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


@functools.cache
def _blocks_per_peer(chunk_elems: int, world_size: int, device) -> int:
    """Two blocks per CU across peers, capped by how much there is to slice.

    Same heuristic as the sibling examples: one block per destination rank would
    idle all but world_size CUs and keep too few writes in flight to cover
    interconnect latency.
    """
    cus = torch.cuda.get_device_properties(device).multi_processor_count
    bpp = max(1, cus * 2 // max(1, world_size))
    return int(min(bpp, max(1, chunk_elems // BLOCK)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk-kib", type=int, default=256, help="payload per peer")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    dist.init_process_group("gloo")
    rank_id = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank_id))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    chunk_bytes = args.chunk_kib * 1024
    elems = chunk_bytes // 4

    group_name = dist.group.WORLD.group_name
    symm_mem.set_backend("MORI")
    recv = symm_mem.empty(world_size * elems, dtype=torch.int32, device=device)
    recv.zero_()
    send = torch.empty(world_size * elems, dtype=torch.int32, device=device)
    for p in range(world_size):
        send[p * elems : (p + 1) * elems] = rank_id * 1000 + p
    torch.cuda.synchronize()

    hdl = symm_mem.rendezvous(recv, group_name)
    window = mori.allocator.window_handle(recv.data_ptr())
    blocks_per_peer = _blocks_per_peer(elems, world_size, device)
    extern_libs = cco.get_extern_libs()

    if rank_id == 0:
        print(
            f"kernel=Triton/LSA  world={world_size}  chunk={args.chunk_kib} KiB  "
            f"blocks_per_peer={blocks_per_peer}"
        )
        print(f"cco window: {window:#x}")

    def push():
        all2all_push_lsa[(world_size * blocks_per_peer,)](
            window,
            send.data_ptr(),
            elems,
            rank_id,
            blocks_per_peer,
            BLOCK=BLOCK,
            num_warps=NUM_WARPS,
            extern_libs=extern_libs,
        )

    # Peers write into this window, so everyone must finish clearing before
    # anyone pushes.
    hdl.barrier(0)
    push()
    torch.cuda.synchronize()
    hdl.barrier(0)

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
        push()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    hdl.barrier(0)
    start.record()
    for _ in range(args.iters):
        push()
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
    hdl.barrier(0)

    dist.destroy_process_group()
    return 0 if failed.item() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
