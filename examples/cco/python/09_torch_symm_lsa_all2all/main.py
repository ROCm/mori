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
"""
CCO Example 09 — mori's torch SymmetricMemory backend, addressed as an LSA window
=================================================================================

Example 08 imports a tensor from whichever symmetric-memory backend torch picked.
This one names mori's own (``symm_mem.set_backend("MORI")``) and then does the
whole exchange through CCO, so the two halves of mori meet:

  * ``mori.allocator`` owns the allocation and its lifetime -- it is a torch
    tensor, freed when Python drops it;
  * CCO owns the peer addressing -- ``rendezvous()`` registers the CCO window,
    and ``window_handle()`` hands it back for the kernel to address through.

No ``Communicator`` is built here: the backend keeps one per process group and
creates it on the first rendezvous. The kernel is Triton (``mori.ir.triton.cco``)
and computes each peer address arithmetically with ``Window.lsa_ptr`` rather than
loading it from a pointer array.

All-to-all: rank r writes its chunk into every peer's slot r, then each rank
checks that slot p holds rank p's sentinel.

    mpirun --allow-run-as-root -np 2 python main.py
"""

import os
import socket
import sys

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py required.  pip install mpi4py")
    sys.exit(1)

import torch  # must be imported before anything that pulls in mori.cco
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

import mori.allocator  # importing registers the "MORI" backend
from mori.ir.triton import cco

CHUNK_ELEMS = 4096  # int32 per peer
BLOCK = 512


def _free_port() -> int:
    with socket.socket() as sk:
        sk.bind(("", 0))
        return int(sk.getsockname()[1])


@triton.jit(do_not_specialize=["chunk_elems", "rank_id"])
def a2a_lsa_kernel(window, send_ptr, chunk_elems, rank_id, BLOCK: tl.constexpr):
    """One program per (peer, block): push my chunk into that peer's slot."""
    peer = tl.program_id(0)
    blk = tl.program_id(1)

    chunk = chunk_elems.to(tl.int64)
    # The peer's window base plus the slot it reserves for me. lsa_ptr is
    # arithmetic on the flat VA, so no pointer table is dereferenced.
    dst_addr = cco.Window.lsa_ptr(window, peer, rank_id.to(tl.int64) * chunk * 4)
    dst = dst_addr.to(tl.pointer_type(tl.int32), bitcast=True)
    src = (
        send_ptr.to(tl.pointer_type(tl.int32), bitcast=True) + peer.to(tl.int64) * chunk
    )

    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < chunk_elems
    tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def main():
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    nranks = comm_mpi.Get_size()
    if nranks < 2:
        if rank == 0:
            print("This example needs at least 2 ranks (mpirun -np 2).")
        return 1

    local = rank % torch.cuda.device_count()
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    # torch's process group is the only bootstrap; the backend builds cco's
    # communicator behind rendezvous(). mpirun does not set the variables torch
    # expects, so carry the master address over MPI.
    master = comm_mpi.bcast(
        (
            (socket.gethostbyname(socket.gethostname()), _free_port())
            if rank == 0
            else None
        ),
        root=0,
    )
    os.environ.setdefault("MASTER_ADDR", master[0])
    os.environ.setdefault("MASTER_PORT", str(master[1]))
    dist.init_process_group("gloo", rank=rank, world_size=nranks)
    group_name = dist.group.WORLD.group_name
    symm_mem.set_backend("MORI")

    recv = symm_mem.empty(nranks * CHUNK_ELEMS, dtype=torch.int32, device=device)
    recv.zero_()
    # Ordinary local memory. Chunk p carries a value identifying (me -> p).
    send = torch.empty(nranks * CHUNK_ELEMS, dtype=torch.int32, device=device)
    for p in range(nranks):
        send[p * CHUNK_ELEMS : (p + 1) * CHUNK_ELEMS] = rank * 1000 + p
    torch.cuda.synchronize()

    hdl = symm_mem.rendezvous(recv, group_name)
    window = mori.allocator.window_handle(recv.data_ptr())
    if rank == 0:
        print(
            f"{nranks} ranks, backend={symm_mem.get_backend(device)}, "
            f"{CHUNK_ELEMS * 4} B per peer"
        )
    print(f"[rank {rank}] cco window {window:#x}", flush=True)

    hdl.barrier(0)
    grid = (nranks, (CHUNK_ELEMS + BLOCK - 1) // BLOCK)
    a2a_lsa_kernel[grid](
        window,
        send.data_ptr(),
        CHUNK_ELEMS,
        rank,
        BLOCK=BLOCK,
        extern_libs=cco.get_extern_libs(),
    )
    torch.cuda.synchronize()
    hdl.barrier(0)

    # Slot p must now hold what rank p sent me. First and last element, since the
    # window started zeroed and a short copy would leave the tail at 0.
    errors = 0
    host = recv.cpu().tolist()
    for p in range(nranks):
        want = p * 1000 + rank
        for i in (0, CHUNK_ELEMS - 1):
            got = host[p * CHUNK_ELEMS + i]
            if got != want:
                errors += 1
                print(
                    f"[rank {rank}] slot {p}[{i}]: got {got}, want {want}", flush=True
                )
    if errors == 0:
        print(
            f"[rank {rank}] all {nranks} slots hold the right sender's data "
            f"(via lsa_ptr, no buffer_ptrs)",
            flush=True,
        )

    dist.destroy_process_group()
    all_errors = comm_mpi.allreduce(errors, op=MPI.SUM)
    if rank == 0:
        print("SUCCESS" if all_errors == 0 else f"FAILED ({all_errors} mismatches)")
    return 0 if all_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
