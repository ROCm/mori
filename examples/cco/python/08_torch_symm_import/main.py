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
CCO Example 08 — Import a torch.symm_mem tensor into the flat LSA space
======================================================================

A tensor allocated by ``torch.distributed._symmetric_memory`` is HIP-VMM
backed, so CCO can alias it into its symmetric flat-VA space with NO copy via
``Communicator.register_external_window()`` (ccoWindowRegister overload C:
retain the tensor's physical handle, map it into this rank's flat-VA slot, then
exchange with peers). The imported buffer then behaves like any ccoMemAlloc
window. This example accesses it two ways:

  Phase 1 (host): each rank reads every peer's tensor via hipMemcpy from the
    peer's flat VA (local_ptr + (peer_lsa - my_lsa) * per_rank_size).
  Phase 2 (kernel): rank 1 launches a FlyDSL kernel that reads rank 0's tensor
    directly through the LSA device API (cco.Window(win.handle).lsa_ptr(peer,
    off) + buffer_load) and copies it into its own slot.

    mpirun -np 2 python main.py
"""

import ctypes
import sys

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py required.  pip install mpi4py")
    sys.exit(1)

import torch
import torch.distributed._symmetric_memory as symm_mem

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops
from flydsl.expr.typing import Int32, Int64, T
from flydsl._mlir.dialects import rocdl

from mori.cco import Communicator, CCODevCommRequirements, GDA_CONNECTION_NONE
from mori.jit.hip_driver import _check, _get_hip_lib
import mori.cco.device.flydsl as cco

PER_RANK_VMM = 1 * 1024 * 1024 * 1024
N = 1024  # int32 elements
NBYTES = N * 4
NUM_PACKS = NBYTES // 16  # 16 B (vector<4xi32>) per pack
THREADS = 256
SRC_RANK = 0  # rank whose torch.symm_mem tensor is read over LSA in phase 2
DST_RANK = 1  # rank that runs the kernel in phase 2


def sentinel(rank):
    return [rank * 100000 + i for i in range(N)]


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def lsa_gather_kernel(dev_comm: Int64, win_handle: Int64, src_rank: Int32):
    """Rank DST reads SRC_RANK's imported torch.symm_mem tensor directly through
    its peer pointer (flat VA) and copies it into its own slot."""
    tid = fx.thread_idx.x
    dc = cco.DevComm(dev_comm)
    win = cco.Window(win_handle)
    src = win.lsa_ptr(src_rank, 0)  # peer's torch tensor (peer VA) — load directly
    dst = win.lsa_ptr(dc.lsa_rank, 0)  # my own torch tensor (local VA)
    src_rsrc = buffer_ops.create_buffer_resource_from_addr(src)
    dst_rsrc = buffer_ops.create_buffer_resource_from_addr(dst)
    for pk in range(tid, NUM_PACKS, THREADS):
        off = pk * 4  # i32-element offset of this 16 B pack
        v = buffer_ops.buffer_load(src_rsrc, off, vec_width=4, dtype=T.i32)
        buffer_ops.buffer_store(v, dst_rsrc, off)
    rocdl.s_waitcnt(0)


@flyc.jit
def run_gather(
    dev_comm: Int64, win_handle: Int64, src_rank: Int32, stream=fx.Stream(None)
):
    lsa_gather_kernel(dev_comm, win_handle, src_rank).launch(
        grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
    )


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
    dev = torch.device("cuda", local)
    hip = _get_hip_lib()

    # torch.symm_mem tensor (HIP VMM backed) with a per-rank sentinel pattern.
    t = symm_mem.empty(N, dtype=torch.int32, device=dev)
    t[:] = torch.tensor(sentinel(rank), dtype=torch.int32, device=dev)
    torch.cuda.synchronize()

    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = comm_mpi.bcast(uid, root=0)

    errors = 0
    with Communicator.init(nranks, rank, uid, per_rank_vmm=PER_RANK_VMM) as comm:
        if rank == 0:
            print(f"CommCreate: {nranks} ranks, PER_RANK_VMM={PER_RANK_VMM >> 30} GiB")

        # ── Import the torch.symm_mem buffer into the flat LSA space (no copy) ──
        win = comm.register_external_window(t.data_ptr(), NBYTES)
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        dc = comm.create_dev_comm(reqs)
        print(
            f"[rank {rank}] torch data_ptr={t.data_ptr():#x} -> "
            f"cco flat local_ptr={win.local_ptr:#x}"
        )
        comm.barrier()

        # ── Phase 1 (host): read every peer's tensor via hipMemcpy over flat VA ──
        my_lsa = dc.lsa_rank
        per_rank_size = dc.per_rank_size
        host_buf = (ctypes.c_int * 8)()
        for pe in range(nranks):
            peer_ptr = win.local_ptr + (pe - my_lsa) * per_rank_size
            _check(
                hip.hipMemcpy(
                    host_buf,
                    ctypes.c_void_p(peer_ptr),
                    ctypes.c_size_t(32),
                    ctypes.c_int(2),
                ),
                "hipMemcpy D2H",
            )
            got = list(host_buf)
            exp = sentinel(pe)[:8]
            tag = "local" if pe == my_lsa else "peer "
            if got == exp:
                print(f"[rank {rank}] [host]   {tag} pe={pe}: {got} OK")
            else:
                print(f"[rank {rank}] [host]   {tag} pe={pe}: {got} MISMATCH exp={exp}")
                errors += 1
        comm.barrier()

        # ── Phase 2 (kernel): rank DST gathers SRC_RANK's tensor via lsa_ptr ──
        if rank == DST_RANK:
            run_gather(dc.ptr, win.handle, SRC_RANK)
            torch.cuda.synchronize()
        comm.barrier()

        if rank == DST_RANK:
            # DST's own torch tensor must now hold SRC_RANK's sentinel.
            host = t.cpu().tolist()
            exp = sentinel(SRC_RANK)
            for i in (0, 1, N // 2, N - 1):
                if host[i] != exp[i]:
                    print(
                        f"[rank {rank}] [kernel] MISMATCH [{i}]: got {host[i]}, exp {exp[i]}"
                    )
                    errors += 1
            if errors == 0:
                print(
                    f"[rank {rank}] [kernel] gathered rank {SRC_RANK}'s torch.symm_mem "
                    f"via lsa_ptr; sample[0,1,-1]={[host[0], host[1], host[N - 1]]} OK"
                )
        comm.barrier()

    all_errors = comm_mpi.allreduce(errors, op=MPI.SUM)
    if rank == 0:
        print("SUCCESS" if all_errors == 0 else f"FAILED ({all_errors} mismatches)")
    return 0 if all_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
