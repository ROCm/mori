#!/usr/bin/env python3
"""CCO Example 04 — FlyDSL LSA put (direct peer-pointer load/store)

The LSA model: cco only hands you the peer's *load/store-accessible* pointer
(``win.lsa_ptr(peer, off)`` over the flat VA); the data movement is done
*directly in the FlyDSL kernel* (buffer_load/store) — cco does NOT do the copy
for you. (Contrast with GDA, where put/get are opaque RDMA ops; see example 03.)

    mpirun -n 2 python main.py     (both ranks on one node)

Rank 0's kernel reads its own window slot and stores it into rank 1's slot
through rank 1's peer VA; the host then validates the payload.
"""

import os
import sys

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py required. pip install mpi4py")
    sys.exit(1)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops
from flydsl.expr.typing import Int64, T
from flydsl._mlir.dialects import rocdl

from mori.cco import Communicator, CCODevCommRequirements, GDA_CONNECTION_NONE
import mori.cco.device.flydsl as cco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cco_example_common import set_device, sync, fill, zero, read  # noqa: E402

PER_RANK_VMM = 256 * 1024 * 1024
NUM_ELEMS = 1024 * 1024 // 8     # 1 MiB of uint64
NBYTES = NUM_ELEMS * 8
NUM_PACKS = NBYTES // 16         # 16 B (vector<4xi32>) per pack
THREADS = 256
DST_RANK = 1                     # rank 1's LSA rank
_CM_UNCACHED = 3                 # SC0|SC1: store bypasses L1+L2 -> reaches peer HBM


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def lsa_put_kernel(dev_comm: Int64, win_handle: Int64):
    """Rank 0: block-strided P2P copy of my window slot into rank 1's slot,
    operating directly on the peer pointer returned by cco."""
    tid = fx.thread_idx.x
    dc = cco.DevComm(dev_comm)
    win = cco.Window(win_handle)
    src = win.lsa_ptr(dc.lsa_rank, 0)     # my own slot (local VA)
    dst = win.lsa_ptr(DST_RANK, 0)        # rank 1's slot (peer VA) — load/store directly
    src_rsrc = buffer_ops.create_buffer_resource_from_addr(src)
    dst_rsrc = buffer_ops.create_buffer_resource_from_addr(dst)
    for pk in range(tid, NUM_PACKS, THREADS):
        off = pk * 4                      # i32-element offset of this 16 B pack
        v = buffer_ops.buffer_load(src_rsrc, off, vec_width=4, dtype=T.i32)
        buffer_ops.buffer_store(v, dst_rsrc, off, cache_modifier=_CM_UNCACHED)
    rocdl.s_waitcnt(0)


@flyc.jit
def run_lsa(dev_comm: Int64, win_handle: Int64, stream=fx.Stream(None)):
    lsa_put_kernel(dev_comm, win_handle).launch(grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream)


def main() -> int:
    mpi = MPI.COMM_WORLD
    rank, nranks = mpi.Get_rank(), mpi.Get_size()
    if nranks != 2:
        if rank == 0:
            print("This example requires exactly 2 ranks on one node (mpirun -n 2).")
        return 1
    set_device(rank)
    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = mpi.bcast(uid, root=0)

    errors = 0
    with Communicator.init(nranks, rank, uid, per_rank_vmm=PER_RANK_VMM) as comm:
        win = comm.alloc_window(NBYTES)
        zero(win.local_ptr, NBYTES)
        if rank == 0:
            fill(win.local_ptr, range(1, NUM_ELEMS + 1))

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        dc = comm.create_dev_comm(reqs)

        comm.barrier()
        if rank == 0:
            run_lsa(dc.ptr, win.handle)
            sync()
        comm.barrier()

        if rank == DST_RANK:
            host = read(win.local_ptr, NUM_ELEMS)
            for i in (0, 1, NUM_ELEMS // 2, NUM_ELEMS - 1):
                if host[i] != i + 1:
                    print(f"[rank {rank}] MISMATCH [{i}]: got {host[i]}, expected {i + 1}", flush=True)
                    errors += 1
            print(f"[rank {rank}] {'payload verified' if errors == 0 else 'FAILED'} "
                  f"({NBYTES} bytes via direct LSA peer-pointer store), "
                  f"sample[0,1,-1]={[host[0], host[1], host[NUM_ELEMS-1]]}", flush=True)

    all_err = mpi.allreduce(errors, op=MPI.SUM)
    if rank == 0:
        print("SUCCESS" if all_err == 0 else "FAILED", flush=True)
    return 0 if all_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
