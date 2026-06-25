#!/usr/bin/env python3
"""
CCO Example 01 — Host Barrier
==============================

Simplest end-to-end CCO example: initialise a communicator and run a
collective host barrier.  No GPU kernels — just the full Python lifecycle.

    mpirun -np 4 python main.py

Each rank prints before and after the barrier so you can see all ranks
arriving before any rank proceeds.
"""

import sys

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py required.  pip install mpi4py")
    sys.exit(1)

from mori.cco import Communicator, CCODevCommRequirements, GDA_CONNECTION_NONE


# How much flat VMM to reserve per rank (bytes).
PER_RANK_VMM = 256 * 1024 * 1024   # 256 MiB


def main() -> int:
    comm_mpi = MPI.COMM_WORLD
    rank     = comm_mpi.Get_rank()
    nranks   = comm_mpi.Get_size()

    # ── [CCO] Step 1: Bootstrap ──────────────────────────────────────────────
    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = comm_mpi.bcast(uid, root=0)

    # ── [CCO] Step 2: Create communicator ───────────────────────────────────
    with Communicator.init(nranks, rank, uid, per_rank_vmm=PER_RANK_VMM) as comm:

        if rank == 0:
            print(f"[rank {rank}] CCO communicator ready ({nranks} ranks, "
                  f"per-rank VMM = {PER_RANK_VMM // (1 << 20)} MiB)")

        # ── [CCO] Step 3: Allocate + register symmetric memory ──────────────
        SCRATCH_BYTES = 4096
        mem = comm.alloc_mem(SCRATCH_BYTES)
        win = comm.register_window(mem.ptr, mem.size)

        # ── [CCO] Step 4: Create DevComm ─────────────────────────────────────
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count    = 0
        reqs.gda_counter_count   = 0
        reqs.lsa_barrier_count   = 1

        dc = comm.create_dev_comm(reqs)

        if rank == 0:
            print(f"[rank {rank}] DevComm ready: {dc}")

        # ── [CCO] Step 5: Host barrier ───────────────────────────────────────
        print(f"[rank {rank}] BEFORE barrier", flush=True)
        comm.barrier()
        print(f"[rank {rank}] AFTER  barrier", flush=True)

    # comm.destroy() is called automatically by the context manager.

    if rank == 0:
        print("SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
