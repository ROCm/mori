#!/usr/bin/env python3
"""
CCO Example 02 — LSA Window Allocation & Validation
=====================================================

Validates the CCO symmetric window allocation path by writing data to
each rank's local window slot from the host, then reading it back via
``hipMemcpy`` to confirm the mapping is correct.

No GPU kernel is launched — this exercises the window allocation and
flat VA mapping without requiring a device communicator.

    mpirun -np 4 python main.py

Flow
----
1.  Bootstrap  : rank 0 generates UniqueId, MPI broadcasts it.
2.  CommCreate : CCO sets up the LSA flat VA.
3.  alloc_window: CCO allocates a symmetric window (overload A).
4.  Host write : each rank writes its rank index into its local slot.
5.  hipMemcpy  : D2H copy to a host buffer to verify the mapping.
6.  barrier    : host-side fence.
7.  Validate   : compare host buffer against expected values.
8.  Teardown   : automatic via Communicator context manager.
"""

import ctypes
import sys

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py required.  pip install mpi4py")
    sys.exit(1)

from mori.cco import Communicator
from mori.jit.hip_driver import _get_hip_lib, _check


# ── Config ───────────────────────────────────────────────────────────────────

NSLOTS     = 4               # uint64 slots per rank
SLOT_BYTES = NSLOTS * 8      # 32 bytes per rank
PER_RANK_VMM = 4 * 1024 * 1024 * 1024   # 4 GiB


# ── HIP helpers ──────────────────────────────────────────────────────────────

def _set_device(rank: int) -> None:
    hip = _get_hip_lib()
    num = ctypes.c_int(0)
    hip.hipGetDeviceCount(ctypes.byref(num))
    _check(hip.hipSetDevice(ctypes.c_int(rank % num.value)), "hipSetDevice")


def _memcpy_d2h(dst: ctypes.Array, src_dev: int, size: int) -> None:
    hipMemcpyDeviceToHost = 2
    _check(_get_hip_lib().hipMemcpy(
        dst, ctypes.c_void_p(src_dev),
        ctypes.c_size_t(size), ctypes.c_int(hipMemcpyDeviceToHost),
    ), "hipMemcpy D2H")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    comm_mpi = MPI.COMM_WORLD
    rank     = comm_mpi.Get_rank()
    nranks   = comm_mpi.Get_size()

    _set_device(rank)

    # ── [CCO] Step 1: Bootstrap ──────────────────────────────────────────────
    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = comm_mpi.bcast(uid, root=0)

    with Communicator.init(nranks, rank, uid, per_rank_vmm=PER_RANK_VMM) as comm:

        if rank == 0:
            print(f"CommCreate: {nranks} ranks, PER_RANK_VMM={PER_RANK_VMM >> 30} GiB")

        # ── [CCO] Step 2: Allocate window ────────────────────────────────────
        win = comm.alloc_window(SLOT_BYTES)

        if rank == 0:
            print(f"Window: handle={win.handle:#x}, local_ptr={win.local_ptr:#x}, "
                  f"size={win.size}")

        # ── Step 3: Host write via local_ptr ─────────────────────────────────
        buf = (ctypes.c_uint64 * NSLOTS).from_address(win.local_ptr)
        for i in range(NSLOTS):
            buf[i] = rank * 1000 + i

        # ── Step 4: hipMemcpy D2H to verify ──────────────────────────────────
        host_buf = (ctypes.c_uint64 * NSLOTS)()
        _memcpy_d2h(host_buf, win.local_ptr, SLOT_BYTES)

        # ── Step 5: Barrier ──────────────────────────────────────────────────
        comm.barrier()

        # ── Step 6: Validate ─────────────────────────────────────────────────
        errors = 0
        for i in range(NSLOTS):
            expected = rank * 1000 + i
            got = host_buf[i]
            if got != expected:
                print(f"[rank {rank}] MISMATCH slot {i}: "
                      f"got {got}, expected {expected}")
                errors += 1
            else:
                print(f"[rank {rank}] OK slot[{i}]={got}")

    all_errors = comm_mpi.allreduce(errors, op=MPI.SUM)
    if rank == 0:
        if all_errors == 0:
            print("SUCCESS")
        else:
            print(f"FAILED ({all_errors} mismatches)")

    return 0 if all_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
