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
CCO Example 02 — LSA Put
=========================

Each rank launches a GPU kernel that writes its lsaRank into every
peer's receive slot via the flat VA.  After a host barrier the host
reads back each rank's local window and validates that all peer
sentinels arrived.

    mpirun -np 2 python main.py
"""

import ctypes
import os
import sys

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py required.  pip install mpi4py")
    sys.exit(1)

from mori.cco import Communicator, CCODevCommRequirements, GDA_CONNECTION_NONE
from mori.jit.core import compile_genco
from mori.jit.hip_driver import HipModule, _get_hip_lib, _check

PER_RANK_VMM = 4 * 1024 * 1024 * 1024
NSLOTS = 8
SLOT_BYTES = NSLOTS * 8


def _set_device(rank):
    hip = _get_hip_lib()
    num = ctypes.c_int(0)
    hip.hipGetDeviceCount(ctypes.byref(num))
    _check(hip.hipSetDevice(ctypes.c_int(rank % num.value)), "hipSetDevice")


def main():
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    nranks = comm_mpi.Get_size()
    _set_device(rank)

    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = comm_mpi.bcast(uid, root=0)

    hip = _get_hip_lib()

    with Communicator.init(nranks, rank, uid, per_rank_vmm=PER_RANK_VMM) as comm:
        if rank == 0:
            print(f"CommCreate: {nranks} ranks, PER_RANK_VMM={PER_RANK_VMM >> 30} GiB")

        mem = comm.alloc_mem(SLOT_BYTES)
        win = comm.register_window(mem.ptr, mem.size)

        # Zero out local window
        _check(
            hip.hipMemset(ctypes.c_void_p(mem.ptr), 0, ctypes.c_size_t(SLOT_BYTES)),
            "hipMemset",
        )
        _check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize")

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0

        dc = comm.create_dev_comm(reqs)

        if rank == 0:
            print(
                f"DevComm: lsa_size={dc._dev_comm.lsa_size}, lsa_rank={dc._dev_comm.lsa_rank}"
            )

        # Resolve the .hip next to this example (absolute), so it works whether
        # mori is run from the source tree or pip-installed (compile_genco joins
        # source_dir onto the mori source root, which is _jit-sources/ when installed).
        _kernel_dir = os.path.dirname(os.path.abspath(__file__))
        hsaco_path = compile_genco("lsa_put_kernel", source_dir=_kernel_dir)
        module = HipModule(hsaco_path)
        func = module.get_function("lsa_put_kernel")

        # Each rank writes into slot 0 of each peer's window.
        # my_buf_off=0, peer_buf_off=0: everyone writes to byte offset 0.
        my_buf_off = 0
        peer_buf_off = 0

        func.launch((1,), (1,), 0, 0, dc.ptr, win.handle, my_buf_off, peer_buf_off)
        _check(hip.hipDeviceSynchronize(), "hipDeviceSynchronize")

        comm.barrier()

        # Read back local window
        host_buf = (ctypes.c_uint64 * NSLOTS)()
        _check(
            hip.hipMemcpy(
                host_buf,
                ctypes.c_void_p(mem.ptr),
                ctypes.c_size_t(SLOT_BYTES),
                ctypes.c_int(2),
            ),
            "hipMemcpy D2H",
        )

        # Validate: slot 0 should contain the peer's lsaRank
        lsa_rank = dc._dev_comm.lsa_rank
        lsa_size = dc._dev_comm.lsa_size
        errors = 0

        val = host_buf[0]
        if lsa_size > 1:
            # With 2 ranks: rank 0 expects value 1, rank 1 expects value 0
            expected_peer = 1 - lsa_rank if lsa_size == 2 else None
            if expected_peer is not None and val != expected_peer:
                print(f"[rank {rank}] MISMATCH slot[0]={val}, expected {expected_peer}")
                errors += 1
            else:
                print(f"[rank {rank}] OK slot[0]={val} (from peer lsaRank={val})")
        else:
            print(f"[rank {rank}] single rank, no peer writes expected")

        for i in range(1, NSLOTS):
            if host_buf[i] != 0:
                print(f"[rank {rank}] unexpected slot[{i}]={host_buf[i]}")

    all_errors = comm_mpi.allreduce(errors, op=MPI.SUM)
    if rank == 0:
        if all_errors == 0:
            print("SUCCESS")
        else:
            print(f"FAILED ({all_errors} mismatches)")

    return 0 if all_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
