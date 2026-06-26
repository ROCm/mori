#!/usr/bin/env python3
"""CCO Example 03 — FlyDSL GDA put + signal/wait

Device-initiated RDMA from a FlyDSL ``@flyc.kernel`` via the cco device bindings
(``mori.cco.device.flydsl``).

Rank 0 fills its send window, then a FlyDSL kernel issues one GDA put into rank
1's recv window with a completion signal.  Rank 1's kernel waits on the signal,
then the host validates the received payload via hipMemcpy D2H.

The cco device communicator crosses the kernel boundary as a device-resident
handle (``dc.ptr``); windows cross as their (already device) handles.

Bootstrap (two modes):
  * MPI:   ``mpirun -n 2 python main.py``  (uses mpi4py to bcast the UniqueId)
  * env:   set ``CCO_RANK`` / ``CCO_WORLD`` / ``CCO_UID_FILE`` (rank 0 writes the
           UniqueId bytes to the file; other ranks read it).  Lets you bring up
           two nodes without cross-host MPI (share the UniqueId file out-of-band).

GDA connection type via ``MORI_CCO_GDA_CONN`` = ``crossnode`` (default; real
2-node) or ``full`` (required when peers share a node, e.g. single-node 2-rank).
"""

import os
import sys
import time

import flydsl.compiler as flyc
import flydsl.expr as fx

from mori.cco import (
    Communicator, CCODevCommRequirements, UniqueId,
    GDA_CONNECTION_CROSSNODE, GDA_CONNECTION_FULL,
)
import mori.cco.device.flydsl as cco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cco_example_common import set_device, sync, fill, zero, read  # noqa: E402

PER_RANK_VMM = 256 * 1024 * 1024     # 256 MiB flat VMM per rank
NUM_ELEMS = 1024 * 1024 // 8         # 1 MiB of uint64
NBYTES = NUM_ELEMS * 8
DST_RANK = 1
SIGNAL_ID = 0


# FlyDSL kernel-author notes (validated on MI300X):
#  1. Handle args (device pointers) MUST be typed fx.Int64 on the @flyc.jit
#     launcher, or the 64-bit pointer is truncated -> GPU memory fault.
#  2. cco.DevComm/Window/Gda are method-carrying FlyDSL structs: build the handle
#     ONCE and reuse it across scf.if / scf.for (it flows through control flow).
#  3. Don't name a @flyc.jit launcher `launch` — it collides with the `.launch()`
#     method name in co_names and self-recurses in the jit cache-key walk.

@flyc.kernel
def cco_put_kernel(dev_comm: fx.Int64, send_win: fx.Int64, recv_win: fx.Int64, nbytes: fx.Int64):
    """Rank 0: thread 0 puts send_win -> rank 1's recv_win, bumps a signal."""
    tid = fx.thread_idx.x
    gda = cco.DevComm(dev_comm).gda(0)          # build once
    if tid == 0:
        gda.put(DST_RANK, recv_win, 0, send_win, 0, nbytes,
                signal_op=cco.SignalOp.INC, signal_id=SIGNAL_ID, coop=cco.CoopScope.THREAD)
    gda.flush(coop=cco.CoopScope.BLOCK)         # same obj, across the dynamic if


@flyc.kernel
def cco_wait_kernel(dev_comm: fx.Int64):
    """Rank 1: thread 0 waits until the signal reaches 1."""
    tid = fx.thread_idx.x
    gda = cco.DevComm(dev_comm).gda(0)
    if tid == 0:
        gda.wait_signal(SIGNAL_ID, 1, coop=cco.CoopScope.THREAD)


@flyc.jit
def run_put(dev_comm: fx.Int64, send_win: fx.Int64, recv_win: fx.Int64, nbytes: fx.Int64,
            stream=fx.Stream(None)):
    cco_put_kernel(dev_comm, send_win, recv_win, nbytes).launch(
        grid=(1, 1, 1), block=[64, 1, 1], stream=stream)


@flyc.jit
def run_wait(dev_comm: fx.Int64, stream=fx.Stream(None)):
    cco_wait_kernel(dev_comm).launch(grid=(1, 1, 1), block=[64, 1, 1], stream=stream)


def _bootstrap():
    """Return (rank, nranks, uid, barrier_fn). Supports MPI or env/file modes."""
    if "CCO_RANK" in os.environ:
        rank = int(os.environ["CCO_RANK"])
        nranks = int(os.environ["CCO_WORLD"])
        uid_file = os.environ["CCO_UID_FILE"]
        have = os.path.exists(uid_file) and os.path.getsize(uid_file) == 128
        if rank == 0 and not have:
            # single-node mode: rank 0 seeds the uid. (For 2-node, the launcher
            # pre-seeds + rsyncs the file so every rank just reads it.)
            uid = Communicator.get_unique_id()
            tmp = uid_file + ".tmp"
            with open(tmp, "wb") as f:
                f.write(bytes(uid))
            os.replace(tmp, uid_file)
        else:
            while not os.path.exists(uid_file) or os.path.getsize(uid_file) != 128:
                time.sleep(0.05)
            with open(uid_file, "rb") as f:
                uid = UniqueId.from_bytes(f.read())
        return rank, nranks, uid, None  # rely on cco's collective barrier

    from mpi4py import MPI
    mpi = MPI.COMM_WORLD
    rank, nranks = mpi.Get_rank(), mpi.Get_size()
    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = mpi.bcast(uid, root=0)
    return rank, nranks, uid, None


def main() -> int:
    rank, nranks, uid, _ = _bootstrap()
    if nranks != 2:
        if rank == 0:
            print("This example requires exactly 2 ranks.")
        return 1

    set_device(rank)

    conn = (GDA_CONNECTION_FULL if os.environ.get("MORI_CCO_GDA_CONN", "crossnode") == "full"
            else GDA_CONNECTION_CROSSNODE)

    errors = 0
    with Communicator.init(nranks, rank, uid, per_rank_vmm=PER_RANK_VMM) as comm:
        send_win = comm.alloc_window(NBYTES)
        recv_win = comm.alloc_window(NBYTES)

        zero(recv_win.local_ptr, NBYTES)
        if rank == 0:
            fill(send_win.local_ptr, range(1, NUM_ELEMS + 1))

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = conn
        reqs.gda_signal_count = 4
        dc = comm.create_dev_comm(reqs)

        comm.barrier()
        if rank == 0:
            run_put(dc.ptr, send_win.handle, recv_win.handle, NBYTES)
        else:
            run_wait(dc.ptr)
        sync()
        comm.barrier()

        errors = 0
        if rank == DST_RANK:
            host = read(recv_win.local_ptr, NUM_ELEMS)
            for i in (0, 1, NUM_ELEMS // 2, NUM_ELEMS - 1):
                if host[i] != i + 1:
                    print(f"[rank {rank}] MISMATCH [{i}]: got {host[i]}, expected {i + 1}", flush=True)
                    errors += 1
            print(f"[rank {rank}] {'payload verified' if errors == 0 else 'FAILED'} "
                  f"({NBYTES} bytes via GDA put), sample[0,1,-1]="
                  f"{[host[0], host[1], host[NUM_ELEMS-1]]}", flush=True)

    if rank == 0:
        print("SUCCESS" if errors == 0 else "FAILED", flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
