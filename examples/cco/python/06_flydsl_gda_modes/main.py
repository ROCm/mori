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
"""CCO Example 06 — FlyDSL GDA template-coverage test

Exercises the monomorphized ``put`` template matrix in libmori_cco_device.bc:

    (thread_mode, coop) ∈ {indep×thread, indep×warp, indep×block, aggr×thread}
                        ×  signal op ∈ {inc, add}

Each (mode, signal) pair selects a *distinct* fully-specialized
``ccoGda<P>::put<..., ThreadMode, Coop>`` / RemoteAction symbol — all three axes
are compile-time constants, so the kernel emits one direct call with no runtime
dispatch. One distinct signal id per combo (so no reset needed): rank 1 waits on
it (proving the signal/op arrived) and the host validates the delivered payload.

Then, for each doorbell mode (DEFERRED, CALLER_ORDERED) and team (WORLD, GDA),
the payload goes as DEFER_CHUNKS puts through a handle of that mode, which ring
nothing, released by that handle's flush_async + wait, followed by a plain
signal -- the batch-then-release pattern of a fused MoE combine.

    mpirun -n 2 python main.py     (one node, FULL connection)
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
from flydsl.expr.typing import Int32, Int64

from mori.cco import Communicator, CCODevCommRequirements, GDA_CONNECTION_FULL
import mori.cco.device.flydsl as cco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cco_example_common import set_device, sync, fill, zero, read  # noqa: E402

PER_RANK_VMM = 256 * 1024 * 1024
NUM_ELEMS = 4096  # uint64 payload per put
NBYTES = NUM_ELEMS * 8
DST_RANK = 1
THREADS = 64  # one wavefront (so aggregate coalesces its lanes)
ADD_VAL = 7  # signal_val for the SignalAdd op

# (name, coop, thread_mode) — aggregate is only valid with thread coop.
MODES = [
    ("indep_thread", cco.CoopScope.THREAD, cco.ThreadMode.INDEPENDENT),
    ("indep_warp", cco.CoopScope.WARP, cco.ThreadMode.INDEPENDENT),
    ("indep_block", cco.CoopScope.BLOCK, cco.ThreadMode.INDEPENDENT),
    ("aggr_thread", cco.CoopScope.THREAD, cco.ThreadMode.AGGREGATE),
]
SIGNALS = [("inc", cco.SignalOp.INC, 1), ("add", cco.SignalOp.ADD, ADD_VAL)]
DEFER_CHUNKS = 8
TEAMS = [("world", cco.Team.WORLD), ("gda", cco.Team.GDA)]
DOORBELLS = [
    ("deferred", cco.GdaDoorbell.DEFERRED),
    ("caller_ordered", cco.GdaDoorbell.CALLER_ORDERED),
]


def _make_put_kernel(coop, thread_mode, signal_op):
    """One kernel per (coop, thread_mode, signal_op) — all baked as constants."""
    # Gate to a single thread only for independent+thread (one logical put).
    # warp/block coops and aggregate need all lanes to enter put together.
    gate_single = (
        coop == cco.CoopScope.THREAD and thread_mode == cco.ThreadMode.INDEPENDENT
    )

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def put_kernel(
        dev_comm: Int64,
        send_win: Int64,
        recv_win: Int64,
        nbytes: Int64,
        sig_id: Int32,
        sig_val: Int64,
    ):
        gda = cco.DevComm(dev_comm).gda(0)
        if gate_single:
            if fx.thread_idx.x == 0:
                gda.put(
                    DST_RANK,
                    recv_win,
                    0,
                    send_win,
                    0,
                    nbytes,
                    signal_op=signal_op,
                    signal_id=sig_id,
                    signal_val=sig_val,
                    coop=coop,
                    thread_mode=thread_mode,
                )
        else:
            gda.put(
                DST_RANK,
                recv_win,
                0,
                send_win,
                0,
                nbytes,
                signal_op=signal_op,
                signal_id=sig_id,
                signal_val=sig_val,
                coop=coop,
                thread_mode=thread_mode,
            )
        gda.flush(coop=cco.CoopScope.BLOCK)

    @flyc.jit
    def run(
        dev_comm: Int64,
        send_win: Int64,
        recv_win: Int64,
        nbytes: Int64,
        sig_id: Int32,
        sig_val: Int64,
        stream=fx.Stream(None),
    ):
        put_kernel(dev_comm, send_win, recv_win, nbytes, sig_id, sig_val).launch(
            grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
        )

    return run


def _make_defer_kernel(team, doorbell):
    """DEFER_CHUNKS block puts through a `doorbell` handle, its flush_async +
    wait, then a signal through an IMMEDIATE handle."""
    chunk = NBYTES // DEFER_CHUNKS

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def defer_kernel(dev_comm: Int64, send_win: Int64, recv_win: Int64, sig_id: Int32):
        dc = cco.DevComm(dev_comm)
        gda = dc.gda(0)
        batch = dc.gda(0, doorbell=doorbell)
        # The QP context may be chosen at run time per call, as a kernel that
        # spreads chunks over contexts would; here every call uses this block's.
        qp = fx.Int32(fx.block_idx.x)
        for c in range(DEFER_CHUNKS):  # unrolled at trace time
            batch.put(
                DST_RANK,
                recv_win,
                c * chunk,
                send_win,
                c * chunk,
                chunk,
                coop=cco.CoopScope.BLOCK,
                team=team,
                ctx=qp,
            )
        # The same thread posts and flushes, so CALLER_ORDERED's ordering holds.
        request = batch.flush_async(
            DST_RANK, coop=cco.CoopScope.BLOCK, team=team, ctx=qp
        )
        batch.wait(request, coop=cco.CoopScope.BLOCK)
        # Nothing is pending on the QP now, so any handle may post to it.
        gda.signal(
            DST_RANK,
            signal_op=cco.SignalOp.INC,
            signal_id=sig_id,
            coop=cco.CoopScope.BLOCK,
            team=team,
            ctx=qp,
        )
        gda.flush_peer(DST_RANK, coop=cco.CoopScope.BLOCK, team=team, ctx=qp)

    @flyc.jit
    def run(
        dev_comm: Int64,
        send_win: Int64,
        recv_win: Int64,
        sig_id: Int32,
        stream=fx.Stream(None),
    ):
        defer_kernel(dev_comm, send_win, recv_win, sig_id).launch(
            grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
        )

    return run


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def wait_kernel(dev_comm: Int64, sig_id: Int32, least: Int64):
    if fx.thread_idx.x == 0:
        cco.DevComm(dev_comm).gda(0).wait_signal(
            sig_id, least, coop=cco.CoopScope.THREAD
        )


@flyc.jit
def run_wait(dev_comm: Int64, sig_id: Int32, least: Int64, stream=fx.Stream(None)):
    wait_kernel(dev_comm, sig_id, least).launch(
        grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
    )


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

    # One specialized kernel per (mode, signal) combo.
    put_runners = {
        (mname, sname): _make_put_kernel(coop, tm, sig_op)
        for mname, coop, tm in MODES
        for sname, sig_op, _ in SIGNALS
    }
    defer_runners = {
        (dname, tname): _make_defer_kernel(team, doorbell)
        for dname, doorbell in DOORBELLS
        for tname, team in TEAMS
    }
    failures = 0

    with Communicator.init(nranks, rank, uid, per_rank_vmm=PER_RANK_VMM) as comm:
        send_mem = comm.alloc_mem(NBYTES)
        send_win = comm.register_window(send_mem.ptr, send_mem.size)
        recv_mem = comm.alloc_mem(NBYTES)
        recv_win = comm.register_window(recv_mem.ptr, recv_mem.size)
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_FULL
        reqs.gda_signal_count = 16
        dc = comm.create_dev_comm(reqs)

        sig_id = 0
        for mname, coop, tm in MODES:
            for sname, sig_op, least in SIGNALS:
                pattern = (sig_id + 1) * 1000  # distinct payload per combo
                if rank == 0:
                    fill(send_win.local_ptr, [pattern] * NUM_ELEMS)
                else:
                    zero(recv_win.local_ptr, NBYTES)
                comm.barrier()

                if rank == 0:
                    put_runners[(mname, sname)](
                        dc.ptr,
                        send_win.handle,
                        recv_win.handle,
                        NBYTES,
                        sig_id,
                        ADD_VAL,
                    )
                else:
                    run_wait(dc.ptr, sig_id, least)
                sync()
                comm.barrier()

                ok = True
                if rank == DST_RANK:
                    host = read(recv_win.local_ptr, NUM_ELEMS)
                    ok = all(
                        host[i] == pattern for i in (0, NUM_ELEMS // 2, NUM_ELEMS - 1)
                    )
                    print(
                        f"[combo mode={mname:12} signal={sname}] "
                        f"{'PASS' if ok else 'FAIL'} (payload {pattern})",
                        flush=True,
                    )
                    failures += 0 if ok else 1
                sig_id += 1

        for dname, tname in defer_runners:
            pattern = (sig_id + 1) * 1000
            if rank == 0:
                fill(send_win.local_ptr, [pattern] * NUM_ELEMS)
            else:
                zero(recv_win.local_ptr, NBYTES)
            comm.barrier()
            if rank == 0:
                defer_runners[(dname, tname)](
                    dc.ptr, send_win.handle, recv_win.handle, sig_id
                )
            else:
                run_wait(dc.ptr, sig_id, 1)
            sync()
            comm.barrier()
            if rank == DST_RANK:
                host = read(recv_win.local_ptr, NUM_ELEMS)
                # One element per chunk: every deferred put must have landed.
                step = NUM_ELEMS // DEFER_CHUNKS
                ok = all(host[c * step] == pattern for c in range(DEFER_CHUNKS))
                ok = ok and host[NUM_ELEMS - 1] == pattern
                print(
                    f"[{dname:14} team={tname:5}] {'PASS' if ok else 'FAIL'} "
                    f"({DEFER_CHUNKS} puts, flush_async + wait)",
                    flush=True,
                )
                failures += 0 if ok else 1
            sig_id += 1

    total = mpi.allreduce(failures, op=MPI.SUM)
    if rank == 0:
        n = len(MODES) * len(SIGNALS) + len(DOORBELLS) * len(TEAMS)
        print(
            (
                f"SUCCESS ({n}/{n} template combos)"
                if total == 0
                else f"FAILED ({total} combos)"
            ),
            flush=True,
        )
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
