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
"""CCO Example 13 — FlyDSL ccoProxy (host-proxy RDMA)

The FlyDSL ``Proxy`` handle on a proxy-only DevComm: the GPU pushes put /
signal requests into D2H queues and a host thread posts them to the NIC, so no
GDA QP is needed. Both ranks run every kernel and send to each other:

  * per-block SPSC queue, BLOCK coop: put + ``SignalOp.SET`` (no NIC atomic)
    for three rounds, then put + ``SignalOp.INC`` in a second launch that
    continues the queues' saved state; each round flushes and waits for the
    peer's signal
  * the same with WARP coop on one-wavefront blocks
  * every block on one shared MPSC queue, put + INC
  * ``put_value`` (plain, then with a SET signal)
  * a bare ADD signal, ``flush_timeout`` and ``failed()``
  * an INC nobody waits for, read and reset through both pools

The host checks every byte the peer put, and the values the kernels stored.

    mpirun -np 2 python3 main.py     (wave64 gfx942 / gfx950, one node)
"""

import ctypes
import os
import sys

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py required. pip install mpi4py")
    sys.exit(1)

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int32, Int64

from mori.cco import Communicator, CCODevCommRequirements, GDA_CONNECTION_FULL
import mori.cco.device.flydsl as cco
from mori.cco.device.flydsl import buffer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cco_example_common import set_device, sync, fill, zero, read  # noqa: E402

WS = 2
PER_RANK_VMM = 256 * 1024 * 1024
PROGRAMS = 4  # blocks per launch, one SPSC queue each
MPSC_QUEUES = 1
BLOCK_THREADS = 256  # 4 wavefronts: BLOCK coop issues from wavefront 0
WARP_THREADS = 64  # 1 wavefront: WARP coop
NUM_ELEMS = 2048  # uint64 payload
NBYTES = NUM_ELEMS * 8
CHUNK = NBYTES // PROGRAMS  # one per block
ITERS = 3
SIGNALS = 8 * PROGRAMS
TIMEOUT_CYCLES = 1 << 36
ADD_VAL = 3
VALUE_TAG = 0xAB00

# Window layout (bytes): [ send | recv | 16 x u64 values | results ]
SEND_OFF = 0
RECV_OFF = NBYTES
VALUE_OFF = 2 * NBYTES
OUT_OFF = VALUE_OFF + 16 * 8
WIN_BYTES = OUT_OFF + 256

SPSC, MPSC = cco.ProxyPool.SPSC, cco.ProxyPool.MPSC
BLOCK, WARP = cco.CoopScope.BLOCK, cco.CoopScope.WARP


def _out_rsrc(win):
    """Buffer resource over this rank's result slots."""
    addr = cco._bindings.cco_local_ptr(win, OUT_OFF)
    return buffer.create_buffer_resource_from_addr(fx.Int64(addr))


def _queue(pool, pid):
    """SPSC: the block's own queue; MPSC: the one shared queue."""
    return pid if pool == SPSC else 0


def _make_put_kernel(signal_op, coop, pool, threads):
    """Block p puts chunk p, then flushes and waits for the peer's chunk p
    signal, ITERS times. SET writes 1, 2, ... so each wait is for 1 more."""

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def put_kernel(dev_comm: Int64, win: Int64, peer: Int32, sig_base: Int32):
        pid = fx.Int32(fx.block_idx.x)
        proxy = cco.DevComm(dev_comm).proxy(pool=pool)
        queue = _queue(pool, pid)
        off = fx.Int64(pid) * fx.Int64(CHUNK)
        sig = sig_base + pid
        for it in range_constexpr(ITERS):
            proxy.put(
                peer,
                win,
                off + fx.Int64(RECV_OFF),
                win,
                off + fx.Int64(SEND_OFF),
                CHUNK,
                signal_op=signal_op,
                signal_id=sig,
                signal_val=it + 1,
                coop=coop,
                queue=queue,
            )
            proxy.flush(coop=coop, queue=queue)
            proxy.wait_signal(sig, 1, coop=coop, queue=queue)

    @flyc.jit
    def run(
        dev_comm: Int64,
        win: Int64,
        peer: Int32,
        sig_base: Int32,
        stream=fx.Stream(None),
    ):
        put_kernel(dev_comm, win, peer, sig_base).launch(
            grid=(PROGRAMS, 1, 1), block=[threads, 1, 1], stream=stream
        )

    return run


@flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
def put_value_kernel(
    dev_comm: Int64, win: Int64, peer: Int32, value_base: Int64, sig_base: Int32
):
    # A plain put_value, then one with a SET signal that covers both.
    pid = fx.Int32(fx.block_idx.x)
    proxy = cco.DevComm(dev_comm).proxy()
    value = value_base + fx.Int64(pid)
    slot = fx.Int64(pid) * fx.Int64(8) + fx.Int64(VALUE_OFF)
    sig = sig_base + pid
    proxy.put_value(peer, win, slot, value, queue=pid)
    proxy.put_value(
        peer,
        win,
        slot + fx.Int64(8 * 8),
        value + fx.Int64(0x100),
        signal_op=cco.SignalOp.SET,
        signal_id=sig,
        signal_val=1,
        queue=pid,
    )
    proxy.flush(queue=pid)
    proxy.wait_signal(sig, 1, queue=pid)


@flyc.jit
def run_put_value(
    dev_comm: Int64,
    win: Int64,
    peer: Int32,
    value_base: Int64,
    sig_base: Int32,
    stream=fx.Stream(None),
):
    put_value_kernel(dev_comm, win, peer, value_base, sig_base).launch(
        grid=(PROGRAMS, 1, 1), block=[BLOCK_THREADS, 1, 1], stream=stream
    )


def _make_signal_kernel(signal_op, value, do_wait):
    """A bare signal, flush_timeout (stored as result 2p) and failed() (2p+1)."""

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def signal_kernel(dev_comm: Int64, win: Int64, peer: Int32, sig_base: Int32):
        pid = fx.Int32(fx.block_idx.x)
        proxy = cco.DevComm(dev_comm).proxy()
        sig = sig_base + pid
        proxy.signal(
            peer, signal_op=signal_op, signal_id=sig, signal_val=value, queue=pid
        )
        rc = proxy.flush_timeout(TIMEOUT_CYCLES, queue=pid)
        if do_wait:
            proxy.wait_signal(sig, value, queue=pid)
        failed = proxy.failed(queue=pid)
        out = _out_rsrc(win)
        buffer.buffer_store(rc, out, pid * 2, cache_modifier=buffer.CM_SC0_SC1)
        buffer.buffer_store(failed, out, pid * 2 + 1, cache_modifier=buffer.CM_SC0_SC1)

    @flyc.jit
    def run(
        dev_comm: Int64,
        win: Int64,
        peer: Int32,
        sig_base: Int32,
        blocks: Int32,
        stream=fx.Stream(None),
    ):
        signal_kernel(dev_comm, win, peer, sig_base).launch(
            grid=(blocks, 1, 1), block=[BLOCK_THREADS, 1, 1], stream=stream
        )

    return run


def _make_read_reset_kernel(pool, reset):
    """Optionally reset slot `sig`, then store its value as u64 result 0."""

    @flyc.kernel(known_block_size=[WARP_THREADS, 1, 1])
    def read_reset_kernel(dev_comm: Int64, win: Int64, sig: Int32):
        proxy = cco.DevComm(dev_comm).proxy(pool=pool)
        if reset:
            proxy.reset_signal(sig)
        value = proxy.read_signal(sig)
        buffer.buffer_store(value, _out_rsrc(win), 0, cache_modifier=buffer.CM_SC0_SC1)

    @flyc.jit
    def run(dev_comm: Int64, win: Int64, sig: Int32, stream=fx.Stream(None)):
        read_reset_kernel(dev_comm, win, sig).launch(
            grid=(1, 1, 1), block=[WARP_THREADS, 1, 1], stream=stream
        )

    return run


def pattern(rank):
    return [rank * 1_000_000 + i for i in range(NUM_ELEMS)]


def main() -> int:
    mpi = MPI.COMM_WORLD
    rank, nranks = mpi.Get_rank(), mpi.Get_size()
    if nranks != WS:
        if rank == 0:
            print(f"This example requires exactly {WS} ranks (mpirun -np {WS}).")
        return 1
    set_device(rank)
    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = mpi.bcast(uid, root=0)
    peer = (rank + 1) % WS
    failures = 0

    def report(label, ok, detail=""):
        nonlocal failures
        failures += 0 if ok else 1
        print(
            f"[rank {rank} {label:26}] {'PASS' if ok else 'FAIL'} {detail}".rstrip(),
            flush=True,
        )

    with Communicator.init(nranks, rank, uid, per_rank_vmm=PER_RANK_VMM) as comm:
        mem = comm.alloc_mem(WIN_BYTES)
        win = comm.register_window(mem.ptr, mem.size)
        zero(win.local_ptr, WIN_BYTES)
        fill(win.local_ptr + SEND_OFF, pattern(rank))

        # Proxy-only: signal slots and window MRs, no GDA QPs.
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_FULL
        reqs.gda_context_count = 0
        reqs.gda_signal_count = SIGNALS
        reqs.gda_counter_count = 0
        reqs.proxy_spsc_queue_count = PROGRAMS
        reqs.proxy_mpsc_queue_count = MPSC_QUEUES
        reqs.proxy_only = 1
        dc = comm.create_dev_comm(reqs)

        def run_puts(label, run, sig_base):
            zero(win.local_ptr + RECV_OFF, NBYTES)
            comm.barrier()
            run(dc.ptr, win.handle, peer, sig_base)
            sync()
            comm.barrier()
            got = list(read(win.local_ptr + RECV_OFF, NUM_ELEMS))
            bad = sum(g != e for g, e in zip(got, pattern(peer)))
            report(label, bad == 0, f"({bad} bad words)" if bad else "")

        # (label, signal_op, coop, pool, threads); SPSC SET then INC continue
        # the same queues across launches.
        put_cases = [
            ("put spsc block SET", cco.SignalOp.SET, BLOCK, SPSC, BLOCK_THREADS),
            ("put spsc block INC", cco.SignalOp.INC, BLOCK, SPSC, BLOCK_THREADS),
            ("put spsc warp SET", cco.SignalOp.SET, WARP, SPSC, WARP_THREADS),
            ("put mpsc block INC", cco.SignalOp.INC, BLOCK, MPSC, BLOCK_THREADS),
        ]
        sig_base = 0
        for label, signal_op, coop, pool, threads in put_cases:
            run = _make_put_kernel(signal_op, coop, pool, threads)
            run_puts(label, run, sig_base)
            sig_base += PROGRAMS

        # put_value: words p and 8 + p of the peer's value area.
        zero(win.local_ptr + VALUE_OFF, 16 * 8)
        comm.barrier()
        run_put_value(dc.ptr, win.handle, peer, (rank << 32) | VALUE_TAG, sig_base)
        sync()
        comm.barrier()
        values = list(read(win.local_ptr + VALUE_OFF, 16))
        base = (peer << 32) | VALUE_TAG
        ok = all(
            values[p] == base + p and values[8 + p] == base + p + 0x100
            for p in range(PROGRAMS)
        )
        report("put_value + SET", ok, "" if ok else str([hex(v) for v in values]))
        sig_base += PROGRAMS

        # ADD, waited for: flush_timeout 0 and failed() 0 in every block.
        zero(win.local_ptr + OUT_OFF, 256)
        comm.barrier()
        _make_signal_kernel(cco.SignalOp.ADD, ADD_VAL, True)(
            dc.ptr, win.handle, peer, sig_base, PROGRAMS
        )
        sync()
        out = list(read(win.local_ptr + OUT_OFF, 2 * PROGRAMS, ctype=ctypes.c_int32))
        report("signal ADD + flush_timeout", out == [0] * (2 * PROGRAMS), str(out))
        sig_base += PROGRAMS

        # An INC nobody waits for reads 1 once the sender's flush returned;
        # after reset_signal it reads 0. Both pools see the same slots.
        comm.barrier()
        zero(win.local_ptr + OUT_OFF, 256)
        _make_signal_kernel(cco.SignalOp.INC, 1, False)(
            dc.ptr, win.handle, peer, sig_base, 1
        )
        sync()
        out = list(read(win.local_ptr + OUT_OFF, 2, ctype=ctypes.c_int32))
        report("signal INC + flush_timeout", out == [0, 0], str(out))
        comm.barrier()
        reads = []
        for pool, reset in ((SPSC, False), (MPSC, False), (MPSC, True), (SPSC, True)):
            fill(win.local_ptr + OUT_OFF, [0xDEAD])
            _make_read_reset_kernel(pool, reset)(dc.ptr, win.handle, sig_base)
            sync()
            reads.append(read(win.local_ptr + OUT_OFF, 1)[0])
        report("read/reset_signal", reads == [1, 1, 0, 0], str(reads))
        comm.barrier()

    total = mpi.allreduce(failures, op=MPI.SUM)
    if rank == 0:
        print(
            (
                "All FlyDSL ccoProxy tests PASSED"
                if total == 0
                else f"FAILED ({total} checks)"
            ),
            flush=True,
        )
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
