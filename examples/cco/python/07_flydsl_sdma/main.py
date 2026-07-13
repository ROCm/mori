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
"""CCO Example 07 — FlyDSL SDMA all-reduce (copy-engine all-gather + local reduce)

A custom all-reduce built on cco's SDMA copy-engine API (``DevComm.sdma()``),
which moves data between intra-node peers via the DMA engines rather than the
compute units:

  * **Scatter (all-gather):** each rank SDMA-``put``s its input region into a
    dedicated per-source slot of every *other* peer's gather region, then
    ``quiet``s each peer to wait for the DMA copies + completion signals to land.
    SDMA is a one-sided copy engine — the put only guarantees *my* writes reached
    the peers, so a host ``barrier`` follows to ensure every peer's writes have
    also reached *me* before I read them.
  * **Reduce:** every rank sums its own input plus the gathered peer slots
    (f32, vectorized 16 B loads) into its output region. All ranks run it, so all
    end up with the full sum (= all-reduce).

SDMA addresses peers through the same flat symmetric-window VA as the LSA model
(``Window.lsa_ptr``); ``put``/``quiet`` just drive the copy through cco's
per-DevComm SDMA queue pool (enabled by ``sdma_queue_count``).

Single node, ``world_size`` ranks (one GPU each):

    mpirun -n 2 python main.py
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
from flydsl.expr import buffer_ops, range_constexpr
from flydsl.expr.typing import Int64, T

from mori.cco import Communicator, CCODevCommRequirements, GDA_CONNECTION_NONE
import mori.cco.device.flydsl as cco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cco_example_common import set_device, sync, fill, zero, read, F32  # noqa: E402

WS = 2  # world size (ranks, one GPU each)
THREADS = 256
NUM_ELEMS = 256 * 1024  # f32 elements to all-reduce (1 MiB)
NBYTES = NUM_ELEMS * 4
ELEMS_PER_PACK = 4  # vector<4xf32> = 16 B load unit
NUM_PACKS = NUM_ELEMS // ELEMS_PER_PACK
SDMA_QUEUES = 8  # per-DevComm SDMA queue pool

# Window layout: [ input | gather (WS slots) | output ].
# gather slot p receives peer p's input; my own slot is unused (own input is
# read straight from the input region during the reduce).
IN_OFF = 0
GATHER_OFF = NBYTES
OUT_OFF = GATHER_OFF + WS * NBYTES
WIN_BYTES = OUT_OFF + NBYTES


def _make_scatter_kernel(my_rank):
    """Scatter kernel specialized to this rank (peer loop constant-folds)."""
    dst_off = GATHER_OFF + my_rank * NBYTES  # my slot in every peer's gather region

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def scatter_kernel(dev_comm: Int64, win: Int64):
        sdma = cco.DevComm(dev_comm).sdma()
        if fx.thread_idx.x == 0:
            # issue one copy per peer, then wait on each.
            for p in range(WS):
                if p != my_rank:
                    sdma.put(
                        p,
                        win,
                        dst_off,
                        win,
                        IN_OFF,
                        NBYTES,
                        0,
                        coop=cco.CoopScope.THREAD,
                    )
            for p in range(WS):
                if p != my_rank:
                    sdma.quiet(p, coop=cco.CoopScope.THREAD)

    @flyc.jit
    def run(dev_comm: Int64, win: Int64, stream=fx.Stream(None)):
        scatter_kernel(dev_comm, win).launch(
            grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
        )

    return run


def _make_reduce_kernel(my_rank):
    """Reduce kernel: sum own input + gathered peer slots into output."""
    # Byte offsets (in my own window) of every contribution to the sum.
    src_offs = [IN_OFF] + [GATHER_OFF + p * NBYTES for p in range(WS) if p != my_rank]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def reduce_kernel(win: Int64):
        tid = fx.thread_idx.x
        w = cco.Window(win)
        # All reads are local memory — my own symmetric-window VA.
        src_rsrc = [
            buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(w.lsa_ptr(my_rank, off))
            )
            for off in src_offs
        ]
        out_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(w.lsa_ptr(my_rank, OUT_OFF))
        )
        for pk in range(tid, NUM_PACKS, THREADS):
            elem_off = pk * ELEMS_PER_PACK
            acc = None
            for s in range_constexpr(len(src_rsrc)):
                v = fx.Vector(
                    buffer_ops.buffer_load(
                        src_rsrc[s], elem_off, vec_width=4, dtype=T.f32
                    )
                )
                acc = v if acc is None else acc + v
            buffer_ops.buffer_store(acc, out_rsrc, elem_off)

    @flyc.jit
    def run(win: Int64, stream=fx.Stream(None)):
        reduce_kernel(win).launch(grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream)

    return run


def main() -> int:
    mpi = MPI.COMM_WORLD
    rank, nranks = mpi.Get_rank(), mpi.Get_size()
    if nranks != WS:
        if rank == 0:
            print(f"This example is built for world_size={WS} (mpirun -n {WS}).")
        return 1
    # SDMA is gated by MORI_ENABLE_SDMA; without it the DevComm builds no SDMA
    # queue pool (sdmaNumQueue=0, null device handles) and the copy-engine put
    # dereferences a null pointer -> opaque GPU memory-access fault.
    if os.environ.get("MORI_ENABLE_SDMA") not in ("1", "true", "TRUE", "on", "ON"):
        if rank == 0:
            print(
                "This example needs SDMA enabled. Re-run with:\n"
                f"    MORI_ENABLE_SDMA=1 mpirun -x MORI_ENABLE_SDMA -n {WS} python main.py",
                flush=True,
            )
        return 1
    set_device(rank)
    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = mpi.bcast(uid, root=0)

    scatter = _make_scatter_kernel(rank)
    reduce = _make_reduce_kernel(rank)
    errors = 0

    with Communicator.init(nranks, rank, uid, per_rank_vmm=256 * 1024 * 1024) as comm:
        mem = comm.alloc_mem(WIN_BYTES)
        win = comm.register_window(mem.ptr, mem.size)

        # Zero the window, then fill my input: input[i] = (rank + 1) * (i + 1)
        # -> allreduce sum = S * (i + 1), S = sum_{r=1..WS} r = WS*(WS+1)/2.
        zero(win.local_ptr, WIN_BYTES)
        fill(
            win.local_ptr + IN_OFF,
            [(rank + 1) * (i + 1) for i in range(NUM_ELEMS)],
            F32,
        )

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = SDMA_QUEUES
        dc = comm.create_dev_comm(reqs)

        comm.barrier()  # all inputs visible + gather regions zeroed
        scatter(dc.ptr, win.handle)  # push my input into every peer's slot
        sync()
        comm.barrier()  # every peer's slot now populated in my window
        reduce(win.handle)  # local sum -> output
        sync()

        S = WS * (WS + 1) // 2
        host = read(win.local_ptr + OUT_OFF, NUM_ELEMS, F32)
        for i in (0, 1, NUM_ELEMS // 2, NUM_ELEMS - 1):
            exp = float(S * (i + 1))
            if abs(host[i] - exp) > 1e-3 * max(1.0, exp):
                print(
                    f"[rank {rank}] MISMATCH [{i}]: got {host[i]} expected {exp}",
                    flush=True,
                )
                errors += 1
        print(
            f"[rank {rank}] {'allreduce verified' if errors == 0 else 'FAILED'} "
            f"(SDMA all-gather + reduce over {WS} ranks of {NUM_ELEMS} f32), "
            f"out[0,1,-1]={[host[0], host[1], host[NUM_ELEMS-1]]}",
            flush=True,
        )

    all_err = mpi.allreduce(errors, op=MPI.SUM)
    if rank == 0:
        print("SUCCESS" if all_err == 0 else "FAILED", flush=True)
    return 0 if all_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
