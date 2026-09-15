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
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Toolchain probe: example 05's LSA all-reduce, ported onto ``_compat``.

Validates the pieces the real kernels depend on, on whichever flydsl is
installed: ``Window.lsa_ptr``, the uncached signal barrier, vectorized peer
reads, and the cco device-bitcode JIT.

    MORI_SOCKET_IFNAME=lo mpirun --allow-run-as-root -np 2 python _probe_lsa.py
"""

import os
import sys

from mpi4py import MPI

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import gpu as fgpu
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int32, Int64

import mori.cco.device.flydsl as cco
from mori.cco import CCODevCommRequirements, Communicator, GDA_CONNECTION_NONE

from mori.ops.gemm_ar._compat import (
    CM_CACHED,
    FLYDSL_VERSION,
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
    i32_type,
    signal_load_u32,
    signal_ptr,
    signal_store_u32,
)

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "examples",
        "cco",
        "python",
    ),
)
from cco_example_common import F32, fill, read, set_device, sync, zero  # noqa: E402

WS = 2
THREADS = 256
NUM_ELEMS = 256 * 1024
ELEMS_PER_PACK = 4
NUM_PACKS = NUM_ELEMS // ELEMS_PER_PACK

SIG_OFF = 0
SIG_BYTES = 256
IN_OFF = SIG_BYTES
OUT_OFF = IN_OFF + NUM_ELEMS * 4
WIN_BYTES = OUT_OFF + NUM_ELEMS * 4


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def probe_ar_kernel(dev_comm: Int64, win: Int64, flag: Int32):
    tid = fx.thread_idx.x
    dc = cco.DevComm(dev_comm)
    w = cco.Window(win)
    rank = dc.lsa_rank

    ins = [fx.Int64(w.lsa_ptr(p, IN_OFF)) for p in range(WS)]
    self_sig = fx.Int64(w.lsa_ptr(rank, SIG_OFF))
    out = fx.Int64(w.lsa_ptr(rank, OUT_OFF))

    # start-sync: publish my arrival to every peer, then wait for all of them.
    if tid < WS:
        peer_base = fx.Int64(w.lsa_ptr(tid, SIG_OFF))
        signal_store_u32(signal_ptr(peer_base + fx.Int64(rank) * fx.Int64(4)), flag)

        wait_addr = self_sig + fx.Int64(tid) * fx.Int64(4)
        wait_rsrc = signal_ptr(wait_addr)
        i32 = i32_type()
        first = signal_load_u32(wait_rsrc)
        first_v = first.ir_value() if hasattr(first, "ir_value") else first
        loop = scf.WhileOp([i32], [first_v])
        cond = ir.Block.create_at_start(loop.before, [i32])
        body = ir.Block.create_at_start(loop.after, [i32])
        with ir.InsertionPoint(cond):
            cur = fx.Int32(cond.arguments[0])
            should_wait = fx.Uint32(cur) < fx.Uint32(flag)
            scf.ConditionOp(should_wait.ir_value(), [cond.arguments[0]])
        with ir.InsertionPoint(body):
            nxt = signal_load_u32(wait_rsrc)
            scf.YieldOp([nxt.ir_value() if hasattr(nxt, "ir_value") else nxt])
    fgpu.barrier()

    out_rsrc = create_buffer_resource_from_addr(out)
    in_rsrc = [create_buffer_resource_from_addr(ins[p]) for p in range(WS)]
    for pk in range(tid, NUM_PACKS, THREADS):
        elem_off = pk * ELEMS_PER_PACK
        acc = None
        for p in range_constexpr(WS):
            raw = fx.Vector(
                buffer_load(in_rsrc[p], elem_off, vec_width=4, dtype=i32_type())
            )
            vf = raw.bitcast(fx.Float32)
            acc = vf if acc is None else acc + vf
        buffer_store(
            acc.bitcast(fx.Int32), out_rsrc, elem_off, cache_modifier=CM_CACHED
        )


@flyc.jit
def run_probe(dev_comm: Int64, win: Int64, flag: Int32, stream=fx.Stream(None)):
    probe_ar_kernel(dev_comm, win, flag).launch(
        grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
    )


def main() -> int:
    mpi = MPI.COMM_WORLD
    rank, nranks = mpi.Get_rank(), mpi.Get_size()
    if nranks != WS:
        if rank == 0:
            print(f"probe is built for world_size={WS} (mpirun -n {WS})")
        return 1
    set_device(rank)
    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = mpi.bcast(uid, root=0)

    errors = 0
    with Communicator.init(nranks, rank, uid, per_rank_vmm=256 * 1024 * 1024) as comm:
        mem = comm.alloc_mem(WIN_BYTES)
        win = comm.register_window(mem.ptr, mem.size)
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
        dc = comm.create_dev_comm(reqs)

        comm.barrier()
        run_probe(dc.ptr, win.handle, 1)
        sync()
        comm.barrier()

        S = WS * (WS + 1) // 2
        host = read(win.local_ptr + OUT_OFF, NUM_ELEMS, F32)
        for i in (0, 1, NUM_ELEMS // 2, NUM_ELEMS - 1):
            exp = float(S * (i + 1))
            if abs(host[i] - exp) > 1e-3 * max(1.0, exp):
                print(f"[rank {rank}] MISMATCH [{i}]: {host[i]} != {exp}", flush=True)
                errors += 1
        print(
            f"[rank {rank}] flydsl={FLYDSL_VERSION} "
            f"{'PROBE OK' if errors == 0 else 'PROBE FAILED'} "
            f"out[0,1,-1]={[host[0], host[1], host[NUM_ELEMS - 1]]}",
            flush=True,
        )

    all_err = mpi.allreduce(errors, op=MPI.SUM)
    if rank == 0:
        print("SUCCESS" if all_err == 0 else "FAILED", flush=True)
    return 0 if all_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
