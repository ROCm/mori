#!/usr/bin/env python3
"""CCO Example 05 — FlyDSL LSA custom all-reduce (device signal protocol)

A vLLM-style *custom all-reduce* over cco symmetric windows, modeled on FlyDSL's
``kernels/custom_all_reduce_kernel.py`` but self-contained and cco-sourced:

  * Peer base addresses for the data and signal regions come from cco's flat-VA
    window via ``DevComm.lsa_ptr(win, peer, off)`` — NO manual HIP-IPC handle
    exchange (cco's symmetric window registration already made peers P2P-reachable).
  * The cross-GPU barrier is a device-side signal protocol (each rank writes a
    flag into every peer's signal slot, then spins until all peers have arrived),
    using uncached buffer loads/stores — no host synchronization inside the kernel.
  * 1-stage reduction: after the barrier, each thread reads the same 16-byte pack
    from every peer's input region, sums (f32), and writes its own output region.
    Every rank runs it, so every rank ends up with the full sum (= all-reduce).

Single node, one block, ``world_size`` ranks (one GPU each):

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
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import buffer_ops, range_constexpr
from flydsl.expr import gpu as fgpu
from flydsl.expr.typing import Int32, Int64, T

from mori.cco import Communicator, CCODevCommRequirements, GDA_CONNECTION_NONE
import mori.cco.device.flydsl as cco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cco_example_common import set_device, sync, fill, zero, read, F32  # noqa: E402

# Cache-modifier aux bits (GFX942): SC0=bypass L1, SC1=bypass L2.
_CM_CACHED = 0
_CM_SC1 = 2          # uncached read  (see peers' fresh signal writes)
_CM_SC0_SC1 = 3      # uncached write (signal stores)

WS = 2                       # world size (ranks, one GPU each)
THREADS = 256
NUM_ELEMS = 256 * 1024       # f32 elements to all-reduce (1 MiB)
ELEMS_PER_PACK = 4           # vector<4xi32> = 16 B atomic unit
NUM_PACKS = NUM_ELEMS // ELEMS_PER_PACK

# Window layout: [ signal | input | output ]
SIG_OFF = 0
SIG_BYTES = 256              # WS u32 arrival slots, padded
IN_OFF = SIG_BYTES
OUT_OFF = IN_OFF + NUM_ELEMS * 4
WIN_BYTES = OUT_OFF + NUM_ELEMS * 4


def _rsrc(addr_i64):
    return buffer_ops.create_buffer_resource_from_addr(addr_i64)


def _store_u32_uncached(rsrc, val):
    buffer_ops.buffer_store(val, rsrc, 0, cache_modifier=_CM_SC0_SC1)


def _load_u32_uncached(rsrc):
    return buffer_ops.buffer_load(rsrc, 0, vec_width=1, dtype=T.i32, cache_modifier=_CM_SC1)


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def custom_ar_kernel(dev_comm: Int64, win: Int64, flag: Int32):
    tid = fx.thread_idx.x
    dc = cco.DevComm(dev_comm)
    w = cco.Window(win)
    rank = dc.lsa_rank  # int32, my index within the node's LSA team

    # Peer base addresses (i64) for each region, straight from cco's flat VA —
    # the LSA model: get peer pointers, then load/store them directly below.
    ins = [fx.Int64(w.lsa_ptr(p, IN_OFF)) for p in range(WS)]
    self_sig = fx.Int64(w.lsa_ptr(rank, SIG_OFF))
    out = fx.Int64(w.lsa_ptr(rank, OUT_OFF))

    # ── device signal barrier (start-sync): lane l (<WS) signals peer l, then
    #    waits until peer l signalled us back; gpu.barrier joins the block. ──
    if tid < WS:
        # write `flag` into peer[tid]'s signal slot reserved for me (rank).
        peer_base = fx.Int64(w.lsa_ptr(tid, SIG_OFF))
        _store_u32_uncached(_rsrc(peer_base + fx.Int64(rank) * fx.Int64(4)), flag)

        # spin until my own slot for peer tid reaches `flag`.
        wait_addr = self_sig + fx.Int64(tid) * fx.Int64(4)
        i32 = T.i32
        first = _load_u32_uncached(_rsrc(wait_addr))
        loop = scf.WhileOp([i32], [first.ir_value() if hasattr(first, "ir_value") else first])
        cond = ir.Block.create_at_start(loop.before, [i32])
        body = ir.Block.create_at_start(loop.after, [i32])
        with ir.InsertionPoint(cond):
            cur = fx.Int32(cond.arguments[0])
            should_wait = fx.Uint32(cur) < fx.Uint32(flag)
            scf.ConditionOp(should_wait.ir_value(), [cond.arguments[0]])
        with ir.InsertionPoint(body):
            nxt = _load_u32_uncached(_rsrc(wait_addr))
            scf.YieldOp([nxt.ir_value() if hasattr(nxt, "ir_value") else nxt])
    fgpu.barrier()

    # ── 1-stage reduction: sum peers' input packs into my output. ──
    out_rsrc = _rsrc(out)
    in_rsrc = [_rsrc(ins[p]) for p in range(WS)]
    for pk in range(tid, NUM_PACKS, THREADS):
        elem_off = pk * ELEMS_PER_PACK
        acc = None
        for p in range_constexpr(WS):
            raw = fx.Vector(buffer_ops.buffer_load(in_rsrc[p], elem_off, vec_width=4, dtype=T.i32))
            vf = raw.bitcast(fx.Float32)
            acc = vf if acc is None else acc + vf
        buffer_ops.buffer_store(acc.bitcast(fx.Int32), out_rsrc, elem_off,
                                cache_modifier=_CM_CACHED)


@flyc.jit
def run_ar(dev_comm: Int64, win: Int64, flag: Int32, stream=fx.Stream(None)):
    custom_ar_kernel(dev_comm, win, flag).launch(grid=(1, 1, 1), block=[THREADS, 1, 1],
                                                 stream=stream)


def main() -> int:
    mpi = MPI.COMM_WORLD
    rank, nranks = mpi.Get_rank(), mpi.Get_size()
    if nranks != WS:
        if rank == 0:
            print(f"This example is built for world_size={WS} (mpirun -n {WS}).")
        return 1
    set_device(rank)
    uid = Communicator.get_unique_id() if rank == 0 else None
    uid = mpi.bcast(uid, root=0)

    errors = 0
    with Communicator.init(nranks, rank, uid, per_rank_vmm=256 * 1024 * 1024) as comm:
        win = comm.alloc_window(WIN_BYTES)

        # Zero the whole window (incl. signal slots), then fill my input region:
        #   input[i] = (rank + 1) * (i + 1)  ->  allreduce sum = S * (i+1),
        #   where S = sum_{r=1..WS} r = WS*(WS+1)/2.
        zero(win.local_ptr, WIN_BYTES)
        fill(win.local_ptr + IN_OFF, [(rank + 1) * (i + 1) for i in range(NUM_ELEMS)], F32)

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0; reqs.gda_counter_count = 0
        dc = comm.create_dev_comm(reqs)

        comm.barrier()                      # all inputs + zeroed signals visible
        run_ar(dc.ptr, win.handle, 1)       # flag = 1 (single round)
        sync()
        comm.barrier()

        S = WS * (WS + 1) // 2
        host = read(win.local_ptr + OUT_OFF, NUM_ELEMS, F32)
        for i in (0, 1, NUM_ELEMS // 2, NUM_ELEMS - 1):
            exp = float(S * (i + 1))
            if abs(host[i] - exp) > 1e-3 * max(1.0, exp):
                print(f"[rank {rank}] MISMATCH [{i}]: got {host[i]} expected {exp}", flush=True)
                errors += 1
        print(f"[rank {rank}] {'allreduce verified' if errors == 0 else 'FAILED'} "
              f"(sum over {WS} ranks of {NUM_ELEMS} f32), "
              f"out[0,1,-1]={[host[0], host[1], host[NUM_ELEMS-1]]}", flush=True)

    all_err = mpi.allreduce(errors, op=MPI.SUM)
    if rank == 0:
        print("SUCCESS" if all_err == 0 else "FAILED", flush=True)
    return 0 if all_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
