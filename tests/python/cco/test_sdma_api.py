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
"""Functional tests for the cco SDMA FlyDSL API — all-to-all via put, drained
with quiet (all queues) and quiet_queue (single queue).

These drive the copy engine end-to-end on real GPUs. Each test spawns
``world_size`` processes (one GPU each) via ``torch.multiprocessing``; rank 0
mints a cco UniqueId and shares it through a file, so no MPI launcher is needed.

SDMA is gated by ``MORI_ENABLE_SDMA``; without it the DevComm builds no queue
pool and the copy-engine ops dereference null. The spawn helper sets it for the
children, and the whole module skips if fewer than 2 GPUs are visible.

Data model (mirrors tests/cpp/cco/test_sdma_put.cpp): each rank fills its input
region with ``rank*1000 + i`` and, after the collective, every rank checks it
received the exact bytes the source rank produced.

Host<->device memcpy goes through the test-owned ``helper`` module (this dir),
so the suite does not depend on examples/.
"""

import os
import tempfile
import traceback

import pytest
import torch

flydsl = pytest.importorskip("flydsl", reason="cco FlyDSL bindings require flydsl")

NUM_ELEMS = 64  # uint32 payload per transfer
NBYTES = NUM_ELEMS * 4
THREADS = 64


def _num_gpus() -> int:
    return torch.cuda.device_count()


def _require_gpus(world_size: int) -> None:
    n = _num_gpus()
    if n < world_size:
        pytest.skip(f"Need {world_size} GPUs, only {n} available")


# ── per-rank worker bodies (run in a spawned process) ──────────────────────
# The workers run in freshly spawned child interpreters, so their imports MUST
# stay inside the function bodies (via _worker_imports): flydsl/cco initialize a
# GPU runtime that must come up only after MORI_ENABLE_SDMA + set_device.


def _worker_imports():
    """The full import set a spawned worker needs (see module docstring)."""
    import ctypes

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr.typing import Int64
    from mori.cco import CCODevCommRequirements, GDA_CONNECTION_NONE
    import mori.cco.device.flydsl as cco

    from .helper import sync, fill, zero, read

    return dict(
        ctypes=ctypes,
        flyc=flyc,
        fx=fx,
        Int64=Int64,
        CCODevCommRequirements=CCODevCommRequirements,
        GDA_CONNECTION_NONE=GDA_CONNECTION_NONE,
        cco=cco,
        sync=sync,
        fill=fill,
        zero=zero,
        read=read,
    )


def _init_comm(rank, world_size, uid_path, per_rank_vmm=128 * 1024 * 1024):
    """Common setup: set device, load shared UniqueId, create Communicator."""
    from mori.cco import Communicator, UniqueId

    torch.cuda.set_device(rank)
    with open(uid_path, "rb") as f:
        uid = UniqueId.from_bytes(f.read())
    comm = Communicator.init(world_size, rank, uid, per_rank_vmm=per_rank_vmm)
    return comm


def _worker_put(rank, world_size, uid_path, coop_name, results_path):
    """All-to-all via SDMA put: rank writes its slice into every peer's slot."""
    err = None
    try:
        mods = _worker_imports()
        flyc, fx, Int64, cco = mods["flyc"], mods["fx"], mods["Int64"], mods["cco"]
        sync, fill, zero, read = mods["sync"], mods["fill"], mods["zero"], mods["read"]
        u32 = mods["ctypes"].c_uint32
        CCODevCommRequirements = mods["CCODevCommRequirements"]
        GDA_CONNECTION_NONE = mods["GDA_CONNECTION_NONE"]

        coop = getattr(cco.CoopScope, coop_name)
        comm = _init_comm(rank, world_size, uid_path)

        # Window layout: [ send | recv (world_size slots) ]
        send_off = 0
        recv_off = NBYTES
        win_bytes = recv_off + world_size * NBYTES

        mem = comm.alloc_mem(win_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        zero(win.local_ptr, win_bytes)
        # send[i] = rank*1000 + i
        fill(win.local_ptr + send_off, [rank * 1000 + i for i in range(NUM_ELEMS)], u32)

        # Before a2a: show my local send buffer (first 8 elems).
        pre = read(win.local_ptr + send_off, NUM_ELEMS, u32)
        print(f"[rank {rank}] BEFORE a2a  send[:8] = {list(pre[:8])}", flush=True)

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 8
        dc = comm.create_dev_comm(reqs)

        my_rank = rank
        # THREAD coop = one logical op from lane 0; WARP coop needs the whole
        # wavefront to enter together (each lane drives one SDMA queue).
        gate_single = coop == cco.CoopScope.THREAD

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def put_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            active = fx.thread_idx.x == 0 if gate_single else fx.thread_idx.x < 64
            if active:
                for p in range(world_size):
                    if p != my_rank:
                        # into peer p's recv slot reserved for me (my_rank)
                        sdma.put(
                            p,
                            w,
                            recv_off + my_rank * NBYTES,
                            w,
                            send_off,
                            NBYTES,
                            0,
                            coop=coop,
                        )
                for p in range(world_size):
                    if p != my_rank:
                        sdma.quiet(p, coop=coop)

        @flyc.jit
        def run(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            put_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        comm.barrier()
        run(dc.ptr, win.handle)
        sync()
        comm.barrier()

        # recv slot s must hold rank s's send buffer: s*1000 + i
        for s in range(world_size):
            if s == my_rank:
                continue
            got = read(win.local_ptr + recv_off + s * NBYTES, NUM_ELEMS, u32)
            print(
                f"[rank {my_rank}] AFTER  a2a  recv[from {s}][:8] = "
                f"{list(got[:8])}",
                flush=True,
            )
            for i in (0, NUM_ELEMS // 2, NUM_ELEMS - 1):
                exp = s * 1000 + i
                if got[i] != exp:
                    raise AssertionError(
                        f"rank {my_rank} recv[{s}][{i}]={got[i]} expected {exp}"
                    )
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


def _worker_put_quiet_queue(rank, world_size, uid_path, coop_name, results_path):
    """Same all-to-all put, but drained per queue with quiet_queue(peer, 0).

    THREAD-coop put uses queue 0 only, so a single quiet_queue on queue 0 fully
    drains it — this exercises the single-(peer,queue) completion path distinct
    from the all-queue quiet() used by _worker_put."""
    err = None
    try:
        mods = _worker_imports()
        flyc, fx, Int64, cco = mods["flyc"], mods["fx"], mods["Int64"], mods["cco"]
        sync, fill, zero, read = mods["sync"], mods["fill"], mods["zero"], mods["read"]
        u32 = mods["ctypes"].c_uint32
        CCODevCommRequirements = mods["CCODevCommRequirements"]
        GDA_CONNECTION_NONE = mods["GDA_CONNECTION_NONE"]

        comm = _init_comm(rank, world_size, uid_path)
        send_off = 0
        recv_off = NBYTES
        win_bytes = recv_off + world_size * NBYTES

        mem = comm.alloc_mem(win_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        zero(win.local_ptr, win_bytes)
        fill(win.local_ptr + send_off, [rank * 1000 + i for i in range(NUM_ELEMS)], u32)

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 8
        dc = comm.create_dev_comm(reqs)

        my_rank = rank

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def put_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            if fx.thread_idx.x == 0:
                for p in range(world_size):
                    if p != my_rank:
                        sdma.put(
                            p,
                            w,
                            recv_off + my_rank * NBYTES,
                            w,
                            send_off,
                            NBYTES,
                            0,
                            coop=cco.CoopScope.THREAD,
                        )
                for p in range(world_size):
                    if p != my_rank:
                        sdma.quiet_queue(p, 0)

        @flyc.jit
        def run(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            put_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        comm.barrier()
        run(dc.ptr, win.handle)
        sync()
        comm.barrier()

        for s in range(world_size):
            if s == my_rank:
                continue
            got = read(win.local_ptr + recv_off + s * NBYTES, NUM_ELEMS, u32)
            for i in (0, NUM_ELEMS // 2, NUM_ELEMS - 1):
                exp = s * 1000 + i
                if got[i] != exp:
                    raise AssertionError(
                        f"rank {my_rank} recv[{s}][{i}]={got[i]} expected {exp}"
                    )
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


def _worker_signals(rank, world_size, uid_path, coop_name, results_path):
    """put / put_ptr / get_ptr with local and remote signals, consumed by
    wait_signal: window put with both, put_ptr remote only, get_ptr local only."""
    err = None
    try:
        mods = _worker_imports()
        flyc, fx, Int64, cco = mods["flyc"], mods["fx"], mods["Int64"], mods["cco"]
        sync, fill, zero, read = mods["sync"], mods["fill"], mods["zero"], mods["read"]
        u32 = mods["ctypes"].c_uint32
        CCODevCommRequirements = mods["CCODevCommRequirements"]
        GDA_CONNECTION_NONE = mods["GDA_CONNECTION_NONE"]

        comm = _init_comm(rank, world_size, uid_path)
        # Window layout: [ send | put recv | put_ptr recv | get_ptr recv ]
        send_off, put_off, ptr_off, get_off = 0, NBYTES, 2 * NBYTES, 3 * NBYTES
        win_bytes = 4 * NBYTES

        mem = comm.alloc_mem(win_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        zero(win.local_ptr, win_bytes)
        fill(win.local_ptr + send_off, [rank * 1000 + i for i in range(NUM_ELEMS)], u32)

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 8
        dc = comm.create_dev_comm(reqs)

        me, peer = rank, 1 - rank

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def signal_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            window = cco.Window(w)
            if fx.thread_idx.x == 0:
                sdma.put(
                    peer,
                    w,
                    put_off,
                    w,
                    send_off,
                    NBYTES,
                    0,
                    local_signal=True,
                    remote_signal=True,
                )
                sdma.put_ptr(
                    peer,
                    window.lsa_ptr(peer, ptr_off),
                    window.lsa_ptr(me, send_off),
                    NBYTES,
                    0,
                    remote_signal=True,
                )
                sdma.get_ptr(
                    peer,
                    window.lsa_ptr(me, get_off),
                    window.lsa_ptr(peer, send_off),
                    NBYTES,
                    0,
                    local_signal=True,
                )
                # Ours: put + get_ptr. The peer's: its put + put_ptr.
                sdma.wait_signal(me, 0, 2)
                sdma.wait_signal(peer, 0, 2)
                sdma.quiet_queue(peer, 0)

        @flyc.jit
        def run(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            signal_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        comm.barrier()
        run(dc.ptr, win.handle)
        sync()
        comm.barrier()

        for off, src, what in (
            (put_off, peer, "put"),
            (ptr_off, peer, "put_ptr"),
            (get_off, peer, "get_ptr"),
        ):
            got = read(win.local_ptr + off, NUM_ELEMS, u32)
            for i in (0, NUM_ELEMS // 2, NUM_ELEMS - 1):
                if got[i] != src * 1000 + i:
                    raise AssertionError(
                        f"rank {me} {what}[{i}]={got[i]} expected {src * 1000 + i}"
                    )
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


def _worker_put_block(rank, world_size, uid_path, coop_name, results_path):
    """All-to-all via BLOCK-coop SDMA put: the whole block drives the queues.

    Uses a 128-thread (2-warp) block so the block path is genuinely exercised
    (multi-warp, threadIdx.x-indexed queues) rather than degenerating to a single
    warp. Every thread must enter put/quiet together; threads beyond the queue
    count return internally."""
    err = None
    try:
        mods = _worker_imports()
        flyc, fx, Int64, cco = mods["flyc"], mods["fx"], mods["Int64"], mods["cco"]
        sync, fill, zero, read = mods["sync"], mods["fill"], mods["zero"], mods["read"]
        u32 = mods["ctypes"].c_uint32
        CCODevCommRequirements = mods["CCODevCommRequirements"]
        GDA_CONNECTION_NONE = mods["GDA_CONNECTION_NONE"]

        comm = _init_comm(rank, world_size, uid_path)
        send_off = 0
        recv_off = NBYTES
        win_bytes = recv_off + world_size * NBYTES

        mem = comm.alloc_mem(win_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        zero(win.local_ptr, win_bytes)
        fill(win.local_ptr + send_off, [rank * 1000 + i for i in range(NUM_ELEMS)], u32)

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 8
        dc = comm.create_dev_comm(reqs)

        my_rank = rank
        block_threads = 128  # 2 warps, > warpSize, to exercise the block path

        @flyc.kernel(known_block_size=[block_threads, 1, 1])
        def put_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            # BLOCK coop: all threads participate (no gating); each put/quiet is
            # a block-collective op internally distributing across the queues.
            for p in range(world_size):
                if p != my_rank:
                    sdma.put(
                        p,
                        w,
                        recv_off + my_rank * NBYTES,
                        w,
                        send_off,
                        NBYTES,
                        0,
                        coop=cco.CoopScope.BLOCK,
                    )
            for p in range(world_size):
                if p != my_rank:
                    sdma.quiet(p, coop=cco.CoopScope.BLOCK)

        @flyc.jit
        def run(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            put_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[block_threads, 1, 1], stream=stream
            )

        comm.barrier()
        run(dc.ptr, win.handle)
        sync()
        comm.barrier()

        for s in range(world_size):
            if s == my_rank:
                continue
            got = read(win.local_ptr + recv_off + s * NBYTES, NUM_ELEMS, u32)
            for i in (0, NUM_ELEMS // 2, NUM_ELEMS - 1):
                exp = s * 1000 + i
                if got[i] != exp:
                    raise AssertionError(
                        f"rank {my_rank} recv[{s}][{i}]={got[i]} expected {exp}"
                    )
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


def _worker_put_nosignal(rank, world_size, uid_path, coop_name, results_path):
    """No-signal (fire-and-forget) put, drained via a trailing signaled put.

    Each transfer is sent in two halves on the SAME queue 0: the first half with
    signal=False (no completion atomic), the second with signal=True. Because the
    queue is in-order, draining the second half with quiet_queue(peer, 0) also
    guarantees the first (no-signal) half has landed. This both validates the
    no-signal path and demonstrates its intended batching use."""
    err = None
    try:
        mods = _worker_imports()
        flyc, fx, Int64, cco = mods["flyc"], mods["fx"], mods["Int64"], mods["cco"]
        sync, fill, zero, read = mods["sync"], mods["fill"], mods["zero"], mods["read"]
        u32 = mods["ctypes"].c_uint32
        CCODevCommRequirements = mods["CCODevCommRequirements"]
        GDA_CONNECTION_NONE = mods["GDA_CONNECTION_NONE"]

        comm = _init_comm(rank, world_size, uid_path)
        send_off = 0
        recv_off = NBYTES
        win_bytes = recv_off + world_size * NBYTES
        half_bytes = NBYTES // 2  # split point between the no-signal / signaled halves

        mem = comm.alloc_mem(win_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        zero(win.local_ptr, win_bytes)
        fill(win.local_ptr + send_off, [rank * 1000 + i for i in range(NUM_ELEMS)], u32)

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 8
        dc = comm.create_dev_comm(reqs)

        my_rank = rank

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def put_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            if fx.thread_idx.x == 0:
                for p in range(world_size):
                    if p != my_rank:
                        dst = recv_off + my_rank * NBYTES
                        # first half: fire-and-forget (no completion atomic)
                        sdma.put(
                            p,
                            w,
                            dst,
                            w,
                            send_off,
                            half_bytes,
                            0,
                            coop=cco.CoopScope.THREAD,
                            signal=False,
                        )
                        # second half: signaled; same queue 0 drains both in-order
                        sdma.put(
                            p,
                            w,
                            dst + half_bytes,
                            w,
                            send_off + half_bytes,
                            half_bytes,
                            0,
                            coop=cco.CoopScope.THREAD,
                            signal=True,
                        )
                for p in range(world_size):
                    if p != my_rank:
                        sdma.quiet_queue(p, 0)

        @flyc.jit
        def run(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            put_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        comm.barrier()
        run(dc.ptr, win.handle)
        sync()
        comm.barrier()

        # Check the whole payload — the no-signal first half must have landed too.
        for s in range(world_size):
            if s == my_rank:
                continue
            got = read(win.local_ptr + recv_off + s * NBYTES, NUM_ELEMS, u32)
            for i in (0, NUM_ELEMS // 2, NUM_ELEMS - 1):
                exp = s * 1000 + i
                if got[i] != exp:
                    raise AssertionError(
                        f"rank {my_rank} recv[{s}][{i}]={got[i]} expected {exp}"
                    )
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


def _worker_put_aggregate(rank, world_size, uid_path, coop_name, results_path):
    """Aggregate puts: 3 chunks/peer on queue 0 with aggregate=True (no doorbell),
    then one commit(peer, 0) rings the batch and quiet_queue drains it."""
    err = None
    try:
        mods = _worker_imports()
        flyc, fx, Int64, cco = mods["flyc"], mods["fx"], mods["Int64"], mods["cco"]
        sync, fill, zero, read = mods["sync"], mods["fill"], mods["zero"], mods["read"]
        u32 = mods["ctypes"].c_uint32
        CCODevCommRequirements = mods["CCODevCommRequirements"]
        GDA_CONNECTION_NONE = mods["GDA_CONNECTION_NONE"]

        comm = _init_comm(rank, world_size, uid_path)
        send_off = 0
        recv_off = NBYTES
        win_bytes = recv_off + world_size * NBYTES
        chunk = NBYTES // 3  # three chunks; the last absorbs the remainder

        mem = comm.alloc_mem(win_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        zero(win.local_ptr, win_bytes)
        fill(win.local_ptr + send_off, [rank * 1000 + i for i in range(NUM_ELEMS)], u32)

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 8
        dc = comm.create_dev_comm(reqs)

        my_rank = rank
        c0, c1 = chunk, 2 * chunk
        last = NBYTES - c1  # remainder for the third chunk

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def put_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            if fx.thread_idx.x == 0:
                for p in range(world_size):
                    if p != my_rank:
                        dst = recv_off + my_rank * NBYTES
                        # aggregate: placed in the SQ, doorbell suppressed
                        sdma.put(
                            p,
                            w,
                            dst,
                            w,
                            send_off,
                            c0,
                            0,
                            coop=cco.CoopScope.THREAD,
                            signal=True,
                            aggregate=True,
                        )
                        sdma.put(
                            p,
                            w,
                            dst + c0,
                            w,
                            send_off + c0,
                            chunk,
                            0,
                            coop=cco.CoopScope.THREAD,
                            signal=True,
                            aggregate=True,
                        )
                        sdma.put(
                            p,
                            w,
                            dst + c1,
                            w,
                            send_off + c1,
                            last,
                            0,
                            coop=cco.CoopScope.THREAD,
                            signal=True,
                            aggregate=True,
                        )
                        # one doorbell rings the whole batch
                        sdma.commit(p, 0, coop=cco.CoopScope.THREAD)
                for p in range(world_size):
                    if p != my_rank:
                        sdma.quiet_queue(p, 0)

        @flyc.jit
        def run(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            put_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        comm.barrier()
        run(dc.ptr, win.handle)
        sync()
        comm.barrier()

        for s in range(world_size):
            if s == my_rank:
                continue
            got = read(win.local_ptr + recv_off + s * NBYTES, NUM_ELEMS, u32)
            for i in (0, NUM_ELEMS // 3, 2 * NUM_ELEMS // 3, NUM_ELEMS - 1):
                exp = s * 1000 + i
                if got[i] != exp:
                    raise AssertionError(
                        f"rank {my_rank} recv[{s}][{i}]={got[i]} expected {exp}"
                    )
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


def _worker_put2d(rank, world_size, uid_path, coop_name, results_path):
    """2D put: a rect of the sender's matrix lands at an offset in the peer's slot.

    THREAD coop: all 64 lanes post one 1-row rect each to the same peer, which
    put2d batches into a single run. WARP/BLOCK: the leader posts the whole
    64-row rect as one packet. The whole slot is read back, so a stray write
    outside the rect fails as surely as a wrong value inside it."""
    err = None
    try:
        mods = _worker_imports()
        flyc, fx, Int64, cco = mods["flyc"], mods["fx"], mods["Int64"], mods["cco"]
        sync, fill, zero, read = mods["sync"], mods["fill"], mods["zero"], mods["read"]
        u32 = mods["ctypes"].c_uint32
        CCODevCommRequirements = mods["CCODevCommRequirements"]
        GDA_CONNECTION_NONE = mods["GDA_CONNECTION_NONE"]

        coop = getattr(cco.CoopScope, coop_name)
        comm = _init_comm(rank, world_size, uid_path)

        # src: S_ROWS x S_COLS u32; each dst slot: D_ROWS x D_COLS u32.
        S_ROWS, S_COLS, D_ROWS, D_COLS = 80, 32, 80, 48
        SY, SX, DY, DX, H, W = 3, 5, 2, 7, 64, 16  # rect origin/extent, u32 units
        s_pitch, d_pitch = S_COLS * 4, D_COLS * 4
        slot_bytes = D_ROWS * d_pitch
        send_off = 0
        recv_off = S_ROWS * s_pitch
        win_bytes = recv_off + world_size * slot_bytes

        mem = comm.alloc_mem(win_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        zero(win.local_ptr, win_bytes)

        def val(r, row, col):
            return r * 100000 + row * 100 + col + 1

        fill(
            win.local_ptr + send_off,
            [val(rank, y, x) for y in range(S_ROWS) for x in range(S_COLS)],
            u32,
        )

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 8
        dc = comm.create_dev_comm(reqs)

        my_rank = rank
        batch = coop == cco.CoopScope.THREAD
        src0 = send_off + (SY * S_COLS + SX) * 4
        dst0 = recv_off + my_rank * slot_bytes + (DY * D_COLS + DX) * 4

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def put2d_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            tid = fx.thread_idx.x
            for p in range(world_size):
                if p != my_rank:
                    if batch:
                        # lane i: row i of the rect, all lanes on (p, queue 0)
                        row = fx.Int64(tid)
                        sdma.put2d(
                            p,
                            w,
                            fx.Int64(dst0) + row * fx.Int64(d_pitch),
                            d_pitch,
                            w,
                            fx.Int64(src0) + row * fx.Int64(s_pitch),
                            s_pitch,
                            W * 4,
                            1,
                            0,
                            coop=coop,
                        )
                    else:
                        sdma.put2d(
                            p,
                            w,
                            dst0,
                            d_pitch,
                            w,
                            src0,
                            s_pitch,
                            W * 4,
                            H,
                            0,
                            coop=coop,
                        )
            if tid == 0:
                for p in range(world_size):
                    if p != my_rank:
                        sdma.quiet(p)

        @flyc.jit
        def run(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            put2d_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        comm.barrier()
        run(dc.ptr, win.handle)
        sync()
        comm.barrier()

        for s in range(world_size):
            if s == my_rank:
                continue
            got = read(win.local_ptr + recv_off + s * slot_bytes, D_ROWS * D_COLS, u32)
            for y in range(D_ROWS):
                for x in range(D_COLS):
                    inside = DY <= y < DY + H and DX <= x < DX + W
                    exp = val(s, SY + y - DY, SX + x - DX) if inside else 0
                    if got[y * D_COLS + x] != exp:
                        raise AssertionError(
                            f"rank {my_rank} slot[{s}] ({y},{x})="
                            f"{got[y * D_COLS + x]} expected {exp}"
                        )
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


def _worker_post_wait_mem(rank, world_size, uid_path, coop_name, results_path):
    """post_wait_mem parks a queue on a local flag: the put posted behind it must
    not move until the host raises the flag, and must all arrive afterwards.

    The kernel returns right after posting, so nothing but the copy engine is
    waiting. ``coop_name`` is the scope of the wait; the put is one thread's."""
    err = None
    try:
        import time

        mods = _worker_imports()
        flyc, fx, Int64, cco = mods["flyc"], mods["fx"], mods["Int64"], mods["cco"]
        sync, fill, zero, read = mods["sync"], mods["fill"], mods["zero"], mods["read"]
        u32 = mods["ctypes"].c_uint32
        CCODevCommRequirements = mods["CCODevCommRequirements"]
        GDA_CONNECTION_NONE = mods["GDA_CONNECTION_NONE"]

        coop = getattr(cco.CoopScope, coop_name)
        comm = _init_comm(rank, world_size, uid_path)
        # Window layout: [ send | recv (world_size slots) | flag ]
        send_off = 0
        recv_off = NBYTES
        flag_off = recv_off + world_size * NBYTES
        win_bytes = flag_off + 4096
        qid = 1  # a waiting queue gets its own queue

        mem = comm.alloc_mem(win_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        zero(win.local_ptr, win_bytes)
        fill(win.local_ptr + send_off, [rank * 1000 + i for i in range(NUM_ELEMS)], u32)

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 8
        dc = comm.create_dev_comm(reqs)

        me = rank
        nxt, prv = (me + 1) % world_size, (me - 1) % world_size
        gate_single = coop == cco.CoopScope.THREAD

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def post_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            # THREAD posts from one lane; WARP needs the whole wave to enter.
            active = fx.thread_idx.x == 0 if gate_single else fx.thread_idx.x < 64
            if active:
                sdma.post_wait_mem(nxt, qid, w, flag_off, 1, coop=coop)
            if fx.thread_idx.x == 0:
                sdma.put(
                    nxt,
                    w,
                    recv_off + me * NBYTES,
                    w,
                    send_off,
                    NBYTES,
                    qid,
                    signal=False,
                )

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def quiet_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            if fx.thread_idx.x == 0:
                sdma.quiet(nxt)

        @flyc.jit
        def post(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            post_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        @flyc.jit
        def drain(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            quiet_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        comm.barrier()
        post(dc.ptr, win.handle)
        sync()  # the kernel is done; only the engine is parked now
        comm.barrier()

        time.sleep(0.5)
        got = read(win.local_ptr + recv_off + prv * NBYTES, NUM_ELEMS, u32)
        if any(got[i] != 0 for i in range(NUM_ELEMS)):
            raise AssertionError(
                f"rank {me}: data from rank {prv} arrived before its flag was raised"
            )
        comm.barrier()

        fill(win.local_ptr + flag_off, [1], u32)  # release our own queue
        drain(dc.ptr, win.handle)
        sync()
        comm.barrier()

        got = read(win.local_ptr + recv_off + prv * NBYTES, NUM_ELEMS, u32)
        for i in range(NUM_ELEMS):
            if got[i] != prv * 1000 + i:
                raise AssertionError(
                    f"rank {me} recv[{prv}][{i}]={got[i]} expected {prv * 1000 + i}"
                )
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


def _worker_post_wait_signal(rank, world_size, uid_path, coop_name, results_path):
    """Engine-side relay: each rank forwards what it received from prev on to
    next once prev's signal lands, with no CU in the forwarding path.

    Queue 1 carries the hop (remote_signal bumps next's slot[me, 1]); queue 2
    waits on our slot[prev, 1] with post_wait_signal, then forwards the hop's data."""
    err = None
    try:
        mods = _worker_imports()
        flyc, fx, Int64, cco = mods["flyc"], mods["fx"], mods["Int64"], mods["cco"]
        sync, fill, zero, read = mods["sync"], mods["fill"], mods["zero"], mods["read"]
        u32 = mods["ctypes"].c_uint32
        CCODevCommRequirements = mods["CCODevCommRequirements"]
        GDA_CONNECTION_NONE = mods["GDA_CONNECTION_NONE"]

        comm = _init_comm(rank, world_size, uid_path)
        # Window layout: [ send | recv (world_size slots) | fwd (world_size slots) ]
        send_off = 0
        recv_off = NBYTES
        fwd_off = recv_off + world_size * NBYTES
        win_bytes = fwd_off + world_size * NBYTES
        hop_q, relay_q = 1, 2

        mem = comm.alloc_mem(win_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        zero(win.local_ptr, win_bytes)
        fill(win.local_ptr + send_off, [rank * 1000 + i for i in range(NUM_ELEMS)], u32)

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 8
        dc = comm.create_dev_comm(reqs)

        me = rank
        nxt, prv = (me + 1) % world_size, (me - 1) % world_size

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def relay_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            if fx.thread_idx.x == 0:
                # Park the relay queue until prev's hop has landed here...
                sdma.post_wait_signal(nxt, relay_q, prv, hop_q, 1)
                # ...then forward it.
                sdma.put(
                    nxt,
                    w,
                    fwd_off + me * NBYTES,
                    w,
                    recv_off + prv * NBYTES,
                    NBYTES,
                    relay_q,
                    signal=False,
                )
                # The hop: our data to next, signalling its slot[me, hop_q].
                sdma.put(
                    nxt,
                    w,
                    recv_off + me * NBYTES,
                    w,
                    send_off,
                    NBYTES,
                    hop_q,
                    local_signal=False,
                    remote_signal=True,
                )
                sdma.quiet(nxt)

        @flyc.jit
        def run(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            relay_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        comm.barrier()
        run(dc.ptr, win.handle)
        sync()
        comm.barrier()

        # fwd slot [prv] holds what prv received from prv's own prev.
        origin = (prv - 1) % world_size
        for what, off, src in (
            ("hop", recv_off + prv * NBYTES, prv),
            ("relay", fwd_off + prv * NBYTES, origin),
        ):
            got = read(win.local_ptr + off, NUM_ELEMS, u32)
            for i in range(NUM_ELEMS):
                if got[i] != src * 1000 + i:
                    raise AssertionError(
                        f"rank {me} {what}[{i}]={got[i]} expected {src * 1000 + i}"
                    )
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


def _worker_put_broadcast(rank, world_size, uid_path, coop_name, results_path):
    """put_broadcast: one source lands in next's slot and in our own local copy
    from a single packet; both signals arrive."""
    err = None
    try:
        mods = _worker_imports()
        flyc, fx, Int64, cco = mods["flyc"], mods["fx"], mods["Int64"], mods["cco"]
        sync, fill, zero, read = mods["sync"], mods["fill"], mods["zero"], mods["read"]
        u32 = mods["ctypes"].c_uint32
        CCODevCommRequirements = mods["CCODevCommRequirements"]
        GDA_CONNECTION_NONE = mods["GDA_CONNECTION_NONE"]

        coop = getattr(cco.CoopScope, coop_name)
        comm = _init_comm(rank, world_size, uid_path)
        # Window layout: [ send | recv (world_size slots) | local copy ]
        send_off = 0
        recv_off = NBYTES
        local_off = recv_off + world_size * NBYTES
        win_bytes = local_off + NBYTES

        mem = comm.alloc_mem(win_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        zero(win.local_ptr, win_bytes)
        fill(win.local_ptr + send_off, [rank * 1000 + i for i in range(NUM_ELEMS)], u32)

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 8
        dc = comm.create_dev_comm(reqs)

        me = rank
        nxt, prv = (me + 1) % world_size, (me - 1) % world_size
        gate_single = coop == cco.CoopScope.THREAD

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def bcast_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            active = fx.thread_idx.x == 0 if gate_single else fx.thread_idx.x < 64
            if active:
                sdma.put_broadcast(
                    nxt,
                    w,
                    recv_off + me * NBYTES,
                    me,
                    w,
                    local_off,
                    w,
                    send_off,
                    NBYTES,
                    0,
                    coop=coop,
                    local_signal=True,
                    remote_signal=True,
                )
                # Ours (local signal) and prv's (its remote signal to us).
                sdma.wait_signal(me, 0, 1)
                sdma.wait_signal(prv, 0, 1)
                sdma.quiet(nxt, coop=coop)

        @flyc.jit
        def run(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            bcast_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        comm.barrier()
        run(dc.ptr, win.handle)
        sync()
        comm.barrier()

        for what, off, src in (
            ("remote", recv_off + prv * NBYTES, prv),
            ("local", local_off, me),
        ):
            got = read(win.local_ptr + off, NUM_ELEMS, u32)
            for i in range(NUM_ELEMS):
                if got[i] != src * 1000 + i:
                    raise AssertionError(
                        f"rank {me} {what}[{i}]={got[i]} expected {src * 1000 + i}"
                    )
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


_STALE_RPTR_SRC = r"""
#include "mori/cco/cco.hpp"
using namespace mori::cco;
extern "C" __global__ void put(ccoDevComm* dc, int peer, ccoWindow_t w, void** host) {
  ccoSdma sdma{*dc};
  sdma.put<ccoCoopThread, true>(peer, w, 0, w, 0, 256);
  sdma.waitSignal(dc->lsaRank, 0, 1);  // the copy has run; retiredWptr has not moved, so quiet must post a drain
  host[0] = dc->sdma.deviceHandles[peer * dc->sdma.sdmaNumQueue]->rptr;
}
extern "C" __global__ void quiet(ccoDevComm* dc, int peer, void** host) {
  ccoSdma{*dc}.quiet(peer);
  host[1] = host;
}
extern "C" __global__ void quietQueue(ccoDevComm* dc, int peer, void** host) {
  ccoSdma{*dc}.quietQueue(peer, 0);
  host[1] = host;
}
"""


def _worker_quiet_stale_rptr(rank, world_size, uid_path, quiet, results_path):
    """#877: quiet must not poll the queue's read pointer (host memory): make it
    stale while the queue is idle, and a quiet that polls it never returns."""
    err = None
    try:
        import ctypes
        import subprocess
        import time

        from mori.cco import CCODevCommRequirements, GDA_CONNECTION_NONE
        from mori.jit.config import detect_build_config, get_mori_source_root
        from mori.jit.hip_driver import HipModule

        comm = _init_comm(rank, world_size, uid_path)
        win = comm.register_window(comm.alloc_mem(NBYTES).ptr, NBYTES)
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.sdma_queue_count = 1
        dc = comm.create_dev_comm(reqs)
        cfg, tmp = detect_build_config(), tempfile.mkdtemp()
        src, co = os.path.join(tmp, "k.hip"), os.path.join(tmp, "k.co")
        with open(src, "w") as f:
            f.write(_STALE_RPTR_SRC)
        inc = os.path.join(str(get_mori_source_root()), "include")
        subprocess.check_call(
            [cfg.hipcc, "--genco", f"--offload-arch={cfg.arch}", "-DBUILD_CCO_SDMA=1"]
            + ["-I", inc, src, "-o", co]
        )
        fn, nxt = HipModule(co).get_function, (rank + 1) % world_size
        host = torch.zeros(2, dtype=torch.int64, pin_memory=True)
        fn("put").launch((1,), (1,), 0, 0, dc.ptr, nxt, win.handle, host.data_ptr())
        torch.cuda.synchronize()
        time.sleep(0.05)  # let the engine's read-pointer write-back land
        rptr = ctypes.c_uint64.from_address(int(host[0]))
        true_rptr = rptr.value
        rptr.value = true_rptr - 32
        fn(quiet).launch((1,), (1,), 0, 0, dc.ptr, nxt, host.data_ptr())
        time.sleep(1.0)
        returned = host[1].item() != 0
        rptr.value = max(rptr.value, true_rptr)  # lets a quiet that polls it finish
        torch.cuda.synchronize()
        comm.barrier()
        assert returned, f"rank {rank}: {quiet} waited on a stale read pointer"
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


def _signal_snapshot_kernel(flyc, fx, Int64, cco, slots, snap_off, me):
    """A kernel that copies the low 32 bits of each ``(src_rank, qid)`` signal
    slot into ``win[snap_off + 4 * i]``, so the host can check exact counts."""
    from mori.cco.device.flydsl.signals import (
        local_store_u32,
        signal_load_u32,
        signal_ptr,
    )

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def snapshot_kernel(dev_comm: Int64, w: Int64):
        sdma = cco.DevComm(dev_comm).sdma()
        window = cco.Window(w)
        if fx.thread_idx.x == 0:
            for i, (src, qid) in enumerate(slots):
                v = signal_load_u32(signal_ptr(fx.Int64(sdma.signal_slot(src, qid))))
                local_store_u32(
                    signal_ptr(fx.Int64(window.lsa_ptr(me, snap_off + 4 * i))), v
                )

    @flyc.jit
    def snapshot(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
        snapshot_kernel(dev_comm, w).launch(
            grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
        )

    return snapshot


def _worker_post_wait_mem_put(rank, world_size, uid_path, variant, results_path):
    """post_wait_mem_put_ptr holds each copy and its signals until its flag is
    raised, then delivers each exactly once; the posting kernel exits at once.

    One lane: ``plain`` rings, ``aggregate`` leaves the doorbell to ``commit``,
    ``zero_bytes`` posts wait + signals only. Four lanes: ``distinct`` (lane l
    on queue l behind flag l, released one at a time) and ``shared`` (one queue
    and one flag: the serialized path); ``aggregate_distinct`` and
    ``aggregate_shared`` are the same with the doorbell left to ``commit``."""
    err = None
    try:
        import time

        mods = _worker_imports()
        flyc, fx, Int64, cco = mods["flyc"], mods["fx"], mods["Int64"], mods["cco"]
        sync, fill, zero, read = mods["sync"], mods["fill"], mods["zero"], mods["read"]
        u32 = mods["ctypes"].c_uint32
        CCODevCommRequirements = mods["CCODevCommRequirements"]
        GDA_CONNECTION_NONE = mods["GDA_CONNECTION_NONE"]

        lanes = 1 if variant in ("plain", "aggregate", "zero_bytes") else 4
        distinct = variant.endswith("distinct")
        aggregate = variant.startswith("aggregate")
        nbytes = 0 if variant == "zero_bytes" else NBYTES
        # distinct posts <remote only>, the other variants <local + remote>.
        local = not distinct
        # distinct: lane l -> queue l, flag l. Otherwise every lane -> queue 1, flag 0.
        q_stride, q_base = (1, 0) if distinct else (0, 1)
        queues = list(range(lanes)) if distinct else [1]

        comm = _init_comm(rank, world_size, uid_path)
        # Window layout: [ send (lanes) | recv (world_size x lanes) | flags | snapshot ]
        send_off = 0
        recv_off = lanes * NBYTES
        flag_off = recv_off + world_size * lanes * NBYTES
        snap_off = flag_off + 256
        win_bytes = snap_off + 256

        mem = comm.alloc_mem(win_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        zero(win.local_ptr, win_bytes)

        def value(src, lane, i):
            return src * 10000 + lane * 1000 + i

        for lane in range(lanes):
            fill(
                win.local_ptr + send_off + lane * NBYTES,
                [value(rank, lane, i) for i in range(NUM_ELEMS)],
                u32,
            )

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 8
        dc = comm.create_dev_comm(reqs)

        me = rank
        nxt, prv = (me + 1) % world_size, (me - 1) % world_size

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def post_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            window = cco.Window(w)
            tid = fx.thread_idx.x
            lane_bytes = fx.Int64(tid) * fx.Int64(NBYTES)
            dst = fx.Int64(window.lsa_ptr(nxt, recv_off + me * lanes * NBYTES))
            src = fx.Int64(window.lsa_ptr(me, send_off))
            q = fx.Int32(tid) * fx.Int32(q_stride) + fx.Int32(q_base)
            flag = fx.Int64(flag_off) + fx.Int64(tid) * fx.Int64(4 * q_stride)
            if tid < lanes:
                sdma.post_wait_mem_put_ptr(
                    nxt,
                    q,
                    w,
                    flag,
                    1,
                    dst + lane_bytes,
                    src + lane_bytes,
                    nbytes,
                    local_signal=local,
                    remote_signal=True,
                    aggregate=aggregate,
                )
                if aggregate:
                    if distinct:
                        sdma.commit(nxt, q)  # each lane rings its own queue
                    elif tid == 0:
                        sdma.commit(nxt, q_base)  # one ring for the shared queue

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def quiet_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            if fx.thread_idx.x == 0:
                sdma.quiet(nxt)

        @flyc.jit
        def post(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            post_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        @flyc.jit
        def drain(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            quiet_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        # Local signals of our puts (if posted), then remote signals of prv's puts.
        slots = [(me, q) for q in queues] if local else []
        slots += [(prv, q) for q in queues]
        snapshot = _signal_snapshot_kernel(flyc, fx, Int64, cco, slots, snap_off, me)

        def check(landed):
            got = read(
                win.local_ptr + recv_off + prv * lanes * NBYTES,
                lanes * NUM_ELEMS,
                u32,
            )
            for lane in range(lanes):
                for i in range(NUM_ELEMS):
                    exp = value(prv, lane, i) if lane in landed and nbytes else 0
                    if got[lane * NUM_ELEMS + i] != exp:
                        raise AssertionError(
                            f"rank {me} lane {lane} recv[{i}]="
                            f"{got[lane * NUM_ELEMS + i]} expected {exp} "
                            f"(landed lanes {sorted(landed)})"
                        )
            snapshot(dc.ptr, win.handle)
            sync()
            counts = list(read(win.local_ptr + snap_off, len(slots), u32))
            if distinct:
                want = [1 if q in landed else 0 for q in queues]
            else:
                want = [len(landed)] * 2
            if counts != want:
                raise AssertionError(
                    f"rank {me}: signals per queue {counts}, "
                    f"expected {want} (landed lanes {sorted(landed)})"
                )

        comm.barrier()
        post(dc.ptr, win.handle)
        sync()  # the kernel is done; only the engine is parked now
        comm.barrier()
        time.sleep(0.5)
        check(set())
        comm.barrier()

        if distinct:
            # Release lane 2 alone: only its copy may move.
            fill(win.local_ptr + flag_off + 4 * 2, [1], u32)
            time.sleep(0.5)
            comm.barrier()
            check({2})
            comm.barrier()
        fill(win.local_ptr + flag_off, [1] * lanes, u32)
        drain(dc.ptr, win.handle)
        sync()
        comm.barrier()
        check(set(range(lanes)))
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


# (cmp, mask, value, flag that must hold the copy, flag that must release it).
# The holding flag passes the compare if the mask is ignored, so a mask that is
# not applied releases the copy early.
_CMP_CASES = {
    "ge_mask": ("GE", 0x00FF0000, 0x00030000, 0xFF02FFFF, 0x00050000),
    "ne_mask": ("NE", 0x0000FFFF, 0x1234, 0xABCD1234, 0xABCD1235),
}


def _worker_post_wait_mem_put_cmp(rank, world_size, uid_path, variant, results_path):
    """A non-EQ compare under a partial mask: the copy is held while
    ``(flag & mask) <cmp> value`` is false, although the unmasked flag would
    pass, and runs once it holds."""
    err = None
    try:
        import time

        mods = _worker_imports()
        flyc, fx, Int64, cco = mods["flyc"], mods["fx"], mods["Int64"], mods["cco"]
        sync, fill, zero, read = mods["sync"], mods["fill"], mods["zero"], mods["read"]
        u32 = mods["ctypes"].c_uint32
        CCODevCommRequirements = mods["CCODevCommRequirements"]
        GDA_CONNECTION_NONE = mods["GDA_CONNECTION_NONE"]
        cmp_name, mask, value, hold, release = _CMP_CASES[variant]
        cmp = getattr(cco.SdmaCmp, cmp_name)
        QID = 1

        comm = _init_comm(rank, world_size, uid_path)
        # Window layout: [ send | recv | flag | snapshot ]
        send_off, recv_off = 0, NBYTES
        flag_off = 2 * NBYTES
        snap_off = flag_off + 256
        win_bytes = snap_off + 256
        mem = comm.alloc_mem(win_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        zero(win.local_ptr, win_bytes)
        fill(win.local_ptr, [rank * 1000 + i for i in range(NUM_ELEMS)], u32)
        fill(win.local_ptr + flag_off, [hold], u32)

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 2
        dc = comm.create_dev_comm(reqs)
        me = rank
        nxt, prv = (me + 1) % world_size, (me - 1) % world_size

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def post_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            window = cco.Window(w)
            dst = fx.Int64(window.lsa_ptr(nxt, recv_off))
            src = fx.Int64(window.lsa_ptr(me, send_off))
            if fx.thread_idx.x == 0:
                sdma.post_wait_mem_put_ptr(
                    nxt,
                    QID,
                    w,
                    flag_off,
                    value,
                    dst,
                    src,
                    NBYTES,
                    cmp=cmp,
                    mask=mask,
                    local_signal=True,
                    remote_signal=True,
                )

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def quiet_kernel(dev_comm: Int64, w: Int64):
            if fx.thread_idx.x == 0:
                cco.DevComm(dev_comm).sdma().quiet(nxt)

        @flyc.jit
        def post(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            post_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        @flyc.jit
        def drain(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            quiet_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        slots = [(me, QID), (prv, QID)]
        snapshot = _signal_snapshot_kernel(flyc, fx, Int64, cco, slots, snap_off, me)

        def check(landed):
            got = list(read(win.local_ptr + recv_off, NUM_ELEMS, u32))
            exp = [prv * 1000 + i if landed else 0 for i in range(NUM_ELEMS)]
            snapshot(dc.ptr, win.handle)
            sync()
            counts = list(read(win.local_ptr + snap_off, 2, u32))
            want = [int(landed)] * 2
            if got != exp or counts != want:
                raise AssertionError(
                    f"rank {me} {variant} flag {'released' if landed else 'held'}:"
                    f" recv[:4]={got[:4]} expected {exp[:4]}, signals {counts}"
                    f" expected {want}"
                )

        comm.barrier()
        post(dc.ptr, win.handle)
        sync()
        comm.barrier()
        time.sleep(0.5)
        try:
            check(False)
        finally:
            comm.barrier()
            fill(win.local_ptr + flag_off, [release], u32)
            drain(dc.ptr, win.handle)
            sync()
            comm.barrier()
        check(True)
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


def _worker_post_wait_mem_put_nothing(
    rank, world_size, uid_path, variant, results_path
):
    """``nbytes == 0`` with no signal posts nothing: a put posted behind it on
    the same queue lands although the flag is never raised."""
    err = None
    try:
        import time

        mods = _worker_imports()
        flyc, fx, Int64, cco = mods["flyc"], mods["fx"], mods["Int64"], mods["cco"]
        sync, fill, zero, read = mods["sync"], mods["fill"], mods["zero"], mods["read"]
        u32 = mods["ctypes"].c_uint32
        CCODevCommRequirements = mods["CCODevCommRequirements"]
        GDA_CONNECTION_NONE = mods["GDA_CONNECTION_NONE"]
        QID = 1

        comm = _init_comm(rank, world_size, uid_path)
        send_off, recv_off, flag_off = 0, NBYTES, 2 * NBYTES
        win_bytes = flag_off + 256
        mem = comm.alloc_mem(win_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        zero(win.local_ptr, win_bytes)
        fill(win.local_ptr, [rank * 1000 + i for i in range(NUM_ELEMS)], u32)

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 2
        dc = comm.create_dev_comm(reqs)
        me = rank
        nxt, prv = (me + 1) % world_size, (me - 1) % world_size

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def post_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            window = cco.Window(w)
            dst = fx.Int64(window.lsa_ptr(nxt, recv_off))
            src = fx.Int64(window.lsa_ptr(me, send_off))
            if fx.thread_idx.x == 0:
                # Flag stays 0, so a posted wait would park the queue for good.
                sdma.post_wait_mem_put_ptr(nxt, QID, w, flag_off, 1, dst, src, 0)
                sdma.put_ptr(nxt, dst, src, NBYTES, QID)

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def quiet_kernel(dev_comm: Int64, w: Int64):
            if fx.thread_idx.x == 0:
                cco.DevComm(dev_comm).sdma().quiet(nxt)

        @flyc.jit
        def post(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            post_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        @flyc.jit
        def drain(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            quiet_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        comm.barrier()
        post(dc.ptr, win.handle)
        sync()
        comm.barrier()
        time.sleep(0.5)
        got = list(read(win.local_ptr + recv_off, NUM_ELEMS, u32))
        comm.barrier()
        # Unpark a queue the old behaviour would have left waiting, then drain.
        fill(win.local_ptr + flag_off, [1], u32)
        drain(dc.ptr, win.handle)
        sync()
        comm.barrier()
        exp = [prv * 1000 + i for i in range(NUM_ELEMS)]
        if got != exp:
            raise AssertionError(
                f"rank {me}: put behind an empty wait+put did not land with the flag"
                f" low: recv[:4]={got[:4]} expected {exp[:4]}"
            )
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


# CCO_SDMA_MAX_COPY_BYTES (cco.hpp): one COPY_LINEAR packet's limit.
_SDMA_MAX_COPY_BYTES = 0x3FFFFFE0
_OVERSIZE_BYTES = _SDMA_MAX_COPY_BYTES + 4096  # two chunks per lane


def _worker_post_wait_mem_put_oversize(
    rank, world_size, uid_path, variant, results_path
):
    """Copies over CCO_SDMA_MAX_COPY_BYTES take the composed fallback (wait +
    chunked put + commit), one lane at a time. Two lanes on ONE queue behind two
    flags: raising the first-posted lane's flag moves its whole copy (both
    chunks) and its signals, and nothing of the other lane's."""
    err = None
    try:
        import time

        from mori.tensor_utils import from_gpu_ptr

        mods = _worker_imports()
        flyc, fx, Int64, cco = mods["flyc"], mods["fx"], mods["Int64"], mods["cco"]
        sync, fill, read = mods["sync"], mods["fill"], mods["read"]
        u32 = mods["ctypes"].c_uint32
        CCODevCommRequirements = mods["CCODevCommRequirements"]
        GDA_CONNECTION_NONE = mods["GDA_CONNECTION_NONE"]
        LANES, QID, NB = 2, 1, _OVERSIZE_BYTES

        # Window layout: [ send (LANES x NB) | recv (LANES x NB) | flags | snapshot ]
        send_off, recv_off = 0, LANES * NB
        flag_off = 2 * LANES * NB
        snap_off = flag_off + 256
        win_bytes = snap_off + 256
        comm = _init_comm(
            rank, world_size, uid_path, per_rank_vmm=win_bytes + (1 << 30)
        )
        mem = comm.alloc_mem(win_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        ints = NB // 4
        whole = from_gpu_ptr(win.local_ptr, (win_bytes // 4,), torch.int32)
        whole.zero_()

        def pattern(src, lane):
            x = torch.arange(ints, device="cuda", dtype=torch.int32)
            return x * 7 + src * 1000003 + lane * 7919

        def region(off, lane):
            return whole[(off + lane * NB) // 4 : (off + (lane + 1) * NB) // 4]

        for lane in range(LANES):
            region(send_off, lane).copy_(pattern(rank, lane))
        torch.cuda.synchronize()

        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = 2
        dc = comm.create_dev_comm(reqs)
        me = rank
        nxt, prv = (me + 1) % world_size, (me - 1) % world_size

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def post_kernel(dev_comm: Int64, w: Int64):
            sdma = cco.DevComm(dev_comm).sdma()
            window = cco.Window(w)
            tid = fx.thread_idx.x
            lane_bytes = fx.Int64(tid) * fx.Int64(NB)
            dst = fx.Int64(window.lsa_ptr(nxt, recv_off)) + lane_bytes
            src = fx.Int64(window.lsa_ptr(me, send_off)) + lane_bytes
            flag = fx.Int64(flag_off) + fx.Int64(tid) * fx.Int64(4)
            if tid < LANES:
                sdma.post_wait_mem_put_ptr(
                    nxt,
                    QID,
                    w,
                    flag,
                    1,
                    dst,
                    src,
                    NB,
                    local_signal=True,
                    remote_signal=True,
                )

        @flyc.kernel(known_block_size=[THREADS, 1, 1])
        def quiet_kernel(dev_comm: Int64, w: Int64):
            if fx.thread_idx.x == 0:
                cco.DevComm(dev_comm).sdma().quiet(nxt)

        @flyc.jit
        def post(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            post_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        @flyc.jit
        def drain(dev_comm: Int64, w: Int64, stream=fx.Stream(None)):
            quiet_kernel(dev_comm, w).launch(
                grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
            )

        slots = [(me, QID), (prv, QID)]
        snapshot = _signal_snapshot_kernel(flyc, fx, Int64, cco, slots, snap_off, me)

        def state():
            """(lanes whose block landed whole, lanes with any byte landed, signals)."""
            whole_lanes, touched = set(), set()
            for lane in range(LANES):
                got = region(recv_off, lane)
                if torch.equal(got, pattern(prv, lane)):
                    whole_lanes.add(lane)
                if bool(torch.count_nonzero(got)):
                    touched.add(lane)
            snapshot(dc.ptr, win.handle)
            sync()
            return whole_lanes, touched, list(read(win.local_ptr + snap_off, 2, u32))

        def expect(landed, what):
            deadline = time.monotonic() + 30.0
            while True:
                whole_lanes, touched, counts = state()
                ok = whole_lanes == landed and touched == landed
                ok = ok and counts == [len(landed)] * 2
                if ok or time.monotonic() > deadline:
                    break
                time.sleep(0.05)
            if not ok:
                raise AssertionError(
                    f"rank {me} {what}: whole {sorted(whole_lanes)} touched"
                    f" {sorted(touched)} signals {counts}, expected lanes"
                    f" {sorted(landed)} and signals {[len(landed)] * 2}"
                )

        def raise_flags(lanes):
            for lane in lanes:
                fill(win.local_ptr + flag_off + 4 * lane, [1], u32)

        comm.barrier()
        post(dc.ptr, win.handle)
        sync()
        comm.barrier()
        time.sleep(0.5)
        try:
            expect(set(), "both flags low")
            comm.barrier()
            # Lane 0 posted first. Only its flag: its copy must not wait for lane 1's.
            raise_flags([0])
            expect({0}, "lane 0's flag raised")
            comm.barrier()
        finally:
            raise_flags(range(LANES))  # never leave a queue parked
            drain(dc.ptr, win.handle)
            sync()
            comm.barrier()
        expect({0, 1}, "both flags raised")
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


# ── spawn harness ──────────────────────────────────────────────────────────


def _run(worker, world_size, coop_name):
    _require_gpus(world_size)
    import mori.cco.device.flydsl  # noqa: F401  (fail early if flydsl broken)
    from mori.cco import Communicator

    os.environ["MORI_ENABLE_SDMA"] = "1"
    os.environ.setdefault("MORI_SOCKET_IFNAME", "lo")

    tmp = tempfile.mkdtemp(prefix="cco_sdma_test_")
    uid_path = os.path.join(tmp, "uid.bin")
    results_path = os.path.join(tmp, "result_%d.txt")

    uid = Communicator.get_unique_id()
    with open(uid_path, "wb") as f:
        f.write(bytes(uid))

    torch.multiprocessing.spawn(
        _spawn_entry,
        args=(worker, world_size, uid_path, coop_name, results_path),
        nprocs=world_size,
        join=True,
    )

    errs = []
    for rank in range(world_size):
        with open(results_path % rank) as f:
            e = f.read()
        if e:
            errs.append(f"rank {rank}:\n{e}")
    assert not errs, "\n\n".join(errs)


def _spawn_entry(rank, worker, world_size, uid_path, coop_name, results_path):
    worker(rank, world_size, uid_path, coop_name, results_path)


# ── tests ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("coop_name", ["THREAD", "WARP"])
def test_sdma_put_alltoall(coop_name, world_size):
    """SDMA put moves the exact source bytes into each peer's slot; quiet drains.

    Scales across 2/4/8 GPUs; cases needing more GPUs than present are skipped."""
    _run(_worker_put, world_size=world_size, coop_name=coop_name)


def test_sdma_quiet_queue():
    """Single-queue completion (quiet_queue) drains a thread-coop put correctly."""
    _run(_worker_put_quiet_queue, world_size=2, coop_name="THREAD")


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_sdma_put_alltoall_block(world_size):
    """BLOCK-coop put/quiet: a multi-warp block drives the queues correctly."""
    _run(_worker_put_block, world_size=world_size, coop_name="BLOCK")


def test_sdma_signals():
    """Local and remote signals of put / put_ptr / get_ptr reach wait_signal."""
    _run(_worker_signals, world_size=2, coop_name="THREAD")


def test_sdma_put_nosignal():
    """No-signal put lands too, drained by a trailing signaled put on the same queue."""
    _run(_worker_put_nosignal, world_size=2, coop_name="THREAD")


def test_sdma_put_aggregate():
    """Aggregate puts batch into the SQ; one commit rings the doorbell for all."""
    _run(_worker_put_aggregate, world_size=2, coop_name="THREAD")


@pytest.mark.parametrize("world_size", [2, 8])
@pytest.mark.parametrize("coop_name", ["THREAD", "WARP", "BLOCK"])
def test_sdma_put2d(coop_name, world_size):
    """put2d moves exactly the rect; THREAD batches 64 one-row rects in one run."""
    _run(_worker_put2d, world_size=world_size, coop_name=coop_name)


@pytest.mark.parametrize("coop_name", ["THREAD", "WARP"])
def test_sdma_post_wait_mem(coop_name):
    """An engine-side wait holds the put behind it until the flag is raised."""
    _run(_worker_post_wait_mem, world_size=2, coop_name=coop_name)


@pytest.mark.parametrize("world_size", [2, 4])
def test_sdma_post_wait_signal_relay(world_size):
    """post_wait_signal forwards a hop's data once its remote signal lands."""
    _run(_worker_post_wait_signal, world_size=world_size, coop_name="THREAD")


@pytest.mark.parametrize("api", ["quiet", "quietQueue"])
def test_sdma_quiet_ignores_host_rptr(api):
    # _run passes the API name through its coop_name slot
    _run(_worker_quiet_stale_rptr, world_size=2, coop_name=api)


@pytest.mark.parametrize("coop_name", ["THREAD", "WARP"])
def test_sdma_put_broadcast(coop_name):
    """put_broadcast writes both destinations from one packet, with both signals."""
    _run(_worker_put_broadcast, world_size=2, coop_name=coop_name)


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("variant", ["plain", "aggregate", "zero_bytes"])
def test_sdma_post_wait_mem_put_ptr(variant, world_size):
    """The fused wait + put holds the copy and its signals until the flag is raised."""
    _run(_worker_post_wait_mem_put, world_size=world_size, coop_name=variant)


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize(
    "layout", ["distinct", "shared", "aggregate_distinct", "aggregate_shared"]
)
def test_sdma_post_wait_mem_put_ptr_lanes(layout, world_size):
    """Lanes on distinct queues wait independently; lanes on one queue in order."""
    _run(_worker_post_wait_mem_put, world_size=world_size, coop_name=layout)


@pytest.mark.parametrize("variant", sorted(_CMP_CASES))
def test_sdma_post_wait_mem_put_ptr_cmp_mask(variant):
    """A non-EQ compare under a partial mask gates the copy on the masked bits."""
    _run(_worker_post_wait_mem_put_cmp, world_size=2, coop_name=variant)


def test_sdma_post_wait_mem_put_ptr_nothing_to_post():
    """nbytes == 0 without signals posts no wait: the queue is not parked."""
    _run(_worker_post_wait_mem_put_nothing, world_size=2, coop_name="zero_nosignal")


def test_sdma_post_wait_mem_put_ptr_oversize_shared_queue():
    """Over CCO_SDMA_MAX_COPY_BYTES: two lanes on one queue, each copy gated only
    by its own flag (wait and chunks posted lane by lane). ~4.3 GiB per GPU."""
    need = 4 * _OVERSIZE_BYTES + (2 << 30)
    for i in range(min(2, _num_gpus())):
        if torch.cuda.get_device_properties(i).total_memory < need:
            pytest.skip(f"needs {need >> 30} GiB per GPU")
    _run(_worker_post_wait_mem_put_oversize, world_size=2, coop_name="shared_queue")
