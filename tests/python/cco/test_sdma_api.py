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

    from helper import sync, fill, zero, read

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


def _init_comm(rank, world_size, uid_path):
    """Common setup: set device, load shared UniqueId, create Communicator."""
    from mori.cco import Communicator, UniqueId

    torch.cuda.set_device(rank)
    with open(uid_path, "rb") as f:
        uid = UniqueId.from_bytes(f.read())
    comm = Communicator.init(world_size, rank, uid, per_rank_vmm=128 * 1024 * 1024)
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

    # This dir holds helper.py; spawned children need it on sys.path to import it.
    this_dir = os.path.dirname(os.path.abspath(__file__))

    torch.multiprocessing.spawn(
        _spawn_entry,
        args=(worker, world_size, uid_path, coop_name, results_path, this_dir),
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


def _spawn_entry(rank, worker, world_size, uid_path, coop_name, results_path, this_dir):
    import sys

    sys.path.insert(0, this_dir)  # so the child can import helper
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
