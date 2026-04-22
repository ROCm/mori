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
"""Single-Process Multi-Thread (SPMT) shmem tests.

Each test spawns one Python thread per GPU.  Threads call ShmemInit via the
socket-bootstrap UniqueId path (no MPI, no torch.distributed).  This is the
JAX/XLA model: one process, N threads, each thread owns one GPU.

Requires MORI_MULTITHREAD_SUPPORT to be compiled in.
"""
import threading
import traceback

import pytest
import torch

import mori.shmem as shmem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_num_gpus() -> int:
    return torch.cuda.device_count()


def _thread_worker(
    rank: int,
    world_size: int,
    unique_id: bytes,
    barrier: threading.Barrier,
    results: list,
):
    """Per-GPU thread body: init, verify, malloc/free, barrier, finalize."""
    error = None
    try:
        torch.cuda.set_device(rank)

        # Phase 1: all threads have their device set → init together
        barrier.wait()

        ret = shmem.shmem_init_attr(
            shmem.MORI_SHMEM_INIT_WITH_UNIQUEID, rank, world_size, unique_id
        )
        assert ret == 0, f"shmem_init_attr failed: {ret}"

        my_pe = shmem.shmem_mype()
        n_pes = shmem.shmem_npes()
        assert my_pe == rank, f"pe mismatch: expected {rank}, got {my_pe}"
        assert n_pes == world_size, f"npes mismatch: expected {world_size}, got {n_pes}"

        # Phase 2: all inited → barrier
        barrier.wait()
        shmem.shmem_barrier_all()

        # Phase 3: symmetric malloc / free
        ptr = shmem.shmem_malloc(4096)
        assert ptr != 0, "shmem_malloc returned NULL"
        barrier.wait()
        shmem.shmem_barrier_all()
        shmem.shmem_free(ptr)

        # Phase 4: finalize
        barrier.wait()
        shmem.shmem_finalize()

    except Exception:
        error = traceback.format_exc()

    results[rank] = error


def _run_spmt(world_size: int):
    """Spawn `world_size` threads and run SPMT shmem init/finalize cycle."""
    import os

    num_gpus = _get_num_gpus()
    if world_size > num_gpus:
        pytest.skip(f"Need {world_size} GPUs, only {num_gpus} available")

    # Set interface before UniqueId is generated so that the socket bootstrap
    # uses the same interface for both root binding and peer connections.
    os.environ["MORI_SOCKET_IFNAME"] = "lo"

    unique_id = shmem.shmem_get_unique_id()

    barrier = threading.Barrier(world_size)
    results = [None] * world_size
    threads = [
        threading.Thread(
            target=_thread_worker,
            args=(rank, world_size, unique_id, barrier, results),
            daemon=True,
        )
        for rank in range(world_size)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive(), f"Thread {t.name} timed out"

    for rank, err in enumerate(results):
        assert err is None, f"Thread {rank} failed:\n{err}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_spmt_shmem_init_finalize(world_size):
    """SPMT: N threads each call ShmemInit + ShmemFinalize on their own GPU."""
    _run_spmt(world_size)
