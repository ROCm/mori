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
import ctypes
import os
import struct
import tempfile
import time
import traceback

import pytest
import torch

NBYTES = 64 * 1024
UNIT_BYTES = 32  # ring bytes per SDMA packet
SDMA_OP_POLL_REGMEM = 8
POLL_MEM_POLL = 1 << 31
GATE_OFF = 0
ARRIVE_OFF = 4096
DATA_OFF = 8192


def _require_gpus(world_size: int) -> None:
    n = torch.cuda.device_count()
    if n < world_size:
        pytest.skip(f"Need {world_size} GPUs, only {n} available")


def _worker(rank, world_size, uid_path, results_path):
    err = None
    try:
        import numpy as np

        from mori.cco import (
            Communicator,
            TriggerBatch,
            TriggerQueue,
            UniqueId,
            peer_ptr,
        )
        from mori.cco.device.ops import SdmaCmp

        from .helper import fill, read, zero

        u32, u64 = ctypes.c_uint32, ctypes.c_uint64
        torch.cuda.set_device(rank)
        with open(uid_path, "rb") as f:
            uid = UniqueId.from_bytes(f.read())
        comm = Communicator.init(world_size, rank, uid, per_rank_vmm=128 * 1024 * 1024)

        win_bytes = DATA_OFF + 2 * world_size * NBYTES
        mem = comm.alloc_mem(win_bytes)
        win = comm.register_window(mem.ptr, mem.size)
        base = win.local_ptr
        zero(base, win_bytes)

        def gate(p):
            return base + GATE_OFF + 64 * p

        def arrive(s):
            return base + ARRIVE_OFF + 64 * s

        def send(p):
            return base + DATA_OFF + p * NBYTES

        def recv(s):
            return base + DATA_OFF + (world_size + s) * NBYTES

        words = NBYTES // 4
        for p in range(world_size):
            fill(send(p), [(rank << 24) ^ (p << 16) ^ i for i in range(words)], u32)

        assert peer_ptr(comm, base, rank) == base
        peers = [p for p in range(world_size) if p != rank]
        assert len({peer_ptr(comm, base, p) for p in peers}) == len(peers)

        queues = {p: TriggerQueue(comm, p) for p in peers}
        ends = {}
        comm.barrier()
        for p, q in queues.items():
            start = q.state()[1]
            end = q.append(
                gate(p),
                1,
                [(send(p), peer_ptr(comm, recv(rank), p), NBYTES)],
                peer_ptr(comm, arrive(rank), p),
            )
            assert end == start + 3 * UNIT_BYTES, (start, end)  # POLL, COPY, ATOMIC
            head = struct.unpack("<8I", q.read_ring(start, UNIT_BYTES))
            poll = SDMA_OP_POLL_REGMEM | (SdmaCmp.GE << 28) | POLL_MEM_POLL
            assert head[0] == poll, hex(head[0])
            assert head[1] | (head[2] << 32) == gate(p)
            assert head[3] == 1 and head[4] == 0xFFFFFFFF
            q.kick()
            ends[p] = end
        time.sleep(0.005)
        for p, q in queues.items():
            assert q.state()[0] < ends[p], "engine ran past a closed gate"
        comm.barrier()
        for s in peers:
            assert read(arrive(s), 1, u64)[0] == 0, "data landed before release"
        comm.barrier()

        for p in peers:
            fill(gate(p), [1], u64)
        for q in queues.values():
            assert q.quiesce(timeout_s=10.0), "quiesce timed out"
        comm.barrier()
        for s in peers:
            assert read(arrive(s), 1, u64)[0] == 1
            got = read(recv(s), words, u32)
            want = [(s << 24) ^ (rank << 16) ^ i for i in range(words)]
            assert list(got) == want, f"payload from {s} differs"
        comm.barrier()

        batch = TriggerBatch()
        for p in peers:
            batch.add(
                queues[p],
                gate(p),
                p,
                [(send(p), peer_ptr(comm, recv(rank), p), NBYTES)],
                peer_ptr(comm, arrive(rank), p),
            )
        assert len(batch) == len(peers)
        targets = np.full(world_size, 2, dtype=np.uint64)
        assert batch.arm(targets) == TriggerBatch.OK
        comm.barrier()
        for p in peers:
            fill(gate(p), [2], u64)
        for q in queues.values():
            assert q.quiesce(timeout_s=10.0), "quiesce timed out"
        comm.barrier()
        for s in peers:
            assert read(arrive(s), 1, u64)[0] == 2
        comm.barrier()

        p0 = peers[0]
        small = TriggerQueue(comm, p0, ring_bytes=4096)
        tiny = TriggerBatch()
        chain = (gate(p0), p0, [(send(p0), peer_ptr(comm, recv(rank), p0), 64)])
        with pytest.raises(ValueError):
            for _ in range(64):
                tiny.add(small, *chain, peer_ptr(comm, arrive(rank), p0))
        parked = small.append(gate(p0), 100, [], base + ARRIVE_OFF + 64 * rank)
        small.kick()
        one = TriggerBatch()
        one.add(small, *chain, base + ARRIVE_OFF + 64 * rank)
        _, w0 = small.state()
        assert one.arm(targets, timeout_s=0.05) == TriggerBatch.DRAIN_TIMEOUT
        assert small.state()[1] == w0, "a refused batch appended something"
        fill(gate(p0), [100], u64)
        assert small.quiesce(index=parked, timeout_s=10.0)
        assert small.close()
        for q in queues.values():
            assert q.close()
        comm.destroy()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


def _spawn_entry(rank, world_size, uid_path, results_path):
    _worker(rank, world_size, uid_path, results_path)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_trigger_queue_bindings(world_size):
    _require_gpus(world_size)
    from mori.cco import Communicator

    os.environ["MORI_ENABLE_SDMA"] = "1"
    os.environ.setdefault("MORI_SOCKET_IFNAME", "lo")
    tmp = tempfile.mkdtemp(prefix="cco_trigger_test_")
    uid_path = os.path.join(tmp, "uid.bin")
    results_path = os.path.join(tmp, "result_%d.txt")
    with open(uid_path, "wb") as f:
        f.write(bytes(Communicator.get_unique_id()))
    torch.multiprocessing.spawn(
        _spawn_entry,
        args=(world_size, uid_path, results_path),
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
