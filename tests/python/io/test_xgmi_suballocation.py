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

"""Cross-process XGMI test for sub-region (offset) memory registration.

KV-cache connectors register each layer as a *view* into one large paged
allocation, so the registered pointer is allocation_base + offset. IPC handles
are keyed to the allocation base, so the importing side must add that offset
back. This test registers a sub-region at a non-zero offset on one GPU and
XGMI-reads it from another process/GPU, then checks the bytes match.

It must be cross-process: within a single process the XGMI backend serves remote
memory via the same-process direct pointer (offset already baked in), so the
offset handling is only exercised across processes (real hipIpcOpenMemHandle).
Before the fix the read pulls from the allocation base and the data mismatches.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.python.utils import TorchDistContext, get_free_port
from mori.io import (
    IOEngineConfig,
    BackendType,
    IOEngine,
    EngineDesc,
    MemoryDesc,
    XgmiBackendConfig,
    set_log_level,
)

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires at least 2 GPUs"
)

# Sub-region starts well past the allocation base so a base-relative (buggy)
# read lands in the zero-filled prefix and clearly mismatches the payload.
OFFSET_BYTES = 64 * 1024 * 1024
REGION_BYTES = 1 * 1024 * 1024
TOTAL_BYTES = OFFSET_BYTES + REGION_BYTES + 4096


def _send_bytes(data: bytes, dst: int):
    dist.send(torch.tensor([len(data)], dtype=torch.long), dst=dst)
    dist.send(torch.ByteTensor(list(data)), dst=dst)


def _recv_bytes(src: int) -> bytes:
    n = torch.zeros(1, dtype=torch.long)
    dist.recv(n, src=src)
    buf = torch.zeros(int(n.item()), dtype=torch.uint8)
    dist.recv(buf, src=src)
    return bytes(buf.tolist())


def _exchange(local_bytes: bytes, rank: int) -> bytes:
    """Symmetric two-rank byte exchange."""
    peer = 1 - rank
    if rank == 0:
        peer_bytes = _recv_bytes(src=peer)
        _send_bytes(local_bytes, dst=peer)
    else:
        _send_bytes(local_bytes, dst=peer)
        peer_bytes = _recv_bytes(src=peer)
    return peer_bytes


def _worker(rank, world_size, master_port, result_queue):
    """rank 0 = initiator/reader (GPU 0); rank 1 = target/payload (GPU 1)."""
    try:
        with TorchDistContext(
            rank=rank,
            world_size=world_size,
            master_addr="localhost",
            master_port=str(master_port),
            device_id=rank,
            backend="gloo",
        ):
            set_log_level("info")
            device = torch.device("cuda", rank)

            engine = IOEngine(
                key=f"xgmi_suballoc_{rank}", config=IOEngineConfig(host="", port=0)
            )
            engine.create_backend(BackendType.XGMI, XgmiBackendConfig())
            engine.register_remote_engine(
                EngineDesc.unpack(_exchange(engine.get_engine_desc().pack(), rank))
            )

            # One large allocation; register only a sub-region at OFFSET_BYTES.
            big = torch.zeros(TOTAL_BYTES, dtype=torch.uint8, device=device)
            sub = big.narrow(0, OFFSET_BYTES, REGION_BYTES)
            assert sub.data_ptr() == big.data_ptr() + OFFSET_BYTES

            # Distinct, nonzero payload so a base-relative read (zeros) is caught.
            pattern = ((torch.arange(REGION_BYTES, device=device) % 255) + 1).to(
                torch.uint8
            )
            if rank == 1:
                sub.copy_(pattern)
            torch.cuda.synchronize()

            mem = engine.register_torch_tensor(sub)
            remote_mem = MemoryDesc.unpack(_exchange(mem.pack(), rank))

            ok, detail = True, ""
            if rank == 0:
                sess = engine.create_session(mem, remote_mem)
                uid = sess.allocate_transfer_uid()
                status = sess.batch_read([0], [0], [REGION_BYTES], uid)
                status.Wait()
                if not status.Succeeded():
                    ok, detail = False, f"batch_read failed: {status.Message()}"
                elif not torch.equal(sub.cpu(), pattern.cpu()):
                    ok, detail = False, (
                        "sub-region XGMI read returned wrong bytes — the "
                        "registered region's base offset was not honored on the "
                        "importing side (got the allocation base instead)."
                    )

            # Both ranks reach the barrier before reporting so a data mismatch on
            # rank 0 doesn't surface as a gloo teardown error on rank 1.
            dist.barrier()
            result_queue.put(("PASS", "") if ok else ("FAIL", detail))
    except Exception as e:
        import traceback

        result_queue.put(("FAIL", f"{e}\n{traceback.format_exc()}"))


def test_xgmi_suballocation_offset_read():
    """Two-process XGMI read of a sub-region registered at a non-zero offset.

    Reproduces the bug where XgmiBackendSession ignored the registered region's
    offset within its allocation: before the fix the reader gets the allocation
    base (zeros) instead of the payload.
    """
    master_port = get_free_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    procs = [
        ctx.Process(target=_worker, args=(rank, 2, master_port, result_queue))
        for rank in range(2)
    ]
    for p in procs:
        p.start()

    results = [result_queue.get(timeout=180) for _ in procs]
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join()
            pytest.fail(f"worker {p.pid} timed out")

    for status, msg in results:
        assert status == "PASS", f"worker failed: {msg}"
    for p in procs:
        assert p.exitcode == 0, f"worker exited with code {p.exitcode}"
