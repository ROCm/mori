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

"""XGMI sessions between processes whose HIP_VISIBLE_DEVICES sets are disjoint.

Disaggregated serving runs prefill and decode as separate processes pinned to
separate GPU ranges (e.g. HIP_VISIBLE_DEVICES=0,1,2,3 vs 4,5,6,7), so each side's
GPUs are *hidden* from the other and XGMI must reach them through IPC handles
plus KFD topology rather than a visible device index.

These tests cover the two ways that used to end in a silent `None` from
`create_session()`, which surfaced downstream as
`AttributeError: 'NoneType' object has no attribute 'batch_read'`:

  * the reader registers the remote engine before creating its XGMI backend --
    registrations only reached backends that already existed, so the backend
    never learned the remote engine and rejected every pair from it;
  * the peer registers its memory before creating its XGMI backend -- its
    MemoryDesc then carries no IPC handle, which the reader cannot repair, so
    session creation must fail with a message that says exactly that.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.python.utils import TorchDistContext, get_free_port

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires at least 2 GPUs"
)

REGION_BYTES = 1 * 1024 * 1024

READER, WRITER = 0, 1


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
    if rank == READER:
        peer_bytes = _recv_bytes(src=peer)
        _send_bytes(local_bytes, dst=peer)
    else:
        _send_bytes(local_bytes, dst=peer)
        peer_bytes = _recv_bytes(src=peer)
    return peer_bytes


def _worker(rank, master_port, scenario, result_queue):
    """rank 0 reads over XGMI from rank 1; each rank sees only its own GPU.

    scenario:
      "backend_first"    -- both sides create the backend before registering.
      "late_backend"     -- the reader registers the remote engine first.
      "peer_memory_first"-- the writer registers its memory first, so the reader
                            must refuse the session and name the empty IPC handle.
    """
    try:
        from mori.io import (
            BackendType,
            EngineDesc,
            IOEngine,
            IOEngineConfig,
            MemoryDesc,
            SessionUnavailableError,
            XgmiBackendConfig,
            set_log_level,
        )

        # The peer's GPU must not be visible here, so bail out loudly rather than silently testing the visible path.
        visible = torch.cuda.device_count()
        if visible != 1:
            result_queue.put(
                (
                    "FAIL",
                    f"rank {rank} sees {visible} GPUs, expected 1; "
                    f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')!r} "
                    "did not take effect",
                )
            )
            return

        with TorchDistContext(
            rank=rank,
            world_size=2,
            master_addr="localhost",
            master_port=str(master_port),
            device_id=0,
            backend="gloo",
        ):
            set_log_level("info")
            device = torch.device("cuda", 0)

            engine = IOEngine(
                key=f"xgmi_split_vis_{rank}", config=IOEngineConfig(host="", port=0)
            )

            def create_backend():
                engine.create_backend(BackendType.XGMI, XgmiBackendConfig())

            pattern = ((torch.arange(REGION_BYTES, device=device) % 251) + 1).to(
                torch.uint8
            )
            buf = torch.zeros(REGION_BYTES, dtype=torch.uint8, device=device)
            if rank == WRITER:
                buf.copy_(pattern)
            torch.cuda.synchronize()

            # The orderings under test differ only in when the backend comes up relative to the registrations it needs.
            defer_backend = (scenario == "late_backend" and rank == READER) or (
                scenario == "peer_memory_first" and rank == WRITER
            )
            if defer_backend:
                if scenario == "peer_memory_first":
                    mem = engine.register_torch_tensor(buf)
                    create_backend()
                    remote_engine_bytes = _exchange(
                        engine.get_engine_desc().pack(), rank
                    )
                    engine.register_remote_engine(
                        EngineDesc.unpack(remote_engine_bytes)
                    )
                else:
                    remote_engine_bytes = _exchange(
                        engine.get_engine_desc().pack(), rank
                    )
                    engine.register_remote_engine(
                        EngineDesc.unpack(remote_engine_bytes)
                    )
                    create_backend()
                    mem = engine.register_torch_tensor(buf)
            else:
                create_backend()
                engine.register_remote_engine(
                    EngineDesc.unpack(_exchange(engine.get_engine_desc().pack(), rank))
                )
                mem = engine.register_torch_tensor(buf)

            remote_mem = MemoryDesc.unpack(_exchange(mem.pack(), rank))

            ok, detail = True, ""
            if rank == READER:
                if scenario == "peer_memory_first":
                    try:
                        engine.create_session(mem, remote_mem)
                        ok, detail = False, (
                            "create_session succeeded even though the peer's "
                            "MemoryDesc has no IPC handle"
                        )
                    except SessionUnavailableError as exc:
                        message = str(exc)
                        for expected in ("IPC handle", str(remote_mem.id)):
                            if expected not in message:
                                ok, detail = False, (
                                    f"SessionUnavailableError message does not mention "
                                    f"{expected!r}: {message}"
                                )
                else:
                    sess = engine.create_session(mem, remote_mem)
                    status = sess.batch_read(
                        [0], [0], [REGION_BYTES], sess.allocate_transfer_uid()
                    )
                    status.Wait()
                    if not status.Succeeded():
                        ok, detail = False, f"batch_read failed: {status.Message()}"
                    elif not torch.equal(buf.cpu(), pattern.cpu()):
                        mismatched = int((buf != pattern).sum().item())
                        ok, detail = False, (
                            f"XGMI read across disjoint HIP_VISIBLE_DEVICES returned "
                            f"{mismatched} wrong bytes"
                        )

            # Both ranks reach the barrier before reporting so a reader-side failure doesn't surface as a gloo teardown error on the writer.
            dist.barrier()
            result_queue.put(("PASS", "") if ok else ("FAIL", detail))
    except Exception as e:
        import traceback

        result_queue.put(("FAIL", f"{e}\n{traceback.format_exc()}"))


MASK_VARS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


def _run(scenario):
    master_port = get_free_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    # A spawned child snapshots the environment at start() and initializes HIP on import, before _worker runs, so the mask must be set here per child, not inside the worker.
    saved = {name: os.environ.get(name) for name in MASK_VARS}
    procs = []
    try:
        for rank in (READER, WRITER):
            os.environ["HIP_VISIBLE_DEVICES"] = str(rank)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            proc = ctx.Process(
                target=_worker, args=(rank, master_port, scenario, result_queue)
            )
            proc.start()
            procs.append(proc)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    results = [result_queue.get(timeout=180) for _ in procs]
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join()
            pytest.fail(f"worker {p.pid} timed out")

    # Report every worker's message: a reader session refusal also fails the writer on a gloo barrier, whose message explains nothing on its own.
    failures = [msg for status, msg in results if status != "PASS"]
    assert not failures, "worker(s) failed:\n" + "\n".join(failures)
    for p in procs:
        assert p.exitcode == 0, f"worker exited with code {p.exitcode}"


def test_xgmi_read_across_disjoint_visible_devices():
    """XGMI read where the remote GPU is hidden from the reader's process."""
    _run("backend_first")


def test_xgmi_session_when_remote_engine_registered_before_backend():
    """The reader registers the remote engine before creating its XGMI backend.

    RegisterRemoteEngine only reached backends that existed at call time, so the
    backend came up blind to the remote engine, CanHandle rejected every pair
    from it, and create_session returned a bare None.
    """
    _run("late_backend")


def test_create_session_names_reason_when_peer_memory_has_no_ipc_handle():
    """Session creation must explain a peer descriptor it cannot use.

    The peer registered its memory before creating its XGMI backend, so the
    descriptor it sent carries no IPC handle. The reader cannot repair that, so
    it must raise a message naming the cause instead of returning None and
    failing later inside batch_read.
    """
    _run("peer_memory_first")
