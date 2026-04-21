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
import pytest
import os
import time
import gc
from contextlib import contextmanager
from tests.python.utils import get_free_port
import torch
from mori.io import (
    IOEngineConfig,
    BackendType,
    IOEngine,
    EngineDesc,
    MemoryDesc,
    StatusCode,
    MemoryLocationType,
    RdmaBackendConfig,
    XgmiBackendConfig,
    set_log_level,
)


def create_connected_engine_pair(
    name_prefix,
    qp_per_transfer,
    post_batch_size,
    num_worker_threads,
    enable_notification=True,
):
    config = IOEngineConfig(
        host="127.0.0.1",
        port=get_free_port(),
    )
    initiator = IOEngine(key=f"{name_prefix}_initiator", config=config)
    config.port = get_free_port()
    target = IOEngine(key=f"{name_prefix}_target", config=config)

    config = RdmaBackendConfig(
        qp_per_transfer=qp_per_transfer,
        post_batch_size=post_batch_size,
        num_worker_threads=num_worker_threads,
        enable_notification=enable_notification,
    )
    with temporary_env("MORI_DISABLE_AUTO_XGMI", "1"):
        initiator.create_backend(BackendType.RDMA, config)
        target.create_backend(BackendType.RDMA, config)

    initiator_desc = initiator.get_engine_desc()
    target_desc = target.get_engine_desc()

    initiator.register_remote_engine(target_desc)
    target.register_remote_engine(initiator_desc)

    return initiator, target


@contextmanager
def temporary_env(env_name: str, value: str):
    old_value = os.environ.get(env_name)
    os.environ[env_name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = old_value


@pytest.fixture(scope="module")
def pre_connected_engine_pair():
    set_log_level("info")
    initiator, target = create_connected_engine_pair(
        "normal", qp_per_transfer=2, post_batch_size=-1, num_worker_threads=1
    )
    multhd_initiator, multhd_target = create_connected_engine_pair(
        "multhd", qp_per_transfer=2, post_batch_size=-1, num_worker_threads=2
    )
    no_notif_initiator, no_notif_target = create_connected_engine_pair(
        "no_notif",
        qp_per_transfer=2,
        post_batch_size=-1,
        num_worker_threads=1,
        enable_notification=False,
    )

    engines = {
        "normal": (initiator, target),
        "multhd": (multhd_initiator, multhd_target),
        "no_notif": (no_notif_initiator, no_notif_target),
    }
    yield engines

    del initiator, target


def test_engine_desc():
    config = IOEngineConfig(
        host="127.0.0.1",
        port=get_free_port(),
    )
    engine = IOEngine(key="engine", config=config)
    engine.create_backend(BackendType.RDMA)

    desc = engine.get_engine_desc()
    assert desc.node_id != ""
    assert desc.pid > 0

    packed_desc = desc.pack()
    unpacked_desc = EngineDesc.unpack(packed_desc)
    assert desc == unpacked_desc


def test_engine_desc_port_zero_auto_bind():
    config = IOEngineConfig(
        host="127.0.0.1",
        port=0,
    )
    engine = IOEngine(key="engine_port0", config=config)
    engine.create_backend(BackendType.RDMA)

    desc = engine.get_engine_desc()
    assert desc.port > 0
    assert desc.node_id != ""
    assert desc.pid > 0

    packed_desc = desc.pack()
    unpacked_desc = EngineDesc.unpack(packed_desc)
    assert desc == unpacked_desc


def test_engine_desc_node_id_env_override(monkeypatch):
    monkeypatch.setenv("MORI_IO_NODE_ID", "node-id-test")
    config = IOEngineConfig(
        host="127.0.0.1",
        port=get_free_port(),
    )
    engine = IOEngine(key="engine_node_id", config=config)
    desc = engine.get_engine_desc()
    assert desc.node_id == "node-id-test"


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="requires GPU")
def test_rdmabackend_auto_creates_xgmi_backend_for_gpu_mem(monkeypatch):
    monkeypatch.setenv("MORI_DISABLE_AUTO_XGMI", "0")
    config = IOEngineConfig(
        host="127.0.0.1",
        port=get_free_port(),
    )
    engine = IOEngine(key="auto_xgmi_engine", config=config)
    engine.create_backend(BackendType.RDMA)

    tensor = torch.ones((1024,), device=torch.device("cuda", 0), dtype=torch.float32)
    mem_desc = engine.register_torch_tensor(tensor)

    # XGMI backend registration should fill IPC handle bytes for GPU memory.
    assert any(b != 0 for b in mem_desc.ipc_handle)


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="requires GPU")
def test_rdmabackend_auto_xgmi_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MORI_DISABLE_AUTO_XGMI", "1")
    config = IOEngineConfig(
        host="127.0.0.1",
        port=get_free_port(),
    )
    engine = IOEngine(key="auto_xgmi_disabled_engine", config=config)
    engine.create_backend(BackendType.RDMA)

    tensor = torch.ones((1024,), device=torch.device("cuda", 0), dtype=torch.float32)
    mem_desc = engine.register_torch_tensor(tensor)

    # Without auto-created XGMI backend, ipc_handle should keep default zero values.
    assert all(b == 0 for b in mem_desc.ipc_handle)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
def test_intra_node_prefers_xgmi_after_rdma_creation(monkeypatch):
    monkeypatch.setenv("MORI_DISABLE_AUTO_XGMI", "0")
    config = IOEngineConfig(
        host="127.0.0.1",
        port=get_free_port(),
    )
    engine = IOEngine(key="auto_xgmi_route_engine", config=config)
    engine.create_backend(BackendType.RDMA)

    src = torch.arange(0, 4096, dtype=torch.float32, device=torch.device("cuda", 0))
    dst = torch.zeros_like(src, device=torch.device("cuda", 1))
    src_mem = engine.register_torch_tensor(src)
    dst_mem = engine.register_torch_tensor(dst)

    transfer_uid = engine.allocate_transfer_uid()
    status = engine.write(
        src_mem, 0, dst_mem, 0, src.numel() * src.element_size(), transfer_uid
    )
    status.Wait()

    assert status.Succeeded()
    assert torch.equal(src.cpu(), dst.cpu())


def test_mem_desc():
    config = IOEngineConfig(
        host="127.0.0.1",
        port=get_free_port(),
    )
    engine = IOEngine(key="engine", config=config)
    engine.create_backend(BackendType.RDMA)

    # Test cpu tensor
    tensor = torch.ones([1, 2, 34, 56])
    mem_desc = engine.register_torch_tensor(tensor)

    assert mem_desc.engine_key == "engine"
    assert mem_desc.device_id == -1
    assert mem_desc.device_bus_id == ""
    assert mem_desc.data == tensor.data_ptr()
    assert mem_desc.size == tensor.nelement() * tensor.element_size()
    assert mem_desc.loc == MemoryLocationType.CPU

    # Test gpu tensor
    device = torch.device("cuda", 0)
    tensor = torch.ones([56, 34, 2, 1]).to(device)
    mem_desc = engine.register_torch_tensor(tensor)

    assert mem_desc.engine_key == "engine"
    assert mem_desc.device_id == 0
    assert mem_desc.device_bus_id != ""
    assert mem_desc.data == tensor.data_ptr()
    assert mem_desc.size == tensor.nelement() * tensor.element_size()
    assert mem_desc.loc == MemoryLocationType.GPU

    # TODO: test mem_desc pack / unpack
    packed_desc = mem_desc.pack()
    unpacked_desc = MemoryDesc.unpack(packed_desc)
    assert mem_desc == unpacked_desc


def wait_status(status):
    while status.InProgress():
        pass


def wait_inbound_status(engine, remote_engine_key, remote_transfer_uid):
    while True:
        target_side_status = engine.pop_inbound_transfer_status(
            remote_engine_key, remote_transfer_uid
        )
        if target_side_status:
            return target_side_status


def wait_inbound_status_with_timeout(
    engine, remote_engine_key, remote_transfer_uid, timeout_s=2.0
):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        target_side_status = engine.pop_inbound_transfer_status(
            remote_engine_key, remote_transfer_uid
        )
        if target_side_status:
            return target_side_status
        time.sleep(0.001)
    return None


def alloc_and_register_mem(engine_pair, shape):
    initiator, target = engine_pair

    # register memory buffer
    device1 = torch.device("cuda", 0)
    device2 = torch.device("cuda", 1)
    tensor1 = torch.randint(1, 256, shape, dtype=torch.uint8, device=device1)
    tensor2 = torch.randint(1, 256, shape, dtype=torch.uint8, device=device2)

    initiator_mem = initiator.register_torch_tensor(tensor1)
    target_mem = target.register_torch_tensor(tensor2)
    return tensor1, tensor2, initiator_mem, target_mem


def register_explicit_mem(engine_pair, initiator_tensor, target_tensor):
    initiator, target = engine_pair
    initiator_mem = initiator.register_torch_tensor(initiator_tensor)
    target_mem = target.register_torch_tensor(target_tensor)
    return initiator_mem, target_mem


def cleanup_engine_pair(initiator, target, initiator_mems=(), target_mems=()):
    if initiator is not None:
        for mem in initiator_mems:
            if mem is None:
                continue
            try:
                initiator.deregister_memory(mem)
            except Exception:
                pass
    if target is not None:
        for mem in target_mems:
            if mem is None:
                continue
            try:
                target.deregister_memory(mem)
            except Exception:
                pass

    del initiator
    del target
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_transfer_result(
    engine_pair,
    initiator_status,
    initiator_tensor,
    initiator_tensor_copy,
    target_tensor,
    target_tensor_copy,
    transfer_uid,
    op_type,
    enable_notification=True,
):
    initiator, target = engine_pair
    wait_status(initiator_status)
    assert initiator_status.Succeeded()

    # Only check target inbound status when notification is enabled
    if enable_notification:
        target_status = wait_inbound_status(
            target, initiator.get_engine_desc().key, transfer_uid
        )
        assert target_status.Succeeded()
    else:
        time.sleep(0.001)

    # Verify data correctness
    assert torch.equal(initiator_tensor.cpu(), target_tensor.cpu())

    if op_type == "read":
        assert not torch.equal(initiator_tensor.cpu(), initiator_tensor_copy.cpu())
    else:
        assert not torch.equal(target_tensor.cpu(), target_tensor_copy.cpu())


@pytest.mark.parametrize("engine_type", ("normal", "multhd", "no_notif"))
@pytest.mark.parametrize(
    "enable_sess",
    (
        True,
        False,
    ),
)
@pytest.mark.parametrize(
    "enable_batch",
    (
        True,
        False,
    ),
)
@pytest.mark.parametrize(
    "op_type",
    (
        "write",
        "read",
    ),
)
@pytest.mark.parametrize(
    "batch_size",
    (
        1,
        64,
    ),
)
@pytest.mark.parametrize(
    "buffer_size",
    (
        8,
        8192,
    ),
)
def test_rdma_backend_ops(
    pre_connected_engine_pair,
    engine_type,
    enable_sess,
    enable_batch,
    op_type,
    batch_size,
    buffer_size,
):
    engine_pair = pre_connected_engine_pair[engine_type]
    initiator, target = engine_pair
    initiator_tensor, target_tensor, initiator_mem, target_mem = alloc_and_register_mem(
        engine_pair, [batch_size, buffer_size]
    )
    initiator_tensor_copy = initiator_tensor.clone()
    target_tensor_copy = target_tensor.clone()

    sess = initiator.create_session(initiator_mem, target_mem)
    offsets = [i * buffer_size for i in range(batch_size)]
    sizes = [buffer_size for _ in range(batch_size)]
    uid_status_list = []

    if enable_batch:
        if enable_sess:
            transfer_uid = sess.allocate_transfer_uid()
            func = sess.batch_read if op_type == "read" else sess.batch_write
            transfer_status = func(offsets, offsets, sizes, transfer_uid)
        else:
            transfer_uid = initiator.allocate_transfer_uid()
            func = initiator.batch_read if op_type == "read" else initiator.batch_write
            transfer_status = func(
                [initiator_mem],
                [offsets],
                [target_mem],
                [offsets],
                [sizes],
                [transfer_uid],
            )[0]
        uid_status_list.append((transfer_uid, transfer_status))
    else:
        for i in range(batch_size):
            if enable_sess:
                transfer_uid = sess.allocate_transfer_uid()
                func = sess.read if op_type == "read" else sess.write
                transfer_status = func(offsets[i], offsets[i], sizes[i], transfer_uid)
            else:
                transfer_uid = initiator.allocate_transfer_uid()
                func = initiator.read if op_type == "read" else initiator.write
                transfer_status = func(
                    initiator_mem,
                    offsets[i],
                    target_mem,
                    offsets[i],
                    sizes[i],
                    transfer_uid,
                )
            uid_status_list.append((transfer_uid, transfer_status))

    enable_notification = engine_type != "no_notif"
    for uid, status in uid_status_list:
        check_transfer_result(
            engine_pair,
            status,
            initiator_tensor,
            initiator_tensor_copy,
            target_tensor,
            target_tensor_copy,
            uid,
            op_type,
            enable_notification,
        )


def test_err_out_of_range(pre_connected_engine_pair):
    engine_pair = pre_connected_engine_pair["normal"]
    initiator, target = engine_pair
    initiator_tensor, target_tensor, initiator_mem, target_mem = alloc_and_register_mem(
        engine_pair,
        (
            2,
            32,
        ),
    )

    sess = initiator.create_session(initiator_mem, target_mem)
    offsets = (0, 32)
    sizes = (32, 34)

    transfer_uid = sess.allocate_transfer_uid()
    transfer_status = sess.batch_read(offsets, offsets, sizes, transfer_uid)

    assert transfer_status.Failed()
    assert transfer_status.Code() == StatusCode.ERR_INVALID_ARGS


def test_notification_disabled():
    """Test that when notification is disabled, pop_inbound_transfer_status doesn't work."""
    initiator, target = create_connected_engine_pair(
        "notif_test",
        qp_per_transfer=1,
        post_batch_size=-1,
        num_worker_threads=1,
        enable_notification=False,
    )

    initiator_tensor, target_tensor, initiator_mem, target_mem = alloc_and_register_mem(
        (initiator, target), [1, 64]
    )

    transfer_uid = initiator.allocate_transfer_uid()
    transfer_status = initiator.write(initiator_mem, 0, target_mem, 0, 64, transfer_uid)

    wait_status(transfer_status)
    assert transfer_status.Succeeded()

    # Verify data was transferred correctly
    assert torch.equal(initiator_tensor.cpu(), target_tensor.cpu())

    # pop_inbound_transfer_status should not work (returns None)
    time.sleep(0.1)
    inbound_status = target.pop_inbound_transfer_status(
        initiator.get_engine_desc().key, transfer_uid
    )
    assert inbound_status is None  # No notification received


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
@pytest.mark.parametrize("op_type", ("write", "read"))
def test_rdma_chunked_session_transfer_crosses_multiple_chunks(op_type):
    with temporary_env("MORI_IO_RDMA_MR_CHUNK_SIZE", "64"):
        initiator, target = create_connected_engine_pair(
            "chunked_single",
            qp_per_transfer=2,
            post_batch_size=-1,
            num_worker_threads=1,
            enable_notification=True,
        )
        initiator_mem = None
        target_mem = None
        sess = None

        try:
            total_bytes = 256
            local_offset = 48
            transfer_size = 160

            if op_type == "write":
                initiator_tensor = torch.arange(
                    total_bytes, dtype=torch.uint8, device=torch.device("cuda", 0)
                )
                target_tensor = torch.full(
                    (total_bytes,),
                    0xA5,
                    dtype=torch.uint8,
                    device=torch.device("cuda", 1),
                )
            else:
                initiator_tensor = torch.full(
                    (total_bytes,),
                    0x5A,
                    dtype=torch.uint8,
                    device=torch.device("cuda", 0),
                )
                target_tensor = torch.arange(
                    total_bytes, dtype=torch.uint8, device=torch.device("cuda", 1)
                )

            initiator_before = initiator_tensor.clone()
            target_before = target_tensor.clone()
            initiator_mem, target_mem = register_explicit_mem(
                (initiator, target), initiator_tensor, target_tensor
            )

            sess = initiator.create_session(initiator_mem, target_mem)
            assert sess is not None
            assert sess.alive()

            transfer_uid = sess.allocate_transfer_uid()
            op = sess.write if op_type == "write" else sess.read
            status = op(local_offset, local_offset, transfer_size, transfer_uid)

            wait_status(status)
            assert status.Succeeded(), status.Message()

            inbound = wait_inbound_status_with_timeout(
                target, initiator.get_engine_desc().key, transfer_uid, timeout_s=3.0
            )
            assert (
                inbound is not None
            ), "Expected inbound notification for chunked transfer"
            assert inbound.Succeeded(), inbound.Message()

            if op_type == "write":
                expected_target = target_before.clone()
                expected_target[local_offset : local_offset + transfer_size] = (
                    initiator_before[local_offset : local_offset + transfer_size]
                )
                assert torch.equal(target_tensor.cpu(), expected_target.cpu())
                assert torch.equal(initiator_tensor.cpu(), initiator_before.cpu())
            else:
                expected_initiator = initiator_before.clone()
                expected_initiator[local_offset : local_offset + transfer_size] = (
                    target_before[local_offset : local_offset + transfer_size]
                )
                assert torch.equal(initiator_tensor.cpu(), expected_initiator.cpu())
                assert torch.equal(target_tensor.cpu(), target_before.cpu())
        finally:
            del sess
            cleanup_engine_pair(initiator, target, [initiator_mem], [target_mem])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
def test_rdma_chunked_multithread_batch_crosses_chunks():
    with temporary_env("MORI_IO_RDMA_MR_CHUNK_SIZE", "64"):
        initiator, target = create_connected_engine_pair(
            "chunked_multhd",
            qp_per_transfer=2,
            post_batch_size=-1,
            num_worker_threads=2,
            enable_notification=True,
        )
        initiator_mem = None
        target_mem = None
        sess = None

        try:
            total_bytes = 512
            initiator_tensor = torch.arange(
                total_bytes, dtype=torch.uint8, device=torch.device("cuda", 0)
            )
            target_tensor = torch.full(
                (total_bytes,), 0x11, dtype=torch.uint8, device=torch.device("cuda", 1)
            )
            target_before = target_tensor.clone()

            initiator_mem, target_mem = register_explicit_mem(
                (initiator, target), initiator_tensor, target_tensor
            )

            sess = initiator.create_session(initiator_mem, target_mem)
            assert sess is not None
            assert sess.alive()

            offsets = [16, 112, 208, 304]
            sizes = [96, 96, 96, 96]
            transfer_uid = sess.allocate_transfer_uid()
            status = sess.batch_write(offsets, offsets, sizes, transfer_uid)

            wait_status(status)
            assert status.Succeeded(), status.Message()

            inbound = wait_inbound_status_with_timeout(
                target, initiator.get_engine_desc().key, transfer_uid, timeout_s=3.0
            )
            assert (
                inbound is not None
            ), "Expected inbound notification for chunked multithread batch"
            assert inbound.Succeeded(), inbound.Message()

            expected_target = target_before.clone()
            for offset, size in zip(offsets, sizes):
                expected_target[offset : offset + size] = initiator_tensor[
                    offset : offset + size
                ]
            assert torch.equal(target_tensor.cpu(), expected_target.cpu())
        finally:
            del sess
            cleanup_engine_pair(initiator, target, [initiator_mem], [target_mem])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
def test_rdma_chunked_session_invalidates_on_deregister_and_reregisters_same_tensor():
    with temporary_env("MORI_IO_RDMA_MR_CHUNK_SIZE", "64"):
        initiator, target = create_connected_engine_pair(
            "chunked_reregister",
            qp_per_transfer=2,
            post_batch_size=-1,
            num_worker_threads=1,
            enable_notification=True,
        )
        initiator_mem = None
        re_registered_mem = None
        target_mem = None
        sess = None
        recovered_sess = None

        try:
            total_bytes = 256
            initiator_tensor = torch.arange(
                total_bytes, dtype=torch.uint8, device=torch.device("cuda", 0)
            )
            target_tensor = torch.zeros(
                total_bytes, dtype=torch.uint8, device=torch.device("cuda", 1)
            )

            initiator_mem, target_mem = register_explicit_mem(
                (initiator, target), initiator_tensor, target_tensor
            )

            sess = initiator.create_session(initiator_mem, target_mem)
            assert sess is not None
            assert sess.alive()

            first_uid = sess.allocate_transfer_uid()
            first_status = sess.write(0, 0, total_bytes, first_uid)
            wait_status(first_status)
            assert first_status.Succeeded(), first_status.Message()
            inbound = wait_inbound_status_with_timeout(
                target, initiator.get_engine_desc().key, first_uid, timeout_s=3.0
            )
            assert inbound is not None
            assert inbound.Succeeded(), inbound.Message()
            assert torch.equal(initiator_tensor.cpu(), target_tensor.cpu())

            target_tensor.zero_()
            initiator.deregister_memory(initiator_mem)
            initiator_mem = None

            assert not sess.alive()
            stale_uid = sess.allocate_transfer_uid()
            stale_status = sess.write(0, 0, total_bytes, stale_uid)
            assert stale_status.Failed()
            assert stale_status.Code() == StatusCode.ERR_INVALID_ARGS
            assert "deregistered" in stale_status.Message().lower()

            re_registered_mem = initiator.register_torch_tensor(initiator_tensor)
            assert re_registered_mem.data == initiator_tensor.data_ptr()

            recovered_sess = initiator.create_session(re_registered_mem, target_mem)
            assert recovered_sess is not None
            assert recovered_sess.alive()

            recovered_uid = recovered_sess.allocate_transfer_uid()
            recovered_status = recovered_sess.write(0, 0, total_bytes, recovered_uid)
            wait_status(recovered_status)
            assert recovered_status.Succeeded(), recovered_status.Message()
            inbound = wait_inbound_status_with_timeout(
                target, initiator.get_engine_desc().key, recovered_uid, timeout_s=3.0
            )
            assert inbound is not None
            assert inbound.Succeeded(), inbound.Message()
            assert torch.equal(initiator_tensor.cpu(), target_tensor.cpu())
        finally:
            del sess
            del recovered_sess
            cleanup_engine_pair(
                initiator, target, [initiator_mem, re_registered_mem], [target_mem]
            )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
def test_rdma_chunked_inflight_dereg_keeps_registration_alive():
    """In-flight transfers must keep their underlying chunk MRs alive even when
    the user deregisters the local memory mid-flight.

    Validates the ``CqCallbackMeta::localRegRef`` ownership anchor: if the
    anchor is missing, deregistering the local memory (which removes the
    manager's and session cache's references to ``RdmaLocalMemoryRegistration``)
    would tear down the chunk MRs while WRs are still pending in the SQ, and
    those in-flight WRs would surface ``LOC_PROT_ERR`` on completion.

    With the patch, the registration object stays alive until the last
    in-flight transfer's ``CqCallbackMeta`` is released, so all submissions
    must complete cleanly. Re-registering the same tensor afterward must work
    against the now-fresh MR set.
    """
    with temporary_env("MORI_IO_RDMA_MR_CHUNK_SIZE", "64"):
        initiator, target = create_connected_engine_pair(
            "chunked_inflight_dereg",
            qp_per_transfer=2,
            post_batch_size=-1,
            num_worker_threads=1,
            enable_notification=True,
        )

        initiator_mem = None
        target_mem = None
        re_registered_mem = None
        recovered_sess = None
        statuses = []

        try:
            # Force the resolved (multi-slice) path: small chunk size so each
            # 4 KiB transfer fans out to ~64 slices, giving the SQ enough
            # outstanding work that deregistration races with completion.
            total_bytes = 4096
            initiator_tensor = torch.arange(
                total_bytes, dtype=torch.uint8, device=torch.device("cuda", 0)
            )
            target_tensor = torch.zeros(
                total_bytes, dtype=torch.uint8, device=torch.device("cuda", 1)
            )

            initiator_mem, target_mem = register_explicit_mem(
                (initiator, target), initiator_tensor, target_tensor
            )

            # Use the cached path (engine.write) on purpose: when we
            # deregister, InvalidateSessionsForMemory drops the cache's
            # session ref so the CqCallbackMeta::localRegRef anchor is the
            # ONLY thing keeping the registration alive.
            num_inflight = 64
            for _ in range(num_inflight):
                uid = initiator.allocate_transfer_uid()
                statuses.append(
                    initiator.write(initiator_mem, 0, target_mem, 0, total_bytes, uid)
                )

            # Race the deregister against the in-flight WRs. We do NOT wait
            # before this call.
            initiator.deregister_memory(initiator_mem)
            initiator_mem = None  # avoid double-deregister in cleanup

            # All in-flight transfers must still complete successfully. Failure
            # here (typically with LOC_PROT_ERR) means the chunk MRs got torn
            # down while WRs were still pending.
            for i, s in enumerate(statuses):
                wait_status(s)
                assert s.Succeeded(), (
                    f"in-flight transfer #{i} failed after dereg: "
                    f"code={s.Code()} msg={s.Message()}"
                )
            # Final state of target is the last successful write.
            assert torch.equal(initiator_tensor.cpu(), target_tensor.cpu())

            # Re-registering the same tensor (same data_ptr) at this point
            # must yield a fresh registration backed by NEW chunk MRs. With
            # the old address-keyed mrPool, this case used to silently reuse
            # the stale MR slot; with owned MRs each registration owns its
            # own ibv_mr* via RAII.
            target_tensor.zero_()
            re_registered_mem = initiator.register_torch_tensor(initiator_tensor)
            assert re_registered_mem.data == initiator_tensor.data_ptr()

            recovered_sess = initiator.create_session(re_registered_mem, target_mem)
            assert recovered_sess is not None
            assert recovered_sess.alive()

            recovered_uid = recovered_sess.allocate_transfer_uid()
            recovered_status = recovered_sess.write(0, 0, total_bytes, recovered_uid)
            wait_status(recovered_status)
            assert recovered_status.Succeeded(), recovered_status.Message()
            inbound = wait_inbound_status_with_timeout(
                target,
                initiator.get_engine_desc().key,
                recovered_uid,
                timeout_s=3.0,
            )
            assert inbound is not None
            assert inbound.Succeeded(), inbound.Message()
            assert torch.equal(initiator_tensor.cpu(), target_tensor.cpu())
        finally:
            # IMPORTANT: do not clear `statuses` until every transfer has
            # finished waiting above. CqCallbackMeta keeps a raw pointer to
            # each TransferStatus, so dropping a status while its WR is still
            # pending would race with the CQ poll thread.
            statuses.clear()
            del recovered_sess
            cleanup_engine_pair(
                initiator,
                target,
                [initiator_mem, re_registered_mem],
                [target_mem],
            )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
def test_multithread_batch_error_path_is_recoverable():
    """Regression for callback meta ownership on error paths.

    Repeatedly trigger failing multithread batch calls, then verify a valid
    transfer still completes end-to-end with notification.
    """

    initiator, target = create_connected_engine_pair(
        "regress_cbmeta",
        qp_per_transfer=2,
        post_batch_size=-1,
        num_worker_threads=2,
        enable_notification=True,
    )

    initiator_tensor, target_tensor, initiator_mem, target_mem = alloc_and_register_mem(
        (initiator, target), (2, 64)
    )

    bad_offsets = [0, 64]
    bad_sizes = [64, 65]  # out-of-range for the second element

    # Run multiple failures to stress repeated cleanup/release behavior.
    for _ in range(20):
        transfer_uid = initiator.allocate_transfer_uid()
        status = initiator.batch_read(
            [initiator_mem],
            [bad_offsets],
            [target_mem],
            [bad_offsets],
            [bad_sizes],
            [transfer_uid],
        )[0]
        wait_status(status)
        assert status.Failed()
        assert status.Code() == StatusCode.ERR_INVALID_ARGS

    # Verify subsequent valid transfer still works.
    transfer_uid = initiator.allocate_transfer_uid()
    full_size = initiator_tensor.numel() * initiator_tensor.element_size()
    status = initiator.write(initiator_mem, 0, target_mem, 0, full_size, transfer_uid)
    wait_status(status)
    assert status.Succeeded(), status.Message()

    inbound = wait_inbound_status_with_timeout(
        target, initiator.get_engine_desc().key, transfer_uid, timeout_s=3.0
    )
    assert (
        inbound is not None
    ), "Expected inbound notification after successful transfer"
    assert inbound.Succeeded()
    assert torch.equal(initiator_tensor.cpu(), target_tensor.cpu())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
def test_successful_writes_always_have_inbound_notification_under_pressure():
    """Regression for notify all-or-none behavior.

    Under low SQ timeout and high-frequency writes, every successful write must
    still produce a corresponding inbound completion notification.
    """

    with temporary_env("MORI_IO_SQ_BACKOFF_TIMEOUT_US", "100"):
        initiator, target = create_connected_engine_pair(
            "regress_notify",
            qp_per_transfer=4,
            post_batch_size=1,
            num_worker_threads=1,
            enable_notification=True,
        )

        initiator_tensor, target_tensor, initiator_mem, target_mem = (
            alloc_and_register_mem((initiator, target), (1, 128))
        )

        initiator_key = initiator.get_engine_desc().key

        for i in range(50):
            transfer_uid = initiator.allocate_transfer_uid()
            status = initiator.write(initiator_mem, 0, target_mem, 0, 128, transfer_uid)
            wait_status(status)

            if status.Succeeded():
                inbound = wait_inbound_status_with_timeout(
                    target, initiator_key, transfer_uid, timeout_s=2.0
                )
                assert (
                    inbound is not None
                ), f"Missing inbound notification for successful transfer {i}"
                assert inbound.Succeeded()
            else:
                # Failed transfers may legitimately miss notification; they
                # should not poison future successful transfers.
                assert status.Code() in (
                    StatusCode.ERR_RDMA_OP,
                    StatusCode.ERR_BAD_STATE,
                )

        assert torch.equal(initiator_tensor.cpu(), target_tensor.cpu())


def test_no_backend():
    config = IOEngineConfig(
        host="127.0.0.1",
        port=get_free_port(),
    )
    initiator = IOEngine(key="no_be_initiator", config=config)
    config.port = get_free_port()
    target = IOEngine(key="no_be_target", config=config)

    initiator_desc = initiator.get_engine_desc()
    target_desc = target.get_engine_desc()

    initiator.register_remote_engine(target_desc)
    target.register_remote_engine(initiator_desc)

    initiator_tensor, target_tensor, initiator_mem, target_mem = alloc_and_register_mem(
        (initiator, target),
        (32,),
    )

    offsets = (0, 16)
    sizes = (16, 16)

    transfer_uid = initiator.allocate_transfer_uid()
    transfer_status = initiator.batch_read(
        [initiator_mem], [offsets], [target_mem], [offsets], [sizes], [transfer_uid]
    )[0]

    assert transfer_status.Failed()
    assert transfer_status.Code() == StatusCode.ERR_BAD_STATE

    sess = initiator.create_session(initiator_mem, target_mem)
    assert sess is None


@pytest.fixture(scope="module")
def xgmi_engine():
    set_log_level("info")
    config = IOEngineConfig(host="", port=0)
    engine = IOEngine(key="xgmi_engine", config=config)
    xgmi_config = XgmiBackendConfig(num_streams=64, num_events=64)
    engine.create_backend(BackendType.XGMI, xgmi_config)
    yield engine


def alloc_xgmi_mem(engine, src_gpu, dst_gpu, shape):
    src_tensor = torch.randn(
        shape, device=torch.device("cuda", src_gpu), dtype=torch.float32
    )
    dst_tensor = torch.zeros(
        shape, device=torch.device("cuda", dst_gpu), dtype=torch.float32
    )
    src_mem = engine.register_torch_tensor(src_tensor)
    dst_mem = engine.register_torch_tensor(dst_tensor)
    return src_tensor, dst_tensor, src_mem, dst_mem


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
@pytest.mark.parametrize("enable_sess", (True, False))
@pytest.mark.parametrize("enable_batch", (True, False))
@pytest.mark.parametrize("op_type", ("write", "read"))
@pytest.mark.parametrize("batch_size", (1, 16))
@pytest.mark.parametrize("buffer_size", (64, 4096))
def test_xgmi_backend_ops(
    xgmi_engine, enable_sess, enable_batch, op_type, batch_size, buffer_size
):
    src_tensor, dst_tensor, src_mem, dst_mem = alloc_xgmi_mem(
        xgmi_engine, src_gpu=0, dst_gpu=1, shape=[batch_size, buffer_size]
    )

    if op_type == "read":
        src_tensor, dst_tensor = dst_tensor, src_tensor
        src_mem, dst_mem = dst_mem, src_mem

    sess = xgmi_engine.create_session(src_mem, dst_mem)
    offsets = [i * buffer_size * 4 for i in range(batch_size)]
    sizes = [buffer_size * 4 for _ in range(batch_size)]

    if enable_batch:
        if enable_sess:
            transfer_uid = sess.allocate_transfer_uid()
            func = sess.batch_read if op_type == "read" else sess.batch_write
            status = func(offsets, offsets, sizes, transfer_uid)
        else:
            transfer_uid = xgmi_engine.allocate_transfer_uid()
            func = (
                xgmi_engine.batch_read if op_type == "read" else xgmi_engine.batch_write
            )
            status = func(
                [src_mem], [offsets], [dst_mem], [offsets], [sizes], [transfer_uid]
            )[0]
    else:
        statuses = []
        for i in range(batch_size):
            if enable_sess:
                transfer_uid = sess.allocate_transfer_uid()
                func = sess.read if op_type == "read" else sess.write
                status = func(offsets[i], offsets[i], sizes[i], transfer_uid)
            else:
                transfer_uid = xgmi_engine.allocate_transfer_uid()
                func = xgmi_engine.read if op_type == "read" else xgmi_engine.write
                status = func(
                    src_mem, offsets[i], dst_mem, offsets[i], sizes[i], transfer_uid
                )
            statuses.append(status)

        for s in statuses:
            s.Wait()
            assert s.Succeeded()
        assert torch.equal(src_tensor.cpu(), dst_tensor.cpu())
        return

    status.Wait()
    assert status.Succeeded()
    assert torch.equal(src_tensor.cpu(), dst_tensor.cpu())


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="requires GPU")
def test_xgmi_same_device(xgmi_engine):
    src_tensor, dst_tensor, src_mem, dst_mem = alloc_xgmi_mem(
        xgmi_engine, src_gpu=0, dst_gpu=0, shape=(1024,)
    )
    transfer_uid = xgmi_engine.allocate_transfer_uid()
    status = xgmi_engine.write(src_mem, 0, dst_mem, 0, 1024 * 4, transfer_uid)
    status.Wait()
    assert status.Succeeded()
    assert torch.equal(src_tensor.cpu(), dst_tensor.cpu())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
def test_xgmi_status_completes_without_wait(xgmi_engine):
    src_tensor, dst_tensor, src_mem, dst_mem = alloc_xgmi_mem(
        xgmi_engine, src_gpu=0, dst_gpu=1, shape=(1024,)
    )
    transfer_uid = xgmi_engine.allocate_transfer_uid()
    status = xgmi_engine.write(src_mem, 0, dst_mem, 0, 1024 * 4, transfer_uid)

    deadline = time.time() + 5
    while status.InProgress() and time.time() < deadline:
        time.sleep(0.01)

    assert not status.InProgress(), "XGMI status should progress without calling Wait()"
    assert status.Succeeded(), status.Message()
    assert torch.equal(src_tensor.cpu(), dst_tensor.cpu())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
def test_xgmi_no_inbound_notification(xgmi_engine):
    src_tensor, dst_tensor, src_mem, dst_mem = alloc_xgmi_mem(
        xgmi_engine, src_gpu=0, dst_gpu=1, shape=(1024,)
    )
    transfer_uid = xgmi_engine.allocate_transfer_uid()
    status = xgmi_engine.write(src_mem, 0, dst_mem, 0, 1024 * 4, transfer_uid)
    status.Wait()
    assert status.Succeeded()
    assert xgmi_engine.pop_inbound_transfer_status("any_key", transfer_uid) is None


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
def test_xgmi_cross_engine_transfer():
    set_log_level("info")

    config = IOEngineConfig(host="", port=0)
    engine_a = IOEngine(key="xgmi_engine_a", config=config)
    engine_b = IOEngine(key="xgmi_engine_b", config=config)

    xgmi_config = XgmiBackendConfig(num_streams=64, num_events=64)
    engine_a.create_backend(BackendType.XGMI, xgmi_config)
    engine_b.create_backend(BackendType.XGMI, xgmi_config)

    engine_a.register_remote_engine(engine_b.get_engine_desc())
    engine_b.register_remote_engine(engine_a.get_engine_desc())

    src_tensor = torch.randn(1024, device=torch.device("cuda", 0), dtype=torch.float32)
    dst_tensor = torch.zeros(1024, device=torch.device("cuda", 1), dtype=torch.float32)

    src_mem = engine_a.register_torch_tensor(src_tensor)
    dst_mem = engine_b.register_torch_tensor(dst_tensor)

    assert any(
        b != 0 for b in src_mem.ipc_handle
    ), "IPC handle should be filled by XGMI backend"
    assert any(
        b != 0 for b in dst_mem.ipc_handle
    ), "IPC handle should be filled by XGMI backend"

    src_mem_packed = src_mem.pack()
    dst_mem_packed = dst_mem.pack()

    remote_src_mem = MemoryDesc.unpack(src_mem_packed)
    remote_dst_mem = MemoryDesc.unpack(dst_mem_packed)

    assert (
        remote_src_mem.ipc_handle == src_mem.ipc_handle
    ), "IPC handle should survive serialization"
    assert (
        remote_dst_mem.ipc_handle == dst_mem.ipc_handle
    ), "IPC handle should survive serialization"

    transfer_uid = engine_b.allocate_transfer_uid()
    status = engine_b.read(dst_mem, 0, remote_src_mem, 0, 1024 * 4, transfer_uid)
    status.Wait()
    assert status.Succeeded()
    assert torch.equal(src_tensor.cpu(), dst_tensor.cpu())

    dst_tensor.zero_()
    assert not torch.equal(src_tensor.cpu(), dst_tensor.cpu())
    transfer_uid = engine_a.allocate_transfer_uid()
    status = engine_a.write(src_mem, 0, remote_dst_mem, 0, 1024 * 4, transfer_uid)
    status.Wait()
    assert status.Succeeded()
    assert torch.equal(src_tensor.cpu(), dst_tensor.cpu())

    dst_tensor.zero_()
    sess = engine_b.create_session(dst_mem, remote_src_mem)
    assert sess is not None
    transfer_uid = sess.allocate_transfer_uid()
    status = sess.read(0, 0, 1024 * 4, transfer_uid)
    status.Wait()
    assert status.Succeeded()
    assert torch.equal(src_tensor.cpu(), dst_tensor.cpu())

    dst_tensor.zero_()
    sess = engine_a.create_session(src_mem, remote_dst_mem)
    assert sess is not None
    transfer_uid = sess.allocate_transfer_uid()
    status = sess.write(0, 0, 1024 * 4, transfer_uid)
    status.Wait()
    assert status.Succeeded()
    assert torch.equal(src_tensor.cpu(), dst_tensor.cpu())

    dst_tensor.zero_()
    offsets = [0 * 4, 256 * 4, 512 * 4, 768 * 4]
    sizes = [256 * 4, 256 * 4, 256 * 4, 256 * 4]
    transfer_uid = engine_b.allocate_transfer_uid()
    status = engine_b.batch_read(
        [dst_mem], [offsets], [remote_src_mem], [offsets], [sizes], [transfer_uid]
    )[0]
    status.Wait()
    assert status.Succeeded()
    assert torch.equal(src_tensor.cpu(), dst_tensor.cpu())

    dst_tensor.zero_()
    transfer_uid = engine_a.allocate_transfer_uid()
    status = engine_a.batch_write(
        [src_mem], [offsets], [remote_dst_mem], [offsets], [sizes], [transfer_uid]
    )[0]
    status.Wait()
    assert status.Succeeded()
    assert torch.equal(src_tensor.cpu(), dst_tensor.cpu())

    dst_tensor.zero_()
    sess = engine_b.create_session(dst_mem, remote_src_mem)
    transfer_uid = sess.allocate_transfer_uid()
    status = sess.batch_read(offsets, offsets, sizes, transfer_uid)
    status.Wait()
    assert status.Succeeded()
    assert torch.equal(src_tensor.cpu(), dst_tensor.cpu())
