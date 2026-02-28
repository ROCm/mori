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
import time
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
    initiator.create_backend(BackendType.RDMA, config)
    target.create_backend(BackendType.RDMA, config)

    initiator_desc = initiator.get_engine_desc()
    target_desc = target.get_engine_desc()

    initiator.register_remote_engine(target_desc)
    target.register_remote_engine(initiator_desc)

    return initiator, target


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
def test_rdmabackend_auto_creates_xgmi_backend_for_gpu_mem():
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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
def test_intra_node_prefers_xgmi_after_rdma_creation():
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
    assert mem_desc.data == tensor.data_ptr()
    assert mem_desc.size == tensor.nelement() * tensor.element_size()
    assert mem_desc.loc == MemoryLocationType.CPU

    # Test gpu tensor
    device = torch.device("cuda", 0)
    tensor = torch.ones([56, 34, 2, 1]).to(device)
    mem_desc = engine.register_torch_tensor(tensor)

    assert mem_desc.engine_key == "engine"
    assert mem_desc.device_id == 0
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
