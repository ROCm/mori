import pytest
from tests.python.utils import get_free_port
import torch
import mori
from mori.io import (
    IOEngineConfig,
    BackendType,
    IOEngine,
    EngineDesc,
    MemoryDesc,
    StatusCode,
    MemoryLocationType,
)


@pytest.fixture(scope="module")
def pre_connected_engine_pair():
    config = IOEngineConfig(
        host="127.0.0.1",
        port=get_free_port(),
    )
    initiator = IOEngine(key="initiator", config=config)
    config.port = get_free_port()
    target = IOEngine(key="target", config=config)

    initiator.create_backend(BackendType.RDMA)
    target.create_backend(BackendType.RDMA)

    initiator_desc = initiator.get_engine_desc()
    target_desc = target.get_engine_desc()

    initiator.register_remote_engine(target_desc)
    target.register_remote_engine(initiator_desc)

    yield (initiator, target)

    del initiator
    del target


def test_engine_desc():
    config = IOEngineConfig(
        host="127.0.0.1",
        port=get_free_port(),
    )
    engine = IOEngine(key="engine", config=config)
    engine.create_backend(BackendType.RDMA)

    desc = engine.get_engine_desc()

    packed_desc = desc.pack()
    unpacked_desc = EngineDesc.unpack(packed_desc)
    assert desc == unpacked_desc


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
    while status.Code() == StatusCode.INIT:
        pass


def wait_inbound_status(engine, remote_engine_key, remote_transfer_uid):
    while True:
        target_side_status = engine.pop_inbound_transfer_status(
            remote_engine_key, remote_transfer_uid
        )
        if target_side_status:
            return target_side_status


def alloc_and_register_mem(pre_connected_engine_pair, shape):
    initiator, target = pre_connected_engine_pair

    # register memory buffer
    device1 = torch.device("cuda", 0)
    device2 = torch.device("cuda", 1)
    tensor1 = torch.randn(shape).to(device1, dtype=torch.uint8)
    tensor2 = torch.randn(shape).to(device2, dtype=torch.uint8)

    initiator_mem = initiator.register_torch_tensor(tensor1)
    target_mem = target.register_torch_tensor(tensor2)
    return tensor1, tensor2, initiator_mem, target_mem


def check_transfer_result(
    pre_connected_engine_pair,
    initiator_status,
    initiator_tensor,
    target_tensor,
    transfer_uid,
):
    initiator, target = pre_connected_engine_pair
    wait_status(initiator_status)
    target_status = wait_inbound_status(
        target, initiator.get_engine_desc().key, transfer_uid
    )
    assert initiator_status.Code() == StatusCode.SUCCESS
    assert target_status.Code() == StatusCode.SUCCESS
    assert torch.equal(initiator_tensor.cpu(), target_tensor.cpu())


def test_read(pre_connected_engine_pair):
    initiator, target = pre_connected_engine_pair
    initiator_tensor, target_tensor, initiator_mem, target_mem = alloc_and_register_mem(
        pre_connected_engine_pair, [128, 8192]
    )

    transfer_uid = initiator.allocate_transfer_uid()
    transfer_status = initiator.read(
        initiator_mem, 0, target_mem, 0, initiator_mem.size, transfer_uid
    )
    check_transfer_result(
        pre_connected_engine_pair,
        transfer_status,
        initiator_tensor,
        target_tensor,
        transfer_uid,
    )


def test_batch_read(pre_connected_engine_pair):
    initiator, target = pre_connected_engine_pair
    batch_size, buffer_size = 128, 8192

    initiator_tensor, target_tensor, initiator_mem, target_mem = alloc_and_register_mem(
        pre_connected_engine_pair, [batch_size, buffer_size]
    )

    transfer_uid = initiator.allocate_transfer_uid()
    offsets = [i * buffer_size for i in range(batch_size)]
    sizes = [buffer_size for _ in range(batch_size)]
    transfer_status = initiator.batch_read(
        initiator_mem, offsets, target_mem, offsets, sizes, transfer_uid
    )

    check_transfer_result(
        pre_connected_engine_pair,
        transfer_status,
        initiator_tensor,
        target_tensor,
        transfer_uid,
    )


def test_sess_batch_read(pre_connected_engine_pair):
    initiator, target = pre_connected_engine_pair
    batch_size, buffer_size = 128, 8192

    initiator_tensor, target_tensor, initiator_mem, target_mem = alloc_and_register_mem(
        pre_connected_engine_pair, [batch_size, buffer_size]
    )

    sess = initiator.create_session(initiator_mem, target_mem)
    transfer_uid = sess.allocate_transfer_uid()
    offsets = [i * buffer_size for i in range(batch_size)]
    sizes = [buffer_size for _ in range(batch_size)]
    transfer_status = sess.batch_read(offsets, offsets, sizes, transfer_uid)

    check_transfer_result(
        pre_connected_engine_pair,
        transfer_status,
        initiator_tensor,
        target_tensor,
        transfer_uid,
    )
