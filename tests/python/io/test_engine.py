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
    set_log_level,
)


def create_connected_engine_pair(
    name_prefix, qp_per_transfer, post_batch_size, num_worker_threads
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
    set_log_level("trace")
    initiator, target = create_connected_engine_pair(
        "normal", qp_per_transfer=2, post_batch_size=-1, num_worker_threads=1
    )
    multhd_initiator, multhd_target = create_connected_engine_pair(
        "multhd", qp_per_transfer=2, post_batch_size=-1, num_worker_threads=2
    )

    engines = {
        "normal": (initiator, target),
        "multhd": (multhd_initiator, multhd_target),
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
    tensor1 = torch.randn(shape).to(device1, dtype=torch.uint8)
    tensor2 = torch.randn(shape).to(device2, dtype=torch.uint8)

    initiator_mem = initiator.register_torch_tensor(tensor1)
    target_mem = target.register_torch_tensor(tensor2)
    return tensor1, tensor2, initiator_mem, target_mem


def check_transfer_result(
    engine_pair,
    initiator_status,
    initiator_tensor,
    target_tensor,
    transfer_uid,
):
    initiator, target = engine_pair
    wait_status(initiator_status)
    target_status = wait_inbound_status(
        target, initiator.get_engine_desc().key, transfer_uid
    )
    assert initiator_status.Succeeded()
    assert target_status.Succeeded()
    assert torch.equal(initiator_tensor.cpu(), target_tensor.cpu())


@pytest.mark.parametrize("engine_type", ("multhd", "normal"))
@pytest.mark.parametrize("enable_sess", (True, False))
@pytest.mark.parametrize("enable_batch", (True, False))
@pytest.mark.parametrize("op_type", ("read",))
@pytest.mark.parametrize(
    "batch_size",
    (
        1,
        64,
    ),
)
@pytest.mark.parametrize("buffer_size", (8, 8192))
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

    sess = initiator.create_session(initiator_mem, target_mem)
    offsets = [i * buffer_size for i in range(batch_size)]
    sizes = [buffer_size for _ in range(batch_size)]
    uid_status_list = []

    if enable_batch:
        if enable_sess:
            transfer_uid = sess.allocate_transfer_uid()
            transfer_status = sess.batch_read(offsets, offsets, sizes, transfer_uid)
        else:
            transfer_uid = initiator.allocate_transfer_uid()
            transfer_status = initiator.batch_read(
                initiator_mem, offsets, target_mem, offsets, sizes, transfer_uid
            )
        uid_status_list.append((transfer_uid, transfer_status))
    else:
        for i in range(batch_size):
            if enable_sess:
                transfer_uid = sess.allocate_transfer_uid()
                transfer_status = sess.read(
                    offsets[i], offsets[i], sizes[i], transfer_uid
                )
            else:
                transfer_uid = initiator.allocate_transfer_uid()
                transfer_status = initiator.read(
                    initiator_mem,
                    offsets[i],
                    target_mem,
                    offsets[i],
                    sizes[i],
                    transfer_uid,
                )
            uid_status_list.append((transfer_uid, transfer_status))

    for uid, status in uid_status_list:
        check_transfer_result(
            engine_pair,
            status,
            initiator_tensor,
            target_tensor,
            uid,
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
