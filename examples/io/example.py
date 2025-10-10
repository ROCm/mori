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
from mori.io import (
    IOEngineConfig,
    BackendType,
    IOEngine,
    RdmaBackendConfig,
    set_log_level,
)
import torch


def basic_read_write_example(initiator, target, size):
    # Allocate and register memory on both engines
    tensor1 = torch.randn(size).to(torch.device("cuda", 0), dtype=torch.uint8)
    tensor2 = torch.randn(size).to(torch.device("cuda", 1), dtype=torch.uint8)

    initiator_mem = initiator.register_torch_tensor(tensor1)
    target_mem = target.register_torch_tensor(tensor2)

    # Perform p2p transfer from initiator to target
    transfer_uid = initiator.allocate_transfer_uid()
    transfer_status = initiator.read(
        initiator_mem,
        0,  # offset of initiator memory
        target_mem,
        0,  # offset of target memory
        size,
        transfer_uid,
    )

    while transfer_status.InProgress():
        pass
    assert torch.equal(tensor1.cpu(), tensor2.cpu())
    print("basic read/write status:", transfer_status.Message())

    initiator.deregister_memory(initiator_mem)
    target.deregister_memory(target_mem)


def batch_read_write_example(initiator, target, size, batch_size):
    # Allocate and register memory on both engines
    tensor1 = torch.randn(size * batch_size).to(
        torch.device("cuda", 0), dtype=torch.uint8
    )
    tensor2 = torch.randn(size * batch_size).to(
        torch.device("cuda", 1), dtype=torch.uint8
    )

    initiator_mem = initiator.register_torch_tensor(tensor1)
    target_mem = target.register_torch_tensor(tensor2)

    # Prepare batch parameters
    local_offsets = [i * size for i in range(batch_size)]
    remote_offsets = [i * size for i in range(batch_size)]
    sizes = [size for _ in range(batch_size)]

    # Perform batch p2p transfer from initiator to target
    transfer_uid = initiator.allocate_transfer_uid()
    transfer_status = initiator.batch_read(
        [initiator_mem],
        [local_offsets],
        [target_mem],
        [remote_offsets],
        [sizes],
        [transfer_uid],
    )[0]

    while transfer_status.InProgress():
        pass
    assert torch.equal(tensor1.cpu(), tensor2.cpu())
    print("batch read/write status:", transfer_status.Message())

    initiator.deregister_memory(initiator_mem)
    target.deregister_memory(target_mem)


def session_batch_read_write_example(initiator, target, size, batch_size):
    # Allocate and register memory on both engines
    tensor1 = torch.randn(size * batch_size).to(
        torch.device("cuda", 0), dtype=torch.uint8
    )
    tensor2 = torch.randn(size * batch_size).to(
        torch.device("cuda", 1), dtype=torch.uint8
    )

    initiator_mem = initiator.register_torch_tensor(tensor1)
    target_mem = target.register_torch_tensor(tensor2)

    # Create session between the two memory regions
    session = initiator.create_session(initiator_mem, target_mem)

    # Prepare batch parameters
    local_offsets = [i * size for i in range(batch_size)]
    remote_offsets = [i * size for i in range(batch_size)]
    sizes = [size for _ in range(batch_size)]

    # Perform batch p2p transfer from initiator to target using session
    transfer_uid = session.allocate_transfer_uid()
    transfer_status = session.batch_read(
        local_offsets,
        remote_offsets,
        sizes,
        transfer_uid,
    )

    while transfer_status.InProgress():
        pass
    assert torch.equal(tensor1.cpu(), tensor2.cpu())

    print("session batch read/write status:", transfer_status.Message())

    initiator.deregister_memory(initiator_mem)
    target.deregister_memory(target_mem)


def example():
    set_log_level("info")

    # Step 1: Create two IOEngines (initiator and target)
    config = IOEngineConfig(
        host="127.0.0.1",
        port=8080,
    )
    initiator = IOEngine(key="initiator", config=config)
    config.port = 8081
    target = IOEngine(key="target", config=config)

    # Step 2: Create RDMA backend for both engines
    config = RdmaBackendConfig(
        qp_per_transfer=1,
    )
    initiator.create_backend(BackendType.RDMA, config)
    target.create_backend(BackendType.RDMA, config)

    # Step 3: Register memory regions on both engines
    initiator_desc = initiator.get_engine_desc()
    target_desc = target.get_engine_desc()

    initiator.register_remote_engine(target_desc)
    target.register_remote_engine(initiator_desc)

    # Step 4 & 5: Allocate, register memory and perform p2p transfers
    size = 1024
    basic_read_write_example(initiator, target, size)
    batch_read_write_example(initiator, target, size, batch_size=4)
    session_batch_read_write_example(initiator, target, size, batch_size=4)

    # Step 6: Deregister engine
    initiator.deregister_remote_engine(target_desc)
    target.deregister_remote_engine(initiator_desc)


if __name__ == "__main__":
    example()
