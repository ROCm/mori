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
)


def test_engine_desc():
    config = IOEngineConfig(
        host="127.0.0.1",
        port=get_free_port(),
        gpuId=0,
        backends=[BackendType.RDMA],
    )
    engine = IOEngine(key="engine", config=config)
    desc = engine.get_engine_desc()

    packed_desc = desc.pack()
    unpacked_desc = EngineDesc.unpack(packed_desc)

    # TODO：add comparison operator and check equality


def test_io_api():
    # Step1: Initialize io engines
    config = IOEngineConfig(
        host="127.0.0.1",
        port=get_free_port(),
        gpuId=0,
        backends=[BackendType.RDMA],
    )
    initiator = IOEngine(key="initiator", config=config)
    config.port = get_free_port()
    target = IOEngine(key="target", config=config)

    # Step2: register remote io engines
    initiator_desc = initiator.get_engine_desc()
    target_desc = target.get_engine_desc()

    initiator.register_remote_engine(target_desc)
    target.register_remote_engine(initiator_desc)

    # Step3: register memory buffer
    shape = [128, 8192]
    device1 = torch.device("cuda", 0)
    device2 = torch.device("cuda", 1)
    tensor1 = torch.ones(shape).to(device1)
    tensor2 = torch.ones(shape).to(device2)

    initiator_mem = initiator.register_torch_tensor(tensor1)
    target_mem = target.register_torch_tensor(tensor2)

    # Step4: initiate tensfer
    transfer_uid = initiator.allocate_transfer_uid()
    transfer_status = initiator.read(
        initiator_mem, 0, target_mem, 0, initiator_mem.size, transfer_uid
    )
    while transfer_status.Code() == StatusCode.INIT:
        pass
    print(
        f"read finished at initiator {transfer_status.Code()} {transfer_status.Message()}"
    )

    while True:
        target_side_status = target.pop_inbound_transfer_status(
            initiator_desc.key, transfer_uid
        )
        if not target_side_status:
            continue
    print(
        f"read finished at target {target_side_status.Code()} {target_side_status.Message()}"
    )

    # Step5: teardown
    initiator.deregister_memory(initiator_mem)
    target.deregister_memory(target_mem)

    initiator.deregister_remote_engine(target_desc)
    target.deregister_remote_engine(initiator_desc)

    del initiator
    del target
