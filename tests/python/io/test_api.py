import pytest
from tests.python.utils import get_free_port
import torch
import mori
from mori.io import IOEngineConfig, BackendType, IOEngine, EngineDesc, MemoryDesc


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
    device1 = torch.device("cuda", 0)
    device2 = torch.device("cuda", 1)
    tensor1 = torch.ones([128, 8192]).to(device1)
    tensor2 = torch.ones([128, 8192]).to(device2)

    initiator_mem = initiator.register_torch_tensor(tensor1)
    target_mem = target.register_torch_tensor(tensor2)

    # Step4: initiate tensfer
    transfer_uid = initiator.allocate_transfer_uid()
    #     future = engine1.write(tensor_1_desc, tensor_2_desc)
    #     status = future.result()
    #     if status != mori.io.StatusCode.SUCC:
    #         print("transfer failed")
    #         exit(1)

    # Step5: teardown
    initiator.deregister_memory(initiator_mem)
    target.deregister_memory(target_mem)

    initiator.deregister_remote_engine(target_desc)
    target.deregister_remote_engine(initiator_desc)

    del initiator
    del target
