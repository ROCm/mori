# import pytest
# import mori


# def test_io_api():
#     # Step1: Initialize io engines
#     backends = [
#         mori.io.BackendType.XGMI,
#         mori.io.BackendType.RDMA,
#         mori.io.BackendType.TCP,
#     ]
#     engine1 = mori.io.IOEngine(
#         backends=backends,
#         config=mori.io.IOEngineConfig(ip="localhost", port=get_free_port()),
#     )
#     engine2 = mori.io.IOEngine(
#         backends=backends,
#         config=mori.io.IOEngineConfig(ip="localhost", port=get_free_port()),
#     )

#     # Step2: register remote io engines
#     engine_1_desc = engine1.get_engine_desc()
#     engine_2_desc = engine2.get_engine_desc()

#     engine_1_desc.register_remote_engine(engine_2_desc)
#     engine_2_desc.register_remote_engine(engine_1_desc)

#     # Step3: register memory buffer
#     device1 = torch.device("cuda", 0)
#     device2 = torch.device("cuda", 1)
#     tensor1 = torch.ones([128, 8192]).to(device1)
#     tensor2 = torch.ones([128, 8192]).to(device2)

#     tensor_1_desc = engine_1_desc.register_memory(tensor1)
#     tensor_2_desc = engine_2_desc.register_memory(tensor2)

#     # Step4: initiate tensfer
#     future = engine1.write(tensor_1_desc, tensor_2_desc)
#     status = future.result()
#     if status != mori.io.StatusCode.SUCC:
#         print("transfer failed")
#         exit(1)

#     # Step5: teardown
#     engine1.deregister_memory(tensor_1_desc)
#     engine2.deregister_memory(tensor_2_desc)
#     engine1.deregister_remote_engine(engine_2_desc)
#     engine2.deregister_remote_engine(engine_1_desc)
#     del engine1
#     del engine2
