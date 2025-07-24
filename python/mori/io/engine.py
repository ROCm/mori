from mori import cpp as mori_cpp
import torch
import ctypes

TORCH_DEVICE_TYPE_MAP = {
    "cpu": mori_cpp.MemoryLocationType.CPU,
    "cuda": mori_cpp.MemoryLocationType.GPU,
}


class IOEngine:
    def __init__(self, key, config: mori_cpp.IOEngineConfig):
        self._engine = mori_cpp.IOEngine(key, config)

    def get_engine_desc(self):
        return self._engine.GetEngineDesc()

    def register_remote_engine(self, engine_desc: mori_cpp.EngineDesc):
        return self._engine.RegisterRemoteEngine(engine_desc)

    def deregister_remote_engine(self, engine_desc: mori_cpp.EngineDesc):
        return self._engine.DeRegisterRemoteEngine(engine_desc)

    def register_torch_tensor(self, tensor: torch.Tensor):
        if not tensor.is_contiguous():
            raise RuntimeError("input tensor must be contiguous")

        data = ctypes.pythonapi.PyCapsule_New(
            ctypes.c_void_p(tensor.data_ptr()), None, None
        )
        total_bytes = tensor.nelement() * tensor.element_size()
        device_id = tensor.device.index
        mem_loc = TORCH_DEVICE_TYPE_MAP[tensor.device.type]
        return self._engine.RegisterMemory(data, total_bytes, device_id, mem_loc)

    def deregister_memory(self, mem_desc: mori_cpp.MemoryDesc):
        return self._engine.DeRegisterMemory(mem_desc)

    def allocate_transfer_uid(self):
        return self._engine.AllocateTransferUniqueId()
