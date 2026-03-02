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
from mori import cpp as mori_cpp
import torch
import ctypes

TORCH_DEVICE_TYPE_MAP = {
    "cpu": mori_cpp.MemoryLocationType.CPU,
    "cuda": mori_cpp.MemoryLocationType.GPU,
}


class IOEngineSession:
    def __init__(self, mori_sess):
        self._sess = mori_sess

    def allocate_transfer_uid(self):
        return self._sess.AllocateTransferUniqueId()

    def _single_side_transfer(
        self, func, local_offset, remote_offset, size, transfer_uid
    ):
        transfer_status = mori_cpp.TransferStatus()
        func(
            local_offset,
            remote_offset,
            size,
            transfer_status,
            transfer_uid,
        )
        return transfer_status

    def read(self, *args):
        return self._single_side_transfer(self._sess.Read, *args)

    def write(self, *args):
        return self._single_side_transfer(self._sess.Write, *args)

    def _batch_single_side_transfer(
        self, func, local_offsets, remote_offsets, sizes, transfer_uid
    ):
        transfer_status = mori_cpp.TransferStatus()
        func(
            local_offsets,
            remote_offsets,
            sizes,
            transfer_status,
            transfer_uid,
        )
        return transfer_status

    def batch_read(self, *args):
        return self._batch_single_side_transfer(self._sess.BatchRead, *args)

    def batch_write(self, *args):
        return self._batch_single_side_transfer(self._sess.BatchWrite, *args)

    def alive(self):
        return self._sess.Alive()


class IOEngine:
    def __init__(self, key, config: mori_cpp.IOEngineConfig):
        self._engine = mori_cpp.IOEngine(key, config)

    def get_engine_desc(self):
        return self._engine.GetEngineDesc()

    def create_backend(self, type: mori_cpp.BackendType, config=None):
        if config is None:
            if type is mori_cpp.BackendType.RDMA:
                config = mori_cpp.RdmaBackendConfig()
            elif type is mori_cpp.BackendType.XGMI:
                config = mori_cpp.XgmiBackendConfig()
            elif type is mori_cpp.BackendType.TCP:
                config = mori_cpp.TcpBackendConfig()
            else:
                raise NotImplementedError("backend not implemented yet")
        return self._engine.CreateBackend(type, config)

    def remove_backend(self, type: mori_cpp.BackendType):
        return self._engine.RemoveBackend(type)

    def register_remote_engine(self, engine_desc: mori_cpp.EngineDesc):
        return self._engine.RegisterRemoteEngine(engine_desc)

    def deregister_remote_engine(self, engine_desc: mori_cpp.EngineDesc):
        return self._engine.DeregisterRemoteEngine(engine_desc)

    def register_memory(
        self, ptr: int, size: int, device_id: int, mem_loc: mori_cpp.MemoryLocationType
    ):
        data = ctypes.pythonapi.PyCapsule_New(ctypes.c_void_p(ptr), None, None)
        return self._engine.RegisterMemory(data, size, device_id, mem_loc)

    def register_torch_tensor(self, tensor: torch.Tensor):
        if not tensor.is_contiguous():
            raise RuntimeError("input tensor must be contiguous")

        total_bytes = tensor.nelement() * tensor.element_size()
        device_id = tensor.device.index
        if device_id is None:
            device_id = -1
        mem_loc = TORCH_DEVICE_TYPE_MAP[tensor.device.type]
        return self.register_memory(tensor.data_ptr(), total_bytes, device_id, mem_loc)

    def deregister_memory(self, mem_desc: mori_cpp.MemoryDesc):
        return self._engine.DeregisterMemory(mem_desc)

    def allocate_transfer_uid(self):
        return self._engine.AllocateTransferUniqueId()

    def _single_side_transfer(
        self,
        func,
        local_dest_mem_desc,
        local_offset,
        remote_src_mem_desc,
        remote_offset,
        size,
        transfer_uid,
    ):
        transfer_status = mori_cpp.TransferStatus()
        func(
            local_dest_mem_desc,
            local_offset,
            remote_src_mem_desc,
            remote_offset,
            size,
            transfer_status,
            transfer_uid,
        )
        return transfer_status

    def read(self, *args):
        return self._single_side_transfer(self._engine.Read, *args)

    def write(self, *args):
        return self._single_side_transfer(self._engine.Write, *args)

    def _batch_single_side_transfer(
        self,
        func,
        local_dest_mem_desc,
        local_offsets,
        remote_src_mem_desc,
        remote_offsets,
        sizes,
        transfer_uid,
    ):
        transfer_status = [mori_cpp.TransferStatus() for _ in range(len(sizes))]
        func(
            local_dest_mem_desc,
            local_offsets,
            remote_src_mem_desc,
            remote_offsets,
            sizes,
            transfer_status,
            transfer_uid,
        )
        return transfer_status

    def batch_read(self, *args):
        return self._batch_single_side_transfer(self._engine.BatchRead, *args)

    def batch_write(self, *args):
        return self._batch_single_side_transfer(self._engine.BatchWrite, *args)

    def create_session(self, local_mem, remote_mem):
        mori_sess = self._engine.CreateSession(local_mem, remote_mem)
        if mori_sess is None:
            return None
        return IOEngineSession(mori_sess)

    def pop_inbound_transfer_status(self, remote_key, transfer_uid):
        transfer_status = mori_cpp.TransferStatus()
        found = self._engine.PopInboundTransferStatus(
            remote_key, transfer_uid, transfer_status
        )
        if found:
            return transfer_status
        return None
