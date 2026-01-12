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

"""
Tensor utilities for MoRI SHMEM - PyTorch integration
"""

import torch
from . import api as mori_shmem


# Simple helper to wrap mori shmem pointer as torch tensor
class MoriShmemBuffer:
    def __init__(self, ptr, nbytes, dtype: torch.dtype):
        self.ptr = ptr
        self.nbytes = nbytes
        self.dtype = dtype
        self.__cuda_array_interface__ = {
            "data": (self.ptr, False),
            "shape": (self.nbytes,),
            "typestr": "<u1",  # uint8
            "strides": None,
            "version": 3,
        }


def mori_shmem_create_tensor(shape, dtype) -> torch.Tensor:
    nbytes = torch.Size(shape).numel() * dtype.itemsize
    torch.cuda.synchronize()
    ptr = mori_shmem.shmem_malloc(nbytes)
    assert ptr != 0, "mori_shmem.shmem_malloc failed"
    torch.cuda.synchronize()
    buffer = MoriShmemBuffer(ptr, nbytes, dtype)
    # First create as uint8 tensor (byte array), then view as target dtype
    tensor = torch.as_tensor(buffer, device='cuda').view(dtype).view(*shape)
    setattr(tensor, "__symm_tensor__", True)
    return tensor

def symm_mori_shmem_tensor(tensor: torch.Tensor, peer: int) -> torch.Tensor:

    assert getattr(tensor, "__symm_tensor__",
                   False), "tensor is not a symm_tensor"

    if peer == mori_shmem.shmem_mype():
        return tensor

    ptr = mori_shmem.shmem_ptr_p2p(tensor.data_ptr(), mori_shmem.shmem_mype(), peer)
    buffer = MoriShmemBuffer(ptr, tensor.nbytes, tensor.dtype)
    return torch.as_tensor(buffer,
                           device="cuda").view(tensor.dtype).view(tensor.shape)


def mori_shmem_create_tensor_list_intra_node(shape, dtype, num_ranks):
    tensor = mori_shmem_create_tensor(shape, dtype)
    return [symm_mori_shmem_tensor(tensor, i) for i in range(num_ranks)]
