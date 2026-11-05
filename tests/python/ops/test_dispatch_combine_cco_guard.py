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
"""EpDispatchCombineHandle must refuse a cross-node config on the CCO backend at construction.

The check is the first thing the constructor does, so a non-null placeholder communicator pointer is
enough: it is never dereferenced, and no GPU, shmem or CCO setup is needed.
"""
import pytest

from mori import cpp as mori_cpp

FAKE_CCO_COMM_PTR = 0x1  # non-null: selects the CCO backend; never dereferenced


def _config(kernel_type, world_size, gpu_per_node):
    return mori_cpp.EpDispatchCombineConfig(
        rank=0,
        world_size=world_size,
        hidden_dim=7168,
        scale_dim=0,
        scale_type_size=1,
        max_token_type_size=2,
        max_num_inp_token_per_rank=128,
        num_experts_per_rank=256 // world_size,
        num_experts_per_token=8,
        warp_num_per_block=8,
        block_num=64,
        use_external_inp_buf=True,
        kernel_type=kernel_type,
        gpu_per_node=gpu_per_node,
        rdma_block_num=16,
        num_qp_per_pe=1,
    )


ALL_KERNEL_TYPES = list(mori_cpp.EpDispatchCombineKernelType.__members__.values())


@pytest.mark.parametrize("kernel_type", ALL_KERNEL_TYPES, ids=lambda k: k.name)
def test_cco_backend_rejects_cross_node_config(kernel_type):
    with pytest.raises(RuntimeError, match="supports intra-node only"):
        mori_cpp.EpDispatchCombineHandle(
            _config(kernel_type, world_size=16, gpu_per_node=8),
            cco_comm_ptr=FAKE_CCO_COMM_PTR,
        )
