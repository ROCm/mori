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
import os

import pytest
import torch

import mori

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "4G")


@pytest.fixture()
def single_rank_shmem():
    uid = mori.shmem.shmem_get_unique_id()
    mori.shmem.shmem_init_attr(mori.shmem.MORI_SHMEM_INIT_WITH_UNIQUEID, 0, 1, uid)
    try:
        yield
    finally:
        torch.cuda.synchronize()
        mori.shmem.shmem_finalize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP device is required")
def test_local_expert_count(single_rank_shmem):
    config = mori.cpp.EpDispatchCombineConfig(
        rank=1,
        world_size=4,
        num_experts_per_rank=3,
        num_experts_per_token=2,
        warp_num_per_block=2,
        block_num=4,
    )

    indices = torch.tensor(
        [
            [3, 0],
            [5, 4],
            [3, 5],
            [1, 5],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    total_recv_token_num = torch.tensor([4], dtype=torch.int32, device="cuda")
    local_expert_count = torch.full((3,), -1, dtype=torch.int32, device="cuda")

    mori.cpp.launch_local_expert_count(
        config,
        indices.data_ptr(),
        total_recv_token_num.data_ptr(),
        local_expert_count.data_ptr(),
        stream=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    expected = torch.tensor([2, 1, 3], dtype=torch.int32)
    assert torch.equal(local_expert_count.cpu(), expected)

    total_recv_token_num.zero_()
    local_expert_count.fill_(9)

    mori.cpp.launch_local_expert_count(
        config,
        indices.data_ptr(),
        total_recv_token_num.data_ptr(),
        local_expert_count.data_ptr(),
        stream=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    assert torch.equal(local_expert_count.cpu(), torch.zeros(3, dtype=torch.int32))
