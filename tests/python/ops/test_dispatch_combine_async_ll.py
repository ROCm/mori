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

import mori
import pytest
import torch
from tests.python.ops.dispatch_combine_test_utils import (
    _all_data_types,
    _is_fp4x2_dtype,
    EpDispatchCombineTestCase,
    assert_worker_results,
    start_torch_dist_process_manager,
)

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "6G")


class AsyncLLDispatchCombineTestCase(EpDispatchCombineTestCase):
    def run_test_once(self, op, test_data):
        (
            _,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data
        (
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            dispatch_recv_num_token,
        ) = op.dispatch_send(
            all_rank_input[self.config.rank],
            all_rank_weights[self.config.rank],
            all_rank_scales[self.config.rank],
            all_rank_indices[self.config.rank],
        )
        op.dispatch_recv()

        self.sync()
        self.check_dispatch_result(
            op,
            test_data,
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            dispatch_recv_num_token,
        )

        # AsyncLL combine weight reconstruction is not exercised in the
        # reference example yet, so validate token reconstruction only.
        combine_output, _ = op.combine_send(
            dispatch_output,
            None,
            all_rank_indices[self.config.rank],
        )
        op.combine_recv()

        self.sync()
        self.check_combine_result(op, test_data, combine_output, None)


@pytest.fixture(scope="session")
def torch_dist_process_manager():
    manager = start_torch_dist_process_manager(world_size=8)
    yield manager
    manager.shutdown()


def _test_dispatch_combine(
    rank,
    world_size,
    data_type,
    hidden_dim,
    scale_dim,
    scale_type_size,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    quant_type="none",
):
    config = mori.ops.EpDispatchCombineConfig(
        data_type=data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim // 2 if _is_fp4x2_dtype(data_type) else hidden_dim,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        max_token_type_size=4,
        block_num=64,
        warp_num_per_block=8,
        kernel_type=mori.ops.EpDispatchCombineKernelType.AsyncLL,
        quant_type=quant_type,
    )
    op = mori.ops.EpDispatchCombineOp(config)
    test_case = AsyncLLDispatchCombineTestCase(config)
    test_data = test_case.gen_test_data()
    test_case.run_test_once(op, test_data)


@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("data_type", _all_data_types())
@pytest.mark.parametrize("hidden_dim", (7168, 4096))
@pytest.mark.parametrize("scale_dim", (0, 32))
@pytest.mark.parametrize("scale_type_size", (1, 4))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (1, 128))
@pytest.mark.parametrize("num_experts_per_rank", (32,))
@pytest.mark.parametrize("num_experts_per_token", (8,))
@pytest.mark.parametrize("quant_type", ("none", "fp8_direct_cast"))
def test_dispatch_combine(
    torch_dist_process_manager,
    world_size,
    data_type,
    hidden_dim,
    scale_dim,
    scale_type_size,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    quant_type,
):
    if quant_type == "fp8_direct_cast" and data_type is not torch.bfloat16:
        pytest.skip("fp8_direct_cast is only supported for bfloat16 data type")

    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_dispatch_combine,
                [
                    world_size,
                    data_type,
                    hidden_dim,
                    scale_dim,
                    scale_type_size,
                    max_num_inp_token_per_rank,
                    num_experts_per_rank,
                    num_experts_per_token,
                    quant_type,
                ],
            )
        )

    assert_worker_results(torch_dist_process_manager, world_size)
