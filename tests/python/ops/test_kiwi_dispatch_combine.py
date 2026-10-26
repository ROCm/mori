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
import pytest
import torch
import torch.distributed as dist
from mori.ops import dispatch_combine_kiwi
from tests.python.ops.dispatch_combine_test_utils import (
    _all_data_types,
    cross_dtype_hidden_dims,
    EpDispatchCombineTestCase,
    assert_worker_results,
)

if not dispatch_combine_kiwi.is_available():
    pytest.skip("mori was built without BUILD_OPS_KIWI", allow_module_level=True)


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
    rcomp_backend,
):
    # max_token_type_size=2 makes KIWI combine in bfloat16 for every dispatch dtype.
    combine_data_type = torch.bfloat16
    dispatch_hidden_dim, combine_hidden_dim, _ = cross_dtype_hidden_dims(
        hidden_dim, data_type, combine_data_type
    )
    config = dispatch_combine_kiwi.EpDispatchCombineKiwiConfig(
        data_type=data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        max_token_type_size=2,
    )
    op = dispatch_combine_kiwi.EpDispatchCombineKiwiOp(config, rcomp_backend=rcomp_backend)
    test_case = EpDispatchCombineTestCase(
        config,
        combine_data_type=combine_data_type,
        combine_hidden_dim=combine_hidden_dim,
        dispatch_hidden_dim=dispatch_hidden_dim,
    )
    num_reps = 8
    for idx in range(num_reps):
        test_data = test_case.gen_test_data(use_max_token_num=idx == 0)
        test_case.run_test_once(op, test_data)
        if rank == 0:
            print(f"Passed {idx}/{num_reps}")
    del op
    dist.barrier()


# TODO: create a sub process group so that we can test worlds size < 8
@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("data_type", _all_data_types())
@pytest.mark.parametrize("hidden_dim", (7168,))
@pytest.mark.parametrize("scale_dim", (56,))
@pytest.mark.parametrize("scale_type_size", (4,))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (1, 32, 128))
@pytest.mark.parametrize("num_experts_per_rank", (32,))
@pytest.mark.parametrize("num_experts_per_token", (8,))
@pytest.mark.parametrize("rcomp_backend", ("rdma-only", "hybrid"))
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
    rcomp_backend,
):
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
                    rcomp_backend,
                ],
            )
        )

    assert_worker_results(torch_dist_process_manager, world_size)
