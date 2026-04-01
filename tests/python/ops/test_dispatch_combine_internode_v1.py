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
import mori
from tests.python.ops.dispatch_combine_test_utils import (
    _all_data_types,
    _is_fp4x2_dtype,
    EpDispatchCombineTestCase,
    assert_worker_results,
    run_ep_dispatch_combine_test,
    start_torch_dist_process_manager,
)

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "6G")

# Kernel-type string → (EpDispatchCombineKernelType, block_num, rdma_block_num, warp_num_per_block)
_KERNEL_CONFIGS = {
    "internode_v1": (
        mori.ops.EpDispatchCombineKernelType.InterNodeV1,
        96,  # block_num
        64,  # rdma_block_num
        8,  # warp_num_per_block
    ),
    "internode_v1_ll": (
        mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
        256,  # block_num
        128,  # rdma_block_num
        8,  # warp_num_per_block
    ),
}


class InterNodeV1DispatchCombineTestCase(EpDispatchCombineTestCase):
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
        ) = op.dispatch(
            all_rank_input[self.config.rank],
            all_rank_weights[self.config.rank],
            all_rank_scales[self.config.rank],
            all_rank_indices[self.config.rank],
        )
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

        combine_output, combine_output_weight = op.combine(
            dispatch_output, dispatch_weights, dispatch_indices, call_reset=False
        )
        self.sync()
        self.check_combine_result(op, test_data, combine_output, combine_output_weight)


@pytest.fixture(scope="session")
def torch_dist_process_manager():
    manager = start_torch_dist_process_manager(world_size=8, disable_p2p=True)
    yield manager
    manager.shutdown()


def _make_internode_v1_config(
    rank,
    world_size,
    kernel_type_str,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    gpu_per_node,
    scale_dim=0,
    scale_type_size=1,
    max_total_recv_tokens=0,
):
    kernel_type, block_num, rdma_block_num, warp_num_per_block = _KERNEL_CONFIGS[
        kernel_type_str
    ]
    return mori.ops.EpDispatchCombineConfig(
        data_type=data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim // 2 if _is_fp4x2_dtype(data_type) else hidden_dim,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        max_token_type_size=2,
        block_num=block_num,
        rdma_block_num=rdma_block_num,
        warp_num_per_block=warp_num_per_block,
        kernel_type=kernel_type,
        gpu_per_node=gpu_per_node,
        max_total_recv_tokens=max_total_recv_tokens,
    )


def _test_dispatch_combine(
    rank,
    world_size,
    kernel_type_str,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    gpu_per_node,
    scale_dim=0,
    scale_type_size=1,
    max_total_recv_tokens=0,
    routing=None,
    use_max_token_num=False,
):
    config = _make_internode_v1_config(
        rank=rank,
        world_size=world_size,
        kernel_type_str=kernel_type_str,
        data_type=data_type,
        hidden_dim=hidden_dim,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        gpu_per_node=gpu_per_node,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        max_total_recv_tokens=max_total_recv_tokens,
    )
    run_ep_dispatch_combine_test(
        config,
        InterNodeV1DispatchCombineTestCase,
        use_max_token_num=use_max_token_num,
        routing=routing,
    )


# TODO: create a sub process group so that we can test world size < 8
@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("kernel_type", ("internode_v1", "internode_v1_ll"))
@pytest.mark.parametrize("data_type", _all_data_types())
@pytest.mark.parametrize("hidden_dim", (7168, 4096))
@pytest.mark.parametrize("scale_dim", (0, 56))
@pytest.mark.parametrize("scale_type_size", (1, 4))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (32, 128))
@pytest.mark.parametrize("num_experts_per_rank", (32,))
@pytest.mark.parametrize("num_experts_per_token", (8,))
# gpu_per_node=8: 1 node × 8 GPUs (exercises intranode paths within the kernels)
# gpu_per_node=4: 2 nodes × 4 GPUs (exercises actual internode/RDMA paths)
@pytest.mark.parametrize("gpu_per_node", (8,))
def test_dispatch_combine(
    torch_dist_process_manager,
    world_size,
    kernel_type,
    data_type,
    hidden_dim,
    scale_dim,
    scale_type_size,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    gpu_per_node,
):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_dispatch_combine,
                [
                    world_size,
                    kernel_type,
                    data_type,
                    hidden_dim,
                    max_num_inp_token_per_rank,
                    num_experts_per_rank,
                    num_experts_per_token,
                    gpu_per_node,
                    scale_dim,
                    scale_type_size,
                    0,  # max_total_recv_tokens
                    None,  # routing
                    True,  # use_max_token_num
                ],
            )
        )

    assert_worker_results(torch_dist_process_manager, world_size)


# ---------------------------------------------------------------------------
# maxTotalRecvTokens tests (InterNodeV1 / InterNodeV1LL)
#
# "spread" routing: each token sends 1 expert to every rank, so after per-rank
# deduplication every rank receives all source tokens.
# actual recv = max_num_inp_token_per_rank * world_size  (true worst case)
# ---------------------------------------------------------------------------


# at_capacity: routing=spread → recv = max_num_inp_token_per_rank * world_size
# (max_num_inp_token_per_rank, max_total_recv_tokens):
#   (32, 0)   → unlimited buffer handles full load of 32*8=256 tokens
#   (32, 256) → exact-fit buffer sized to 256, exactly 256 tokens arrive
@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("kernel_type", ("internode_v1", "internode_v1_ll"))
@pytest.mark.parametrize("data_type", _all_data_types())
@pytest.mark.parametrize("hidden_dim", (7168, 4096))
@pytest.mark.parametrize(
    "max_num_inp_token_per_rank, max_total_recv_tokens",
    [
        (32, 0),  # unlimited: verify a fully-loaded buffer works with no cap
        (32, 256),  # exact worst case: buffer=256, recv=32*8=256 tokens arrive
    ],
)
@pytest.mark.parametrize("num_experts_per_rank", (32,))
def test_dispatch_combine_max_total_recv_tokens_at_capacity(
    torch_dist_process_manager,
    world_size,
    kernel_type,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    max_total_recv_tokens,
    num_experts_per_rank,
):
    # spread routing requires num_experts_per_token == world_size
    num_experts_per_token = world_size
    gpu_per_node = world_size
    routing = "spread"
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_dispatch_combine,
                [
                    world_size,
                    kernel_type,
                    data_type,
                    hidden_dim,
                    max_num_inp_token_per_rank,
                    num_experts_per_rank,
                    num_experts_per_token,
                    gpu_per_node,
                    0,  # scale_dim
                    1,  # scale_type_size
                    max_total_recv_tokens,
                    routing,
                    True,  # use_max_token_num
                ],
            )
        )

    assert_worker_results(torch_dist_process_manager, world_size)


# under_budget: routing=spread → recv = max_num_inp_token_per_rank * world_size
# (max_num_inp_token_per_rank, max_total_recv_tokens):
#   (1,  128) → recv=1*8=8,   well under the 128 budget
#   (16, 128) → recv=16*8=128, exactly at the 128 budget
@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("kernel_type", ("internode_v1", "internode_v1_ll"))
@pytest.mark.parametrize("data_type", _all_data_types())
@pytest.mark.parametrize("hidden_dim", (7168, 4096))
@pytest.mark.parametrize(
    "max_num_inp_token_per_rank, max_total_recv_tokens",
    [
        (1, 128),  # recv=1*8=8,   well under the 128 budget
        (16, 128),  # recv=16*8=128, exactly at the 128 budget
    ],
)
@pytest.mark.parametrize("num_experts_per_rank", (32,))
def test_dispatch_combine_max_total_recv_tokens_under_budget(
    torch_dist_process_manager,
    world_size,
    kernel_type,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    max_total_recv_tokens,
    num_experts_per_rank,
):
    # spread routing requires num_experts_per_token == world_size
    num_experts_per_token = world_size
    gpu_per_node = world_size
    routing = "spread"
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_dispatch_combine,
                [
                    world_size,
                    kernel_type,
                    data_type,
                    hidden_dim,
                    max_num_inp_token_per_rank,
                    num_experts_per_rank,
                    num_experts_per_token,
                    gpu_per_node,
                    0,  # scale_dim
                    1,  # scale_type_size
                    max_total_recv_tokens,
                    routing,
                    True,  # use_max_token_num
                ],
            )
        )

    assert_worker_results(torch_dist_process_manager, world_size)
