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
import torch
from tests.python.ops.dispatch_combine_test_utils import (
    _all_data_types,
    _is_fp4x2_dtype,
    EpDispatchCombineTestCase,
    assert_worker_results,
    run_ep_dispatch_combine_test,
    start_torch_dist_process_manager,
)

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "4G")


@pytest.fixture(scope="session")
def torch_dist_process_manager():
    manager = start_torch_dist_process_manager(world_size=8)
    yield manager
    manager.shutdown()


def _make_intranode_config(
    rank,
    world_size,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    use_external_inp_buf=True,
    scale_dim=0,
    scale_type_size=1,
    max_total_recv_tokens=0,
    quant_type="none",
):
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
        max_token_type_size=4,
        block_num=40,
        warp_num_per_block=8,
        use_external_inp_buf=use_external_inp_buf,
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
        max_total_recv_tokens=max_total_recv_tokens,
        quant_type=quant_type,
    )


def _test_dispatch_combine(
    rank,
    world_size,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    use_external_inp_buf,
    scale_dim=0,
    scale_type_size=1,
    quant_type="none",
    max_total_recv_tokens=0,
    routing=None,
    use_max_token_num=False,
):
    config = _make_intranode_config(
        rank=rank,
        world_size=world_size,
        data_type=data_type,
        hidden_dim=hidden_dim,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        use_external_inp_buf=use_external_inp_buf,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        quant_type=quant_type,
        max_total_recv_tokens=max_total_recv_tokens,
    )
    run_ep_dispatch_combine_test(
        config,
        EpDispatchCombineTestCase,
        use_max_token_num=use_max_token_num,
        routing=routing,
    )


# TODO: create a sub process group so that we can test worlds size < 8
@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("data_type", _all_data_types())
@pytest.mark.parametrize("hidden_dim", (7168, 4096))
@pytest.mark.parametrize("scale_dim", (0, 32))
@pytest.mark.parametrize("scale_type_size", (1, 4))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (1, 128))
@pytest.mark.parametrize("num_experts_per_rank", (32,))
@pytest.mark.parametrize("num_experts_per_token", (8,))
@pytest.mark.parametrize("use_external_inp_buf", (True, False))
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
    use_external_inp_buf,
    quant_type,
):
    # fp8_direct_cast is not supported in zero-copy mode (use_external_inp_buf=False)
    if quant_type == "fp8_direct_cast" and not use_external_inp_buf:
        pytest.skip("fp8_direct_cast is not supported in zero-copy mode")
    if quant_type == "fp8_direct_cast" and data_type is not torch.bfloat16:
        pytest.skip("fp8_direct_cast is only supported for bfloat16 data type")

    for i in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_dispatch_combine,
                [
                    world_size,
                    data_type,
                    hidden_dim,
                    max_num_inp_token_per_rank,
                    num_experts_per_rank,
                    num_experts_per_token,
                    use_external_inp_buf,
                    scale_dim,
                    scale_type_size,
                    quant_type,
                ],
            )
        )

    assert_worker_results(torch_dist_process_manager, world_size)


# ---------------------------------------------------------------------------
# maxTotalRecvTokens tests (IntraNode only)
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
@pytest.mark.parametrize("use_external_inp_buf", (True, False))
def test_dispatch_combine_max_total_recv_tokens_at_capacity(
    torch_dist_process_manager,
    world_size,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    max_total_recv_tokens,
    num_experts_per_rank,
    use_external_inp_buf,
):
    # spread routing requires num_experts_per_token == world_size
    num_experts_per_token = world_size
    routing = "spread"
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_dispatch_combine,
                [
                    world_size,
                    data_type,
                    hidden_dim,
                    max_num_inp_token_per_rank,
                    num_experts_per_rank,
                    num_experts_per_token,
                    use_external_inp_buf,
                    0,  # scale_dim
                    1,  # scale_type_size
                    "none",  # quant_type
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
@pytest.mark.parametrize("use_external_inp_buf", (True, False))
def test_dispatch_combine_max_total_recv_tokens_under_budget(
    torch_dist_process_manager,
    world_size,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    max_total_recv_tokens,
    num_experts_per_rank,
    use_external_inp_buf,
):
    # spread routing requires num_experts_per_token == world_size
    num_experts_per_token = world_size
    routing = "spread"
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_dispatch_combine,
                [
                    world_size,
                    data_type,
                    hidden_dim,
                    max_num_inp_token_per_rank,
                    num_experts_per_rank,
                    num_experts_per_token,
                    use_external_inp_buf,
                    0,  # scale_dim
                    1,  # scale_type_size
                    "none",  # quant_type
                    max_total_recv_tokens,
                    routing,
                    True,  # use_max_token_num
                ],
            )
        )

    assert_worker_results(torch_dist_process_manager, world_size)


# ---------------------------------------------------------------------------
# Overflow assert test (IntraNode only, isolated subprocess)
# ---------------------------------------------------------------------------


# def _overflow_subprocess(rank, world_size, port):
#     """Run in a spawned subprocess so the device assert doesn't poison the main test process."""
#     from tests.python.utils import TorchDistContext

#     with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
#         mori.shmem.shmem_torch_process_group_init("default")
#         config = mori.ops.EpDispatchCombineConfig(
#             data_type=torch.bfloat16,
#             rank=rank,
#             world_size=world_size,
#             hidden_dim=4096,
#             scale_dim=0,
#             scale_type_size=1,
#             max_token_type_size=4,
#             max_num_inp_token_per_rank=32,
#             num_experts_per_rank=32,
#             num_experts_per_token=1,
#             max_total_recv_tokens=64,  # per-PE limit = ceil(64/8) = 8
#             block_num=40,
#             warp_num_per_block=8,
#             use_external_inp_buf=True,
#             kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
#         )
#         op = mori.ops.EpDispatchCombineOp(config)
#         test_case = EpDispatchCombineTestCase(config)
#         # all_to_one: all tokens -> expert 0 (PE 0).
#         # PE 0 receives 32 * 7 = 224 tokens but per-PE limit is 8 -> overflow.
#         test_data = test_case.gen_test_data(
#             use_max_token_num=True, routing="all_to_one"
#         )
#         (_, all_rank_indices, all_rank_input, all_rank_weights, all_rank_scales) = (
#             test_data
#         )
#         op.dispatch(
#             all_rank_input[rank],
#             all_rank_weights[rank],
#             all_rank_scales[rank],
#             all_rank_indices[rank],
#         )
#         torch.cuda.synchronize()  # device assert surfaces here


# def test_dispatch_combine_overflow_assert():
#     """Verify per-PE overflow assert fires when routing exceeds maxTotalRecvTokens budget."""
#     from tests.python.utils import get_free_port

#     port = get_free_port()
#     # spawn must raise due to device assert in at least one worker
#     with pytest.raises(Exception):
#         torch.multiprocessing.spawn(
#             _overflow_subprocess,
#             args=(8, port),
#             nprocs=8,
#             join=True,
#         )
