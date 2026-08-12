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
import mori
from tests.python.ops.dispatch_combine_test_utils import (
    _all_data_types,
    cross_dtype_hidden_dims,
    cross_dtype_skip_reason,
    EpDispatchCombineTestCase,
    assert_worker_results,
    run_ep_dispatch_combine_test,
    run_ep_dispatch_local_expert_count_test,
)

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
    quant_type="none",
    combine_data_type=None,
):
    kernel_type, block_num, rdma_block_num, warp_num_per_block = _KERNEL_CONFIGS[
        kernel_type_str
    ]
    _, _, config_hidden_dim = cross_dtype_hidden_dims(
        hidden_dim, data_type, combine_data_type or data_type
    )
    return mori.ops.EpDispatchCombineConfig(
        data_type=data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=config_hidden_dim,
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
        quant_type=quant_type,
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
    check_results=True,
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
        EpDispatchCombineTestCase,
        use_max_token_num=use_max_token_num,
        routing=routing,
        check_results=check_results,
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
# Quantized / cross-dtype coverage for the InterNodeV1 kernels.
#
# The matrix above sweeps dtypes but leaves quant_type at its default (none)
# and derives the combine element type from config.data_type, so the quantized
# internode configurations the tuning matrix actually ships
# (tools/run_all_internode_tuning.sh pairs v1 and v1_ll with FP4/FP8 dispatch
# and a BF16 combine) had no correctness coverage at all. These tests add the
# quant_type axis and an independent combine dtype.
# ---------------------------------------------------------------------------


def _test_dispatch_combine_cross_dtype(
    rank,
    world_size,
    kernel_type_str,
    data_type,
    combine_data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    gpu_per_node,
    quant_type,
):
    dispatch_hidden_dim, combine_hidden_dim, _ = cross_dtype_hidden_dims(
        hidden_dim, data_type, combine_data_type
    )
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
        quant_type=quant_type,
        combine_data_type=combine_data_type,
    )
    run_ep_dispatch_combine_test(
        config,
        EpDispatchCombineTestCase,
        use_max_token_num=True,
        combine_data_type=combine_data_type,
        combine_hidden_dim=combine_hidden_dim,
        dispatch_hidden_dim=dispatch_hidden_dim,
    )


@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("kernel_type", ("internode_v1", "internode_v1_ll"))
@pytest.mark.parametrize("data_type", _all_data_types())
@pytest.mark.parametrize("combine_data_type", (torch.bfloat16,), ids=("combine_bf16",))
@pytest.mark.parametrize("hidden_dim", (7168, 4096))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (128,))
@pytest.mark.parametrize("num_experts_per_rank", (32,))
@pytest.mark.parametrize("num_experts_per_token", (8,))
@pytest.mark.parametrize("quant_type", ("none", "fp8_direct_cast"))
# gpu_per_node=8 keeps this on a single node; gpu_per_node=4 would additionally
# route half the traffic over RDMA (see the note on test_dispatch_combine).
@pytest.mark.parametrize("gpu_per_node", (8,))
def test_dispatch_combine_cross_dtype(
    torch_dist_process_manager,
    world_size,
    kernel_type,
    data_type,
    combine_data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    quant_type,
    gpu_per_node,
):
    skip = cross_dtype_skip_reason(quant_type, data_type, combine_data_type)
    if skip:
        pytest.skip(skip)

    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_dispatch_combine_cross_dtype,
                [
                    world_size,
                    kernel_type,
                    data_type,
                    combine_data_type,
                    hidden_dim,
                    max_num_inp_token_per_rank,
                    num_experts_per_rank,
                    num_experts_per_token,
                    gpu_per_node,
                    quant_type,
                ],
            )
        )

    assert_worker_results(torch_dist_process_manager, world_size)


def _test_blockwise_combine_unsupported(
    rank,
    world_size,
    kernel_type_str,
    quant_type,
):
    config = _make_internode_v1_config(
        rank=rank,
        world_size=world_size,
        kernel_type_str=kernel_type_str,
        data_type=torch.bfloat16,
        hidden_dim=4096,
        max_num_inp_token_per_rank=64,
        num_experts_per_rank=32,
        num_experts_per_token=8,
        gpu_per_node=world_size,
        quant_type=quant_type,
    )
    op = mori.ops.EpDispatchCombineOp(config)
    device = torch.device("cuda", rank)
    inp = torch.zeros(64, 4096, dtype=torch.bfloat16, device=device)
    indices = torch.zeros(64, 8, dtype=torch.int32, device=device)
    weights = torch.ones(64, 8, dtype=torch.float32, device=device)

    dispatch_output, dispatch_weights, _, _, _ = op.dispatch(
        inp, weights, None, indices
    )
    try:
        op.combine(dispatch_output, dispatch_weights, indices, call_reset=False)
    except ValueError as e:
        assert "only supports" in str(e), f"unexpected rejection message: {e}"
        return
    raise AssertionError(
        f"{quant_type} combine unexpectedly succeeded on {kernel_type_str}. "
        "Blockwise combine transports per-block scales alongside the token and "
        "has no internode kernel (only bf16_nop2p_fp8bwq_* / _fp4bwq_* are "
        "registered, all IntraNode). If internode blockwise combine has now "
        "been implemented, replace this test with real numerical coverage in "
        "test_dispatch_combine_cross_dtype."
    )


# Pins the internode blockwise-combine gap. fp8/fp4 blockwise quant is
# implemented for IntraNode (and, for fp8, AsyncLL) combine only; the internode
# kernels have no blockwise variant registered. Asserting the rejection keeps
# the boundary explicit and makes this test fail loudly -- with instructions --
# the moment internode support lands, rather than leaving the new path silently
# untested.
@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("kernel_type", ("internode_v1", "internode_v1_ll"))
@pytest.mark.parametrize("quant_type", ("fp8_blockwise", "fp4_blockwise"))
def test_blockwise_combine_unsupported_internode(
    torch_dist_process_manager,
    world_size,
    kernel_type,
    quant_type,
):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_blockwise_combine_unsupported,
                [world_size, kernel_type, quant_type],
            )
        )

    assert_worker_results(torch_dist_process_manager, world_size)


# ---------------------------------------------------------------------------
# -1 routing sentinel tests (InterNodeV1, default dispatch/combine path)
# gpu_per_node=4: 2 nodes × 4 GPUs — exercises RDMA send/recv with sentinels
# gpu_per_node=8: 1 node × 8 GPUs — XGMI-only paths inside InterNodeV1
# ---------------------------------------------------------------------------


def _test_dispatch_combine_sentinel(
    rank,
    world_size,
    kernel_type_str,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    gpu_per_node,
    sentinel_pattern,
    scale_dim=0,
    scale_type_size=1,
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
    )
    run_ep_dispatch_combine_test(
        config,
        EpDispatchCombineTestCase,
        sentinel_pattern=sentinel_pattern,
    )


@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("kernel_type", ("internode_v1",))
@pytest.mark.parametrize("data_type", (torch.bfloat16,))
@pytest.mark.parametrize("hidden_dim", (4096,))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (1, 32))
@pytest.mark.parametrize("num_experts_per_rank", (4,))
@pytest.mark.parametrize("num_experts_per_token", (4,))
@pytest.mark.parametrize("gpu_per_node", (4, 8))
@pytest.mark.parametrize(
    "sentinel_pattern",
    ("every_other", "first_only", 1),
)
def test_dispatch_combine_minus_one_sentinel(
    torch_dist_process_manager,
    world_size,
    kernel_type,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    gpu_per_node,
    sentinel_pattern,
):
    """InterNodeV1 dispatch + combine must skip -1 routing sentinel entries."""
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_dispatch_combine_sentinel,
                [
                    world_size,
                    kernel_type,
                    data_type,
                    hidden_dim,
                    max_num_inp_token_per_rank,
                    num_experts_per_rank,
                    num_experts_per_token,
                    gpu_per_node,
                    sentinel_pattern,
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


# ---------------------------------------------------------------------------
# Large token num test (InterNodeV1 / InterNodeV1LL)
#
# Stress-test with 65536 tokens per rank (512K tokens total across 8 ranks)
# and hidden_dim=7168.  Only checks that dispatch+combine complete without
# error; correctness checks are skipped because they are too slow at this scale.
# ---------------------------------------------------------------------------


def test_dispatch_combine_large_token_num(
    torch_dist_process_manager,
):
    """Dispatch + combine with max_num_inp_token_per_rank=65536, hidden_dim=7168.

    Tested for both InterNodeV1 and InterNodeV1LL kernel types.
    Correctness is not verified — only that the kernel completes without error.
    """
    world_size = 8
    for kernel_type in ("internode_v1", "internode_v1_ll"):
        for _ in range(world_size):
            torch_dist_process_manager.task_queue.put(
                (
                    _test_dispatch_combine,
                    [
                        world_size,
                        kernel_type,
                        torch.bfloat16,  # data_type
                        7168,  # hidden_dim
                        65536,  # max_num_inp_token_per_rank
                        32,  # num_experts_per_rank
                        8,  # num_experts_per_token
                        8,  # gpu_per_node
                        0,  # scale_dim
                        1,  # scale_type_size
                        0,  # max_total_recv_tokens
                        None,  # routing
                        True,  # use_max_token_num
                        False,  # check_results
                    ],
                )
            )

        assert_worker_results(torch_dist_process_manager, world_size)


# local_expert_count tests (InterNodeV1 / InterNodeV1LL)
# ---------------------------------------------------------------------------


def _test_dispatch_local_expert_count(
    rank,
    world_size,
    kernel_type_str,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    gpu_per_node,
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
    )
    run_ep_dispatch_local_expert_count_test(config)


@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("kernel_type", ("internode_v1", "internode_v1_ll"))
@pytest.mark.parametrize("data_type", (torch.bfloat16,))
@pytest.mark.parametrize("hidden_dim", (4096,))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (1, 32))
@pytest.mark.parametrize("num_experts_per_rank", (32,))
@pytest.mark.parametrize("num_experts_per_token", (8,))
@pytest.mark.parametrize("gpu_per_node", (8,))
def test_dispatch_local_expert_count(
    torch_dist_process_manager,
    world_size,
    kernel_type,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    gpu_per_node,
):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_dispatch_local_expert_count,
                [
                    world_size,
                    kernel_type,
                    data_type,
                    hidden_dim,
                    max_num_inp_token_per_rank,
                    num_experts_per_rank,
                    num_experts_per_token,
                    gpu_per_node,
                ],
            )
        )

    assert_worker_results(torch_dist_process_manager, world_size)
