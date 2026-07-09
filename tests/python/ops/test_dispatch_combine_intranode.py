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
import mori
import torch
from tests.python.ops.dispatch_combine_test_utils import (
    _all_data_types,
    _is_fp4x2_dtype,
    EpDispatchCombineTestCase,
    assert_worker_results,
    run_ep_dispatch_combine_test,
    run_ep_dispatch_local_expert_count_test,
)


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
    kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
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
        block_num=64,
        warp_num_per_block=4,
        use_external_inp_buf=use_external_inp_buf,
        kernel_type=kernel_type,
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
    check_results=True,
    sentinel_pattern=None,
    weightless=False,
    expect_combine_kernel_substr=None,
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
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
    )
    run_ep_dispatch_combine_test(
        config,
        EpDispatchCombineTestCase,
        use_max_token_num=use_max_token_num,
        routing=routing,
        check_results=check_results,
        weightless=weightless,
        expect_combine_kernel_substr=expect_combine_kernel_substr,
    )


def _test_dispatch_combine_ll(
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
    check_results=True,
    sentinel_pattern=None,
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
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNodeLL,
    )
    run_ep_dispatch_combine_test(
        config,
        EpDispatchCombineTestCase,
        use_max_token_num=use_max_token_num,
        routing=routing,
        check_results=check_results,
        sentinel_pattern=sentinel_pattern,
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
@pytest.mark.parametrize(
    "quant_type", ("none", "fp8_direct_cast", "fp8_blockwise", "fp4_blockwise")
)
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
    if quant_type in ("fp8_blockwise", "fp4_blockwise"):
        if data_type is not torch.bfloat16:
            pytest.skip(f"{quant_type} only supports bfloat16 input")
        if not use_external_inp_buf:
            pytest.skip(f"{quant_type} requires use_external_inp_buf=True")
        # blockwise combine ignores scale_dim/scale_type_size (driven by
        # MORI_FP8_COMBINE_SCALE_DIM internally). fp4_blockwise reuses the same path
        # but transports packed FP4 (E2M1) instead of FP8.

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


# Exercises the fast fp8_blockwise weightless vec8 combine kernels, including the
# topk=9 (AccumNum=9) shared-experts-fusion specialization. hidden_dim=7168 with the
# default MORI_FP8_COMBINE_SCALE_DIM=56 yields block_elems=128, so the gate fires and
# we assert the specialized kernel was actually selected (guards against silent
# fallback to the generic path, which would produce identical weightless numbers).
@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("data_type", (torch.bfloat16,))
@pytest.mark.parametrize("hidden_dim", (7168,))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (128,))
@pytest.mark.parametrize("num_experts_per_rank", (32,))
@pytest.mark.parametrize("num_experts_per_token", (8, 9))
@pytest.mark.parametrize("use_external_inp_buf", (True,))
@pytest.mark.parametrize("quant_type", ("fp8_blockwise", "fp4_blockwise"))
def test_dispatch_combine_weightless_vec8(
    torch_dist_process_manager,
    world_size,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    use_external_inp_buf,
    quant_type,
):
    expect_combine_kernel_substr = (
        "noweight_block128_vec8_top9"
        if num_experts_per_token == 9
        else "noweight_block128_vec8"
    )
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
                    0,  # scale_dim
                    1,  # scale_type_size
                    quant_type,
                    0,  # max_total_recv_tokens
                    None,  # routing
                    False,  # use_max_token_num
                    True,  # check_results
                    None,  # sentinel_pattern
                    True,  # weightless
                    expect_combine_kernel_substr,
                ],
            )
        )

    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("data_type", _all_data_types())
@pytest.mark.parametrize("hidden_dim", (7168, 4096))
@pytest.mark.parametrize("scale_dim", (0, 32))
@pytest.mark.parametrize("scale_type_size", (1, 4))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (1, 64))
@pytest.mark.parametrize("num_experts_per_rank", (32,))
@pytest.mark.parametrize("num_experts_per_token", (8,))
@pytest.mark.parametrize("use_external_inp_buf", (True, False))
@pytest.mark.parametrize("quant_type", ("none", "fp8_direct_cast", "fp8_blockwise"))
def test_dispatch_combine_ll(
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
    if quant_type in ("fp8_blockwise", "fp4_blockwise"):
        if data_type is not torch.bfloat16:
            pytest.skip(f"{quant_type} only supports bfloat16 input")
        if not use_external_inp_buf:
            pytest.skip(f"{quant_type} requires use_external_inp_buf=True")
        # blockwise combine ignores scale_dim/scale_type_size (driven by
        # MORI_FP8_COMBINE_SCALE_DIM internally). fp4_blockwise reuses the same path
        # but transports packed FP4 (E2M1) instead of FP8.

    for i in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_dispatch_combine_ll,
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
# local_expert_count tests (IntraNode only)
# ---------------------------------------------------------------------------


def _test_dispatch_local_expert_count(
    rank,
    world_size,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
):
    config = _make_intranode_config(
        rank=rank,
        world_size=world_size,
        data_type=data_type,
        hidden_dim=hidden_dim,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
    )
    run_ep_dispatch_local_expert_count_test(config)


@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("data_type", (torch.bfloat16,))
@pytest.mark.parametrize("hidden_dim", (4096,))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (1, 32))
@pytest.mark.parametrize("num_experts_per_rank", (32,))
@pytest.mark.parametrize("num_experts_per_token", (8,))
def test_dispatch_local_expert_count(
    torch_dist_process_manager,
    world_size,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_dispatch_local_expert_count,
                [
                    world_size,
                    data_type,
                    hidden_dim,
                    max_num_inp_token_per_rank,
                    num_experts_per_rank,
                    num_experts_per_token,
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
# Large token num test (IntraNode only)
#
# Stress-test with 65536 tokens per rank (512K tokens total across 8 ranks)
# and hidden_dim=7168.  Only checks that dispatch+combine complete without
# error; correctness checks are skipped because they are too slow at this scale.
# ---------------------------------------------------------------------------


# -1 routing sentinel tests (IntraNode only)


@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("data_type", (torch.bfloat16,))
@pytest.mark.parametrize("hidden_dim", (4096,))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (1, 32))
@pytest.mark.parametrize("num_experts_per_rank", (32,))
@pytest.mark.parametrize("num_experts_per_token", (8,))
@pytest.mark.parametrize(
    "sentinel_pattern",
    ("every_other", "first_only", 1, 7),
)
def test_dispatch_combine_minus_one_sentinel(
    torch_dist_process_manager,
    world_size,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    sentinel_pattern,
):
    """Dispatch + combine must treat -1 routing entries as routing sentinels."""
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
                    True,  # use_external_inp_buf
                    0,  # scale_dim
                    1,  # scale_type_size
                    "none",  # quant_type
                    0,  # max_total_recv_tokens
                    None,  # routing (default random)
                    False,  # use_max_token_num
                    True,  # check_results
                    sentinel_pattern,
                ],
            )
        )

    assert_worker_results(torch_dist_process_manager, world_size)


def test_dispatch_combine_large_token_num(
    torch_dist_process_manager,
):
    """Dispatch + combine with max_num_inp_token_per_rank=65536, hidden_dim=7168.

    Correctness is not verified — only that the kernel completes without error.
    """
    world_size = 8
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_dispatch_combine,
                [
                    world_size,
                    torch.bfloat16,  # data_type
                    7168,  # hidden_dim
                    65536,  # max_num_inp_token_per_rank
                    32,  # num_experts_per_rank
                    8,  # num_experts_per_token
                    True,  # use_external_inp_buf
                    0,  # scale_dim
                    1,  # scale_type_size
                    "none",  # quant_type
                    0,  # max_total_recv_tokens
                    None,  # routing
                    True,  # use_max_token_num
                    False,  # check_results
                ],
            )
        )

    assert_worker_results(torch_dist_process_manager, world_size)
