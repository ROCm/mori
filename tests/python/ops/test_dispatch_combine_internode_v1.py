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
    _is_fp4x2_dtype,
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


# ---------------------------------------------------------------------------
# DeepEP-style two-mode dispatch (replay) test (InterNodeV1 only)
#
# Same shape as the IntraNode replay test, but exercises the InterNodeV1 RDMA
# path. With gpu_per_node == world_size the V1 kernel only takes the XGMI
# (intra-node) branch, which mirrors what the IntraNode test covers but
# through the V1 launch sequence (EpDispatchCopyToStaging + V1 dispatch +
# EpCombineSync + EpCombineSyncBarrier + V1 combine + EpCombineAll). Smaller
# gpu_per_node values exercise the cached interNodeDispSendMap /
# interNodeDispDestTokIdMap replay paths but require multi-node test
# scaffolding to actually move data over RDMA.
# ---------------------------------------------------------------------------
def _test_dispatch_combine_replay(
    rank,
    world_size,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    gpu_per_node,
    sentinel_pattern,
):
    config = _make_internode_v1_config(
        rank=rank,
        world_size=world_size,
        kernel_type_str="internode_v1",
        data_type=data_type,
        hidden_dim=hidden_dim,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        gpu_per_node=gpu_per_node,
    )
    op = mori.ops.EpDispatchCombineOp(config)
    test_case = EpDispatchCombineTestCase(config)
    test_data = test_case.gen_test_data(sentinel_pattern=sentinel_pattern)
    (
        all_rank_num_token,
        all_rank_indices,
        all_rank_input,
        all_rank_weights,
        all_rank_scales,
    ) = test_data

    # ----- Forward dispatch (mode-1) + forward combine (pin layout) -----
    (
        fwd_recv_x,
        fwd_recv_w,
        fwd_recv_s,
        fwd_recv_idx,
        fwd_total_recv,
    ) = op.dispatch(
        all_rank_input[rank],
        all_rank_weights[rank],
        all_rank_scales[rank],
        all_rank_indices[rank],
    )
    test_case.sync()
    total = int(fwd_total_recv[0].item())
    fwd_src_pos = op.get_dispatch_src_token_pos().clone()
    fwd_recv_x_snap = fwd_recv_x[:total].clone()

    fwd_out, _ = op.combine(
        fwd_recv_x,
        fwd_recv_w,
        fwd_recv_idx,
        call_reset=False,
        release=False,  # pin totalRecvTokenNum across this combine
    )
    test_case.sync()

    # ----- Backward dispatch (mode-2 replay) with a *different* input -----
    grad_y = torch.randn_like(all_rank_input[rank].to(torch.float32)).to(
        all_rank_input[rank].dtype
    )
    (bwd_recv_x, _, _, _, bwd_total_recv) = op.dispatch(
        grad_y,
        all_rank_weights[rank],
        all_rank_scales[rank],
        all_rank_indices[rank],
        replay=True,
    )
    test_case.sync()

    # Routing must be identical to forward.
    bwd_src_pos = op.get_dispatch_src_token_pos()
    assert torch.equal(fwd_src_pos, bwd_src_pos), (
        f"Rank[{rank}] InterNodeV1 replay routing mismatch:\n"
        f"  forward src_pos[:8]: {fwd_src_pos[:8].tolist()}\n"
        f"  replay  src_pos[:8]: {bwd_src_pos[:8].tolist()}"
    )
    assert int(bwd_total_recv[0].item()) == total, (
        f"Rank[{rank}] InterNodeV1 replay total recv mismatch: forward={total}, "
        f"replay={int(bwd_total_recv[0].item())}"
    )

    # Sanity: replay output must differ from forward (both are random tensors,
    # so element-wise equality is essentially impossible).
    if total > 0:
        diff = (
            bwd_recv_x[:total].to(torch.float32) - fwd_recv_x_snap.to(torch.float32)
        ).abs()
        assert (diff > 0).any(), (
            f"Rank[{rank}] InterNodeV1 replay output is identical to forward output — "
            "the replay either didn't run or re-used forward data."
        )

    # ----- Backward combine (release=True default) -----
    bwd_out, _ = op.combine(
        bwd_recv_x,
        fwd_recv_w,
        fwd_recv_idx,
        call_reset=False,
    )
    test_case.sync()

    # ----- Second forward round on the same op: must work cleanly -----
    test_data2 = test_case.gen_test_data(sentinel_pattern=sentinel_pattern)
    (
        _,
        all_rank_indices2,
        all_rank_input2,
        all_rank_weights2,
        all_rank_scales2,
    ) = test_data2
    (
        fwd_recv_x2,
        fwd_recv_w2,
        _,
        fwd_recv_idx2,
        _,
    ) = op.dispatch(
        all_rank_input2[rank],
        all_rank_weights2[rank],
        all_rank_scales2[rank],
        all_rank_indices2[rank],
    )
    test_case.sync()
    fwd_out2, _ = op.combine(
        fwd_recv_x2,
        fwd_recv_w2,
        fwd_recv_idx2,
        call_reset=False,
    )
    test_case.sync()
    test_case.check_combine_result(op, test_data2, fwd_out2)


@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("data_type", (torch.bfloat16,))
@pytest.mark.parametrize("hidden_dim", (4096,))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (32,))
@pytest.mark.parametrize("num_experts_per_rank", (32,))
@pytest.mark.parametrize("num_experts_per_token", (8,))
@pytest.mark.parametrize("gpu_per_node", (8,))
@pytest.mark.parametrize(
    "sentinel_pattern",
    (None, "every_other", 1),
)
def test_dispatch_combine_replay(
    torch_dist_process_manager,
    world_size,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    gpu_per_node,
    sentinel_pattern,
):
    """DeepEP-style mode-1/mode-2 dispatch replay through the InterNodeV1 kernel.

    Verifies that a single ``EpDispatchCombineOp`` configured with InterNodeV1
    can be used as both the forward and backward dispatch via ``replay=True``:
    the cached routing layout from mode-1 must produce a bit-identical layout
    when re-dispatched, and ``EpCombineAll`` must skip its
    ``totalRecvTokenNum`` reset when ``release=False``.
    """
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_dispatch_combine_replay,
                [
                    world_size,
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
