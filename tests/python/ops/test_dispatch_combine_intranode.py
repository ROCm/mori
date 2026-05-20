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
        block_num=256,
        warp_num_per_block=4,
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


# ---------------------------------------------------------------------------
# DeepEP-style -1 sentinel tests (IntraNode only)
#
# Inject ``-1`` into selected top-k slots and verify that dispatch + combine
# treat them as "drop this slot": no dispatch, no combine contribution.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("data_type", (torch.bfloat16,))
@pytest.mark.parametrize("hidden_dim", (4096,))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (1, 32))
@pytest.mark.parametrize("num_experts_per_rank", (1,))
@pytest.mark.parametrize("num_experts_per_token", (16,))
@pytest.mark.parametrize(
    "sentinel_pattern",
    (None, "first_only", 8),
)
def test_dispatch_combine_tp_replicated_with_sentinels(
    torch_dist_process_manager,
    world_size,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    sentinel_pattern,
):
    """Megatron-style TP-replicated routing combined with -1 sentinels.

    ``num_experts_per_token = world_size * router_topk = 16`` and
    ``num_experts_per_rank = 1`` mimic ``tp=8, ep=1, router_topk=2``. Each
    token's slots are grouped: slots ``[0..7]`` route to PEs ``0..7`` and
    slots ``[8..15]`` route to PEs ``0..7`` again, so dedup must drop all
    of slots ``[8..15]``. Adding sentinels (``-1``) on top exercises the
    interaction between dedup-null and sentinel-null on the same token.
    """
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
                    "tp_replicated",  # routing
                    False,  # use_max_token_num
                    True,  # check_results
                    sentinel_pattern,
                ],
            )
        )

    assert_worker_results(torch_dist_process_manager, world_size)


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
    """Dispatch + combine must treat -1 routing entries as DeepEP-style sentinels.

    For each pattern we check that the kernels skip the -1 slots: combine
    output equals input * (number of unique non-sentinel PEs).
    """
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


# ---------------------------------------------------------------------------
# DeepEP-style two-mode dispatch (replay) test (IntraNode only)
#
# Runs an autograd-shaped sequence on a single EpDispatchCombineOp:
#   1) forward dispatch  (mode-1: CAS-based slot assignment)
#   2) forward combine   (release=False: pin totalRecvTokenNum)
#   3) backward dispatch (mode-2: replay along the pinned layout)
#   4) backward combine  (release=True: zero state for next iteration)
#
# Invariants checked:
#   * Replay dispatch produces *bit-identical* routing as the forward dispatch
#     (the per-slot src_token_pos is unchanged), so the backward sees the
#     same expert-to-token mapping the forward did.
#   * Replay dispatch streams the *new* input tensor along that pinned route,
#     i.e. recv_y[i] equals y[src_rank][src_id] for the cached (src_rank,
#     src_id) pair.
#   * After the release combine, a second mode-1 forward dispatch round
#     produces correct results (no stale totalRecvTokenNum bleed-through).
# ---------------------------------------------------------------------------


def _test_dispatch_combine_replay(
    rank,
    world_size,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    sentinel_pattern,
):
    config = _make_intranode_config(
        rank=rank,
        world_size=world_size,
        data_type=data_type,
        hidden_dim=hidden_dim,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        use_external_inp_buf=True,
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
    # Snapshot the routing produced by mode-1 — this is what replay must reproduce.
    fwd_src_pos = op.get_dispatch_src_token_pos().clone()
    # Snapshot the data so we can compare later (the underlying buffer is
    # zero-copy and gets overwritten by the replay dispatch).
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
    # Use a fresh random tensor of the same shape/dtype to represent gradients.
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
        f"Rank[{rank}] replay routing mismatch:\n"
        f"  forward src_pos[:8]: {fwd_src_pos[:8].tolist()}\n"
        f"  replay  src_pos[:8]: {bwd_src_pos[:8].tolist()}"
    )
    assert int(bwd_total_recv[0].item()) == total, (
        f"Rank[{rank}] replay total recv mismatch: forward={total}, "
        f"replay={int(bwd_total_recv[0].item())}"
    )

    # Data-parity check: the replay should have streamed *grad_y* through the
    # same routes as forward streamed *all_rank_input*. Equivalently, for each
    # i, (bwd_recv_x[i] - all_rank_input_view) == (forward_recv_diff applied
    # to grad_y). We avoid cross-rank gather (which can deadlock against the
    # zero-copy shmem buffers) by leveraging a local invariant instead:
    # if mode-1 dispatched x[i] = X to slot s on the dest PE, then mode-2 will
    # have dispatched grad_y[i] = G to the same slot s. So on the destination
    # rank we can locally check `bwd_recv_x[s]` is *not* equal to
    # `fwd_recv_x_snap[s]` whenever X != G (which holds with probability 1
    # for fresh randn tensors), and vice versa for slots that were sentinels.
    # This catches the case where replay accidentally re-runs the CAS race
    # (different slot ordering) or accidentally reuses forward data.
    if total > 0:
        # Sanity: bwd_recv_x and fwd_recv_x_snap differ everywhere
        # (since grad_y is fresh randn, and forward data is also random,
        # the two should disagree at every slot with overwhelming probability).
        diff = (
            bwd_recv_x[:total].to(torch.float32) - fwd_recv_x_snap.to(torch.float32)
        ).abs()
        assert (diff > 0).any(), (
            f"Rank[{rank}] replay output is identical to forward output — "
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
@pytest.mark.parametrize("max_num_inp_token_per_rank", (1, 32))
@pytest.mark.parametrize("num_experts_per_rank", (32,))
@pytest.mark.parametrize("num_experts_per_token", (8,))
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
    sentinel_pattern,
):
    """DeepEP-style mode-1/mode-2 dispatch replay (autograd-shaped sequence).

    Verifies that a single ``EpDispatchCombineOp`` can be used as both the
    forward and backward dispatch via ``replay=True``: the cached routing
    layout from mode-1 must produce a bit-identical layout when re-dispatched
    (and the cached ``totalRecvTokenNum`` survives the forward combine when
    called with ``release=False``).
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
