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
"""DeepEP-style routing-handle (cached-mode) tests for IntraNode and InterNodeV1."""
import pytest
import torch
import mori
from tests.python.ops.dispatch_combine_test_utils import (
    EpDispatchCombineTestCase,
    assert_worker_results,
)


def _make_intranode_config(rank, world_size):
    return mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world_size,
        hidden_dim=4096,
        scale_dim=0,
        scale_type_size=1,
        max_num_inp_token_per_rank=32,
        num_experts_per_rank=4,
        num_experts_per_token=4,
        max_token_type_size=4,
        block_num=64,
        warp_num_per_block=4,
        use_external_inp_buf=True,
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
    )


def _make_internode_v1_config(rank, world_size):
    return mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world_size,
        hidden_dim=4096,
        scale_dim=0,
        scale_type_size=1,
        max_num_inp_token_per_rank=32,
        num_experts_per_rank=4,
        num_experts_per_token=4,
        max_token_type_size=2,
        block_num=96,
        rdma_block_num=64,
        warp_num_per_block=8,
        kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1,
        gpu_per_node=world_size,
    )


def _gen_layer_data(test_case, seed):
    test_case.rng.manual_seed(seed)
    return test_case.gen_test_data(use_max_token_num=True)


def _do_dispatch(op, test_data, *, return_routing=False, routing=None):
    rank = op.config.rank
    (_, all_idx, all_inp, all_w, all_s) = test_data
    res = op.dispatch(
        all_inp[rank],
        all_w[rank],
        all_s[rank],
        all_idx[rank],
        return_routing=return_routing,
        routing=routing,
    )
    if return_routing:
        out, out_w, out_s, out_idx, total_recv, R = res
    else:
        out, out_w, out_s, out_idx, total_recv = res
        R = None
    # Clone so a later dispatch on the same op doesn't stomp shared scratch.
    n = int(total_recv[0].item())
    return (
        {
            "out": out[:n].clone(),
            "out_w": out_w[:n].clone() if out_w is not None else None,
            "out_idx": out_idx[:n].clone(),
            "total_recv": int(n),
        },
        R,
    )


def _do_combine(op, dispatch_out, indices_local, *, routing=None):
    out, out_w = op.combine(
        dispatch_out["out"],
        dispatch_out["out_w"],
        dispatch_out["out_idx"],
        routing=routing,
    )
    n = int(indices_local.size(0))
    return out[:n].clone(), (out_w[:n].clone() if out_w is not None else None)


def _default_vs_routing_handle_parity(rank, world_size, kernel):
    """Mode-1 default path (op-owned buffers) vs routing-handle path must match."""
    config = (
        _make_intranode_config(rank, world_size)
        if kernel == "intra"
        else _make_internode_v1_config(rank, world_size)
    )

    op_default = mori.ops.EpDispatchCombineOp(config)
    op_routing = mori.ops.EpDispatchCombineOp(config)

    tc = EpDispatchCombineTestCase(config)
    test_data = _gen_layer_data(tc, seed=4242)

    default_disp, _ = _do_dispatch(op_default, test_data, return_routing=False)
    tc.sync()
    default_combine_out, _ = _do_combine(op_default, default_disp, test_data[1][rank])
    tc.sync()

    rh_disp, R = _do_dispatch(op_routing, test_data, return_routing=True)
    tc.sync()
    rh_combine_out, _ = _do_combine(op_routing, rh_disp, test_data[1][rank], routing=R)
    tc.sync()

    assert torch.allclose(
        default_combine_out.float(), rh_combine_out.float(), atol=1e-3, rtol=1e-3
    ), (
        f"rank {rank}: default-path/routing-handle combine outputs differ; max diff = "
        f"{(default_combine_out.float() - rh_combine_out.float()).abs().max().item()}"
    )


def _replay_correctness(rank, world_size, kernel):
    config = (
        _make_intranode_config(rank, world_size)
        if kernel == "intra"
        else _make_internode_v1_config(rank, world_size)
    )

    op = mori.ops.EpDispatchCombineOp(config)
    tc = EpDispatchCombineTestCase(config)
    test_data = _gen_layer_data(tc, seed=7777)
    rank_idx = test_data[1][rank]

    fwd_disp, R = _do_dispatch(op, test_data, return_routing=True)
    tc.sync()
    pre_disp_dest = R.disp_dest_tok_id_map.clone()
    pre_total_recv = R.total_recv_token_num.clone()

    combine_out, _ = _do_combine(op, fwd_disp, rank_idx, routing=R)
    tc.sync()

    # Mode-2 dispatch with same indices but a different payload.
    rng = torch.Generator(device=torch.device("cuda", rank))
    rng.manual_seed(91234 + rank)
    grad_input = torch.randn(
        test_data[2][rank].shape,
        dtype=test_data[2][rank].dtype,
        device=test_data[2][rank].device,
        generator=rng,
    )
    # Same 5-tuple layout as gen_test_data(); only activations change on this rank.
    grad_test_data = (
        test_data[0],  # num_token per rank (unchanged)
        test_data[1],  # all_rank_indices / expert routing (unchanged for replay)
        [
            grad_input if r == rank else test_data[2][r] for r in range(world_size)
        ],  # all_rank_input: new payload on this rank only
        test_data[3],  # all_rank_weights (unchanged)
        test_data[4],  # all_rank_scales (unchanged)
    )

    rep_disp, _ = _do_dispatch(op, grad_test_data, routing=R)
    tc.sync()

    assert torch.equal(
        R.disp_dest_tok_id_map, pre_disp_dest
    ), f"rank {rank}: dispDestTokIdMap changed after replay dispatch"
    assert torch.equal(
        R.total_recv_token_num, pre_total_recv
    ), f"rank {rank}: totalRecvTokenNum changed after replay dispatch"

    assert rep_disp["total_recv"] == fwd_disp["total_recv"], (
        f"rank {rank}: replay total_recv {rep_disp['total_recv']} differs from "
        f"mode-1 total_recv {fwd_disp['total_recv']}"
    )
    assert combine_out is not None


def _multi_layer_correctness(rank, world_size, kernel, num_layers):
    """Shared op + per-layer routing handles vs fresh op default path per layer."""
    config = (
        _make_intranode_config(rank, world_size)
        if kernel == "intra"
        else _make_internode_v1_config(rank, world_size)
    )

    op_shared = mori.ops.EpDispatchCombineOp(config)
    tc = EpDispatchCombineTestCase(config)

    layer_data = []
    layer_disp = []
    layer_R = []
    for L in range(num_layers):
        td = _gen_layer_data(tc, seed=10_000 + L)
        d, R = _do_dispatch(op_shared, td, return_routing=True)
        tc.sync()
        layer_data.append(td)
        layer_disp.append(d)
        layer_R.append(R)

    layer_combine_out = []
    for L in range(num_layers):
        out, _ = _do_combine(
            op_shared, layer_disp[L], layer_data[L][1][rank], routing=layer_R[L]
        )
        tc.sync()
        layer_combine_out.append(out)

    # Baseline: fresh op per layer on the default (op-owned buffer) path.
    for L in range(num_layers):
        op_baseline = mori.ops.EpDispatchCombineOp(config)
        td = layer_data[L]
        d, _ = _do_dispatch(op_baseline, td, return_routing=False)
        tc.sync()
        out, _ = _do_combine(op_baseline, d, td[1][rank])
        tc.sync()
        assert torch.allclose(
            out.float(), layer_combine_out[L].float(), atol=1e-2, rtol=1e-2
        ), (
            f"rank {rank} layer {L}: shared-op routing-handle combine differs from "
            f"fresh-op default-path baseline; max diff = "
            f"{(out.float() - layer_combine_out[L].float()).abs().max().item()}"
        )


def _stale_symmetric_buffer_guard(rank, world_size):
    """Verify the disp_tok_id_to_src_tok_id_local snapshot is layer-private (IntraNode only)."""
    config = _make_intranode_config(rank, world_size)
    # No-P2P combine reads disp_tok_id_to_src_tok_id_local; external inp buf forces that path.
    config.use_external_inp_buf = False
    tc = EpDispatchCombineTestCase(config)
    op_shared = mori.ops.EpDispatchCombineOp(config)

    td0 = _gen_layer_data(tc, seed=51)
    td1 = _gen_layer_data(tc, seed=151)

    d0, R0 = _do_dispatch(op_shared, td0, return_routing=True)
    tc.sync()
    _ = _do_dispatch(op_shared, td1, return_routing=True)
    tc.sync()

    out_shared, _ = _do_combine(op_shared, d0, td0[1][rank], routing=R0)
    tc.sync()

    # Fresh op, layer 0 only: combine reads op-owned symmetric inverse map (no snapshot).
    op_baseline = mori.ops.EpDispatchCombineOp(config)
    d_base, _ = _do_dispatch(op_baseline, td0, return_routing=False)
    tc.sync()
    out_base, _ = _do_combine(op_baseline, d_base, td0[1][rank])
    tc.sync()

    assert torch.allclose(out_shared.float(), out_base.float(), atol=1e-2, rtol=1e-2), (
        f"rank {rank}: stale-symmetric-buffer guard failed; combine on shared op "
        f"diverged from baseline (max diff = "
        f"{(out_shared.float() - out_base.float()).abs().max().item()})."
    )


@pytest.mark.parametrize("kernel", ("intra", "v1"))
def test_default_vs_routing_handle_parity(torch_dist_process_manager, kernel):
    world_size = 8
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_default_vs_routing_handle_parity, [world_size, kernel])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("kernel", ("intra", "v1"))
def test_replay_correctness(torch_dist_process_manager, kernel):
    world_size = 8
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_replay_correctness, [world_size, kernel])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("kernel", ("intra", "v1"))
@pytest.mark.parametrize("num_layers", (2, 4))
def test_multi_layer_correctness(torch_dist_process_manager, kernel, num_layers):
    world_size = 8
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_multi_layer_correctness, [world_size, kernel, num_layers])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


def test_stale_symmetric_buffer_guard(torch_dist_process_manager):
    world_size = 8
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_stale_symmetric_buffer_guard, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)
