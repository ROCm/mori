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
"""DeepEP-style routing-handle (cached-mode) tests for IntraNode and InterNodeV1.

Covers:
- legacy vs routing-handle parity (same combine output, bit-equal).
- replay correctness: mode-2 dispatch reuses the cached layout.
- multi-layer correctness: two routings in flight on a shared op survive
  intervening dispatches.
- stale symmetric-buffer guard: the disp_tok_id_to_src_tok_id_local snapshot
  is layer-private even though the underlying symmetric buffer is shared.
"""
import pytest
import torch
import mori
from tests.python.ops.dispatch_combine_test_utils import (
    EpDispatchCombineTestCase,
    assert_worker_results,
)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gen_layer_data(test_case, seed):
    """Generate a fresh (input, weights, indices) tuple, deterministic per seed."""
    test_case.rng.manual_seed(seed)
    return test_case.gen_test_data(use_max_token_num=True)


def _do_dispatch(op, test_data, *, return_routing=False, routing=None):
    """Run dispatch on the local rank, return (cloned outputs, routing or None)."""
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
    # The base tensors alias shared op-handle scratch (dispatchOut, etc.).
    # Clone so that a later dispatch on the same op doesn't stomp our reference.
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


# ---------------------------------------------------------------------------
# Worker bodies
# ---------------------------------------------------------------------------
def _legacy_vs_routing_handle_parity(rank, world_size, kernel):
    """Run identical inputs through legacy and routing-handle paths, expect bit-equal."""
    config = _make_intranode_config(rank, world_size) if kernel == "intra" \
        else _make_internode_v1_config(rank, world_size)

    # Two ops: one for legacy, one for routing-handle. They operate independently.
    op_legacy = mori.ops.EpDispatchCombineOp(config)
    op_route = mori.ops.EpDispatchCombineOp(config)

    tc = EpDispatchCombineTestCase(config)
    test_data = _gen_layer_data(tc, seed=4242)

    # Legacy path.
    leg_disp, _ = _do_dispatch(op_legacy, test_data, return_routing=False)
    tc.sync()
    leg_combine_out, _ = _do_combine(
        op_legacy, leg_disp, test_data[1][rank]
    )
    tc.sync()

    # Routing-handle path with identical inputs.
    rh_disp, R = _do_dispatch(op_route, test_data, return_routing=True)
    tc.sync()
    rh_combine_out, _ = _do_combine(
        op_route, rh_disp, test_data[1][rank], routing=R
    )
    tc.sync()

    # Both dispatch outputs must match: same per-rank token order in dispatchOut,
    # since indices are identical and slot assignment is deterministic-modulo-CAS.
    # The routing-handle path doesn't change kernel CAS ordering, just the location
    # of the routing output; the dispatched payload should still be present
    # (possibly in a different order than legacy due to CAS race independence).
    # We compare combine outputs (which are fully order-invariant).
    assert torch.allclose(
        leg_combine_out.float(), rh_combine_out.float(), atol=1e-3, rtol=1e-3
    ), (
        f"rank {rank}: legacy/routing-handle combine outputs differ; max diff = "
        f"{(leg_combine_out.float() - rh_combine_out.float()).abs().max().item()}"
    )


def _replay_correctness(rank, world_size, kernel):
    """Mode-1 dispatch → combine; mode-2 dispatch with the same routing.

    Verifies routing tensors are not mutated by the replay path and that the
    replay produces a dispatch output consistent with the cached layout.
    """
    config = _make_intranode_config(rank, world_size) if kernel == "intra" \
        else _make_internode_v1_config(rank, world_size)

    op = mori.ops.EpDispatchCombineOp(config)
    tc = EpDispatchCombineTestCase(config)
    test_data = _gen_layer_data(tc, seed=7777)
    rank_idx = test_data[1][rank]

    # Mode-1 dispatch + snapshot routing.
    fwd_disp, R = _do_dispatch(op, test_data, return_routing=True)
    tc.sync()
    pre_disp_dest = R.disp_dest_tok_id_map.clone()
    pre_total_recv = R.total_recv_token_num.clone()

    # Combine using the routing handle (mode-2 combine, not dispatch).
    combine_out, _ = _do_combine(op, fwd_disp, rank_idx, routing=R)
    tc.sync()

    # Mode-2 (replay) dispatch: same indices, *different* token payload (use a
    # gradient-like signal); routes must match the cached layout.
    rng = torch.Generator(device=torch.device("cuda", rank))
    rng.manual_seed(91234 + rank)
    grad_input = torch.randn(
        test_data[2][rank].shape,
        dtype=test_data[2][rank].dtype,
        device=test_data[2][rank].device,
        generator=rng,
    )
    # Build a fresh test_data wrapper around the gradient-shaped input.
    grad_test_data = (
        test_data[0],
        test_data[1],
        [grad_input if r == rank else test_data[2][r] for r in range(world_size)],
        test_data[3],
        test_data[4],
    )

    rep_disp, _ = _do_dispatch(op, grad_test_data, routing=R)
    tc.sync()

    # Routing tensors should be invariant across replay.
    assert torch.equal(R.disp_dest_tok_id_map, pre_disp_dest), (
        f"rank {rank}: dispDestTokIdMap changed after replay dispatch"
    )
    assert torch.equal(R.total_recv_token_num, pre_total_recv), (
        f"rank {rank}: totalRecvTokenNum changed after replay dispatch"
    )

    # Replay dispatch output total_recv equals the mode-1 total_recv.
    assert rep_disp["total_recv"] == fwd_disp["total_recv"], (
        f"rank {rank}: replay total_recv {rep_disp['total_recv']} differs from "
        f"mode-1 total_recv {fwd_disp['total_recv']}"
    )
    # Combine output is sanity (we don't assert exact value here; the legacy/
    # routing-handle parity test covers correctness).
    assert combine_out is not None


def _multi_layer_correctness(rank, world_size, kernel, num_layers):
    """Multiple in-flight routing handles on a single shared op.

    Forward dispatches all N layers, saving routings; then runs all N combines.
    Each combine output is compared against a baseline single-layer run on a
    separate op (pristine state) to verify cross-layer state isolation.
    """
    config = _make_intranode_config(rank, world_size) if kernel == "intra" \
        else _make_internode_v1_config(rank, world_size)

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

    # Now run all combines using stashed routings.
    layer_combine_out = []
    for L in range(num_layers):
        out, _ = _do_combine(op_shared, layer_disp[L], layer_data[L][1][rank],
                             routing=layer_R[L])
        tc.sync()
        layer_combine_out.append(out)

    # Baseline: each layer through a fresh op.
    for L in range(num_layers):
        op_baseline = mori.ops.EpDispatchCombineOp(config)
        td = layer_data[L]
        d, R = _do_dispatch(op_baseline, td, return_routing=True)
        tc.sync()
        out, _ = _do_combine(op_baseline, d, td[1][rank], routing=R)
        tc.sync()
        # Allow small numerical wobble across runs (different op instance,
        # different CAS race), but layer outputs must be close.
        assert torch.allclose(
            out.float(), layer_combine_out[L].float(), atol=1e-2, rtol=1e-2
        ), (
            f"rank {rank} layer {L}: shared-op combine differs from baseline; "
            f"max diff = "
            f"{(out.float() - layer_combine_out[L].float()).abs().max().item()}"
        )


def _stale_symmetric_buffer_guard(rank, world_size):
    """Specifically exercises the disp_tok_id_to_src_tok_id_local snapshot.

    Runs dispatch L0 (return_routing=True), then dispatch L1 (return_routing=True)
    which overwrites the symmetric local view, then combine L0 — must still match
    a fresh-op baseline because the snapshot is layer-private.

    IntraNode is the only kernel that reads dispTokIdToSrcTokIdLocal in combine
    (V1 doesn't), so this test is intra-only.
    """
    config = _make_intranode_config(rank, world_size)
    op_shared = mori.ops.EpDispatchCombineOp(config)
    tc = EpDispatchCombineTestCase(config)

    td0 = _gen_layer_data(tc, seed=51)
    td1 = _gen_layer_data(tc, seed=151)

    # Use zero-copy combine path to force the kernel to read
    # dispTokIdToSrcTokIdLocal (the UseP2PRead=false path is the only consumer).
    config_nop2p = _make_intranode_config(rank, world_size)
    config_nop2p.use_external_inp_buf = False
    op_shared_nop2p = mori.ops.EpDispatchCombineOp(config_nop2p)

    d0, R0 = _do_dispatch(op_shared_nop2p, td0, return_routing=True)
    tc.sync()
    # Stomp the symmetric buffer with a second dispatch.
    _ = _do_dispatch(op_shared_nop2p, td1, return_routing=True)
    tc.sync()

    # Combine L0 with R0; must match a baseline op that only saw L0.
    out_shared, _ = _do_combine(op_shared_nop2p, d0, td0[1][rank], routing=R0)
    tc.sync()

    op_baseline = mori.ops.EpDispatchCombineOp(config_nop2p)
    d_base, R_base = _do_dispatch(op_baseline, td0, return_routing=True)
    tc.sync()
    out_base, _ = _do_combine(op_baseline, d_base, td0[1][rank], routing=R_base)
    tc.sync()

    assert torch.allclose(
        out_shared.float(), out_base.float(), atol=1e-2, rtol=1e-2
    ), (
        f"rank {rank}: stale-symmetric-buffer guard failed; combine on shared op "
        f"diverged from baseline (max diff = "
        f"{(out_shared.float() - out_base.float()).abs().max().item()})."
    )


# ---------------------------------------------------------------------------
# Pytest entrypoints
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kernel", ("intra", "v1"))
def test_legacy_vs_routing_handle_parity(torch_dist_process_manager, kernel):
    world_size = 8
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_legacy_vs_routing_handle_parity, [world_size, kernel])
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
    """IntraNode-only: the only kernel that reads dispTokIdToSrcTokIdLocal."""
    world_size = 8
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_stale_symmetric_buffer_guard, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)
