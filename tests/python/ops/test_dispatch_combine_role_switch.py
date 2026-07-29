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
"""EpDispatchCombineOp buffer teardown / rebuild (`reconfigure`, `finalize`).

Covers the mori side of an sglang PD role switch: a prefill role sizes its
all-to-all dispatch buffer for a large chunked-prefill token count, a decode
role for a small per-step batch. On a flip the buffers must be freed and
re-allocated for the new capacity, in place, with the shmem context intact --
and dispatch/combine must still be numerically correct afterwards.
"""
import pytest
import torch

import mori
from tests.python.ops.dispatch_combine_test_utils import (
    EpDispatchCombineTestCase,
    assert_worker_results,
)

# "prefill" (large chunked-prefill token count) vs "decode" (per-step batch).
PREFILL_TOKENS = 128
DECODE_TOKENS = 8


# Kernel types reachable with a single node of 8 GPUs. The InterNode* paths
# need real multi-node RDMA and AsyncLL needs MORI_ENABLE_SDMA set before shmem
# init, so both are left to their own coverage (see backlog in RESULTS_M.md).
_KERNEL_TYPES = ("IntraNode", "IntraNodeLL")


def _make_config(
    rank,
    world_size,
    max_num_inp_token_per_rank,
    kernel_type="IntraNode",
    quant_type="none",
):
    return mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world_size,
        hidden_dim=4096,
        scale_dim=0,
        scale_type_size=1,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=32,
        num_experts_per_token=8,
        max_token_type_size=4,
        block_num=64,
        warp_num_per_block=4,
        use_external_inp_buf=True,
        kernel_type=getattr(mori.ops.EpDispatchCombineKernelType, kernel_type),
        quant_type=quant_type,
    )


def _run_once(op, config):
    """One dispatch+combine at the op's current capacity, fully checked."""
    test_case = EpDispatchCombineTestCase(config)
    test_data = test_case.gen_test_data(use_max_token_num=True)
    test_case.run_test_once(op, test_data, check_results=True)


def _assert_capacity(op, expected_tokens):
    """The rebuilt buffers must actually be sized for the new role."""
    assert op.config.max_num_inp_token_per_rank == expected_tokens
    assert op._handle_info["max_num_inp_token_per_rank"] == expected_tokens
    assert op.max_num_tokens_to_send_per_rank() == expected_tokens
    assert op.max_num_tokens_to_send() == expected_tokens * op.config.world_size
    assert op.is_initialized


# ---------------------------------------------------------------------------
# workers (run one per rank via the torch-dist process manager)
# ---------------------------------------------------------------------------
def _worker_flip_and_flip_back(rank, world_size, kernel_type="IntraNode",
                               quant_type="none"):
    """P -> D -> P: correct numerics at every capacity, buffers really resized."""
    config = _make_config(rank, world_size, PREFILL_TOKENS, kernel_type, quant_type)
    op = mori.ops.EpDispatchCombineOp(config)

    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)

    # flip P -> D: buffers shrink
    op.reconfigure(max_num_inp_token_per_rank=DECODE_TOKENS)
    _assert_capacity(op, DECODE_TOKENS)
    _run_once(op, config)

    # flip back D -> P: buffers grow again (the OOM-risk direction)
    op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)


def _worker_leak_stress(rank, world_size, cycles):
    """N flip cycles must not grow device memory once past the first cycle.

    The first cycle is excluded from the baseline: it also warms up caching
    allocators / lazily-created kernel state, so its delta is not a leak
    signal. From cycle 2 onward a steady state is expected.
    """
    config = _make_config(rank, world_size, PREFILL_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)
    _run_once(op, config)

    def free_bytes():
        torch.cuda.synchronize()
        free, _total = torch.cuda.mem_get_info(config.rank)
        return free

    baseline = None
    for i in range(cycles):
        op.reconfigure(max_num_inp_token_per_rank=DECODE_TOKENS)
        _run_once(op, config)
        op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
        _run_once(op, config)
        if i == 0:
            baseline = free_bytes()

    end = free_bytes()
    leaked = baseline - end
    # Tolerate 64 MiB of allocator noise across the remaining cycles; a real
    # per-cycle leak of the a2a buffers is orders of magnitude larger than that
    # (the prefill dispatch-out buffer alone is 128*8*4096*4 B = 16 MiB).
    assert leaked < 64 * 1024 * 1024, (
        f"rank {rank}: device memory dropped by {leaked / 2**20:.1f} MiB over "
        f"{cycles - 1} flip cycles -- a2a buffers are leaking on reconfigure"
    )


def _worker_reconfigure_rejects_layout_change(rank, world_size):
    """Fields the peers' symmetric layout depends on must not be changeable."""
    config = _make_config(rank, world_size, PREFILL_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)

    for field, bad_value in (
        ("hidden_dim", config.hidden_dim * 2),
        ("num_experts_per_rank", config.num_experts_per_rank + 1),
        ("num_experts_per_token", config.num_experts_per_token + 1),
        ("world_size", config.world_size + 1),
    ):
        good = getattr(op._cpp_config, field)
        setattr(op._cpp_config, field, bad_value)
        try:
            with pytest.raises(RuntimeError, match="must not change"):
                op._handle.reconfigure(op._cpp_config)
        finally:
            setattr(op._cpp_config, field, good)

    # A rejected reconfigure must leave the op fully usable -- validation runs
    # before anything is freed.
    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)


def _worker_reconfigure_noop_and_finalize(rank, world_size):
    """Same-capacity reconfigure is a no-op; finalize is idempotent."""
    config = _make_config(rank, world_size, PREFILL_TOKENS)
    op = mori.ops.EpDispatchCombineOp(config)

    op.reconfigure(max_num_inp_token_per_rank=PREFILL_TOKENS)
    _assert_capacity(op, PREFILL_TOKENS)
    _run_once(op, config)

    op.finalize()
    assert not op.is_initialized
    op.finalize()  # idempotent
    assert not op.is_initialized
    # The destructor must not double-free after an explicit finalize.
    del op


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("kernel_type", _KERNEL_TYPES)
def test_reconfigure_flip_and_flip_back(
    torch_dist_process_manager, world_size, kernel_type
):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_flip_and_flip_back, [world_size, kernel_type])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


# fp8 paths allocate extra scale buffers off the same capacity fields, so the
# rebuild has to resize those too.
@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("quant_type", ("fp8_direct_cast", "fp8_blockwise"))
def test_reconfigure_flip_and_flip_back_quant(
    torch_dist_process_manager, world_size, quant_type
):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_flip_and_flip_back, [world_size, "IntraNode", quant_type])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("cycles", (5,))
def test_reconfigure_leak_stress(torch_dist_process_manager, world_size, cycles):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_leak_stress, [world_size, cycles])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
def test_reconfigure_rejects_layout_change(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_reconfigure_rejects_layout_change, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)


@pytest.mark.parametrize("world_size", (8,))
def test_reconfigure_noop_and_finalize(torch_dist_process_manager, world_size):
    for _ in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (_worker_reconfigure_noop_and_finalize, [world_size])
        )
    assert_worker_results(torch_dist_process_manager, world_size)
