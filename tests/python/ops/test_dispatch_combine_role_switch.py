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
"""Runtime resize of the a2a buffers, as a PD role switch drives it.

Every case runs on all 8 ranks: the outcome of a resize is a group verdict, so a
single-rank test cannot observe the behaviour these guard.
"""

import os
import traceback

import mori
import pytest
import torch
from mori.ops.dispatch_combine import ResizeFatal, ResizeRejected, ResizeRolledBack

from tests.python.ops.dispatch_combine_test_utils import (
    EpDispatchCombineTestCase,
    assert_worker_results,
)

WORLD_SIZE = 8
SMALL_TOKENS = 128
LARGE_TOKENS = 1024
# Allocated after the symmetric buffers, so an injected failure here exercises
# the partial-cleanup path and not just the first allocation.
FAULT_TARGET = "dispReceiverIdxMap"


def _config(rank, max_num_inp_token_per_rank):
    return mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=WORLD_SIZE,
        hidden_dim=1024,
        scale_dim=0,
        scale_type_size=1,
        max_token_type_size=2,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=4,
        num_experts_per_token=2,
        block_num=64,
        warp_num_per_block=4,
        kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
    )


def _dispatch_ok(op):
    """A full round trip at the op's CURRENT capacity, checked against the
    reference: this is what proves a resized buffer is usable, not just sized.
    The checker prints the mismatch itself, so a bool loses nothing."""
    test_case = EpDispatchCombineTestCase(op.config)
    try:
        test_case.run_test_once(op, test_case.gen_test_data(), check_results=True)
        return True
    except AssertionError:
        return False


class _FaultInjection:
    """The C++ side re-arms on every change of the variable, so the worker pool
    can run several cases in the same process."""

    def __init__(self, enabled, times):
        self.enabled = enabled
        self.times = times

    def __enter__(self):
        if self.enabled:
            os.environ["MORI_TEST_FAIL_ALLOC"] = FAULT_TARGET
            os.environ["MORI_TEST_FAIL_ALLOC_TIMES"] = str(self.times)

    def __exit__(self, *exc):
        os.environ.pop("MORI_TEST_FAIL_ALLOC", None)
        os.environ.pop("MORI_TEST_FAIL_ALLOC_TIMES", None)


def _run(rank, worker_name, *args):
    """The rig hands a worker's exception to a result queue that the collector
    cannot drain while a peer is still inside a collective, so the first fault of
    a hung run is invisible and the group looks like a resize deadlock. Print it
    where it happens. Named at module level because the queue pickles by
    reference."""
    try:
        return globals()[worker_name](rank, *args)
    except BaseException:
        traceback.print_exc()
        raise


def _heap_addr(op):
    """mori sub-allocates every buffer from one symmetric heap reserved at shmem
    init, so a driver-level free-VRAM reading cannot see a resize at all. The
    heap's own allocator is address-ordered first fit, which makes a buffer's
    address the reading that does move when memory is not given back."""
    return op._dispatch_out_ptrs[0]


def _worker_round_trip(rank, cycles):
    op = mori.ops.EpDispatchCombineOp(_config(rank, SMALL_TOKENS))
    fresh_op_ok = _dispatch_ok(op)
    heap_at_start = _heap_addr(op)

    per_cycle = []
    usable = []
    for _ in range(cycles):
        for target in (LARGE_TOKENS, SMALL_TOKENS):
            op.reconfigure(target)
            assert op.config.max_num_inp_token_per_rank == target
            usable.append(_dispatch_ok(op))
        per_cycle.append(_heap_addr(op))

    # The allocation is a high-water mark, so only the first grow takes anything
    # and a flip cycle repeated forever costs nothing.
    assert len(set(per_cycle)) == 1, [hex(a) for a in per_cycle]
    op.finalize()
    # A shrink deliberately keeps the memory; finalize is what hands it back. A
    # fresh op can only land where the first one did if all of it came back.
    reborn = mori.ops.EpDispatchCombineOp(_config(rank, SMALL_TOKENS))
    assert _heap_addr(reborn) == heap_at_start, (
        hex(_heap_addr(reborn)),
        hex(heap_at_start),
    )
    assert all(usable), "the round after a capacity change is wrong"
    # Last, so a failure here still leaves the memory verdict above measured.
    assert fresh_op_ok, "the baseline dispatch on a fresh op is wrong"


def _worker_rejected(rank):
    op = mori.ops.EpDispatchCombineOp(_config(rank, SMALL_TOKENS))
    with pytest.raises(ResizeRejected):
        op.reconfigure(0)
    assert op.is_initialized
    assert op.config.max_num_inp_token_per_rank == SMALL_TOKENS
    assert _dispatch_ok(op), "a rejected resize left the op unusable"
    op.finalize()


def _worker_rolled_back(rank):
    """Only rank 0 fails to grow. Every rank must end up back at the old
    capacity: symmetric buffers where 7 of 8 ranks grew are corrupt, not
    degraded, so the ranks that succeeded have to give the capacity back."""
    op = mori.ops.EpDispatchCombineOp(_config(rank, SMALL_TOKENS))
    with _FaultInjection(enabled=(rank == 0), times=1):
        with pytest.raises(ResizeRolledBack):
            op.reconfigure(LARGE_TOKENS)
    assert op.is_initialized
    assert op.config.max_num_inp_token_per_rank == SMALL_TOKENS
    assert _dispatch_ok(op), "the round after a rolled-back resize is wrong"
    # An op that has reallocated must not outlive the shmem heap: the rig finalizes
    # shmem at pool shutdown and the destructor would then free into nothing.
    op.finalize()


def _worker_fatal(rank):
    from mori import cpp as mori_cpp

    op = mori.ops.EpDispatchCombineOp(_config(rank, SMALL_TOKENS))
    with _FaultInjection(enabled=True, times=-1):
        with pytest.raises(ResizeFatal):
            op.reconfigure(LARGE_TOKENS)
    assert not op.is_initialized
    # The point of the guard: reading the freed buffers back must raise, not
    # SIGSEGV the rank before the error can reach its peers.
    with pytest.raises(RuntimeError):
        mori_cpp.get_dispatch_output_ptrs(op._handle, True)


@pytest.mark.parametrize(
    "worker, args",
    [
        ("_worker_round_trip", (3,)),
        ("_worker_rejected", ()),
        ("_worker_rolled_back", ()),
        ("_worker_fatal", ()),
    ],
    ids=["round_trip", "rejected", "rolled_back", "fatal"],
)
def test_role_switch_resize(torch_dist_process_manager, worker, args):
    for _ in range(WORLD_SIZE):
        torch_dist_process_manager.task_queue.put((_run, [worker, *args]))
    assert_worker_results(torch_dist_process_manager, WORLD_SIZE)
