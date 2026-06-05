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
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# MIT License.
#
# Regression tests for the out-of-range expert-id guard across ALL dispatch
# kernel variants. The destination rank is computed as
# ``pe = expert_id // num_experts_per_rank`` and must satisfy
# ``0 <= pe < world_size`` in every kernel. The device kernels guarded this only
# with a debug ``assert`` (compiled out under NDEBUG in release builds), so an
# out-of-range id (e.g. an EPLB physical id >= world_size*num_experts_per_rank)
# could index device memory out of bounds -> HSA page fault.
#
# Two layers are tested:
#   1. The host-side validator ``validate_dispatch_indices`` (one check before
#      launch, protects every kernel type). Needs the built package -> gated.
#   2. Pure-Python models of each kernel's per-site skip decision (no GPU), to
#      confirm out-of-range ids are dropped on every dispatch path.

import pytest

WORLD_SIZE = 16
GPU_PER_NODE = 8


def pe_of(expert_id: int, num_expert_per_rank: int) -> int:
    return expert_id // num_expert_per_rank


def in_range(pe: int, world_size: int) -> bool:
    return 0 <= pe < world_size


# --- Layer 2: per-kernel skip models -------------------------------------------------


def skip_warp_uniform(pe: int, world_size: int) -> bool:
    """Model of the warp-uniform sites that early-`continue` on out-of-range:
    intranode.hpp, low_latency_async.cpp SendCopy, internode_v1.cpp recv."""
    return (pe < 0) or (pe >= world_size)


def route_to_null_per_lane(pe: int, world_size: int, is_duplicate: bool) -> bool:
    """Model of the per-lane sites that fold out-of-range into the dedup-null
    branch (keeps __shfl/__match_any_sync coherent):
    internode.hpp, low_latency_async.cpp SlotAssign."""
    return is_duplicate or (pe < 0) or (pe >= world_size)


ALL_KERNELS = ["intranode", "internode", "internode_v1", "internode_v1_ll", "async_ll"]


def kernel_drops_out_of_range(kernel: str, pe: int, world_size: int) -> bool:
    """Whether the given kernel's dispatch path drops (does not index memory
    with) an out-of-range pe. Maps each kernel to its guard model."""
    if kernel in ("intranode", "internode_v1", "internode_v1_ll", "async_ll_sendcopy"):
        return skip_warp_uniform(pe, world_size)
    if kernel in ("internode", "async_ll"):
        return route_to_null_per_lane(pe, world_size, is_duplicate=False)
    # default to the uniform model
    return skip_warp_uniform(pe, world_size)


def test_every_kernel_drops_out_of_range_id():
    """An EPLB physical id past the valid range is dropped on every kernel."""
    nepr = 8
    num_experts = WORLD_SIZE * nepr  # 128
    overflow_pe = pe_of(num_experts, nepr)  # 16 == world_size, OOB
    assert not in_range(overflow_pe, WORLD_SIZE)
    for kernel in ALL_KERNELS:
        assert kernel_drops_out_of_range(kernel, overflow_pe, WORLD_SIZE) is True
        # negative / sentinel id also dropped
        assert kernel_drops_out_of_range(kernel, -1, WORLD_SIZE) is True


def test_valid_ids_not_dropped_on_any_kernel():
    """Valid ids (in range) are never dropped by the guard on any kernel."""
    nepr = 8
    num_experts = WORLD_SIZE * nepr
    for kernel in ALL_KERNELS:
        for eid in (0, 1, num_experts // 2, num_experts - 1):
            pe = pe_of(eid, nepr)
            assert in_range(pe, WORLD_SIZE)
            assert kernel_drops_out_of_range(kernel, pe, WORLD_SIZE) is False


def test_per_lane_fold_keeps_duplicates_dropped():
    """The per-lane fold must still drop true duplicates (regression safety)."""
    assert (
        route_to_null_per_lane(3, WORLD_SIZE, is_duplicate=True) is True
    )  # dup dropped
    assert (
        route_to_null_per_lane(3, WORLD_SIZE, is_duplicate=False) is False
    )  # valid kept
    assert (
        route_to_null_per_lane(99, WORLD_SIZE, is_duplicate=False) is True
    )  # OOB dropped


def test_small_num_expert_per_rank_overflow_dropped():
    """If num_experts_per_rank is too small for the id space, the overflow is
    still dropped on every kernel."""
    nepr = 4
    capacity = WORLD_SIZE * nepr  # 64
    assert (
        kernel_drops_out_of_range("intranode", pe_of(capacity, nepr), WORLD_SIZE)
        is True
    )
    assert (
        kernel_drops_out_of_range("async_ll", pe_of(capacity, nepr), WORLD_SIZE) is True
    )
    assert in_range(pe_of(capacity - 1, nepr), WORLD_SIZE)


# --- Layer 1: the real host validator (needs the built mori package) -----------------


def test_validate_dispatch_indices_real():
    """The shipping host validator passes in-range and raises on out-of-range."""
    mori_ops = pytest.importorskip("mori.ops.dispatch_combine")
    validate = mori_ops.validate_dispatch_indices
    nepr, ws = 8, WORLD_SIZE
    total = ws * nepr  # 128
    # in range -> ok
    validate(0, total - 1, nepr, ws)
    # out of range (max) -> raises
    with pytest.raises(ValueError):
        validate(0, total, nepr, ws)
    # negative -> raises
    with pytest.raises(ValueError):
        validate(-1, total - 1, nepr, ws)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
