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
# Regression test for the InterNode dispatch out-of-range destination-PE guard.
#
# The InterNode dispatch kernel computes the destination PE from an expert id:
#
#   src/ops/dispatch_combine/internode_v1.cpp
#       lanePe = indices[laneId] / config.numExpertPerRank;
#       assert((lanePe < config.worldSize) && (lanePe >= 0));  # no-op under NDEBUG
#       ...
#       destPe = __shfl(lanePe, e);
#       GetAs<uint8_t*>(destPe) / WarpCopy / atomicAdd          # OOB if destPe >= worldSize
#
# In a Release build (the default) NDEBUG strips the assert, so an out-of-range
# expert id (e.g. an EPLB physical id >= worldSize * numExpertPerRank) makes
# destPe >= worldSize and drives an out-of-bounds device write -> HSA page fault.
# A real GPU run on wide-ep=16 (2 nodes x 8 GPU) is not exercised here; this test
# models the exact integer mapping the kernel performs and the runtime guard that
# folds an out-of-range destPe into the existing `shouldSkip` path.

import pytest

# wide-ep=16 topology (worldSize=16, gpuPerNode=8 => 2 nodes)
WORLD_SIZE = 16
GPU_PER_NODE = 8
N_NODES = WORLD_SIZE // GPU_PER_NODE  # 2


def lane_pe(expert_id: int, num_expert_per_rank: int) -> int:
    """Mirror internode_v1.cpp: lanePe = indices[laneId] / numExpertPerRank."""
    return expert_id // num_expert_per_rank


def is_in_range(lane_pe_val: int, world_size: int) -> bool:
    """The assert predicate (compiled out under NDEBUG)."""
    return 0 <= lane_pe_val < world_size


def guarded_dest_pe(expert_id: int, num_expert_per_rank: int, world_size: int):
    """Device-side guard: skip out-of-range ids instead of an OOB write.

    Returns the destPe to use, or None meaning 'skip this expert slot'
    (the kernel writes NullFlatTokenIndex and performs no copy).
    """
    lp = lane_pe(expert_id, num_expert_per_rank)
    if not is_in_range(lp, world_size):
        return None
    return lp


def test_valid_ids_map_in_range():
    """Every id < worldSize*numExpertPerRank maps to a valid PE and node."""
    num_expert_per_rank = 8  # 16 ranks * 8 = 128 experts
    num_experts = WORLD_SIZE * num_expert_per_rank
    for eid in range(num_experts):
        lp = lane_pe(eid, num_expert_per_rank)
        assert is_in_range(lp, WORLD_SIZE), (eid, lp)
        assert 0 <= lp // GPU_PER_NODE < N_NODES


def test_out_of_range_id_overflows_lanepe():
    """An id >= numExperts drives lanePe >= worldSize (the unguarded OOB case)."""
    num_expert_per_rank = 8
    num_experts = WORLD_SIZE * num_expert_per_rank  # 128
    physical_id = num_experts  # first id past the valid range
    lp = lane_pe(physical_id, num_expert_per_rank)
    assert lp == WORLD_SIZE  # exactly one past the last valid PE (0..15)
    assert not is_in_range(lp, WORLD_SIZE)  # assert WOULD have fired in Debug


def test_guard_drops_overflow_instead_of_oob():
    """The guard skips out-of-range / negative ids and keeps valid ones."""
    num_expert_per_rank = 8
    num_experts = WORLD_SIZE * num_expert_per_rank
    assert guarded_dest_pe(0, num_expert_per_rank, WORLD_SIZE) == 0
    assert (
        guarded_dest_pe(num_experts - 1, num_expert_per_rank, WORLD_SIZE)
        == WORLD_SIZE - 1
    )
    assert guarded_dest_pe(num_experts, num_expert_per_rank, WORLD_SIZE) is None
    assert guarded_dest_pe(10 * num_experts, num_expert_per_rank, WORLD_SIZE) is None
    assert guarded_dest_pe(-1, num_expert_per_rank, WORLD_SIZE) is None


def test_small_num_expert_per_rank_also_overflows():
    """If numExpertPerRank is too small for the id space, ids overflow the PE table."""
    num_expert_per_rank = 4
    capacity = WORLD_SIZE * num_expert_per_rank  # 64
    assert is_in_range(lane_pe(capacity - 1, num_expert_per_rank), WORLD_SIZE)
    assert not is_in_range(lane_pe(capacity, num_expert_per_rank), WORLD_SIZE)


def kernel_should_skip(dest_pe: int, my_node: int) -> bool:
    """Mirror the guard in internode_v1.cpp:

        bool peOutOfRange = (destPe < 0) || (destPe >= config.worldSize);
        bool shouldSkip = peOutOfRange || (destNode != myNode) || <dedup>;

    The dedup term is omitted in this single-slot model; only the
    out-of-range + wrong-node filtering that prevents the OOB write is asserted.
    """
    pe_out_of_range = dest_pe < 0 or dest_pe >= WORLD_SIZE
    dest_node = dest_pe // GPU_PER_NODE if not pe_out_of_range else -1
    return pe_out_of_range or (dest_node != my_node)


def test_guard_skips_out_of_range_on_every_node():
    """An out-of-range or negative destPe is skipped on every node (no OOB path);
    valid PEs on their own node still proceed, so valid traffic is unchanged."""
    num_expert_per_rank = 8
    num_experts = WORLD_SIZE * num_expert_per_rank
    overflow_pe = lane_pe(num_experts, num_expert_per_rank)  # == 16, OOB
    for my_node in range(N_NODES):
        assert kernel_should_skip(overflow_pe, my_node) is True
    for my_node in range(N_NODES):
        assert kernel_should_skip(-1, my_node) is True
    for pe in range(WORLD_SIZE):
        my_node = pe // GPU_PER_NODE
        assert kernel_should_skip(pe, my_node) is False
    assert kernel_should_skip(0, my_node=1) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
