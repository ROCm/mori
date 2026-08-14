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
"""Lightweight (no-GPU) guard for the combine() indices contract (ROCm/mori#475).

The InterNodeV1 combine keys ``tokenIndices`` by this rank's own token id, the
same key as ``interNodeDispSendMap``, so passing dispatch()'s returned
recv-slot-layout ``out_idx`` reduces cross-node tokens against an unrelated
token's routing and corrupts the result without any error. The cross-node
numerics themselves are covered by the GPU test
``test_dispatch_combine_internode_v1.py``.
"""

import pytest
import torch

from mori.ops.dispatch_combine import EpDispatchCombineOp

TOPK = 8
NUM_TOKENS = 128


def test_own_rank_routing_accepted():
    indices = torch.zeros(NUM_TOKENS, TOPK, dtype=torch.int32)
    EpDispatchCombineOp._check_combine_indices(indices, NUM_TOKENS)


def test_recv_layout_indices_rejected():
    # dispatch() returns out_idx sized by the recv buffer, always >= cur_n.
    max_recv = NUM_TOKENS * 4
    out_idx = torch.zeros(max_recv, TOPK, dtype=torch.int32)
    with pytest.raises(ValueError, match="out_idx"):
        EpDispatchCombineOp._check_combine_indices(out_idx, NUM_TOKENS)


def test_short_indices_rejected():
    indices = torch.zeros(NUM_TOKENS - 1, TOPK, dtype=torch.int32)
    with pytest.raises(ValueError, match="rows but this rank dispatched"):
        EpDispatchCombineOp._check_combine_indices(indices, NUM_TOKENS)


def test_same_shape_stand_in_is_not_detectable():
    # Documents the limit of a row-count check: another layer's routing, or a
    # sliced out_idx whose recv count happens to equal the send count, has
    # exactly the right shape and passes. Identity cannot close this either --
    # data_ptr() has both false positives (a cloned tensor is still correct)
    # and false negatives (the caching allocator hands the same address back
    # across iterations) -- so correctness here rests on the caller contract.
    foreign = torch.zeros(NUM_TOKENS, TOPK, dtype=torch.int32)
    EpDispatchCombineOp._check_combine_indices(foreign, NUM_TOKENS)
