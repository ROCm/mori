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
"""Active QP bounds, fixed allocation, and precompiled subset selection."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig
from mori.ops.dispatch_combine_v2.hip_backend import EpDispatchCombineOpHip


@pytest.mark.parametrize("allocated", [0, -1, True, 2.0])
def test_reject_invalid_allocation(allocated):
    with pytest.raises(ValueError, match="num_qp_per_pe"):
        config(num_qp_per_pe=allocated)


def config(**kwargs):
    values = dict(
        rank=0,
        world_size=16,
        gpu_per_node=8,
        hidden_dim=1024,
        max_num_inp_token_per_rank=128,
        num_experts_per_rank=4,
        num_experts_per_token=4,
        kernel_backend="hip",
    )
    values.update(kwargs)
    return EpDispatchCombineConfig(**values)


@pytest.mark.parametrize("counts", [(), (0, 8), (-1, 8), (9, 8), (True, 8), (1.5, 8)])
def test_reject_invalid_variants(counts):
    with pytest.raises(ValueError, match="active_qp_counts"):
        config(num_qp_per_pe=8, active_qp_counts=counts)


def test_fixed_allocation_remains_replaceable():
    cfg = config(num_qp_per_pe=2)
    assert cfg.active_qp_counts is None
    assert replace(cfg, num_qp_per_pe=8).active_qp_counts is None


def test_normalize_variants_without_changing_allocation():
    cfg = config(num_qp_per_pe=8, active_qp_counts=(8, 1, 4, 2, 4))
    assert cfg.num_qp_per_pe == 8
    assert cfg.active_qp_counts == (1, 2, 4, 8)


@pytest.mark.parametrize("active", [0, 9, True, 1.5])
def test_reject_invalid_configured_active_qps(active):
    with pytest.raises(ValueError, match="active_qps"):
        config(num_qp_per_pe=8, active_qps=active)


def test_configured_active_qps_does_not_change_allocation():
    cfg = config(num_qp_per_pe=8, active_qps=3, active_qp_counts=(1, 2))
    assert EpDispatchCombineOpHip._resolve_active_qps(cfg) == 3
    assert cfg.num_qp_per_pe == 8


def test_dev_comm_allocation_is_independent_of_kernel_requests(monkeypatch):
    from mori.cco import communicator

    created = []
    handle = SimpleNamespace(world_size=16, lsa_size=8, lsa_rank=0)

    def create(comm, *, requirements):
        created.append(requirements.gda_context_count)
        return handle

    monkeypatch.setattr(communicator, "DevCommHandle", create)
    cfg = config(active_qps=3, active_qp_counts=(1, 2, 3, 8))
    op = object.__new__(EpDispatchCombineOpHip)
    op._multi_processor_count = 80
    assert op._make_dev_comm(cfg, None) is handle
    for active in cfg.active_qp_counts:
        request = op._internode_request(cfg, "bf16", 32, 4, 8, active)
        assert request["numQpPerPe"] == active
    assert created == [8]
    assert cfg.num_qp_per_pe == 8


@pytest.mark.parametrize("active", [0, 3, 9, True, 2.0])
def test_reject_uncompiled_variant_before_touching_inputs(active):
    op = object.__new__(EpDispatchCombineOpHip)
    op.cfg = SimpleNamespace(
        is_internode=True, num_qp_per_pe=8, active_qp_counts=(1, 2, 4, 8)
    )
    op._default_active_qps = 8
    op._compiled_active_qps = (1, 2, 4, 8)
    with pytest.raises(ValueError, match="was not compiled"):
        op.set_active_qps(active)


@pytest.mark.parametrize("allocated,expected", [(1, 1), (2, 2), (8, 2)])
def test_default_reserves_unused_qps(allocated, expected):
    cfg = config()
    assert cfg.num_qp_per_pe == 8
    cfg = replace(cfg, num_qp_per_pe=allocated)
    assert EpDispatchCombineOpHip._resolve_active_qps(cfg) == expected


def test_selection_keeps_pending_dispatch_count():
    op = object.__new__(EpDispatchCombineOpHip)
    op.cfg = SimpleNamespace(is_internode=True)
    op._compiled_active_qps = (1, 2, 8)
    op._active_qps = 2
    op._dispatch_active_qps = 2
    op.set_active_qps(8)
    assert op._active_qps == 8
    assert op._dispatch_active_qps == 2
