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
    op.cfg = SimpleNamespace(is_internode=True)
    op._selectable_active_qps = (1, 2, 4, 8)
    op._active_qps_override = None
    with pytest.raises(ValueError, match="was not compiled"):
        op.set_active_qps(active)
    assert op._active_qps_override is None


@pytest.mark.parametrize("allocated,expected", [(1, 1), (2, 2), (8, 2)])
def test_default_reserves_unused_qps(allocated, expected):
    cfg = config()
    assert cfg.num_qp_per_pe == 8
    cfg = replace(cfg, num_qp_per_pe=allocated)
    assert EpDispatchCombineOpHip._resolve_active_qps(cfg) == expected


def test_override_is_set_and_cleared():
    op = object.__new__(EpDispatchCombineOpHip)
    op.cfg = SimpleNamespace(is_internode=True)
    op._selectable_active_qps = (1, 8)
    op._active_qps_override = None
    op.set_active_qps(8)
    assert op._active_qps_override == 8
    op.set_active_qps(None)
    assert op._active_qps_override is None


def test_allocation_is_capped_by_the_combine_marker_total():
    # QP 0 carries the remainder of a fixed per-combine total (64), so a kernel
    # can send on at most 64 QPs and allocating more cannot help.
    assert config(num_qp_per_pe=64).num_qp_per_pe == 64
    with pytest.raises(ValueError, match="num_qp_per_pe"):
        config(num_qp_per_pe=65)


@pytest.mark.parametrize(
    "raw,expected",
    [("64,32,8", (64, 32, 8, None)), ("64, 32, 8, 4", (64, 32, 8, 4))],
)
def test_env_geometry_takes_an_optional_active_count(monkeypatch, raw, expected):
    from mori.ops.dispatch_combine_v2.hip_backend import _geometry_from_env

    monkeypatch.setenv("MORI_EP_DISP_GEOM", raw)
    assert _geometry_from_env("MORI_EP_DISP_GEOM") == expected


@pytest.mark.parametrize("raw", ["64,32", "64,32,8,4,1"])
def test_env_geometry_rejects_other_lengths(monkeypatch, raw):
    from mori.ops.dispatch_combine_v2.hip_backend import _geometry_from_env

    monkeypatch.setenv("MORI_EP_DISP_GEOM", raw)
    with pytest.raises(ValueError, match="block,rdma,warp"):
        _geometry_from_env("MORI_EP_DISP_GEOM")


def test_env_pinned_count_above_the_allocation_is_rejected(monkeypatch):
    # Clamping would let a sweep report a count it never ran.
    monkeypatch.setenv("MORI_EP_DISP_GEOM", "64,32,8,4")
    op = object.__new__(EpDispatchCombineOpHip)
    with pytest.raises(ValueError, match="MORI_EP_DISP_GEOM"):
        op._internode_geometry_buckets_raw(config(num_qp_per_pe=2))


@pytest.mark.parametrize(
    "allocated,active,geometry,expected",
    [
        # Left open: the configured count, else two within the allocation.
        (8, None, (64, 32, 8, None), (64, 32, 8, 2)),
        (8, 3, (64, 32, 8, None), (64, 32, 8, 3)),
        (1, None, (64, 32, 8, None), (64, 32, 8, 1)),
        # A table count above the allocation is clamped, like block to CUs.
        (2, None, (64, 32, 8, 4), (64, 32, 8, 2)),
        # And rdma is still forced below block.
        (8, None, (16, 16, 8, 1), (16, 15, 8, 1)),
    ],
)
def test_fit_resolves_and_clamps_the_active_count(
    allocated, active, geometry, expected
):
    op = object.__new__(EpDispatchCombineOpHip)
    cfg = config(num_qp_per_pe=allocated, active_qps=active)
    assert op._fit_internode_geometry(cfg, geometry) == expected


def test_configured_count_pins_both_phases_over_the_table():
    op = object.__new__(EpDispatchCombineOpHip)
    cfg = config(active_qps=4)
    for phase in ("dispatch", "combine"):
        assert op._overlay_pinned_geometry(cfg, phase, (64, 32, 8, 1))[3] == 4
    cfg = config()
    assert op._overlay_pinned_geometry(cfg, "dispatch", (64, 32, 8, 1))[3] == 1


@pytest.mark.parametrize(
    "row,dispatch_qps,combine_qps",
    [
        ((None, 64, 32, 8, 32, 21, 6), None, None),
        ((None, 64, 32, 8, 32, 21, 6, 1), 1, 1),
        ((None, 64, 32, 8, 32, 21, 6, 1, 4), 1, 4),
    ],
)
def test_table_rows_may_carry_active_counts(
    monkeypatch, row, dispatch_qps, combine_qps
):
    from mori.ops.dispatch_combine_v2 import internode_tuning_configs as table

    monkeypatch.setattr(table, "_device_key", lambda: "test-device")
    monkeypatch.setattr(
        table, "_TABLE", {("test-device", 16, 1024, 4): {"fp8": (row,)}}
    )
    monkeypatch.setattr(table.gpu_utils, "cu_count", lambda: 80)
    result = table.lookup(16, 1024, 4, 8)
    assert result["dispatch"] == (64, 32, 8, dispatch_qps)
    assert result["combine"] == (32, 21, 6, combine_qps)


def test_table_rejects_more_than_two_active_count_fields(monkeypatch):
    from mori.ops.dispatch_combine_v2 import internode_tuning_configs as table

    row = (None, 64, 32, 8, 32, 21, 6, 1, 2, 4)
    monkeypatch.setattr(table, "_device_key", lambda: "test-device")
    monkeypatch.setattr(
        table, "_TABLE", {("test-device", 16, 1024, 4): {"fp8": (row,)}}
    )
    with pytest.raises(ValueError, match="at most two"):
        table.lookup(16, 1024, 4, 8)


def test_python_allocation_cap_matches_the_kernel_marker_total():
    # The config's cap is a copy of kCombineBarrierMarkerTotal; keep them equal.
    import pathlib
    import re

    header = pathlib.Path(__file__).resolve().parents[4] / (
        "include/mori/ops/dispatch_combine_v2/ep_internode_cfg.hpp"
    )
    match = re.search(r"kCombineBarrierMarkerTotal\s*=\s*(\d+)", header.read_text())
    assert match, "kCombineBarrierMarkerTotal not found"
    total = int(match.group(1))
    assert config(num_qp_per_pe=total).num_qp_per_pe == total
    with pytest.raises(ValueError, match="num_qp_per_pe"):
        config(num_qp_per_pe=total + 1)


@pytest.mark.parametrize("family", ["auto", "v2", "v2_ll"])
@pytest.mark.parametrize("selectable", [None, (1, 8)])
def test_every_launch_key_is_built(monkeypatch, family, selectable):
    # A launch looks up (geometry of its bucket, optionally with the
    # set_active_qps override, (phase, low latency)). A missing key is a KeyError
    # at run time, not build time, so walk every token count and override.
    from mori.jit.v2 import plan_api
    from mori.ops.dispatch_combine_v2 import ep_plans
    from mori.ops.dispatch_combine_v2 import internode_tuning_configs as table

    rows = (
        (4, 32, 16, 4, 32, 21, 6, 1),  # both phases on one QP
        (16, 80, 40, 4, 80, 40, 4, 1, 4),  # dispatch 1, combine 4
        (None, 80, 48, 8, 64, 48, 6),  # no count: the default
    )
    monkeypatch.setattr(table, "_device_key", lambda: "test-device")
    monkeypatch.setattr(table, "_TABLE", {("test-device", 16, 1024, 4): {"fp8": rows}})
    monkeypatch.setattr(table.gpu_utils, "cu_count", lambda: 80)

    class FakePlan:
        def __init__(self, **request):
            self.request = request

        def bind(self, **_):
            pass

    monkeypatch.setattr(
        ep_plans,
        "EP_INTERNODE_PLANS",
        {name: FakePlan for name in ep_plans.EP_INTERNODE_PLANS},
    )
    monkeypatch.setattr(plan_api, "make_launch_group", lambda plans: tuple(plans))

    cfg = config(
        internode_kernel=family,
        internode_auto_ll_max_tokens=8,
        active_qp_counts=selectable,
    )
    op = object.__new__(EpDispatchCombineOpHip)
    op.cfg = cfg
    op._selectable_active_qps = tuple(cfg.active_qp_counts or ())
    op._active_qps_override = None
    op._multi_processor_count = 80
    op._build_internode_kernels(cfg)

    for override in (None, *(selectable or ())):
        for num_tokens in range(1, cfg.max_num_inp_token_per_rank + 1):
            low_latency = op._internode_use_ll(num_tokens)
            for phase in ("dispatch", "combine"):
                geometry = op._internode_geom_for(phase, num_tokens)
                if override is not None:
                    geometry = (*geometry[:3], override)
                assert (geometry, (phase, low_latency)) in op._internode_groups

    # Every plan drains the op's largest active count, one op-wide value: the
    # table's 4 (combine, 16 tokens), or 8 when set_active_qps can select it.
    expected_drain = 8 if selectable else 4
    assert {plan.request["numQpToDrain"] for plan in op._plans} == {expected_drain}

    # And the kernels really carry the count: the table's per-phase choice.
    assert op._internode_geom_for("dispatch", 16)[3] == 1
    assert op._internode_geom_for("combine", 16)[3] == 4
    assert op._internode_geom_for("combine", 128)[3] == 2
