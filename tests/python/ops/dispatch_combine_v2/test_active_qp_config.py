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
from mori.ops.dispatch_combine_v2 import hip_tuning_configs as tuning
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
    assert tuning.resolve_active_qps(cfg) == 3
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
        request = op._internode_request(cfg, "bf16", 32, 4, 8, active, 8)
        assert request["numQpPerPe"] == active
        assert request["numQpToDrain"] == 8
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
    assert tuning.resolve_active_qps(cfg) == expected


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
    [("64,32,8", (64, 32, 8, 2)), ("64, 32, 8, 4", (64, 32, 8, 4))],
)
def test_env_geometry_takes_an_optional_active_count(monkeypatch, raw, expected):
    # Three fields resolve the count like an untuned one; a fourth names it.
    monkeypatch.setenv("MORI_EP_DISP_GEOM", raw)
    assert tuning._geometry_from_env("MORI_EP_DISP_GEOM", config()) == expected


@pytest.mark.parametrize("raw", ["64,32", "64,32,8,4,1"])
def test_env_geometry_rejects_other_lengths(monkeypatch, raw):
    monkeypatch.setenv("MORI_EP_DISP_GEOM", raw)
    with pytest.raises(ValueError, match="block,rdma,warp"):
        tuning._geometry_from_env("MORI_EP_DISP_GEOM", config())


def test_env_pinned_count_above_the_allocation_is_rejected(monkeypatch):
    # Clamping would let a sweep report a count it never ran.
    monkeypatch.setenv("MORI_EP_DISP_GEOM", "64,32,8,4")
    with pytest.raises(ValueError, match="MORI_EP_DISP_GEOM"):
        tuning.internode_schedule(config(num_qp_per_pe=2))


@pytest.mark.parametrize(
    "allocated,active,table_count,expected",
    [
        # Left open: the configured count, else two within the allocation.
        (8, None, None, 2),
        (8, 3, None, 3),
        (1, None, None, 1),
        # A table count above the allocation is clamped, like block to CUs.
        (2, None, 4, 2),
        # A configured count beats the table's.
        (8, 3, 1, 3),
    ],
)
def test_resolve_active_qps(allocated, active, table_count, expected):
    cfg = config(num_qp_per_pe=allocated, active_qps=active)
    assert tuning.resolve_active_qps(cfg, table_count) == expected


def test_fit_keeps_the_active_count_and_forces_rdma_below_block():
    assert tuning.fit_internode_geometry((16, 16, 8, 1)) == (16, 15, 8, 1)


# -- tables -----------------------------------------------------------------

ARCH, MODEL = "gfx942", "testgpu"


def write_tables(directory, kernel_type, dispatch_rules, combine_rules):
    rules = [
        dict(
            num_experts=64,
            topk=4,
            phase=phase,
            dtype="bf16",
            hidden_dim=1024,
            num_tokens=tokens,
            block_num=block,
            rdma_block_num=rdma,
            warp_per_block=warp,
            **(
                {"zero_copy": False, "quant_type": "none"} if phase == "combine" else {}
            ),
            **({"active_qps": qps} if qps is not None else {}),
        )
        for phase, phase_rules in (
            ("dispatch", dispatch_rules),
            ("combine", combine_rules),
        )
        for tokens, block, rdma, warp, qps in phase_rules
    ]
    name = tuning.v2_table_filename(ARCH, MODEL, kernel_type, 16)
    tuning.write_v2_table(directory / name, rules, kernel_type)


@pytest.fixture
def tables(tmp_path, monkeypatch):
    """Point lookup at a temp directory as a fake 80-CU model."""
    monkeypatch.setenv("MORI_EP_V2_TUNING_DIR", str(tmp_path))
    monkeypatch.delenv("MORI_EP_DISP_GEOM", raising=False)
    monkeypatch.delenv("MORI_EP_COMB_GEOM", raising=False)
    monkeypatch.setattr(tuning._gpu, "detect_model", lambda: MODEL)
    monkeypatch.setattr(tuning._gpu, "arch_name", lambda: ARCH)
    monkeypatch.setattr(tuning._gpu, "cu_count", lambda: 80)
    tuning._V2_RULES_CACHE.clear()
    # Per-phase counts at 16 tokens (dispatch 1, combine 4); none above it.
    rules = (
        [(16, 80, 40, 4, 1), (128, 80, 48, 8, None)],
        [(16, 80, 40, 4, 4), (128, 64, 48, 6, None)],
    )
    write_tables(tmp_path, "InterNodeV2", *rules)
    write_tables(tmp_path, "InterNodeV2LL", *rules)
    yield tmp_path
    tuning._V2_RULES_CACHE.clear()


def test_table_rules_may_carry_active_counts(tables):
    rows = tuning.internode_buckets(16, 1024, 4, bf16(), experts_per_rank=4)
    assert [(row[0], row[1][3], row[2][3]) for row in rows] == [
        (16, 1, 4),
        (128, None, None),
    ]


def test_configured_count_pins_both_phases_over_the_table(tables):
    cfg = config(internode_kernel="v2", active_qps=3)
    for entry in tuning.resolve_schedule(cfg):
        assert entry.dispatch.active_qps == entry.combine.active_qps == 3
    cfg = config(internode_kernel="v2")
    first = tuning.resolve_schedule(cfg)[0]
    assert (first.dispatch.active_qps, first.combine.active_qps) == (1, 4)


def bf16():
    import torch

    return torch.bfloat16


class FakePlan:
    def __init__(self, **request):
        self.request = request

    def bind(self, **_):
        pass


def build(monkeypatch, cfg):
    from mori.jit.v2 import plan_api
    from mori.ops.dispatch_combine_v2 import ep_plans

    monkeypatch.setattr(
        ep_plans,
        "EP_INTERNODE_PLANS",
        {name: FakePlan for name in ep_plans.EP_INTERNODE_PLANS},
    )
    monkeypatch.setattr(plan_api, "make_launch_group", lambda plans: tuple(plans))
    op = object.__new__(EpDispatchCombineOpHip)
    op.cfg = cfg
    op._selectable_active_qps = tuple(cfg.active_qp_counts or ())
    op._active_qps_override = None
    op._multi_processor_count = 80
    op._schedule = tuning.resolve_schedule(cfg)
    op._dispatch_specs = tuple(dict.fromkeys(e.dispatch for e in op._schedule))
    op._combine_specs = tuple(dict.fromkeys(e.combine for e in op._schedule))
    op._plans = []
    op._kernels = op._build_internode_kernels(cfg)
    return op


@pytest.mark.parametrize("family", ["auto", "v2", "v2_ll"])
@pytest.mark.parametrize("selectable", [None, (1, 8)])
def test_every_launch_key_is_built(tables, monkeypatch, family, selectable):
    # A launch looks its kernel up by the bucket's KernelConfig, with the
    # set_active_qps override applied. A missing key is a KeyError at run time,
    # not build time, so walk every token count and override.
    cfg = config(
        internode_kernel=family,
        internode_auto_ll_max_tokens=8,
        active_qp_counts=selectable,
    )
    op = build(monkeypatch, cfg)
    for override in (None, *(selectable or ())):
        op._active_qps_override = override
        for num_tokens in range(1, cfg.max_num_inp_token_per_rank + 1):
            dispatch, combine = op._pick(num_tokens)
            assert dispatch in op._kernels.dispatch
            assert combine in op._kernels.combine
            if override is not None:
                assert dispatch.active_qps == combine.active_qps == override
    op._active_qps_override = None

    # Every plan drains the op's largest active count, one op-wide value: the
    # table's 4 (combine, 16 tokens), or 8 when set_active_qps can select it.
    expected_drain = 8 if selectable else 4
    assert {plan.request["numQpToDrain"] for plan in op._plans} == {expected_drain}
    # And the kernels carry each phase's own count; untuned buckets default to 2.
    assert {plan.request["numQpPerPe"] for plan in op._plans} >= {1, 4, 2}


def test_launch_uses_the_override_then_the_bucket_count(tables, monkeypatch):
    cfg = config(internode_kernel="v2_ll", active_qp_counts=(8,))
    op = build(monkeypatch, cfg)

    def counts(num_tokens):
        dispatch, combine = op._pick(num_tokens)
        return dispatch.active_qps, combine.active_qps

    # The table's per-phase counts for the 16-token bucket, the default above it.
    assert counts(16) == (1, 4)
    assert counts(64) == (2, 2)
    op.set_active_qps(8)
    assert counts(16) == (8, 8)
    assert op.kernel_config(64).combine.active_qps == 8
    op.set_active_qps(None)
    assert counts(16) == (1, 4)


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
