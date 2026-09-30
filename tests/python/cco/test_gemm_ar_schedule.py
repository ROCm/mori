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
"""Public scheduling contracts, without acquiring GPU resources."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("flydsl")
from mori.ops.gemm_ar import GemmAllReduceOp  # noqa: E402
from mori.ops.gemm_ar.op import padded_m, supports  # noqa: E402


@pytest.fixture
def make_op(monkeypatch):
    monkeypatch.setattr("mori.ops.gemm_ar.op.from_gpu_ptr", Mock())

    def make(world=8, **kwargs):
        comm = Mock(nranks=world, rank=0)
        comm.alloc_mem.side_effect = lambda size: SimpleNamespace(
            ptr=4096, size=size, close=Mock()
        )
        defaults = dict(
            n=7168 if world == 8 else 5120,
            k=2048,
            m_max=16384,
            quant="blockscale" if world == 8 else "mxfp8",
            schedule="wo_b",
        )
        defaults.update(kwargs)
        return GemmAllReduceOp(comm, **defaults)

    return make


@pytest.mark.parametrize("world", [8, 4])
@pytest.mark.parametrize("m_max", [4200, 8200, 11264, 16384])
def test_capacity_covers_all_plans_and_matches_allocation(make_op, world, m_max):
    op = make_op(world, m_max=m_max)
    assert op.window_bytes == GemmAllReduceOp.window_bytes_for(
        world, n=op.n, k=op.k, m_max=m_max, quant=op.quant, schedule="wo_b"
    )
    cfgs = []
    # All legacy physical shapes and all measured logical shapes can coexist.
    ms = list(range(world * op.block_m, op.m_max + 1, world * op.block_m))
    ms += [m for m in (4200, 8200, 11264, 13312) if m <= m_max]
    for m in ms:
        mp = op.padded_m(m)
        cfgs.append(op._make_cfg(mp, m))
    assert (
        len(
            {
                (c.lock_off, c.input_off, c.output_off, c.recv_off, c.window_bytes)
                for c in cfgs
            }
        )
        == 1
    )
    assert all(c.counter_chunks <= c.counter_capacity_eff for c in cfgs)
    assert all(c.window_bytes == op.window_bytes for c in cfgs)
    ranges = sorted(
        set((c.counter_off, c.counter_off + c.counter_set_bytes) for c in cfgs)
    )
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))


def test_same_padded_m_different_chunk_protocols_have_disjoint_slots(make_op):
    op = make_op()
    tuned = op._make_cfg(op.padded_m(8200), 8200)
    fallback = op._make_cfg(9216)
    assert tuned.m == fallback.m == 9216
    assert (tuned.counter_chunks, fallback.counter_chunks) == (5, 3)
    assert tuned.counter_off != fallback.counter_off
    assert op._row_plan(9216, 8200) != op._row_plan(9216)
    assert op._make_cfg(9216, 8200) == tuned


def test_preparation_order_does_not_select_the_next_call(make_op):
    op = make_op(4)
    m1, m2 = op.padded_m(4200), op.padded_m(8200)
    assert (m1, m2) == (4224, 8256)
    assert op._make_cfg(m1, 4200).counter_chunks == 5
    assert op._make_cfg(m2, 8200).counter_chunks == 9
    with pytest.raises(ValueError, match="logical_m"):
        op._row_plan(m1)
    with pytest.raises(ValueError, match="logical_m"):
        op._row_plan(m1, 8200)


@pytest.mark.parametrize("world", [8, 4])
def test_unmeasured_inputs_keep_legacy_policy(make_op, world):
    op = make_op(world)
    for m in (1, 4096, 4201, 8199, 11265, 16384):
        plan = op._row_plan(op.padded_m(m), m)
        assert plan == (padded_m(m, world, op.block_m), 0, False)
    for kwargs in (
        dict(k=4096),
        dict(n=8192),
        dict(schedule="default"),
        dict(gather_dtype="fp8"),
    ):
        other = make_op(world, **kwargs)
        assert not other._schedule.overrides
        assert other._schedule.counter_capacity == 8


@pytest.mark.parametrize("gather", ["bf16", "fp8"])
def test_fp8_wire_has_explicit_storage_and_profile(make_op, gather):
    op = make_op(4, scatter_dtype="fp8", gather_dtype=gather)
    cfg = op._make_cfg(op.padded_m(8200), 8200)
    assert cfg.fp8_scatter and cfg.counter_chunks == 9
    assert op.window_bytes == GemmAllReduceOp.window_bytes_for(
        4,
        n=op.n,
        k=op.k,
        m_max=16384,
        quant=op.quant,
        schedule="wo_b",
        scatter_dtype="fp8",
        gather_dtype=gather,
    )
    assert op._row_plan(op.padded_m(11264), 11264).chunk_bands == 0
    assert supports(
        8200,
        op.n,
        op.k,
        4,
        quant=op.quant,
        scatter_dtype="fp8",
        gather_dtype=gather,
        schedule="wo_b",
    )


@pytest.mark.parametrize("kwargs", [dict(schedule="typo"), dict(scatter_dtype="FP8")])
def test_invalid_modes_fail_before_allocation(make_op, kwargs):
    assert not supports(4200, 5120, 2048, 4, quant="mxfp8", **kwargs)
    with pytest.raises(ValueError):
        make_op(4, **kwargs)


def test_sizing_requires_target_k_and_positive_slots():
    with pytest.raises(ValueError, match="k is required"):
        GemmAllReduceOp.window_bytes_for(8, m_max=16384, n=7168, schedule="wo_b")
    with pytest.raises(ValueError, match="positive"):
        GemmAllReduceOp.window_bytes_for(8, m_max=16384, n=7168, max_shapes=0)


def test_tuned_self_test_rejects_nonfinite_output(make_op, monkeypatch):
    import torch

    op = make_op(4)
    plan = SimpleNamespace(
        input=Mock(), output=torch.full((1,), float("nan")), parts={"order": ()}
    )
    monkeypatch.setattr(op, "_compiled", lambda *args: plan)
    # Preserve injected output through the self-test's zeroing operation.
    monkeypatch.setattr(plan.output, "zero_", lambda: None)
    monkeypatch.setattr("mori.ops.gemm_ar.op.fx.Stream", lambda s: s)
    monkeypatch.setattr(torch.cuda, "current_stream", Mock())
    with pytest.raises(RuntimeError, match="self-test failed"):
        op.self_test(4200)
