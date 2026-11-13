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
"""Tuning profiles: "peak" is the tuned tables, "capped" keeps every grid within 64 blocks."""

import pytest

from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig
from mori.ops.dispatch_combine_v2 import hip_tuning_configs as tc

DTYPES = ("bf16", "fp8", "fp4")
QUANTS = ("none", *tc._COMBINE_PUSH_TABLE)


@pytest.fixture
def gfx1250(monkeypatch):
    monkeypatch.delenv("MORI_EP_TUNING_PROFILE", raising=False)
    monkeypatch.setattr(tc, "_device_key", lambda: "gfx1250")
    monkeypatch.setattr(tc._gpu, "asic_revision", lambda: 1)
    monkeypatch.setattr(tc._gpu, "cu_count", lambda: 0)  # 0: no CU clamp


def shapes():
    found = set()
    for table in (
        tc._DISPATCH_TABLE,
        tc._COMBINE_TABLE,
        tc._CAPPED_DISPATCH_TABLE,
        tc._CAPPED_COMBINE_TABLE,
        *tc._COMBINE_PUSH_TABLE.values(),
        *tc._CAPPED_COMBINE_PUSH_TABLE.values(),
    ):
        for dev in ("gfx1250", "gfx1250-strict"):
            found |= set(table.get(dev, {}))
    return sorted(found, key=str)


def every_lookup(**kw):
    for world, hidden, topk, epr in shapes():
        for dtype in DTYPES:
            for quant in QUANTS:
                key = (world, hidden, topk, epr, dtype, quant)
                args = dict(dtype=dtype, experts_per_rank=epr, quant_type=quant)
                yield key, tc.lookup(world, hidden, topk, **args, **kw)


def widest(t):
    blocks = [t["dispatch_block_num"], t["combine_block_num"]]
    return max(blocks + [max(b[1], b[3]) for b in t["schedule"] or ()])


@pytest.mark.parametrize("rev", [1, 0])
def test_capped_keeps_every_grid_within_64_blocks(gfx1250, monkeypatch, rev):
    monkeypatch.setattr(tc._gpu, "asic_revision", lambda: rev)
    for key, t in every_lookup(profile="capped"):
        assert widest(t) <= tc._CAPPED_BLOCKS, key


def test_ll128_and_bf16_push_rows_fit_capped():
    for per_dev in tc._LL128_TABLE.values():
        for per_shape in per_dev.values():
            for plans in per_shape.values():
                for plan in plans.values():
                    assert plan["dispatch"][0] <= tc._CAPPED_BLOCKS
                    assert plan["combine"][0] <= tc._CAPPED_BLOCKS
    for per_shape in tc._BF16_PUSH_TABLE.values():
        for rows in per_shape.values():
            assert all(row[1] <= tc._CAPPED_BLOCKS for row in rows)


def test_peak_is_the_default(gfx1250):
    peak = dict(every_lookup(profile="peak"))
    assert dict(every_lookup()) == peak


def test_capped_reuses_peak_rows_that_fit(gfx1250):
    peak = dict(every_lookup(profile="peak"))
    for key, t in every_lookup(profile="capped"):
        if widest(peak[key]) <= tc._CAPPED_BLOCKS:
            assert t == peak[key], key


def test_capped_takes_its_own_rows(gfx1250):
    for topk, epr in ((6, 96), (3, 32)):
        t = tc.lookup(4, 5120, topk, experts_per_rank=epr, profile="capped")
        assert t["schedule"] == (
            (128, 64, 4, 64, 8),
            (256, 64, 4, 64, 16),
            (1024, 64, 8, 64, 16),
            (None, 64, 16, 64, 16),
        )


def test_capped_without_rows_keeps_the_buckets_that_fit_and_warns(gfx1250, caplog):
    tc._warn_no_capped_rows.cache_clear()
    peak = tc.lookup(4, 7168, 8, dtype="bf16", profile="peak")
    capped = tc.lookup(4, 7168, 8, dtype="bf16", profile="capped")

    def dispatch_at(t, ct):
        return tc.pick_bucket(t["schedule"], ct)[1:3]

    assert dispatch_at(peak, 256) == dispatch_at(capped, 256) == (64, 8)
    assert dispatch_at(peak, 8192)[0] > tc._CAPPED_BLOCKS
    assert dispatch_at(capped, 8192) == (64, 16)
    assert "no capped dispatch rows for EP4 hidden 7168 topk 8" in caplog.text


def test_capped_keeps_a_fitting_bucket_after_a_wide_one(gfx1250, monkeypatch):
    shape = (4, 4096, 8, None)
    peak = ((512, 64, 8), (4096, 256, 8), (8192, 64, 16), (None, 192, 32))
    monkeypatch.setitem(tc._DISPATCH_TABLE, "gfx1250", {shape: {None: peak}})
    t = tc.lookup(4, 4096, 8, profile="capped")
    assert tuple(row[:3] for row in t["schedule"]) == (
        (512, 64, 8),
        (8192, 64, 16),
        (None, 64, 32),
    )


def test_capped_push_buckets_keep_their_warps(gfx1250):
    kw = dict(dtype="fp4", quant_type="fp4_blockwise")
    peak = tc.lookup(4, 7168, 6, profile="peak", **kw)["schedule"]
    capped = tc.lookup(4, 7168, 6, profile="capped", **kw)["schedule"]
    wide = [(row[0], row[4]) for row in peak if row[3] > tc._CAPPED_BLOCKS]
    assert wide
    for ct, warps in wide:
        assert tc.pick_bucket(capped, ct)[3:] == (tc._CAPPED_BLOCKS, warps)


def test_capped_push_row_without_a_peak_push_row(gfx1250, monkeypatch):
    shape = (4, 4096, 8, None)
    rows = ((None, 64, 24),)
    monkeypatch.setitem(
        tc._CAPPED_COMBINE_PUSH_TABLE, "fp8_blockwise", {"gfx1250": {shape: rows}}
    )
    t = tc.lookup(4, 4096, 8, quant_type="fp8_blockwise", profile="capped")
    assert {row[3:] for row in t["schedule"]} == {(64, 24)}


def test_profile_precedence(monkeypatch):
    monkeypatch.delenv("MORI_EP_TUNING_PROFILE", raising=False)
    assert tc.tuning_profile() == "peak"
    assert tc.tuning_profile(default="capped") == "capped"
    monkeypatch.setenv("MORI_EP_TUNING_PROFILE", "capped")
    assert tc.tuning_profile() == "capped"
    assert tc.tuning_profile("peak", default="capped") == "peak"
    for bad in ("Capped", "max"):
        with pytest.raises(ValueError, match="tuning profile"):
            tc.tuning_profile(bad)
    monkeypatch.setenv("MORI_EP_TUNING_PROFILE", "fast")
    with pytest.raises(ValueError, match="tuning profile"):
        tc.tuning_profile()


@pytest.mark.parametrize(
    "key, rev, want",
    [("gfx1250", 1, "capped"), ("gfx1250", 0, "capped"), ("mi355x", None, "peak")],
)
def test_default_profile(monkeypatch, key, rev, want):
    monkeypatch.setattr(tc, "_device_key", lambda: key)
    monkeypatch.setattr(tc._gpu, "asic_revision", lambda: rev)
    assert tc.default_profile() == want


def config(**kwargs):
    values = dict(
        rank=0,
        world_size=4,
        hidden_dim=5120,
        max_num_inp_token_per_rank=8192,
        num_experts_per_rank=96,
        num_experts_per_token=6,
        kernel_backend="hip",
    )
    values.update(kwargs)
    return EpDispatchCombineConfig(**values)


def test_op_is_capped_on_gfx1250_unless_told(gfx1250, monkeypatch):
    capped = tc.lookup(4, 5120, 6, experts_per_rank=96, profile="capped")["schedule"]
    peak = tc.lookup(4, 5120, 6, experts_per_rank=96, profile="peak")["schedule"]
    assert capped != peak
    cfg = config()
    assert (cfg.tuning_profile, cfg.schedule) == ("capped", capped)
    cfg = config(tuning_profile="peak")
    assert (cfg.tuning_profile, cfg.schedule) == ("peak", peak)
    monkeypatch.setenv("MORI_EP_TUNING_PROFILE", "peak")
    assert config().schedule == peak
    assert config(tuning_profile="capped").schedule == capped


def test_asyncll_stays_outside_the_profiles(gfx1250, caplog):
    tc._warn_no_capped_rows.cache_clear()
    cfg = config(
        hidden_dim=7168, num_experts_per_rank=64, num_experts_per_token=8, async_ll=True
    )
    assert cfg.tuning_profile is None
    assert "no capped" not in caplog.text


def test_a_bad_profile_is_refused_even_with_pinned_geometry(gfx1250):
    with pytest.raises(ValueError, match="tuning_profile"):
        config(tuning_profile="lean", dispatch_block_num=64)
