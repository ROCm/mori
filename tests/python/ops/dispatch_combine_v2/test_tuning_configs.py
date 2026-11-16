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
"""Host-only checks of the flydsl EP v2 geometry lookup (no GPU needed).

The untuned fallback sizes combine blocks from the CU count. A part with more
CUs than the xdb flag array has slots used to fail at op construction.
"""

import pytest

from mori.ops.dispatch_combine_v2 import tuning_configs as tc


@pytest.mark.parametrize("cu, expected", [(80, 80), (256, 256), (304, 256)])
def test_untuned_combine_blocks_fit_flag_slots(monkeypatch, cu, expected):
    monkeypatch.setattr(tc, "_cu_count", lambda: cu)
    monkeypatch.setattr(tc, "_device_key", lambda: None)
    cfg = tc.lookup(8, 7168, 8)
    assert cfg["combine_block_num"] == expected
    assert cfg["dispatch_block_num"] == cu  # only combine has the slot limit


def test_device_without_default_uses_the_clamped_fallback(monkeypatch):
    # A known device whose default is None takes the same CU-scaled path.
    monkeypatch.setattr(tc, "_cu_count", lambda: 304)
    monkeypatch.setattr(tc, "_device_key", lambda: "tuned_none")
    monkeypatch.setitem(tc._DEVICES, "tuned_none", (None, {}))
    assert tc.lookup(2, 7168, 8)["combine_block_num"] <= tc._MAX_COMBINE_BLOCKS


def test_max_combine_blocks_matches_kernel_flag_slots():
    pytest.importorskip("flydsl")
    from mori.ops.dispatch_combine_v2.intranode_kernels import xdb_flag_slots

    assert tc._MAX_COMBINE_BLOCKS == xdb_flag_slots


# (CU count, wave size) per device table.
_DEVICE_LIMITS = {
    "mi308x": (80, 64),
    "mi325x": (304, 64),
    "mi300x": (304, 64),
    "mi355x": (256, 64),
    "gfx1250": (256, 32),
}


def _schedules(dev_default, dev_table):
    if dev_default and dev_default.get("schedule"):
        yield "default", dev_default["schedule"]
    for key, by_dtype in dev_table.items():
        for dtype, schedule in by_dtype.items():
            yield f"{key} {dtype}", schedule


def _check_geometry(name, cu, wave, disp_block, disp_warp, comb_block, comb_warp):
    assert 1 <= disp_block <= cu, name
    assert 1 <= comb_block <= min(cu, tc._MAX_COMBINE_BLOCKS), name
    assert 1 <= disp_warp and disp_warp * wave <= 1024, name
    assert 1 <= comb_warp and comb_warp * wave <= 1024, name


@pytest.mark.parametrize("device", sorted(tc._DEVICES))
def test_every_schedule_is_well_formed(device):
    cu, wave = _DEVICE_LIMITS[device]
    dev_default, dev_table = tc._DEVICES[device]
    if dev_default:
        keys = (
            "dispatch_block_num",
            "warp_num_per_block",
            "combine_block_num",
            "combine_warp_num_per_block",
        )
        _check_geometry("single-shot", cu, wave, *(dev_default[k] for k in keys))
    for name, schedule in _schedules(dev_default, dev_table):
        ceilings = [row[0] for row in schedule[:-1]]
        assert schedule[-1][0] is None and None not in ceilings, name
        assert ceilings == sorted(set(ceilings)) and min(ceilings, default=1) >= 1, name
        for row in schedule:
            _check_geometry(name, cu, wave, *row[1:])
