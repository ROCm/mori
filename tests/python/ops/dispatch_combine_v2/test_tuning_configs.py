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
