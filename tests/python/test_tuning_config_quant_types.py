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
"""Regression tests for quant_type normalization in the AUTO tuning config.

`quant_type_to_config_str` translates a quant type into the string the tuned
lookup tables are keyed by. `dispatch_combine.py` gained full `fp4_blockwise`
support -- `_QUANT_TYPE_MAP`, `_BLOCKWISE_COMBINE_QUANT_TYPES`,
`_FP4_COMBINE_KERNELS` -- but this function was not updated alongside it, so
passing `Fp4BlockwiseQuant` raised `ValueError`.

That mattered more than a missing combine lookup: the exception escapes before
*either* table is consulted, so a config using fp4 blockwise combine also lost
its tuned **dispatch** rules and fell back to hard-coded launch geometry. The
only signal is one `logger.warning` from the `except` in `dispatch_combine.py`
("AUTO tuning: failed to load config"); results stay correct, so the cost is
throughput nobody is looking for.

These pin both directions of the mapping and, deliberately, that a genuinely
unknown type still raises -- the fix must not degrade into accepting anything.
"""

import pytest
from mori.ops.dispatch_combine import EpDispatchCombineQuantType
from mori.ops.tuning_config import (
    _QUANT_TYPE_CONFIG_STRS,
    quant_type_to_config_str,
)


@pytest.mark.parametrize(
    "quant_type,expected",
    [
        (EpDispatchCombineQuantType.None_, "none"),
        (EpDispatchCombineQuantType.Fp8DirectCast, "fp8_direct_cast"),
        (EpDispatchCombineQuantType.Fp8BlockwiseQuant, "fp8_blockwise"),
        (EpDispatchCombineQuantType.Fp4BlockwiseQuant, "fp4_blockwise"),
    ],
)
def test_enum_maps_to_config_str(quant_type, expected):
    """Every quant type dispatch_combine knows must be nameable here, or its
    config silently loses tuned rules."""
    assert quant_type_to_config_str(quant_type) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("none", "none"),
        ("none_", "none"),
        ("fp8_direct_cast", "fp8_direct_cast"),
        ("fp8_blockwise", "fp8_blockwise"),
        ("fp4_blockwise", "fp4_blockwise"),
        ("  FP4_BlockWise  ", "fp4_blockwise"),
    ],
)
def test_string_forms_normalize(text, expected):
    assert quant_type_to_config_str(text) == expected


def test_every_enum_member_maps_into_the_config_str_set():
    """Every member must map, and map into the set that gates the string path.

    Deliberately no try/except around the call: an unmapped member is exactly
    the defect this PR fixes, so swallowing its ValueError would make the test
    green for the very drift it exists to catch. A member added to the enum
    without a mapping must fail here.
    """
    # pybind11 enums are not directly iterable; go through __members__.
    for name, quant_type in EpDispatchCombineQuantType.__members__.items():
        got = quant_type_to_config_str(quant_type)
        assert (
            got in _QUANT_TYPE_CONFIG_STRS
        ), f"{name} maps to {got!r}, which is not an accepted config string"


@pytest.mark.parametrize("bad", ["", "fp16_blockwise", "fp4", "totally_unknown"])
def test_unknown_quant_types_still_raise(bad):
    """The fix adds one mapping; it must not turn this into a function that
    accepts anything. An unnamed type should fail loudly rather than key the
    lookup on a string no table contains."""
    with pytest.raises(ValueError):
        quant_type_to_config_str(bad)
