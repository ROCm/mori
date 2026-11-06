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
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Check calibrated launch choices and supported RDNA4 configurations."""

import csv
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "rdna4_policy_test", ROOT / "python/mori/ops/_rdna4_ep.py"
)
policy_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy_module
spec.loader.exec_module(policy_module)
Rdna4EpPolicy = policy_module.Rdna4EpPolicy
with (Path(__file__).parent / "fixtures/rdna4_ep_launches.csv").open() as stream:
    RECORDED = list(csv.DictReader(stream))


@pytest.mark.parametrize("row", RECORDED)
def test_recorded_kernel_and_launch(row):
    policy = Rdna4EpPolicy(
        int(row["world_size"]), int(row["hidden"]), int(row["capacity"]), row["dtype"]
    )
    tokens = int(row["tokens"])
    assert policy.dispatch_kernel(tokens) == row["dispatch_kernel"]
    assert policy.combine_kernel(tokens) == row["combine_kernel"]
    for phase in ("dispatch", "combine"):
        assert policy.launch(tokens, dispatch=phase == "dispatch") == (
            int(row[f"{phase}_blocks"]),
            int(row[f"{phase}_warps"]),
        )


@pytest.mark.parametrize("tokens", [0, 1, 64, 192, 256])
def test_ragged_small_rank_uses_compatible_counts_in_large_allocation(tokens):
    policy = Rdna4EpPolicy(2, 4096, 512, "bf16")
    assert policy.dispatch_kernel(tokens).endswith("_small_v2_compat")
    assert policy.dispatch_kernel(257).endswith("Kernel_bf16")
    assert "small_v2" in policy.combine_kernel(tokens)


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("hidden", [2056, 8184])
def test_fixed_layout_guard(world, hidden):
    policy = Rdna4EpPolicy(world, hidden, 256, "fp16")
    policy.check_input("fp16", hidden)
    for dtype, width, external in [
        ("fp16", 4096, True),
        ("bf16", hidden, True),
        ("fp16", hidden, False),
    ]:
        with pytest.raises(ValueError, match="fixed config.hidden_dim"):
            policy.check_input(dtype, width, external=external)


@pytest.mark.parametrize(
    "args",
    [
        (8, 4096, 256, "bf16"),
        (2, 2050, 256, "bf16"),
        (4, 1024, 256, "fp16"),
        (2, 8193, 256, "bf16"),
        (4, 4096, 0, "bf16"),
        (2, 4096, 256, "f32"),
        (2, 4096, 256, "bf16", 0),
        (4, 4096, 256, "fp16", 65),
    ],
)
def test_invalid_policy_rejected(args):
    with pytest.raises(ValueError, match="MORI_RDNA4_EP"):
        Rdna4EpPolicy(*args)


@pytest.mark.parametrize("world", [2, 4])
def test_tile_threshold_and_token_boundary(world):
    policy = Rdna4EpPolicy(world, 4096, 512, "bf16")
    assert policy.combine_kernel(64).endswith("512")
    assert policy.combine_kernel(65).endswith("1024")
    assert policy.combine_kernel(256).endswith("1024")
    assert not policy.combine_kernel(257).endswith(("512", "1024"))
