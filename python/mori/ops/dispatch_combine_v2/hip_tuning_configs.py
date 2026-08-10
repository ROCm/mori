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
"""Launch-geometry tuning for the HIP/JIT EP kernels — separate from FlyDSL's.

The C++/HIP kernels are a different implementation from the FlyDSL ones: different
register pressure, occupancy, and best warp count (e.g. HIP combine peaks at 8
warps on gfx950 EP8 where FlyDSL uses 4). So they carry their OWN tuned geometry
and never borrow FlyDSL's schedule. This module deliberately does NOT import
``tuning_configs`` (FlyDSL's table); it shares only the hardware-detection
primitive (``mori.ops.utils``), the same one FlyDSL's table uses.

Return contract matches ``tuning_configs.lookup`` so the op's ``_resolve_geometry``
treats both backends uniformly:

    {dispatch_block_num, combine_block_num, warp_num_per_block,
     combine_warp_num_per_block, schedule}

``schedule`` (or None) is a per-token-count launch plan: a tuple of
(max_tok_inclusive | None, disp_block, disp_warp, comb_block, comb_warp) buckets,
ascending; the op precompiles the distinct (block, warp) variants and picks a
bucket at runtime from cur_rank_num_token. When schedule is None the single-shot
block/warp fields are used.

SKELETON: the per-device/per-shape tables (``_HIP_TABLES``) are intentionally
empty. Until a shape is separately tuned FOR THE HIP KERNEL, lookup returns HIP's
single-shot default (schedule=None) -- one (block, warp) variant, whose values
mirror the C++ ``MakeEpCfg`` arch default so a bare C++ caller and this path agree.
Fill ``_HIP_TABLES`` from a ``bench_ep.py`` sweep of the HIP kernel; do NOT paste
FlyDSL's ``tuning_configs`` numbers in -- different kernel, different optimum.
"""

from __future__ import annotations

from mori.ops import utils as _gpu

# Models with no table of their own that reuse a tuned sibling's key.
_MODEL_ALIAS = {"mi350x": "mi355x"}  # same die / CU count


def _device_key():
    """Map the current GPU to a table key: PCI model first, then arch. None if
    unknown. Same detection FlyDSL's table uses, kept here so this module does not
    depend on tuning_configs."""
    model = _gpu.detect_model()
    if model is not None:
        return _MODEL_ALIAS.get(model, model)
    if _gpu.topology()[1] == 120500:  # gfx1250 has no MI model name
        return "gfx1250"
    return None


def _hip_default() -> dict:
    """HIP single-shot default (no tuned schedule). Mirrors the C++ MakeEpCfg arch
    default: dispatch 64 blocks / 16 warps, combine 80 blocks / 8 warps -- the
    latter measured for the HIP combine kernel on gfx950 EP8."""
    return dict(
        dispatch_block_num=64,
        combine_block_num=80,
        warp_num_per_block=16,
        combine_warp_num_per_block=8,
        schedule=None,
    )


# Per-device HIP schedule tables:
#   device_key -> {(world_size, hidden_dim, topk): {dtype: schedule}}
# schedule = ((max_tok|None, disp_block, disp_warp, comb_block, comb_warp), ...)
#
# EMPTY on purpose (skeleton). Fill from a HIP-kernel bench sweep; never copy
# FlyDSL's tuning_configs values in.
_HIP_TABLES: dict = {
    # "gfx950": {(8, 7168, 8): {"bf16": (...buckets...)}},   # TODO: tune the HIP kernel
}


def lookup(world_size, hidden_dim, topk, dtype="bf16") -> dict:
    """HIP geometry for this device/shape/dtype, from HIP's own table.

    Skeleton behaviour: no shape is tuned yet, so this returns the HIP single-shot
    default (schedule=None) for everything. Add entries to ``_HIP_TABLES`` to give
    a shape a per-token schedule."""
    base = _hip_default()
    table = _HIP_TABLES.get(_device_key())
    if table:
        entry = table.get((world_size, hidden_dim, topk))
        if entry:
            base["schedule"] = entry.get(dtype) or entry.get("bf16") or base["schedule"]
    return base
