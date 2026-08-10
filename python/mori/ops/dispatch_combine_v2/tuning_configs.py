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
"""Per-device, per-shape block/warp tuning for the cco-LSA dispatch/combine
kernels.

Geometry is picked per device (MI308X / MI300X / MI355X), then by
(world_size, hidden_dim, topk). Devices are told apart by PCI device id (MI300X
and MI308X are both gfx942, differing only in CU count), falling back to arch for
gfx950 — all from KFD sysfs, no torch/HIP dependency. block_num must stay
<= CU count; re-tune per GPU.
"""

from mori.ops import utils as gpu_utils


# Dispatch and combine are tuned independently: each phase has its own per-token
# SCHEDULE of (max_tok_inclusive | None, block, warp) buckets and its own per-shape
# table keyed by (world_size, hidden_dim, topk). The two tables are looked up
# separately, so a phase may reuse a schedule across topk while the other doesn't.

# ── MI308X (gfx942, 80 CU) — EP8.
_MI308X_DISP = (
    (256, 64, 8),
    (2048, 64, 16),
    (None, 64, 16),
)
_MI308X_COMB = (
    (256, 64, 4),
    (2048, 64, 4),
    (None, 80, 4),
)
# Single-shot fallback (schedule ignored) = peak-optimal.
_MI308X_DEFAULT = dict(
    dispatch_block_num=64,
    combine_block_num=80,
    warp_num_per_block=16,
    combine_warp_num_per_block=4,
    dispatch_schedule=_MI308X_DISP,
    combine_schedule=_MI308X_COMB,
)
# Tuned as fp8-dispatch + bf16-combine, so filed under "fp8" (fallback for untuned
# dtypes). hidden 4096 / 2048 reuse the 7168 schedule until separately tuned.
_MI308X_DISP_TABLE = {
    (8, 7168, 8): {"fp8": _MI308X_DISP},
    (8, 4096, 8): {"fp8": _MI308X_DISP},
    (8, 2048, 8): {"fp8": _MI308X_DISP},
}
_MI308X_COMB_TABLE = {
    (8, 7168, 8): {"fp8": _MI308X_COMB},
    (8, 4096, 8): {"fp8": _MI308X_COMB},
    (8, 2048, 8): {"fp8": _MI308X_COMB},
}

# ── MI325X (gfx942, 304 CU, DID 0x74a5) — EP8, selected by min latency. dispatch
# wants warp 8, block grows with tok (64->304); combine is latency-bound at
# small/mid tok (small block 64 + warp 4 wins), only large tok (>1024) want
# block ~0.5*CU (152) + warp 2.
_MI325X_DISP = (
    (8, 64, 8),  # <=8 tok: tiny
    (64, 64, 8),  # <=64:   small block
    (1024, 152, 8),  # <=1024: disp 0.5*CU
    (4096, 228, 8),  # <=4096
    (None, 304, 8),  # >4096 (peak)
)
_MI325X_COMB = (
    (8, 64, 2),  # <=8 tok: warp 2
    (64, 64, 4),  # <=64:   small block
    (1024, 64, 4),  # <=1024: small-block/warp4 (latency)
    (4096, 152, 2),  # <=4096: 0.5*CU/warp2 (bandwidth)
    (None, 152, 2),  # >4096 (peak)
)
_MI325X_DEFAULT = dict(
    dispatch_block_num=304,
    combine_block_num=152,
    warp_num_per_block=8,
    combine_warp_num_per_block=2,
    dispatch_schedule=_MI325X_DISP,
    combine_schedule=_MI325X_COMB,
)
# hidden 4096 / 2048 reuse the 7168 schedule until separately tuned.
_MI325X_DISP_TABLE = {
    (8, 7168, 8): {"fp8": _MI325X_DISP},
    (8, 4096, 8): {"fp8": _MI325X_DISP},
    (8, 2048, 8): {"fp8": _MI325X_DISP},
}
_MI325X_COMB_TABLE = {
    (8, 7168, 8): {"fp8": _MI325X_COMB},
    (8, 4096, 8): {"fp8": _MI325X_COMB},
    (8, 2048, 8): {"fp8": _MI325X_COMB},
}

# ── MI300X (gfx942, 304 CU) — TODO: re-tune. Falls back to CU-scaled default. ──
_MI300X_DEFAULT = None  # None => derive from CU count (see _cu_scaled_default)
_MI300X_DISP_TABLE = {}
_MI300X_COMB_TABLE = {}

# ── MI355X (gfx950, wave64) — EP8, vec4 combine gather. combine wants a small
# block (32-48) + warp up to 16; dispatch grows block 96->160. wave64 => warp <= 16
# (1024-thread block). Geometry is topk-independent.
# dispatch (bf16 token dtype):
_MI355X_DISP_BF16 = (
    (128, 128, 8),
    (256, 96, 8),
    (1024, 96, 8),
    (4096, 160, 8),
    (None, 128, 16),
)
# dispatch (fp8 token dtype):
_MI355X_DISP_FP8 = (
    (128, 128, 8),
    (256, 160, 8),
    (1024, 128, 8),
    (4096, 160, 8),
    (None, 128, 16),
)
# dispatch (fp4 token dtype, 0.5 B/elem):
_MI355X_DISP_FP4 = (
    (256, 128, 4),
    (2048, 144, 4),
    (None, 128, 8),
)
# combine (bf16 / fp8 dispatch share the same combine geometry):
_MI355X_COMB = (
    (128, 32, 8),
    (256, 64, 8),
    (1024, 32, 16),
    (4096, 32, 16),
    (None, 48, 16),
)
# combine (fp4):
_MI355X_COMB_FP4 = (
    (256, 32, 8),
    (2048, 64, 16),
    (None, 64, 16),
)
_MI355X_DEFAULT = dict(
    dispatch_block_num=128,
    combine_block_num=48,
    warp_num_per_block=16,
    combine_warp_num_per_block=16,
    dispatch_schedule=_MI355X_DISP_BF16,
    combine_schedule=_MI355X_COMB,
)
_MI355X_DISP_TABLE = {
    (8, 7168, 8): {
        "bf16": _MI355X_DISP_BF16,
        "fp8": _MI355X_DISP_FP8,
        "fp4": _MI355X_DISP_FP4,
    },
    (8, 7168, 6): {
        "bf16": _MI355X_DISP_BF16,
        "fp8": _MI355X_DISP_FP8,
        "fp4": _MI355X_DISP_FP4,
    },
}
_MI355X_COMB_TABLE = {
    (8, 7168, 8): {
        "bf16": _MI355X_COMB,
        "fp8": _MI355X_COMB,
        "fp4": _MI355X_COMB_FP4,
    },
    (8, 7168, 6): {
        "bf16": _MI355X_COMB,
        "fp8": _MI355X_COMB,
        "fp4": _MI355X_COMB_FP4,
    },
}

_GFX1250_DISP_BF16 = (
    (256, 128, 16),  # <=128:  latency-bound
    (512, 192, 32),  # <=512:  disp peak
    (4096, 192, 32),  # <=4096
    (None, 192, 32),  # >4096
)
_GFX1250_COMB_BF16 = (
    (256, 64, 4),  # <=128:  combine warp 4 (fewest warps win)
    (512, 128, 4),  # <=512:  comb 128/4
    (4096, 128, 8),  # <=4096: comb block 128
    (None, 192, 8),  # >4096:  comb warp 8
)

_GFX1250_DEFAULT = dict(
    dispatch_block_num=192,
    combine_block_num=192,
    warp_num_per_block=32,
    combine_warp_num_per_block=16,
    dispatch_schedule=_GFX1250_DISP_BF16,
    combine_schedule=_GFX1250_COMB_BF16,
)
# DeepSeek-V4-Pro shape (hidden 7168, topk 6, 384 experts). EP4 bf16, load-once/
# store-many + 4-way dispatch kernel (host-selected by load_once_threshold=4096).
# dispatch wants block=256 + warp 32 above 1024 tok (<=256 tok is latency-flat).
# combine also wants block=256 (=CU, blocks stay co-resident 1/CU), warp ramps 4->16;
# only tiny tok (<=256) wants a small block (256 starves it).
_GFX1250_DISP_BF16_T6 = (
    (256, 128, 16),  # <=256:  disp latency-flat
    (512, 192, 32),  # <=512:  disp warp32 (192x32==256x32, min block)
    (1024, 192, 32),  # <=1024
    (None, 256, 32),  # >1024:  disp 256x32
)
_GFX1250_COMB_BF16_T6 = (
    (256, 64, 4),  # <=256:  comb small-block 64/4
    (512, 128, 4),  # <=512:  comb 128/4
    (1024, 128, 8),  # <=1024: comb 128/8
    (None, 256, 16),  # >1024:  comb 256x16
)
# EP8 cross-node (2 nodes x 4 GPUs over UALink), bf16, vec4 combine-gather.
# dispatch block 128, warp ramps 16->32. combine block 64 uniformly best, warp
# ramps 4->16. Fabric caps disp ~200 GB/s; geometry is world_size-independent so
# this also serves single-node EP8. topk6 reuses this schedule (topk-independent).
_GFX1250_DISP_BF16_EP8 = (
    (256, 128, 16),  # <=256:  disp warp 16 (latency-bound)
    (1024, 128, 32),  # <=1024: disp warp 32
    (None, 128, 32),  # >1024 (peak)
)
_GFX1250_COMB_BF16_EP8 = (
    (256, 64, 4),  # <=256:  comb 64/4 (latency-bound)
    (1024, 64, 8),  # <=1024: comb 64/8
    (None, 64, 16),  # >1024 (peak)
)
# bf16-tuned (EP4 + EP8). fp8/fp4 fall back to the bf16 schedule until separately tuned.
_GFX1250_DISP_TABLE = {
    (4, 7168, 8): {"bf16": _GFX1250_DISP_BF16},
    (4, 7168, 6): {"bf16": _GFX1250_DISP_BF16_T6},  # DeepSeek-V4-Pro
    (8, 7168, 8): {"bf16": _GFX1250_DISP_BF16_EP8},  # cross-node / single-node EP8
    # V4-Pro topk=6 EP8: topk-independent, reuse the topk=8 schedule.
    (8, 7168, 6): {"bf16": _GFX1250_DISP_BF16_EP8},
}
_GFX1250_COMB_TABLE = {
    (4, 7168, 8): {"bf16": _GFX1250_COMB_BF16},
    (4, 7168, 6): {"bf16": _GFX1250_COMB_BF16_T6},  # DeepSeek-V4-Pro
    (8, 7168, 8): {"bf16": _GFX1250_COMB_BF16_EP8},  # cross-node / single-node EP8
    (8, 7168, 6): {"bf16": _GFX1250_COMB_BF16_EP8},
}

# key -> (single-shot default, dispatch table, combine table)
_DEVICES = {
    "mi308x": (_MI308X_DEFAULT, _MI308X_DISP_TABLE, _MI308X_COMB_TABLE),
    "mi325x": (_MI325X_DEFAULT, _MI325X_DISP_TABLE, _MI325X_COMB_TABLE),
    "mi300x": (_MI300X_DEFAULT, _MI300X_DISP_TABLE, _MI300X_COMB_TABLE),
    "mi355x": (_MI355X_DEFAULT, _MI355X_DISP_TABLE, _MI355X_COMB_TABLE),
    "gfx1250": (_GFX1250_DEFAULT, _GFX1250_DISP_TABLE, _GFX1250_COMB_TABLE),
}


# Kept as a module-level name: tests and benches call tuning_configs._topology().
_topology = gpu_utils.topology


def _cu_count():
    return gpu_utils.cu_count()


# Models with no table of their own that reuse a tuned sibling's schedule.
_MODEL_ALIAS = {"mi350x": "mi355x"}  # same die, same 256 CU; clocks differ


def _device_key():
    """Map the current GPU to a device table key: PCI DID first, then arch.
    Returns None if unknown (caller uses a CU-scaled default)."""
    model = gpu_utils.detect_model()
    if model is not None:
        return _MODEL_ALIAS.get(model, model)
    if gpu_utils.topology()[1] == 120500:  # gfx1250 has no MI model name
        return "gfx1250"
    return None


def _cu_scaled_default():
    """Untuned fallback: ~1 block/CU for combine (<= CU count), many warps for
    dispatch, few warps for combine, no per-token schedule (single-shot). Used
    for devices without a measured table."""
    cu = _cu_count() or 80
    return dict(
        dispatch_block_num=cu,
        combine_block_num=cu,
        warp_num_per_block=16,
        combine_warp_num_per_block=4,
        dispatch_schedule=None,
        combine_schedule=None,
    )


def lookup(world_size, hidden_dim, topk, dtype="fp8"):
    """Return {dispatch_block_num, combine_block_num, warp_num_per_block,
    combine_warp_num_per_block, dispatch_schedule, combine_schedule} for the
    current GPU, shape, and dtype.

    `dtype` (the token / dispatch dtype: "bf16" | "fp8" | "fp4") selects the
    per-dtype schedule, because dtype sets the communication volume (fp4 = 0.5 B,
    fp8 = 1 B, bf16 = 2 B per element) and thus the best block/warp. It falls back
    to the "fp8" schedule, then to the device default, when a dtype isn't tuned.

    dispatch_schedule / combine_schedule (or None) are independent per-token-count
    launch plans: a tuple of (max_tok_inclusive | None, block, warp) buckets,
    ascending; the op precompiles the distinct (block, warp) variants and picks a
    bucket at runtime from cur_rank_num_token. The two phases have separate tables
    (each keyed by (world_size, hidden_dim, topk)) and are looked up independently.
    dispatch_block_num / warp_num_per_block / combine_block_num /
    combine_warp_num_per_block are the single-shot fallback used when a schedule
    is None."""
    key = _device_key()
    if key is None or key not in _DEVICES:
        return _cu_scaled_default()
    dev_default, disp_table, comb_table = _DEVICES[key]
    base = dict(dev_default) if dev_default is not None else _cu_scaled_default()
    base.setdefault("dispatch_schedule", None)
    base.setdefault("combine_schedule", None)
    shape = (world_size, hidden_dim, topk)
    d_entry = disp_table.get(shape)
    if d_entry:
        base["dispatch_schedule"] = (
            d_entry.get(dtype) or d_entry.get("fp8") or base["dispatch_schedule"]
        )
    c_entry = comb_table.get(shape)
    if c_entry:
        base["combine_schedule"] = (
            c_entry.get(dtype) or c_entry.get("fp8") or base["combine_schedule"]
        )
    return base
