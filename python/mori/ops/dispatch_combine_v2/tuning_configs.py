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
from .flydsl_compat import HAS_BUFFER_OPS


# ── MI308X (gfx942, 80 CU) — EP8. Per-token SCHEDULE of
# (max_tok_inclusive | None, disp_block, disp_warp, comb_block, comb_warp) buckets.
_MI308X_SCHEDULE = (
    (256, 64, 8, 64, 4),
    (2048, 64, 16, 64, 4),
    (None, 64, 16, 80, 4),
)
# Single-shot fallback (schedule ignored) = peak-optimal.
_MI308X_DEFAULT = dict(
    dispatch_block_num=64,
    combine_block_num=80,
    warp_num_per_block=16,
    combine_warp_num_per_block=4,
    schedule=_MI308X_SCHEDULE,
)
# Tuned as fp8-dispatch + bf16-combine, so filed under "fp8" (fallback for untuned
# dtypes). hidden 4096 / 2048 reuse the 7168 schedule until separately tuned.
_MI308X_TABLE = {
    (8, 7168, 8): {"fp8": _MI308X_SCHEDULE},
    (8, 4096, 8): {"fp8": _MI308X_SCHEDULE},
    (8, 2048, 8): {"fp8": _MI308X_SCHEDULE},
}

# ── MI325X (gfx942, 304 CU, DID 0x74a5) — EP8, selected by min latency. dispatch
# wants warp 8, block grows with tok (64->304); combine is latency-bound at
# small/mid tok (small block 64 + warp 4 wins), only large tok (>1024) want
# block ~0.5*CU (152) + warp 2.
_MI325X_SCHEDULE = (
    (8, 64, 8, 64, 2),  # <=8 tok: tiny, combine warp 2
    (64, 64, 8, 64, 4),  # <=64:   small block both
    (1024, 152, 8, 64, 4),  # <=1024: disp 0.5*CU, comb small-block/warp4 (latency)
    (4096, 228, 8, 152, 2),  # <=4096: comb 0.5*CU/warp2 (bandwidth)
    (None, 304, 8, 152, 2),  # >4096 (peak)
)
_MI325X_DEFAULT = dict(
    dispatch_block_num=304,
    combine_block_num=152,
    warp_num_per_block=8,
    combine_warp_num_per_block=2,
    schedule=_MI325X_SCHEDULE,
)
# hidden 4096 / 2048 reuse the 7168 schedule until separately tuned.
_MI325X_TABLE = {
    (8, 7168, 8): {"fp8": _MI325X_SCHEDULE},
    (8, 4096, 8): {"fp8": _MI325X_SCHEDULE},
    (8, 2048, 8): {"fp8": _MI325X_SCHEDULE},
}

# ── MI300X (gfx942, 304 CU) — TODO: re-tune. Falls back to CU-scaled default. ──
_MI300X_DEFAULT = None  # None => derive from CU count (see _cu_scaled_default)
_MI300X_TABLE = {}

# ── MI355X (gfx950, wave64) — EP8, vec4 combine gather. combine wants a small
# block (32-48) + warp up to 16; dispatch uses 64-160 blocks. wave64 => warp <= 16
# (1024-thread block).
# bf16 dispatch + bf16 combine, topk=8. Re-tuned 2026-08-11: after the dispatch
# atomic/cache-policy optimizations, 96x4 wins the 109-128 bucket; <=108 and
# topk=6 still prefer their pre-existing plan, so keep the topk=6/default
# schedule separate.
_MI355X_SCHED_BF16 = (
    (108, 128, 8, 32, 8),
    (128, 96, 4, 32, 8),
    (256, 96, 8, 64, 8),
    (1024, 96, 8, 32, 16),
    (4096, 160, 8, 32, 16),
    (None, 128, 16, 48, 16),
)
# FlyDSL >=0.3 ignores buffer cache modifiers. Keep the pre-cache-policy
# geometry there; 96x4 was tuned together with SC0|NT stores.
_MI355X_SCHED_BF16_NO_CACHE_HINT = (
    (108, 128, 8, 32, 8),
    (128, 64, 8, 32, 8),
    (256, 96, 8, 64, 8),
    (1024, 96, 8, 32, 16),
    (4096, 160, 8, 32, 16),
    (None, 128, 16, 48, 16),
)
_MI355X_TOKEN_CENTRIC_BF16 = (
    (512, 64, 8),
    (1024, 96, 8),
    (4096, 160, 8),
)
# Preserve the pre-existing plan for topk=6 and untuned shapes. On the measured
# topk=6 / 384-expert shape, 64x8 regresses 112-token dispatch by 3.5%.
_MI355X_SCHED_BF16_BASE = (
    (128, 128, 8, 32, 8),
    (256, 96, 8, 64, 8),
    (1024, 96, 8, 32, 16),
    (4096, 160, 8, 32, 16),
    (None, 128, 16, 48, 16),
)
# fp8 dispatch + bf16 combine (combine geometry shared with bf16):
_MI355X_SCHED_FP8 = (
    (128, 128, 8, 32, 8),
    (256, 160, 8, 64, 8),
    (1024, 128, 8, 32, 16),
    (4096, 160, 8, 32, 16),
    (None, 128, 16, 48, 16),
)
# fp4 dispatch + fp4 combine (0.5 B/elem):
_MI355X_SCHED_FP4 = (
    (256, 128, 4, 32, 8),
    (2048, 144, 4, 64, 16),
    (None, 128, 8, 64, 16),
)
# fp4 dispatch + bf16 combine (ASYMMETRIC, the SGLang/aiter path): dispatch
# geometry from the fp4 sweep, combine geometry from bf16 -- combine moves 2 B/elem
# here, not 0.5, so the fp4-combine geometry above does not apply. Thresholds are
# the union of both schedules' buckets. Mirrors what _MI355X_SCHED_FP8 already does.
_MI355X_SCHED_FP4_DISP_BF16_COMB = (
    # MEASURED 2026-08-03: 2-pass block x warp sweep, EP8 gfx950, fp4 dispatch +
    # bf16 combine, hidden=7168 topk=6 (DSv4-Pro, 384 experts). Large buckets want
    # dispatch warp=4 (not 8: 8192 551->481us, 4096 300->271us); small buckets need
    # their own combine geometry (256: 197->114us, 128: 172->139us).
    (128, 128, 4, 128, 16),
    (256, 160, 4, 32, 8),
    (2048, 144, 8, 32, 16),
    (None, 128, 4, 48, 16),
)
# fp4 dispatch + bf16 combine, RE-TUNED 2026-08-04 specifically for topk=6
# (DSv4-Pro, hidden=7168, 384 experts, EP8 gfx950). 2-pass block x warp sweep
# (public op.dispatch/op.combine API, fp4-asym), 2x-unroll production kernel.
# Supersedes the shared _MI355X_SCHED_FP4_DISP_BF16_COMB for topk=6. Key fixes vs that table:
#   ct=2048 dispatch 144x8 was the WORST geometry in-bucket (138 GB/s); 96x8 -> 205
#           GB/s (+47%). ct=4096 dispatch 128x4 224 -> 96x8 265 GB/s (+18%): the
#           large buckets want block 96 / warp 8, not block 128-144.
#   ct=256  combine 32x8 was near-worst (240us); 48x8 -> 138us (-42%).
#   combine 32x16 is peak at every large bucket (2048: 354, 4096: 360 GB/s).
# Small buckets (<=128) are launch/barrier-overhead-bound (~135-140us, GB/s
# meaningless) so their geometry is within measurement noise. Large buckets share
# one plan (96,8,32,16) -> collapsed into the None bucket. Verified end-to-end on
# DSv4-Pro 8k/1k c256: 32.8k tok/s (== baseline, tune is e2e-neutral since
# dispatch/combine are BW-bound and a small fraction of the decode/prefill step).
_MI355X_SCHED_FP4_DISP_BF16_COMB_T6 = (
    (128, 96, 4, 160, 16),
    (256, 160, 4, 48, 8),
    (None, 96, 8, 32, 16),
)
_MI355X_DEFAULT = dict(
    dispatch_block_num=128,
    combine_block_num=48,
    warp_num_per_block=16,
    combine_warp_num_per_block=16,
    schedule=_MI355X_SCHED_BF16_BASE,
)
_MI355X_TABLE = {
    (8, 7168, 8): {
        "bf16": (
            _MI355X_SCHED_BF16
            if HAS_BUFFER_OPS
            else _MI355X_SCHED_BF16_NO_CACHE_HINT
        ),
        "fp8": _MI355X_SCHED_FP8,
        "fp4": _MI355X_SCHED_FP4,
        "fp4_disp_bf16_comb": _MI355X_SCHED_FP4_DISP_BF16_COMB,
        # BF16 dispatch A/B, 2026-08-11: tok_off counting plus SC0|SC1 peer
        # stores wins at <=512 tokens; metadata bypass remains useful through
        # 256. Cached stores recover the large-message write-combining path.
        "_options": {
            "bf16": {
                "use_tok_off_total_recv": True,
                "replay_fast_path": True,
                "uncached_token_store_max_tokens": 512 if HAS_BUFFER_OPS else 0,
                "uncached_metadata_store_max_tokens": (
                    256 if HAS_BUFFER_OPS else 0
                ),
                # Workgroup-per-token load-once/store-many wins from 512
                # through the largest measured bucket. Keep small tokens and
                # >4096 on the work-centric kernel.
                "token_centric_min_tokens": 512 if HAS_BUFFER_OPS else 0,
                "token_centric_max_tokens": 4096 if HAS_BUFFER_OPS else 0,
                "token_centric_schedule": (
                    _MI355X_TOKEN_CENTRIC_BF16
                    if HAS_BUFFER_OPS
                    else None
                ),
                "token_centric_rotate_peer_order": HAS_BUFFER_OPS,
            }
        },
    },
    (8, 7168, 6): {
        "bf16": _MI355X_SCHED_BF16_BASE,
        "fp8": _MI355X_SCHED_FP8,
        "fp4": _MI355X_SCHED_FP4,
        "fp4_disp_bf16_comb": _MI355X_SCHED_FP4_DISP_BF16_COMB_T6,
    },
}

# ── gfx1250 (256 CU, wave32) — EP4, bf16, vec4 combine + load-first scheduling.
# dispatch warp grows to 32, block 192. combine is warp-sensitive at small tok
# (warp 2 <=128, warp 4 mid, warp 8 large); block grows 64->128. GUARDRAIL:
# block_num < CU (256); 192 safe ceiling (Phase-2 grid barrier needs co-residence).
_GFX1250_SCHED_BF16 = (
    (256, 128, 16, 64, 4),  # <=128:  latency-bound; combine warp 2 (fewest warps win)
    (512, 192, 32, 128, 4),  # <=512:  disp peak; comb 128/4
    (4096, 192, 32, 128, 8),  # <=4096: comb block 128
    (None, 192, 32, 192, 8),  # >4096:  comb warp 8
)

_GFX1250_DEFAULT = dict(
    dispatch_block_num=192,
    combine_block_num=192,
    warp_num_per_block=32,
    combine_warp_num_per_block=16,
    schedule=_GFX1250_SCHED_BF16,
)
# DeepSeek-V4-Pro shape (hidden 7168, topk 6, 384 experts). EP4 bf16, load-once/
# store-many + 4-way dispatch kernel (host-selected by load_once_threshold=4096).
# dispatch wants block=256 + warp 32 above 1024 tok (<=256 tok is latency-flat).
# combine also wants block=256 (=CU, blocks stay co-resident 1/CU), warp ramps 4->16;
# only tiny tok (<=256) wants a small block (256 starves it).
_GFX1250_SCHED_BF16_T6 = (
    (256, 128, 16, 64, 4),  # <=256:  disp latency-flat; comb small-block 64/4
    (
        512,
        192,
        32,
        128,
        4,
    ),  # <=512:  disp warp32 (192x32==256x32, min block); comb 128/4
    (1024, 192, 32, 128, 8),  # <=1024: comb 128/8
    (None, 256, 32, 256, 16),  # >1024: disp 256x32; comb 256x16
)
# EP8 cross-node (2 nodes x 4 GPUs over UALink), bf16, vec4 combine-gather.
# dispatch block 128, warp ramps 16->32. combine block 64 uniformly best, warp
# ramps 4->16. Fabric caps disp ~200 GB/s; geometry is world_size-independent so
# this also serves single-node EP8. topk6 reuses this schedule (topk-independent).
_GFX1250_SCHED_BF16_EP8 = (
    (256, 128, 16, 64, 4),  # <=256:  disp warp 16, comb 64/4 (latency-bound)
    (1024, 128, 32, 64, 8),  # <=1024: disp warp 32, comb 64/8
    (None, 128, 32, 64, 16),  # >1024 (peak)
)
# bf16-tuned (EP4 + EP8). fp8/fp4 fall back to the bf16 schedule until separately tuned.
_GFX1250_TABLE = {
    (4, 7168, 8): {"bf16": _GFX1250_SCHED_BF16},
    (4, 7168, 6): {"bf16": _GFX1250_SCHED_BF16_T6},  # DeepSeek-V4-Pro
    (8, 7168, 8): {"bf16": _GFX1250_SCHED_BF16_EP8},  # cross-node / single-node EP8
    # V4-Pro topk=6 EP8: topk-independent, reuse the topk=8 schedule.
    (8, 7168, 6): {"bf16": _GFX1250_SCHED_BF16_EP8},
}

_DEVICES = {
    "mi308x": (_MI308X_DEFAULT, _MI308X_TABLE),
    "mi325x": (_MI325X_DEFAULT, _MI325X_TABLE),
    "mi300x": (_MI300X_DEFAULT, _MI300X_TABLE),
    "mi355x": (_MI355X_DEFAULT, _MI355X_TABLE),
    "gfx1250": (_GFX1250_DEFAULT, _GFX1250_TABLE),
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
        schedule=None,
    )


def lookup(world_size, hidden_dim, topk, dtype="fp8"):
    """Return {dispatch_block_num, combine_block_num, warp_num_per_block,
    combine_warp_num_per_block, schedule, ...optional dispatch policy fields}
    for the current GPU, shape, and dtype.

    `dtype` (the token / dispatch dtype: "bf16" | "fp8" | "fp4") selects the
    per-dtype schedule, because dtype sets the communication volume (fp4 = 0.5 B,
    fp8 = 1 B, bf16 = 2 B per element) and thus the best block/warp. It falls back
    to the "fp8" schedule, then to the device default, when a dtype isn't tuned.

    `schedule` (or None) is a per-token-count launch plan: a tuple of
    (max_tok_inclusive | None, disp_block, disp_warp, comb_block, comb_warp)
    buckets, ascending; the op precompiles the distinct (block, warp) variants
    and picks a bucket at runtime from cur_rank_num_token. dispatch_block_num /
    warp_num_per_block / combine_block_num / combine_warp_num_per_block are the
    single-shot fallback used when schedule is None."""
    key = _device_key()
    if key is None or key not in _DEVICES:
        return _cu_scaled_default()
    dev_default, dev_table = _DEVICES[key]
    base = dict(dev_default) if dev_default is not None else _cu_scaled_default()
    base.setdefault("schedule", None)
    entry = dev_table.get((world_size, hidden_dim, topk))
    if entry:
        base["schedule"] = entry.get(dtype) or entry.get("fp8") or base["schedule"]
        options = entry.get("_options", {}).get(dtype, {})
        base.update(options)
    return base
