"""Per-device, per-shape block/warp tuning for the cco-LSA dispatch/combine
kernels.

Geometry is picked per device (MI308X / MI300X / MI355X), then by
(world_size, hidden_dim, topk). Devices are told apart by PCI device id (MI300X
and MI308X are both gfx942, differing only in CU count), falling back to arch for
gfx950 — all from KFD sysfs, no torch/HIP dependency. block_num must stay
<= CU count; re-tune per GPU.
"""

import functools
import glob


# ── MI308X (gfx942, 80 CU) — measured 2026-07-08, EP8, from a block x warp sweep.
# Best (block, warp) is token-count dependent, so this is a per-token SCHEDULE of
# (max_tok_inclusive | None, disp_block, disp_warp, comb_block, comb_warp) buckets;
# the op precompiles the distinct (block, warp) variants and picks by token count.
_MI308X_SCHEDULE = (
    (256,  64, 8,  64, 4),
    (2048, 64, 16, 64, 4),
    (None, 64, 16, 80, 4),
)
# Single-shot fallback (schedule ignored) = peak-optimal.
_MI308X_DEFAULT = dict(dispatch_block_num=64, combine_block_num=80,
                       warp_num_per_block=16, combine_warp_num_per_block=4,
                       schedule=_MI308X_SCHEDULE)
# Per-shape -> per-dtype schedules. These were tuned as fp8-dispatch + bf16-combine,
# so they live under "fp8" (the default / fallback for any untuned dtype).
# hidden 4096 / 2048 reuse the 7168 schedule until separately tuned.
_MI308X_TABLE = {
    (8, 7168, 8): {"fp8": _MI308X_SCHEDULE},
    (8, 4096, 8): {"fp8": _MI308X_SCHEDULE},
    (8, 2048, 8): {"fp8": _MI308X_SCHEDULE},
}

# ── MI325X (gfx942, 304 CU, DID 0x74a5) — measured 2026-07-08, EP8, full 2D block x
# warp sweep (graph, ITERS>=300, tok 8..8192), selected by min LATENCY (us). Key
# lessons vs MI308X: (a) dispatch always wants warp 8 (not 16), block grows with tok
# (64->152->228->304); (b) combine is latency-bound at small/mid tok — on this big
# 304-CU part a SMALL block (64) + warp 4 wins (the 0.5*CU block's grid-barrier cost
# dominates), only bandwidth-bound large tok (>1024) want b~0.5*CU (152) + warp 2;
# at the tiniest tok (8) combine warp 2 edges warp 4. b64 combine beats MI308X at
# every small size. Measured combine us: 8=20 64=38 128=53 256=83 512=145 1024=267
# 2048=497 4096=961 8192=1839; dispatch us: 8=18 64=30 128=47 256=78 512=138 1024=256
# 2048=490 4096=960 8192=1881.
_MI325X_SCHEDULE = (
    (8,     64, 8,   64, 2),   # <=8 tok: tiny, combine warp 2
    (64,    64, 8,   64, 4),   # <=64:   small block both
    (1024, 152, 8,   64, 4),   # <=1024: disp 0.5*CU, comb small-block/warp4 (latency)
    (4096, 228, 8,  152, 2),   # <=4096: comb 0.5*CU/warp2 (bandwidth)
    (None, 304, 8,  152, 2),   # >4096 (peak)
)
_MI325X_DEFAULT = dict(dispatch_block_num=304, combine_block_num=152,
                       warp_num_per_block=8, combine_warp_num_per_block=2,
                       schedule=_MI325X_SCHEDULE)
# hidden 4096 / 2048 reuse the 7168 schedule until separately tuned.
_MI325X_TABLE = {
    (8, 7168, 8): {"fp8": _MI325X_SCHEDULE},
    (8, 4096, 8): {"fp8": _MI325X_SCHEDULE},
    (8, 2048, 8): {"fp8": _MI325X_SCHEDULE},
}

# ── MI300X (gfx942, 304 CU) — TODO: re-tune. Falls back to CU-scaled default. ──
_MI300X_DEFAULT = None   # None => derive from CU count (see _cu_scaled_default)
_MI300X_TABLE = {}

# ── MI355X (gfx950, 256 CU) — measured 2026-07-08, EP8, fine block x warp sweep.
# dispatch: warp 8 throughout (warp 16 worse); block 128 (<=2048) / 192 (>=4096).
# combine : warp 4; block 128 (>=256), 48 for <=64 tok (small blocks win at tiny
#           batches; 256 blocks are worse — unlike MI308X's ~1 block/CU).
# fp8-dispatch + bf16-combine schedule (also the fallback for bf16/untuned dtypes):
_MI355X_SCHED_FP8 = (
    (64,   128, 8, 48, 8),
    (2048, 128, 8, 128, 4),
    (None, 192, 8, 128, 4),
)
# fp4 dispatch + fp4 combine (data_type=fp4 does both) — measured 2026-07-09 via a
# 2D block x warp sweep on MI355X (DTYPE=fp4). fp4 = 0.5 B/elem, so both phases are
# less bandwidth-bound than fp8: dispatch block 128 throughout (warp 4 <=2048,
# 8 >=4096); combine small block at small tok (64/8 <=256), 128/4 mid, 128/8 large.
_MI355X_SCHED_FP4 = (
    (256,  128, 4, 64,  8),
    (2048, 128, 4, 128, 4),
    (None, 128, 8, 128, 8),
)
_MI355X_DEFAULT = dict(dispatch_block_num=192, combine_block_num=128,
                       warp_num_per_block=8, combine_warp_num_per_block=4,
                       schedule=_MI355X_SCHED_FP8)
_MI355X_TABLE = {
    (8, 7168, 8): {"fp8": _MI355X_SCHED_FP8, "fp4": _MI355X_SCHED_FP4},
}

_DEVICES = {
    "mi308x": (_MI308X_DEFAULT, _MI308X_TABLE),
    "mi325x": (_MI325X_DEFAULT, _MI325X_TABLE),
    "mi300x": (_MI300X_DEFAULT, _MI300X_TABLE),
    "mi355x": (_MI355X_DEFAULT, _MI355X_TABLE),
}


# PCI device IDs (KFD `device_id`) → device table key. gfx942 family only; the
# gfx950 parts (MI350/MI355X) are matched by arch below since their DIDs vary.
_DID_TO_KEY = {
    0x74A1: "mi300x",   # MI300X
    0x74A5: "mi325x",   # MI325X (304-CU gfx942; tuned 2026-07-08)
    0x74A2: "mi308x",   # MI308X (80 CU)
}


@functools.lru_cache(maxsize=1)
def _topology():
    """(cu_count, gfx_target_version, device_id) of the first GPU node from KFD
    sysfs. torch/HIP-free. gfx_target_version is e.g. 90402 (gfx942), 90500
    (gfx950); device_id is the PCI DID (e.g. 0x74a2 = MI308X). Homogeneous host
    assumed. Returns (0, 0, 0) if sysfs is unavailable (no KFD mounted)."""
    for props in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")):
        try:
            vals = {}
            with open(props) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 2:
                        vals[parts[0]] = int(parts[1])
            simd = vals.get("simd_count", 0)
            if simd <= 0:               # CPU / non-GPU node
                continue
            spc = vals.get("simd_per_cu", 0) or 1
            return simd // spc, vals.get("gfx_target_version", 0), vals.get("device_id", 0)
        except Exception:
            continue
    return 0, 0, 0


def _cu_count():
    return _topology()[0]


def _device_key():
    """Map the current GPU to a device table key: exact PCI DID first, then arch
    for gfx950. Returns None if unknown (caller uses a CU-scaled default)."""
    _, gfx, did = _topology()
    key = _DID_TO_KEY.get(did)
    if key is not None:
        return key
    if gfx == 90500:                    # gfx950 (MI350 / MI355X), DID varies
        return "mi355x"
    return None


def _cu_scaled_default():
    """Untuned fallback: ~1 block/CU for combine (<= CU count), many warps for
    dispatch, few warps for combine, no per-token schedule (single-shot). Used
    for devices without a measured table."""
    cu = _cu_count() or 80
    return dict(dispatch_block_num=cu, combine_block_num=cu,
                warp_num_per_block=16, combine_warp_num_per_block=4, schedule=None)


def lookup(world_size, hidden_dim, topk, dtype="fp8"):
    """Return {dispatch_block_num, combine_block_num, warp_num_per_block,
    combine_warp_num_per_block, schedule} for the current GPU, shape, and dtype.

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
    return base
