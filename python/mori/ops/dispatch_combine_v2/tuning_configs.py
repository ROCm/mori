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
# hidden 4096 / 2048 reuse the 7168 schedule until separately tuned.
_MI308X_TABLE = {
    (8, 7168, 8): dict(schedule=_MI308X_SCHEDULE),
    (8, 4096, 8): dict(schedule=_MI308X_SCHEDULE),
    (8, 2048, 8): dict(schedule=_MI308X_SCHEDULE),
}

# ── MI300X (gfx942, 304 CU) — TODO: re-tune. Falls back to CU-scaled default. ──
_MI300X_DEFAULT = None   # None => derive from CU count (see _cu_scaled_default)
_MI300X_TABLE = {}

# ── MI355X (gfx950) — TODO: re-tune. Falls back to CU-scaled default. ──
_MI355X_DEFAULT = None
_MI355X_TABLE = {}

_DEVICES = {
    "mi308x": (_MI308X_DEFAULT, _MI308X_TABLE),
    "mi300x": (_MI300X_DEFAULT, _MI300X_TABLE),
    "mi355x": (_MI355X_DEFAULT, _MI355X_TABLE),
}


# PCI device IDs (KFD `device_id`) → device table key. gfx942 family only; the
# gfx950 parts (MI350/MI355X) are matched by arch below since their DIDs vary.
_DID_TO_KEY = {
    0x74A1: "mi300x",   # MI300X
    0x74A5: "mi300x",   # MI325X (same 304-CU gfx942 class; share until re-tuned)
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


def lookup(world_size, hidden_dim, topk):
    """Return {dispatch_block_num, combine_block_num, warp_num_per_block,
    combine_warp_num_per_block, schedule} for the current GPU and shape.

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
    base.update(dev_table.get((world_size, hidden_dim, topk), {}))
    return base
