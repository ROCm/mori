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

"""Per-device launch geometry for the v2 CCO internode (InterNodeV1LL)
dispatch/combine kernels.

The internode kernels reach their grid a different way than the intranode ones.
An intranode kernel is compiled once per (block, warp) the schedule can name and
picks among those at launch; an internode *pass sequence* is compiled per
geometry, because ``rdma_block_num`` splits the grid between the RDMA blocks and
the intra-node ones and the kernel branches on it. So the host resolves a bucket
here from the live ``num_tokens`` and launches the plan set built for it --
``HipBackend._internode_geometry_buckets`` walks this whole table at build time
so that resolution can never trigger a compile.

That is also why this table, unlike ``tuning_configs.py`` (flydsl intranode) and
``hip_tuning_configs.py``, carries an ``rdma_block_num`` per phase: dispatch and
combine are tuned to different values at the same token count over one shared
arena, which a single geometry per bucket cannot express.

Devices are told apart the same way as ``tuning_configs.py`` -- PCI DID first
(MI300X and MI308X are both gfx942, differing only in CU count), then arch. The
one hard invariant is the same one that module states: **block_num must stay
<= CU count**; a grid wider than the CUs runs the tail in a second wave, which a
latency-bound small-token kernel cannot afford. ``lookup`` clamps to it.

Buckets are ``(max_tok_inclusive | None, disp_block, disp_rdma, disp_warp,
comb_block, comb_rdma, comb_warp)``, ascending; the first whose ``max_tok``
covers ``num_tokens`` wins. Filed under the dispatch/token dtype ("fp8" here
means fp8-dispatch + bf16-combine, the pairing the internode bench measures);
untuned dtypes fall back to "fp8".
"""

from mori.ops import utils as gpu_utils

# ── MI308X (gfx942, 80 CU) — EP16, hidden 6144, topk 8. Tuned fp8-dispatch +
# bf16-combine on skyriver07+04 (2-node), block_num <= 80.
#
# dispatch and combine are coupled, not independent: dispatch and combine share
# the same CUDA-graph replay and the same QPs, and a dispatch with too few
# rdma_block_num leaves the combine that follows it markedly slower (~+18us at
# 4/8 tokens for rdma 16 vs 32). So the small-token dispatch is NOT tuned for
# dispatch latency alone -- it holds rdma at 32 to keep the paired combine fast,
# which is the lower total. combine wants a small block at small tok (32/64) and
# block 80 / rdma 40 at mid tok.
#
# Re-tune (2026-09-04, full-scope sweep x3 + A/B validation, same 2-node rig):
# the 4/8/32 rows were confirmed at or better than anything the sweep found (the
# independent per-phase argmins the sweep prints do NOT reproduce once dispatch
# and combine run at different geometries -- the coupling above -- so tok8 stays
# on the current 32/21/6 combine, which A/B-beat the sweep's 64/16/8). Only tok16
# moved: a single shared 80/rdma40/warp4 geometry for both phases beat the old
# 80/48/8 + 80/40/8 by ~4us total (disp 41.3 vs 43.6, comb 51.1 vs 53.1),
# reproducible across two A/B batches.
#
# Re-tune (2026-09-08, v2 CCO path, `--cmd tuning` in the v2 harness): only the
# 4-token DISPATCH moved, 64/32/8 -> 32/16/4. Everything else in this table was
# re-swept and held: 4-token combine, both phases at 8, 16 and 32 tokens -- 0
# reproducible wins over 3 repeats each (16 and 32 gave 0 wins in all 6 runs).
#
# That row is the one change because it is the only one that reproduced. Three
# independent 29-candidate sweeps ranked 32/16/4 first every time (-2.4, -2.6,
# -2.8us paired against the fixed incumbent), and eight 51-rep head-to-heads gave
# a median of -2.3us with 6 of 8 clearing the margin. The per-phase split is
# consistent across all eight: dispatch 36.9-38.4us against 39.2-40.9us, combine
# 46.2-46.9 against 45.8-46.6 -- so the gain is ~6% of dispatch and the combine
# after it is unchanged.
#
# That last part matters, because the coupling note above predicts the opposite:
# it records rdma 16 costing ~+18us on the paired combine at 4/8 tokens, which is
# why this row held rdma at 32. On the v2 CCO path that penalty does not appear
# (46.5 vs 46.5 in the validation runs), so the reason for keeping rdma high at 4
# tokens has gone with it. The note is left standing for the 8-token row, whose
# sweep found nothing better and which still carries rdma 32.
#
# Re-tune (2026-09-09) after the CCO NUMA-binding fix. EVERY number above this
# line was tuned with the ranks unbound, i.e. every A/B in it raced a +-40us
# random term from CPU placement, so the table had to be re-derived rather than
# trusted. ONE row moved: 32-token combine 80/40/8 -> 64/48/6.
#
# Method, FOUR stages. The fourth is not optional and I learned that the hard way:
#   1. Full sweep, 165 candidates (the rdma grid was widened from three points to
#      eighths -- the old one could not even reach the shipped 32-token rdma=48),
#      three repeats per (token, phase).
#   2. Keep only candidates that won in at least two of the three repeats. At 4
#      tokens the three sweeps returned 15 winners and NONE repeated, which is
#      the whole argument for this stage.
#   3. Head-to-head against the shipped row, 21 paired reps, three times.
#
# Stage 3 is not a formality. 16-token combine (64,40,6) won all THREE sweeps and
# then lost all three head-to-heads -- picking a winner out of 165 candidates is
# a multiple-comparison problem and the sweep alone cannot tell a real effect
# from the best of 165 draws. The bar applied here is: the tuned phase's median
# must be better in all three head-to-heads.
#   4. A plain interleaved bench A/B of the resulting TABLE. Stage 3 runs two ops
#      at once -- incumbent and candidate each holding a symmetric window, run
#      alternately -- which is not the shipping condition, so its verdict does
#      not transfer on its own.
#
# Stage 4 rejected 8-token dispatch (32,12,6), which had passed stage 3 on both
# the median (39.0/39.2/40.2 against 40.9/41.0/41.3) and the worst. In a plain
# bench it read 118.1/118.8/115.9 against the shipped row's 92.0/94.3, with one
# dispatch worst of 374us -- best-in-class once and much worse three times. That
# geometry leaves only 20 of 32 blocks for the intra-node half and is bimodal;
# 8-token dispatch is therefore UNCHANGED.
#
# 32-token combine (64,48,6) passed stage 4: interleaved against the shipped row,
# three pairs, 113.6/114.6/114.3 against 116.1/116.1/118.3, with combine itself
# 65.0/64.9/64.9 against 66.7/66.8/67.3. That is the one row this re-tune moved.
# 4- and 32-token dispatch had no candidate reproduce at all and are unchanged.
#
# Two candidates cleared the tail bar but not the median one and are NOT applied,
# recorded so they are not re-derived: 4-token combine (64,16,8) and 8-token
# combine (80,50,4). Both tie on the median and cut the worst of the paired reps
# by 15-25us. They are a real trade, not noise; they want their own decision.
#
# Methodology, because a sweep on this path is easy to get wrong: candidates are
# judged by a PAIRED comparison against a FIXED incumbent, not by a chain. v1's
# greedy shape (a winner becomes the incumbent) is unusable at this noise level
# -- five repeats of one sweep returned five different winners. See _tune in
# tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py.
_MI308X_EP16_H6144 = (
    # max_tok, disp_block, disp_rdma, disp_warp, comb_block, comb_rdma, comb_warp
    (4, 32, 16, 4, 32, 21, 6),
    (8, 64, 32, 8, 32, 21, 6),
    (16, 80, 40, 4, 80, 40, 4),
    (None, 80, 48, 8, 64, 48, 6),
)

# (device_key, world_size, hidden_dim, topk) -> {dtype: schedule}
_TABLE = {
    ("mi308x", 16, 6144, 8): {"fp8": _MI308X_EP16_H6144},
}

# Same die / CU count as a tuned sibling, reuse its table.
_MODEL_ALIAS = {"mi350x": "mi355x"}


def _device_key():
    """Device table key for the current GPU (PCI DID, then arch), or None."""
    model = gpu_utils.detect_model()
    if model is not None:
        return _MODEL_ALIAS.get(model, model)
    return None


def lookup(world_size, hidden_dim, topk, num_tokens, dtype="fp8"):
    """Per-phase internode geometry for the current GPU/shape/token-count.

    Returns ``{"dispatch": (block, rdma, warp), "combine": (block, rdma, warp)}``
    with ``block <= CU count``, or ``None`` when this GPU/shape is not tuned (the
    caller keeps whatever geometry it already had).
    """
    key = _device_key()
    if key is None:
        return None
    entry = _TABLE.get((key, world_size, hidden_dim, topk))
    if not entry:
        return None
    sched = entry.get(dtype) or entry.get("fp8")
    if not sched:
        return None

    bucket = sched[-1]
    for b in sched:
        if b[0] is None or num_tokens <= b[0]:
            bucket = b
            break
    _, db, dr, dw, cb, cr, cw = bucket

    cu = gpu_utils.cu_count() or 80
    db, cb = min(db, cu), min(cb, cu)  # never over-subscribe the CUs
    # rdma_block_num partitions the SAME grid: blocks below it talk to the
    # network, the rest do the intra-node half. Clamping block without clamping
    # rdma can leave rdma >= block, which is not slow but wrong -- no block is
    # left for the intra-node side and the dispatch barrier never completes.
    dr, cr = min(dr, max(1, db - 1)), min(cr, max(1, cb - 1))
    return {"dispatch": (db, dr, dw), "combine": (cb, cr, cw)}
