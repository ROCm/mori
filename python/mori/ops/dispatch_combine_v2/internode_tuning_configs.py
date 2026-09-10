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

"""Per-device launch geometry for the v2 CCO internode dispatch/combine kernels.

One table for BOTH internode families: "v2" and "v2_ll" are compiled from the
same request, so a row's geometry is whichever family the launch resolves to.
The coarsest row also spans the token counts above ``internode_ll_max_tokens``,
where "auto" stops picking the LL family -- there it is "v2" geometry only.

The internode kernels reach their grid a different way than the intranode ones.
An intranode kernel is compiled once per (block, warp) the schedule can name and
picks among those at launch; an internode *pass sequence* is compiled per
geometry, because ``rdma_block_num`` splits the grid between the RDMA blocks and
the intra-node ones and the kernel branches on it. So the host resolves a bucket
here from the live ``num_tokens`` and launches the plan set built for it --
``EpDispatchCombineOpHip._internode_geometry_buckets`` walks this whole table
at build time so that resolution can never trigger a compile.

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

# MI308X (gfx942, 80 CU) -- EP16, hidden 6144, topk 8. Tuned fp8-dispatch +
# bf16-combine on a 2-node rig, block_num <= 80.
#
# Invariant: dispatch and combine are COUPLED -- same graph replay, same QPs, one
# shared arena -- so these rows are the best PAIR, not the per-phase argmins. Do
# not re-tune one phase in isolation; the per-phase winners a sweep prints do not
# reproduce once the two phases run at different geometries. The 4-token row is
# the one place a low dispatch rdma_block_num is safe: the extra cost it used to
# impose on the paired combine does not occur on the v2 CCO path. The 8-token row
# still holds rdma at 32 for that reason.
#
# Re-tuning protocol -- a sweep on this path is easy to get wrong, and all three
# guards are required. Only the first is automated.
#   1. Judge candidates by a PAIRED comparison against a FIXED incumbent, never
#      greedily: a "winner becomes the next incumbent" chain returns a different
#      winner on every repeat of the same sweep at this noise level. This is what
#      `--cmd tuning` does; see _tune in
#      tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py.
#   2. MANUAL. Picking the best of a large candidate set is a multiple-comparison
#      problem, so re-run the sweep winner with `--tuning-candidate` three
#      separate times and require it to beat the shipped row's median every time.
#      Sweep-only winners routinely fail this.
#   3. MANUAL. Finally A/B the resulting TABLE in a plain interleaved bench. The tuner
#      cannot substitute for this: it takes a median over paired passes, which
#      discards the rare multi-x spike a bench mean sees, and it holds two ops
#      (two symmetric windows) alive at once, which makes BOTH arms spike and so
#      masks a candidate's own spike. Geometries that leave the intra-node half
#      too few blocks are bimodal and have been caught only at this stage.
#
# Ranks must be NUMA-bound before any of this: unbound CPU placement adds a large
# random term that swamps every effect measured here.
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


def _device_key():
    """Device table key for the current GPU (PCI DID, then arch), or None."""
    return gpu_utils.detect_model()


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
