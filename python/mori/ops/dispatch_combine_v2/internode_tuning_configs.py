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

The internode kernels reach their grid a different way than the intranode ones:
the launch redirect (see the v2 internode test) inherits geometry from
``dispatch_combine.py`` -- the shmem resolve -- rather than a per-token schedule
the kernel selects at runtime. So the *host* picks a per-token bucket here and
pins the resulting block/rdma/warp on ``op.dispatch`` / ``op.combine``. That is
why this table, unlike ``tuning_configs.py`` (flydsl intranode) and
``hip_tuning_configs.py``, carries an ``rdma_block_num`` per phase and is looked
up with the live ``num_tokens`` rather than compiled into a schedule.

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
# bf16-combine on skyriver07+04 (2-node). dispatch is fabric-bound (low rdma at
# small tok, block<=80); combine is xGMI-bandwidth-bound (small block at small
# tok, block 80 / rdma 40 / warp 8 at mid tok). All block_num <= 80.
_MI308X_EP16_H6144 = (
    # max_tok, disp_block, disp_rdma, disp_warp, comb_block, comb_rdma, comb_warp
    (4, 32, 16, 6, 32, 21, 6),
    (8, 64, 16, 4, 64, 32, 4),
    (16, 80, 48, 8, 80, 40, 8),
    (None, 80, 48, 8, 80, 40, 8),
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
    return {"dispatch": (db, dr, dw), "combine": (cb, cr, cw)}
