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

Different kernels, different optima, so never borrow FlyDSL's schedule. This module
does not import ``tuning_configs``; it shares only ``mori.ops.utils`` for device
detection.

Dispatch and combine get INDEPENDENT tables: dispatch depends on the dtype it
transports, combine does not (it only ever reduces the bf16/fp32 staging region).
Both are keyed by (world_size, hidden_dim, topk, experts_per_rank); an
``experts_per_rank`` of None is a wildcard, used only where a sweep showed the expert
count does not move the optimum. A bucket is (max_tok_inclusive | None, block, warp),
ascending, and ``lookup`` merges the two into the op's
(max_tok, disp_block, disp_warp, comb_block, comb_warp) schedule.

An unswept shape returns schedule=None and the single-shot default below. Add one by
sweeping with ``bench_ep.py``. fp32 combine is untuned and takes the bf16 buckets.

dtype keys are whatever ``EpDispatchCombineConfig.dtype_str`` produces, hence
"fp4_disp_bf16_comb": hip rejects an fp4 combine outright, so an fp4 dispatch here is
always paired with bf16 -- which is the configuration the sweep measured.
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
    """Fallback for an unswept shape, mirroring the C++ MakeEpCfg arch default so a
    bare C++ caller and this path agree. Still not a tuned answer -- it is one shape
    for every device, token count and dtype -- but it is no longer a known-worse one:
    combine moved 80x8 -> 64x8 in both places once the sweep showed 64 wins at every
    size on mi355x."""
    return dict(
        dispatch_block_num=64,
        combine_block_num=64,
        warp_num_per_block=16,
        combine_warp_num_per_block=8,
        schedule=None,
    )


# ---------------------------------------------------------------------------
# The tables. bucket = (max_tok_inclusive | None, block, warp), ascending.
# key = (world_size, hidden_dim, topk, experts_per_rank); experts_per_rank None
# means "measured not to matter here", and an exact key wins over the wildcard.
# ---------------------------------------------------------------------------

# 4x gfx1250 EP8-equivalent (EP4) hidden 7168, 2026-08-11, same grid. Entries are the
# smallest geometry within ~3% of the best -- fewer blocks means fewer CUs held, which
# matters when this overlaps an expert GEMM. Those ties are policy, not measurement: a
# single bench point can be off 20%. The bucket EDGES come from 10-40% effects.
#
# topk moves the edges here (it sets _tpi), the expert count does not (64 vs 96
# agreed within ~2%), and the dtype only does for fp4 at topk 6.
_DISPATCH_TABLE: dict = {
    # MI355X/gfx950 EP8 hidden 7168, 2026-08-11. No TDM here: the portable dispatch
    # reserves no LDS and copies with plain vectors, so bf16/fp8 are bandwidth-bound and
    # the grid barely registers. Cost of the smallest geometry (64x8) vs the best of
    # {64x8, 64x16, 128x8, 128x16, 256x8} over 64..16384 tokens: bf16 0-2%, fp8 1-2%
    # from ct>=512, fp4 6-69% from ct>=128. Only fp4 cares -- a quarter of the payload
    # tips it out of bandwidth-bound -- and it wants 128x8 flat.
    #
    # topk 6 and 8 measured identical. Two entries rather than a topk wildcard: _tpi
    # makes topk matter in principle, and it does on gfx1250, so an unmeasured topk
    # should get the default instead of inheriting one that happened to agree.
    "mi355x": {
        (8, 7168, 8, None): {None: ((None, 64, 8),), "fp4_disp_bf16_comb": ((None, 128, 8),)},
        (8, 7168, 6, None): {None: ((None, 64, 8),), "fp4_disp_bf16_comb": ((None, 128, 8),)},
    },
    "gfx1250": {
        # topk 8 (256 experts at EP4). All three dtypes agree here.
        #   ct     64x8   64x16  128x16 256x16      (bf16 / fp8 / fp4)
        #   512    75.0    92.7   92.5   92.4  |  72.7 73.9 73.2 73.6 | 71.2 72.8 73.5 73.0
        #   2048   97.5   108.8  113.4  105.0  |  80.2 77.8 77.8 78.5 | 76.5 75.0 74.7 74.5
        #   4096  163.4   158.0  161.1  156.4  | 127.3 99.0 99.3 98.4 |121.4 80.5 81.0 80.7
        #   16384 560.3   552.4  516.8  507.4  | 428.6 307. 278. 282.9|418.8 243. 174.6 172.1
        (4, 7168, 8, None): {
            None: ((2048, 64, 8), (4096, 64, 16), (None, 128, 16)),
        },
        # topk 6 (384 experts at EP4). The edges move in: 64x8 stops paying at 512.
        #   ct     64x8   64x16  128x16 256x16      (bf16 / fp8 / fp4)
        #   512    54.4    55.3   55.1   55.3  |  50.4 47.9 48.1  --  | 48.3 46.6 46.6 46.3
        #   1024   75.8    68.8   68.1   69.4  |  68.0 55.2 55.6 54.7 | 66.6 50.6 50.8 51.3
        #   2048  101.3   103.5  102.3  101.7  |  86.5 78.5 76.6 76.2 | 85.8 67.9 64.2 64.5
        #   16384 551.7   518.5  478.5  470.7  | 423.8 303. 264.9 266.| 414.9 233.7 172.6 166.4
        (4, 7168, 6, None): {
            None: ((512, 64, 8), (4096, 64, 16), (None, 128, 16)),
            "fp4_disp_bf16_comb": ((512, 64, 8), (1024, 64, 16), (None, 128, 16)),
        },
    },
}

# No dtype axis: combine reduces the bf16 staging region whatever dispatch carried,
# and three runs per shape (one per dispatch dtype) agreed to 1-5%.
#
# mi355x 64x8 wins at every size and both topk, 2-3% over the 80x8 v1 carried and with
# 16 fewer blocks; MakeEpCfg and _hip_default() were moved to 64 to match, so the
# untuned fallback is no longer a geometry the tables already knew was worse. It is a
# real optimum, not just the smallest tried: 32x8 costs 37% at ct=4096. On gfx1250,
# 128x4 never wins and 256x8 collapses 2x at 16384.
_COMBINE_TABLE: dict = {
    # MI355X / gfx950 EP8: 64x8 wins at every token count and both topk, against
    # 32x8 / 48x8 / 64x16 / 80x8 (us at ct=4096, topk 8: 991.6 / 750.5 / 724.3 / 738.5 /
    # 745.2). 80x8 was v1's value, measured once on a narrower grid; both MakeEpCfg and
    # _hip_default() now say 64 as well. Fewer blocks than 64 is not free here: 32x8
    # costs 37% at ct=4096, so this is a real optimum, not just the smallest thing tried.
    "mi355x": {
        (8, 7168, 8, None): ((None, 64, 8),),
        (8, 7168, 6, None): ((None, 64, 8),),
    },
    "gfx1250": {
        (4, 7168, 8, None): ((512, 64, 8), (None, 128, 8)),
        (4, 7168, 6, None): ((512, 64, 8), (None, 128, 8)),
    },
}

# THE ROUND RULE outranks every shape choice above: _tpi = warpSize/topk tokens are
# consumed per warp-iteration, so one round covers block*warp*_tpi tokens and coming up
# short costs more than any geometry difference (gfx1250 fp4 at ct=4096: 64x8 121.4us
# against 64x16 80.5). A new topk moves _tpi and every edge with it.
#
# Health-check the box before re-tuning: a degraded machine once produced a
# self-consistent, entirely wrong answer. Canary: gfx1250 bf16 dispatch at ct=4096/64x16
# is ~157us healthy and ~172 degraded, while combine sits at ~145 either way.

def _bucket_key(table, world_size, hidden_dim, topk, experts_per_rank):
    """Exact expert count first, then the "any expert count" wildcard."""
    for epr in (experts_per_rank, None):
        entry = table.get((world_size, hidden_dim, topk, epr))
        if entry is not None:
            return entry
    return None


def _merge(disp, comb):
    """Interleave two independent bucket lists into the op's one schedule.

    The edges need not line up: the merged list breaks at the union of both, and each
    half keeps whatever it asked for on either side of the other's edge.
    """
    edges = sorted(
        {b[0] for b in disp if b[0] is not None} | {b[0] for b in comb if b[0] is not None}
    ) + [None]

    def pick(buckets, edge):
        for mx, blk, wrp in buckets:
            if mx is None or (edge is not None and edge <= mx):
                return blk, wrp
        return buckets[-1][1], buckets[-1][2]

    return tuple(
        (edge,) + pick(disp, edge) + pick(comb, edge) for edge in edges
    )


def lookup(world_size, hidden_dim, topk, dtype="bf16", experts_per_rank=None) -> dict:
    """HIP geometry for this device/shape/dtype, composed from HIP's own two tables.

    An unswept shape gets the HIP single-shot default (schedule=None). A swept one
    gets a per-token-count schedule built from the dispatch and combine tables
    independently, so either half can be re-tuned without touching the other.
    """
    base = _hip_default()
    dev = _device_key()
    disp = _bucket_key(_DISPATCH_TABLE.get(dev, {}), world_size, hidden_dim, topk,
                       experts_per_rank)
    comb = _bucket_key(_COMBINE_TABLE.get(dev, {}), world_size, hidden_dim, topk,
                       experts_per_rank)
    if disp is None or comb is None:
        return base  # half a schedule is not a schedule
    # None is the "every dtype measured the same" key; an exact dtype overrides it.
    disp = disp.get(dtype) or disp.get(None)
    if disp is None:
        return base
    base["schedule"] = _merge(disp, comb)
    return base
