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

Dispatch dtype keys name only what dispatch transports: "bf16", "fp8", "fp4"
(``EpDispatchCombineConfig.dispatch_dtype_str``). The combine side has no dtype here --
it always reduces bf16 rows, and whether those go out as mxfp4 is ``quant_type``'s
table. "fp4_disp_bf16_comb" is FlyDSL's key for the same fp4 dispatch (its table does
name both halves); ``lookup`` still reads it as "fp4" for callers that pass
``dtype_str``.
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
    """Fallback for an unswept shape, mirroring the C++ MakeEpCfg default so a bare
    C++ caller and this path agree. One shape for every device, token count and
    dtype -- not a tuned answer."""
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

# Entries are the smallest geometry within ~3% of the best: fewer blocks holds fewer
# CUs, which matters when this overlaps an expert GEMM. Those ties are policy, not
# measurement -- a single bench point can be off 20% -- while the bucket EDGES come
# from 10-40% effects.
_DISPATCH_TABLE: dict = {
    # MI355X/gfx950 EP8 hidden 7168, 2026-08-11. No TDM here: the portable dispatch
    # reserves no LDS and copies with plain vectors, so bf16/fp8 stay bandwidth-bound
    # and the grid barely registers. Cost of 64x8 against the best of {64x8, 64x16,
    # 128x8, 128x16, 256x8} over 64..16384 tokens: bf16 0-2%, fp8 1-2% from ct>=512,
    # fp4 6-69% from ct>=128. Only fp4 cares -- a quarter of the payload tips it out
    # of bandwidth-bound -- and it wants 128x8 flat.
    #
    # topk 6 and 8 measured identical; listed twice rather than wildcarded, so an
    # unmeasured topk gets the default instead of inheriting an agreement by accident.
    "mi355x": {
        (8, 7168, 8, None): {
            None: ((None, 64, 8),),
            "fp4": ((None, 128, 8),),
        },
        (8, 7168, 6, None): {
            None: ((None, 64, 8),),
            "fp4": ((None, 128, 8),),
        },
    },
    # 4x gfx1250 at EP4, hidden 7168, 2026-08-11. topk moves the edges (it sets _tpi),
    # the expert count does not (64 vs 96 agreed within ~2%), and the dtype only does
    # for fp4 at topk 6.
    "gfx1250": {
        # topk 8 (256 experts at EP4). All three dtypes agree here.
        #   ct     64x8   64x16  128x16 256x16      (bf16 / fp8 / fp4)
        #   512    75.0    92.7   92.5   92.4  |  72.7 73.9 73.2 73.6 | 71.2 72.8 73.5 73.0
        #   2048   97.5   108.8  113.4  105.0  |  80.2 77.8 77.8 78.5 | 76.5 75.0 74.7 74.5
        #   4096  163.4   158.0  161.1  156.4  | 127.3 99.0 99.3 98.4 |121.4 80.5 81.0 80.7
        #   16384 560.3   552.4  516.8  507.4  | 428.6 307. 278. 282.9|418.8 243. 174.6 172.1
        (4, 7168, 8, None): {
            None: ((512, 64, 8), (4096, 256, 8), (None, 256, 16)),
        },
        # topk 9 = 8 routed + 1 shared (what ATOM dispatches: kernel name k9).
        # The engine selects this key, not topk 8; block 64 default starves the
        # 256-CU GPU at the ~3456-recv operating point. Block 256 saturates it.
        # fp4, 14 geometries swept (us, random / grouped routing; every pick within 3% of
        # the best under both):
        #   ct     64x8        256x8       256x24      256x32      128x16
        #   512    28.9/21.4   28.9/21.7   35.1/33.3   35.1/33.5   35.8/34.5
        #   1536   57.2/47.6   37.5/27.7   42.5/37.4   43.4/38.4   39.7/30.2
        #   2048   59.9/47.3   40.7/30.5   45.7/33.1   46.5/40.0   42.7/32.7
        #   4096  108.2/90.8   68.3/56.3   55.6/47.5   57.3/46.4   67.8/56.0
        #   16384 371.5/369.8 187.6/146.1 157.2/128.0 153.4/124.5 164.0/124.3
        (4, 7168, 9, None): {
            None: ((512, 64, 8), (4096, 256, 8), (None, 256, 16)),
            "fp4": (
                (512, 64, 8),
                (2048, 256, 8),
                (4096, 256, 24),
                (None, 256, 32),
            ),
        },
        # topk 6 (384 experts at EP4). The edges move in: 64x8 stops paying at 512.
        #   ct     64x8   64x16  128x16 256x16      (bf16 / fp8 / fp4)
        #   512    54.4    55.3   55.1   55.3  |  50.4 47.9 48.1  --  | 48.3 46.6 46.6 46.3
        #   1024   75.8    68.8   68.1   69.4  |  68.0 55.2 55.6 54.7 | 66.6 50.6 50.8 51.3
        #   2048  101.3   103.5  102.3  101.7  |  86.5 78.5 76.6 76.2 | 85.8 67.9 64.2 64.5
        #   16384 551.7   518.5  478.5  470.7  | 423.8 303. 264.9 266.| 414.9 233.7 172.6 166.4
        (4, 7168, 6, None): {
            None: ((512, 64, 8), (2048, 64, 16), (None, 128, 16)),
            "fp4": ((512, 64, 8), (1024, 64, 16), (None, 128, 16)),
        },
        # EP8 on two 4-GPU hosts, topk 6, fp4 dispatch, 2026-10-03; the grid stays at 64 blocks
        # (the rest of the CUs belong to the co-resident GEMM). What wins is one token a warp:
        # past 1024 tokens 64x16 gives some warps a second token, and its metadata split
        # (kPlainMeta in ep_intranode_1250x.hpp) up to four a payload warp, while 64x24 holds
        # 1536 and 64x32 2048 at one each. Past 2048 64x16 wins again up to about 2900, 64x32
        # from there to 4096 (its metadata warps borrow the payload warps' metadata slabs), and
        # 64x16 above that; 8192 is the one point past 4096 where 64x32 is ahead (-2). Dispatch
        # us, graph, mean over the ranks, 3 alternating rounds (up to 2304 with 4096 tokens a
        # rank allocated, past it 8192):
        #   ct     64x16  64x24  64x32
        #   1088    63.6   54.0   56.3
        #   1536    66.6   62.4   63.7
        #   1792    72.4   76.5   69.6
        #   2048    78.8   79.6   74.8
        #   2304    82.5   83.0   86.8
        #   2816    91.1          92.1
        #   3328   109.5         100.7
        #   4096   123.2         119.9
        #   5120   149.4         161.0
        (8, 7168, 6, None): {
            None: ((2048, 64, 16), (None, 128, 16)),
            "fp4": (
                (1024, 64, 16),
                (1536, 64, 24),
                (2048, 64, 32),
                (None, 128, 16),
            ),
        },
    },
}

# No dtype axis: combine reduces the bf16 staging region whatever dispatch carried,
# and three runs per shape (one per dispatch dtype) agreed to 1-5%.
_COMBINE_TABLE: dict = {
    # MI355X/gfx950 EP8: 64x8 wins at every token count and both topk against
    # 32x8 / 48x8 / 64x16 / 80x8 (us at ct=4096, topk 8: 991.6 / 750.5 / 724.3 / 738.5 /
    # 745.2). Not merely the smallest tried -- 32x8 costs 37% at ct=4096 -- so MakeEpCfg
    # and _hip_default() moved off v1's 80 to match. On gfx1250, 128x4 never wins and
    # 256x8 collapses 2x at 16384.
    "mi355x": {
        (8, 7168, 8, None): ((None, 64, 8),),
        (8, 7168, 6, None): ((None, 64, 8),),
    },
    "gfx1250": {
        (4, 7168, 8, None): ((None, 64, 8),),
        # topk 9 = 8 routed + 1 shared (ATOM's actual dispatch key), bf16 pull combine, 9
        # geometries swept (us, random / grouped routing). 192x8 is slow at 256-512 under
        # both routings, so it only takes the large counts.
        #   ct     64x8        128x8       256x8       192x8
        #   16     15.6/14.7   14.5/13.9   23.0/15.9   14.4/13.8
        #   256    24.7/18.4   21.3/17.5   22.2/18.1   41.0/29.6
        #   1536   71.8/69.0   59.0/48.2   60.6/44.4   58.2/43.7
        #   4096  163.5/163.3 130.7/104.5 129.6/94.3  130.9/95.1
        #   16384 599.0/609.7 470.8/369.5 481.1/388.8 484.7/338.3
        (4, 7168, 9, None): ((256, 128, 8), (4096, 256, 8), (None, 192, 8)),
        (4, 7168, 6, None): ((None, 64, 8),),
        # The single-shot default, listed so the EP8 dispatch schedule has a combine half.
        (8, 7168, 6, None): ((None, 64, 8),),
    },
}

_COMBINE_FP4_TABLE: dict = {
    "fp4_blockwise": {
        "gfx1250": {
            # EP4 64x24 runs the overlapped path too; from 2048 tokens a rank it is ahead of 64x16
            # (-1.9 at 2048, -17 at 4096, -46 at 8192, -109 at 16384: 361.9 -> 252.4) and below it
            # behind (+0.7 at 1792, +4.0 at 1536, +8.1 at 1024).
            (4, 7168, 6, None): (
                (256, 64, 8),
                (2047, 64, 16),
                (2048, 64, 24),
                (16384, 128, 24),
                (None, 64, 16),
            ),
            # topk 9: MegaMoE runs nothing beside the combine, so the whole grid pays. 10
            # geometries swept (us, random / grouped routing):
            #   ct     64x16       256x8       256x16      192x24      256x24
            #   1024   31.3/ --    23.8/19.7   25.0/19.4   30.2/26.0   29.3/27.1
            #   1536   43.0/ --    31.8/25.7   30.9/23.9   37.1/30.5   35.0/30.6
            #   4096  107.8/ --    70.5/58.3   74.7/53.2   60.5/51.5   63.2/54.3
            #   16384 396.0/ --   273.7/168.8 283.3/161.7 158.5/132.6 167.8/133.2
            (4, 7168, 9, None): (
                (1024, 256, 8),
                (2048, 256, 16),
                (None, 192, 24),
            ),
            # 64x24 runs the overlapped path with eight producers and twelve reducing waves:
            # 329.6 -> 314.8 us at 16384 tokens a rank. It tied with 64x16 at 4096 until its store
            # waves kept one chunk in flight; since then it is ahead at 2048 (-0.4) and 4096 (-1.7).
            (8, 7168, 6, None): (
                (1536, 64, 8),
                (2048, 64, 24),
                (16384, 128, 24),
                (None, 64, 8),
            ),
        },
    },
    "fp4_blockwise_fp32": {
        "gfx1250": {
            (4, 7168, 6, None): ((128, 64, 8), (None, 64, 16)),
        },
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
        {b[0] for b in disp if b[0] is not None}
        | {b[0] for b in comb if b[0] is not None}
    ) + [None]

    def pick(buckets, edge):
        for mx, blk, wrp in buckets:
            if mx is None or (edge is not None and edge <= mx):
                return blk, wrp
        return buckets[-1][1], buckets[-1][2]

    return tuple((edge,) + pick(disp, edge) + pick(comb, edge) for edge in edges)


# FlyDSL's dtype_str spelling, still accepted from callers that pass it (aiter's MegaMoE).
_DISPATCH_DTYPE_ALIAS = {"fp4_disp_bf16_comb": "fp4"}


def lookup(
    world_size,
    hidden_dim,
    topk,
    dtype="bf16",
    experts_per_rank=None,
    quant_type="none",
) -> dict:
    """HIP geometry for this device/shape/dtype, composed from HIP's own two tables.

    ``dtype`` is the dispatch wire ("bf16" / "fp8" / "fp4") and picks the dispatch
    half; ``quant_type`` picks the combine half. An unswept shape gets the HIP
    single-shot default (schedule=None). A swept one gets a per-token-count schedule
    built from the dispatch and combine tables independently, so either half can be
    re-tuned without touching the other.
    """
    base = _hip_default()
    dev = _device_key()
    disp = _bucket_key(
        _DISPATCH_TABLE.get(dev, {}), world_size, hidden_dim, topk, experts_per_rank
    )
    comb = None
    if quant_type in _COMBINE_FP4_TABLE:
        comb = _bucket_key(
            _COMBINE_FP4_TABLE[quant_type].get(dev, {}),
            world_size,
            hidden_dim,
            topk,
            experts_per_rank,
        )
    if comb is None:
        comb = _bucket_key(
            _COMBINE_TABLE.get(dev, {}), world_size, hidden_dim, topk, experts_per_rank
        )
    if disp is None or comb is None:
        return base  # half a schedule is not a schedule
    # None is the "every dtype measured the same" key; an exact dtype overrides it.
    disp = disp.get(_DISPATCH_DTYPE_ALIAS.get(dtype, dtype)) or disp.get(None)
    if disp is None:
        return base
    base["schedule"] = _merge(disp, comb)
    return base
