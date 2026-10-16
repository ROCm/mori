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
            "fp4_disp_bf16_comb": ((None, 128, 8),),
        },
        (8, 7168, 6, None): {
            None: ((None, 64, 8),),
            "fp4_disp_bf16_comb": ((None, 128, 8),),
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
        (4, 7168, 9, None): {
            None: ((512, 64, 8), (4096, 256, 8), (None, 256, 16)),
            "fp4_disp_bf16_comb": ((512, 64, 8), (1024, 64, 16), (None, 128, 16)),
        },
        # topk 6 (384 experts at EP4). The edges move in: 64x8 stops paying at 512.
        #   ct     64x8   64x16  128x16 256x16      (bf16 / fp8 / fp4)
        #   512    54.4    55.3   55.1   55.3  |  50.4 47.9 48.1  --  | 48.3 46.6 46.6 46.3
        #   1024   75.8    68.8   68.1   69.4  |  68.0 55.2 55.6 54.7 | 66.6 50.6 50.8 51.3
        #   2048  101.3   103.5  102.3  101.7  |  86.5 78.5 76.6 76.2 | 85.8 67.9 64.2 64.5
        #   16384 551.7   518.5  478.5  470.7  | 423.8 303. 264.9 266.| 414.9 233.7 172.6 166.4
        (4, 7168, 6, None): {
            None: ((512, 64, 8), (2048, 64, 16), (None, 128, 16)),
            "fp4_disp_bf16_comb": ((512, 64, 8), (1024, 64, 16), (None, 128, 16)),
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
            "fp4_disp_bf16_comb": (
                (1024, 64, 16),
                (1536, 64, 24),
                (2048, 64, 32),
                (None, 128, 16),
            ),
        },
        # EP16 on four 4-GPU hosts, topk 6, fp4 dispatch, 2026-10-11. 128x16 is the only geometry
        # that builds the pinned-destination dispatch (kDPin in ep_intranode_1250x.hpp). 128x16
        # minus 64x16, dispatch us, graph, 3 alternating rounds: M=512 plan, 64 / 256 / 384 / 512
        # tokens a rank +3.0 / +3.9..+4.1 / +2.5..+2.8 / +1.7..+2.0 (ab_1011_233137_978008); M=4096
        # plan, 768 / 1024 / 1280 / 1536 / 2048 / 2560 / 3072 / 4096 -18.2..-18.7 / -18.4..-18.6 /
        # -12.4..-12.5 / -9.3 / -4.8..-5.1 / -29.0..-29.1 / -35.9..-36.0 / -37.0..-37.1
        # (ab_1011_232944_972647); M=16384 plan, 8192 -14.7..-15.3, 16384 -80.9..-83.3. The 512
        # bucket is its own variant: at most 512 tokens a call compiles out the tile path (kCuAlways).
        (16, 7168, 6, None): {
            None: ((None, 64, 16),),
            "fp4_disp_bf16_comb": ((512, 64, 16), (None, 128, 16)),
        },
        # EP32 on eight 4-GPU hosts, topk 6, fp4 dispatch, 2026-10-11, one M=16384 plan, 3
        # alternating rounds, dispatch us (ab_1011_213858_566770):
        #   ct     64x16  128x16 256x16
        #   64     33.60  33.97  33.12
        #   512    38.14  39.26  41.06
        #   1024   57.01  47.75  50.79
        #   2048   97.24  80.83  68.47
        #   4096  144.96 145.69 133.17
        #   8192  239.43 226.28 239.29
        #   16384 459.61 432.08 414.95
        # 384 and 512 blocks hang (384 at 16384 tokens, 512 at the first call). The edges between
        # the swept points are unswept.
        (32, 7168, 6, None): {
            None: ((None, 64, 16),),
            "fp4_disp_bf16_comb": (
                (512, 64, 16),
                (1024, 128, 16),
                (4096, 256, 16),
                (8192, 128, 16),
                (None, 256, 16),
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
        # topk 9 = 8 routed + 1 shared (ATOM's actual dispatch key). Needed so the
        # dispatch schedule has a combine half -- lookup() returns the block-64
        # single-shot default unless BOTH halves exist. Combine geometry mirrors
        # topk 8 (unchanged); only the dispatch half is retuned to block 256.
        (4, 7168, 9, None): ((None, 64, 8),),
        (4, 7168, 6, None): ((None, 64, 8),),
        # The single-shot default, listed so the EP8 and EP16 dispatch schedules have a combine half.
        (8, 7168, 6, None): ((None, 64, 8),),
        (16, 7168, 6, None): ((None, 64, 8),),
        (32, 7168, 6, None): ((None, 64, 8),),
    },
}

_COMBINE_PUSH_TABLE: dict = {
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
            (4, 7168, 9, None): (
                (256, 64, 8),
                (2047, 64, 16),
                (2048, 64, 24),
                (16384, 128, 24),
                (None, 64, 16),
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
            # EP16, 2026-10-11, dispatch 128x16, one M=16384 plan, 3 alternating rounds, combine us:
            #   ct     64x8   64x16  64x24  128x16 128x24
            #   64     16.0   24.4   24.9   22.6   22.9
            #   512    28.3   34.4   34.7   30.7   31.3
            #   1024   46.4                 40.7   41.6
            #   2048   74.0                 57.0   57.5
            #   4096  127.9  111.8  109.3   88.2   87.6
            #   8192  232.7                159.4  159.0
            #   16384 434.2  369.1  353.5  306.8  303.5
            # The edge between 512 and 1024 is unswept.
            (16, 7168, 6, None): (
                (512, 64, 8),
                (8192, 128, 16),
                (None, 128, 24),
            ),
            # EP32, 2026-10-11, the run of the EP32 dispatch entry, combine us:
            #   ct     64x8   128x16 128x24 256x8  256x16
            #   64     16.19  22.65  23.19  16.07  22.02
            #   512    32.70  34.52  35.56  28.03  34.50
            #   1024   48.40  45.94  47.71  46.74  44.17
            #   2048   77.71  63.25  65.13  76.20  64.69
            #   4096  134.05 105.14 105.87 112.95 106.61
            #   8192  243.99 191.68 197.61 209.69 195.29
            #   16384 462.90 371.53 358.19 402.34 368.26
            (32, 7168, 6, None): (
                (512, 256, 8),
                (1024, 256, 16),
                (8192, 128, 16),
                (None, 128, 24),
            ),
        },
    },
    "fp4_blockwise_fp32": {
        "gfx1250": {
            (4, 7168, 6, None): ((128, 64, 8), (None, 64, 16)),
        },
    },
    "fp8_blockwise": {
        "gfx1250": {
            # 128x16 is ahead of 64x16 from 512 to 1536 tokens a rank (-3.6, -7.0, -14.6 at topk 6).
            # It stays ahead of 128x24 through 2048 at topk 6 (-3.5) but only through 1792 at topk 9
            # (-3.2); at 2048 topk 9 128x24 is ahead by 1.8.
            (4, 7168, 6, None): (
                (256, 64, 8),
                (2048, 128, 16),
                (16384, 128, 24),
                (None, 64, 16),
            ),
            (4, 7168, 9, None): (
                (256, 64, 8),
                (2047, 128, 16),
                (16384, 128, 24),
                (None, 64, 16),
            ),
        },
    },
}

# The LL128 protocol (EpCfg::ll128 in ep_cfg.hpp): an op of at most `max_tok` tokens a rank
# runs every call on one dispatch and one combine plan, block x warp each, instead of the
# schedule. One plan each because the per-block call numbers are module globals, which a
# second plan would not share. The plans stay on 64 blocks at these token counts (the rest of
# the CUs belong to the co-resident GEMM); dispatch and combine may differ in warps a block.
# max_tok is the largest op whose dispatch and combine are both faster than the schedule's at
# every token count. Keyed by quant_type, then (world, hidden, topk, experts a rank) and the
# dispatch dtype. EP4 on C17, 2026-10-16, 3 alternating rounds, us against the schedule:
#   fp4 on 64x16 + 64x16, an op of 256, at 1 / 64 / 128 / 256: dispatch 18.0 -> 8.1,
#     22.9 -> 10.9, 21.1 -> 12.2, 22.4 -> 18.9; combine 12.2 -> 7.6, 14.7 -> 8.7, 15.1 -> 9.9,
#     16.7 -> 15.1. At 512, in an op of 512, both lose (24.3 -> 29.0, 20.4 -> 26.1).
#   fp8 on 64x8 + 64x8, an op of 128, at 1 / 16 / 64 / 128: dispatch 18.2 -> 8.2, 22.8 -> 9.5,
#     24.2 -> 12.6, 24.6 -> 14.5; combine 12.2 -> 8.1, 13.7 -> 8.6, 15.2 -> 9.9, 16.3 -> 13.7.
#     Its combine loses at 256 (18.9 -> 21.9), and on 64x16 from 64 (15.2 -> 15.8).
#   bf16 on 64x16 + 64x16, an op of 64, at 1 / 16 / 64: dispatch 18.5 -> 9.6, 25.2 -> 10.8,
#     27.2 -> 14.7; combine 19.5 -> 9.9, 16.1 -> 10.6, 16.4 -> 16.0. Its combine loses at 128
#     (17.6 -> 23.6).
# The fp4 push combine on 64x32 does not build.
_LL128_TABLE: dict = {
    "fp4_blockwise": {
        "gfx1250": {
            (4, 7168, 6, None): {"fp4": dict(max_tok=256, dispatch=(64, 16), combine=(64, 16))},
            # EP8: the combine on 64x8, where it keeps 365 VGPRs; on 64x16 (256 a lane) it spills 230.
            (8, 7168, 6, None): {"fp4": dict(max_tok=128, dispatch=(64, 16), combine=(64, 8))},
        },
    },
    "fp8_blockwise": {
        "gfx1250": {
            (4, 7168, 6, None): {"fp8": dict(max_tok=128, dispatch=(64, 8), combine=(64, 8))},
            (8, 7168, 6, None): {"fp8": dict(max_tok=128, dispatch=(64, 8), combine=(64, 8))},
        },
    },
    "none": {
        "gfx1250": {
            (4, 7168, 6, None): {"bf16": dict(max_tok=64, dispatch=(64, 16), combine=(64, 16))},
            # EP8: the combine on 64x8 (333 VGPRs, no spill); past 16 tokens it is slower than the pull.
            (8, 7168, 6, None): {"bf16": dict(max_tok=16, dispatch=(64, 16), combine=(64, 8))},
        },
    },
}


def ll128(world_size, hidden_dim, topk, dtype, max_tok, experts_per_rank=None, quant_type="none"):
    """The LL128 plan geometry for an op of this shape, dispatch dtype and capacity:
    {"dispatch": (block, warp), "combine": (block, warp)} when both kernels take LL128,
    {"combine": (block, warp)} when only the combine does, or None when the op keeps the
    schedule."""
    per_shape = _LL128_TABLE.get(quant_type, {}).get(_device_key(), {})
    entry = _bucket_key(per_shape, world_size, hidden_dim, topk, experts_per_rank)
    plan = entry.get(dtype) if entry else None
    if plan is None:
        return None
    if max_tok <= plan["max_tok"]:
        return {"dispatch": plan["dispatch"], "combine": plan["combine"]}
    if max_tok <= plan.get("combine_max_tok", 0):
        return {"combine": plan["combine"]}
    return None


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


def lookup(
    world_size,
    hidden_dim,
    topk,
    dtype="bf16",
    experts_per_rank=None,
    quant_type="none",
) -> dict:
    """HIP geometry for this device/shape/dtype, composed from HIP's own two tables.

    An unswept shape gets the HIP single-shot default (schedule=None). A swept one
    gets a per-token-count schedule built from the dispatch and combine tables
    independently, so either half can be re-tuned without touching the other.
    """
    base = _hip_default()
    dev = _device_key()
    disp = _bucket_key(
        _DISPATCH_TABLE.get(dev, {}), world_size, hidden_dim, topk, experts_per_rank
    )
    comb = None
    if quant_type in _COMBINE_PUSH_TABLE:
        comb = _bucket_key(
            _COMBINE_PUSH_TABLE[quant_type].get(dev, {}),
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
    disp = disp.get(dtype) or disp.get(None)
    if disp is None:
        return base
    base["schedule"] = _merge(disp, comb)
    return base
