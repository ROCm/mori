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

The C++/HIP kernels are a different implementation from the FlyDSL ones: different
register pressure, occupancy, and best warp count (e.g. HIP combine peaks at 8
warps on gfx950 EP8 where FlyDSL uses 4). So they carry their OWN tuned geometry
and never borrow FlyDSL's schedule. This module deliberately does NOT import
``tuning_configs`` (FlyDSL's table); it shares only the hardware-detection
primitive (``mori.ops.utils``), the same one FlyDSL's table uses.

THE TWO KERNELS ARE TUNED SEPARATELY, in two independent tables, because they are
not tuned by the same things:

  * dispatch depends on the DTYPE it transports (2 B, 1 B, or 2 e2m1 to a byte, which
    changes both the LDS tile per warp and how much payload there is to hide behind)
    and on topk, through the round rule below;
  * combine does NOT depend on the dispatch dtype at all. It only ever reduces the
    bf16/fp32 staging region, whatever dispatch carried. Giving it a dtype axis would
    invent a dimension and then oblige three copies of one answer to stay in sync.

Both are keyed by (world_size, hidden_dim, topk, experts_per_rank), with
``experts_per_rank=None`` meaning "any" -- use that until a sweep shows the expert
count actually moves the optimum, so the wildcard is a measured claim and not an
assumption nobody wrote down. A bucket is (max_tok_inclusive | None, block, warp),
ascending; lookup composes the two into the op's schedule.

Return contract matches ``tuning_configs.lookup`` so the op's ``_resolve_geometry``
treats both backends uniformly:

    {dispatch_block_num, combine_block_num, warp_num_per_block,
     combine_warp_num_per_block, schedule}

``schedule`` (or None) is a per-token-count launch plan: a tuple of
(max_tok_inclusive | None, disp_block, disp_warp, comb_block, comb_warp) buckets,
ascending; the op precompiles the distinct (block, warp) variants and picks a
bucket at runtime from cur_rank_num_token. When schedule is None the single-shot
block/warp fields are used. That the op wants the two halves interleaved is a detail
of its variant selector, not a reason to tune them together.

Unswept shapes fall back to HIP's single-shot default (schedule=None), whose values
mirror the C++ ``MakeEpCfg`` arch default so a bare C++ caller and this path agree.
Add a shape by sweeping it with ``bench_ep.py``; do NOT paste FlyDSL's
``tuning_configs`` numbers in -- different kernel, different optimum.

The combine table assumes a 2-byte combine dtype. An fp32 combine moves twice the
bytes and is untuned; it will take the bf16 buckets, which is a guess, not a result.
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
    """HIP single-shot default (no tuned schedule). Mirrors the C++ MakeEpCfg arch
    default: dispatch 64 blocks / 16 warps, combine 80 blocks / 8 warps -- the
    latter measured for the HIP combine kernel on gfx950 EP8."""
    return dict(
        dispatch_block_num=64,
        combine_block_num=80,
        warp_num_per_block=16,
        combine_warp_num_per_block=8,
        schedule=None,
    )


# ---------------------------------------------------------------------------
# The tables. bucket = (max_tok_inclusive | None, block, warp), ascending.
# key = (world_size, hidden_dim, topk, experts_per_rank); experts_per_rank None
# means "measured not to matter here", and an exact key wins over the wildcard.
# ---------------------------------------------------------------------------

# MEASURED on 4x gfx1250, hidden 7168, 2026-08-11: 64..16384 tokens x {64x8, 64x16,
# 128x16, 256x16} x {bf16, fp8, fp4} x {topk 8 / 64 experts, topk 6 / 96 experts},
# plus an expert-count isolation pass. Rule for picking: take the SMALLEST geometry
# within ~3% of the best, because fewer blocks means fewer CUs held and this overlaps
# an expert GEMM in production. Every entry below is at most 3.1% off its column's best.
#
# WHAT MOVES THE ANSWER, and what does not:
#   * topk does, a lot -- it sets _tpi = warpSize/topk, the tokens one warp consumes
#     per iteration, so it moves where a geometry stops covering the token count in one
#     round. At topk 8, 64x8 is 75.0us at ct=512 against 92.7 for 64x16; at topk 6 the
#     same pair is 54.6 vs 55.8, a tie. Hence separate entries per topk.
#   * the expert count does NOT. Measured at 64 and 96 experts/rank across both topk
#     values, both a wide and a narrow dtype, and all four geometries: every pair agrees
#     within ~2%, i.e. noise. That is what the None wildcard below records -- a result,
#     not an assumption.
#   * the dtype mostly does not, which is why most rows use the None dtype key. It only
#     bites at topk 6, where fp4 wants the wider grid two buckets earlier (ct=2048:
#     128x16 64.2us against 64x16 67.9).
_DISPATCH_TABLE: dict = {
    # MI355X / gfx950, EP8 hidden 7168, 2026-08-11: same grid, 64..16384 tokens x
    # {64x8, 64x16, 128x8, 128x16, 256x8} x {bf16, fp8, fp4} x {topk 8 / 32 experts,
    # topk 6 / 48 experts}, ITERS=50.
    #
    # THE ANSWER IS NOT gfx1250'S, and not because the numbers came out differently --
    # the shape of the problem is different. There is no TDM here: the portable dispatch
    # reserves no LDS (sharedBytes is 0) and moves its payload with plain vector copies,
    # so bf16 and fp8 are bandwidth-bound and the grid barely registers. Cost of taking
    # the smallest geometry, 64x8, against the best of the five at each token count:
    #
    #   bf16   0-2% everywhere, both topk        -> flat, take the smallest
    #   fp8    1-2% from ct>=512, both topk      -> same (the 4-16% at ct<=128 is noise:
    #                                               the ITERS=20 pass had it the other way)
    #   fp4    6-69% from ct>=128                -> the one dtype that cares
    #
    # fp4 is 1/4 the payload, which is what tips it out of bandwidth-bound and into
    # caring about how many blocks are issuing -- the same mechanism as on gfx1250, at a
    # different threshold. It wants 128x8 flat; at ct=64 that costs nothing (32.4 against
    # 32.5), so it gets one bucket rather than an edge.
    #
    # topk 6 and 8 came out identical here, unlike gfx1250 where _tpi moves the edges.
    # Written as two entries rather than a topk wildcard on purpose: the MECHANISM says
    # topk should matter (it sets tokens-per-warp-iteration), and it demonstrably does on
    # the other arch, so an unmeasured topk should fall back to the single-shot default
    # rather than inherit a schedule from a topk that happened to agree.
    "mi355x": {
        (8, 7168, 8, None): {None: ((None, 64, 8),), "fp4": ((None, 128, 8),)},
        (8, 7168, 6, None): {None: ((None, 64, 8),), "fp4": ((None, 128, 8),)},
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
            "fp4": ((512, 64, 8), (1024, 64, 16), (None, 128, 16)),
        },
    },
}

# No dtype axis on purpose: combine reduces the bf16 staging region no matter what
# dispatch transported, and the sweep ran it three times per shape (once per dispatch
# dtype) with the three agreeing to 1-5%. Both topk values also land on the same
# buckets, so these two entries are equal today and kept apart only because topk is
# part of the key -- do not merge them into a wildcard without measuring a third topk.
#   ct      64x8   128x4  128x8  256x8       (topk 8 / topk 6, us)
#   512     36.8    38.9   38.0   45.8   |   36.5  39.1  37.2  42.7
#   4096   160.4   161.2  146.1  154.7   |  157.5 155.1 135.8 146.0
#   16384  574.1   565.0  524.9  996.3   |  564.4 542.7 469.2 1027.0
# 256x8 is not a near miss at the top end, it is a 2x collapse; 128x4 never wins.
_COMBINE_TABLE: dict = {
    # MI355X / gfx950 EP8: 64x8 wins at every token count and both topk, against
    # 32x8 / 48x8 / 64x16 / 80x8 (us at ct=4096, topk 8: 991.6 / 750.5 / 724.3 / 738.5 /
    # 745.2). Note this REPLACES the 80x8 that MakeEpCfg still uses as its arch default
    # and that _hip_default() mirrors -- 80x8 was measured once, on a narrower grid; 64x8
    # is 2-3% faster with 16 fewer blocks. Fewer blocks than 64 is not free here: 32x8
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

# THE ROUND RULE, which outranks every shape choice above. _tpi = warpSize/topk
# tokens are consumed per warp-iteration, so one round covers block*warp*_tpi tokens;
# coming up one round short costs more than any geometry difference. At topk=8 on
# wave32 that is 4 tokens per warp-iteration, so 64x8 covers 2048 -- exactly where its
# bucket ends above. Measured at ct=4096, dispatch us: fp4 64x8 (one round short) 121.4
# against 64x16 80.5, and at ct=16384 it is 418.8 against 174.6. A new topk moves _tpi
# and therefore moves every edge; do not copy a bucket across topk without re-measuring.
#
# HEALTH-CHECK THE BOX BEFORE RE-TUNING. An earlier pass ran on a degraded machine and
# produced a self-consistent, entirely wrong answer (fp8/fp4 at 128x8, "+12%"), because
# the fault also flipped bf16's own optimum. Canary: bf16 dispatch at ct=4096 / 64x16 is
# ~157us healthy and ~172 degraded, while combine sits at ~145 either way -- combine
# being unchanged is what distinguishes a broken box from a busy one.

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
