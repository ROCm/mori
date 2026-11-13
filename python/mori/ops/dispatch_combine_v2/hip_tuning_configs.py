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
"""HIP V2 configuration: resolve_schedule(cfg) -> ScheduleEntry rows.

Both paths use KernelConfig keys and pick_bucket() with inclusive token ceilings.
Intra lookup() reads the IntraNodeV2 table first and falls back to the Python
tables per phase; internode_buckets() reads the CSV table v2_table_filename()
names for the GPU model, kernel family and EP size (read_v2_table()).
"""

from __future__ import annotations

import csv
import functools
import io
import numbers
import os
import tempfile
from pathlib import Path
from typing import NamedTuple

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


@functools.lru_cache(maxsize=None)
def _detected_arch():
    try:
        from mori.jit.config import detect_gpu_arch

        return detect_gpu_arch()
    except Exception:
        return None


def _device_keys():
    """Table keys to try, most specific first. KFD reports gfx1250 for every ASIC
    revision and either code target, so the parts that can run gfx1250-strict are
    told apart by their revision (0), or by running it; whichever code they run, they
    use the gfx1250-strict rows when there is one.
    """
    key = _device_key()
    if key != "gfx1250":
        return (key,)
    rev = _gpu.asic_revision()
    if rev == 0 or (rev is None and _detected_arch() == "gfx1250-strict"):
        return ("gfx1250-strict", "gfx1250")
    cu = _gpu.cu_count()
    return (f"{key}/{cu}cu", key) if cu else (key,)


def _first_bucket(table, devs, *shape):
    for dev in devs:
        found = _bucket_key(table.get(dev, {}), *shape)
        if found is not None:
            return found
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
# Dispatch dtype keys name only what dispatch transports: "bf16" / "fp8" / "fp4"
# (EpDispatchCombineConfig.dispatch_dtype_str). Combine has no dtype key; whether
# its bf16 rows go out as mxfp4 is quant_type's table.
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
    # A shape the IntraNodeV2 table has rules for uses those first; this serves the
    # other expert counts and the dtypes the table lacks (fp4, and fp8 at 256
    # experts topk 6).
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
        (4, 7168, 9, None): {
            None: ((512, 64, 8), (4096, 256, 8), (None, 256, 16)),
            "fp4": (
                (512, 64, 8),
                (1024, 128, 8),
                (1536, 192, 8),
                (2048, 256, 8),
                (8192, 192, 24),
                (None, 192, 32),
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
        # hidden 5120 (DeepSeek-V4.1-Flash), topk 6, 96 experts a rank. Swept 64-256 blocks x
        # 4-16 warps for each dispatch dtype. Graph us, median of 3 rounds, these rows against
        # 128x8 for bf16 and the 64x16 default for fp8 / fp4:
        #   ct     bf16 128x8   rows   fp8 64x16   rows   fp4 64x16   rows
        #   64          24.8    18.8        24.8   19.4        25.8   19.4
        #   512         27.2    26.2        26.3   23.4        32.1   22.0
        #   2048        55.9    52.9        46.7   37.5        45.5   29.6
        #   8192       169.0   165.2       107.8  100.9       100.1   69.2
        #   16384      321.8   308.8       197.9  195.9       180.4  132.6
        (4, 5120, 6, 96): {
            None: ((256, 64, 4), (512, 64, 8), (1024, 128, 8), (None, 256, 8)),
            "bf16": ((256, 64, 4), (512, 128, 4), (2048, 256, 4), (None, 128, 16)),
        },
        # Its 128-expert topk-3 draft MoE (32 experts a rank) takes the same rows: at 2048 tokens
        # bf16 / fp8 / fp4 run 43.0 / 31.6 / 26.1 against 47.1 / 42.5 / 41.2.
        (4, 5120, 3, 32): {
            None: ((256, 64, 4), (512, 64, 8), (1024, 128, 8), (None, 256, 8)),
            "bf16": ((256, 64, 4), (512, 128, 4), (2048, 256, 4), (None, 128, 16)),
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
            "bf16": ((None, 64, 16),),
            "fp4": (
                (1024, 64, 16),
                (1536, 64, 24),
                (2048, 64, 32),
                (None, 128, 16),
            ),
        },
        # EP16 on four 4-GPU hosts, topk 6, fp4 dispatch, 2026-10-11. 128x16 is the only geometry
        # that builds the pinned-destination dispatch (kDPin in ep_intranode_1250x.hpp). 128x16
        (16, 7168, 6, None): {
            None: ((None, 64, 16),),
            "fp4": ((512, 64, 16), (None, 128, 16)),
        },
        (32, 7168, 6, None): {
            None: ((512, 64, 16), (2048, 64, 16), (None, 128, 16)),
            "fp4": ((512, 64, 16), (1024, 64, 16), (None, 128, 16)),
        },
        # EP4, hidden 2048, topk 8, 64 experts a rank (Qwen3.5-35B-A3B at DP4 + EP4), 2026-10-30.
        # bench_ep.py, graph, ITERS=200, mean of 3 alternating rounds (spread <= 1.3 us). The
        # 64x16 default costs bf16 up to 9% to 512 tokens a rank and 19-46% past it; fp4 12-26% to
        # 1024 and about 2x past it, and on 64x16 fp4 is slower than bf16 at 192-256 and 2048-4096
        # tokens. bf16 and fp4 take the same schedule; fp8 and other expert counts are unswept.
        # Dispatch us:
        #   ct     32x8   64x8  64x16  128x8  256x8     (bf16 | fp4)
        #   64     24.4   24.5   26.6   24.4   24.4  |  21.3   21.1   23.8   21.0   20.9
        #   256    25.3   25.1   25.3   25.0   25.0  |  21.6   21.5   26.8   21.5   21.5
        #   512    31.4   26.3   26.9   26.0   26.0  |  31.5   21.9   26.0   21.7   21.7
        #   1024   41.2   32.5   34.7   29.1   28.8  |  41.0   31.8   26.6   22.9   22.7
        #   2048   64.2   43.6   51.8   38.3   35.5  |  67.6   41.6   52.7   32.4   25.7
        #   4096  112.6   68.4   64.9   54.5   50.9  | 119.4   68.9   68.3   43.1   34.5
        (4, 2048, 8, 64): {
            "bf16": ((256, 32, 8), (512, 64, 8), (1024, 128, 8), (None, 256, 8)),
            "fp4": ((256, 32, 8), (512, 64, 8), (1024, 128, 8), (None, 256, 8)),
        },
    },
    # Parts that can run gfx1250-strict (ASIC revision 0), either code target, EP4.
    # Dispatch and combine hold one block per CU (LDS) while a block waits for the whole
    # grid, and these parts have fewer CUs than the 256-block gfx1250 rows assume, so
    # those would hang on them.
    # Graph-timed, median of 3 rounds; within 3-5% the smaller geometry wins.
    "gfx1250-strict": {
        (4, 7168, 8, None): {
            None: ((512, 64, 8), (1024, 64, 16), (None, 64, 8)),
            "fp8": ((256, 64, 8), (1024, 64, 16), (1536, 64, 24), (None, 64, 32)),
            "fp4": (
                (256, 64, 8),
                (512, 128, 8),
                (1024, 64, 16),
                (1536, 64, 24),
                (8192, 64, 32),
                (None, 128, 16),
            ),
        },
        (4, 7168, 9, None): {
            None: ((4096, 64, 8), (None, 64, 16)),
            "fp8": ((256, 64, 8), (1024, 64, 16), (1536, 64, 24), (None, 64, 32)),
            "fp4": (
                (512, 64, 8),
                (1024, 64, 16),
                (1536, 64, 24),
                (8192, 64, 32),
                (None, 128, 16),
            ),
        },
        (4, 7168, 6, None): {
            None: ((4096, 64, 8), (None, 64, 16)),
            "fp8": ((1024, 64, 8), (1536, 64, 24), (2048, 64, 16), (None, 64, 32)),
            "fp4": (
                (512, 64, 8),
                (1024, 64, 16),
                (1536, 64, 24),
                (8192, 64, 32),
                (None, 128, 16),
            ),
        },
        # 16 warps from 512 tokens; the gfx1250 rows' 8-warp grids trail well behind here.
        (4, 2048, 8, 64): {
            "bf16": ((256, 32, 8), (512, 32, 16), (1024, 64, 16), (None, 128, 16)),
            "fp4": ((256, 32, 8), (512, 32, 16), (1024, 64, 16), (None, 128, 16)),
        },
    },
    "gfx1250/96cu": {
        (8, 7168, 6, None): {
            None: ((2048, 64, 16), (None, 128, 16)),
            "bf16": ((1024, 32, 16), (None, 64, 16)),
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
# and three runs per shape (one per dispatch dtype) agreed to 1-5%. A shape with no
# combine rule runs _hip_default()'s 64x8, which an early MI355X EP8 sweep picked
# (us at ct=4096, topk 8, for 32x8 / 48x8 / 64x8 / 64x16 / 80x8: 991.6 / 750.5 /
# 724.3 / 738.5 / 745.2) and which also moved MakeEpCfg off v1's 80; MI355X has since
# been tuned per bucket in its IntraNodeV2 table. On gfx1250, 128x4 never wins and
# 256x8 collapses 2x at 16384.
_COMBINE_TABLE: dict = {
    "gfx1250": {
        # topk 9 = 8 routed + 1 shared.
        (4, 7168, 9, None): (
            (64, 64, 4),
            (128, 64, 8),
            (256, 128, 8),
            (1024, 256, 8),
            (None, 192, 8),
        ),
        # hidden 5120: 64x8 up to 128 tokens, then 128x8; topk 3 moves to 256x4 past 512 tokens,
        # where it leads the 128-block grids by 4-10% up to 4096 (128x16 is 6% ahead at 8192).
        # Graph us, median of 3 rounds, topk 6 | topk 3:
        #   ct     64x8  128x8 128x16  256x4 |  64x8  128x8 128x16  256x4
        #   128    13.6   15.0   15.2   14.9 |  13.0   13.9   14.5   13.8
        #   512    24.4   21.4   20.9   21.2 |  23.0   19.5   18.9   19.4
        #   2048   65.5   49.1   50.7   48.5 |  54.7   42.8   40.5   39.1
        #   8192  228.8  163.8  189.7  165.5 | 172.1  131.0  124.4  131.5
        (4, 5120, 6, 96): ((128, 64, 8), (None, 128, 8)),
        (4, 5120, 3, 32): ((128, 64, 8), (512, 128, 8), (None, 256, 4)),
    },
    # The gfx1250-strict parts, as their dispatch rows.
    "gfx1250-strict": {
        (4, 7168, 8, None): ((128, 64, 4), (512, 128, 8), (None, 64, 8)),
        (4, 7168, 9, None): ((128, 64, 4), (256, 128, 8), (None, 64, 8)),
        (4, 7168, 6, None): ((128, 64, 4), (256, 128, 8), (None, 64, 8)),
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
                (128, 64, 8),
                (512, 192, 8),
                (1024, 256, 8),
                (1536, 192, 16),
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
            (32, 7168, 6, None): (
                (512, 64, 8),
                (2047, 64, 16),
                (2048, 64, 24),
                (16384, 128, 24),
                (None, 64, 16),
            ),
        },
        # gfx1250-strict parts, EP4. No 24-warp geometry: several times slower there.
        "gfx1250-strict": {
            (4, 7168, 6, None): ((256, 64, 8), (1024, 64, 16), (None, 128, 8)),
            (4, 7168, 9, None): (
                (512, 64, 8),
                (1024, 64, 16),
                (1536, 128, 8),
                (2048, 128, 16),
                (4096, 192, 8),
                (None, 128, 8),
            ),
        },
    },
    "fp4_blockwise_fp32": {
        "gfx1250": {
            (4, 7168, 6, None): ((128, 64, 8), (None, 64, 16)),
        },
        "gfx1250-strict": {
            (4, 7168, 6, None): (
                (512, 64, 8),
                (1024, 64, 16),
                (4096, 128, 8),
                (8192, 192, 8),
                (None, 128, 8),
            ),
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
            (32, 7168, 6, None): (
                (256, 64, 8),
                (384, 128, 8),
                (2048, 128, 16),
                (16384, 128, 24),
                (None, 64, 16),
            ),
        },
        # gfx1250-strict parts, EP4: no 24-warp geometry, as above.
        "gfx1250-strict": {
            (4, 7168, 6, None): ((256, 64, 8), (1536, 64, 16), (None, 128, 16)),
            (4, 7168, 9, None): ((256, 64, 8), (1536, 64, 16), (None, 128, 16)),
        },
    },
}

_BF16_PUSH_TABLE: dict = {
    "gfx1250": {
        (4, 7168, 6, None): ((128, 64, 8), (512, 64, 16), (None, 64, 24)),
        (8, 7168, 6, None): ((192, 64, 8), (512, 64, 16), (None, 64, 24)),
    },
    "gfx1250/96cu": {
        (8, 7168, 6, None): ((128, 32, 8), (None, 32, 16)),
    },
}

_LL128_TABLE: dict = {
    "fp4_blockwise": {
        "gfx1250": {
            (4, 7168, 6, None): {
                "fp4": dict(max_tok=256, dispatch=(64, 16), combine=(64, 16))
            },
            (8, 7168, 6, None): {
                "fp4": dict(max_tok=128, dispatch=(64, 16), combine=(64, 8))
            },
            (16, 7168, 6, None): {
                "fp4": dict(max_tok=128, dispatch=(64, 16), combine=(64, 8))
            },
            (32, 7168, 6, None): {
                "fp4": dict(max_tok=64, dispatch=(64, 16), combine=(64, 8))
            },
        },
    },
    "fp8_blockwise": {
        "gfx1250": {
            (4, 7168, 6, None): {
                "fp8": dict(max_tok=128, dispatch=(64, 8), combine=(64, 8))
            },
            (8, 7168, 6, None): {
                "fp8": dict(max_tok=128, dispatch=(64, 16), combine=(64, 8))
            },
            (16, 7168, 6, None): {
                "fp8": dict(max_tok=128, dispatch=(64, 16), combine=(64, 8))
            },
            (32, 7168, 6, None): {
                "fp8": dict(max_tok=16, dispatch=(64, 16), combine=(64, 8))
            },
        },
    },
    "none": {
        "gfx1250": {
            (4, 7168, 6, None): {
                "bf16": dict(max_tok=64, dispatch=(64, 16), combine=(64, 16)),
                "fp8": dict(max_tok=64, dispatch=(64, 8), combine=(64, 16)),
            },
            (8, 7168, 6, None): {
                "bf16": dict(max_tok=64, dispatch=(64, 16), combine=(64, 16)),
                "fp8": dict(max_tok=64, dispatch=(64, 16), combine=(64, 16)),
            },
            (16, 7168, 6, None): {
                "bf16": dict(max_tok=16, dispatch=(64, 16), combine=(64, 16)),
                "fp8": dict(max_tok=16, dispatch=(64, 16), combine=(64, 16)),
            },
        },
        "gfx1250/96cu": {
            (8, 7168, 6, None): {
                "bf16": dict(
                    max_tok=176, card_max_tok=160, dispatch=(32, 16), combine=(32, 16)
                ),
            },
        },
    },
}


def ll128(
    world_size,
    hidden_dim,
    topk,
    dtype,
    max_tok,
    experts_per_rank=None,
    quant_type="none",
    card_fold=False,
):
    devs = _device_keys()
    if devs[0] == "gfx1250-strict":
        return None  # LL128's polls miss peer writes on these parts; use the schedule
    per_device = _LL128_TABLE.get(quant_type, {})
    plan = None
    for key in devs:
        entry = _bucket_key(
            per_device.get(key, {}), world_size, hidden_dim, topk, experts_per_rank
        )
        plan = entry.get(dtype) if entry else None
        if plan is not None:
            break
    if plan is None:
        return None
    if card_fold and "card_max_tok" in plan:
        plan = dict(plan, max_tok=plan["card_max_tok"])
    if max_tok <= plan["max_tok"]:
        return {"dispatch": plan["dispatch"], "combine": plan["combine"]}
    if max_tok <= plan.get("combine_max_tok", 0):
        return {"combine": plan["combine"]}
    return {
        "dispatch": plan["dispatch"],
        "combine": plan["combine"],
        "max_tok": plan["max_tok"],
    }


def bf16_push(world_size, hidden_dim, topk, experts_per_rank=None):
    devs = _device_keys()
    if devs[0] == "gfx1250-strict":
        return None
    return _first_bucket(
        _BF16_PUSH_TABLE, devs, world_size, hidden_dim, topk, experts_per_rank
    )


_CARD_TABLE: dict = {
    "gfx1250/96cu": {
        (8, 7168, 6, None): dict(pull_blocks=8, pull_launch_max=256),
    },
}


def card(world_size, hidden_dim, topk, experts_per_rank=None):
    devs = _device_keys()
    if devs[0] == "gfx1250-strict":
        return None
    return _first_bucket(
        _CARD_TABLE, devs, world_size, hidden_dim, topk, experts_per_rank
    )


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


def pick_bucket(buckets, num_tokens):
    """Pick an inclusive ceiling, clamping above the last bucket.

    A None bound is open-ended; a None query requests the terminal bucket.
    Empty tables return None so callers can apply their own fallback policy.
    """
    for row in buckets:
        if row[0] is None or (num_tokens is not None and num_tokens <= row[0]):
            return row
    return buckets[-1] if buckets else None


def _ceilings(*bucket_lists, extra=()):
    """Every finite ceiling of the bucket lists and ``extra``, ascending, then
    None for the open-ended tail: the edges a merge of the lists breaks at."""
    finite = {row[0] for rows in bucket_lists for row in rows if row[0] is not None}
    return [*sorted(finite.union(extra)), None]


def _probe(ceiling, ceilings):
    """A token count inside the bucket ending at ``ceiling``: the ceiling
    itself, or for the open-ended tail one past every finite ceiling."""
    if ceiling is not None:
        return ceiling
    return max((c for c in ceilings if c is not None), default=0) + 1


class KernelConfig(NamedTuple):
    """One HIP kernel choice; intra has no RDMA split."""

    family: str
    block_num: int
    warp_per_block: int
    rdma_block_num: int | None = None
    # Ordinary-v2 combine only: compile the load-first gather (the table's
    # combine_load_first). Part of the key, so buckets that share a geometry
    # but differ here get separate kernels.
    load_first: bool = False
    # Internode only: how many of the allocated QPs (cfg.num_qp_per_pe) this
    # kernel sends on. A compile-time constant of each kernel, chosen per phase
    # and per bucket (the table's active_qps), so it is part of the key too.
    active_qps: int | None = None


class ScheduleEntry(NamedTuple):
    max_tokens: int | None
    dispatch: KernelConfig
    combine: KernelConfig


def resolve_schedule(cfg, cu_count=None) -> tuple[ScheduleEntry, ...]:
    """The HIP V2 schedule consumed by both construction and token selection.

    Intra adapts the schedule Config resolved: per phase, the IntraNodeV2 table's
    rules when quant_type is "none", else the Python table's; an explicit
    cfg.schedule is taken as given, and pinned or untuned geometry is one bucket.
    Inter names each bucket of internode_schedule() with its kernel family; tuned
    grids are clamped to ``cu_count``, the backend's own CU count. Loading and
    selection never depend on the declared token capacity; that includes the
    combine load-first choice, which comes from each bucket's rule
    (combine_load_first). AsyncLL is one entry: it compiles a single geometry,
    and its specs derive every pass's grid themselves.
    """
    if getattr(cfg, "async_ll", False):
        spec = KernelConfig("asyncll", cfg.dispatch_block_num, cfg.warp_num_per_block)
        return (ScheduleEntry(None, spec, spec),)
    if not cfg.is_internode:
        rows = cfg.schedule or (
            (
                None,
                cfg.dispatch_block_num,
                cfg.warp_num_per_block,
                cfg.combine_block_num,
                cfg.combine_warp_num_per_block,
            ),
        )
        return tuple(
            ScheduleEntry(
                ceiling, KernelConfig("intra", db, dw), KernelConfig("intra", cb, cw)
            )
            for ceiling, db, dw, cb, cw in rows
        )

    geometries = internode_schedule(cfg, cu_count=cu_count)
    ceilings = [row[0] for row in geometries]
    entries = []
    for ceiling, dispatch, combine, load_first in geometries:
        family = internode_kernel_family(cfg, _probe(ceiling, ceilings))
        entries.append(
            ScheduleEntry(
                ceiling,
                KernelConfig(
                    family,
                    block_num=dispatch[0],
                    warp_per_block=dispatch[2],
                    rdma_block_num=dispatch[1],
                    active_qps=dispatch[3],
                ),
                KernelConfig(
                    family,
                    block_num=combine[0],
                    warp_per_block=combine[2],
                    rdma_block_num=combine[1],
                    load_first=load_first,
                    active_qps=combine[3],
                ),
            )
        )
    return tuple(entries)


def describe_schedule(cfg, schedule) -> str:
    """Multi-line summary of a resolved schedule and the table files looked up for
    its shape (found or missing)."""

    def fmt(spec):
        if spec.rdma_block_num is None:
            return f"{spec.family} B/W={spec.block_num}/{spec.warp_per_block}"
        return (
            f"{spec.family} B/R/W="
            f"{spec.block_num}/{spec.rdma_block_num}/{spec.warp_per_block}"
            f" qps={spec.active_qps}" + (" load-first" if spec.load_first else "")
        )

    num_experts = cfg.world_size * cfg.num_experts_per_rank
    lines = [
        f"HIP V2 schedule: {_gpu.arch_name()}/{_gpu.detect_model()} "
        f"ep{cfg.world_size} expert{num_experts} top{cfg.num_experts_per_token} "
        f"hidden{cfg.hidden_dim} dispatch={cfg.dispatch_dtype} combine={cfg.combine_dtype}"
    ]
    if cfg.is_internode:
        families = [_V2_KERNEL_TYPES[f] for f in internode_kernel_families(cfg)]
        overrides = [
            name
            for name in ("MORI_EP_DISP_GEOM", "MORI_EP_COMB_GEOM")
            if os.environ.get(name)
        ]
        if overrides:
            lines.append(
                "  table rules overridden by "
                + ", ".join(f"{name}={os.environ[name]}" for name in overrides)
            )
        lines.append(
            f"  QPs allocated per peer: {cfg.num_qp_per_pe}"
            + (f", active pinned to {cfg.active_qps}" if cfg.active_qps else "")
        )
        pinned = sorted(getattr(cfg, "_pinned_geometry", ()))
        if pinned:
            lines.append(f"  pinned fields: {', '.join(pinned)}")
    else:
        families = [_V2_KERNEL_TYPES["intra"]]
    model = _device_key()
    for kernel_type in families:
        if model is None:
            lines.append(f"  no {kernel_type} table: unknown GPU model")
            continue
        path = internode_table_dir() / v2_table_filename(
            _gpu.arch_name(), model, kernel_type, cfg.world_size
        )
        lines.append(f"  {'found' if path.is_file() else 'missing'}: {path}")
    for row in schedule:
        ceiling = "rest" if row.max_tokens is None else f"<={row.max_tokens}"
        lines.append(
            f"  tokens {ceiling:>7}: dispatch {fmt(row.dispatch)}, "
            f"combine {fmt(row.combine)}"
        )
    return "\n".join(lines)


def _merge(disp, comb):
    """Interleave two independent bucket lists into the op's one schedule.

    The edges need not line up: the merged list breaks at the union of both, and each
    half keeps whatever it asked for on either side of the other's edge.
    """
    return tuple(
        (edge,) + pick_bucket(disp, edge)[1:] + pick_bucket(comb, edge)[1:]
        for edge in _ceilings(disp, comb)
    )


# FlyDSL's dtype_str spelling, still accepted from callers that pass it (aiter's MegaMoE).
_DISPATCH_DTYPE_ALIAS = {"fp4_disp_bf16_comb": "fp4"}


def _wire_dtypes(wire):
    """(dispatch, combine) torch dtypes for a dispatch wire name -- "bf16",
    "fp8" in the arch's native encoding, or "fp4" -- with BF16 combine, the
    only combine dtype the IntraNodeV2 tables hold. (None, None) otherwise."""
    from mori.ops.tuning_config import CONFIG_STR_TO_DTYPE

    if wire == "fp8":
        wire = "fp8_e4m3_fnuz" if _gpu.arch_name() == "gfx942" else "fp8_e4m3"
    dispatch = CONFIG_STR_TO_DTYPE.get(wire)
    if dispatch is None:
        return None, None
    return dispatch, CONFIG_STR_TO_DTYPE["bf16"]


def lookup(
    world_size,
    hidden_dim,
    topk,
    dtype="bf16",
    experts_per_rank=None,
    quant_type="none",
    dispatch_dtype=None,
    combine_dtype=None,
) -> dict:
    """HIP geometry for this device/shape/dtype, composed from HIP's own two tables.

    ``dtype`` is the dispatch wire ("bf16" / "fp8" / "fp4") and picks the dispatch
    half; ``quant_type`` picks the combine half. Each half is built independently,
    so either can be re-tuned without touching the other: its IntraNodeV2 table
    rules when it has any (``quant_type="none"`` only), else the Python table's,
    else the HIP single-shot default. A shape with neither half tuned gets the
    default and no schedule.

    The table is matched with the legs' torch dtypes; a caller that passes only
    ``dtype`` gets the ones it names (_wire_dtypes), so both ways of calling read
    the same rules.
    """
    base = _hip_default()
    devs = _device_keys()
    shape = (world_size, hidden_dim, topk, experts_per_rank)
    wire = _DISPATCH_DTYPE_ALIAS.get(dtype, dtype)
    disp = _first_bucket(_DISPATCH_TABLE, devs, *shape)
    if disp is not None:
        # None is the "every dtype measured the same" key; an exact dtype overrides it.
        disp = disp.get(wire) or disp.get(None)
    comb = None
    if quant_type in _COMBINE_PUSH_TABLE:
        comb = _first_bucket(_COMBINE_PUSH_TABLE[quant_type], devs, *shape)
    if comb is None:
        comb = _first_bucket(_COMBINE_TABLE, devs, *shape)
    if dispatch_dtype is None:
        dispatch_dtype, combine_dtype = _wire_dtypes(wire)
    if dispatch_dtype is not None and quant_type == "none":
        table_disp, table_comb = intranode_buckets(
            world_size,
            hidden_dim,
            topk,
            dispatch_dtype,
            combine_dtype,
            experts_per_rank=experts_per_rank,
        )
        disp = table_disp or disp
        comb = table_comb or comb
    if disp is None and comb is None:
        return _gpu.fit_blocks_to_cus(base)
    disp = disp or ((None, base["dispatch_block_num"], base["warp_num_per_block"]),)
    comb = comb or (
        (None, base["combine_block_num"], base["combine_warp_num_per_block"]),
    )
    base["schedule"] = _merge(disp, comb)
    return _gpu.fit_blocks_to_cus(base)


_INTERNODE_TABLE_DIR = Path(__file__).parent / "tuning_configs"
# Table kernel_type per family. "intra" is the single-node HIP op, which
# shares the internode loader, matching rules and save path.
_V2_KERNEL_TYPES = {
    "v2": "InterNodeV2",
    "v2_ll": "InterNodeV2LL",
    "intra": "IntraNodeV2",
}
_INTERNODE_PHASES = ("dispatch", "combine")
_INTERNODE_GEOMETRY_FIELDS = (
    ("dispatch_block_num", "dispatch_rdma_block_num", "warp_num_per_block"),
    ("combine_block_num", "combine_rdma_block_num", "combine_warp_num_per_block"),
)


def internode_kernel_family(cfg, num_tokens):
    """Resolve auto using live tokens, independently of the declared capacity."""
    if cfg.internode_kernel != "auto":
        return cfg.internode_kernel
    return "v2_ll" if num_tokens <= cfg.internode_auto_ll_max_tokens else "v2"


def internode_kernel_families(cfg):
    """Families to prepare before launch; only auto needs both."""
    return (
        ("v2", "v2_ll") if cfg.internode_kernel == "auto" else (cfg.internode_kernel,)
    )


def fit_internode_geometry(geometry, *, clamp_to_cu=False, cu_count=None):
    """Cap rdma at max(1, block - 1), which leaves at least one intra-node block
    whenever block > 1; optionally cap tuned grids to the CU count: ``cu_count``,
    the count the op launches against, else KFD's. An unknown count caps nothing.
    Fields after block, rdma, warp (the active QP count) pass through."""
    if geometry is None:
        return None
    block, rdma, warp, *rest = geometry
    limit = (cu_count or _gpu.cu_count()) if clamp_to_cu else 0
    if limit:
        block = min(block, limit)
    return (block, min(rdma, max(1, block - 1)), warp, *rest)


def resolve_active_qps(cfg, active_qps=None):
    """The active QP count a kernel runs with: a configured cfg.active_qps wins;
    otherwise the table's count, else min(2, allocation). A table count above
    the allocation is clamped to it, the way block_num is clamped to the CU
    count, so a row tuned under one allocation stays usable under another."""
    if cfg.active_qps is not None:
        return cfg.active_qps
    if active_qps is None:
        active_qps = 2
    return min(active_qps, cfg.num_qp_per_pe)


def _geometry_from_env(name, cfg):
    """Optional sweep override, read once when the op is built:
    B,R,W[,active_qps]. A fourth field beats cfg.active_qps and is rejected,
    not clamped, outside the allocation: a sweep that silently ran a smaller
    count would report a winner it never measured."""
    raw = os.environ.get(name)
    if not raw:
        return None
    geometry = tuple(int(x) for x in raw.replace(" ", "").split(","))
    if len(geometry) == 3:
        return geometry + (resolve_active_qps(cfg),)
    if len(geometry) != 4:
        raise ValueError(
            f"{name}={raw!r}: want block,rdma,warp or block,rdma,warp,active_qps"
        )
    if not 1 <= geometry[3] <= cfg.num_qp_per_pe:
        raise ValueError(
            f"{name}: active_qps {geometry[3]} must be in "
            f"[1, num_qp_per_pe={cfg.num_qp_per_pe}]"
        )
    return geometry


def internode_schedule(cfg, cu_count=None):
    """Resolve (inclusive token ceiling, dispatch B/R/W/QPs, combine B/R/W/QPs,
    combine load-first).

    A sweep env override replaces its phase's geometry in every bucket, as
    given, and a combine override runs the plain gather; with both set the
    tables are not read. Otherwise explicit config fields override the
    selected family's table, with config defaults for an untuned phase. Table
    values and unpinned defaults are clamped to ``cu_count``
    (fit_internode_geometry); pinned fields are kept as given. All paths cap R
    at max(1, B - 1), so R < B whenever B > 1. Under auto the family crossover
    is always a bucket edge.
    """
    defaults = [
        tuple(getattr(cfg, name) for name in fields) + (resolve_active_qps(cfg),)
        for fields in _INTERNODE_GEOMETRY_FIELDS
    ]
    overrides = [
        _geometry_from_env("MORI_EP_DISP_GEOM", cfg),
        _geometry_from_env("MORI_EP_COMB_GEOM", cfg),
    ]
    crossover = (
        (cfg.internode_auto_ll_max_tokens,) if cfg.internode_kernel == "auto" else ()
    )
    if all(overrides):
        fitted = tuple(fit_internode_geometry(g) for g in overrides)
        return [(ceiling, *fitted, False) for ceiling in _ceilings(extra=crossover)]

    tables = {
        family: internode_buckets(
            cfg.world_size,
            cfg.hidden_dim,
            cfg.num_experts_per_token,
            cfg.dispatch_dtype,
            cfg.combine_dtype,
            kernel_family=family,
            experts_per_rank=cfg.num_experts_per_rank,
            cu_count=cu_count,
        )
        or []
        for family in internode_kernel_families(cfg)
    }
    ceilings = _ceilings(*tables.values(), extra=crossover)

    pinned = getattr(cfg, "_pinned_geometry", frozenset())
    rows, families = [], []
    for ceiling in ceilings:
        probe = _probe(ceiling, ceilings)
        family = internode_kernel_family(cfg, probe)
        tuned = pick_bucket(tables[family], probe)
        geometries = []
        for index, (fields, fallback) in enumerate(
            zip(_INTERNODE_GEOMETRY_FIELDS, defaults), start=1
        ):
            geometry = (
                tuned[index]
                if tuned and tuned[index]
                else fit_internode_geometry(
                    fallback, clamp_to_cu=True, cu_count=cu_count
                )
            )
            geometry = tuple(
                getattr(cfg, name) if name in pinned else value
                for name, value in zip(fields, geometry[:3])
            ) + (
                resolve_active_qps(cfg, geometry[3]),
            )
            geometries.append(fit_internode_geometry(geometry))
        geometries = [
            geometry if override is None else fit_internode_geometry(override)
            for geometry, override in zip(geometries, overrides)
        ]
        # The load-first gather exists in the ordinary combine only.
        load_first = (
            bool(tuned and tuned[3]) and family == "v2" and overrides[1] is None
        )
        row = (ceiling, *geometries, load_first)
        # Equal neighbours of one family share a bucket. A family boundary
        # stays an edge even between equal geometries, so each bucket's family
        # follows from its ceiling alone (resolve_schedule).
        if rows and rows[-1][1:] == row[1:] and families[-1] == family:
            rows[-1] = row
        else:
            rows.append(row)
            families.append(family)
    return rows


def internode_table_dir():
    """V2 reads only this directory; V1's MORI_EP_TUNING_CONFIG never applies."""
    return Path(os.environ.get("MORI_EP_V2_TUNING_DIR") or _INTERNODE_TABLE_DIR)


def v2_table_filename(gpu_arch, gpu_model, kernel_type, ep_size):
    """One V2 table per GPU model, kernel family and EP size, holding both
    phases, e.g. ``gfx950_mi355x_InterNodeV2_ep16.csv``."""
    return f"{gpu_arch}_{gpu_model}_{kernel_type}_ep{ep_size}.csv"


# One rule per row. The first six columns are the selector (rows are ordered by
# _table_key), then the geometry and its per-bucket flags, then metrics, which are
# for readers only. An empty cell is an absent field: active_qps or
# combine_load_first (combine-only) left at the default.
_TABLE_SELECTOR = (
    "num_experts",
    "topk",
    "phase",
    "dtype",
    "hidden_dim",
    "num_tokens",
)
_TABLE_COLUMNS = _TABLE_SELECTOR + (
    "block_num",
    "rdma_block_num",
    "warp_per_block",
    "active_qps",
    "combine_load_first",
    "avg_latency_us",
    "bandwidth_gbps",
    "avg_rdma_bandwidth_gbps",
    "avg_xgmi_bandwidth_gbps",
    "avg_ll_bandwidth_gbps",
    "bandwidth_metric",
)
_TABLE_REQUIRED = (
    "num_experts",
    "topk",
    "phase",
    "dtype",
    "hidden_dim",
    "num_tokens",
    "block_num",
    "rdma_block_num",
    "warp_per_block",
)
_TABLE_INT_MINIMUM = {
    "num_experts": 1,
    "topk": 1,
    "hidden_dim": 1,
    "num_tokens": 1,
    "block_num": 1,
    "rdma_block_num": 0,
    "warp_per_block": 1,
    "active_qps": 1,
}
_TABLE_BOOLS = frozenset({"combine_load_first"})
_TABLE_FLOATS = frozenset(
    {
        "avg_latency_us",
        "bandwidth_gbps",
        "avg_rdma_bandwidth_gbps",
        "avg_xgmi_bandwidth_gbps",
        "avg_ll_bandwidth_gbps",
    }
)
# Config strings rather than this torch build's dtype registry, so a table
# loads the same whichever dtypes torch has.
_TABLE_DTYPES = frozenset({"bf16", "fp8_e4m3_fnuz", "fp8_e4m3", "fp4"})


# A table's rows group by shape (experts, top-k, hidden), dispatch before combine within a
# shape. num_tokens stays last: lookup relies on each selector's rules ascending in it.
_PHASE_ORDER = {"dispatch": 0, "combine": 1}


def _table_key(rule):
    return (
        rule["num_experts"],
        rule["topk"],
        rule["hidden_dim"],
        _PHASE_ORDER[rule["phase"]],
        rule["dtype"],
        rule["num_tokens"],
    )


def _parse_table_row(cells, kernel_type, where):
    """One rule from a row's cells: typed, empty cells left out, and checked
    against what this kernel family's lookup reads."""
    rule = {}
    for name in _TABLE_COLUMNS:
        text = cells.get(name)
        if text is None:
            raise ValueError(f"{where}: fewer cells than columns")
        text = text.strip()
        if not text:
            continue
        try:
            if name in _TABLE_INT_MINIMUM:
                value = int(text)
            elif name in _TABLE_BOOLS:
                if text not in ("0", "1"):
                    raise ValueError("want 0 or 1")
                value = text == "1"
            elif name in _TABLE_FLOATS:
                value = float(text)
            else:
                value = text
        except ValueError as exc:
            raise ValueError(f"{where}: {name}={text!r}: {exc}") from None
        rule[name] = value
    problems = [f"missing {name}" for name in _TABLE_REQUIRED if name not in rule]
    phase = rule.get("phase")
    if phase == "dispatch" and "combine_load_first" in rule:
        problems.append("combine_load_first is combine-only")
    elif phase not in (None, "dispatch", "combine"):
        problems.append(f"phase {phase!r} is neither dispatch nor combine")
    if "dtype" in rule and rule["dtype"] not in _TABLE_DTYPES:
        problems.append(f"unknown dtype {rule['dtype']!r}")
    if "combine_load_first" in rule and kernel_type != "InterNodeV2":
        problems.append("combine_load_first is read by InterNodeV2 only")
    if "active_qps" in rule and kernel_type == "IntraNodeV2":
        problems.append("active_qps is internode only")
    if kernel_type != "IntraNodeV2" and rule.get("rdma_block_num") == 0:
        # No block would run the RDMA leg, so the node-count signal would
        # never fire and the op would hang.
        problems.append("rdma_block_num must be at least 1 for an internode rule")
    problems += [
        f"{name}={rule[name]} is below {low}"
        for name, low in _TABLE_INT_MINIMUM.items()
        if name in rule and rule[name] < low
    ]
    if problems:
        raise ValueError(f"{where}: {'; '.join(problems)}")
    return rule


def read_v2_table(path, kernel_type):
    """Rules of one V2 table, sorted by selector. Anything malformed raises: a
    table that loaded wrong would run untuned geometry without a word."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if len(header) != len(set(header)) or set(header) != set(_TABLE_COLUMNS):
            raise ValueError(
                f"{path}: columns must be {','.join(_TABLE_COLUMNS)}, "
                f"got {','.join(header)}"
            )
        rules, first_line = [], {}
        for cells in reader:
            where = f"{path}:{reader.line_num}"
            if None in cells:
                raise ValueError(f"{where}: more cells than columns")
            rule = _parse_table_row(cells, kernel_type, where)
            key = _table_key(rule)
            if key in first_line:
                raise ValueError(f"{where}: same selector as line {first_line[key]}")
            first_line[key] = reader.line_num
            rules.append(rule)
    return tuple(sorted(rules, key=_table_key))


def _table_cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        # float(): repr of a NumPy scalar is "np.float64(...)", not a number.
        # A whole number drops ".0" so a cell rewrites as it was read.
        value = float(value)
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    return str(value)


def _checked_rule(rule, kernel_type, where):
    """(the rule as read_v2_table() will read it back, its cells)."""
    extra = sorted(set(rule) - set(_TABLE_COLUMNS))
    if extra:
        raise ValueError(f"{where}: fields the table has no column for: {extra}")
    cells = {name: _table_cell(rule.get(name)) for name in _TABLE_COLUMNS}
    return _parse_table_row(cells, kernel_type, where), cells


def write_v2_table(path, rules, kernel_type):
    """Write rules as one V2 table, sorted by selector, replacing the file
    atomically. Each rule is checked as read_v2_table() reads it, so a save
    cannot leave a table that then fails to load."""
    path = Path(path)
    rows = {}
    for rule in rules:
        selector = ",".join(_table_cell(rule.get(name)) for name in _TABLE_SELECTOR)
        where = f"{path}: rule {selector}"
        parsed, cells = _checked_rule(rule, kernel_type, where)
        key = _table_key(parsed)
        if key in rows:
            raise ValueError(f"{where}: selector appears twice")
        rows[key] = cells
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(_TABLE_COLUMNS)
    for key in sorted(rows):
        writer.writerow([rows[key][name] for name in _TABLE_COLUMNS])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.stem, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            f.write(text.getvalue())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    # Container sweeps may run as root; keep their output readable on the host.
    os.chmod(path, 0o644)


_V2_RULES_CACHE: dict = {}


def _v2_rules(gpu_arch, gpu_model, kernel_type, ep_size):
    """This model's own V2 table for one family and EP size; () when there is none."""
    path = internode_table_dir() / v2_table_filename(
        gpu_arch, gpu_model, kernel_type, ep_size
    )
    if path not in _V2_RULES_CACHE:
        _V2_RULES_CACHE[path] = (
            read_v2_table(path, kernel_type) if path.is_file() else ()
        )
    return _V2_RULES_CACHE[path]


def internode_buckets(
    world_size,
    hidden_dim,
    topk,
    dispatch_dtype,
    combine_dtype=None,
    kernel_family="v2",
    experts_per_rank=None,
    cu_count=None,
):
    """Return (token ceiling, dispatch geometry, combine geometry,
    combine load-first) rows. A geometry is (block, rdma, warp, active_qps),
    clamped to ``cu_count`` (fit_internode_geometry); active_qps is the
    rule's optional count, None when the rule names none.

    Each phase matches its own dtype and stored width (packed for FP4 dispatch).
    A rule must match every selector exactly -- total experts (world_size *
    experts_per_rank), top-k, dtype and width -- so no shape borrows another's.
    An untuned phase is None, and a wholly untuned shape returns None.
    A combine rule's optional ``combine_load_first`` (default false) selects
    the load-first gather for its bucket.
    """
    if topk is None or experts_per_rank is None:
        return None

    from mori.ops.tuning_config import DTYPE_TO_CONFIG_STR

    kernel_type = _V2_KERNEL_TYPES.get(kernel_family)
    if kernel_type is None:
        return None
    model, arch = _device_key(), _gpu.arch_name()
    if model is None or arch is None:
        return None
    num_experts = world_size * experts_per_rank
    dtypes = {
        "dispatch": dispatch_dtype,
        "combine": combine_dtype if combine_dtype is not None else dispatch_dtype,
    }
    packed_dispatch = DTYPE_TO_CONFIG_STR.get(dispatch_dtype) == "fp4"
    if packed_dispatch and hidden_dim % 2:
        return None
    hidden_dims = {
        "dispatch": hidden_dim // 2 if packed_dispatch else hidden_dim,
        "combine": hidden_dim,
    }
    rules = _v2_rules(arch, model, kernel_type, world_size)
    # Each phase's matches as (ceiling, rule) buckets. The table is sorted by
    # selector, so they ascend in num_tokens.
    buckets = {
        phase: [
            (rule["num_tokens"], rule)
            for rule in rules
            if rule["phase"] == phase
            and rule["num_experts"] == num_experts
            and rule["topk"] == topk
            and rule["dtype"] == DTYPE_TO_CONFIG_STR.get(dtypes[phase])
            and rule["hidden_dim"] == hidden_dims[phase]
        ]
        for phase in _INTERNODE_PHASES
    }
    # Every rule has a ceiling, so the open-ended tail is not a row of its own.
    ceilings = _ceilings(*buckets.values())[:-1]
    if not ceilings:
        return None

    table = []
    for num_tokens in ceilings:
        selected = [
            (pick_bucket(buckets[phase], num_tokens) or (None, None))[1]
            for phase in _INTERNODE_PHASES
        ]
        geometries = [
            fit_internode_geometry(
                (
                    (
                        rule["block_num"],
                        rule["rdma_block_num"],
                        rule["warp_per_block"],
                        rule.get("active_qps"),
                    )
                    if rule is not None
                    else None
                ),
                clamp_to_cu=True,
                cu_count=cu_count,
            )
            for rule in selected
        ]
        combine = selected[1]
        load_first = bool(combine is not None and combine.get("combine_load_first"))
        table.append((num_tokens, *geometries, load_first))
    return table


def intranode_buckets(
    world_size,
    hidden_dim,
    topk,
    dispatch_dtype,
    combine_dtype=None,
    experts_per_rank=None,
):
    """(dispatch, combine) bucket lists from the IntraNodeV2 table.

    Each is ((max_tok_inclusive, block, warp), ...) ascending, or None when that
    phase has no rule. Matching is internode_buckets', so dtype, top-k and the
    expert count must all be exact; intranode has no RDMA split to keep.
    """
    rows = internode_buckets(
        world_size,
        hidden_dim,
        topk,
        dispatch_dtype,
        combine_dtype,
        kernel_family="intra",
        experts_per_rank=experts_per_rank,
    )
    if not rows:
        return None, None
    phases = []
    for index in (1, 2):
        buckets = tuple(
            (row[0], row[index][0], row[index][2])
            for row in rows
            if row[index] is not None
        )
        phases.append(buckets or None)
    return phases[0], phases[1]


def lookup_internode(
    world_size,
    hidden_dim,
    topk,
    num_tokens,
    dispatch_dtype,
    combine_dtype=None,
    kernel_family="v2",
    experts_per_rank=None,
    cu_count=None,
):
    """Resolve one token count; backend construction uses internode_buckets."""
    table = internode_buckets(
        world_size,
        hidden_dim,
        topk,
        dispatch_dtype,
        combine_dtype,
        kernel_family,
        experts_per_rank,
        cu_count=cu_count,
    )
    row = pick_bucket(table, num_tokens) if table else None
    if row is None or (row[1] is None and row[2] is None):
        return None
    return {"dispatch": row[1], "combine": row[2], "combine_load_first": row[3]}


def save_internode_result(
    world_size,
    phase,
    entry,
    *,
    kernel_family="v2",
    directory=None,
    path=None,
):
    """Merge one rule into the table v2_table_filename() names for this GPU
    model, kernel family and EP size, replacing the rule with the same selector,
    then invalidate the cache. The rule's total experts is world_size *
    entry["experts_per_rank"]; the table stores the total."""
    from mori.ops.tuning_config import _report_rule_added, _report_rule_change

    if phase not in _INTERNODE_PHASES:
        raise ValueError(f"phase must be one of {_INTERNODE_PHASES}, got {phase!r}")
    kernel_type = _V2_KERNEL_TYPES.get(kernel_family)
    if kernel_type is None:
        raise ValueError(f"unsupported V2 kernel family: {kernel_family!r}")
    entry = dict(entry)
    for name in ("topk", "experts_per_rank"):
        value = entry.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"V2 internode tuning requires a positive integer {name}")
    entry["num_experts"] = world_size * entry.pop("experts_per_rank")
    entry["phase"] = phase
    if path is None:
        model, arch = _device_key(), _gpu.arch_name()
        if model is None or arch is None:
            raise RuntimeError(
                "cannot save internode tuning for an unknown GPU; pass an explicit path"
            )
        path = Path(directory or internode_table_dir()) / v2_table_filename(
            arch, model, kernel_type, world_size
        )
    else:
        path = Path(path)
    entry, _ = _checked_rule(entry, kernel_type, f"{path}: new {phase} rule")
    rules = list(read_v2_table(path, kernel_type)) if path.is_file() else []
    key = _table_key(entry)
    for index, rule in enumerate(rules):
        if _table_key(rule) == key:
            _report_rule_change(phase, key, rule, entry)
            rules[index] = entry
            break
    else:
        _report_rule_added(phase, key, entry)
        rules.append(entry)
    write_v2_table(path, rules, kernel_type)
    _V2_RULES_CACHE.clear()
    return path
