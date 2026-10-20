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
Intra lookup() reads the IntraNodeV2 JSON tables first and falls back to the
Python tables per phase; internode_buckets() reads the JSON table named by
v2_table_filename() for its shape, validated by mori.ops.tuning_config.
"""

from __future__ import annotations

import os
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
        # EP16 on four 4-GPU hosts, topk 6, fp4 dispatch, 2026-10-11. 128x16 is the only geometry
        # that builds the pinned-destination dispatch (kDPin in ep_intranode_1250x.hpp). 128x16
        # minus 64x16, dispatch us, graph, one M=16384 plan, 3 alternating rounds: 1024 tokens a
        # rank -11.0..-11.8, 2048 +2.2..+3.3, 3072 -6.0..-6.8, 8192 -14.7..-15.3, 16384
        # -80.9..-83.3. 2048 is the only point 64x16 wins; the edge between them is unswept.
        (16, 7168, 6, None): {
            None: ((None, 64, 16),),
            "fp4": ((4095, 64, 16), (None, 128, 16)),
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
        # topk 9 = 8 routed + 1 shared. Also the combine half of the topk-9 dispatch
        # schedule: lookup() returns the single-shot default unless both halves exist.
        (4, 7168, 9, None): (
            (64, 64, 4),
            (128, 64, 8),
            (256, 128, 8),
            (1024, 256, 8),
            (None, 192, 8),
        ),
        (4, 7168, 6, None): ((None, 64, 8),),
        # The single-shot default, listed so the EP8 and EP16 dispatch schedules have a combine half.
        (8, 7168, 6, None): ((None, 64, 8),),
        (16, 7168, 6, None): ((None, 64, 8),),
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


class KernelConfig(NamedTuple):
    """One HIP kernel choice; intra has no RDMA split."""

    family: str
    block_num: int
    warp_per_block: int
    rdma_block_num: int | None = None
    # Ordinary-v2 combine only: compile the load-first gather (JSON
    # combine_load_first). Part of the key, so buckets that share a geometry
    # but differ here get separate kernels.
    load_first: bool = False
    # Internode only: how many of the allocated QPs (cfg.num_qp_per_pe) this
    # kernel sends on. A compile-time constant of each kernel, chosen per phase
    # and per bucket (JSON active_qps), so it is part of the key too.
    active_qps: int | None = None


class ScheduleEntry(NamedTuple):
    max_tokens: int | None
    dispatch: KernelConfig
    combine: KernelConfig


def resolve_schedule(cfg) -> tuple[ScheduleEntry, ...]:
    """The HIP V2 schedule consumed by both construction and token selection.

    Intra adapts the Python-table/explicit schedule resolved by Config. Inter
    resolves JSON and overrides, then includes the family crossover in the
    schedule. Loading and selection never depend on the declared token capacity;
    that includes the combine load-first choice, which comes from each bucket's
    JSON rule (combine_load_first).
    """
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

    geometries = internode_schedule(cfg)
    edges = {row[0] for row in geometries if row[0] is not None}
    if cfg.internode_kernel == "auto":
        # Keep a zero-token LL entry when the crossover is zero, too.
        edges.add(cfg.internode_auto_ll_max_tokens)
    entries = []
    for ceiling in [*sorted(edges), None]:
        probe = ceiling if ceiling is not None else max(edges, default=0) + 1
        family = internode_kernel_family(cfg, probe)
        _, dispatch, combine, load_first = pick_bucket(geometries, probe)
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
    """Multi-line summary of a resolved schedule and the tables it came from."""

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
            lines.append(f"  tables bypassed by {', '.join(overrides)}")
        lines.append(
            f"  QPs allocated per peer: {cfg.num_qp_per_pe}"
            + (f", active pinned to {cfg.active_qps}" if cfg.active_qps else "")
        )
        pinned = sorted(getattr(cfg, "_pinned_geometry", ()))
        if pinned:
            lines.append(f"  pinned fields: {', '.join(pinned)}")
    else:
        families = [_V2_KERNEL_TYPES["intra"]]
    for kernel_type in families:
        for phase in _INTERNODE_PHASES:
            path = internode_table_dir() / v2_table_filename(
                _gpu.arch_name(),
                _gpu.detect_model(),
                kernel_type,
                cfg.world_size,
                num_experts,
                cfg.num_experts_per_token,
                phase,
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
    edges = sorted(
        {b[0] for b in disp if b[0] is not None}
        | {b[0] for b in comb if b[0] is not None}
    ) + [None]

    return tuple(
        (edge,) + pick_bucket(disp, edge)[1:] + pick_bucket(comb, edge)[1:]
        for edge in edges
    )


# FlyDSL's dtype_str spelling, still accepted from callers that pass it (aiter's MegaMoE).
_DISPATCH_DTYPE_ALIAS = {"fp4_disp_bf16_comb": "fp4"}


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
    half; ``quant_type`` picks the combine half. An unswept shape gets the HIP
    single-shot default (schedule=None). A swept one gets a per-token-count schedule
    built from the dispatch and combine tables independently, so either half can be
    re-tuned without touching the other.

    Given both legs' dtypes, the IntraNodeV2 JSON tables come first. Each phase
    takes its JSON rules when it has any and the Python table otherwise, so the
    JSON can be filled one phase at a time.
    """
    base = _hip_default()
    dev = _device_key()
    disp = _bucket_key(
        _DISPATCH_TABLE.get(dev, {}), world_size, hidden_dim, topk, experts_per_rank
    )
    if disp is not None:
        # None is the "every dtype measured the same" key; an exact dtype overrides it.
        disp = disp.get(_DISPATCH_DTYPE_ALIAS.get(dtype, dtype)) or disp.get(None)
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
    if dispatch_dtype is not None and quant_type == "none":
        json_disp, json_comb = intranode_buckets(
            world_size,
            hidden_dim,
            topk,
            dispatch_dtype,
            combine_dtype,
            experts_per_rank=experts_per_rank,
        )
        if json_disp or json_comb:
            disp = json_disp or disp
            comb = json_comb or comb
            disp = disp or (
                (None, base["dispatch_block_num"], base["warp_num_per_block"]),
            )
            comb = comb or (
                (None, base["combine_block_num"], base["combine_warp_num_per_block"]),
            )
            base["schedule"] = _merge(disp, comb)
            return base
    if disp is None or comb is None:
        return base  # half a schedule is not a schedule
    base["schedule"] = _merge(disp, comb)
    return base


_INTERNODE_TABLE_DIR = Path(__file__).parent / "tuning_configs"
# JSON kernel_type per table family. "intra" is the single-node HIP op, which
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


def fit_internode_geometry(geometry, *, clamp_to_cu=False):
    """Leave at least one intra-node block; optionally cap tuned grids to CUs.
    Fields after block, rdma, warp (the active QP count) pass through."""
    if geometry is None:
        return None
    block, rdma, warp, *rest = geometry
    if clamp_to_cu:
        block = min(block, _gpu.cu_count() or 80)
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


def internode_schedule(cfg):
    """Resolve (inclusive token ceiling, dispatch B/R/W/QPs, combine B/R/W/QPs,
    combine load-first).

    Either sweep env override bypasses both tables. Otherwise explicit config
    fields override the selected family's table, with config defaults for an
    untuned phase. Table values and unpinned defaults are CU-clamped; pinned
    fields are kept as given. All paths keep R < B.
    """
    defaults = [
        tuple(getattr(cfg, name) for name in fields) + (resolve_active_qps(cfg),)
        for fields in _INTERNODE_GEOMETRY_FIELDS
    ]
    overrides = [
        _geometry_from_env("MORI_EP_DISP_GEOM", cfg),
        _geometry_from_env("MORI_EP_COMB_GEOM", cfg),
    ]
    if any(overrides):
        return [
            (
                None,
                *(fit_internode_geometry(g or d) for g, d in zip(overrides, defaults)),
                False,
            )
        ]

    tables = {
        family: internode_buckets(
            cfg.world_size,
            cfg.hidden_dim,
            cfg.num_experts_per_token,
            cfg.dispatch_dtype,
            cfg.combine_dtype,
            kernel_family=family,
            experts_per_rank=cfg.num_experts_per_rank,
        )
        or []
        for family in internode_kernel_families(cfg)
    }
    edges = {row[0] for table in tables.values() for row in table if row[0] is not None}
    if cfg.internode_kernel == "auto":
        edges.add(cfg.internode_auto_ll_max_tokens)

    pinned = getattr(cfg, "_pinned_geometry", frozenset())
    rows = []
    for ceiling in [*sorted(edges), None]:
        # The terminal bucket uses the family selected past every finite edge.
        probe = ceiling if ceiling is not None else max(edges, default=0) + 1
        family = internode_kernel_family(cfg, probe)
        tuned = pick_bucket(tables[family], probe)
        geometries = []
        for index, (fields, fallback) in enumerate(
            zip(_INTERNODE_GEOMETRY_FIELDS, defaults), start=1
        ):
            geometry = (
                tuned[index]
                if tuned and tuned[index]
                else fit_internode_geometry(fallback, clamp_to_cu=True)
            )
            geometry = tuple(
                getattr(cfg, name) if name in pinned else value
                for name, value in zip(fields, geometry[:3])
            ) + (
                resolve_active_qps(cfg, geometry[3]),
            )
            geometries.append(fit_internode_geometry(geometry))
        # The load-first gather exists in the ordinary combine only.
        load_first = bool(tuned and tuned[3]) and family == "v2"
        row = (ceiling, *geometries, load_first)
        # Equal geometry can span a crossover here. resolve_schedule() restores
        # the family boundary when producing the executable configurations.
        if rows and rows[-1][1:] == row[1:]:
            rows[-1] = row
        else:
            rows.append(row)
    return rows


def internode_table_dir():
    """V2 reads only this directory; V1's MORI_EP_TUNING_CONFIG never applies."""
    return Path(os.environ.get("MORI_EP_V2_TUNING_DIR") or _INTERNODE_TABLE_DIR)


def v2_table_filename(
    gpu_arch, gpu_model, kernel_type, ep_size, num_experts, topk, phase
):
    """One V2 table per model shape and phase, e.g.
    ``gfx950_mi355x_InterNodeV2_ep16_expert256_top8_dispatch.json``."""
    return (
        f"{gpu_arch}_{gpu_model}_{kernel_type}_ep{ep_size}_"
        f"expert{num_experts}_top{topk}_{phase}.json"
    )


_V2_RULES_CACHE: dict = {}


def _v2_phase_rules(
    gpu_arch, gpu_model, kernel_type, ep_size, num_experts, topk, phase
):
    """Validated, sorted rules of this model's own V2 table; [] when there is none."""
    from mori.ops.tuning_config import (
        _COMBINE_RULE_REQUIRED,
        _DISPATCH_RULE_REQUIRED,
        TuningConfigManager,
    )

    path = internode_table_dir() / v2_table_filename(
        gpu_arch, gpu_model, kernel_type, ep_size, num_experts, topk, phase
    )
    if path not in _V2_RULES_CACHE:
        required = (
            _DISPATCH_RULE_REQUIRED if phase == "dispatch" else _COMBINE_RULE_REQUIRED
        )
        _V2_RULES_CACHE[path] = TuningConfigManager._load_phase_rules(
            path, gpu_arch, phase, required
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
):
    """Return (token ceiling, dispatch geometry, combine geometry,
    combine load-first) rows. A geometry is (block, rdma, warp, active_qps);
    active_qps is the rule's optional count, None when the rule names none.

    Each phase matches its own dtype and stored width (packed for FP4 dispatch).
    Total experts (world_size * experts_per_rank) and top-k name the table file
    (v2_table_filename), so a file only holds its own expert count. Rules still
    record top-k, and a rule without it is not a wildcard.
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
    model, arch = _gpu.detect_model(), _gpu.arch_name()
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
    phase_rules = {
        phase: _v2_phase_rules(
            arch, model, kernel_type, world_size, num_experts, topk, phase
        )
        for phase in _INTERNODE_PHASES
    }
    edges, matched = set(), {}
    for phase, rules in phase_rules.items():
        dtype_str = DTYPE_TO_CONFIG_STR.get(dtypes[phase])
        # The shared loader permits broader shape fallback. Restrict its input
        # here so V2 never borrows another hidden dimension or top-k shape.
        matched[phase] = [
            rule
            for rule in rules
            if rule["dtype"] == dtype_str
            and rule["hidden_dim"] == hidden_dims[phase]
            and rule.get("topk") == topk
            and (
                phase == "dispatch"
                or (not rule["zero_copy"] and rule["quant_type"] == "none")
            )
        ]
        edges.update(rule["num_tokens"] for rule in matched[phase])
    if not edges:
        return None

    table = []
    for num_tokens in sorted(edges):
        selected = {}
        for phase in _INTERNODE_PHASES:
            # Rules are sorted by num_tokens: the tightest ceiling, else the last.
            rules = matched[phase]
            selected[phase] = next(
                (rule for rule in rules if num_tokens <= rule["num_tokens"]),
                rules[-1] if rules else None,
            )
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
            )
            for rule in selected.values()
        ]
        combine = selected["combine"]
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
    """(dispatch, combine) bucket lists from the IntraNodeV2 JSON tables.

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
    """Save one rule into the table v2_table_filename names for its total experts
    (world_size * entry["experts_per_rank"]) and top-k, then invalidate the cache.
    experts_per_rank only picks the file; the stored rule does not repeat it."""
    from mori.ops.tuning_config import TuningConfigManager

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
    model, arch = _gpu.detect_model(), _gpu.arch_name()
    num_experts = world_size * entry.pop("experts_per_rank")
    if path is None:
        if model is None or arch is None:
            raise RuntimeError(
                "cannot save internode tuning for an unknown GPU; pass an explicit path"
            )
        path = Path(directory or internode_table_dir()) / v2_table_filename(
            arch, model, kernel_type, world_size, num_experts, entry["topk"], phase
        )
    else:
        path = Path(path)
    if phase == "combine":
        entry.setdefault("zero_copy", False)
        entry.setdefault("quant_type", "none")
    TuningConfigManager.save_tuning_result(
        path,
        dict(
            gpu_arch=arch,
            gpu_model=model,
            kernel_type=kernel_type,
            ep_size=world_size,
            num_experts=num_experts,
            topk=entry["topk"],
            phase=phase,
        ),
        entry,
        phase=phase,
    )
    # Container sweeps may run as root; keep their output readable on the host.
    os.chmod(path, 0o644)
    _V2_RULES_CACHE.clear()
    return path
