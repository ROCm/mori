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
The data sources retain their existing formats: intra lookup() uses Python
tables; internode_buckets() reads JSON through mori.ops.tuning_config.
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
        (4, 7168, 6, None): ((None, 64, 8),),
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


class ScheduleEntry(NamedTuple):
    max_tokens: int | None
    dispatch: KernelConfig
    combine: KernelConfig


def resolve_schedule(cfg) -> tuple[ScheduleEntry, ...]:
    """The HIP V2 schedule consumed by both construction and token selection.

    Intra adapts the Python-table/explicit schedule resolved by Config. Inter
    resolves JSON and overrides, then includes the family crossover in the
    schedule. Loading and selection never depend on the declared token capacity.
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
        _, dispatch, combine = pick_bucket(geometries, probe)
        entries.append(
            ScheduleEntry(
                ceiling,
                KernelConfig(
                    family,
                    block_num=dispatch[0],
                    warp_per_block=dispatch[2],
                    rdma_block_num=dispatch[1],
                ),
                KernelConfig(
                    family,
                    block_num=combine[0],
                    warp_per_block=combine[2],
                    rdma_block_num=combine[1],
                ),
            )
        )
    return tuple(entries)


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


def lookup(world_size, hidden_dim, topk, dtype="bf16", experts_per_rank=None) -> dict:
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


_INTERNODE_TABLE_DIR = Path(__file__).parent / "tuning_configs"
_INTERNODE_KERNEL_TYPES = {"v2": "InterNodeV2", "v2_ll": "InterNodeV2LL"}
_INTERNODE_PHASES = ("dispatch", "combine")
_INTERNODE_GEOMETRY_FIELDS = (
    ("dispatch_block_num", "dispatch_rdma_block_num", "warp_num_per_block"),
    ("combine_block_num", "combine_rdma_block_num", "combine_warp_num_per_block"),
)

# Legacy MI308X FP8-dispatch/BF16-combine measurements. Preserve the historical
# fallback across dtypes and families when JSON has no rules for the shape.
# Rows: token ceiling, dispatch B/R/W, combine B/R/W.
_LEGACY_INTERNODE_TABLE = {
    ("mi308x", 16, 6144, 8): (
        (4, 32, 16, 4, 32, 21, 6),
        (8, 64, 32, 8, 32, 21, 6),
        (16, 80, 40, 4, 80, 40, 4),
        (None, 80, 48, 8, 64, 48, 6),
    ),
    ("mi308x", 16, 7168, 8): (
        (4, 64, 42, 8, 32, 16, 16),
        (8, 32, 21, 16, 32, 16, 4),
        (16, 80, 40, 4, 80, 40, 4),
        (None, 80, 48, 8, 64, 48, 6),
    ),
}


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
    """Leave at least one intra-node block; optionally cap tuned grids to CUs."""
    if geometry is None:
        return None
    block, rdma, warp = geometry
    if clamp_to_cu:
        block = min(block, _gpu.cu_count() or 80)
    return (block, min(rdma, max(1, block - 1)), warp)


def _geometry_from_env(name):
    """Optional sweep override, read once when the op is built: B,R,W."""
    raw = os.environ.get(name)
    if not raw:
        return None
    geometry = tuple(int(x) for x in raw.replace(" ", "").split(","))
    if len(geometry) != 3:
        raise ValueError(f"{name}={raw!r}: want three ints, block,rdma,warp")
    return geometry


def internode_schedule(cfg):
    """Resolve (inclusive token ceiling, dispatch B/R/W, combine B/R/W).

    Either sweep env override bypasses both tables. Otherwise explicit config
    fields override the selected family's table, with config defaults for an
    untuned phase. Only table values are CU-clamped; all paths keep R < B.
    """
    defaults = [
        tuple(getattr(cfg, name) for name in fields)
        for fields in _INTERNODE_GEOMETRY_FIELDS
    ]
    overrides = [
        _geometry_from_env("MORI_EP_DISP_GEOM"),
        _geometry_from_env("MORI_EP_COMB_GEOM"),
    ]
    if any(overrides):
        return [
            (
                None,
                *(fit_internode_geometry(g or d) for g, d in zip(overrides, defaults)),
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
    if cfg.internode_kernel == "auto" and cfg.internode_auto_ll_max_tokens >= 1:
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
            geometry = tuned[index] if tuned and tuned[index] else fallback
            geometry = tuple(
                getattr(cfg, name) if name in pinned else value
                for name, value in zip(fields, geometry)
            )
            geometries.append(fit_internode_geometry(geometry))
        row = (ceiling, *geometries)
        # Equal geometry can span a crossover here. resolve_schedule() restores
        # the family boundary when producing the executable configurations.
        if rows and rows[-1][1:] == row[1:]:
            rows[-1] = row
        else:
            rows.append(row)
    return rows


def internode_table_dir():
    """V2's explicit directory also isolates it from MORI_EP_TUNING_CONFIG."""
    return Path(os.environ.get("MORI_EP_V2_TUNING_DIR") or _INTERNODE_TABLE_DIR)


def _legacy_internode_buckets(world_size, hidden_dim, topk):
    schedule = _LEGACY_INTERNODE_TABLE.get(
        (_gpu.detect_model(), world_size, hidden_dim, topk)
    )
    if not schedule:
        return None
    return [
        (
            row[0],
            fit_internode_geometry(row[1:4], clamp_to_cu=True),
            fit_internode_geometry(row[4:7], clamp_to_cu=True),
        )
        for row in schedule
    ]


def internode_buckets(
    world_size,
    hidden_dim,
    topk,
    dispatch_dtype,
    combine_dtype=None,
    kernel_family="v2",
    experts_per_rank=None,
):
    """Return (token ceiling, dispatch geometry, combine geometry) rows.

    Each phase matches its own dtype. Exact expert counts override wildcards
    only at the same token ceiling; other ceilings keep their wildcard rows.
    An untuned phase is None, and a wholly untuned shape returns None.
    """
    from mori.ops.tuning_config import DTYPE_TO_CONFIG_STR, TuningConfigManager

    kernel_type = _INTERNODE_KERNEL_TYPES.get(kernel_family)
    if kernel_type is None:
        return None
    model, arch = _gpu.detect_model(), _gpu.arch_name()
    if model is None or arch is None:
        return _legacy_internode_buckets(world_size, hidden_dim, topk)
    manager = TuningConfigManager.get_instance(
        arch, kernel_type, world_size, model, directory=internode_table_dir()
    )
    dtypes = {
        "dispatch": dispatch_dtype,
        "combine": combine_dtype if combine_dtype is not None else dispatch_dtype,
    }
    phase_rules = {
        "dispatch": manager.dispatch_rules,
        "combine": manager.combine_rules,
    }
    edges, matched = set(), {}
    for phase, rules in phase_rules.items():
        dtype_str = DTYPE_TO_CONFIG_STR.get(dtypes[phase])
        # The shared loader permits broader shape fallback. Restrict its input
        # here so internode never borrows another hidden dimension or top-k.
        rules = [
            rule
            for rule in rules
            if rule["dtype"] == dtype_str
            and rule["hidden_dim"] == hidden_dim
            and rule.get("topk") in (None, topk)
            and (
                experts_per_rank is None
                or rule.get("experts_per_rank") in (None, experts_per_rank)
            )
        ]
        exact = {
            rule["num_tokens"]
            for rule in rules
            if rule.get("experts_per_rank") is not None
        }
        matched[phase] = [
            rule
            for rule in rules
            if rule.get("experts_per_rank") is not None
            or rule["num_tokens"] not in exact
        ]
        edges.update(rule["num_tokens"] for rule in matched[phase])
    if not edges:
        return _legacy_internode_buckets(world_size, hidden_dim, topk)

    table = []
    for num_tokens in sorted(edges):
        geometries = []
        for phase in _INTERNODE_PHASES:
            filters = (
                {"zero_copy": False, "quant_type": "none"} if phase == "combine" else {}
            )
            params = TuningConfigManager.lookup(
                matched[phase],
                dtype=dtypes[phase],
                num_tokens=num_tokens,
                hidden_dim=hidden_dim,
                topk=topk,
                **filters,
            )
            geometry = (
                (params.block_num, params.rdma_block_num, params.warp_per_block)
                if params is not None
                else None
            )
            geometries.append(fit_internode_geometry(geometry, clamp_to_cu=True))
        table.append((num_tokens, *geometries))
    return table


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
    return {"dispatch": row[1], "combine": row[2]}


def save_internode_result(
    world_size,
    phase,
    entry,
    *,
    kernel_family="v2",
    directory=None,
    path=None,
):
    """Merge one phase's rule with the shared JSON writer and invalidate cache."""
    from mori.ops.tuning_config import TuningConfigManager, build_config_filename

    if phase not in _INTERNODE_PHASES:
        raise ValueError(f"phase must be one of {_INTERNODE_PHASES}, got {phase!r}")
    kernel_type = _INTERNODE_KERNEL_TYPES.get(kernel_family)
    if kernel_type is None:
        raise ValueError(f"unsupported internode kernel family: {kernel_family!r}")
    model, arch = _gpu.detect_model(), _gpu.arch_name()
    if path is None:
        if model is None or arch is None:
            raise RuntimeError(
                "cannot save internode tuning for an unknown GPU; pass an explicit path"
            )
        path = Path(directory or internode_table_dir()) / build_config_filename(
            arch, kernel_type, world_size, model, phase
        )
    else:
        path = Path(path)
    entry = dict(entry)
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
            phase=phase,
        ),
        entry,
        phase=phase,
    )
    # Container sweeps may run as root; keep their output readable on the host.
    os.chmod(path, 0o644)
    TuningConfigManager._cache.clear()
    return path
