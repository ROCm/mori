#!/usr/bin/env python3
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
"""Render any of the benchmark JSONL files into tables.

One reader for every producer, because the two it replaced each knew about one
of them and dropped whatever it did not recognise.

Two rules it keeps that the originals did not:

* **The grouping key is every configuration field present**, not a hand-listed
  few. `report_fused.py` keyed on `(label, world_size, m)` and stored by mode,
  so a sweep that varied `gather_transport` or `fuse_quantize` -- which is
  exactly what `sweep.py fused-wire` does -- had every such run overwrite the
  one before it, silently, and the table showed the last one as if it were the
  only one.
* **Nothing is dropped for being unrecognised.** A label or an impl this file
  has never seen still gets a row. The old `show(labels, ...)` matched a fixed
  list and printed nothing at all for a `fp8gather` sweep.

    python report.py gemm.jsonl
    python report.py fused.jsonl --col cold_us
    python report.py gemm-full.jsonl --by impl --baseline sglang
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

#: Fields that identify *a measurement* rather than a configuration. Everything
#: else in the row is part of the key, which is what keeps distinct runs
#: distinct without this file having to know the axes in advance.
MEASURED = {
    "hot_us",
    "cold_us",
    "us",
    "max_rank_time_us",
    "rel_l2",
    "validated",
    "copies",
    "cold_reps",
    "working_set_mb",
    "vram_before",
    "vram_after",
    "error",
    "timing",
    "route",
    "supported",
    "reason",
    # Resolved by the op, not chosen by the caller: `critical_rank` is elected
    # at runtime and `resolved_tile_order` is what `--tile-order auto` became.
    # They are never sweep axes, so they can be excluded even when a file
    # predates `sweep_axes`.
    "critical_rank",
    "resolved_tile_order",
}
#: Shown as the row label when present, in this order; the rest go in the key.
ROW_KEYS = ("label", "shape", "n", "k", "world_size", "quant", "scope")
#: Preferred column axis, first one present wins.
COL_KEYS = ("impl", "mode", "config")


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("RESULT_JSON"):
                    line = line.split(" ", 1)[1]
                rows.append(json.loads(line))
    return rows


def time_of(row, col):
    for k in (col, "cold_us", "us", "max_rank_time_us", "hot_us"):
        if k in row and isinstance(row[k], (int, float)):
            return row[k]
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument(
        "--col", default="cold_us", help="which timing field to table (default cold_us)"
    )
    p.add_argument(
        "--by",
        default=None,
        help="column axis; default is the first of " + ", ".join(COL_KEYS),
    )
    p.add_argument(
        "--baseline",
        default=None,
        help="column to show the others as a percentage against",
    )
    p.add_argument("--keep-invalid", action="store_true")
    args = p.parse_args()

    rows = load(args.paths)
    if not rows:
        print("no rows")
        return 1

    bad = [r for r in rows if r.get("validated") is False]
    if bad and not args.keep_invalid:
        rows = [r for r in rows if r.get("validated") is not False]
        print(
            f"!! {len(bad)} row(s) failed validation, excluded "
            f"(--keep-invalid to see them)"
        )
        for r in bad[:5]:
            who = r.get("impl") or r.get("mode") or "?"
            print(
                f"     {r.get('shape','?')} M={r.get('m','?')} {who}: "
                f"{r.get('error') or f'rel_l2={r.get("rel_l2")}'}"
            )
    if not rows:
        print("nothing left after validation filter")
        return 1

    col_key = args.by
    if col_key is None:
        for c in COL_KEYS:
            if any(c in r for r in rows):
                col_key = c
                break
    if col_key is None:
        print("no column axis found; pass --by")
        return 1

    # Key on what the sweep varied when it said so, and on everything
    # configuration-ish otherwise. The distinction matters: bench_gemm_ar.py
    # echoes knobs it resolved for itself -- `chunks`, `critical_rank`, the
    # tile order -- and keying on those turns one matrix into one table per
    # point. `sweep_axes` is how sweep.py says which fields were axes.
    def keyof(r):
        axes = r.get("sweep_axes")
        if axes is not None:
            keep = [k for k in axes if k not in (col_key, "m")]
            keep += [
                k
                for k in ("label", "shape", "n", "k", "world_size", "quant", "scope")
                if k in r
            ]
            return tuple((k, r[k]) for k in sorted(set(keep)))
        return tuple(
            (k, r[k])
            for k in sorted(r)
            if k not in MEASURED
            and k not in (col_key, "m", "bench", "sweep_axes")
            and not isinstance(r[k], (dict, list))
        )

    table = defaultdict(lambda: defaultdict(list))
    for r in rows:
        t = time_of(r, args.col)
        if t is not None:
            table[keyof(r)][(r.get("m"), r.get(col_key))].append(t)

    for key, cells in sorted(table.items(), key=lambda kv: str(kv[0])):
        d = dict(key)
        head = "  ".join(f"{k}={d[k]}" for k in ROW_KEYS if k in d)
        rest = "  ".join(
            f"{k}={v}" for k, v in key if k not in ROW_KEYS and v not in (None, "")
        )
        print(f"\n### {head}" + (f"   [{rest}]" if rest else ""))
        ms = sorted({m for m, _ in cells if m is not None})
        cols = sorted({c for _, c in cells if c is not None}, key=str)
        if not cols:
            continue
        base = args.baseline if args.baseline in cols else None

        print(
            f"{'M':>8}"
            + "".join(f"{str(c):>16}" for c in cols)
            + ("   (vs " + base + ")" if base else "")
        )
        for m in ms:
            line = f"{m:>8}"
            bt = None
            if base:
                vs = cells.get((m, base))
                bt = statistics.median(vs) if vs else None
            for c in cols:
                vs = cells.get((m, c))
                if not vs:
                    line += f"{'-':>16}"
                    continue
                t = statistics.median(vs)
                spread = f"*{len(vs)}" if len(vs) > 1 else ""
                if bt and c != base:
                    line += f"{t:9.1f}{(t / bt - 1) * 100:+6.0f}%"
                else:
                    line += f"{t:>12.1f}{spread:>4}"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
