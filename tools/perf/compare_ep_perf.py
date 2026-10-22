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
"""Compare two internode EP perf JSONL files (base vs head) as a Markdown table.

Both inputs are records written by :mod:`tests.python.perf_report` (``MORI_PERF_OUT``)
from ``--cmd bench`` runs. Rows are matched on the run configuration; the last record
wins when a configuration appears more than once. Stdlib only.
"""

from __future__ import annotations

import argparse
import json

_PARAM_KEYS = (
    "kernel_type",
    "max_tokens",
    "world_size",
    "dtype",
    "combine_dtype",
    "quant_type",
    "num_qp",
    "hidden_dim",
)

# (metric, column title, True if higher is better)
_METRICS = (
    ("dispatch_lat_us", "Dispatch lat (us)", False),
    ("combine_lat_us", "Combine lat (us)", False),
    ("dispatch_rdma_bw_gbps", "Dispatch RDMA BW (GB/s)", True),
    ("combine_rdma_bw_gbps", "Combine RDMA BW (GB/s)", True),
)


def load(path):
    rows = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("category") != "internode_ep":
                    continue
                p = rec.get("params", {})
                rows[tuple(p.get(k) for k in _PARAM_KEYS)] = rec.get("metrics", {})
    except FileNotFoundError:
        pass
    return rows


def _cell(base, head, higher_is_better):
    if base is None or head is None:
        return "n/a", ""
    if not base:
        return f"{base:.2f} → {head:.2f}", ""
    delta = (head - base) / base * 100.0
    better = delta > 0 if higher_is_better else delta < 0
    sign = "+" if delta >= 0 else ""
    mark = "" if abs(delta) < 3.0 else (" (better)" if better else " (worse)")
    return f"{base:.2f} → {head:.2f}", f"{sign}{delta:.1f}%{mark}"


def render(base_rows, head_rows, base_label, head_label):
    keys = sorted(
        set(base_rows) | set(head_rows),
        key=lambda k: (str(k[0]), k[1] if isinstance(k[1], int) else 0, str(k[2:])),
    )
    out = [
        f"### MORI-EP internode perf: `{base_label}` (base) → `{head_label}` (PR)",
        "",
        "Single bench pass per side on the same nodes; |Δ| < 3% is left unmarked as noise.",
        "",
    ]
    header = ["Kernel", "Tokens"]
    for _, title, _ in _METRICS:
        header += [title, "Δ"]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for k in keys:
        b, h = base_rows.get(k, {}), head_rows.get(k, {})
        row = [str(k[0]), str(k[1])]
        for metric, _, hib in _METRICS:
            row += list(_cell(b.get(metric), h.get(metric), hib))
        out.append("| " + " | ".join(row) + " |")
    if not keys:
        out.append("")
        out.append("_No internode_ep records found in either input._")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="base JSONL")
    ap.add_argument("--head", required=True, help="head (PR) JSONL")
    ap.add_argument("--base-label", default="base")
    ap.add_argument("--head-label", default="head")
    args = ap.parse_args()
    print(
        render(load(args.base), load(args.head), args.base_label, args.head_label),
        end="",
    )


if __name__ == "__main__":
    main()
