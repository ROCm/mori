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
"""Build a README-style Markdown/HTML MORI performance report from perf JSONL.

Reads every ``*.jsonl`` under ``--input-dir`` (recursively), each line being one
record emitted by :mod:`tests.python.perf_report`, deduplicates them, keeps the
latest record per (hardware, kernel, tokens) config, and renders tables that
mirror the project README's ``## Benchmarks`` section:

* **MORI-EP**: a *Bandwidth* table and a *Latency* table, each with the columns
  ``Hardware | Kernels | Tokens | Dispatch ... | Combine ...`` and the hardware
  column merged with ``rowspan`` (HTML tables render in GitHub job summaries).
  Intra-node runs appear as ``EP8`` (XGMI only, RDMA shown as ``x``); inter-node
  runs as ``EP16-V1`` / ``EP16-V1-LL`` with both XGMI and RDMA.
* **MORI-IO**: an RDMA/XGMI transfer table (bandwidth + latency per message size).

Also writes ``history.jsonl`` (merged/deduped records) so the next run can feed
it back in. Stdlib only; append the report to ``$GITHUB_STEP_SUMMARY`` and/or
upload it as a build artifact - no HTML dashboard, no gh-pages.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os

# Kernel display + sort order for the EP tables.
_KERNEL_TYPE_LABEL = {"v0": "V0", "v1": "V1", "v1_ll": "V1-LL", "async_ll": "ASYNC-LL"}
_KERNEL_TYPE_ORDER = {"v0": 3, "v1": 1, "v1_ll": 2, "async_ll": 4}


def _series_key(category, params):
    p = params or {}
    if category == "intra_ep":
        return (
            f"EP{p.get('world_size')} tok{p.get('max_tokens')} "
            f"{p.get('dtype')} q={p.get('quant_type')} zc={int(bool(p.get('zero_copy')))}"
        )
    if category == "internode_ep":
        return (
            f"{p.get('kernel_type')} EP{p.get('world_size')} "
            f"tok{p.get('max_tokens')} {p.get('dtype')}"
        )
    if category == "io":
        return (
            f"{p.get('op_type')}/{p.get('backend')} "
            f"msg{p.get('msg_size')} bs{p.get('batch_size')}"
        )
    return json.dumps(params, sort_keys=True)


def _dedup_key(rec):
    return (
        rec.get("category"),
        rec.get("platform"),
        rec.get("python"),
        rec.get("run_id"),
        _series_key(rec.get("category"), rec.get("params", {})),
    )


def load_records(input_dir):
    records, seen = [], set()
    for path in sorted(
        glob.glob(os.path.join(input_dir, "**", "*.jsonl"), recursive=True)
    ):
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict) or "category" not in rec:
                        continue
                    key = _dedup_key(rec)
                    if key in seen:
                        continue
                    seen.add(key)
                    records.append(rec)
        except OSError:
            continue
    return records


# ── formatting helpers ─────────────────────────────────────────────────────


def _bw(v):
    return "x" if v is None else f"{float(v):g} GB/s"


def _lat(v):
    return "—" if v is None else f"{float(v):g} µs"


def _num(v, fmt="{:.2f}"):
    if v is None:
        return "-"
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def _ascii_table(title, headers, rows):
    """Render a PrettyTable-style ASCII table (README MORI-IO format)."""
    cols = len(headers)
    cells = [[str(c) for c in row] for row in rows]
    widths = [len(headers[i]) for i in range(cols)]
    for r in cells:
        for i in range(cols):
            widths[i] = max(widths[i], len(r[i]))

    def sep():
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def line(vals):
        return (
            "|" + "|".join(" " + v.center(w) + " " for v, w in zip(vals, widths)) + "|"
        )

    inner = sum(w + 2 for w in widths) + (cols - 1)
    out = [sep()]
    if title:
        out.append("|" + title.center(inner) + "|")
    out.append(sep())
    out.append(line(headers))
    out.append(sep())
    for r in cells:
        out.append(line(r))
    out.append(sep())
    return "\n".join(out)


def _esc(s):
    return html.escape(str(s))


def _html_table(headers, groups):
    """Render an HTML table; *groups* is a list of (group_label, rows) where the
    first column is the group label merged with rowspan (README style)."""
    out = ["<table>", "  <tr>"]
    out += [f"    <th>{_esc(h)}</th>" for h in headers]
    out.append("  </tr>")
    for group_label, rows in groups:
        for i, row in enumerate(rows):
            out.append("  <tr>")
            if i == 0:
                out.append(f'    <td rowspan="{len(rows)}">{_esc(group_label)}</td>')
            out += [f"    <td>{_esc(c)}</td>" for c in row]
            out.append("  </tr>")
    out.append("</table>")
    return "\n".join(out)


# ── record normalization ───────────────────────────────────────────────────


def _normalize_ep(rec):
    """Return a flat EP row dict, or None if not an EP record."""
    cat = rec.get("category")
    p = rec.get("params") or {}
    m = rec.get("metrics") or {}
    ws = p.get("world_size")
    tokens = p.get("max_tokens")
    plat = rec.get("platform") or "unknown"
    ts = rec.get("ts", 0)

    if cat == "intra_ep":
        return {
            "platform": plat,
            "kernel": f"EP{ws}",
            "order": 0,
            "tokens": tokens,
            "disp_xgmi": m.get("dispatch_bw_gbps"),
            "disp_rdma": None,
            "comb_xgmi": m.get("combine_bw_gbps"),
            "comb_rdma": None,
            "disp_lat": m.get("dispatch_lat_us"),
            "comb_lat": m.get("combine_lat_us"),
            "disp_bw": m.get("dispatch_bw_gbps"),
            "comb_bw": m.get("combine_bw_gbps"),
            "ts": ts,
        }
    if cat == "internode_ep":
        kt = p.get("kernel_type")
        label = _KERNEL_TYPE_LABEL.get(kt, str(kt).upper())
        return {
            "platform": plat,
            "kernel": f"EP{ws}-{label}",
            "order": _KERNEL_TYPE_ORDER.get(kt, 9),
            "tokens": tokens,
            "disp_xgmi": m.get("dispatch_xgmi_bw_gbps"),
            "disp_rdma": m.get("dispatch_rdma_bw_gbps"),
            "comb_xgmi": m.get("combine_xgmi_bw_gbps"),
            "comb_rdma": m.get("combine_rdma_bw_gbps"),
            "disp_lat": m.get("dispatch_lat_us"),
            "comb_lat": m.get("combine_lat_us"),
            "disp_bw": m.get("dispatch_rdma_bw_gbps"),
            "comb_bw": m.get("combine_rdma_bw_gbps"),
            "ts": ts,
        }
    return None


def _collapse(rows, key):
    """Keep the most recent row per *key* (a function of the row)."""
    best = {}
    for r in rows:
        k = key(r)
        if k not in best or r["ts"] > best[k]["ts"]:
            best[k] = r
    return list(best.values())


def _group_by_platform(rows):
    plats = {}
    for r in rows:
        plats.setdefault(r["platform"], []).append(r)
    return [(p, plats[p]) for p in sorted(plats)]


# ── section builders ───────────────────────────────────────────────────────


def _ep_sections(records):
    ep_rows = [r for r in (_normalize_ep(rec) for rec in records) if r]
    if not ep_rows:
        return ""
    ep_rows = _collapse(ep_rows, lambda r: (r["platform"], r["kernel"], r["tokens"]))
    ep_rows.sort(key=lambda r: (r["platform"], r["order"], r["tokens"] or 0))

    # descriptive params for the titles
    hidden = next(
        (
            rec.get("params", {}).get("hidden_dim")
            for rec in records
            if rec.get("category") in ("intra_ep", "internode_ep")
            and rec.get("params", {}).get("hidden_dim")
        ),
        None,
    )
    experts = next(
        (
            rec.get("params", {}).get("num_experts_per_token")
            for rec in records
            if rec.get("category") == "intra_ep"
            and rec.get("params", {}).get("num_experts_per_token")
        ),
        8,
    )
    cfg = ", ".join(
        x
        for x in [
            f"{hidden} hidden" if hidden else None,
            f"top-{experts} experts",
            "BF16 dispatch/combine",
        ]
        if x
    )

    lines = ["## MORI-EP", ""]

    # Bandwidth table
    lines.append(f"**Bandwidth** ({cfg})")
    lines.append("")
    bw_headers = [
        "Hardware",
        "Kernels",
        "Tokens",
        "Dispatch XGMI",
        "Dispatch RDMA",
        "Combine XGMI",
        "Combine RDMA",
    ]
    bw_groups = []
    for plat, rows in _group_by_platform(ep_rows):
        grows = [
            [
                r["kernel"],
                r["tokens"],
                _bw(r["disp_xgmi"]),
                _bw(r["disp_rdma"]),
                _bw(r["comb_xgmi"]),
                _bw(r["comb_rdma"]),
            ]
            for r in rows
        ]
        bw_groups.append((plat, grows))
    lines.append(_html_table(bw_headers, bw_groups))
    lines.append("")

    # Latency table
    lines.append(f"**Latency** ({cfg})")
    lines.append("")
    lat_headers = [
        "Hardware",
        "Kernels",
        "Tokens",
        "Dispatch Latency",
        "Dispatch BW",
        "Combine Latency",
        "Combine BW",
    ]
    lat_groups = []
    for plat, rows in _group_by_platform(ep_rows):
        grows = [
            [
                r["kernel"],
                r["tokens"],
                _lat(r["disp_lat"]),
                _bw(r["disp_bw"]),
                _lat(r["comb_lat"]),
                _bw(r["comb_bw"]),
            ]
            for r in rows
        ]
        lat_groups.append((plat, grows))
    lines.append(_html_table(lat_headers, lat_groups))
    lines.append("")
    return "\n".join(lines)


_IO_HEADERS = [
    "MsgSize (B)",
    "BatchSize",
    "TotalSize (MB)",
    "Max BW (GB/s)",
    "Avg Bw (GB/s)",
    "Min Lat (us)",
    "Avg Lat (us)",
]


def _io_section(records):
    io = [r for r in records if r.get("category") == "io"]
    if not io:
        return ""

    rows = []
    for rec in io:
        p = rec.get("params") or {}
        m = rec.get("metrics") or {}
        rows.append(
            {
                "platform": rec.get("platform") or "unknown",
                "backend": (p.get("backend") or "rdma"),
                "op": (p.get("op_type") or "write"),
                "msg": p.get("msg_size"),
                "batch": p.get("batch_size"),
                "total_mb": m.get("total_mb"),
                "avg_bw": m.get("avg_bw_gbps"),
                "max_bw": m.get("max_bw_gbps"),
                "avg_lat": m.get("avg_lat_us"),
                "min_lat": m.get("min_lat_us"),
                "ts": rec.get("ts", 0),
            }
        )
    rows = _collapse(
        rows, lambda r: (r["platform"], r["backend"], r["op"], r["msg"], r["batch"])
    )

    lines = ["## MORI-IO", ""]

    # One code-block table per (platform, backend, op), README style.
    groups = {}
    for r in rows:
        groups.setdefault((r["platform"], r["backend"], r["op"]), []).append(r)

    for plat, backend, op in sorted(groups):
        grp = sorted(groups[(plat, backend, op)], key=lambda r: r["msg"] or 0)
        batches = sorted({r["batch"] for r in grp if r["batch"] is not None})
        batch_txt = (
            f"{batches[0]} consecutive transfers" if len(batches) == 1 else "batched"
        )
        lines.append(
            f"GPU Direct {str(backend).upper()} {str(op).upper()}, "
            f"pairwise, {batch_txt}, {plat}:"
        )
        lines.append("")
        lines.append("```")
        trows = [
            [
                (
                    _num(r["msg"], "{:d}")
                    if isinstance(r["msg"], int)
                    else _num(r["msg"], "{:.0f}")
                ),
                (
                    _num(r["batch"], "{:d}")
                    if isinstance(r["batch"], int)
                    else _num(r["batch"], "{:.0f}")
                ),
                _num(r["total_mb"]),
                _num(r["max_bw"]),
                _num(r["avg_bw"]),
                _num(r["min_lat"]),
                _num(r["avg_lat"]),
            ]
            for r in grp
        ]
        title = f"{str(backend).upper()} {str(op).upper()} sweep ({plat})"
        lines.append(_ascii_table(title, _IO_HEADERS, trows))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def build_markdown(records):
    total = len(records)
    platforms = sorted({r.get("platform", "") for r in records if r.get("platform")})
    runs = sorted(
        {(r.get("date") or "") + " " + (r.get("commit") or "")[:8] for r in records}
    )

    parts = ["# MORI Nightly Performance Report", ""]
    parts.append(
        f"_{total} records · {', '.join(platforms) or 'n/a'} · "
        f"{len([x for x in runs if x.strip()])} run(s)._"
    )
    parts.append("")

    if not records:
        parts.append("> No perf records found for this run.")
        parts.append("")
        return "\n".join(parts)

    ep = _ep_sections(records)
    if ep:
        parts.append(ep)
    io = _io_section(records)
    if io:
        parts.append(io)

    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Build MORI perf report (README style)")
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing perf *.jsonl files (searched recursively).",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write report.md and history.jsonl into.",
    )
    args = ap.parse_args()

    records = load_records(args.input_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    history_path = os.path.join(args.output_dir, "history.jsonl")
    with open(history_path, "w", encoding="utf-8") as fh:
        for rec in sorted(records, key=lambda r: r.get("ts", 0)):
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    report_path = os.path.join(args.output_dir, "report.md")
    md = build_markdown(records)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(md + "\n")

    print(f"Loaded {len(records)} records from {args.input_dir}")
    print(f"Wrote {report_path}")
    print(f"Wrote {history_path}")
    if not records:
        print("WARNING: no perf records found; report is empty.")


if __name__ == "__main__":
    main()
