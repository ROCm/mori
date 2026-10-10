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
  ``read`` records are excluded - the nightly read step is a fixed-size smoke
  check, not a sweep, so its numbers are not comparable to the write sweep.

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
        # The atomic-free run has the same config as its baseline; without this suffix the dedup in
        # load_records would drop it as a duplicate of the atomics run from the same CI run.
        variant = " atomic-free" if p.get("atomic_free") else ""
        return (
            f"{p.get('kernel_type')} EP{p.get('world_size')} "
            f"tok{p.get('max_tokens')} {p.get('dtype')}{variant}"
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
            "base_kernel": f"EP{ws}",
            "atomic_free": False,
            "run_id": rec.get("run_id", ""),
            "python": rec.get("python", ""),
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
        atomic_free = bool(p.get("atomic_free"))
        base_kernel = f"EP{ws}-{label}"
        return {
            "platform": plat,
            # Atomic-free rows are listed right after their baseline kernel (order + 0.5).
            "kernel": f"{base_kernel} (atomic-free)" if atomic_free else base_kernel,
            "base_kernel": base_kernel,
            "atomic_free": atomic_free,
            "run_id": rec.get("run_id", ""),
            "python": rec.get("python", ""),
            "order": _KERNEL_TYPE_ORDER.get(kt, 9) + (0.5 if atomic_free else 0),
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


def _delta_pct(base, new):
    """(new - base) / base in percent, or None when it cannot be computed."""
    try:
        base, new = float(base), float(new)
    except (TypeError, ValueError):
        return None
    if base == 0:
        return None
    return (new - base) / base * 100.0


def _cmp_cell(base, new, unit):
    """'<base> -> <new> <unit> (<delta>%)', or a dash when either side is missing."""
    if base is None or new is None:
        return "—"
    d = _delta_pct(base, new)
    delta = "n/a" if d is None else f"{d:+.1f}%"
    return f"{float(base):g} → {float(new):g} {unit} ({delta})"


def _atomic_free_pairs(records):
    """Pair each atomic-free EP record with its atomics baseline.

    A pair is only formed within one CI run (same run_id, platform, python, kernel and token
    count), so the comparison never sets today's atomic-free number against an older night's
    baseline, which is what the "latest record per config" collapse in the main tables would do
    if the atomic-free step were skipped or failed. Returns [(baseline_row, atomic_free_row)],
    keeping the newest pair per (platform, kernel, tokens).
    """
    rows = [r for r in (_normalize_ep(rec) for rec in records) if r]
    baseline = {}
    for r in rows:
        if r["atomic_free"]:
            continue
        key = (r["run_id"], r["platform"], r["python"], r["base_kernel"], r["tokens"])
        if key not in baseline or r["ts"] > baseline[key]["ts"]:
            baseline[key] = r

    newest = {}
    for r in rows:
        if not r["atomic_free"]:
            continue
        key = (r["run_id"], r["platform"], r["python"], r["base_kernel"], r["tokens"])
        base = baseline.get(key)
        if base is None:
            continue
        out_key = (r["platform"], r["base_kernel"], r["tokens"])
        if out_key not in newest or r["ts"] > newest[out_key][1]["ts"]:
            newest[out_key] = (base, r)

    return sorted(
        newest.values(),
        key=lambda p: (p[0]["platform"], p[0]["order"], p[0]["tokens"] or 0),
    )


def _atomic_free_section(records):
    pairs = _atomic_free_pairs(records)
    if not pairs:
        return ""

    lines = [
        "## MORI-EP atomic-free vs atomics",
        "",
        "Same CI run, kernel and token count; the only difference is "
        "`MORI_EP_DISABLE_RDMA_ATOMICS=1`, which sends the cross-node signals as RDMA WRITEs "
        "instead of RDMA atomics. Δ = (atomic-free − atomics) / atomics. For latency a positive "
        "Δ is slower; for bandwidth a negative Δ is slower. Each cell is one nightly run, so "
        "treat small Δ as noise.",
        "",
    ]
    headers = [
        "Hardware",
        "Kernels",
        "Tokens",
        "Dispatch Latency",
        "Dispatch BW",
        "Combine Latency",
        "Combine BW",
    ]
    by_platform = {}
    for base, af in pairs:
        by_platform.setdefault(base["platform"], []).append(
            [
                base["base_kernel"],
                base["tokens"],
                _cmp_cell(base["disp_lat"], af["disp_lat"], "µs"),
                _cmp_cell(base["disp_bw"], af["disp_bw"], "GB/s"),
                _cmp_cell(base["comb_lat"], af["comb_lat"], "µs"),
                _cmp_cell(base["comb_bw"], af["comb_bw"], "GB/s"),
            ]
        )
    lines.append(_html_table(headers, sorted(by_platform.items())))
    lines.append("")

    # One-line summary per platform: the mean latency change across all compared configs.
    for plat in sorted(by_platform):
        disp = [
            _delta_pct(b["disp_lat"], a["disp_lat"])
            for b, a in pairs
            if b["platform"] == plat
        ]
        comb = [
            _delta_pct(b["comb_lat"], a["comb_lat"])
            for b, a in pairs
            if b["platform"] == plat
        ]
        disp = [d for d in disp if d is not None]
        comb = [d for d in comb if d is not None]
        parts = []
        if disp:
            parts.append(f"dispatch latency {sum(disp) / len(disp):+.1f}%")
        if comb:
            parts.append(f"combine latency {sum(comb) / len(comb):+.1f}%")
        if parts:
            lines.append(
                f"**{plat}** mean over {len(by_platform[plat])} configs: "
                + ", ".join(parts)
                + "."
            )
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
    # Only the write sweep is a real perf signal. The read step is a fixed 4KB
    # single-point smoke check whose numbers sit far below the sweep's, so it is
    # dropped here as well as at the source (nightly clears MORI_PERF_OUT for it)
    # -- history from earlier runs still carries read records.
    io = [
        r
        for r in records
        if r.get("category") == "io"
        and (r.get("params") or {}).get("op_type") != "read"
    ]
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
    atomic_free = _atomic_free_section(records)
    if atomic_free:
        parts.append(atomic_free)
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
