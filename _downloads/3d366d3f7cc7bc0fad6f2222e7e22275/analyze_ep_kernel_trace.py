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
"""
Analyze internode_v1 kernel trace.

Produces one Gantt subplot per dispatch+combine iteration.  Each subplot shows
the merged single-timeline view (wall-clock span: earliest thread start →
latest thread end) with concurrent kernels stacked into separate rows.
Time is relative to the start of each iteration.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_events(path: str):
    """Return sorted list of (name, start_us, end_us, tid)."""
    with open(path) as f:
        data = json.load(f)

    # Timestamps are always in microseconds regardless of displayTimeUnit.
    by_tid: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for e in data["traceEvents"]:
        by_tid[e["tid"]][e["ph"]].append(e)

    intervals = []
    for tid, phases in by_tid.items():
        begins = sorted(phases.get("B", []), key=lambda x: x["ts"])
        ends = sorted(phases.get("E", []), key=lambda x: x["ts"])
        for b, e in zip(begins, ends):
            assert (
                b["name"] == e["name"]
            ), f"tid {tid}: B/E mismatch: {b['name']} vs {e['name']}"
            intervals.append((b["name"], b["ts"], e["ts"], tid))

    intervals.sort(key=lambda x: x[1])
    return intervals


# ---------------------------------------------------------------------------
# Iteration detection
# ---------------------------------------------------------------------------


def _merge_spans(spans):
    spans = sorted(spans)
    cs, ce = spans[0]
    result = []
    for s, e in spans[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            result.append((cs, ce))
            cs, ce = s, e
    result.append((cs, ce))
    return result


def detect_iterations(intervals):
    """
    An iteration is the period from the start of ep_dispatch_copy_to_staging
    to the end of ep_combine_all.  Returns list of (iter_start_ms, iter_end_ms).
    """
    by_kernel: dict[str, list] = defaultdict(list)
    for name, ts, te, _ in intervals:
        by_kernel[name].append((ts, te))

    disp_spans = _merge_spans(by_kernel["ep_dispatch_copy_to_staging"])
    comb_spans = _merge_spans(by_kernel["ep_combine_all"])

    assert len(disp_spans) == len(
        comb_spans
    ), f"Iteration count mismatch: {len(disp_spans)} dispatches vs {len(comb_spans)} combines"

    iters = [(ds, ce) for (ds, _), (_, ce) in zip(disp_spans, comb_spans)]
    return iters


# ---------------------------------------------------------------------------
# Per-iteration merge
# ---------------------------------------------------------------------------


def merge_to_single_timeline(intervals):
    """Merge per-thread spans by kernel name → wall-clock union."""
    by_kernel: dict[str, list] = defaultdict(list)
    for name, ts, te, _ in intervals:
        by_kernel[name].append((ts, te))

    return {name: _merge_spans(spans) for name, spans in by_kernel.items()}


# ---------------------------------------------------------------------------
# Semantic lane assignment
# ---------------------------------------------------------------------------
#
# Reflects the actual internode_v1 kernel execution path:
#
#   Lane 0  ep_dispatch_copy_to_staging   ← sequential pre-stage (separate kernel)
#   ────────────────────────────────────────────────────────────────────────
#   Lane 1  dispatch_inter_node_send      ← rdma blocks: send then recv
#           dispatch_inter_node_recv        (sequential within those blocks)
#   Lane 2  dispatch_intra               ← xgmi blocks: run CONCURRENTLY
#                                           with lane-1 (different SMs, same launch)
#   Lane 3  dispatch_sync                ← all-block barrier after lanes 1+2
#   ────────────────────────────────────────────────────────────────────────
#   Lane 4  combine_inter_node           ← internode combine (rdma blocks)
#   Lane 5  combine_intra_node           ← intranode combine  [concurrent with lane 4]
#   Lane 6  combine_sync                 ← all-block barrier after lanes 4+5
#   Lane 7  ep_combine_all               ← sequential post-stage
#
# Kernels unknown to this table fall back to sweep-line assignment on lanes
# starting at FALLBACK_LANE_START.

KERNEL_LANE: dict[str, int] = {
    # internode lane — dispatch and combine share the same row
    "dispatch_inter_node_send": 0,
    "dispatch_inter_node_recv": 0,
    "dispatch_inter_node_ll_send": 0,
    "dispatch_inter_node_ll_recv": 0,
    "combine_inter_node": 0,
    "combine_inter_node_ll": 0,
    # intranode lane — dispatch and combine share the same row
    "dispatch_intra": 1,
    "combine_intra_node": 1,
    "combine_intra_node_ll": 1,
}

LANE_LABELS: dict[int, str] = {
    0: "internode",
    1: "intranode",
}

FALLBACK_LANE_START = 2

# Kernels that are serial — drawn full-height spanning all parallel lanes
SERIAL_KERNELS = {
    "ep_dispatch_copy_to_staging",
    "dispatch_sync",
    "combine_sync",
    "ep_combine_sync_barrier",
    "ep_combine_all",
}


def assign_rows(merged_timeline):
    """
    Assign each merged span to a fixed semantic lane based on KERNEL_LANE.
    Kernels not in KERNEL_LANE are placed on fallback lanes using a sweep-line.
    Returns (span_row dict, total_rows).
    """
    span_row: dict[tuple[str, int], int] = {}
    known_lanes: set[int] = set()

    # Pass 1 – assign known kernels to their fixed lanes.
    # Serial kernels are skipped entirely: they span all rows in the plot
    # and don't need a lane assignment.
    unknown_kernels: list[tuple[str, list]] = []
    for kernel, spans in merged_timeline.items():
        if kernel in SERIAL_KERNELS:
            pass  # drawn full-height in plot; no lane needed
        elif kernel in KERNEL_LANE:
            lane = KERNEL_LANE[kernel]
            known_lanes.add(lane)
            for i in range(len(spans)):
                span_row[(kernel, i)] = lane
        else:
            unknown_kernels.append((kernel, spans))

    # Pass 2 – sweep-line for unknown kernels on fallback lanes
    fallback_row_end: list[float] = (
        []
    )  # indexed from 0; real lane = idx + FALLBACK_LANE_START
    fallback_events = []
    for kernel, spans in unknown_kernels:
        for i, (s, e) in enumerate(spans):
            fallback_events.append((s, 1, kernel, i))
            fallback_events.append((e, -1, kernel, i))
    fallback_events.sort(key=lambda x: (x[0], x[1]))

    for time, delta, kernel, idx in fallback_events:
        key = (kernel, idx)
        if delta == 1:
            assigned = next(
                (r for r, end in enumerate(fallback_row_end) if end <= time), None
            )
            if assigned is None:
                assigned = len(fallback_row_end)
                fallback_row_end.append(0.0)
            span_row[key] = FALLBACK_LANE_START + assigned
        else:
            local = span_row[key] - FALLBACK_LANE_START
            fallback_row_end[local] = time

    total_rows = (
        FALLBACK_LANE_START + len(fallback_row_end)
        if fallback_row_end
        else (max(known_lanes) + 1 if known_lanes else 1)
    )
    return span_row, total_rows


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

KERNEL_COLORS = {
    "ep_dispatch_copy_to_staging": "#4C72B0",
    "dispatch_intra": "#55A868",
    "dispatch_inter_node_send": "#C44E52",
    "dispatch_inter_node_recv": "#8172B2",
    "dispatch_inter_node_ll_send": "#E05050",
    "dispatch_inter_node_ll_recv": "#9B6FC8",
    "dispatch_sync": "#CCB974",
    "combine_intra_node": "#64B5CD",
    "combine_intra_node_ll": "#4FA8C0",
    "combine_inter_node": "#E07B39",
    "combine_inter_node_ll": "#D06020",
    "combine_sync": "#DD8452",
    "ep_combine_sync_barrier": "#B07030",
    "ep_combine_all": "#937860",
}

SHORT_NAME = {
    "ep_dispatch_copy_to_staging": "copy_to_stg",
    "dispatch_intra": "disp_intra",
    "dispatch_inter_node_send": "disp_send",
    "dispatch_inter_node_recv": "disp_recv",
    "dispatch_inter_node_ll_send": "disp_ll_send",
    "dispatch_inter_node_ll_recv": "disp_ll_recv",
    "dispatch_sync": "disp_sync",
    "combine_intra_node": "comb_intra",
    "combine_intra_node_ll": "comb_ll_intra",
    "combine_inter_node": "comb_inter",
    "combine_inter_node_ll": "comb_ll_inter",
    "combine_sync": "comb_sync",
    "ep_combine_sync_barrier": "comb_sync_bar",
    "ep_combine_all": "comb_all",
}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def detect_variant(iter_data):
    """Return 'v1_ll' if low-latency kernels are present, else 'v1'."""
    for d in iter_data:
        if any(
            k in d["merged"]
            for k in (
                "dispatch_inter_node_ll_send",
                "dispatch_inter_node_ll_recv",
                "combine_inter_node_ll",
                "combine_intra_node_ll",
            )
        ):
            return "v1_ll"
    return "v1"


def plot_iterations(iter_data, out_path):
    """
    iter_data: list of dicts, each with keys:
        label        – e.g. "Iteration 0"
        merged       – dict kernel->[(start,end)] relative to iter start
        span_row     – row assignment dict
        num_rows     – int
        total_ms     – float (duration of this iteration)
    """
    n = len(iter_data)
    max_total = max(d["total_ms"] for d in iter_data)
    max_rows = max(d["num_rows"] for d in iter_data)

    fig_w = max(20, max_total / 25)
    fig_h = n * max(3.5, max_rows * 0.7 + 1.5)
    fig, axes = plt.subplots(n, 1, figsize=(fig_w, fig_h), squeeze=False)

    bar_height = 0.75

    all_kernels = sorted({k for d in iter_data for k in d["merged"]})
    legend_patches = [
        mpatches.Patch(color=KERNEL_COLORS.get(k, "#999"), label=k) for k in all_kernels
    ]

    for ax, d in zip(axes[:, 0], iter_data):
        merged = d["merged"]
        span_row = d["span_row"]
        num_rows = d["num_rows"]
        total_ms = d["total_ms"]

        # Serial kernels span all rows; parallel kernels sit on their own lane.
        # Draw serial kernels first (lower z-order) so parallel bars render on top.
        center_y = (num_rows - 1) / 2.0  # vertical centre of all rows
        full_height = (
            num_rows - 1
        ) + bar_height  # bottom of lane-0 → top of lane-(N-1)

        for kernel, spans in sorted(merged.items()):
            color = KERNEL_COLORS.get(kernel, "#999999")
            is_serial = kernel in SERIAL_KERNELS
            for i, (s, e) in enumerate(spans):
                width = e - s
                if is_serial:
                    # Full-height bar to signal "everything is blocked here"
                    ax.barh(
                        center_y,
                        width,
                        left=s,
                        height=full_height,
                        color=color,
                        alpha=0.55,
                        edgecolor="white",
                        linewidth=0.6,
                        zorder=2,
                    )
                    if width > total_ms * 0.01:
                        label = f"{SHORT_NAME.get(kernel, kernel)}\n{width:.1f}us"
                        ax.text(
                            s + width / 2,
                            center_y,
                            label,
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white",
                            fontweight="bold",
                            zorder=3,
                        )
                else:
                    row = span_row.get((kernel, i), 0)
                    ax.barh(
                        row,
                        width,
                        left=s,
                        height=bar_height,
                        color=color,
                        alpha=0.88,
                        edgecolor="white",
                        linewidth=0.4,
                        zorder=4,
                    )
                    if width > total_ms * 0.02:
                        label = f"{SHORT_NAME.get(kernel, kernel)}\n{width:.1f}us"
                        ax.text(
                            s + width / 2,
                            row,
                            label,
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white",
                            fontweight="bold",
                            zorder=5,
                        )

        ax.set_title(d["label"], fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("Time relative to iteration start (us)", fontsize=9)
        ax.set_xlim(0, total_ms * 1.02)
        ax.set_ylim(-0.6, num_rows - 0.2)
        ax.set_yticks([])
        ax.yaxis.set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

        # Draw faint brackets to mark concurrent internode/intranode regions
        # (dispatch phase)
        disp_inter_spans = d["merged"].get(
            "dispatch_inter_node_send",
            d["merged"].get("dispatch_inter_node_ll_send", []),
        )
        disp_intra_spans = d["merged"].get("dispatch_intra", [])
        if disp_inter_spans and disp_intra_spans:
            region_s = min(s for s, _ in disp_inter_spans + disp_intra_spans)
            region_e = max(e for _, e in disp_inter_spans + disp_intra_spans)
            ax.axvspan(
                region_s,
                region_e,
                ymin=0,
                ymax=1,
                color="#AAAAAA",
                alpha=0.10,
                zorder=0,
            )
            ax.annotate(
                "concurrent",
                xy=((region_s + region_e) / 2, num_rows - 0.55),
                ha="center",
                va="top",
                fontsize=7,
                color="#666666",
                style="italic",
            )
        # (combine phase)
        comb_inter_spans = d["merged"].get(
            "combine_inter_node",
            d["merged"].get("combine_inter_node_ll", []),
        )
        comb_intra_spans = d["merged"].get(
            "combine_intra_node",
            d["merged"].get("combine_intra_node_ll", []),
        )
        if comb_inter_spans and comb_intra_spans:
            region_s = min(s for s, _ in comb_inter_spans + comb_intra_spans)
            region_e = max(e for _, e in comb_inter_spans + comb_intra_spans)
            ax.axvspan(
                region_s,
                region_e,
                ymin=0,
                ymax=1,
                color="#AAAAAA",
                alpha=0.10,
                zorder=0,
            )
            ax.annotate(
                "concurrent",
                xy=((region_s + region_e) / 2, num_rows - 0.55),
                ha="center",
                va="top",
                fontsize=7,
                color="#666666",
                style="italic",
            )

    variant = detect_variant(iter_data)
    # Shared legend at top
    fig.legend(
        handles=legend_patches, loc="upper right", fontsize=9, framealpha=0.9, ncol=3
    )
    fig.suptitle(
        f"internode_{variant} – one subplot per dispatch+combine iteration\n"
        "Execution path: copy_to_staging → [internode send+recv ‖ intranode] → sync → combine\n"
        "(bars = wall-clock span merged across all threads; time relative to iter start, unit: us)",
        fontsize=11,
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------


def print_summary(all_intervals, iterations, iter_intervals):
    print("\n" + "=" * 78)
    print("ITERATION SUMMARY  (wall-clock, us)")
    print("=" * 78)

    for i, ((it_s, it_e), ivs) in enumerate(zip(iterations, iter_intervals)):
        merged = merge_to_single_timeline(ivs)
        print(
            f"\n  Iteration {i}  [{it_s:.2f} – {it_e:.2f} us]"
            f"  total={it_e - it_s:.2f}us"
        )
        print(f"  {'Kernel':<35} {'start':>8} {'end':>8} {'dur':>8}")
        print(f"  {'-'*63}")
        all_spans = [(s, e, k) for k, spans in merged.items() for s, e in spans]
        for s, e, k in sorted(all_spans):
            print(f"  {k:<35} {s - it_s:8.2f} {e - it_s:8.2f} {e - s:8.2f}")

    print()
    print("PER-THREAD DURATION STATS across all iterations (us)")
    print("-" * 78)
    print(f"{'Kernel':<35} {'mean':>8} {'min':>8} {'max':>8} {'#calls':>7}")
    print("-" * 78)
    per_thread: dict[str, list[float]] = defaultdict(list)
    for name, ts, te, _ in all_intervals:
        per_thread[name].append(te - ts)
    for name, durs in sorted(
        per_thread.items(), key=lambda x: np.mean(x[1]), reverse=True
    ):
        print(
            f"  {name:<33} {np.mean(durs):8.2f} {np.min(durs):8.2f}"
            f" {np.max(durs):8.2f} {len(durs):7d}"
        )
    print("=" * 78)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Analyze internode_v1 kernel trace")
    parser.add_argument(
        "trace", nargs="?", default="/apps/ditian12/mori/trace_rank_0_0414_074134.json"
    )
    parser.add_argument("--out", default="timeline_internode_v1.png")
    args = parser.parse_args()

    print(f"[loading] {args.trace}")
    intervals = parse_events(args.trace)
    n_tids = len(set(t for *_, t in intervals))
    print(f"[parsed]  {len(intervals)} intervals  /  {n_tids} threads")

    iterations = detect_iterations(intervals)
    print(f"[iters]   {len(iterations)} iteration(s) detected")
    for i, (s, e) in enumerate(iterations):
        print(f"          iter {i}: {s:.2f} – {e:.2f} us  (dur={e-s:.2f}us)")

    # Split intervals per iteration
    iter_intervals: list[list] = [[] for _ in iterations]
    for iv in intervals:
        _, ts, te, _ = iv
        for i, (it_s, it_e) in enumerate(iterations):
            # include if the span overlaps this iteration window
            if ts < it_e and te > it_s:
                iter_intervals[i].append(iv)

    # Build per-iteration plot data (times relative to iter start)
    iter_data = []
    for i, (it_s, it_e) in enumerate(iterations):
        # Shift times to be relative to this iteration's start
        shifted = [
            (n, ts - it_s, te - it_s, tid) for n, ts, te, tid in iter_intervals[i]
        ]
        merged = merge_to_single_timeline(shifted)

        # For dispatch_sync / combine_sync: keep only the span that finishes last.
        # This represents the moment the final warp crosses the barrier — the one
        # that actually gates forward progress — and drops the noisy earlier fragments.
        for sync_key in ("dispatch_sync", "combine_sync"):
            if sync_key in merged and len(merged[sync_key]) > 1:
                last = max(merged[sync_key], key=lambda x: x[1])
                merged[sync_key] = [last]

        span_row, num_rows = assign_rows(merged)
        iter_data.append(
            {
                "label": f"Iteration {i}  (abs {it_s:.1f}–{it_e:.1f} us, "
                f"dur={it_e - it_s:.1f}us)",
                "merged": merged,
                "span_row": span_row,
                "num_rows": num_rows,
                "total_ms": it_e - it_s,
            }
        )

    print_summary(intervals, iterations, iter_intervals)

    out = Path(args.trace).parent / args.out
    plot_iterations(iter_data, str(out))


if __name__ == "__main__":
    main()
