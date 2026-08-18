#!/usr/bin/env python3
import argparse
import collections
import csv
import json
import os
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse SDMA profiler counters and plot tasks per submission."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Profile result directory containing blocks_<N>_tokens_<N> directories.",
    )
    parser.add_argument(
        "--output-prefix",
        default="tasks_per_submission_vs_max_tokens",
        help="Output filename prefix inside the result directory.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg", "pdf"),
        default=("png", "svg"),
        help="Plot formats to generate. Defaults to png and svg.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=60,
        help="Theoretical maximum tasks per submission. Defaults to 60.",
    )
    parser.add_argument(
        "--capture-iters",
        type=int,
        default=None,
        help="Capture iterations. Inferred from metadata.txt or defaults to 3.",
    )
    return parser.parse_args()


def load_metadata(root):
    path = root / "metadata.txt"
    if not path.exists():
        return {}
    return dict(
        line.split("=", 1)
        for line in path.read_text().splitlines()
        if "=" in line
    )


def count_case(case_dir, capture_iters, max_tasks):
    traces = sorted(case_dir.glob("trace_rank*.json"))
    if not traces:
        raise ValueError(f"No trace_rank*.json files found in {case_dir}")

    counts = collections.Counter()
    for trace in traces:
        with trace.open() as f:
            counts.update(event["name"] for event in json.load(f)["traceEvents"])

    submit_calls = counts["dispatch_sdma_submit_call"]
    if submit_calls == 0:
        raise ValueError(f"No dispatch_sdma_submit_call events found in {case_dir}")

    submitted_tasks = sum(
        (1 << bit) * counts[f"dispatch_sdma_active_count_bit{bit}"]
        for bit in range(6)
    )
    avg = submitted_tasks / submit_calls
    ranks = len(traces)
    return {
        "trace_files": ranks,
        "submitted_tasks_total": submitted_tasks,
        "submit_calls_total": submit_calls,
        "avg_tasks_per_submit": avg,
        "batch_utilization_percent": avg / max_tasks * 100,
        "tasks_per_rank_per_dispatch": submitted_tasks / (ranks * capture_iters),
        "submits_per_rank_per_dispatch": submit_calls / (ranks * capture_iters),
    }


def main():
    args = parse_args()
    root = Path(args.input).resolve()
    metadata = load_metadata(root)
    capture_iters = args.capture_iters or int(metadata.get("capture_iters", 3))
    dtype = metadata.get("dtype", "unknown dtype")

    pattern = re.compile(r"blocks_(\d+)_tokens_(\d+)$")
    rows = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        match = pattern.match(case_dir.name)
        if not match:
            continue
        blocks, tokens = map(int, match.groups())
        result = count_case(case_dir, capture_iters, args.max_tasks)
        rows.append(
            {
                "blocks": blocks,
                "max_tokens_per_rank": tokens,
                "submitted_tasks_total": result["submitted_tasks_total"],
                "submit_calls_total": result["submit_calls_total"],
                "avg_tasks_per_submit": round(result["avg_tasks_per_submit"], 3),
                "theoretical_max_tasks_per_submit": args.max_tasks,
                "batch_utilization_percent": round(result["batch_utilization_percent"], 2),
                "tasks_per_rank_per_dispatch": round(result["tasks_per_rank_per_dispatch"], 3),
                "submits_per_rank_per_dispatch": round(result["submits_per_rank_per_dispatch"], 3),
                "trace_directory": case_dir.name,
            }
        )

    if not rows:
        raise SystemExit(f"No blocks_<N>_tokens_<N> directories found in {root}")
    rows.sort(key=lambda row: (row["blocks"], row["max_tokens_per_rank"]))

    csv_path = root / "batch_occupancy.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    md = [
        f"# {dtype} SDMA Batch Occupancy",
        "",
        f"The theoretical maximum is {args.max_tasks} token-copy tasks per submission.",
        "",
        "| Blocks | Max tokens/rank | Submitted tasks | Submit calls | Avg tasks/submit | Utilization |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['blocks']} | {row['max_tokens_per_rank']} | "
            f"{row['submitted_tasks_total']:,} | {row['submit_calls_total']:,} | "
            f"{row['avg_tasks_per_submit']:.2f} | "
            f"{row['batch_utilization_percent']:.1f}% |"
        )
    md.append("")
    (root / "BATCH_OCCUPANCY.md").write_text("\n".join(md))

    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
    import matplotlib.pyplot as plt

    colors = {8: "#1f77b4", 16: "#2ca02c", 32: "#d62728", 64: "#9467bd"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for blocks in sorted({row["blocks"] for row in rows}):
        series = [row for row in rows if row["blocks"] == blocks]
        ax.plot(
            [row["max_tokens_per_rank"] for row in series],
            [row["avg_tasks_per_submit"] for row in series],
            marker="o",
            linewidth=2,
            markersize=6,
            color=colors.get(blocks),
            label=f"blocks={blocks}",
        )
    ax.axhline(
        args.max_tasks,
        color="#555555",
        linestyle="--",
        linewidth=1.3,
        label=f"theoretical max ({args.max_tasks})",
    )
    tokens = sorted({row["max_tokens_per_rank"] for row in rows})
    ax.set_xscale("log", base=2)
    ax.set_xticks(tokens)
    ax.set_xticklabels([str(token) for token in tokens])
    ax.set_ylim(bottom=0, top=args.max_tasks * 1.07)
    ax.set_xlabel("max_tokens per rank")
    ax.set_ylabel("average token-copy tasks per SDMA submission")
    ax.set_title(f"{dtype} SDMA batch occupancy vs max_tokens")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(ncol=2)
    fig.tight_layout()

    for fmt in args.formats:
        output = root / f"{args.output_prefix}.{fmt}"
        fig.savefig(output, dpi=180)
        print(f"Wrote {output}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {root / 'BATCH_OCCUPANCY.md'}")


if __name__ == "__main__":
    main()
