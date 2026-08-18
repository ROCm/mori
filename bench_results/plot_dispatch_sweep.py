#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot dispatch latency vs max_tokens for the SDMA sweep."
    )
    parser.add_argument(
        "--input",
        default="summary.csv",
        help="Input summary CSV. Defaults to summary.csv next to this script.",
    )
    parser.add_argument(
        "--output",
        default="dispatch_vs_max_tokens.png",
        help="Output image path. Defaults to dispatch_vs_max_tokens.png next to this script.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "svg", "pdf"),
        default=None,
        help="Optional explicit output format.",
    )
    parser.add_argument("--dtype", default=None, help="Dtype label for the plot title; inferred from metadata.txt when omitted.")
    return parser.parse_args()


def resolve_path(path, script_dir):
    path = Path(path)
    if path.is_absolute():
        return path
    return script_dir / path


def load_rows(csv_path):
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "sdma": int(row["sdma"]),
                    "blocks": int(row["blocks"]),
                    "max_tokens": int(row["max_tokens"]),
                    "dispatch_us": float(row["dispatch_us_med"]),
                }
            )
    return rows


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    os.environ.setdefault("MPLCONFIGDIR", str(script_dir / ".matplotlib"))
    csv_path = resolve_path(args.input, script_dir)
    output_path = resolve_path(args.output, script_dir)

    import matplotlib.pyplot as plt

    dtype_label = args.dtype
    if dtype_label is None:
        metadata_path = csv_path.parent / "metadata.txt"
        if metadata_path.exists():
            metadata = dict(line.split("=", 1) for line in metadata_path.read_text().splitlines() if "=" in line)
            dtype_label = metadata.get("dtype")
    dtype_label = dtype_label or "unknown dtype"
    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows found in {csv_path}")

    colors = {8: "#1f77b4", 16: "#2ca02c", 32: "#d62728", 64: "#9467bd"}
    markers = {0: "o", 1: "s"}
    linestyles = {0: "--", 1: "-"}
    labels = {0: "no SDMA", 1: "SDMA"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for blocks in sorted({row["blocks"] for row in rows}):
        for sdma in (0, 1):
            series = sorted(
                (
                    row
                    for row in rows
                    if row["blocks"] == blocks and row["sdma"] == sdma
                ),
                key=lambda row: row["max_tokens"],
            )
            if not series:
                continue
            ax.plot(
                [row["max_tokens"] for row in series],
                [row["dispatch_us"] for row in series],
                marker=markers[sdma],
                linestyle=linestyles[sdma],
                linewidth=2,
                markersize=6,
                color=colors.get(blocks),
                label=f"blocks={blocks}, {labels[sdma]}",
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("max_tokens")
    ax.set_ylabel("dispatch latency median (us)")
    ax.set_title(f"{dtype_label} dispatch latency vs max_tokens")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(ncol=2, fontsize=9)

    tokens = sorted({row["max_tokens"] for row in rows})
    ax.set_xticks(tokens)
    ax.set_xticklabels([str(token) for token in tokens])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, format=args.format)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
