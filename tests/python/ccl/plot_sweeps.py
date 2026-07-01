#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# render the two benchmark charts from the sweep CSVs:
#   logs/sweep_standalone.csv   -> logs/chart_standalone.png    (mori vs rccl GB/s bars)
#   logs/sweep_gemm_overlap.csv -> logs/chart_gemm_overlap.png  (rccl vs sdma total-ms bars)
# Run on the host (no GPU needed):  python3 tests/python/ccl/plot_sweeps.py
import csv
import os

_OUT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)
_LOGS = os.path.join(_OUT_ROOT, "logs")


def _read(path, cols):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("dtype", "fp32") != "fp32":
                continue
            rows.append([float(r[c]) if "." in r[c] or c != "size_mb"
                         else int(r[c]) for c in cols])
    rows.sort(key=lambda x: x[0])
    return rows


def _bars(ax, sizes, a, b, la, lb):
    import numpy as np
    x = np.arange(len(sizes))
    w = 0.38
    ax.bar(x - w / 2, a, w, label=la, color="#4C72B0")
    ax.bar(x + w / 2, b, w, label=lb, color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s)}" for s in sizes])
    ax.set_xlabel("AllGather size (MiB/rank, fp32)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # the benchmark: standalone GB/s.
    p1 = os.path.join(_LOGS, "sweep_standalone.csv")
    if os.path.exists(p1):
        rows = _read(p1, ["size_mb", "mori_gbs", "rccl_gbs"])
        sizes = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(9, 5))
        _bars(ax, sizes, [r[1] for r in rows], [r[2] for r in rows],
              "mori SDMA hier", "RCCL")
        ax.set_ylabel("Algorithm bandwidth (GB/s)")
        ax.set_title("Standalone AllGather — mori SDMA hier vs RCCL (higher=better)")
        out = os.path.join(_LOGS, "chart_standalone.png")
        fig.tight_layout(); fig.savefig(out, dpi=130); print("wrote", out)

    # the benchmark: GEMM-overlap total time (lower=better).
    p2 = os.path.join(_LOGS, "sweep_gemm_overlap.csv")
    if os.path.exists(p2):
        rows = _read(p2, ["size_mb", "gemm_rccl_total_ms", "gemm_sdma_total_ms"])
        sizes = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(9, 5))
        _bars(ax, sizes, [r[1] for r in rows], [r[2] for r in rows],
              "gemm overlap with RCCL AG", "gemm overlap with SDMA AG")
        ax.set_ylabel("Overlapped total time (ms)")
        ax.set_title("GEMM overlap total time — RCCL AG vs mori SDMA AG (lower=better)")
        out = os.path.join(_LOGS, "chart_gemm_overlap.png")
        fig.tight_layout(); fig.savefig(out, dpi=130); print("wrote", out)


if __name__ == "__main__":
    main()
