#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
"""Render the cross-node HierAllGather benchmark figures from the CSVs in
bench_data/. No GPU required.

  python3 make_bench_charts.py

Produces:
  ut_allgather_bw.png   standalone AllGather bandwidth vs RCCL (world=8 and 16)
  gemm_overlap.png      AllGather bandwidth in isolation vs under a concurrent GEMM
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "bench_data")


def _rows(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


def ut_chart():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (csvname, title) in zip(
        axes,
        [("ut_w8.csv", "world=8  (2 nodes x 4 GPU)"),
         ("ut_w16.csv", "world=16  (2 nodes x 8 GPU)")],
    ):
        rows = [r for r in _rows(csvname) if r["dtype"] == "fp32"]
        x = [int(r["size_mb"]) for r in rows]
        mori = [float(r["mori_gbs"]) for r in rows]
        rccl = [float(r["rccl_gbs"]) for r in rows]
        ratio = [float(r["ratio"]) for r in rows]
        ax.plot(x, rccl, "-o", color="tab:orange", label="RCCL all_gather")
        ax.plot(x, mori, "-o", color="tab:blue", label="HierAllGather (SDMA intra + RDMA inter)")
        for xi, yi, ri in zip(x, mori, ratio):
            ax.annotate(f"{ri:.2f}x", (xi, yi), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=8, color="tab:blue")
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x])
        ax.set_xlabel("per-rank message size (MiB)")
        ax.set_ylabel("all-gather bandwidth (GB/s)")
        ax.set_title(title + "  (fp32, bit-exact)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Cross-node AllGather bandwidth: HierAllGather vs RCCL "
                 "(MI300X, mlx5 RoCEv2)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(HERE, "ut_allgather_bw.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


def overlap_chart():
    rows = _rows("overlap_w8.csv")
    sizes = [f'{float(r["per_rank_mb"]):.0f}MB' for r in rows]
    x = np.arange(len(sizes))
    w = 0.2
    iso_m = [float(r["iso_mori_gbs"]) for r in rows]
    iso_r = [float(r["iso_rccl_gbs"]) for r in rows]
    gem_m = [float(r["gemm_mori_gbs"]) for r in rows]
    gem_r = [float(r["gemm_rccl_gbs"]) for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax.bar(x - 1.5 * w, iso_r, w, color="#f4a582", label="RCCL, isolated")
    ax.bar(x - 0.5 * w, gem_r, w, color="#ca0020", label="RCCL, under GEMM")
    ax.bar(x + 0.5 * w, iso_m, w, color="#92c5de", label="HierAllGather, isolated")
    ax.bar(x + 1.5 * w, gem_m, w, color="#0571b0", label="HierAllGather, under GEMM")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("per-rank message size")
    ax.set_ylabel("all-gather bandwidth (GB/s)")
    ax.set_title("AllGather bandwidth: isolated vs under a concurrent GEMM")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    slow_m = [float(r["mori_slowdown"]) for r in rows]
    slow_r = [float(r["rccl_slowdown"]) for r in rows]
    ax2.bar(x - w / 2, slow_r, w, color="#ca0020", label="RCCL")
    ax2.bar(x + w / 2, slow_m, w, color="#0571b0", label="HierAllGather (SDMA)")
    ax2.axhline(1.0, color="#888", lw=0.8, ls="--")
    for xi, v in zip(x - w / 2, slow_r):
        ax2.annotate(f"{v:.2f}x", (xi, v), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, slow_m):
        ax2.annotate(f"{v:.2f}x", (xi, v), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sizes)
    ax2.set_xlabel("per-rank message size")
    ax2.set_ylabel("slowdown under concurrent GEMM (lower=better)")
    ax2.set_title("Copy-engine SDMA keeps bandwidth under GEMM; RCCL's CU copy does not")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("No-CU-contention dividend (world=8, bf16)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(HERE, "gemm_overlap.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    ut_chart()
    overlap_chart()
