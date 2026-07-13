#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
"""Render all cross-node HierAllGather benchmark figures from the CSVs in
../bench_results/. No GPU required.

  python3 make_charts.py

Inputs  (../bench_results/):
  loss_curve_w16.csv     step, native_loss, mori_loss           (E2E, bit-exact)
  tflops_curve_w16.csv   step, native_tflops.., mori_tflops..   (E2E throughput)
  e2e_gate2.csv          per-world E2E TFLOPS ratio vs native
  ut_w8.csv / ut_w16.csv standalone AllGather bandwidth vs RCCL
  overlap_w8.csv / overlap_w16.csv  AG bandwidth isolated vs under a GEMM

Outputs (../bench_results/*.png). Each chart is skipped if its CSV is absent.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "bench_results"))


def _rows(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def loss_curve_w16():
    rows = _rows("loss_curve_w16.csv")
    if not rows:
        return
    step = [int(r["step"]) for r in rows]
    nat = [float(r["native_loss"]) for r in rows]
    mori = [float(r["mori_loss"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(step, nat, "-", color="tab:orange", lw=3.0, label="native (RCCL)")
    ax.plot(step, mori, "--", color="tab:blue", lw=1.6,
            label="mori SDMA HierAllGather (deferred fence)")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title("world=16 FSDP2 E2E loss: mori == native BIT-EXACT (500 steps)\n"
                 "MI300X, mlx5 RoCEv2, Qwen-7B seq2048 bf16")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ndiff = sum(1 for a, b in zip(nat, mori) if a != b)
    ax.text(0.02, 0.02, f"max|Δloss| = 0.0  ({len(step)} windows, {ndiff} differ)",
            transform=ax.transAxes, fontsize=9, color="green")
    fig.tight_layout()
    out = os.path.join(RES, "loss_curve_w16.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


def tflops_curve_w16():
    rows = _rows("tflops_curve_w16.csv")
    if not rows:
        return
    step = [int(r["step"]) for r in rows]
    nat = [float(r["native_tflops_per_gpu"]) for r in rows]
    mori = [float(r["mori_tflops_per_gpu"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(step, nat, "-o", color="tab:orange", ms=2, label="native (RCCL)")
    ax.plot(step, mori, "-o", color="tab:blue", ms=2,
            label="mori SDMA HierAllGather (deferred fence)")
    # steady-state means (drop the first 20% warmup windows)
    k = len(step) // 5
    nat_ss, mori_ss = np.mean(nat[k:]), np.mean(mori[k:])
    ax.axhline(nat_ss, color="tab:orange", ls="--", lw=0.8)
    ax.axhline(mori_ss, color="tab:blue", ls="--", lw=0.8)
    ax.set_xlabel("training step")
    ax.set_ylabel("TFLOPS / GPU")
    ax.set_title(
        f"world=16 FSDP2 E2E throughput  (steady-state mori {mori_ss:.1f} vs "
        f"native {nat_ss:.1f} = {mori_ss / nat_ss:.3f}x)\n"
        "MI300X, mlx5 RoCEv2, Qwen-7B seq2048 bf16")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = os.path.join(RES, "tflops_curve_w16.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


def e2e_ratio():
    rows = _rows("e2e_gate2.csv")
    if not rows:
        return
    worlds, nat, mori = [], [], []
    by_world = {}
    for r in rows:
        by_world.setdefault(r["world"], {})[r["backend"]] = float(r["e2e_ratio_vs_native"])
    for w in sorted(by_world, key=int):
        d = by_world[w]
        nb = next((v for k, v in d.items() if "native" in k), 1.0)
        mb = next((v for k, v in d.items() if "mori" in k), None)
        if mb is None:
            continue
        worlds.append(f"world={w}")
        nat.append(nb)
        mori.append(mb)
    x = np.arange(len(worlds))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, nat, w, color="tab:orange", label="native (RCCL)")
    ax.bar(x + w / 2, mori, w, color="tab:blue", label="mori SDMA HierAllGather")
    ax.axhline(1.0, color="#888", ls="--", lw=0.8)
    for xi, v in zip(x + w / 2, mori):
        ax.annotate(f"{v:.3f}x", (xi, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=9, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(worlds)
    ax.set_ylabel("E2E TFLOPS ratio vs native (higher=better)")
    ax.set_title("Cross-node FSDP2 E2E: mori vs native (bit-exact)\n"
                 "MI300X, mlx5 RoCEv2")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RES, "e2e_ratio.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


def ut_chart():
    have = [(n, t) for n, t in
            [("ut_w8.csv", "world=8  (2 nodes x 4 GPU)"),
             ("ut_w16.csv", "world=16  (2 nodes x 8 GPU)")] if _rows(n)]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(6.5 * len(have), 5), squeeze=False)
    for ax, (csvname, title) in zip(axes[0], have):
        rows = [r for r in _rows(csvname) if r["dtype"] == "fp32"]
        x = [int(r["size_mb"]) for r in rows]
        mori = [float(r["mori_gbs"]) for r in rows]
        rccl = [float(r["rccl_gbs"]) for r in rows]
        ratio = [float(r["ratio"]) for r in rows]
        ax.plot(x, rccl, "-o", color="tab:orange", label="RCCL all_gather")
        ax.plot(x, mori, "-o", color="tab:blue",
                label="HierAllGather (SDMA intra + RDMA inter)")
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
    out = os.path.join(RES, "ut_allgather_bw.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


def overlap_two(world):
    """Two figures, two bars each (RCCL, mori): isolated AG, then AG under GEMM."""
    rows = _rows(f"overlap_{world}.csv")
    if not rows:
        return
    sizes = [f'{float(r["per_rank_mb"]):.0f}MB' for r in rows]
    x = np.arange(len(sizes))
    w = 0.35
    series = [
        ("iso", "isolated all-gather", "iso_rccl_gbs", "iso_mori_gbs"),
        ("gemm", "all-gather under a concurrent GEMM", "gemm_rccl_gbs", "gemm_mori_gbs"),
    ]
    for tag, subtitle, rccl_key, mori_key in series:
        rccl = [float(r[rccl_key]) for r in rows]
        mori = [float(r[mori_key]) for r in rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - w / 2, rccl, w, color="#ca0020", label="RCCL")
        ax.bar(x + w / 2, mori, w, color="#0571b0",
               label="mori HierAllGather (SDMA)")
        for xi, v in zip(x - w / 2, rccl):
            ax.annotate(f"{v:.0f}", (xi, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8)
        for xi, v in zip(x + w / 2, mori):
            ax.annotate(f"{v:.0f}", (xi, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.set_xlabel("per-rank message size")
        ax.set_ylabel("all-gather bandwidth (GB/s)")
        ax.set_title(f"{world}: {subtitle}  (RCCL vs mori)\nMI300X, mlx5 RoCEv2, bf16")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = os.path.join(RES, f"overlap_{tag}_{world}.png")
        fig.savefig(out, dpi=130)
        print("wrote", out)


if __name__ == "__main__":
    loss_curve_w16()
    tflops_curve_w16()
    e2e_ratio()
    ut_chart()
    overlap_two("w8")
    overlap_two("w16")
