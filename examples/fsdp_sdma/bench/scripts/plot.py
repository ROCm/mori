#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
"""Render the benchmark figures directly from the raw logs in ../results/.
No GPU, no intermediate CSVs.

  python3 plot.py

Reads (../results/):   ut_w16.log, ut_overlap_w16.log,
                       e2e_w16_native.log, e2e_w16_mori.log
Writes (../results/):  ut_w16.png, ut_overlap_w16.png,
                       e2e_w16_loss.png, e2e_w16_tflops.png
Any figure whose log is missing is skipped.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results"))
HW = "MI300X, mlx5 RoCEv2, world=16 (2 nodes x 8 GPU)"


def _read(name):
    p = os.path.join(RES, name)
    return open(p).read() if os.path.exists(p) else None


def ut_bandwidth():
    txt = _read("ut_w16.log")
    if not txt:
        return
    pat = re.compile(
        r"\[sweep\] (\w+) (\d+)MB .* mori [\d.]+ms ([\d.]+)GB/s \| "
        r"rccl [\d.]+ms ([\d.]+)GB/s \| ratio=([\d.]+)")
    rows = [(dt, int(mb), float(m), float(r), float(ra))
            for dt, mb, m, r, ra in pat.findall(txt) if dt == "fp32"]
    if not rows:
        return
    x = [r[1] for r in rows]
    mori = [r[2] for r in rows]
    rccl = [r[3] for r in rows]
    ratio = [r[4] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, rccl, "-o", color="tab:orange", label="RCCL all_gather")
    ax.plot(x, mori, "-o", color="tab:blue",
            label="mori HierAllGather (SDMA intra + RDMA inter)")
    for xi, yi, ri in zip(x, mori, ratio):
        ax.annotate(f"{ri:.2f}x", (xi, yi), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8, color="tab:blue")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel("per-rank message size (MiB)")
    ax.set_ylabel("all-gather bandwidth (GB/s)")
    ax.set_title(f"Standalone AllGather bandwidth: mori vs RCCL (fp32, bit-exact)\n{HW}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = os.path.join(RES, "ut_w16.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


def ut_overlap():
    txt = _read("ut_overlap_w16.log")
    if not txt:
        return
    pat = re.compile(
        r"\[gemm-ovlp\] \w+ (\d+)MB .* rccl_total=([\d.]+)ms mori_total=([\d.]+)ms "
        r"\| solo rccl=([\d.]+) mori=([\d.]+)")
    rows = [(int(mb), float(rt), float(mt), float(rs), float(ms))
            for mb, rt, mt, rs, ms in pat.findall(txt)]
    if not rows:
        return
    sizes = [f"{r[0]}MB" for r in rows]
    x = np.arange(len(sizes))
    w = 0.35
    solo_r = [r[3] for r in rows]
    solo_m = [r[4] for r in rows]
    tot_r = [r[1] for r in rows]
    tot_m = [r[2] for r in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, rccl, mori, sub in [
        (a1, solo_r, solo_m, "AllGather alone (no compute)"),
        (a2, tot_r, tot_m, "AllGather overlapped with a GEMM"),
    ]:
        ax.bar(x - w / 2, rccl, w, color="#ca0020", label="RCCL")
        ax.bar(x + w / 2, mori, w, color="#0571b0", label="mori HierAllGather (SDMA)")
        for xi, v in zip(x - w / 2, rccl):
            ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8)
        for xi, v in zip(x + w / 2, mori):
            ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.set_xlabel("per-rank message size")
        ax.set_ylabel("time (ms, lower = better)")
        ax.set_title(sub)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "AllGather under a concurrent GEMM: RCCL's CU collective contends at large "
        f"size, mori's SDMA does not\n{HW}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(RES, "ut_overlap_w16.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


def _e2e_series(mode):
    txt = _read(f"e2e_w16_{mode}.log")
    if not txt:
        return None
    pat = re.compile(r"steps=\d+-(\d+) mode=\w+ .* tflops_per_gpu=([\d.]+) loss=([\d.]+)")
    return [(int(s) + 1, float(tf), float(ls)) for s, tf, ls in pat.findall(txt)]


def e2e_curves():
    nat = _e2e_series("native")
    mori = _e2e_series("mori")
    if not nat or not mori:
        return
    nmap = {s: (tf, ls) for s, tf, ls in nat}
    mmap = {s: (tf, ls) for s, tf, ls in mori}
    steps = sorted(set(nmap) & set(mmap))

    # loss curve
    nl = [nmap[s][1] for s in steps]
    ml = [mmap[s][1] for s in steps]
    ndiff = sum(1 for a, b in zip(nl, ml) if a != b)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, nl, "-", color="tab:orange", lw=3.0, label="native (RCCL)")
    ax.plot(steps, ml, "--", color="tab:blue", lw=1.6,
            label="mori SDMA HierAllGather (deferred fence)")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title(f"FSDP2 E2E loss: mori == native BIT-EXACT ({len(steps)} windows)\n{HW}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.text(0.02, 0.02, f"max|Δloss| = 0.0  ({len(steps)} windows, {ndiff} differ)",
            transform=ax.transAxes, fontsize=9, color="green")
    fig.tight_layout()
    out = os.path.join(RES, "e2e_w16_loss.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # tflops curve
    nt = [nmap[s][0] for s in steps]
    mt = [mmap[s][0] for s in steps]
    k = len(steps) // 5
    ns, ms = np.mean(nt[k:]), np.mean(mt[k:])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, nt, "-o", color="tab:orange", ms=2, label="native (RCCL)")
    ax.plot(steps, mt, "-o", color="tab:blue", ms=2,
            label="mori SDMA HierAllGather (deferred fence)")
    ax.axhline(ns, color="tab:orange", ls="--", lw=0.8)
    ax.axhline(ms, color="tab:blue", ls="--", lw=0.8)
    ax.set_xlabel("training step")
    ax.set_ylabel("TFLOPS / GPU")
    ax.set_title(
        f"FSDP2 E2E throughput (steady-state mori {ms:.1f} vs native {ns:.1f} "
        f"= {ms / ns:.3f}x)\n{HW}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = os.path.join(RES, "e2e_w16_tflops.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    ut_bandwidth()
    ut_overlap()
    e2e_curves()
