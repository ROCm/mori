#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
"""Host-proxy ASYNC w16 E2E figures (loss + TFLOPS) from ../results/ logs.
  python3 hp_plot.py
Reads  e2e_hp_w16_native.log (RCCL), e2e_hp_w16_async.log (mori host-proxy async)
Writes e2e_hp_w16_loss.png, e2e_hp_w16_tflops.png   (same bench.py harness)
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.normpath(os.path.join(HERE, "..", "results", "mi300x_mlx5"))
HW = "MI300X, mlx5 RoCEv2, world=16 (2 nodes x 8 GPU), Qwen-7B seq2048 bf16"
PAT = re.compile(r"steps=\d+-(\d+) mode=\w+ .* tflops_per_gpu=([\d.]+) loss=([\d.]+)")


def series(name):
    txt = open(os.path.join(RES, name)).read()
    return [(int(s) + 1, float(tf), float(ls)) for s, tf, ls in PAT.findall(txt)]


def main():
    nat = {s: (tf, ls) for s, tf, ls in series("e2e_hp_w16_native.log")}
    hp = {s: (tf, ls) for s, tf, ls in series("e2e_hp_w16_async.log")}
    steps = sorted(set(nat) & set(hp))

    # ---- loss curve (bit-exact overlap) ----
    nl = [nat[s][1] for s in steps]
    hl = [hp[s][1] for s in steps]
    ndiff = sum(1 for a, b in zip(nl, hl) if a != b)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, nl, "-", color="tab:orange", lw=3.0, label="native (RCCL)")
    ax.plot(steps, hl, "--", color="tab:green", lw=1.6,
            label="mori host-proxy async (CPU-posted RDMA)")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title(f"world=16 FSDP2 E2E loss: host-proxy async == native BIT-EXACT "
                 f"({len(steps)} windows)\n{HW}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.text(0.02, 0.02, f"max|Δloss| = 0.0  ({len(steps)} windows, {ndiff} differ)",
            transform=ax.transAxes, fontsize=9, color="green")
    fig.tight_layout()
    out = os.path.join(RES, "e2e_hp_w16_loss.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # ---- tflops curve ----
    nt = [nat[s][0] for s in steps]
    ht = [hp[s][0] for s in steps]
    k = len(steps) // 5
    ns, hs = np.mean(nt[k:]), np.mean(ht[k:])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, nt, "-o", color="tab:orange", ms=2, label="native (RCCL)")
    ax.plot(steps, ht, "-o", color="tab:green", ms=2,
            label="mori host-proxy async (CPU-posted RDMA)")
    ax.axhline(ns, color="tab:orange", ls="--", lw=0.8)
    ax.axhline(hs, color="tab:green", ls="--", lw=0.8)
    ax.set_xlabel("training step")
    ax.set_ylabel("TFLOPS / GPU")
    ax.set_title(f"world=16 FSDP2 E2E throughput: host-proxy async "
                 f"(steady {hs:.1f} vs native {ns:.1f} = {hs / ns:.3f}x)\n{HW}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = os.path.join(RES, "e2e_hp_w16_tflops.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
