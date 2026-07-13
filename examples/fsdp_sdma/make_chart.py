#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
"""Build compare_chart.png: end-to-end FSDP2 training-step throughput of
the mori SDMA cross-node HierAllGather vs the framework-default ("native")
AllGather, for the two target topologies (2-node x 4 GPU and 2-node x 8 GPU).

Throughput is normalized to native = 1.0 (per-window paired comparison); the
bit-exact training loss is annotated under each config as the correctness result.
Reads e2e_gate2.csv. No GPU needed:  python3 make_chart.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(_HERE, "e2e_gate2.csv"))))

configs = []
for r in rows:
    if r["config"] not in configs:
        configs.append(r["config"])

fig, ax = plt.subplots(figsize=(9, 5.4))
w = 0.36
x = np.arange(len(configs))
native_r = [1.0 for _ in configs]
mori_r, mori_lo, mori_hi, mori_loss = [], [], [], []
for c in configs:
    m = next(r for r in rows if r["config"] == c and r["backend"] == "mori SDMA hier")
    mori_r.append(float(m["e2e_ratio_vs_native"]))
    mori_lo.append(float(m["e2e_ratio_vs_native"]) - float(m["e2e_ratio_lo"]))
    mori_hi.append(float(m["e2e_ratio_hi"]) - float(m["e2e_ratio_vs_native"]))
    mori_loss.append(m["last_loss"])

ax.bar(x - w / 2, native_r, w, label="native baseline", color="#4C72B0")
ax.bar(x + w / 2, mori_r, w, yerr=[mori_lo, mori_hi], capsize=5,
       label="mori SDMA hier (this work)", color="#55A868")
ax.axhline(1.0, color="#888", lw=0.8, ls="--")

for i, c in enumerate(configs):
    ax.annotate(f"{mori_r[i]:.3f}x", (i + w / 2, mori_r[i]),
                textcoords="offset points", xytext=(0, 6), ha="center",
                fontsize=10, fontweight="bold")
    ax.annotate(f"loss={mori_loss[i]}\n(bit-exact vs native)", (i, 0.04),
                ha="center", va="bottom", fontsize=8, color="#1a1a1a")

ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n(world={next(r['world'] for r in rows if r['config']==c)})"
                    for c in configs])
ax.set_ylabel("E2E FSDP2 training-step throughput\n(normalized, native = 1.0; higher=better)")
ax.set_title("Cross-node FSDP2 (Qwen-7B) end-to-end throughput — "
             "mori SDMA HierAllGather vs native baseline (bit-exact)")
ax.set_ylim(0, max(mori_r) + 0.25)
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out = os.path.join(_HERE, "compare_chart.png")
fig.savefig(out, dpi=130)
print("wrote", out)
