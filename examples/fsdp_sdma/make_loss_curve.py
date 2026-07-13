#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
"""Plot the long-run (500-step) FSDP2 training loss curve of the mori SDMA
cross-node HierAllGather against the framework-default ("native") AllGather, for
both target topologies (world=8 = 2-node x 4 GPU, world=16 = 2-node x 8 GPU).

Both runs use identical config/seed and the mori path uses the ZERO-TUNING
auto-defaults (no MORI_* env set), so the two curves overlap step-for-step and
the final loss matches to full float precision. Reads loss_curve.csv (world=8)
and, if present, loss_curve_w16.csv (world=16). No GPU needed:
    python3 make_loss_curve.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    rows = list(csv.DictReader(open(os.path.join(_HERE, name))))
    step = [int(r["step"]) for r in rows]
    native = [float(r["native_loss"]) for r in rows]
    sdma = [float(r["sdma_loss"]) for r in rows]
    return step, native, sdma


panels = [("loss_curve.csv", "world=8  (2-node x 4 GPU)")]
if os.path.exists(os.path.join(_HERE, "loss_curve_w16.csv")):
    panels.append(("loss_curve_w16.csv", "world=16  (2-node x 8 GPU)"))

fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 5.2),
                         squeeze=False)
for ax, (csvname, title) in zip(axes[0], panels):
    step, native, sdma = _load(csvname)
    md = max(abs(n - s) for n, s in zip(native, sdma))
    exact = sum(1 for n, s in zip(native, sdma) if n == s)
    ax.plot(step, native, "o-", color="#1f77b4", lw=2, ms=5,
            label="native baseline AllGather")
    ax.plot(step, sdma, "x--", color="#d62728", lw=1.6, ms=7,
            label="mori SDMA hier (auto, this work)")
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss (per-window)")
    ax.set_title(f"{title}\nbit-exact {exact}/{len(step)} windows, "
                 f"max|Δ|={md:.1e}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.annotate(f"final loss\nnative={native[-1]:.6f}\nSDMA={sdma[-1]:.6f}",
                (0.03, 0.03), xycoords="axes fraction", fontsize=8.5,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="#f2f2f2", ec="#999"))

fig.suptitle("Cross-node FSDP2 (Qwen-7B, seq2048) 500-step training loss — "
             "mori SDMA HierAllGather vs native (zero env tuning; bit-exact)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(_HERE, "loss_curve.png")
fig.savefig(out, dpi=130)
print("wrote", out)
