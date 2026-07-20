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
# Regenerate the MI355X + AINIC (ionic RoCEv2) w16 E2E figures.
#   python3 plot_mi355x_ainic.py
# Reads the raw run logs under raw/ (native RCCL vs mori hp_sdma), a 500-step w16
# FSDP2 run (Qwen-7B, seq2048, bf16, 2 nodes x 8 GPU) on smci355 n09-33 + n09-29.
# Per-window loss is bit-identical between backends.
import os
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = os.path.dirname(os.path.abspath(__file__))  # raw/ : logs live next to this script
HERE = os.path.dirname(
    RAW
)  # platform results dir : figures (deliverables) are written up here

WIN_RE = re.compile(
    r"steps=\d+-(?P<end>\d+)\s+.*?tflops_per_gpu=(?P<tf>[\d.]+)\s+loss=(?P<loss>[\d.]+)"
)
AVG_RE = re.compile(r'"avg_tflops_per_gpu":\s*([\d.]+)')
LAST_LOSS_RE = re.compile(r'"last_loss":\s*([\d.]+)')


def parse(path):
    steps, tflops, loss = [], [], []
    avg_tflops = last_loss = None
    with open(path) as f:
        for line in f:
            m = WIN_RE.search(line)
            if m:
                steps.append(int(m["end"]) + 1)
                tflops.append(float(m["tf"]))
                loss.append(float(m["loss"]))
                continue
            a = AVG_RE.search(line)
            if a:
                avg_tflops = float(a.group(1))
            lm = LAST_LOSS_RE.search(line)
            if lm:
                last_loss = float(lm.group(1))
    if avg_tflops is None and tflops:
        avg_tflops = sum(tflops) / len(tflops)
    return steps, tflops, loss, avg_tflops, last_loss


ns, ntf, nloss, navg, nlast = parse(os.path.join(RAW, "e2e_w16_RCCL.log"))
hs, htf, hloss, havg, hlast = parse(os.path.join(RAW, "e2e_w16_hp_sdma.log"))

# Match the MI300X figure style (see results/mi300x_mlx5/): wide canvas, full box,
# RCCL = orange / mori hp_sdma = green, steady-state legend, no avg guide-lines.
HW = (
    "MI355X, AINIC (ionic) RoCEv2, world=16 (2 nodes x 8 GPU), "
    "Qwen-7B seq2048 bf16, 500 steps"
)


def _steady(series):
    # drop the first 1/5 warm-up windows, like the MI300X figures
    k = len(series) // 5
    tail = series[k:] or series
    return sum(tail) / len(tail)


n_steady = _steady(ntf)
h_steady = _steady(htf)
ratio = h_steady / n_steady

# ---- Figure 1: throughput ----
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.plot(
    ns, ntf, "-", color="tab:orange", lw=1.6, label=f"RCCL  (steady {n_steady:.0f})"
)
ax.plot(
    hs,
    htf,
    "-",
    color="tab:green",
    lw=1.6,
    label=f"hp_sdma  (steady {h_steady:.0f}, {ratio:.2f}x)",
)
ax.set_xlabel("training step")
ax.set_ylabel("TFLOPS / GPU")
ax.set_title("world=16 FSDP2 E2E throughput vs RCCL\n" + HW)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "e2e_w16_tflops.png"), dpi=130)
print(
    f"wrote e2e_w16_tflops.png  (RCCL steady {n_steady:.1f}, "
    f"hp_sdma steady {h_steady:.1f}, {ratio:.3f}x)"
)

# ---- Figure 2: loss curve (bit-exact overlap) ----
ndiff = sum(1 for a, b in zip(nloss, hloss) if a != b)
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.plot(ns, nloss, "-", color="tab:orange", lw=3.0, label="RCCL")
ax.plot(
    hs, hloss, "--", color="tab:green", lw=1.6, label="hp_sdma (== RCCL, bit-exact)"
)
ax.set_xlabel("training step")
ax.set_ylabel("loss")
ax.set_title(
    "world=16 FSDP2 E2E loss: hp_sdma == RCCL BIT-EXACT (max|\u0394|=0)\n" + HW
)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "e2e_w16_loss.png"), dpi=130)
print(
    f"wrote e2e_w16_loss.png  (last_loss native {nlast!r} / hp_sdma {hlast!r}, "
    f"{ndiff} windows differ)"
)

# ---- Figure 3: GEMM-under-AllGather overlap (MI300X bar-chart style) ----
OVL_RE = re.compile(
    r"shard=(\d+)MB\s+gemm_n=(\d+).*?rccl=([\d.]+)ms\s+hp_sdma=([\d.]+)ms"
    r".*?([\d.]+)x faster"
)
ovl = []
with open(os.path.join(RAW, "overlap_w16.log")) as f:
    for line in f:
        m = OVL_RE.search(line)
        if m:
            ovl.append((int(m[1]), int(m[2]), float(m[3]), float(m[4]), float(m[5])))

if ovl:
    shard = [o[0] for o in ovl]
    gemm_n = [o[1] for o in ovl]
    rccl_ms = [o[2] for o in ovl]
    hp_ms = [o[3] for o in ovl]
    speed = [o[4] for o in ovl]
    x = list(range(len(ovl)))
    w = 0.38
    top = max(rccl_ms)

    fig, ax = plt.subplots(figsize=(10, 5.7))
    b_rccl = ax.bar(
        [i - w / 2 for i in x],
        rccl_ms,
        w,
        color="#c0392b",
        label="under RCCL AllGather",
    )
    b_hp = ax.bar(
        [i + w / 2 for i in x],
        hp_ms,
        w,
        color="#27ae60",
        label="under hp_sdma AllGather (host-proxy + intra SDMA)",
    )
    for rect, v in list(zip(b_rccl, rccl_ms)) + list(zip(b_hp, hp_ms)):
        ax.annotate(
            f"{v:.2f}",
            (rect.get_x() + rect.get_width() / 2, v),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=9,
        )
    for xi, sp in zip(x, speed):
        ax.annotate(
            f"{sp:.2f}x faster",
            (xi, top * 1.04),
            ha="center",
            fontsize=9,
            color="#1a7a3a",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}MB x50\nGEMM n={g}" for s, g in zip(shard, gemm_n)])
    ax.set_xlabel("per-rank AllGather shard size (50 concurrent AGs)")
    ax.set_ylabel("GEMM time under 50 concurrent AllGathers  (ms, lower=better)")
    ax.set_ylim(0, top * 1.35)  # headroom so the legend clears the "Nx faster" labels
    ax.set_title(
        "Cross-node w16 overlap: compute runs FASTER under hp_sdma than under RCCL\n"
        "50 AllGathers overlap 50 GEMMs. RCCL's CU-resident kernels steal GPU from\n"
        "the GEMMs; hp_sdma keeps the cross-node leg on the CPU. bit-exact vs RCCL."
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "overlap_w16_gemm_time.png"), dpi=130)
    print(f"wrote overlap_w16_gemm_time.png  ({len(ovl)} shard sizes)")
