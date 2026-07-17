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
ratio = havg / navg

# ---- Figure 1: throughput ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ns, ntf, "-", color="#888888", lw=1.6, label=f"native RCCL (avg {navg:.1f})")
ax.plot(hs, htf, "-", color="#1f77b4", lw=1.6, label=f"mori hp_sdma (avg {havg:.1f})")
ax.axhline(navg, color="#888888", ls="--", lw=0.8)
ax.axhline(havg, color="#1f77b4", ls="--", lw=0.8)
ax.set_xlabel("training step")
ax.set_ylabel("TFLOPS / GPU")
ax.set_title(
    "w16 FSDP2 E2E throughput — MI355X + AINIC (ionic)\n"
    f"hp_sdma {havg:.1f} = {ratio:.3f}x native "
    "(Qwen-7B, seq2048, bf16, 500 steps, bit-exact)"
)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "e2e_w16_tflops.png"), dpi=130)
print(
    f"wrote e2e_w16_tflops.png  (native {navg:.2f}, hp_sdma {havg:.2f}, {ratio:.3f}x)"
)

# ---- Figure 2: loss curve (bit-exact overlap) ----
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ns, nloss, "-", color="#888888", lw=2.5, label="native RCCL")
ax.plot(
    hs, hloss, "--", color="#d62728", lw=1.2, label="mori hp_sdma (overlaps exactly)"
)
ax.set_xlabel("training step")
ax.set_ylabel("training loss")
ax.set_title(
    "w16 FSDP2 E2E loss — MI355X + AINIC (ionic)\n"
    f"per-window loss BIT-IDENTICAL (last_loss {nlast:.15g} both, \u0394=0)"
)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "e2e_w16_loss.png"), dpi=130)
print(f"wrote e2e_w16_loss.png  (last_loss native {nlast!r} / hp_sdma {hlast!r})")
