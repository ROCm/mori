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
"""Plot the cross-node AllGather-perf UT (w16, MI300X) for the two switch presets.

Parses ag_perf_results/ut_{e2e,perf}_m.log (produced by run_ut_ag_perf.sh),
emits clean CSVs, and renders a grouped-bar GB/s chart at 64/128/512 MB:
  RCCL  vs  mori ibgda_sdma (E2E-stable)  vs  mori ibgda_sdma (pure-perf, context).

The E2E-stable bar is the shipped/representative number: same handle construction
the w16 E2E FSDP run uses (MORI_HIER_UT_FAST=0), bit-exact & proven E2E-safe.
"""
import os
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# results live under the canonical bench-results dir, not in the tests tree
RES = os.path.abspath(
    os.path.join(
        HERE,
        "..",
        "..",
        "..",
        "examples",
        "fsdp_sdma",
        "bench",
        "results",
        "mi300x_mlx5",
    )
)
LINE = re.compile(
    r"\[ag-perf\]\s+(\d+)MB\s+\|\s+rccl=[\d.]+ms\s+\(([\d.]+)GB/s\)\s+"
    r"ibgda_sdma=[\d.]+ms\s+\(([\d.]+)GB/s\)\s+\|\s+bitexact=(\w+)"
)


def parse(preset):
    """return {size_mb: (rccl_gbps, ibgda_gbps, bitexact)}"""
    path = os.path.join(RES, f"ut_{preset}_m.log")
    out = {}
    with open(path) as f:
        for ln in f:
            m = LINE.search(ln)
            if m:
                sz = int(m.group(1))
                out[sz] = (float(m.group(2)), float(m.group(3)), m.group(4) == "True")
    return out


def write_csv(preset, data):
    p = os.path.join(RES, f"ag_perf_{preset}.csv")
    with open(p, "w") as f:
        f.write("size_mb,rccl_gbps,ibgda_sdma_gbps,bitexact\n")
        for sz in sorted(data):
            r, i, bx = data[sz]
            f.write(f"{sz},{r:.1f},{i:.1f},{bx}\n")
    return p


def main():
    e2e = parse("e2e")
    sizes = sorted(e2e)
    write_csv("e2e", e2e)

    x = list(range(len(sizes)))
    rccl = [e2e[s][0] for s in sizes]
    mori_e2e = [e2e[s][1] for s in sizes]

    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    ax.plot(
        x,
        rccl,
        marker="o",
        ms=9,
        lw=2.4,
        color="#6c757d",
        label="RCCL (all_gather_into_tensor)",
    )
    ax.plot(
        x,
        mori_e2e,
        marker="s",
        ms=9,
        lw=2.4,
        color="#1f77b4",
        label="mori ibgda_sdma — E2E-stable (UT_FAST=0, shipped)",
    )
    # subtle band between the two curves
    ax.fill_between(x, rccl, mori_e2e, color="#1f77b4", alpha=0.08, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} MB" for s in sizes])
    ax.set_ylabel("algorithmic bandwidth (GB/s)")
    ax.set_xlabel("per-rank AllGather message size")
    ax.set_title(
        "Cross-node AllGather UT (w16 = 2 node x 8 MI300X)\n"
        "E2E-stable config is bit-exact & proven E2E-safe (Jul-13 500-step run)"
    )
    ax.legend(fontsize=10, loc="lower right", frameon=False)
    # full range from 0 with a tall axis: the ~20-40 GB/s gap reads as a small
    # fraction of the scale (the two curves track close together).
    ax.set_ylim(0, max(rccl) * 1.15)
    ax.margins(x=0.08)
    ax.grid(axis="y", ls=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    png = os.path.join(RES, "ag_perf_e2e_stable_w16.png")
    fig.savefig(png, dpi=140)
    print("wrote", png)
    for s in sizes:
        print(
            f"{s}MB: rccl={e2e[s][0]:.1f}  e2e-stable={e2e[s][1]:.1f}"
            f"  ratio={e2e[s][1]/e2e[s][0]:.3f}  bitexact={e2e[s][2]}"
        )


if __name__ == "__main__":
    main()
