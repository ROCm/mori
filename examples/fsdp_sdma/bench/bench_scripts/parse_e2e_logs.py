#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
"""Parse the raw w16 E2E bench logs (native + mori, 500 steps) into the curve
CSVs consumed by make_charts.py. No GPU required.

  python3 parse_e2e_logs.py

Reads   bench_results/raw_logs/{gt500_w16_m.log, defer500_w16_m.log}
Writes  bench_results/{loss_curve_w16.csv, tflops_curve_w16.csv}
"""
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "bench_results"))
RAW = os.path.join(RESULTS, "raw_logs")

# steps=485-489 mode=native avg_time_s=0.54 ... tflops_per_gpu=153.05 loss=10.402088
LINE = re.compile(
    r"steps=(\d+)-(\d+)\s+mode=(\w+).*?tflops_per_gpu=([\d.]+)\s+loss=([\d.]+)"
)


def parse(logname):
    """Return {end_step: (tflops, loss)} for every reported window."""
    out = {}
    with open(os.path.join(RAW, logname)) as f:
        for line in f:
            m = LINE.search(line)
            if m:
                end = int(m.group(2))
                out[end] = (float(m.group(4)), float(m.group(5)))
    return out


def main():
    nat = parse("gt500_w16_m.log")
    mori = parse("defer500_w16_m.log")
    steps = sorted(set(nat) & set(mori))

    with open(os.path.join(RESULTS, "loss_curve_w16.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "native_loss", "mori_loss"])
        for s in steps:
            w.writerow([s + 1, f"{nat[s][1]:.6f}", f"{mori[s][1]:.6f}"])

    with open(os.path.join(RESULTS, "tflops_curve_w16.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "native_tflops_per_gpu", "mori_tflops_per_gpu"])
        for s in steps:
            w.writerow([s + 1, f"{nat[s][0]:.2f}", f"{mori[s][0]:.2f}"])

    print(f"parsed {len(steps)} windows -> loss_curve_w16.csv, tflops_curve_w16.csv")


if __name__ == "__main__":
    main()
