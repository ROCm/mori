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
"""Calibrate CPU copies, then validate held-out transfer and UMBP API timings."""

import argparse
import csv
import hashlib
import itertools
import json
import os
import statistics
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def fit_profile(records, threads):
    """Fit only real training transfers; control-path time is never added twice."""
    groups = defaultdict(list)
    for row in records:
        if (
            row["phase"] == "train"
            and row["mode"] == "real"
            and row.get("concurrency", 1) == 1
        ):
            key = (row["effective_threads"], row["block_bytes"], row["batch"])
            groups[key].append(row["elapsed_ns"])
    curves = {}
    for p in threads:
        cases = [
            (b, n, statistics.median(v)) for (t, b, n), v in groups.items() if t == p
        ]
        x = np.array([[1, b * n / 1048576, n] for b, n, _ in cases])
        y = np.array([ns for _, _, ns in cases])
        if (
            len(cases) < 3
            or not np.all(np.isfinite(y) & (y > 0))
            or np.linalg.matrix_rank(x) < 3
        ):
            raise ValueError(f"insufficient or invalid training data for {p} threads")
        weights = 1 / np.maximum(y, 1000)
        best = None
        # Three nonnegative coefficients: enumerate active sets, no optimizer dependency.
        for fixed, per_copy in itertools.product([False, True], repeat=2):
            active = [fixed, True, per_copy]  # Bandwidth must always be fitted.
            coef = np.zeros(3)
            coef[active] = np.linalg.lstsq(
                x[:, active] * weights[:, None], y * weights, rcond=None
            )[0]
            if not np.all(np.isfinite(coef) & (coef >= 0)) or coef[1] <= 0:
                continue
            error = np.sum(((x @ coef - y) * weights) ** 2)
            if best is None or error < best[0]:
                best = (error, coef)
        if best is None:
            raise ValueError(f"no valid fit for {p} threads")
        curves[p] = best[1]
    # Interpolate unmeasured thread counts in time/byte, without extrapolating.
    values = [
        np.interp(range(1, max(threads) + 1), threads, [curves[p][i] for p in threads])
        for i in range(3)
    ]
    return {
        "channel": "host-memory",
        "latency_ns": values[0].tolist(),
        "copy_latency_ns": values[2].tolist(),
        "bandwidth_bytes_per_second": (1048576e9 / values[1]).tolist(),
    }


def compare(records):
    fields = ("phase", "threads", "concurrency", "block_bytes", "batch", "api")
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row["phase"] != "train":
            key = tuple(row.get(k, 1) if k == "concurrency" else row[k] for k in fields)
            groups[key][row["mode"]].append(row["elapsed_ns"])
    results = []
    for key, modes in sorted(groups.items()):
        real, sim = (statistics.median(modes[mode]) for mode in ("real", "sim"))
        results.append(
            dict(
                zip(fields, key),
                real_median_ns=real,
                sim_median_ns=sim,
                error_pct=100 * (sim / real - 1),
                within_tolerance=abs(sim - real) <= max(2000, 0.1 * real),
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="new output directory"
    )
    parser.add_argument(
        "--cpus", help="ordered CPU IDs, comma separated; first N used for N threads"
    )
    parser.add_argument("--threads", nargs="+", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--cache", choices=("cold", "reuse"), default="cold")
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        choices=(1, 2, 4),
        default=[1, 2, 4],
        help="simultaneous callers sharing one client; validation only",
    )
    parser.add_argument(
        "--max-mib",
        type=int,
        default=256,
        help="maximum aggregate payload MiB per case, across all callers",
    )
    for phase, option in itertools.product(
        ("train", "validation"), ("block-bytes", "batches")
    ):
        label = "block sizes in bytes" if option == "block-bytes" else "batch sizes"
        parser.add_argument(
            f"--{phase}-{option}",
            type=int,
            nargs="+",
            help=f"{phase} {label}; replaces the default list",
        )
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    allowed = sorted(os.sched_getaffinity(0))
    cpus = [int(c) for c in args.cpus.split(",")] if args.cpus else allowed[:8]
    threads = sorted(set(args.threads or [t for t in (1, 2, 4, 8) if t <= len(cpus)]))
    if (
        not cpus
        or len(set(cpus)) != len(cpus)
        or not set(cpus) <= set(allowed)
        or not threads
        or threads[0] != 1
        or max(threads) > len(cpus)
        or min(args.samples, args.rounds) <= 0
    ):
        parser.error(
            "need valid distinct CPUs, threads including 1, and positive samples/rounds"
        )
    if args.max_mib < 1 or 1 not in args.concurrency:
        parser.error("need positive --max-mib and --concurrency including 1")
    concurrency = sorted(set(args.concurrency))
    workloads = {}
    for phase in ("train", "validation"):
        workloads[phase] = {
            "block-bytes": getattr(args, f"{phase}_block_bytes"),
            "batches": getattr(args, f"{phase}_batches"),
        }
        if any(
            values is not None and min(values) <= 0
            for values in workloads[phase].values()
        ):
            parser.error("block and batch sizes must be positive integers")
    training_shapes = set()
    probe = args.probe.resolve(strict=True)
    env = dict(
        os.environ, MORI_UMBP_LOG_LEVEL="warn", UMBP_HBM_GATHER_PREWARM_STREAMS="0"
    )
    records = []
    args.output.mkdir(parents=True, exist_ok=False)

    def collect(phase, mode, round_id, p, c=1, profile=None):
        print(
            f"{phase} {mode}: round {round_id + 1}, {p} threads, {c} callers",
            flush=True,
        )
        command = [
            "taskset",
            "-c",
            ",".join(map(str, cpus[: p * c])),
            str(probe),
            phase,
            args.cache,
            str(args.samples),
            "--concurrency",
            str(c),
            "--max-mib",
            str(args.max_mib),
        ]
        for option, values in workloads[
            "train" if phase == "train" else "validation"
        ].items():
            if values is not None:
                command.extend([f"--{option}", ",".join(map(str, values))])
        if profile:
            command.append(str(profile))
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            check=True,
            env=dict(
                env, UMBP_DRAM_COPY_THREADS=str(p), UMBP_DRAM_WRITE_THREADS=str(p)
            ),
        )
        rows = list(csv.DictReader(result.stdout.splitlines()))
        if not rows:
            raise ValueError("probe returned no measurements")
        for row in rows:
            row = {k: v if k == "api" else int(v) for k, v in row.items()}
            if any(
                row[k] <= 0
                for k in (
                    "block_bytes",
                    "batch",
                    "effective_threads",
                    "concurrency",
                    "elapsed_ns",
                )
            ):
                raise ValueError("probe returned invalid measurements")
            shape = (row["block_bytes"], row["batch"])
            if phase == "train":
                training_shapes.add(shape)
            elif shape in training_shapes:
                raise ValueError(
                    f"training and validation overlap at block_bytes,batch={shape}"
                )
            records.append(dict(row, phase=phase, mode=mode, round=round_id, threads=p))

    for r in range(args.rounds):
        for p in threads:
            collect("train", "real", r, p)
    created = datetime.now(timezone.utc).isoformat()
    profile = {
        "id": f"cpu-cpu-{args.cache}-{created}",
        "paths": {"cpu_cpu": fit_profile(records, threads)},
    }
    with tempfile.TemporaryDirectory(dir=args.output.resolve()) as temporary:
        candidate = Path(temporary) / "profile.json"
        candidate.write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n")
        for r in range(args.rounds):
            for p, c, phase in itertools.product(
                threads, concurrency, ("engine", "api")
            ):
                # Alternate order to reduce systematic real/sim drift.
                for mode in ("real", "sim") if r % 2 == 0 else ("sim", "real"):
                    collect(phase, mode, r, p, c, candidate if mode == "sim" else None)
        comparisons = compare(records)
        report = {
            "profile_id": profile["id"],
            "profile_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "cpus": cpus,
            "threads": threads,
            "concurrency": concurrency,
            "max_mib": args.max_mib,
            "workloads": workloads,
            "cache": args.cache,
            "samples": args.samples,
            "rounds": args.rounds,
            "tolerance": "absolute median error <= max(2000 ns, 10% of real median)",
            "comparisons": comparisons,
            "measurements": records,
        }
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        candidate.replace(args.output / "profile.json")
    print(f"Profile and measurements: {args.output.resolve()}")


if __name__ == "__main__":
    main()
