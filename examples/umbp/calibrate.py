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
"""Calibrate real UMBP transfer engines and validate independent workloads."""

import argparse
import csv
import hashlib
import itertools
import json
import os
import shlex
import signal
import statistics
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CHANNELS = dict(
    cpu_cpu="host-memory",
    cpu_staging="host-memory",
    cpu_staging_read="host-memory",
    cpu_gpu="host-device-link",
    cpu_gpu_gather="host-device-link",
    gpu_cpu="host-device-link",
    gpu_cpu_gather="host-device-link",
    gpu_gpu="device-memory",
    remote_read="network-link",
    remote_write="network-link",
    file_cpu="ssd-device",
    cpu_file="ssd-device",
    file_gpu="ssd-device",
    file_cpu_staged="ssd-device",
    file_cpu_staged_no_crc="ssd-device",
    cpu_file_staged="ssd-device",
)
TRANSFER_PATHS = (
    "cpu_cpu",
    "cpu_gpu",
    "gpu_cpu",
    "gpu_gpu",
    "remote_read",
    "remote_write",
    "file_cpu",
    "cpu_file",
    "file_gpu",
    "file_cpu_staged",
    "cpu_file_staged",
)


def fit_nonnegative(x, y):
    weights = 1 / np.maximum(y, 1000)
    best = None
    # Enumerate nonnegative active sets, without an optimizer dependency.
    for active in itertools.product([False, True], repeat=x.shape[1]):
        if not any(active):
            continue
        active = list(active)
        coef = np.zeros(x.shape[1])
        coef[active] = np.linalg.lstsq(
            x[:, active] * weights[:, None], y * weights, rcond=None
        )[0]
        if not np.all(np.isfinite(coef) & (coef >= 0)):
            continue
        error = np.sum(((x @ coef - y) * weights) ** 2)
        if best is None or error < best[0]:
            best = (error, coef)
    return best


def fit_profile(records, threads, *, log_size=False, staging=False):
    """Fit only real training transfers; control-path time is never added twice."""
    if staging and (threads != [1] or log_size):
        raise ValueError("staging calibration requires one scalar copy curve")
    groups = defaultdict(list)
    for row in records:
        if (
            row["phase"] == "train"
            and row["mode"] == "real"
            and row.get("concurrency", 1) == 1
            and row.get("api") != "register"
        ):
            key = (
                row["effective_threads"],
                row.get("model_bytes", row["block_bytes"] * row["batch"]),
                row.get("segments", row["batch"]),
            )
            groups[key].append(row["elapsed_ns"])
    width = 3 + 2 * log_size
    curves = {}
    stage_curve = {}
    for p in threads:
        cases = [
            (b, n, statistics.median(v)) for (t, b, n), v in groups.items() if t == p
        ]
        x = np.array([[1, b / 1048576, n] for b, n, _ in cases])
        y = np.array([ns for _, _, ns in cases])
        if (
            len(cases) < 3
            or not np.all(np.isfinite(y) & (y > 0))
            or np.linalg.matrix_rank(x) < 3
        ):
            raise ValueError(f"insufficient or invalid training data for {p} threads")
        columns = list(range(3))
        if log_size:
            logs = np.array([np.log2(max(1, b / n / 4096)) for b, n, _ in cases])
            # Separate serial per-I/O latency from a shared batch completion cost.
            features = np.column_stack((x, logs * x[:, 2], logs))
            for column in (3, 4):
                candidate = features[:, columns + [column]]
                if np.linalg.matrix_rank(candidate) == candidate.shape[1]:
                    columns.append(column)
                    x = candidate
        best = fit_nonnegative(x, y)
        if best is None:
            raise ValueError(f"no valid fit for {p} threads")
        if staging:
            # Identify a cache-to-DRAM slope change only from real TRAIN sizes.
            # Two distinct sizes on each side and full rank prevent unidentifiable knees.
            for knee in sorted({b for b, _, _ in cases})[2:-2]:
                features = np.column_stack((x, np.maximum(0, x[:, 1] - knee / 1048576)))
                if np.linalg.matrix_rank(features) != features.shape[1]:
                    continue
                candidate = fit_nonnegative(features, y)
                if (
                    candidate is not None
                    and candidate[1][-1] > 0
                    and candidate[0] < best[0] - 1e-6
                ):
                    best = candidate
                    stage_curve = {
                        "size_knee_bytes": knee,
                        "large_size_ns_per_byte": candidate[1][-1] / 1048576,
                    }
        curves[p] = np.zeros(width)
        curves[p][columns] = best[1][: len(columns)]
    # Interpolate unmeasured thread counts in time/byte, without extrapolating.
    values = [
        np.interp(range(1, max(threads) + 1), threads, [curves[p][i] for p in threads])
        for i in range(width)
    ]
    curve = {
        "channel": "host-memory",
        "latency_ns": values[0].tolist(),
        "copy_latency_ns": values[2].tolist(),
        # A zero fitted size cost is unresolved bandwidth, encoded without JSON infinity.
        "bandwidth_bytes_per_second": np.divide(
            1048576e9,
            values[1],
            out=np.full(len(values[1]), np.finfo(float).max),
            where=values[1] > 0,
        ).tolist(),
    }
    if log_size:
        for name, value in zip(
            ("log_size_latency_ns", "batch_log_size_latency_ns"), values[3:]
        ):
            if np.any(value > 0):
                curve[name] = value.tolist()
    return {**curve, **stage_curve}


def fit_paths(records, threads, base, *, ssd_io="posix"):
    """Keep unmeasured estimates; fit each engine path without mixing implementations."""
    paths = dict(base)
    measured = {
        row.get("path", "cpu_cpu")
        for row in records
        if row["phase"] == "train"
        and row["mode"] == "real"
        and row.get("concurrency", 1) == 1
    }
    for name in sorted(measured):
        selected = [row for row in records if row.get("path", "cpu_cpu") == name]
        curve = fit_profile(
            selected,
            threads if name == "cpu_cpu" else [1],
            log_size=name
            in ("file_cpu", "file_gpu", "file_cpu_staged", "file_cpu_staged_no_crc"),
            staging=name in ("cpu_staging", "cpu_staging_read"),
        )
        if name in ("cpu_gpu", "gpu_cpu"):
            curve["interleave_copies"] = True
        base_name = name.removesuffix("_no_crc")
        fallback = (
            base.get("cpu_staging", base.get("cpu_cpu", {}))
            if name.startswith("cpu_staging")
            else base.get(
                base_name,
                base.get(
                    base_name.removesuffix("_gather").removesuffix("_staged"),
                    {},
                ),
            )
        )
        curve["channel"] = base.get(name, fallback).get("channel", CHANNELS[name])
        if name != "cpu_cpu":
            curve = {
                key: value[0] if isinstance(value, list) else value
                for key, value in curve.items()
            }
        registrations = [
            row["elapsed_ns"]
            for row in records
            if row.get("path") == name
            and row.get("api") == "register"
            and row["phase"] == "train"
            and row["mode"] == "real"
            and row.get("concurrency", 1) == 1
        ]
        if registrations:
            curve["registration_latency_ns"] = statistics.median(registrations)
        if ssd_io == "io_uring" and name in (
            "file_cpu_staged",
            "file_cpu_staged_no_crc",
        ):
            curve["serialize_latency"] = (
                True  # SSDTier holds io_mu_ through native completion.
            )
        paths[name] = curve
    return paths


def compare(records):
    fields = (
        "path",
        "staged",
        "gap",
        "phase",
        "threads",
        "concurrency",
        "block_bytes",
        "batch",
        "api",
        "caller_gpu",
    )
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row["phase"] != "train":
            defaults = dict(
                path="cpu_cpu", staged=0, gap=0, concurrency=1, caller_gpu=0
            )
            key = tuple(row.get(k, defaults.get(k)) for k in fields)
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
        "--paths",
        nargs="+",
        default=["cpu_cpu"],
        choices=["all", *TRANSFER_PATHS],
    )
    parser.add_argument("--device", type=int, default=0, help="local GPU ordinal")
    parser.add_argument(
        "--ssd-dir", type=Path, help="writable SSD directory; never use NFS/tmpfs"
    )
    parser.add_argument("--ssd-io", choices=("posix", "io_uring"), default="posix")
    parser.add_argument(
        "--ssd-durability", choices=("relaxed", "strict"), default="strict"
    )
    parser.add_argument("--ssd-no-crc", action="store_true")
    parser.add_argument(
        "--ssd-read-prefill-mib",
        type=int,
        default=0,
        help="write filler after seeding read fixtures; API reads retain data instead of rewriting",
    )
    parser.add_argument(
        "--ssd-settle-ms",
        type=int,
        default=10,
        help="read seed settling before engine timing",
    )
    parser.add_argument(
        "--ssd-api-settle-ms",
        type=int,
        default=0,
        help="wait after offload before API load; 0 keeps immediate reload",
    )
    parser.add_argument(
        "--base-profile",
        type=Path,
        default=Path(__file__).with_name("sim_profile.json"),
        help="preserve unmeasured paths as explicitly reported estimates",
    )
    parser.add_argument("--remote-probe", type=Path)
    parser.add_argument("--local-address", help="local RDMA IPv4 address")
    parser.add_argument("--remote-address", help="owner RDMA IPv4 address")
    parser.add_argument(
        "--remote-command",
        help="owner launcher command, including probe executable; owner arguments are appended",
    )
    parser.add_argument("--port", type=int, default=59000)
    parser.add_argument(
        "--output", type=Path, required=True, help="new output directory"
    )
    parser.add_argument(
        "--cpus",
        help="CPU IDs, comma separated; CPU copies use first N, GPU/RDMA use the full set",
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
    parser.add_argument(
        "--probe-timeout", type=int, default=300, help="seconds per probe"
    )
    args = parser.parse_args()
    paths = (
        list(TRANSFER_PATHS) if "all" in args.paths else list(dict.fromkeys(args.paths))
    )
    remote = any(p.startswith("remote_") for p in paths)
    ssd = any("file" in p for p in paths)
    if ssd and args.ssd_dir is None:
        parser.error(
            "SSD paths require --ssd-dir on the storage device being calibrated"
        )
    if (
        min(args.ssd_settle_ms, args.ssd_api_settle_ms, args.ssd_read_prefill_mib) < 0
        or max(args.ssd_settle_ms, args.ssd_api_settle_ms, args.ssd_read_prefill_mib)
        > 2147483647
    ):
        parser.error("SSD delays and read prefill must be nonnegative 32-bit integers")
    if args.device < 0 or not 1 <= args.port <= 65535:
        parser.error("need a nonnegative GPU ordinal and valid TCP port")
    if remote and not all(
        (args.local_address, args.remote_address, args.remote_command)
    ):
        parser.error(
            "remote paths need --local-address, --remote-address and --remote-command"
        )
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
        or min(args.samples, args.rounds, args.probe_timeout) <= 0
    ):
        parser.error(
            "need valid distinct CPUs, threads including 1, and positive samples/rounds/timeout"
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
    remote_probe = (
        (args.remote_probe or probe.with_name("umbp_sim_remote_probe")).resolve(
            strict=True
        )
        if remote
        else None
    )
    base_paths = json.loads(args.base_profile.read_text())["paths"]
    env = dict(
        os.environ, MORI_UMBP_LOG_LEVEL="warn", UMBP_HBM_GATHER_PREWARM_STREAMS="0"
    )
    records = []
    args.output.mkdir(parents=True, exist_ok=False)

    def collect(
        phase,
        mode,
        round_id,
        p,
        c=1,
        profile=None,
        path="cpu_cpu",
        staged=False,
        caller_gpu=False,
    ):
        print(
            f"{path} {phase} {mode}: round {round_id + 1}, {p} threads, {c} callers, staged={staged}",
            flush=True,
        )
        command = [
            "taskset",
            "-c",
            ",".join(map(str, cpus[: p * c] if path == "cpu_cpu" else cpus)),
            str(probe),
            phase,
            args.cache,
            str(args.samples),
            "--concurrency",
            str(c),
            "--max-mib",
            str(args.max_mib),
        ]
        if path == "remote":
            command[3:] = [
                str(remote_probe),
                "caller",
                args.local_address,
                args.remote_address,
                str(args.port),
                phase,
                str(args.samples),
                "--concurrency",
                str(c),
                "--max-mib",
                str(args.max_mib),
            ]
            if staged:
                command.append("--staged")
        else:
            command.extend(["--path", path, "--device", str(args.device)])
            if "file" in path:
                command.extend(
                    [
                        "--ssd-dir",
                        str(args.ssd_dir),
                        "--ssd-io",
                        args.ssd_io,
                        "--ssd-durability",
                        args.ssd_durability,
                        "--ssd-settle-ms",
                        str(args.ssd_settle_ms),
                        "--ssd-api-settle-ms",
                        str(args.ssd_api_settle_ms),
                        "--ssd-read-prefill-mib",
                        str(args.ssd_read_prefill_mib),
                    ]
                )
                if args.ssd_no_crc:
                    command.append("--ssd-no-crc")
                if caller_gpu:
                    command.append("--gpu-caller")
        for option, values in workloads[
            "train" if phase == "train" else "validation"
        ].items():
            if values is not None:
                command.extend([f"--{option}", ",".join(map(str, values))])
        if profile:
            command.append(str(profile))
        owner = None
        process_env = dict(
            env, UMBP_DRAM_COPY_THREADS=str(p), UMBP_DRAM_WRITE_THREADS=str(p)
        )
        with (args.output / "owner.log").open("a") as log:
            try:
                if path == "remote":
                    owner_args = ["owner", args.remote_address, *command[6:]]
                    owner = subprocess.Popen(
                        [
                            "taskset",
                            "-c",
                            ",".join(map(str, cpus)),
                            "sh",
                            "-c",
                            args.remote_command + " " + shlex.join(owner_args),
                        ],
                        start_new_session=True,
                        stdout=log,
                        stderr=log,
                        env=process_env,
                    )
                result = subprocess.run(
                    command,
                    text=True,
                    stdout=subprocess.PIPE,
                    check=True,
                    env=process_env,
                    timeout=args.probe_timeout,
                )
                if owner and owner.wait(timeout=30) != 0:
                    raise RuntimeError("remote owner failed; see owner.log")
            finally:
                if owner and owner.poll() is None:
                    os.killpg(owner.pid, signal.SIGTERM)
                    try:
                        owner.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(owner.pid, signal.SIGKILL)
                        owner.wait()
        lines = result.stdout.splitlines()
        header = next(
            i for i, line in enumerate(lines) if line.startswith("block_bytes,batch,")
        )
        rows = list(csv.DictReader(lines[header:]))
        if not rows:
            raise ValueError("probe returned no measurements")
        for row in rows:
            row = {k: v if k in ("api", "path") else int(v) for k, v in row.items()}
            row.setdefault("path", path)
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
            shape = (
                row["path"].removesuffix("_gather"),
                row["block_bytes"],
                row["batch"],
                row.get("gap", 0),
            )
            if phase == "train":
                training_shapes.add(shape)
            elif shape in training_shapes:
                raise ValueError(
                    f"training and validation overlap at block_bytes,batch={shape}"
                )
            records.append(dict(row, phase=phase, mode=mode, round=round_id, threads=p))

    local_paths = [p for p in paths if not p.startswith("remote_")]
    if (
        "file_cpu_staged" in local_paths
        and not args.ssd_no_crc
        and "file_cpu_staged_no_crc" not in local_paths
    ):
        local_paths.append("file_cpu_staged_no_crc")
    for r in range(args.rounds):
        for path in local_paths:
            for p in threads if path == "cpu_cpu" else [1]:
                collect("train", "real", r, p, path=path)
        if remote:
            collect("train", "real", r, 1, path="remote")
    created = datetime.now(timezone.utc).isoformat()
    measured_paths = {row["path"] for row in records}
    profile = {
        "id": f"umbp-measured-{','.join(sorted(measured_paths))}-estimates-{len(set(base_paths) - measured_paths)}-{created}",
        "paths": fit_paths(records, threads, base_paths, ssd_io=args.ssd_io),
    }
    with tempfile.TemporaryDirectory(dir=args.output.resolve()) as temporary:
        candidate = Path(temporary) / "profile.json"
        candidate.write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n")
        for r in range(args.rounds):
            for path in local_paths:
                for p, c, phase in itertools.product(
                    threads if path == "cpu_cpu" else [1],
                    concurrency,
                    ("engine", "api"),
                ):
                    # The staged CPU API uses the same SSD client configuration.
                    if phase == "api" and path in (
                        "file_cpu",
                        "cpu_file",
                        "file_cpu_staged_no_crc",
                    ):
                        continue
                    for gpu in (
                        (False, True)
                        if phase == "api" and path.endswith("_staged")
                        else (False,)
                    ):
                        for mode in ("real", "sim") if r % 2 == 0 else ("sim", "real"):
                            collect(
                                phase,
                                mode,
                                r,
                                p,
                                c,
                                candidate if mode == "sim" else None,
                                path,
                                caller_gpu=gpu,
                            )
            if remote:
                for c, staged in itertools.product(concurrency, (False, True)):
                    for mode in ("real", "sim") if r % 2 == 0 else ("sim", "real"):
                        collect(
                            "engine",
                            mode,
                            r,
                            1,
                            c,
                            candidate if mode == "sim" else None,
                            "remote",
                            staged,
                        )
        comparisons = compare(records)
        report = {
            "profile_id": profile["id"],
            "profile_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
            "base_profile_sha256": hashlib.sha256(
                args.base_profile.read_bytes()
            ).hexdigest(),
            "path_sources": {
                p: "measured" if p in measured_paths else "estimate"
                for p in profile["paths"]
            },
            "zero_byte_cost_paths": [
                name
                for name, curve in profile["paths"].items()
                if np.any(
                    np.asarray(curve["bandwidth_bytes_per_second"])
                    == np.finfo(float).max
                )
            ],
            "device": args.device,
            "ssd": (
                {
                    "directory": str(args.ssd_dir),
                    "io": args.ssd_io,
                    "durability": args.ssd_durability,
                    "verify_crc": not args.ssd_no_crc,
                    "tier_io_threads": next(
                        (
                            row["ssd_tier_threads"]
                            for row in records
                            if "ssd_tier_threads" in row
                        ),
                        None,
                    ),
                    "settle_ms": args.ssd_settle_ms,
                    "api_settle_ms": args.ssd_api_settle_ms,
                    "read_prefill_mib": args.ssd_read_prefill_mib,
                    "api_read_mode": (
                        "retained-after-prefill"
                        if args.ssd_read_prefill_mib
                        else "reload-after-offload"
                    ),
                }
                if ssd
                else None
            ),
            "remote_addresses": (
                [args.local_address, args.remote_address] if remote else None
            ),
            "cpus": cpus,
            "threads": threads,
            "concurrency": concurrency,
            "max_mib": args.max_mib,
            "workloads": workloads,
            "cache": args.cache,
            "samples": args.samples,
            "probe_timeout_seconds": args.probe_timeout,
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
