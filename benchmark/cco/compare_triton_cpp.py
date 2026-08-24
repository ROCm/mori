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
"""Run matching C++/Triton CCO benchmarks and report performance deltas."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
TRITON_BENCH = REPO_ROOT / "benchmark" / "cco" / "triton" / "bench_p2p.py"
HEADER_RE = re.compile(r"^# p2p_(put|get)_(bw|latency) unidirection (.*)$")


def parse_size_label(value: str, unit: str) -> int:
    multiplier = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}[unit]
    return int(value) * multiplier


def parse_cpp_table(output: str) -> list[dict]:
    """Parse ``PrintPerfTable`` output into normalized result dictionaries."""

    rows: list[dict] = []
    context = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        header = HEADER_RE.match(line)
        if header:
            op, kind, fields_text = header.groups()
            fields = dict(re.findall(r"(\w+)=([^\s]+)", fields_text))
            context = {
                "impl": "cpp",
                "op": op,
                "metric": "bandwidth" if kind == "bw" else "latency",
                "transport": fields["transport"],
                "scope": fields["scope"],
                "grid": int(fields["grid"]),
                "threads": int(fields["block"]),
                "iters": int(fields["iters"]),
                "warmup": int(fields["warmup"]),
            }
            continue
        if context is None or not line or line.startswith(("size", "[", "#")):
            continue
        parts = line.split()
        if len(parts) < 6 or parts[1] not in ("B", "KB", "MB", "GB"):
            continue
        if "skip" in parts:
            continue
        size_bytes = parse_size_label(parts[0], parts[1])
        value = float(parts[5])
        row = dict(context)
        row.update(
            {
                "size_bytes": size_bytes,
                "value": value,
                "unit": "GB/s" if context["metric"] == "bandwidth" else "us",
            }
        )
        rows.append(row)
    return rows


def parse_triton_json(output: str) -> list[dict]:
    prefix = "RESULT_JSON "
    return [
        json.loads(line[len(prefix) :])
        for line in output.splitlines()
        if line.startswith(prefix)
    ]


def resolve_cpp_binary(name: str, cpp_dir: str | None) -> Path:
    candidates = []
    if cpp_dir:
        candidates.append(Path(cpp_dir) / name)
    candidates.append(REPO_ROOT / "build" / "benchmark" / name)
    try:
        import mori

        candidates.append(Path(mori.__file__).resolve().parent / "benchmarks" / "cco" / name)
    except ImportError:
        pass
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{name} not found; searched: {candidates}")


def benchmark_geometry(transport: str, metric: str) -> tuple[int, int]:
    return (1, 256) if metric == "latency" or transport == "sdma" else (32, 256)


def transport_env(base: dict[str, str], transport: str) -> dict[str, str]:
    env = base.copy()
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    if transport == "sdma":
        env["MORI_ENABLE_SDMA"] = "1"
        env["BUILD_CCO_SDMA"] = "ON"
    elif transport == "ibgda":
        devices = env.get("MORI_CCO_TRITON_GDA_DEVICE") or env.get(
            "MORI_RDMA_DEVICES", ""
        )
        if not devices:
            raise RuntimeError(
                "IBGDA needs MORI_CCO_TRITON_GDA_DEVICE or MORI_RDMA_DEVICES"
            )
        env["MORI_RDMA_DEVICES"] = devices.split(",", 1)[0]
        env["MORI_DISABLE_TOPO"] = "1"
    return env


def run_command(command: list[str], env: dict[str, str], timeout: int) -> str:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{output}"
        )
    return output


def run_cpp_case(args, transport: str, op: str, metric: str) -> list[dict]:
    suffix = "bw" if metric == "bandwidth" else "latency"
    binary = resolve_cpp_binary(f"cco_p2p_{op}_{suffix}", args.cpp_dir)
    grid, threads = benchmark_geometry(transport, metric)
    min_size = args.bw_min if metric == "bandwidth" else args.lat_min
    max_size = args.bw_max if metric == "bandwidth" else args.lat_max
    command = [
        "mpirun",
        "--allow-run-as-root",
        "-np",
        "2",
        str(binary),
        "-t",
        transport,
        "-s",
        "block",
        "-b",
        min_size,
        "-e",
        max_size,
        "-f",
        str(args.step_factor),
        "-n",
        str(args.iters),
        "-w",
        str(args.warmup),
        "-c",
        str(grid),
        "-T",
        str(threads),
    ]
    output = run_command(command, transport_env(os.environ.copy(), transport), args.timeout)
    rows = parse_cpp_table(output)
    if not rows:
        raise RuntimeError(f"no C++ results parsed from:\n{output}")
    return rows


def run_triton_case(args, transport: str, op: str, metric: str) -> list[dict]:
    grid, threads = benchmark_geometry(transport, metric)
    min_size = args.bw_min if metric == "bandwidth" else args.lat_min
    max_size = args.bw_max if metric == "bandwidth" else args.lat_max
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(TRITON_BENCH),
        "--transport",
        transport,
        "--op",
        op,
        "--metric",
        metric,
        "--min-size",
        min_size,
        "--max-size",
        max_size,
        "--step-factor",
        str(args.step_factor),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--grid",
        str(grid),
        "--threads",
        str(threads),
    ]
    output = run_command(command, transport_env(os.environ.copy(), transport), args.timeout)
    rows = parse_triton_json(output)
    if not rows:
        raise RuntimeError(f"no Triton results parsed from:\n{output}")
    return rows


def compare_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (
            row["transport"],
            row["op"],
            row["metric"],
            row["scope"],
            row["size_bytes"],
        )
        grouped.setdefault(key, {})[row["impl"]] = row

    comparisons = []
    for key, implementations in sorted(grouped.items()):
        if "cpp" not in implementations or "triton" not in implementations:
            continue
        cpp = implementations["cpp"]
        triton = implementations["triton"]
        ratio = triton["value"] / cpp["value"]
        comparisons.append(
            {
                "transport": key[0],
                "op": key[1],
                "metric": key[2],
                "scope": key[3],
                "size_bytes": key[4],
                "unit": cpp["unit"],
                "cpp": cpp["value"],
                "triton": triton["value"],
                "ratio": ratio,
                "delta_pct": (ratio - 1.0) * 100.0,
            }
        )
    return comparisons


def markdown_report(comparisons: list[dict]) -> str:
    lines = [
        "# CCO Triton vs C++",
        "",
        "| Transport | Op | Metric | Size (B) | C++ | Triton | Ratio | Delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['transport']} | {row['op']} | {row['metric']} | "
            f"{row['size_bytes']} | {row['cpp']:.4f} {row['unit']} | "
            f"{row['triton']:.4f} {row['unit']} | {row['ratio']:.3f}x | "
            f"{row['delta_pct']:+.1f}% |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transports", default="lsa,sdma,ibgda")
    parser.add_argument("--ops", default="put,get")
    parser.add_argument("--metrics", default="latency,bandwidth")
    parser.add_argument("--lat-min", default="8")
    parser.add_argument("--lat-max", default="64K")
    parser.add_argument("--bw-min", default="64K")
    parser.add_argument("--bw-max", default="8M")
    parser.add_argument("--step-factor", type=int, default=2)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cpp-dir")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    transports = [item.strip() for item in args.transports.split(",") if item.strip()]
    ops = [item.strip() for item in args.ops.split(",") if item.strip()]
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    rows: list[dict] = []
    for transport in transports:
        for op in ops:
            for metric in metrics:
                print(f"[compare] C++ {transport} {op} {metric}", flush=True)
                rows.extend(run_cpp_case(args, transport, op, metric))
                print(f"[compare] Triton {transport} {op} {metric}", flush=True)
                rows.extend(run_triton_case(args, transport, op, metric))

    comparisons = compare_rows(rows)
    report = markdown_report(comparisons)
    print(report, end="")
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"results": rows, "comparisons": comparisons}, indent=2))
    if args.markdown_out:
        output = Path(args.markdown_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
