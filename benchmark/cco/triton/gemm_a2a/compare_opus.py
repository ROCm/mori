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
"""Compare Opus fused/split GEMM+A2A with Triton local/split-LSA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
TRITON_BENCH = Path(__file__).resolve().with_name("bench_gemm_a2a.py")
DEFAULT_OPUS_ROOT = Path("/workspace/gcnasm/opus_gemm_dist/opus_gemm_a2a_lsa")


def _number(pattern: str, text: str, default: float = 0.0) -> float:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else default


def parse_opus_output(output: str) -> dict:
    line = next(
        (line for line in output.splitlines() if line.startswith("quad_gemm_a2a ")),
        None,
    )
    if line is None or not line.endswith("SUCCESS"):
        raise ValueError(f"Opus result line not found or failed:\n{output}")
    mode = re.search(r"output=([^\s]+)", line).group(1)
    return {
        "impl": "opus",
        "mode": mode,
        "max_rank_time_ms": _number(r"max_rank_time=([\d.]+)", line),
        "avg_rank_time_ms": _number(r"avg_rank_time=([\d.]+)", line),
        "critical_rank": int(_number(r"critical_rank=(\d+)", line)),
        "critical_compute_ms": _number(r"critical_compute_ms=([\d.]+)", line),
        "critical_comm_ms": _number(r"critical_comm_ms=([\d.]+)", line),
        "barrier_idle_residual_ms": _number(
            r"barrier_idle_residual_ms=([\d.]+)", line
        ),
        "aggregate_tflops": _number(r"aggregate=([\d.]+)", line),
    }


def parse_triton_output(output: str) -> dict:
    prefix = "RESULT_JSON "
    line = next(
        (line for line in output.splitlines() if line.startswith(prefix)),
        None,
    )
    if line is None:
        raise ValueError(f"Triton result not found:\n{output}")
    return json.loads(line[len(prefix) :])


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


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    try:
        import mori

        package_dir = str(Path(mori.__file__).resolve().parent)
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            package_dir if not current else f"{package_dir}:{current}"
        )
    except ImportError:
        pass
    return env


def run_opus(args, mode: str) -> dict:
    executable = Path(args.opus_exe)
    command = [
        "mpirun",
        "--allow-run-as-root",
        "-np",
        "4",
        str(executable),
        "--output-mode",
        mode,
        "-m",
        str(args.m),
        "-n",
        str(args.n),
        "-k",
        str(args.k),
        "--shard-n",
        str(args.shard_n),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--strict-timing",
        "1",
    ]
    return parse_opus_output(run_command(command, base_env(), args.timeout))


def run_triton(args, mode: str) -> dict:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        str(TRITON_BENCH),
        "--mode",
        mode,
        "-m",
        str(args.m),
        "-n",
        str(args.n),
        "-k",
        str(args.k),
        "--shard-n",
        str(args.shard_n),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--block-m",
        str(args.block_m),
        "--block-n",
        str(args.block_n),
        "--block-k",
        str(args.block_k),
        "--num-warps",
        str(args.num_warps),
        "--num-stages",
        str(args.num_stages),
        "--loop-unroll",
        str(args.loop_unroll),
        "--tile-order",
        args.tile_order,
        "--group-m",
        str(args.group_m),
    ]
    return parse_triton_output(run_command(command, base_env(), args.timeout))


def median_record(records: list[dict]) -> dict:
    ordered = sorted(records, key=lambda item: item["max_rank_time_ms"])
    result = dict(ordered[len(ordered) // 2])
    result["repeats"] = len(records)
    result["all_max_rank_time_ms"] = [
        item["max_rank_time_ms"] for item in records
    ]
    return result


def markdown_report(records: list[dict]) -> str:
    opus_direct = next(
        item for item in records if item["impl"] == "opus" and item["mode"] == "direct"
    )
    opus_split = next(
        item
        for item in records
        if item["impl"] == "opus" and item["mode"] == "split-lsa"
    )
    lines = [
        "# Opus vs Triton GEMM+A2A",
        "",
        "| Implementation | Mode | Max rank (ms) | Compute (ms) | Comm (ms) | TFLOP/s | vs Opus Direct |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in records:
        ratio = item["max_rank_time_ms"] / opus_direct["max_rank_time_ms"]
        lines.append(
            f"| {item['impl']} | {item['mode']} | "
            f"{item['max_rank_time_ms']:.4f} | "
            f"{item.get('critical_compute_ms', 0.0):.4f} | "
            f"{item.get('critical_comm_ms', 0.0):.4f} | "
            f"{item['aggregate_tflops']:.2f} | {ratio:.3f}x |"
        )
    triton_split = next(
        item
        for item in records
        if item["impl"] == "triton" and item["mode"] == "split-lsa"
    )
    lines.extend(
        [
            "",
            f"- Opus fusion gain: split/direct = "
            f"{opus_split['max_rank_time_ms'] / opus_direct['max_rank_time_ms']:.3f}x.",
            f"- Triton split vs Opus split = "
            f"{triton_split['max_rank_time_ms'] / opus_split['max_rank_time_ms']:.3f}x.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opus-exe",
        default=str(DEFAULT_OPUS_ROOT / "build" / "quad_lsa_direct.exe"),
    )
    parser.add_argument("-m", type=int, default=2048)
    parser.add_argument("-n", type=int, default=18432)
    parser.add_argument("-k", type=int, default=8192)
    parser.add_argument("--shard-n", type=int, default=2560)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--block-m", type=int, default=256)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--loop-unroll", type=int, default=1)
    parser.add_argument(
        "--tile-order",
        choices=("linear", "grouped", "opus"),
        default="opus",
    )
    parser.add_argument("--group-m", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--json-out")
    parser.add_argument("--markdown-out")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not Path(args.opus_exe).is_file():
        raise FileNotFoundError(f"Opus executable not found: {args.opus_exe}")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")

    medians = []
    for impl, mode in (
        ("opus", "direct"),
        ("opus", "split-lsa"),
        ("opus", "local"),
        ("triton", "local"),
        ("triton", "split-lsa"),
    ):
        runs = []
        for repeat in range(args.repeats):
            print(f"[compare] {impl} {mode} run {repeat + 1}/{args.repeats}", flush=True)
            runs.append(
                run_opus(args, mode) if impl == "opus" else run_triton(args, mode)
            )
        medians.append(median_record(runs))

    report = markdown_report(medians)
    print(report, end="")
    payload = {"config": vars(args), "results": medians}
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.markdown_out:
        output = Path(args.markdown_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
