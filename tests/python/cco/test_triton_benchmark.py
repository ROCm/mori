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

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRITON_DIR = REPO_ROOT / "benchmark" / "cco" / "triton"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(TRITON_DIR))
bench = _load("cco_triton_bench", TRITON_DIR / "bench_p2p.py")
compare = _load(
    "cco_triton_compare",
    REPO_ROOT / "benchmark" / "cco" / "compare_triton_cpp.py",
)


def test_benchmark_size_and_metric_helpers():
    assert bench.parse_size("8") == 8
    assert bench.parse_size("64K") == 64 * 1024
    assert bench.parse_size("8MiB") == 8 * 1024 * 1024
    assert bench.size_sweep(8, 64, 2) == [8, 16, 32, 64]
    assert bench.latency_us(0.1, 10) == pytest.approx(10.0)
    assert bench.bandwidth_gbps(1_000_000, 1.0, 10) == pytest.approx(10.0)


def test_cpp_table_parser():
    output = """\
# p2p_put_bw unidirection transport=sdma scope=block grid=1 block=256 warpSize=64 units=1 iters=10 warmup=2
size       msg        scope       Bandwidth GB/s        Mpps
 64 KB     64 KB      block          17.250 GB/s      0.263
# p2p_get_latency unidirection transport=lsa scope=block grid=1 block=256 warpSize=64 units=1 iters=10 warmup=2
size       msg        scope         Latency us
  8 B       8 B       block           1.250 us
"""
    rows = compare.parse_cpp_table(output)
    assert rows == [
        {
            "impl": "cpp",
            "op": "put",
            "metric": "bandwidth",
            "transport": "sdma",
            "scope": "block",
            "grid": 1,
            "threads": 256,
            "iters": 10,
            "warmup": 2,
            "size_bytes": 64 * 1024,
            "value": 17.25,
            "unit": "GB/s",
        },
        {
            "impl": "cpp",
            "op": "get",
            "metric": "latency",
            "transport": "lsa",
            "scope": "block",
            "grid": 1,
            "threads": 256,
            "iters": 10,
            "warmup": 2,
            "size_bytes": 8,
            "value": 1.25,
            "unit": "us",
        },
    ]


@pytest.mark.parametrize(
    "transport,op,metric,size",
    [
        ("lsa", "put", "latency", "8"),
        ("sdma", "get", "bandwidth", "64K"),
    ],
)
def test_triton_benchmark_gpu_smoke(transport, op, metric, size):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    env = os.environ.copy()
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    env.setdefault("BUILD_CCO_SDMA", "ON")
    if transport == "sdma":
        try:
            from mori.cco.device._build_flags import BUILD_CCO_SDMA
        except ImportError:
            BUILD_CCO_SDMA = False
        if not BUILD_CCO_SDMA:
            pytest.skip("MORI was not built with BUILD_CCO_SDMA=ON")
        env["MORI_ENABLE_SDMA"] = "1"

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(TRITON_DIR / "bench_p2p.py"),
        "--transport",
        transport,
        "--op",
        op,
        "--metric",
        metric,
        "--min-size",
        size,
        "--max-size",
        size,
        "--iters",
        "5",
        "--warmup",
        "2",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    records = [
        json.loads(line.removeprefix("RESULT_JSON "))
        for line in output.splitlines()
        if line.startswith("RESULT_JSON ")
    ]
    assert len(records) == 1
    assert records[0]["value"] > 0
