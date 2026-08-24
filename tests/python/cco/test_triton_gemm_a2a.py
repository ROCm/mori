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
GEMM_A2A_DIR = REPO_ROOT / "benchmark" / "cco" / "triton" / "gemm_a2a"
sys.path.insert(0, str(GEMM_A2A_DIR))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


layout = _load("gemm_a2a_layout", GEMM_A2A_DIR / "layout.py")
compare = _load("gemm_a2a_compare", GEMM_A2A_DIR / "compare_opus.py")


def test_gemm_a2a_layout_indices():
    config = layout.GemmA2AConfig(
        world_size=4,
        m=256,
        n=1024,
        k=128,
        shard_n=256,
        block_m=64,
        block_n=64,
        block_k=32,
    )
    config.validate()
    assert config.scatter_n == 1024
    assert config.staging_index(3, 17, 29) == 3 * 256 * 256 + 17 * 256 + 29
    assert config.recv_index(2, 17, 29) == 2 * 256 * 256 + 17 * 256 + 29
    assert config.remote_bytes_per_rank == 3 * 256 * 256 * 2


def test_opus_result_parser():
    line = (
        "quad_gemm_a2a M=2048 output=direct schedule=serial shard_n=2560 "
        "strict_timing=1 persistent grid=304 avg_rank_time=0.6000 ms "
        "max_rank_time=0.6200 ms critical_rank=2 critical_compute_ms=0.5900 "
        "critical_comm_ms=0.0100 barrier_idle_residual_ms=0.0200 "
        "aggregate=3990.00 TFLOP/s SUCCESS"
    )
    result = compare.parse_opus_output(line)
    assert result["mode"] == "direct"
    assert result["max_rank_time_ms"] == pytest.approx(0.62)
    assert result["critical_rank"] == 2
    assert result["aggregate_tflops"] == pytest.approx(3990.0)


@pytest.mark.parametrize(
    "world_size,m,n,k,shard_n",
    [
        (2, 128, 512, 256, 256),
        (4, 128, 1024, 256, 256),
    ],
)
def test_triton_gemm_a2a_gpu_smoke(world_size, m, n, k, shard_n):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    env = os.environ.copy()
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        str(GEMM_A2A_DIR / "bench_gemm_a2a.py"),
        "--mode",
        "split-lsa",
        "-m",
        str(m),
        "-n",
        str(n),
        "-k",
        str(k),
        "--shard-n",
        str(shard_n),
        "--warmup",
        "1",
        "--iters",
        "2",
        "--block-m",
        "64",
        "--block-n",
        "64",
        "--block-k",
        "32",
        "--num-warps",
        "4",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    records = [
        json.loads(line.removeprefix("RESULT_JSON "))
        for line in output.splitlines()
        if line.startswith("RESULT_JSON ")
    ]
    assert len(records) == 1
    assert records[0]["validated"] is True
    assert records[0]["max_rank_time_ms"] > 0
