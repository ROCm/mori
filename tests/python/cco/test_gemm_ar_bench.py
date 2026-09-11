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
"""The fused GEMM+AR benchmark, run as a test: accuracy gates, timings recorded.

What is asserted and what is not, deliberately:

* **Accuracy is a gate.** ``validated`` and ``rel_l2`` come out of
  ``bench_gemm_ar.py``'s own host-side check and are hard assertions. A kernel
  that is fast and wrong fails here.
* **Absolute timings are only recorded.** CI runs on a shared runner, so a
  microsecond threshold would be flaky in both directions -- it would fail on a
  busy box and pass a real regression on an idle one. The numbers are printed
  (run with ``-s``) so a bisect has something to read.
* **One relative ordering is asserted**: ``fused-sdma`` must beat
  ``split-sdma``. That is the entire claim of the op, it is a 20%+ margin at the
  model's shape, and both sides are measured in the same process on the same
  box within seconds of each other -- so load affects them together.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH = REPO_ROOT / "benchmark" / "cco" / "flydsl" / "gemm_ar" / "bench_gemm_ar.py"

# Small enough that FlyDSL compiles it quickly, large enough that the fused path
# has more than one chunk per destination to publish (M / (ws * 128) = 4 bands).
SMOKE = dict(world_size=8, m=4096, n=1024, k=512)

# The DSV4-Pro wo_b prefill chunk the op was built for: a --chunked-prefill-size
# 16384 TP8 chunk, N=7168 K=2048.
MODEL_SHAPE = dict(world_size=8, m=16384, n=7168, k=2048)


def _run(world_size, mode, m, n, k, quant="blockscale", iters=11):
    env = os.environ.copy()
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    env.setdefault("MORI_ENABLE_SDMA", "1")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        str(BENCH),
        "--mode", mode,
        "--quant", quant,
        "-m", str(m), "-n", str(n), "-k", str(k),
        "--warmup", "3", "--iters", str(iters),
    ]
    result = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=1800
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    records = [
        json.loads(line.removeprefix("RESULT_JSON "))
        for line in output.splitlines()
        if line.startswith("RESULT_JSON ")
    ]
    assert len(records) == 1, output
    return records[0]


def _require(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")


@pytest.mark.parametrize("mode", ["gemm-only", "split-sdma", "fused-sdma"])
def test_mode_is_accurate_and_reports_a_time(mode):
    """Every mode computes the right thing, and says how long it took."""
    _require(SMOKE["world_size"])
    r = _run(mode=mode, **SMOKE)
    print(
        f"\n[gemm_ar] {mode:<11s} m={r['m']} n={r['n']} k={r['k']} "
        f"quant={r['quant']}  {r['max_rank_time_us']:9.1f} us  "
        f"relL2={r['rel_l2']:.2e}"
    )
    assert r["validated"] is True, f"{mode} failed the benchmark's own check"
    assert r["rel_l2"] < 3e-3, f"{mode} relL2 {r['rel_l2']:.3e} is past the fp8 floor"
    assert r["max_rank_time_us"] > 0


def test_fusing_beats_splitting_at_the_model_shape():
    """The op's whole claim, measured back to back on one box.

    Recorded on an idle 8x MI355X at this shape: fused-sdma 1146.2us against
    split-sdma 1465.5us (-21.8%), with the GEMM alone at 388.5us. The assertion
    is only the ordering -- see this module's docstring for why.
    """
    _require(MODEL_SHAPE["world_size"])
    fused = _run(mode="fused-sdma", **MODEL_SHAPE)
    split = _run(mode="split-sdma", **MODEL_SHAPE)

    f, s = fused["max_rank_time_us"], split["max_rank_time_us"]
    print(
        f"\n[gemm_ar] model shape m={MODEL_SHAPE['m']} n={MODEL_SHAPE['n']} "
        f"k={MODEL_SHAPE['k']}\n"
        f"           fused-sdma {f:9.1f} us  relL2={fused['rel_l2']:.2e}\n"
        f"           split-sdma {s:9.1f} us  relL2={split['rel_l2']:.2e}\n"
        f"           fused is {(1 - f / s) * 100:+.1f}% against split"
    )
    for name, r in (("fused-sdma", fused), ("split-sdma", split)):
        assert r["validated"] is True, f"{name} failed the benchmark's own check"
        assert r["rel_l2"] < 3e-3, f"{name} relL2 {r['rel_l2']:.3e}"
    assert f < s, (
        f"fused-sdma ({f:.1f} us) did not beat split-sdma ({s:.1f} us). "
        f"The recorded margin is 21.8%, so this is a real regression unless the "
        f"runner was heavily contended."
    )
