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
"""``GemmA2AOp``, the public wrapper, on 8 ranks.

``test_gemm_a2a.py`` covers the layout on the CPU; the kernels are exercised by
``bench_gemm_a2a.py``, which calls the builders directly and so says nothing
about this layer. The race test is the one this file exists for: it provokes
the recv-reuse race and shows the credit is what stops it, with the credit
switched off as the negative control.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flydsl")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
WORKER = pathlib.Path(__file__).with_name("gemm_a2a_op_worker.py")
WORLD = 8

#: SDMA queues from the previous 8-rank launch are not always reclaimed at
#: once (anvil.cpp:237). That is a resource race in the harness, not a defect
#: in the op, and the only failure worth one retry.
_QUEUE_EXHAUSTED = "anvil.cpp"
_SETTLE_SECONDS = 20

requires_world = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD,
    reason=f"needs {WORLD} GPUs",
)


def _spawn(*args: str, timeout: int = 900, last: bool = False, env=None):
    time.sleep(_SETTLE_SECONDS)
    env = {**os.environ, **(env or {})}
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    env.setdefault("MORI_ENABLE_SDMA", "1")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={WORLD}",
            str(WORKER),
            *args,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        if _QUEUE_EXHAUSTED in output and not last:
            return None
        raise AssertionError(output[-4000:])
    lines = [ln for ln in output.splitlines() if ln.startswith("RESULT_JSON ")]
    assert lines, output[-4000:]
    return json.loads(lines[-1].removeprefix("RESULT_JSON "))


def _run(*args: str, env=None):
    rec = _spawn(*args, env=env)
    if rec is None:
        time.sleep(_SETTLE_SECONDS * 3)
        rec = _spawn(*args, last=True, env=env)
    return rec["ranks"]


@requires_world
def test_self_test_sees_every_peer():
    assert all(r["ok"] for r in _run("--case", "self_test"))


@requires_world
@pytest.mark.parametrize("quant", ["ptpc", "mxfp8"])
@pytest.mark.parametrize("mode", ["fused", "split"])
def test_op_lands_at_the_fp8_floor(mode, quant):
    """Two M values on one instance, alternating, then graph replays with fresh
    inputs: every received block, from every source, against fp32."""
    ranks = _run("--case", "correct", "--mode", mode, "--quant", quant)
    bad = [(i, r["checks"]) for i, r in enumerate(ranks) if not r["ok"]]
    assert not bad, bad


@requires_world
@pytest.mark.parametrize("mode", ["fused", "split"])
def test_credit_keeps_a_slow_reader_intact(mode):
    """The victim reads call 1's result 50 ms late while its peers run call 2."""
    victim = next(r for r in _run("--case", "race", "--mode", mode) if r["victim"])
    assert victim["ok"], victim


@requires_world
@pytest.mark.parametrize("mode", ["fused", "split"])
def test_without_the_credit_the_race_is_real(mode):
    """Negative control: the same schedule with the credit off must corrupt the
    victim's unread recv. If this passes cleanly, the race test above proves
    nothing."""
    victim = next(
        r for r in _run("--case", "race", "--mode", mode, "--no-credit") if r["victim"]
    )
    assert not victim["ok"], victim


@requires_world
def test_constructor_refuses_what_would_return_wrong_bytes():
    """K below two K blocks, and more SDMA queues than the communicator has: both
    used to construct fine and return relL2 ~1.0 on every rank."""
    for r in _run("--case", "reject"):
        assert r["queues_on_comm"] == 1, r
        assert r["k128"].startswith("rejected: ValueError"), r
        assert r["one_queue"] == "accepted", r
        assert r["two_queues"].startswith("rejected: RuntimeError"), r


@requires_world
def test_constructor_refuses_a_communicator_without_sdma():
    for r in _run("--case", "reject", env={"MORI_ENABLE_SDMA": "0"}):
        assert r["queues_on_comm"] == 0, r
        assert r["one_queue"].startswith("rejected: RuntimeError"), r
