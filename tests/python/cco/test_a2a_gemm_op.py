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
"""``A2aGemmOp``, the public wrapper, on 8 ranks.

``test_a2a_gemm.py`` covers the layout and drives the kernels through the
benchmark, which calls the builders directly; this file covers the op: its
per-M plans and chunk counts, the in-place ``send_buffer`` path, graph replay,
back-to-back calls under rank skew, and what the constructor refuses.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import pytest

from . import kfd_settle

torch = pytest.importorskip("torch")
pytest.importorskip("flydsl")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
WORKER = pathlib.Path(__file__).with_name("a2a_gemm_op_worker.py")
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
    env = {**os.environ, **(env or {})}
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    env.setdefault("MORI_ENABLE_SDMA", "1")
    result = kfd_settle.run(
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
@pytest.mark.parametrize("mode", ["fused", "split"])
def test_op_lands_at_the_fp8_floor(mode):
    """Four M on one instance (1, 2 and 4 bands, both wait modes), a call from
    ``send_buffer``, then graph replays with fresh inputs."""
    ranks = _run("--case", "correct", "--mode", mode)
    bad = [(i, r["checks"]) for i, r in enumerate(ranks) if not r["ok"]]
    assert not bad, bad


@requires_world
@pytest.mark.parametrize("mode", ["fused", "split"])
def test_back_to_back_with_a_late_rank(mode):
    """No host sync between calls, and a different rank 20 ms late each time."""
    bad = [
        (i, r["rels"])
        for i, r in enumerate(_run("--case", "skew", "--mode", mode))
        if not r["ok"]
    ]
    assert not bad, bad


@requires_world
def test_constructor_refuses_what_the_kernel_cannot_express():
    for r in _run("--case", "reject"):
        assert r["ok_args"] == "accepted", r
        assert r["k_not_whole_steps"].startswith("rejected: ValueError"), r
        assert r["n_not_tiled"].startswith("rejected: ValueError"), r
        assert r["bad_mode"].startswith("rejected: ValueError"), r


@requires_world
def test_constructor_refuses_a_communicator_without_sdma():
    """With SDMA off every put is a no-op and every band wait would hang."""
    for r in _run("--case", "reject", env={"MORI_ENABLE_SDMA": "0"}):
        assert r["ok_args"].startswith("rejected: RuntimeError"), r
