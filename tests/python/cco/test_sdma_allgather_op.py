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
"""``SdmaAllGatherOp`` on 2, 4 and 8 ranks, one torchrun spawn of
``sdma_allgather_op_worker.py`` per world size."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

WORKER = Path(__file__).with_name("sdma_allgather_op_worker.py")
#: SDMA queues of a finished process group take seconds to come back.
_SETTLE_SECONDS = 15
#: Marker of anvil's SDMA queue-creation failure while those queues come back.
_QUEUE_EXHAUSTED = "anvil.cpp"


def _spawn(world, last=False):
    time.sleep(_SETTLE_SECONDS)
    env = os.environ.copy()
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    env.setdefault("MORI_ENABLE_SDMA", "1")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world}",
            str(WORKER),
        ],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    out = result.stdout + result.stderr
    if result.returncode != 0 and _QUEUE_EXHAUSTED in out and not last:
        time.sleep(_SETTLE_SECONDS * 3)
        return _spawn(world, last=True)
    assert result.returncode == 0, out[-8000:]
    return {
        r["case"]: r["ranks"]
        for r in (
            json.loads(ln.removeprefix("RESULT_JSON "))
            for ln in out.splitlines()
            if ln.startswith("RESULT_JSON ")
        )
    }


@pytest.fixture(scope="module", params=[2, 4, 8])
def cases(request):
    world = request.param
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} GPUs")
    pytest.importorskip("flydsl")
    return _spawn(world)


def test_a_setup_step_failing_on_one_rank_fails_every_rank(cases):
    assert all(r["raised"] for r in cases["setup_fault"]), cases["setup_fault"]


def test_regions_chosen_per_call_gather_in_place(cases):
    assert all(r["ok"] and not r["faulted"] for r in cases["regions"]), cases["regions"]


def test_a_captured_pair_replays_on_new_data(cases):
    assert all(r["ok"] for r in cases["graph"]), cases["graph"]


def test_a_missing_peer_is_a_bounded_sticky_fault(cases):
    for r in cases["fault"]:
        if not r["missing"]:
            assert r["faulted"] and r["device_ok"], r
            assert r["give_up_s"] < 10 and r["sticky_s"] < 1, r
