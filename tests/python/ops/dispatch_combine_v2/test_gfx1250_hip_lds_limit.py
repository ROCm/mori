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
"""EP4 HIP dispatch/combine at the 64x16 dispatch geometry on gfx1250, eager and graph.

At hidden 5120 bf16 this launch used to request more LDS than a workgroup has, so it
never got scheduled and every rank waited forever. A hang shows up here as a timeout.
"""
import os
import signal
import subprocess
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.join(_HERE, "bench_ep.py")
_NPROC = 4
_TIMEOUT = 300


def _arch():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    return ""


pytestmark = pytest.mark.skipif(
    _arch() != "gfx1250" or torch.cuda.device_count() < _NPROC,
    reason=f"needs {_NPROC} gfx1250 GPUs",
)


@pytest.mark.parametrize(
    "hidden,topk,epr", [(5120, 6, 96), (5120, 3, 32), (4096, 6, 96), (7168, 6, 96)]
)
def test_ep4_64x16_dispatch_completes(hidden, topk, epr):
    env = os.environ.copy()
    env.update(
        BACKENDS="hip",
        DISP="bf16",
        COMB_MODE="pull",
        HIDDEN=str(hidden),
        TOPK=str(topk),
        EPR=str(epr),
        DBN="64",
        DWPB="16",
        CBN="64",
        CWPB="8",
        MODES="eager,graph",
        SWEEP="64,512",
        ITERS="5",
    )
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={_NPROC}",
        _BENCH,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=_HERE,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=_TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        pytest.fail(f"hidden {hidden} 64x16 did not finish in {_TIMEOUT} s:\n{out}")
    assert proc.returncode == 0, f"rc={proc.returncode}\n{out}"
    assert "# PASS: 0 failed" in out, out
