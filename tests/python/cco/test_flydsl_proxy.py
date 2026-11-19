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
"""Runs examples/cco/python/14_flydsl_proxy on two ranks."""

import importlib.util
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

import pytest


def test_flydsl_cco_proxy():
    for module in ("flydsl", "mpi4py"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"FlyDSL proxy example needs {module}")
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("FlyDSL proxy example needs mpirun")
    if not os.environ.get("MORI_RDMA_DEVICES"):
        pytest.skip("FlyDSL proxy example needs MORI_RDMA_DEVICES")
    from mori.jit.config import detect_gpu_arch

    try:
        arch = detect_gpu_arch()
    except RuntimeError as e:
        pytest.skip(f"FlyDSL proxy example needs a GPU: {e}")
    # The ccoProxy host engine only accepts queues on wave64 gfx942/gfx950.
    if arch not in ("gfx942", "gfx950"):
        pytest.skip(f"ccoProxy needs gfx942/gfx950, not {arch}")

    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "examples" / "cco" / "python" / "14_flydsl_proxy" / "main.py"
    env = os.environ.copy()
    env.setdefault("MORI_SOCKET_IFNAME", "lo")
    # Open MPI refuses root without these; others ignore them.
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    # Own session so a timeout also reaps ranks stuck in a signal wait.
    proc = subprocess.Popen(
        [mpirun, "-np", "2", sys.executable, str(script)],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        # TERM first: mpirun forwards it to ranks it may have put in other groups.
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            output, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            output, _ = proc.communicate()
        pytest.fail(f"FlyDSL proxy example timed out after 600 s\n{output}")
    assert proc.returncode == 0, output
    assert "All FlyDSL ccoProxy tests PASSED" in output, output
