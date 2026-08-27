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

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_PREFIX = "ROCM_IMPORT_ORDER_RESULT "


@pytest.mark.parametrize("order", ["mori_first", "torch_first"])
def test_mori_cco_and_torch_share_one_hip_runtime(order):
    child_code = f"""
import json
import os

if {order!r} == "mori_first":
    import mori.cco as cco
    import torch
else:
    import torch
    import mori.cco as cco

cco.Communicator.get_unique_id()
torch.cuda.is_available()

paths = set()
with open("/proc/self/maps", encoding="utf-8") as maps:
    for line in maps:
        fields = line.rstrip("\\n").split(maxsplit=5)
        if len(fields) < 6:
            continue
        path = fields[5]
        if path.endswith(" (deleted)"):
            path = path[:-10]
        if path.startswith("/") and os.path.basename(path).startswith("libamdhip64.so"):
            paths.add(os.path.realpath(path))

paths = sorted(paths)
if len(paths) != 1:
    raise RuntimeError(f"expected one libamdhip64 mapping, got {{paths}}")
print({RESULT_PREFIX!r} + json.dumps({{"order": {order!r}, "paths": paths}}))
"""
    env = os.environ.copy()
    env.pop("MORI_DISABLE_ROCM_SDK", None)
    env["PYTHONFAULTHANDLER"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout + result.stderr

    assert result.returncode not in (
        -signal.SIGABRT,
        128 + signal.SIGABRT,
    ), f"{order} aborted while loading ROCm:\n{output}"
    assert result.returncode == 0, output

    records = [
        json.loads(line.removeprefix(RESULT_PREFIX))
        for line in output.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    assert len(records) == 1, output
    assert records[0]["order"] == order
    assert len(records[0]["paths"]) == 1
