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
"""Under pytest-xdist, give each worker GPUs of its own; on a PR, skip ci_full.

The ``single_gpu`` cases run one kernel on device 0 with no cco, and the
``gpu_group`` cases launch 2 or 4 ranks, so either kind can run side by side --
but only if no two workers share a device. Each worker gets
MORI_TEST_GPUS_PER_WORKER (default 1) devices of its own. HIP reads
HIP_VISIBLE_DEVICES once, at the first torch.cuda call, and the test modules
make that call at import (their skipif marks), so the pin has to be set before
they are collected; the ranks a case launches inherit it.

ci_cco.yml sets MORI_CI_FULL=0 on a pull request without the ci:full label,
which skips the ``ci_full`` cases there.
"""

import os
from pathlib import Path

import pytest

_KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def _devices():
    visible = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    visible = os.environ.get("HIP_VISIBLE_DEVICES") or visible
    if visible:
        return visible.split(",")
    # Counted from KFD: asking torch would initialise HIP with every device
    # visible, and the pin could no longer take effect.
    ids = [p.read_text().strip() for p in _KFD_NODES.glob("*/gpu_id")]
    return [str(i) for i in range(sum(1 for i in ids if i not in ("", "0")))]


_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker:
    _all = _devices()
    if _all:
        _per = int(os.environ.get("MORI_TEST_GPUS_PER_WORKER", "1"))
        _first = int(_worker.removeprefix("gw")) * _per % len(_all)
        os.environ["HIP_VISIBLE_DEVICES"] = ",".join(_all[_first : _first + _per])


def pytest_configure(config):
    for line in (
        "single_gpu: one GPU, in-process, no cco; CI runs these under "
        "pytest-xdist with a GPU per worker",
        "gpu_group: launches 2 or 4 ranks; CI runs these under pytest-xdist "
        "with 4 GPUs per worker",
        "ci_full: skipped on a pull request unless it has the ci:full label",
    ):
        config.addinivalue_line("markers", line)


def pytest_collection_modifyitems(config, items):
    if os.environ.get("MORI_CI_FULL") != "0":
        return
    skip = pytest.mark.skip(
        reason="ci_full: runs on push to main, workflow_dispatch and ci:full"
    )
    for item in items:
        if item.get_closest_marker("ci_full"):
            item.add_marker(skip)
