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

import os

import pytest

from .helper import run_triton_example


def test_triton_cco_proxy():
    if not (
        os.environ.get("MORI_CCO_TRITON_PROXY_DEVICE")
        or os.environ.get("MORI_RDMA_DEVICES")
    ):
        pytest.skip(
            "Proxy test needs MORI_CCO_TRITON_PROXY_DEVICE or MORI_RDMA_DEVICES"
        )
    from mori.jit.config import detect_gpu_arch

    try:
        arch = detect_gpu_arch()
    except RuntimeError as e:
        pytest.skip(f"Proxy test needs a GPU: {e}")
    # The ccoProxy host engine only accepts queues on wave64 gfx942/gfx950.
    if arch not in ("gfx942", "gfx950"):
        pytest.skip(f"ccoProxy needs gfx942/gfx950, not {arch}")
    output = run_triton_example("proxy", timeout=300)
    assert output.count("Proxy PASS") == 2
    assert "All CCO Triton proxy tests PASSED" in output
