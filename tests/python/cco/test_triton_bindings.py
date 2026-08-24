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

from pathlib import Path
import subprocess

import pytest

from mori.cco.device.bitcode import _sdma_enabled, find_cco_bitcode
from mori.cco.device.ops import CCO_DEVICE_FUNCTIONS
from mori.ir.triton import cco
from mori.jit.config import detect_build_config


def test_all_cco_device_symbols_are_exported_to_triton():
    assert len(CCO_DEVICE_FUNCTIONS) == 67
    assert set(CCO_DEVICE_FUNCTIONS).issubset(cco.__all__)
    for name in CCO_DEVICE_FUNCTIONS:
        assert callable(getattr(cco, name))
    assert all(meta["ret"] != "void" for meta in CCO_DEVICE_FUNCTIONS.values())


def test_cov5_bitcode_contains_every_enabled_wrapper_symbol():
    bitcode = find_cco_bitcode(cov=5)
    cfg = detect_build_config()
    llvm_nm = Path(cfg.opt).with_name("llvm-nm")
    if not llvm_nm.is_file():
        pytest.skip(f"llvm-nm not found next to {cfg.opt}")

    result = subprocess.run(
        [str(llvm_nm), "--defined-only", "--format=posix", bitcode],
        check=True,
        capture_output=True,
        text=True,
    )
    defined = {line.split()[0] for line in result.stdout.splitlines() if line}
    expected = {
        meta["symbol"]
        for meta in CCO_DEVICE_FUNCTIONS.values()
        if _sdma_enabled() or not meta["family"].startswith("sdma_")
    }
    assert expected <= defined


def test_flydsl_and_triton_share_the_same_scalar_abi():
    pytest.importorskip("flydsl")
    from mori.cco.device.flydsl import _bindings

    externs = [
        *_bindings.PUT.values(),
        *_bindings.PUT_VALUE.values(),
        *_bindings.GET.values(),
        *_bindings.SIGNAL.values(),
        *_bindings.WAIT_SIGNAL.values(),
        *_bindings.FLUSH.values(),
        *_bindings.FLUSH_PEER.values(),
        *_bindings.SDMA_XFER.values(),
        *_bindings.SDMA_QUIET.values(),
        *_bindings.SDMA_COMMIT.values(),
        _bindings.cco_sdma_quiet_queue,
        _bindings.cco_lsa_ptr,
        _bindings.cco_devcomm_rank,
        _bindings.cco_devcomm_world_size,
        _bindings.cco_devcomm_lsa_rank,
        _bindings.cco_devcomm_lsa_size,
        _bindings.cco_gda_read_signal,
        _bindings.cco_gda_reset_signal,
    ]
    by_symbol = {extern._symbol: extern for extern in externs}
    assert set(by_symbol) == {
        meta["symbol"] for meta in CCO_DEVICE_FUNCTIONS.values()
    }
    for meta in CCO_DEVICE_FUNCTIONS.values():
        extern = by_symbol[meta["symbol"]]
        assert extern._args == meta["args"]
        assert extern._ret == meta["ret"]
