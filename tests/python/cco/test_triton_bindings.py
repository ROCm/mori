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
import re
import subprocess

import pytest
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError

from mori.cco.device.bitcode import _sdma_enabled, find_cco_bitcode
from mori.cco.device.ops import (
    CCO_DEVICE_FUNCTIONS,
    CCO_DEVICE_FUNCTIONS_BY_SYMBOL,
)
from mori.ir.triton import cco
from mori.jit.config import detect_build_config


@triton.jit
def invalid_constexpr_axis_kernel(
    dev_comm,
    window,
    axis: tl.constexpr,
):
    if axis == 0:
        cco.Gda.get(dev_comm, 0, window, 0, window, 0, 8, coop=99)
    elif axis == 1:
        cco.Gda.get(dev_comm, 0, window, 0, window, 0, 8, thread_mode=99)
    elif axis == 2:
        cco.Gda.signal(dev_comm, 0, signal_op=99)
    else:
        cco.Sdma.quiet(dev_comm, 0, coop=-1)


def test_all_cco_device_symbols_are_exported_to_triton():
    assert len(CCO_DEVICE_FUNCTIONS) == 68
    assert set(CCO_DEVICE_FUNCTIONS).issubset(cco.__all__)
    for name in CCO_DEVICE_FUNCTIONS:
        assert callable(getattr(cco, name))
    assert all(meta["ret"] != "void" for meta in CCO_DEVICE_FUNCTIONS.values())


def test_triton_facades_match_flydsl_handle_names():
    assert {"rank", "world_size", "lsa_rank", "lsa_size"} <= set(vars(cco.DevComm))
    assert {"lsa_ptr"} <= set(vars(cco.Window))
    assert {
        "put",
        "put_value",
        "get",
        "signal",
        "read_signal",
        "reset_signal",
        "wait_signal",
        "flush",
        "flush_peer",
    } <= set(vars(cco.Gda))
    assert {"put", "get", "commit", "quiet", "quiet_queue"} <= set(vars(cco.Sdma))


@pytest.mark.parametrize(
    "axis,error",
    [
        (0, "GDA coop must be"),
        (1, "GDA thread_mode must be"),
        (2, "GDA signal requires"),
        (3, "SDMA coop must be"),
    ],
)
def test_triton_facades_reject_invalid_constexpr_axes(axis, error):
    with pytest.raises(CompilationError) as exc_info:
        invalid_constexpr_axis_kernel.warmup(
            0,
            0,
            axis=axis,
            grid=(1,),
            extern_libs=cco.get_extern_libs(),
            num_warps=1,
        )
    messages = []
    exc = exc_info.value
    while exc is not None:
        messages.append(str(exc))
        exc = exc.__cause__
    assert error in "\n".join(messages)


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


def test_cov5_bitcode_return_types_match_the_declared_abi():
    bitcode = find_cco_bitcode(cov=5)
    cfg = detect_build_config()
    llvm_dis = Path(cfg.opt).with_name("llvm-dis")
    if not llvm_dis.is_file():
        pytest.skip(f"llvm-dis not found next to {cfg.opt}")

    result = subprocess.run(
        [str(llvm_dis), "-o", "-", bitcode],
        check=True,
        capture_output=True,
        text=True,
    )
    pattern = re.compile(
        r"^\s*define\b.*?\b(?P<ret>void|i\d+)\s+" r"@(?P<symbol>[A-Za-z0-9_.$-]+)\("
    )
    definitions = {
        match.group("symbol"): match.group("ret")
        for line in result.stdout.splitlines()
        if (match := pattern.match(line))
    }
    expected = {
        symbol: "i32" if meta["ret"] == "int32" else "i64"
        for symbol, (_, meta) in CCO_DEVICE_FUNCTIONS_BY_SYMBOL.items()
        if _sdma_enabled() or not meta["family"].startswith("sdma_")
    }

    assert expected.keys() <= definitions.keys()
    assert {
        symbol: definitions[symbol]
        for symbol in expected
        if definitions[symbol] != expected[symbol]
    } == {}


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
        _bindings.cco_system_fence,
        _bindings.cco_devcomm_rank,
        _bindings.cco_devcomm_world_size,
        _bindings.cco_devcomm_lsa_rank,
        _bindings.cco_devcomm_lsa_size,
        _bindings.cco_gda_read_signal,
        _bindings.cco_gda_reset_signal,
    ]
    by_symbol = {extern._symbol: extern for extern in externs}
    assert set(by_symbol) == {meta["symbol"] for meta in CCO_DEVICE_FUNCTIONS.values()}
    for meta in CCO_DEVICE_FUNCTIONS.values():
        extern = by_symbol[meta["symbol"]]
        assert extern._args == meta["args"]
        assert extern._ret == meta["ret"]
