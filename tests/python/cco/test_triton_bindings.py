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
    GDA_DOORBELL_TAG,
    GDA_FLUSH_DOORBELL_TAG,
    GdaDoorbell,
    GdaFlags,
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
    elif axis == 3:
        cco.Sdma.quiet(dev_comm, 0, coop=-1)
    else:
        cco.Sdma.put_ptr(
            dev_comm, 0, window, window, 8, coop=cco.CoopScope.WARP, same_queue=True
        )


@triton.jit
def sdma_constant_flags_kernel(dev_comm, dst, src):
    cco.Sdma.put_ptr(dev_comm, 1, dst, src, 4096, 0, remote_signal=True)


@triton.jit
def sdma_runtime_flags_kernel(dev_comm, dst, src, flags):
    cco.sdma_put_ptr_thread(dev_comm, 1, dst, src, 4096, 0, flags)


def test_sdma_constant_flags_fold_to_one_instantiation():
    # The handles pass SdmaFlags as a constant so the wrapper's dispatch over
    # every instantiation compiles away; a runtime `flags` keeps all of them.
    if not _sdma_enabled():
        pytest.skip("MORI was not built with BUILD_CCO_SDMA=ON")

    def instructions(kernel, *args):
        compiled = kernel.warmup(
            *args, grid=(1,), extern_libs=cco.get_extern_libs(), num_warps=1
        )
        return sum(
            1
            for line in compiled.asm["amdgcn"].splitlines()
            if line.startswith("\t") and not line.lstrip().startswith((".", ";"))
        )

    handle = (1 << 40) + 3
    constant = instructions(sdma_constant_flags_kernel, handle, handle, handle)
    runtime = instructions(sdma_runtime_flags_kernel, handle, handle, handle, 12345)
    assert constant * 4 < runtime, (constant, runtime)


@triton.jit
def gda_constant_flags_kernel(dev_comm, win):
    cco.GdaDeferred.put(
        dev_comm,
        1,
        win,
        0,
        win,
        0,
        4096,
        coop=cco.CoopScope.WARP,
        team=cco.Team.GDA,
    )


@triton.jit
def gda_runtime_flags_kernel(dev_comm, win, flags):
    cco.gda_put_iw_none_deferred(dev_comm, 0, 1, win, 0, win, 0, 4096, 0, 0, flags)


def test_gda_constant_flags_fold_to_one_instantiation():
    # Same contract as SDMA's: a constant GdaFlags word folds the wrapper's
    # dispatch over team (and its trap on other bits) away; a runtime one keeps
    # both instantiations. They differ only in the peer translation, so the gap
    # is small; the doorbell mode is in the symbol and never dispatched.
    def instructions(kernel, *args):
        compiled = kernel.warmup(
            *args, grid=(1,), extern_libs=cco.get_extern_libs(), num_warps=1
        )
        return sum(
            1
            for line in compiled.asm["amdgcn"].splitlines()
            if line.startswith("\t") and not line.lstrip().startswith((".", ";"))
        )

    handle = (1 << 40) + 3
    constant = instructions(gda_constant_flags_kernel, handle, handle)
    runtime = instructions(gda_runtime_flags_kernel, handle, handle, 12345)
    assert constant < runtime, (constant, runtime)


@triton.jit
def gda_doorbell_kernel(dev_comm, win, mode: tl.constexpr, op: tl.constexpr):
    if mode == 0:
        if op == 0:
            cco.Gda.put(dev_comm, 1, win, 0, win, 0, 64, signal_op=cco.SignalOp.INC)
        elif op == 1:
            cco.Gda.put_value(dev_comm, 1, win, 0, 7, coop=cco.CoopScope.WARP)
        elif op == 2:
            cco.Gda.get(dev_comm, 1, win, 0, win, 0, 64, coop=cco.CoopScope.BLOCK)
        elif op == 3:
            cco.Gda.flush(dev_comm, coop=cco.CoopScope.BLOCK)
        elif op == 4:
            cco.Gda.flush_peer(dev_comm, 1)
        else:
            cco.Gda.wait(dev_comm, cco.Gda.flush_async(dev_comm, 1))
    elif mode == 1:
        if op == 0:
            cco.GdaDeferred.put(
                dev_comm, 1, win, 0, win, 0, 64, signal_op=cco.SignalOp.INC
            )
        elif op == 1:
            cco.GdaDeferred.put_value(dev_comm, 1, win, 0, 7, coop=cco.CoopScope.WARP)
        elif op == 2:
            cco.GdaDeferred.get(
                dev_comm, 1, win, 0, win, 0, 64, coop=cco.CoopScope.BLOCK
            )
        elif op == 3:
            cco.GdaDeferred.flush(dev_comm, coop=cco.CoopScope.BLOCK)
        elif op == 4:
            cco.GdaDeferred.flush_peer(dev_comm, 1)
        else:
            cco.GdaDeferred.wait(dev_comm, cco.GdaDeferred.flush_async(dev_comm, 1))
    else:
        if op == 0:
            cco.GdaCallerOrdered.put(
                dev_comm, 1, win, 0, win, 0, 64, signal_op=cco.SignalOp.INC
            )
        elif op == 1:
            cco.GdaCallerOrdered.put_value(
                dev_comm, 1, win, 0, 7, coop=cco.CoopScope.WARP
            )
        elif op == 2:
            cco.GdaCallerOrdered.get(
                dev_comm, 1, win, 0, win, 0, 64, coop=cco.CoopScope.BLOCK
            )
        elif op == 3:
            cco.GdaCallerOrdered.flush(dev_comm, coop=cco.CoopScope.BLOCK)
        elif op == 4:
            cco.GdaCallerOrdered.flush_peer(dev_comm, 1)
        else:
            cco.GdaCallerOrdered.wait(
                dev_comm, cco.GdaCallerOrdered.flush_async(dev_comm, 1)
            )


_DOORBELL_OPS = [
    ("cco_gda_put__it__inc", GDA_DOORBELL_TAG),
    ("cco_gda_put_value__iw__none", GDA_DOORBELL_TAG),
    ("cco_gda_get__ib", GDA_DOORBELL_TAG),
    ("cco_gda_flush__block", GDA_FLUSH_DOORBELL_TAG),
    ("cco_gda_flush_peer__warp", GDA_FLUSH_DOORBELL_TAG),
    ("cco_gda_flush_async__thread", GDA_FLUSH_DOORBELL_TAG),
]


@pytest.mark.parametrize(
    "mode", [GdaDoorbell.IMMEDIATE, GdaDoorbell.DEFERRED, GdaDoorbell.CALLER_ORDERED]
)
@pytest.mark.parametrize("op", range(len(_DOORBELL_OPS)))
def test_triton_gda_namespace_picks_its_doorbell_symbol(mode, op):
    # Gda / GdaDeferred / GdaCallerOrdered each call their mode's wrapper; a
    # deferred handle flushes with the plain symbols.
    handle = (1 << 40) + 3
    compiled = gda_doorbell_kernel.warmup(
        handle,
        handle,
        mode=mode,
        op=op,
        grid=(1,),
        extern_libs=cco.get_extern_libs(),
        num_warps=1,
    )
    base, tags = _DOORBELL_OPS[op]
    called = set(re.findall(r'symbol = "(cco_gda_[^"]+)"', compiled.asm["ttir"]))
    assert base + tags[mode] in called, called


def test_gda_doorbell_symbols_are_registered_per_mode():
    by_symbol = CCO_DEVICE_FUNCTIONS_BY_SYMBOL
    data = ("gda_put", "gda_put_value", "gda_get")
    flush = ("gda_flush", "gda_flush_peer", "gda_flush_async")
    for name, meta in CCO_DEVICE_FUNCTIONS.items():
        if meta["family"] not in data + flush:
            continue
        tags = GDA_DOORBELL_TAG if meta["family"] in data else GDA_FLUSH_DOORBELL_TAG
        if any(t and meta["symbol"].endswith(t) for t in tags.values()):
            continue  # a moded variant, checked from its base
        for mode, tag in tags.items():
            moded = by_symbol[meta["symbol"] + tag][0]
            assert moded == name + tag[1:], (mode, moded)
    assert not any(s.endswith("__deferred") for s in by_symbol if "flush" in s)


def test_all_cco_device_symbols_are_exported_to_triton():
    assert len(CCO_DEVICE_FUNCTIONS) == 171
    assert set(CCO_DEVICE_FUNCTIONS).issubset(cco.__all__)
    for name in CCO_DEVICE_FUNCTIONS:
        assert callable(getattr(cco, name))
    assert all(meta["ret"] != "void" for meta in CCO_DEVICE_FUNCTIONS.values())


def test_triton_facades_match_flydsl_handle_names():
    assert {"rank", "world_size", "lsa_rank", "lsa_size"} <= set(vars(cco.DevComm))
    assert {"lsa_ptr"} <= set(vars(cco.Window))
    gda = {
        "put",
        "put_value",
        "get",
        "signal",
        "read_signal",
        "reset_signal",
        "wait_signal",
        "flush",
        "flush_peer",
        "flush_async",
        "wait",
    }
    assert gda <= set(vars(cco.Gda))
    assert gda <= set(dir(cco.GdaDeferred))
    # A CALLER_ORDERED handle has no signal, in FlyDSL a trace-time error.
    assert gda - {"signal"} <= set(vars(cco.GdaCallerOrdered))
    assert not hasattr(cco.GdaCallerOrdered, "signal")
    assert {
        "put",
        "get",
        "commit",
        "quiet",
        "quiet_queue",
        "put_ptr",
        "get_ptr",
        "wait_signal",
        "signal_slot",
    } <= set(vars(cco.Sdma))


@pytest.mark.parametrize(
    "axis,error",
    [
        (0, "GDA coop must be"),
        (1, "GDA thread_mode must be"),
        (2, "GDA signal requires"),
        (3, "SDMA coop must be"),
        (4, "SDMA same_queue requires"),
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
    # And the other way: a wrapper nobody registered is unreachable from
    # FlyDSL and Triton, so it must not ship silently.
    unregistered = {symbol for symbol in defined if symbol.startswith("cco_")} - {
        meta["symbol"] for meta in CCO_DEVICE_FUNCTIONS.values()
    }
    assert unregistered == set(), f"not in CCO_DEVICE_FUNCTIONS: {sorted(unregistered)}"


def test_cov5_bitcode_signatures_match_the_declared_abi():
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
        r"^\s*define\b[^@]*?\b(?P<ret>void|i\d+)\s+"
        r"@(?P<symbol>[A-Za-z0-9_.$-]+)\((?P<args>[^)]*)\)"
    )
    definitions = {
        match.group("symbol"): (
            match.group("ret"),
            [arg.split()[0] for arg in match.group("args").split(",") if arg.strip()],
        )
        for line in result.stdout.splitlines()
        if (match := pattern.match(line))
    }
    llvm_type = {"int32": "i32", "uint64": "i64"}
    expected = {
        symbol: (llvm_type[meta["ret"]], [llvm_type[arg] for arg in meta["args"]])
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

    externs = list(_bindings.ALL.values())
    by_symbol = {extern._symbol: extern for extern in externs}
    assert set(by_symbol) == {meta["symbol"] for meta in CCO_DEVICE_FUNCTIONS.values()}
    for meta in CCO_DEVICE_FUNCTIONS.values():
        extern = by_symbol[meta["symbol"]]
        assert extern._args == meta["args"]
        assert extern._ret == meta["ret"]


def test_flydsl_gda_doorbell_picks_symbols(monkeypatch):
    pytest.importorskip("flydsl")
    from mori.cco.device.flydsl import _internal, handles

    called = []
    monkeypatch.setattr(
        _internal._LazyExtern,
        "__call__",
        lambda self, *args: called.append((self._symbol, args)) or 0,
    )

    class FakeGda:  # Gda's methods on plain attributes; no trace needed
        dev_comm, ctx = 1, 0

        def __init__(self, doorbell):
            self.doorbell = doorbell

    for name in ("_ctx", "_db", "put", "put_value", "get", "signal"):
        setattr(FakeGda, name, getattr(handles.Gda, name))
    for name in ("flush", "flush_peer", "flush_async"):
        setattr(FakeGda, name, getattr(handles.Gda, name))

    team = handles.Team.GDA
    for mode, tag in GDA_DOORBELL_TAG.items():
        called.clear()
        gda = FakeGda(mode)
        gda.put(1, 0, 0, 0, 0, 64, signal_op=handles.SignalOp.INC, team=team)
        gda.put_value(1, 0, 0, 7, coop=handles.CoopScope.WARP, team=team)
        gda.get(1, 0, 0, 0, 0, 64, coop=handles.CoopScope.BLOCK, team=team)
        gda.flush_peer(1, team=team)
        gda.flush_async(1, team=team)
        gda.flush(coop=handles.CoopScope.BLOCK)
        flush_tag = GDA_FLUSH_DOORBELL_TAG[mode]
        assert [symbol for symbol, _ in called] == [
            f"cco_gda_put__it__inc{tag}",
            f"cco_gda_put_value__iw__none{tag}",
            f"cco_gda_get__ib{tag}",
            f"cco_gda_flush_peer__warp{flush_tag}",
            f"cco_gda_flush_async__thread{flush_tag}",
            f"cco_gda_flush__block{flush_tag}",
        ]
        # The flags word of every op that takes a peer carries the team only.
        assert all(args[-1] == GdaFlags.TEAM_GDA for _, args in called[:-1])
        if mode == GdaDoorbell.CALLER_ORDERED:
            with pytest.raises(ValueError, match="no signal"):
                gda.signal(1)
        else:
            gda.signal(1)
    with pytest.raises(ValueError, match="doorbell must be"):
        handles.DevComm.gda(None, 0, doorbell=3)


def test_cco_bitcode_cache_key_follows_ccqe(monkeypatch, tmp_path):
    """-DIONIC_CCQE changes the bitcode, so each mode gets its own cache dir."""
    from types import SimpleNamespace

    import mori.jit.cache as jit_cache
    import mori.jit.config as jit_config
    import mori.jit.core as jit_core
    from mori.cco.device import bitcode

    repo = Path(__file__).resolve().parents[3]
    monkeypatch.setattr(jit_config, "get_mori_source_root", lambda: repo)
    monkeypatch.setattr(
        jit_config, "detect_build_config", lambda: SimpleNamespace(arch="gfx950")
    )
    monkeypatch.setattr(jit_config, "detect_nic_type", lambda: "ionic")
    monkeypatch.setattr(bitcode, "_sdma_enabled", lambda: False)
    keys = []

    def fake_cache_dir(arch, paths, nic, profiler=False, *, cov=None, ccqe=False):
        keys.append(ccqe)
        cache_dir = tmp_path / f"{arch}_{nic}_cov{cov}_ccqe{int(ccqe)}"
        cache_dir.mkdir(exist_ok=True)
        (cache_dir / bitcode._BC_FILENAME).write_bytes(b"")  # cached: nothing compiles
        return cache_dir

    monkeypatch.setattr(jit_cache, "get_cache_dir", fake_cache_dir)
    paths = []
    for ccqe in (True, False):
        monkeypatch.setattr(jit_core, "is_ccqe_enabled", lambda on=ccqe: on)
        paths.append(bitcode._jit_compile(6))
    assert keys == [True, False]
    assert None not in paths and paths[0] != paths[1]
