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
"""Signal and atomic accessors for the gemm_ar kernels, on flydsl 0.2.x and later.

Buffer load/store with a working cache policy lives in
:mod:`mori.cco.device.flydsl.buffer`, shared with the cco examples, and is
re-exported here so the kernels keep one import site.
"""

from __future__ import annotations

import flydsl
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import rocdl as _rocdl

from mori.cco.device.flydsl.buffer import (  # noqa: F401  (re-exported)
    CM_CACHED,
    CM_NT,
    CM_SC0,
    CM_SC0_SC1,
    CM_SC1,
    HAS_BUFFER_OPS,
    _dtype,
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

FLYDSL_VERSION = getattr(flydsl, "__version__", "unknown")

# --- signal slot accessors -------------------------------------------------
#
# These are **atomic**, not plain buffer accesses with a cache modifier, for two
# independent reasons -- both found by reading the emitted ISA:
#
# 1. A ``rocdl.raw.ptr.buffer.load`` is side-effect-free, so the spin loop that
#    reloads it was dead-code-eliminated outright: the barrier compiled down to
#    a signal store and nothing else. An atomic is never removed.
# 2. The syncscope tells the AMDGPU backend which cache levels to bypass, so we
#    stop hand-encoding ``aux`` bits whose meaning changes between CDNA3 and
#    CDNA4 (see the CM_* note in mori.cco.device.flydsl.buffer).
#
# Scopes mirror aiter's start_sync/end_sync: publish at **system** scope (the
# store must cross xGMI), poll at **agent** scope (we only read our own HBM).
from flydsl._mlir.dialects import llvm as _llvm_d  # noqa: E402

_SYSTEM_SCOPE = ""  # LLVM AMDGPU: the empty syncscope is system-wide
_AGENT_SCOPE = "agent"


def wave_uniform_i64(addr):
    """Force a 64-bit address into scalar registers.

    ``Window.lsa_ptr`` is an opaque extern call, so even for a compile-time peer
    index the compiler cannot prove its result is wave-uniform. Building a
    buffer descriptor from a value it believes divergent makes the backend wrap
    **every** load and store in a readfirstlane waterfall loop, serialising each
    16-byte access lane by lane -- worth ~2.5x on this kernel, visible in the ISA
    as ``s_cbranch_execnz`` around each ``buffer_load_dwordx4``.

    Same idiom as aiter's
    ``kernels/flydsl_dispatch_combine_intranode_kernel.py::_wave_uniform_i64``.
    """
    v = fx.Uint64(addr)
    lo = _rocdl.readfirstlane(_dtype("i32"), fx.Uint32(v))
    hi = _rocdl.readfirstlane(_dtype("i32"), fx.Uint32(v >> 32))
    return (fx.Uint64(hi) << 32) | fx.Uint64(lo)


#: AMDGPU's LLVM address space for global memory. `fx.to_llvm_ptr` would
#: resolve this from the active backend, but it does not exist before flydsl
#: 0.3, and 0.2.x is still supported (nightly.yml's flydsl-compat job).
#: `llvm.inttoptr` is identical in
#: both -- same signature, same dialect module -- so going straight to an
#: `!llvm.ptr` keeps one code path instead of a version probe.
_GLOBAL_LLVM_AS = 1
_LLVM_GLOBAL_PTR = f"!llvm.ptr<{_GLOBAL_LLVM_AS}>"


def signal_ptr(addr_i64):
    """Raw i64 device address -> ``!llvm.ptr`` over one u32 signal slot.

    The llvm dialect ops below (atomic store/load/cmpxchg) need a lowered
    ``!llvm.ptr``, not the ``!fly.ptr`` that ``fly``'s own ``inttoptr`` yields.
    """
    return _llvm_d.inttoptr(
        ir.Type.parse(_LLVM_GLOBAL_PTR), fx.Int64(addr_i64).ir_value()
    )


def signal_store_u32(ptr, value):
    _llvm_d.store(
        fx.Int32(value).ir_value(),
        ptr,
        alignment=4,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope=_SYSTEM_SCOPE,
    )


def signal_load_u32(ptr):
    return _llvm_d.load(
        _dtype("i32"),
        ptr,
        alignment=4,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope=_AGENT_SCOPE,
    )


# The per-block ``_flag`` counter is private to this rank: only our own block
# reads and writes it, so it needs no atomic and no scope. Making it atomic had
# all 512 lanes issue an agent-scope atomic against one address every launch,
# which measurably serialised the prologue. ``volatile`` is still required so the
# load is not hoisted above the barrier that publishes the previous value.
def local_load_u32(ptr):
    return _llvm_d.load(_dtype("i32"), ptr, alignment=4, volatile_=True)


def local_store_u32(ptr, value):
    _llvm_d.store(fx.Int32(value).ir_value(), ptr, alignment=4, volatile_=True)


def atomic_add_u32(ptr, value, *, ordering="monotonic"):
    """Device-scope ``fetch_add``; returns the value *before* the add.

    Agent scope, because the tile counters it serves are only ever touched by
    blocks of the same kernel on this GPU -- a system-scope RMW would be a fabric
    round trip per tile.

    ``monotonic`` (relaxed), not ``acq_rel``, and that is worth 5% of the fused
    GEMM. An ordered RMW is bracketed by its own ``buffer_wbl2`` /
    ``buffer_inv``, and a thread trace showed those costing 91.6k cycles -- more
    per wave than the explicit release fence itself. Both halves are redundant
    here: the caller has already issued that release, and the winner never reads
    the data it is counting (the copy engine does), so it needs no acquire. Pass
    ``ordering="acq_rel"`` if a caller ever does need the fence.
    """
    return _llvm_d.atomicrmw(
        _llvm_d.AtomicBinOp.add,
        ptr,
        fx.Int32(value).ir_value(),
        getattr(_llvm_d.AtomicOrdering, ordering),
        syncscope=_AGENT_SCOPE,
        alignment=4,
    )


def atomic_xchg_u32(ptr, value, *, ordering="acquire"):
    """Device-scope exchange; returns the previous value. Used as a test-and-set."""
    return _llvm_d.atomicrmw(
        _llvm_d.AtomicBinOp.xchg,
        ptr,
        fx.Int32(value).ir_value(),
        getattr(_llvm_d.AtomicOrdering, ordering),
        syncscope=_AGENT_SCOPE,
        alignment=4,
    )


def atomic_store_u32(ptr, value, *, ordering="release"):
    _llvm_d.store(
        fx.Int32(value).ir_value(),
        ptr,
        alignment=4,
        ordering=getattr(_llvm_d.AtomicOrdering, ordering),
        syncscope=_AGENT_SCOPE,
    )


def release_fence(scope="agent"):
    """Release fence at an explicit scope, for producers of copy-engine input.

    cco only exposes ``cco_system_fence`` (``__threadfence_system``), which on
    gfx950 writes back through to memory. An SDMA engine reading a buffer we just
    wrote lives on the *same* device, so agent scope is the scope that actually
    describes the handoff; system scope over-synchronises against host and peers
    that are not the reader here.
    """
    _llvm_d.fence(_llvm_d.AtomicOrdering.release, syncscope=scope)


def i32_type():
    return _dtype("i32")


__all__ = [
    "FLYDSL_VERSION",
    "HAS_BUFFER_OPS",
    "buffer_load",
    "buffer_store",
    "create_buffer_resource_from_addr",
    "signal_ptr",
    "wave_uniform_i64",
    "signal_store_u32",
    "signal_load_u32",
    "local_load_u32",
    "local_store_u32",
    "atomic_add_u32",
    "release_fence",
    "atomic_xchg_u32",
    "atomic_store_u32",
    "i32_type",
    "CM_CACHED",
    "CM_SC1",
    "CM_SC0_SC1",
]
