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
"""Atomic accessors and fences for the gemm_ar kernels, on flydsl 0.2.x and later.

Buffer load/store with a working cache policy lives in
:mod:`mori.cco.device.flydsl.buffer` and the u32 signal-slot accessors in
:mod:`mori.cco.device.flydsl.signals`, both shared with the cco examples. They
are re-exported here so the kernels keep one import site. What stays here is
tuned for gemm_ar: the scope and ordering of its tile-counter atomics and the
copy-engine release fence.
"""

from __future__ import annotations

import flydsl
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl.expr import rocdl as _rocdl

from mori.cco.device.flydsl.buffer import (  # noqa: F401  (re-exported)
    CM_CACHED,
    CM_NT,
    CM_SC0,
    CM_SC0_SC1,
    CM_SC1,
    HAS_BUFFER_OPS,
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from mori.cco.device.flydsl.signals import (  # noqa: F401  (re-exported)
    _AGENT_SCOPE,
    i32_type,
    local_load_u32,
    local_store_u32,
    signal_load_u32,
    signal_ptr,
    signal_store_u32,
)

FLYDSL_VERSION = getattr(flydsl, "__version__", "unknown")


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
    lo = _rocdl.readfirstlane(i32_type(), fx.Uint32(v))
    hi = _rocdl.readfirstlane(i32_type(), fx.Uint32(v >> 32))
    return (fx.Uint64(hi) << 32) | fx.Uint64(lo)


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
    "CM_SC0",
    "CM_NT",
    "CM_SC1",
    "CM_SC0_SC1",
]
