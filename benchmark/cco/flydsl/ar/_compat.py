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
"""Buffer load/store with working cache policy, on flydsl 0.2.x and 0.3.x.

flydsl 0.3.0 removed ``flydsl.expr.buffer_ops``. ``mori.ops.dispatch_combine_v2
.flydsl_compat`` already bridges that for EPv2, but it documents that
``cache_modifier`` is *accepted and ignored* on 0.3.0. For EPv2 that is a hint;
here it is a correctness requirement:

* the cross-GPU barrier spins on a slot a **peer** writes, so a cached load can
  sit on a stale L2 line forever, and
* the signal store must be visible past L1+L2 or no peer ever leaves the spin.

So the 0.3.0 path is built directly on ``rocdl.MakeBufferRsrcOp`` +
``rocdl.raw_ptr_buffer_{load,store}``, which do still carry an explicit ``aux``
cache-policy operand. Only the handful of primitives the all-reduce needs are
implemented; ``aiter/ops/flydsl/kernels/buffer_ops.py`` is the full-featured
version of the same idea and was the reference for the descriptor encoding.

Cache-policy (``aux``) bits on CDNA: SC0 = bypass L1, SC1 = bypass L2 -- the
same constants aiter's ``custom_all_reduce.cuh`` uses for ``start_sync`` /
``end_sync``.
"""

from __future__ import annotations

import flydsl
import flydsl.expr as fx
from flydsl._mlir import ir

FLYDSL_VERSION = getattr(flydsl, "__version__", "unknown")

# Cache-modifier (``aux``) bits. **These are CDNA3/CDNA4-dependent.**
#
# On gfx950 the encoding is bit0=SC0, bit1=NT, bit2=SC1, so the gfx942 spelling
# used by ``examples/cco/python/05_flydsl_lsa_allreduce`` (SC1=2, SC0|SC1=3)
# silently compiles to ``nt`` and ``sc0 nt`` here -- non-temporal hints that do
# **not** bypass L2. Verified by reading the emitted ISA: aux=3 produced
# ``buffer_store_dwordx4 ... sc0 nt``, aux=5 produces ``... sc0 sc1``.
#
# The examples get away with it because their inputs are host-written before the
# kernel, so nothing there actually depends on the barrier ordering.
CM_CACHED = 0
CM_SC0 = 1  # bypass L1
CM_NT = 2  # non-temporal
CM_SC1 = 4  # bypass L2 -- required to observe a peer's fresh write
CM_SC0_SC1 = CM_SC0 | CM_SC1  # 5: publish past L1+L2

# V# flags word for CDNA (gfx9xx): DATA_FORMAT=7, NUM_FORMAT=4. Bit 24 and
# OOB_SELECT are RDNA-only; this benchmark is gfx95x/gfx94x.
_CDNA_BUFFER_FLAGS = (7 << 12) | (4 << 15)
_NO_OOB_LIMIT = 0xFFFFFFFF

try:  # flydsl <= 0.2.x
    # ImportError, not ModuleNotFoundError: on 0.3.0 ``buffer_ops`` is neither a
    # submodule nor a name in ``flydsl.expr``, so this form raises the parent.
    from flydsl.expr import buffer_ops as _bo
    from flydsl.expr.typing import T as _T

    HAS_BUFFER_OPS = True

    def _dtype(name):
        return getattr(_T, name)

    create_buffer_resource_from_addr = _bo.create_buffer_resource_from_addr
    buffer_load = _bo.buffer_load
    buffer_store = _bo.buffer_store

except ImportError:  # flydsl >= 0.3.0
    from flydsl._mlir.dialects import arith as _arith
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr import rocdl as _rocdl
    from flydsl.expr.typing import T as _T

    HAS_BUFFER_OPS = False

    def _dtype(name):
        # 0.3.0 turned T.<dtype> from a factory into a property.
        attr = getattr(_T, name)
        return attr() if callable(attr) else attr

    def _raw(v):
        return v.ir_value() if hasattr(v, "ir_value") else v

    def _const(value, width):
        ty = ir.IntegerType.get_signless(width)
        return _arith.ConstantOp(ty, ir.IntegerAttr.get(ty, int(value))).result

    def _as_i32(v):
        if isinstance(v, int):  # literal element offsets, e.g. the signal slot
            return _const(v, 32)
        v = _raw(v)
        ty = v.type
        if isinstance(ty, ir.IndexType):
            return _arith.IndexCastOp(ir.IntegerType.get_signless(32), v).result
        if isinstance(ty, ir.IntegerType) and ty.width != 32:
            i32 = ir.IntegerType.get_signless(32)
            if ty.width > 32:
                return _arith.TruncIOp(i32, v).result
            return _arith.ExtSIOp(i32, v).result
        return v

    def create_buffer_resource_from_addr(addr_i64, *, num_records_bytes=None):
        """Raw i64 device address -> ``!llvm.ptr<8>`` buffer descriptor (V#)."""
        base = _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), _raw(addr_i64)).result
        nrec = (
            _NO_OOB_LIMIT
            if num_records_bytes is None
            else max(0, min(int(num_records_bytes), _NO_OOB_LIMIT))
        )
        return _rocdl.MakeBufferRsrcOp(
            ir.Type.parse("!llvm.ptr<8>"),
            base,
            _const(0, 16),  # stride: raw (non-structured) buffer
            _const(nrec, 64),
            _const(_CDNA_BUFFER_FLAGS, 32),
        ).result

    def _byte_offset(offset, elem_ty):
        elem_bytes = elem_ty.width // 8
        return _arith.MulIOp(_as_i32(offset), _const(elem_bytes, 32)).result

    def buffer_load(rsrc, offset, vec_width=4, dtype=None, cache_modifier=CM_CACHED):
        """Load ``vec_width`` x ``dtype`` at ELEMENT ``offset`` of ``rsrc``."""
        elem = dtype if dtype is not None else _dtype("f32")
        if hasattr(elem, "ir_type"):
            elem = elem.ir_type
        res = elem if vec_width == 1 else ir.VectorType.get([vec_width], elem)
        return _rocdl.raw_ptr_buffer_load(
            res,
            rsrc,
            _byte_offset(offset, elem),
            _const(0, 32),
            _const(cache_modifier, 32),
        )

    def buffer_store(data, rsrc, offset, cache_modifier=CM_CACHED):
        """Store ``data`` at ELEMENT ``offset`` of ``rsrc``."""
        value = _raw(data)
        ty = value.type
        try:  # this binding has no VectorType.isinstance; construction is the probe
            elem = ir.VectorType(ty).element_type
        except (ValueError, TypeError):
            elem = ty
        return _rocdl.raw_ptr_buffer_store(
            value,
            rsrc,
            _byte_offset(offset, elem),
            _const(0, 32),
            _const(cache_modifier, 32),
        )


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
#    CDNA4 (see the CM_* note above).
#
# Scopes mirror aiter's start_sync/end_sync: publish at **system** scope (the
# store must cross xGMI), poll at **agent** scope (we only read our own HBM).
from flydsl._mlir.dialects import llvm as _llvm_d  # noqa: E402
from flydsl.expr.typing import AddressSpace as _AS  # noqa: E402
from flydsl.expr.typing import PointerType as _PT  # noqa: E402
from flydsl.expr.typing import inttoptr as _inttoptr  # noqa: E402

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


def signal_ptr(addr_i64):
    """Raw i64 device address -> ``!llvm.ptr`` over one u32 signal slot.

    ``inttoptr`` yields a ``!fly.ptr``; the llvm dialect ops need the lowered
    ``!llvm.ptr``, hence the extra hop (same shape as aiter's
    ``kernels/mxfp4_gemm_common.py::_global_base_ptr1``).
    """
    return fx.to_llvm_ptr(
        _inttoptr(_PT.get(_dtype("i32"), _AS.Global), fx.Int64(addr_i64))
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
    "i32_type",
    "CM_CACHED",
    "CM_SC1",
    "CM_SC0_SC1",
]
