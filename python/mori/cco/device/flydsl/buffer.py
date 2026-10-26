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
"""Buffer load/store with a working cache policy, on FlyDSL 0.2.x through 0.4.x.

flydsl 0.3.0 removed ``flydsl.expr.buffer_ops``. ``mori.ops.dispatch_combine_v2
.flydsl_compat`` bridges that for EPv2, but there ``cache_modifier`` is
*accepted and ignored* on 0.3.0 and later. That is fine for a hint and wrong
wherever the policy is the point -- reading what a **peer** wrote into our memory
needs a load that bypasses L2 (a remote store never invalidates our L2), and a
store meant for a peer must be visible past L1+L2:

* ``mori.ops.gemm_ar``'s loads of data a peer's copy engine delivered
  (``kernels_sdma``, ``CM_SC1``), and
* the cco FlyDSL examples, which load and store through LSA peer pointers.

gemm_ar's barrier and signal slots are atomics instead
(:mod:`mori.cco.device.flydsl.signals`) and use no ``aux`` bits.

So on 0.3.0 and later this is built directly on ``rocdl.MakeBufferRsrcOp`` +
``rocdl.raw_ptr_buffer_{load,store}``, which still carry an explicit ``aux``
cache-policy operand; on 0.2.x it is ``flydsl.expr.buffer_ops`` itself. Only
the primitives those callers need are implemented;
``aiter/ops/flydsl/kernels/buffer_ops.py`` is the full-featured version of the
same idea and was the reference for the descriptor encoding.

Offsets are in elements and are scaled to bytes in i32, as 0.2.x ``buffer_ops``
does, so a byte offset must stay below 2 GiB.
"""

from __future__ import annotations

import inspect

import flydsl
from flydsl._mlir import ir

# Cache-modifier (``aux``) values. This is the LLVM ``CPol`` operand, **not** the
# hardware bit positions, and the two do not line up -- which is worth stating
# because getting it wrong is silent.
#
# Established by assembling each policy and diffing the encoding::
#
#     echo 'buffer_store_short v0, v1, s[0:3], 0 offen <pol>' \
#       | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx950 -show-encoding
#
#     pol         encoding byte1/byte2      delta
#     (none)      0x10 0x68                 --
#     sc0         0x50 0x68                 byte1 bit6  -> insn bit 14
#     sc1         0x90 0x68                 byte1 bit7  -> insn bit 15
#     nt          0x10 0x6a                 byte2 bit1  -> insn bit 17
#
# Insn bit 14 is GLC, 15 is SCC, 17 is SLC, and CPol spells those 1, 16, 2. So
# SC1 is **16**, not 4 -- 4 is DLC, which is RDNA-only and silently ignored on
# CDNA. An earlier version of this file had ``CM_SC1 = 4`` and a comment
# claiming aux=5 emitted ``sc0 sc1``; it emits ``sc0`` alone, leaving the line
# dirty in L2. Nothing depended on it (the barrier uses atomics and the payload
# stores use CM_CACHED), but two experiments were invalidated before it showed
# up in the ISA.
#
# Note also that ``fx.rocdl.BufferCopy*(cache_modifier)`` does **not** take these
# values: its modifier is a two-value enum (0=cached, 2=nt). Reaching sc0/sc1
# needs ``buffer_store`` below, which passes aux through.
CM_CACHED = 0
CM_SC0 = 1  # bypass L1  (GLC)
CM_NT = 2  # non-temporal (SLC)
CM_SC1 = 16  # bypass L2 -- required to observe a peer's fresh write  (SCC)
CM_SC0_SC1 = CM_SC0 | CM_SC1  # 17: publish past L1+L2

# V# flags word for CDNA (gfx9xx): DATA_FORMAT=7, NUM_FORMAT=4. Bit 24 and
# OOB_SELECT are RDNA-only; the callers are gfx95x/gfx94x.
_CDNA_BUFFER_FLAGS = (7 << 12) | (4 << 15)
_NO_OOB_LIMIT = 0xFFFFFFFF

try:  # flydsl <= 0.2.x
    import flydsl.expr.buffer_ops as _bo
except ModuleNotFoundError as e:  # flydsl >= 0.3.0 removed the module
    if e.name != "flydsl.expr.buffer_ops":  # anything else is a broken install
        raise
    _bo = None

HAS_BUFFER_OPS = _bo is not None

if HAS_BUFFER_OPS:
    from flydsl.expr.typing import T as _T

    def _dtype(name):
        return getattr(_T, name)

    create_buffer_resource_from_addr = _bo.create_buffer_resource_from_addr
    buffer_load = _bo.buffer_load
    buffer_store = _bo.buffer_store

else:
    from flydsl._mlir.dialects import arith as _arith
    from flydsl._mlir.dialects import llvm as _llvm

    # The generated ops, not `flydsl.expr.rocdl`: from 0.3.1 that wraps the load
    # (only) and drops an `aux` that is neither int nor Attribute.
    from flydsl._mlir.dialects import rocdl as _rocdl
    from flydsl.expr.typing import T as _T

    def _dtype(name):
        # A property returning the type, or a factory returning it when called.
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

    # `aux` on the raw buffer ops is a positional i32 *value* on 0.3.0 and a
    # keyword-only *attribute* from 0.3.1 (checked against the 0.3.0 to 0.4.0
    # wheels). Probe once rather than pin, and refuse any other shape.
    def _aux_is_kwarg():
        param = inspect.Parameter
        kinds = {
            getattr(inspect.signature(op).parameters.get("aux"), "kind", None)
            for op in (_rocdl.raw_ptr_buffer_load, _rocdl.raw_ptr_buffer_store)
        }
        if kinds == {param.KEYWORD_ONLY}:
            return True
        if kinds == {param.POSITIONAL_OR_KEYWORD}:
            return False
        raise ImportError(
            f"flydsl {getattr(flydsl, '__version__', '?')}: unrecognised `aux` on "
            f"rocdl.raw_ptr_buffer_load/store ({kinds}); "
            "mori.cco.device.flydsl.buffer needs updating"
        )

    _AUX_IS_KWARG = _aux_is_kwarg()

    def _aux_args(cache_modifier):
        """Positional tail / kwargs for the raw buffer ops' ``aux`` operand."""
        if _AUX_IS_KWARG:
            return (), {
                "aux": ir.IntegerAttr.get(
                    ir.IntegerType.get_signless(32), int(cache_modifier)
                )
            }
        return (_const(cache_modifier, 32),), {}

    def buffer_load(rsrc, offset, vec_width=4, dtype=None, cache_modifier=CM_CACHED):
        """Load ``vec_width`` x ``dtype`` at ELEMENT ``offset`` of ``rsrc``.

        The byte offset is computed in i32, so it must stay below 2 GiB.
        """
        elem = dtype if dtype is not None else _dtype("f32")
        if hasattr(elem, "ir_type"):
            elem = elem.ir_type
        res = elem if vec_width == 1 else ir.VectorType.get([vec_width], elem)
        tail, kw = _aux_args(cache_modifier)
        return _rocdl.raw_ptr_buffer_load(
            res, rsrc, _byte_offset(offset, elem), _const(0, 32), *tail, **kw
        )

    def buffer_store(data, rsrc, offset, cache_modifier=CM_CACHED):
        """Store ``data`` at ELEMENT ``offset`` of ``rsrc``.

        The byte offset is computed in i32, so it must stay below 2 GiB.
        """
        value = _raw(data)
        ty = value.type
        try:  # this binding has no VectorType.isinstance; construction is the probe
            elem = ir.VectorType(ty).element_type
        except (ValueError, TypeError):
            elem = ty
        tail, kw = _aux_args(cache_modifier)
        return _rocdl.raw_ptr_buffer_store(
            value, rsrc, _byte_offset(offset, elem), _const(0, 32), *tail, **kw
        )


__all__ = [
    "HAS_BUFFER_OPS",
    "create_buffer_resource_from_addr",
    "buffer_load",
    "buffer_store",
    "CM_CACHED",
    "CM_SC0",
    "CM_NT",
    "CM_SC1",
    "CM_SC0_SC1",
]
