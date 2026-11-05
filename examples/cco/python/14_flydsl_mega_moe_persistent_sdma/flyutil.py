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
"""FlyDSL helpers used by the kernels: buffer resources, typed global / LDS pointers, the
cross-rank atomics and fences, the 8-wave GEMM's global->LDS->register loaders.

Copied from ROCm/aiter (``aiter/ops/flydsl/kernels``: communication_ops_utils, buffer_ops,
tensor_shim, mxfp4_gemm_common, kernels_common, gemm_a8w8_8wave, mega_moe/dispatch and
mega_moe/mega_moe_fused_gemm), only the definitions this example uses; each carries a
``# from`` comment naming its source module. Kept as they are so the example runs without
aiter installed.
"""

from flydsl._mlir import ir
from flydsl._mlir.dialects import fly
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.extras import types as T_buffer_ops
from flydsl.compiler.protocol import extract_to_ir_values
from flydsl.expr import arith
from flydsl.expr import const_expr
from flydsl.expr import math as fmath
from flydsl.expr import ptrtoint
from flydsl.expr import range_constexpr
from flydsl.expr import rocdl
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import is_rdna_arch
from flydsl.utils import env
from typing import NamedTuple
import flydsl.compiler as flyc
import flydsl.expr as fx
import inspect


# from aiter.ops.flydsl.kernels.communication_ops_utils
def _to_ptr_global(v, dtype=fx.Int32, alignment=4):
    """Cast an integer address to an aligned global FlyDSL pointer."""
    ptr_type = fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, alignment)
    return fx.inttoptr(ptr_type, fx.Int64(v))


# from aiter.ops.flydsl.kernels.communication_ops_utils
def atomic_add_global_at(addr_i64, val, syncscope="one-as"):
    """Monotonic global fetch-add with configurable agent/system visibility."""
    dtype = (
        fx.Int32
        if isinstance(val, int)
        else fx.Int64 if fx.as_ir_value(val).type == T.i64 else fx.Int32
    )
    ptr = _to_ptr_global(addr_i64, dtype, dtype.width // 8)
    return fx.atomic_add(ptr, dtype(val), syncscope=syncscope)


# from aiter.ops.flydsl.kernels.communication_ops_utils
def atomic_add_agent(addr_i64, val):
    """Agent-scope monotonic global fetch-and-add."""
    return atomic_add_global_at(addr_i64, val, syncscope=fx.rocdl.SyncScope.Agent)


# from aiter.ops.flydsl.kernels.communication_ops_utils
def atomic_add_system(addr_i64, val):
    """System-scope monotonic global fetch-and-add."""
    return atomic_add_global_at(addr_i64, val)


# from aiter.ops.flydsl.kernels.communication_ops_utils
def fence_acquire(syncscope):
    """Emit an acquire fence for the selected AMDGPU memory scope."""
    fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope=syncscope)


# from aiter.ops.flydsl.kernels.communication_ops_utils
def fence_agent_acquire():
    """Agent-scope acquire fence."""
    fence_acquire(fx.rocdl.SyncScope.AgentOneAs)


# from aiter.ops.flydsl.kernels.communication_ops_utils
def fence_release(syncscope):
    """Emit a release fence for the selected AMDGPU memory scope."""
    fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope=syncscope)


# from aiter.ops.flydsl.kernels.communication_ops_utils
def fence_agent_release():
    """Agent-scope release fence."""
    fence_release(fx.rocdl.SyncScope.AgentOneAs)


# from aiter.ops.flydsl.kernels.communication_ops_utils
def fence_system_acquire():
    """System-scope acquire fence."""
    fence_acquire(fx.rocdl.SyncScope.OneAs)


# from aiter.ops.flydsl.kernels.communication_ops_utils
def fence_system_release():
    """System-scope release fence."""
    fence_release(fx.rocdl.SyncScope.OneAs)


# from aiter.ops.flydsl.kernels.communication_ops_utils
def load_i32_global_agent(addr_i64):
    """Volatile monotonic i32 load with agent-one-as visibility."""
    return fx.generic_load(
        _to_ptr_global(addr_i64),
        volatile=True,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


# from aiter.ops.flydsl.kernels.communication_ops_utils
@flyc.jit
def spin_until_ge_i32_agent(addr_i64, val, *, sleep=True):
    """Spin on an agent-visible i32 flag until it reaches ``val``."""
    cur = fx.Int32(load_i32_global_agent(addr_i64))
    while cur < fx.Int32(val):
        if fx.const_expr(sleep):
            fx.rocdl.s_sleep(1)
        cur = fx.Int32(load_i32_global_agent(addr_i64))
    return cur


# from aiter.ops.flydsl.kernels.communication_ops_utils
def load_i32_global_system(addr_i64, *, acquire=False):
    """Volatile i32 load with system visibility for a spin-wait."""
    return fx.generic_load(
        _to_ptr_global(addr_i64),
        volatile=True,
        memory_order=(
            fx.AtomicOrdering.Acquire if acquire else fx.AtomicOrdering.Monotonic
        ),
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


# from aiter.ops.flydsl.kernels.communication_ops_utils
@flyc.jit
def spin_until_ge_i32_system(addr_i64, val, *, acquire=False, sleep=True):
    """Spin on a system-visible i32 flag until it reaches ``val``."""
    cur = fx.Int32(load_i32_global_system(addr_i64, acquire=acquire))
    while cur < fx.Int32(val):
        if fx.const_expr(sleep):
            fx.rocdl.s_sleep(1)
        cur = fx.Int32(load_i32_global_system(addr_i64, acquire=acquire))
    return cur


# from aiter.ops.flydsl.kernels.communication_ops_utils
def store_i32_global_system_monotonic(addr_i64, val):
    """System-visible monotonic store to a global i32 address."""
    fx.generic_store(
        _to_ptr_global(addr_i64),
        fx.Int32(val),
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


# from aiter.ops.flydsl.kernels.communication_ops_utils
def store_i32_global_system_release(addr_i64, val):
    """System-visible release store to a global i32 address."""
    fx.generic_store(
        _to_ptr_global(addr_i64),
        fx.Int32(val),
        memory_order=fx.AtomicOrdering.Release,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


# from aiter.ops.flydsl.kernels.communication_ops_utils
def store_i32_system(addr_i64, offset, val):
    """System-scope release i32 store at ``addr_i64 + offset*4``."""
    off = arith.unwrap(offset)
    off64 = fx.Uint64(fx.Uint32(off)) if off.type == T.i32 else fx.Int64(off)
    addr = fx.Int64(addr_i64) + fx.Int64(off64) * 4
    fx.generic_store(
        _to_ptr_global(addr),
        fx.Int32(val),
        memory_order=fx.AtomicOrdering.Release,
        syncscope="one-as",
    )


# from aiter.ops.flydsl.kernels.communication_ops_utils
def load_i32_acquire(addr_i64):
    """Volatile monotonic i32 load suitable for a spin-wait."""
    return fx.generic_load(
        _to_ptr_global(addr_i64),
        volatile=True,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope="one-as",
    )


# from aiter.ops.flydsl.kernels.communication_ops_utils
@flyc.jit
def spin_until_eq_i32(addr_i64, val):
    """Spin until an i32 flag equals ``val``."""
    cur = fx.Int32(load_i32_acquire(addr_i64))
    while cur != fx.Int32(val):
        cur = fx.Int32(load_i32_acquire(addr_i64))
    return cur


# from aiter.ops.flydsl.kernels.communication_ops_utils
def wait_i32_until_equals(addr_i64, expected):
    """Compatibility wrapper for the historical MegaMoE wait helper."""
    spin_until_eq_i32(addr_i64, expected)


# from aiter.ops.flydsl.kernels.communication_ops_utils
@flyc.jit
def spin_until_gt_i32(addr_i64, val):
    """Spin until an i32 flag exceeds ``val`` and return the observed value."""
    cur = fx.Int32(load_i32_acquire(addr_i64))
    while cur <= fx.Int32(val):
        cur = fx.Int32(load_i32_acquire(addr_i64))
    return cur


# from aiter.ops.flydsl.kernels.communication_ops_utils
def wait_i32_until_greater_than(addr_i64, expected):
    """Compatibility wrapper returning the first i32 value above ``expected``."""
    return spin_until_gt_i32(addr_i64, expected)


# from aiter.ops.flydsl.kernels.communication_ops_utils
def load_i32_system(addr_i64, index):
    """Compatibility system-scope i32 load at ``addr_i64 + index * 4``."""
    item_addr = fx.Int64(addr_i64) + fx.Int64(index) * 4
    return fx.Int32(load_i32_acquire(item_addr))


# from aiter.ops.flydsl.kernels.mxfp4_gemm_common
def global_typed_ptr(arg, elem_ty, align=4, *, byte_offset=None):
    """Typed global fx.Pointer over a raw i64 device address; index in ELEMENTS (ptr[i]), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    if byte_offset is not None:
        byte_ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Global, align)
        base = fx.inttoptr(byte_ptr_ty, fx.Int64(arg))
        return fx.recast_iter(ptr_ty, fx.add_offset(base, byte_offset))
    return fx.inttoptr(ptr_ty, fx.Int64(arg))


# from aiter.ops.flydsl.kernels.mxfp4_gemm_common
def lds_typed_ptr(base_i32, elem_ty, align=4, *, byte_offset=None):
    """Typed LDS (Shared) fx.Pointer over an i32 LDS base; index in ELEMENTS (ptr[i]), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    if byte_offset is not None:
        byte_ptr_ty = fx.PointerType.get(T.i8, fx.AddressSpace.Shared, align)
        base = fx.inttoptr(byte_ptr_ty, fx.Int32(base_i32))
        return fx.recast_iter(ptr_ty, fx.add_offset(base, byte_offset))
    return fx.inttoptr(ptr_ty, fx.Int32(base_i32))


# from aiter.ops.flydsl.kernels.mxfp4_gemm_common
def lds_vec_load(base_i32, byte_off_i32, result_type, elem_ty, align=4):
    """Typed LDS ds-read at a BYTE offset from the i32 LDS base; mirrors raw llvm.load (vector or scalar)."""
    elem_ir_ty = elem_ty.ir_type if hasattr(elem_ty, "ir_type") else elem_ty
    ptr = lds_typed_ptr(base_i32, elem_ir_ty, align=align, byte_offset=byte_off_i32)
    return fx.ptr_load(ptr, result_type=result_type)


# from aiter.ops.flydsl.kernels.mxfp4_gemm_common
def _fabs_f32(x):
    return fmath.absf(x)


# from aiter.ops.flydsl.kernels.tensor_shim
BUF_VIEW_MAX_ELEMS = 0xFFFFFFFF


# from aiter.ops.flydsl.kernels.tensor_shim
def buf_base_i64(base):
    """i64 base address of *base*: an fx pointer, a tensor/memref, or an address.

    Lets a caller narrow a descriptor to one row (``ptr + row * row_bytes``)
    without first materialising a pointer for it.
    """
    raw = extract_to_ir_values(base)[0]
    if str(raw.type).startswith(("!fly.ptr", "!llvm.ptr")):
        return fx.Int64(ptrtoint(base))
    if isinstance(raw.type, (ir.IntegerType, ir.IndexType)):
        return fx.Int64(base)
    # tensor / memref kernel arg: take its aligned base pointer.
    aligned = fly.extract_aligned_pointer_as_index(ir.Type.parse("!llvm.ptr<1>"), raw)
    return fx.Int64(llvm.PtrToIntOp(T.i64, aligned).result)


# from aiter.ops.flydsl.kernels.tensor_shim
def ptr_buf_tensor(
    ptr,
    elem=fx.Int32,
    n=BUF_VIEW_MAX_ELEMS,
    unit_elems=1,
    num_records_bytes=None,
    unit_stride=None,
):
    """Buffer-resource (V#) view of *ptr*, so ``t[i]`` / ``fx.slice`` index it.

    *ptr* is an fx.Pointer kernel arg or a raw i64 address (see
    :func:`buf_base_i64`).

    Keeps the addressing `buffer_ops` used: descriptor in SGPRs, 32-bit voffset
    per access. Indexing a plain typed pointer instead builds a full 64-bit
    address in VGPRs on every access.

    ``unit_elems`` sets the access width and hence the rank:
      1  -> flat ``(n,)``; ``t[i]`` is one element. No atom, no fragment.
      >1 -> ``(n, unit_elems)``; ``fx.slice(t, (u, None))`` is one wide access
            and ``fx.copy`` over it emits a single ``buffer_load_dwordx{2,4}``.
            A vector *element type* would be the obvious spelling for this but
            MLIR rejects it (``AlignAttr`` takes integer/float only), so the
            width lives in the layout.

    ``unit_stride`` is the element distance between consecutive units and
    defaults to ``unit_elems`` (units tile the buffer, so ``u`` counts whole
    units). Pass 1 for a wide access at an arbitrary *element* offset: units
    then overlap -- meaningless to iterate but exact for the one slice a caller
    takes, and the only way to express e.g. a dwordx4 store at a row base that
    is only dword-aligned. It also sets the pointer alignment, which is all such
    an access can rely on.

    ``num_records_bytes`` may be a runtime value, for a hardware OOB check that
    zero-fills rather than reading stale bytes.
    """
    unit_stride = unit_elems if unit_stride is None else unit_stride
    layout = (
        fx.make_layout((n,), (1,))
        if unit_elems == 1
        else fx.make_layout((n, unit_elems), (unit_stride, 1))
    )
    pt = fx.PointerType.get(
        elem.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=unit_stride * (elem.width // 8),
    )
    view = fx.make_view(fx.inttoptr(pt, buf_base_i64(ptr)), layout)
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)


# from aiter.ops.flydsl.kernels.tensor_shim
_BUF_COPY_ATOM = {
    16: fx.rocdl.BufferCopy128b,
    8: fx.rocdl.BufferCopy64b,
    4: fx.rocdl.BufferCopy32b,
    2: fx.rocdl.BufferCopy16b,
    1: fx.rocdl.BufferCopy8b,
}


# from aiter.ops.flydsl.kernels.tensor_shim
def buf_copy_atom(unit_bytes, elem=fx.Int32, cache_modifier=0):
    """Copy atom for a ``unit_bytes``-wide buffer access (0=cached, 2=nt).

    ``cache_modifier`` is the instruction's raw aux/CPOL field, and its width is
    architecture-dependent: two bits (sc0, nt) on CDNA3, but gfx1250 puts scope
    bits above them and passes values like ``CPOL_DEV`` (0x10). It is therefore
    NOT range-checked here -- a bound that fits one generation rejects a legal
    value on another, which is what happened when this was briefly capped at 3
    and five gfx1250 mxfp8 GEMM kernels stopped building. Callers that know
    their own encoding bound it themselves; the M3 scorers do.
    """
    if unit_bytes not in _BUF_COPY_ATOM:
        raise ValueError(
            f"unit_bytes {unit_bytes} is not a buffer access width; "
            f"expected one of {sorted(_BUF_COPY_ATOM)}"
        )
    return fx.make_copy_atom(_BUF_COPY_ATOM[unit_bytes](cache_modifier), elem)


# from aiter.ops.flydsl.kernels.tensor_shim
def _buf_copy_slice(buffer, index, unit_elems):
    if unit_elems == 1:
        grouped = fx.logical_divide(buffer, fx.make_layout(1, 1))
        return fx.slice(grouped, (None, index))
    return fx.slice(buffer, (index, None))


# from aiter.ops.flydsl.kernels.tensor_shim
def buf_copy_load(buffer, index, elem=fx.Int32, unit_elems=1, cache_modifier=0):
    """Load one vector unit, preserving an explicit buffer cache policy."""
    fragment = fx.make_rmem_tensor(unit_elems, elem)
    fx.copy(
        buf_copy_atom(
            unit_elems * (elem.width // 8), elem, cache_modifier=cache_modifier
        ),
        _buf_copy_slice(buffer, index, unit_elems),
        fragment,
    )
    value = Vec(fragment.load())
    return value[0] if unit_elems == 1 else value


# from aiter.ops.flydsl.kernels.tensor_shim
def buf_copy_store(buffer, index, value, elem=fx.Int32, unit_elems=1, cache_modifier=0):
    """Store one vector unit, preserving an explicit buffer cache policy."""
    fragment = fx.make_rmem_tensor(unit_elems, elem)
    fragment.store(Vec.from_elements([value], elem) if unit_elems == 1 else Vec(value))
    fx.copy(
        buf_copy_atom(
            unit_elems * (elem.width // 8), elem, cache_modifier=cache_modifier
        ),
        fragment,
        _buf_copy_slice(buffer, index, unit_elems),
    )


# from aiter.ops.flydsl.kernels.tensor_shim
def _compile_and_run(exe, *args):
    """``flyc.compile(exe, *args)``: compiles **and** executes the kernel, and
    returns its ``CompiledFunction`` (None under compile-only)."""
    try:
        return flyc.compile(exe, *args)
    except Exception:
        # flyc.compile leaks ir.Context on failure; pop it so a retry takes the right path.
        try:
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:  # noqa: BLE001, S110
            pass
        raise


# from aiter.ops.flydsl.kernels.tensor_shim
def _run_compiled(exe, *args, specialization_key=None):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel.
    Subsequent calls: fast dispatch via the cached ``CompiledFunction``.

    A specialization key gives a multi-constexpr JitFunction one compiled
    callable per configuration. Factory-style launchers should leave it unset.
    """
    if env.compile.compile_only:
        # A callable warmed during execution must not dispatch when a later
        # preload switches to compile-only. Preserve it for execution to resume.
        _compile_and_run(exe, *args)
        return
    if specialization_key is None:
        cf = getattr(exe, "_cf", None)
    else:
        cache = getattr(exe, "_cf_by_specialization", None)
        cf = cache.get(specialization_key) if cache is not None else None
    if cf is not None:
        cf(*args)
        return
    cf = _compile_and_run(exe, *args)
    if specialization_key is None:
        exe._cf = cf
    else:
        if not hasattr(exe, "_cf_by_specialization"):
            exe._cf_by_specialization = {}
        exe._cf_by_specialization[specialization_key] = cf


# from aiter.ops.flydsl.kernels.kernels_common
_slp_once_used_available = None


# from aiter.ops.flydsl.kernels.kernels_common
_SLP_VECTORIZE_ONCE_USED = "slp-vectorize-once-used"


# from aiter.ops.flydsl.kernels.kernels_common
def set_slp_vectorize_once_used(jit_fn, enabled):
    """Set LLVM slp-vectorize-once-used on this JIT.

    ``enabled`` is the value the kernel wants. flydsl 0.4.1 defaults the flag
    on. flydsl 0.4.0 has no such option, and passing it would fail compilation,
    so the hint is attached only when the cl::opt exists.
    """
    global _slp_once_used_available
    if _slp_once_used_available is None:
        try:
            from flydsl._mlir._mlir_libs import _mlirDialectsFly as fly

            previous = fly.set_llvm_option_bool(_SLP_VECTORIZE_ONCE_USED, False)
            fly.set_llvm_option_bool(_SLP_VECTORIZE_ONCE_USED, previous)
            _slp_once_used_available = True
        except RuntimeError:
            _slp_once_used_available = False
    if _slp_once_used_available:
        options = jit_fn.compile_hints.setdefault("llvm_options", {})
        options[_SLP_VECTORIZE_ONCE_USED] = bool(enabled)
    return jit_fn


# from aiter.ops.flydsl.kernels.kernels_common
def ceildiv(numer, denom):
    """Ceiling division preserving Python-int or DSL-scalar operand types."""
    return (numer + denom - 1) // denom


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_fused_gemm
class AddrBuffer(NamedTuple):
    """A raw device address and an optional hardware OOB bound in bytes."""

    addr: object
    num_records_bytes: object = None


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_fused_gemm
def _addr_view(buf, elem, width):
    # unit_stride=1: a wide access may start at any element offset.
    return ptr_buf_tensor(
        buf.addr,
        elem,
        unit_elems=width,
        unit_stride=1,
        num_records_bytes=buf.num_records_bytes,
    )


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_fused_gemm
def load_at(buf, index, elem=fx.Int32, width=1, cache_modifier=0, mask=None):
    """Load ``width`` x ``elem`` at element ``index`` of ``buf``.

    A false ``mask`` moves the access past the buffer bound.
    """
    if mask is not None:
        index = fx.Boolean(mask).select(
            index, fx.Int32(0x7FFFFFFF // (elem.width // 8))
        )
    return buf_copy_load(
        _addr_view(buf, elem, width), index, elem, width, cache_modifier
    )


# from aiter.ops.flydsl.kernels.tensor_shim
_FX_ELEM = {
    "i8": fx.Int8,
    "i16": fx.Int16,
    "i32": fx.Int32,
    "i64": fx.Int64,
    "f16": fx.Float16,
    "bf16": fx.BFloat16,
    "f32": fx.Float32,
}


# from aiter.ops.flydsl.kernels.tensor_shim
def _fx_elem(dtype):
    """fx element class for an MLIR scalar type."""
    key = str(dtype() if callable(dtype) else dtype)
    try:
        return _FX_ELEM[key]
    except KeyError:
        raise TypeError(f"no fx element class for MLIR type {key!r}") from None


# from aiter.ops.flydsl.kernels.tensor_shim
def _fx_value_elem(value):
    """(fx element class, element count) of a scalar or vector SSA value."""
    ty = extract_to_ir_values(value)[0].type
    # Duck-typed vector check, as buffer_store did: this MLIR binding has no
    # ir.VectorType.isinstance.
    if hasattr(ty, "element_type"):
        return _fx_elem(ty.element_type), ty.shape[0]
    return _fx_elem(ty), 1


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_fused_gemm
def store_at(value, buf, index, cache_modifier=0, offset_is_bytes=False):
    """Store ``value`` (scalar or vector) at element ``index`` of ``buf``."""
    elem, width = _fx_value_elem(value)
    if offset_is_bytes:
        index = fx.Int32(index) // fx.Int32(elem.width // 8)
    buf_copy_store(
        _addr_view(buf, elem, width), index, value, elem, width, cache_modifier
    )


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_fused_gemm
def _buffer_load(buffer, group_index, elem_ty, width=1, cache_modifier=0):
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(elem_ty.width * width, cache_modifier), elem_ty
    )
    fragment = fx.make_rmem_tensor(width, elem_ty)
    fx.copy(atom, fx.slice(buffer, (None, group_index)), fragment)
    value = Vec(fragment.load())
    return value[0] if width == 1 else value


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_fused_gemm
def _buffer_store(buffer, group_index, value, elem_ty, width=1, cache_modifier=0):
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(elem_ty.width * width, cache_modifier), elem_ty
    )
    fragment = fx.make_rmem_tensor(width, elem_ty)
    fragment.store(Vec.from_elements([value], elem_ty) if width == 1 else Vec(value))
    fx.copy(atom, fragment, fx.slice(buffer, (None, group_index)))


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_fused_gemm
def _make_buffer_from_addr(addr, elem_ty, width=1, *, num_records_bytes=None):
    alignment = max(1, elem_ty.width * width // 8)
    ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Global, alignment)
    base = fx.inttoptr(ptr_ty, fx.Int64(addr))
    view = fx.Tensor(fx.make_view(base, fx.make_layout((width, 1), (1, width))))
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)


# from aiter.ops.flydsl.kernels.mega_moe.mega_moe_fused_gemm
def _make_buffer(tensor, elem_ty, width=1, *, max_size=True, num_records_bytes=None):
    alignment = max(1, elem_ty.width * width // 8)
    ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Global, alignment)
    base = fx.inttoptr(ptr_ty, fx.Int64(fx.ptrtoint(fx.get_iter(tensor))))
    view = fx.Tensor(fx.make_view(base, fx.make_layout((width, 1), (1, width))))
    return fx.rocdl.make_buffer_tensor(
        view, max_size=max_size, num_records_bytes=num_records_bytes
    )


# from aiter.ops.flydsl.kernels.mega_moe.dispatch
@flyc.jit
def _wave_inclusive_scan_i32(value, lane):
    value_raw = value.ir_value()
    zero_raw = fx.Int32(0).ir_value()
    for shift, dpp in ((1, 0x111), (2, 0x112), (4, 0x114), (8, 0x118)):
        remote = fx.rocdl.update_dpp(T.i32, zero_raw, value_raw, dpp, 0xF, 0xF, True)
        value = (lane >= fx.Int32(shift)).select(value + fx.Int32(remote), value)
        value_raw = value.ir_value()
    source16 = (lane & fx.Int32(0x30)) - fx.Int32(1)
    remote16 = fx.rocdl.ds_bpermute(T.i32, source16 * fx.Int32(4), value)
    value = (lane >= fx.Int32(16)).select(value + fx.Int32(remote16), value)
    source32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
    remote32 = fx.rocdl.ds_bpermute(T.i32, source32 * fx.Int32(4), value)
    return (lane >= fx.Int32(32)).select(value + fx.Int32(remote32), value)


# from aiter.ops.flydsl.kernels.mega_moe.dispatch
@flyc.jit
def _wave_reduce_max_i32(value, lane):
    for distance in (1, 2, 4, 8, 16, 32):
        peer = fx.Int32(
            fx.rocdl.ds_bpermute(
                T.i32, (lane ^ fx.Int32(distance)) * fx.Int32(4), value
            )
        )
        value = (peer > value).select(peer, value)
    return value


# from aiter.ops.flydsl.kernels.gemm_a8w8_8wave
class G2SLoader:
    def __init__(self, gl_src, gl_offsets, n_load_steps, lds_dtype, wave_id):
        self.g2lds_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        self.LdsPtr_t = fx.PointerType.get(lds_dtype, 2, 512)
        self.gl_src = gl_src
        self.gl_offsets = gl_offsets
        self.n_load_steps = n_load_steps
        self.wave_id = wave_id
        self.n_waves = fx.block_dim.x // 64

    def _lds_dst_at(self, lds_dst, step):
        step_off = self.wave_id * 1024 + step * (self.n_waves * 1024)
        base_i32 = fx.Int32(fx.ptrtoint(lds_dst.ptr))
        sum_i32 = base_i32 + fx.Int32(step_off)
        lds_ptr = fx.inttoptr(self.LdsPtr_t, sum_i32)
        return fx.make_view(lds_ptr, fx.make_layout(1, 1))

    def load(self, lds_dst, k_offset):
        for step in range_constexpr(self.n_load_steps):
            src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
            dst = self._lds_dst_at(lds_dst, step)
            fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))

    def load_one(self, lds_dst, k_offset, step):
        src = fx.slice(self.gl_src, (None, fx.Int32(self.gl_offsets[step])))
        dst = self._lds_dst_at(lds_dst, step)
        fx.copy(self.g2lds_atom, src, dst, soffset=fx.Int32(k_offset))


# from aiter.ops.flydsl.kernels.gemm_a8w8_8wave
def pack_i32x4_i32x8(lo, hi):
    # Pack two i32x4 as one i32x8
    return lo.shuffle(hi, list(range(8)))


# from aiter.ops.flydsl.kernels.gemm_a8w8_8wave
def swizzle_128(row, col):
    offset = row * 128 + col
    swizzle = ((offset % (16 * 128)) >> 8) << 4
    swizzled_offset = offset ^ swizzle
    return swizzled_offset // 128, swizzled_offset % 128


# from aiter.ops.flydsl.kernels.gemm_a8w8_8wave
class S2RLoader:
    def __init__(self, wave_idx, n_tiles):
        self.lane_id = fx.thread_idx.x % 64
        self.wave_idx = wave_idx
        self.n_tiles = n_tiles

    def _vec_load_16xf8(self, lds_src, offset):
        off_tup = fx.make_int_tuple(offset)
        ptr_off = fx.add_offset(lds_src.ptr, off_tup)
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        view = fx.make_view(i8_iter, fx.make_layout(16, 1))
        return view.load()

    def load(self, lds_src, preshuffled=False):
        frag = []
        for i in range_constexpr(self.n_tiles):
            halves = []
            row = self.wave_idx * (self.n_tiles * 16) + i * 16 + self.lane_id % 16
            for step in range_constexpr(2):
                col = (self.lane_id // 16) * 16 + step * 64
                if const_expr(preshuffled):
                    offset = (row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128
                else:
                    row_swz, col_swz = swizzle_128(row, col)
                    offset = row_swz * 128 + col_swz
                v = self._vec_load_16xf8(lds_src, offset)
                halves.append(v.bitcast(fx.Int32))
            frag.append(pack_i32x4_i32x8(halves[0], halves[1]))
        return frag

    def load_one(self, lds_src, lds_offset):
        v = self._vec_load_16xf8(lds_src, lds_offset)
        return v.bitcast(fx.Int32)


# from aiter.ops.flydsl.kernels.gemm_a8w8_8wave
def compute_global_swizzle(lane_id, wave_id, K, n_rounds, preshuffled):
    offsets = []
    n_waves = fx.block_dim.x // 64
    for round in range_constexpr(n_rounds):
        if const_expr(preshuffled):
            row = lane_id % 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id // 8) * 16
            offsets.append(
                (row // 16) * (K * 16)
                + (row % 16) * 16
                + (col // 64) * 1024
                + ((col % 64) // 16) * 256
                + (col % 16)
            )
        else:
            row = lane_id // 8 + wave_id * 8 + round * (n_waves * 8)
            col = (lane_id % 8) * 16
            r, c = swizzle_128(row, col)
            offsets.append(r * K + c)
    return offsets


# from aiter.ops.flydsl.kernels.gemm_a8w8_8wave
def make_fp8_buffer_tensor(arg_i8, fp8_ir_t):
    # max_size=False with no num_records_bytes: cosize(layout) becomes a
    # runtime expression because TensorAdaptor defaults to layout-dynamic
    # memref (post #554), so the descriptor adapts to the actual tensor
    # extent and no longer bakes the first-call's shape into IR.
    t_i8 = fx.rocdl.make_buffer_tensor(arg_i8, max_size=False)
    iter_i8 = fx.get_iter(t_i8)
    f8_buf_ptr_ty = fx.PointerType.get(
        elem_ty=fp8_ir_t,
        address_space=rocdl.TargetAddressSpace.BufferDesc,
        alignment=fx.PointerType(iter_i8.type).alignment,
    )
    iter_f8 = fx.recast_iter(f8_buf_ptr_ty, iter_i8)
    return fx.Tensor(fx.make_view(iter_f8, fx.get_layout(t_i8)))


# from aiter.ops.flydsl.kernels.gemm_a8w8_8wave
def wait_barrier(count):
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt vmcnt({count})\ns_barrier",
        constraints="",
        has_side_effects=True,
    )


# from aiter.ops.flydsl.kernels.buffer_ops
def _unwrap_value(value):
    """Recursively unwrap ArithValue or similar wrappers to get the actual MLIR value.

    Handles:
    - FlyDSL ArithValue (has ._value)
    - flyc DSL Numeric like fx.Int32 (has .ir_value() method)
    - flyc ArithValue (is already ir.Value subclass)
    """
    # DSL Numeric (Int32, Float32, etc.) — use ir_value() to materialize
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    max_depth = 10  # Safety limit
    depth = 0
    while depth < max_depth and not isinstance(value, ir.Value):
        if hasattr(value, "_value"):
            value = value._value
        elif hasattr(value, "value"):
            value = value.value
        else:
            break
        depth += 1
    return value


# from aiter.ops.flydsl.kernels.buffer_ops
def _get_buffer_flags(arch=None):
    """Get AMD buffer resource descriptor (V#) flags word (bits 127:96).

    Constructs the 32-bit flags field for rocdl.make.buffer.rsrc, following the
    same logic as LLVM's AMDGPUToROCDL makeBufferRsrc():
      https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp

    Bit layout (common to all architectures):
      bits [11:0]  - DST_SEL: ignored by raw buffer intrinsics
      bits [14:12] - DATA_FORMAT: must be nonzero, 7 = float
      bits [18:15] - NUM_FORMAT:  must be nonzero, 4 = 32-bit
      bit  [19]    - In nested heap (0)
      bit  [20]    - Behavior on unmap (0 = return 0 / ignore)
      bits [22:21] - Index stride for swizzles (0)
      bit  [23]    - Add thread ID (0)
      bit  [24]    - Reserved: must be 1 on RDNA, 0 on CDNA
      bits [26:25] - Reserved (0)
      bit  [27]    - Non-volatile (CDNA only, 0)
      bits [29:28] - OOB_SELECT (RDNA only): 0=structured, 2=none, 3=check offset
      bits [31:30] - Type (must be 0)

    CDNA (gfx9xx):    (7 << 12) | (4 << 15)                         = 0x20070
    RDNA (gfx10+):    (7 << 12) | (4 << 15) | (1 << 24) | (2 << 28) = 0x21020070
      - bit 24 set to 1 (required on RDNA)
      - OOB_SELECT=2 (no bounds checking, matching LLVM boundsCheck=false)
    """
    import os

    if arch is None:
        arch = os.environ.get("FLYDSL_GPU_ARCH")
    flags = (7 << 12) | (4 << 15)
    if is_rdna_arch(arch):
        flags |= 1 << 24  # reserved bit, must be 1 on RDNA
        flags |= 2 << 28  # OOB_SELECT = 2 (no bounds checking)
    return flags


# from aiter.ops.flydsl.kernels.buffer_ops
@dsl_loc_tracing
def create_buffer_resource_from_addr(
    addr_i64: ir.Value, *, num_records_bytes: int | ir.Value | None = None
) -> ir.Value:
    """Create AMD buffer resource descriptor from a raw i64 device address.

    Useful when working with runtime pointer arrays (e.g. IPC-mapped addresses
    or device-side pointer tables) where no fly.memref is available.
    The full address is encoded as the buffer base; callers should pass
    byte offset 0 to buffer_load / buffer_store.

    Args:
        addr_i64: Raw 64-bit device address (i64 MLIR value).
        num_records_bytes: Optional buffer size in bytes for hardware OOB checking.

    Returns:
        ROCDL buffer resource descriptor (!llvm.ptr<8>).

    Example:
        >>> rsrc = create_buffer_resource_from_addr(raw_addr_i64)
        >>> data = buffer_load(rsrc, i32_zero, vec_width=4, dtype=T.i32)
    """
    addr_i64 = _unwrap_value(addr_i64)
    ptr_type = fx.PointerType.get(T_buffer_ops.i8(), fx.AddressSpace.Generic, 1)
    base_ptr = fx.to_llvm_ptr(fx.inttoptr(ptr_type, addr_i64))
    flags = fx.Int32(_get_buffer_flags()).ir_value()
    stride = fx.Int16(0).ir_value()
    if num_records_bytes is None:
        num_records = fx.Int64(4294967295).ir_value()
    elif isinstance(num_records_bytes, int):
        nbytes = max(0, min(int(num_records_bytes), 4294967295))
        num_records = fx.Int64(nbytes).ir_value()
    else:
        num_records = fx.Int64(_unwrap_value(num_records_bytes)).ir_value()
    rsrc_type = ir.Type.parse("!llvm.ptr<8>")
    return fx.rocdl.make_buffer_rsrc(rsrc_type, base_ptr, stride, num_records, flags)


# from aiter.ops.flydsl.kernels.buffer_ops
_RAW_PTR_BUFFER_AUX_IS_ATTRIBUTE = (
    inspect.signature(fx.rocdl.RawPtrBufferLoadOp).parameters["aux"].kind
    is inspect.Parameter.KEYWORD_ONLY
)


# from aiter.ops.flydsl.kernels.buffer_ops
@dsl_loc_tracing
def _ptr8_to_v4i32(ptr8_val) -> ir.Value:
    """Reinterpret a buffer resource (!llvm.ptr<8>) as a <4 x i32> vector.

    Required by the scalar ``s.buffer.load`` intrinsic, whose resource operand is
    a v4i32 rather than the opaque buffer pointer used by the vector path.
    """
    i128_ty = ir.IntegerType.get_signless(128)
    v4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
    i128_val = llvm.ptrtoint(i128_ty, _unwrap_value(ptr8_val))
    return llvm.bitcast(v4i32_ty, i128_val)


# from aiter.ops.flydsl.kernels.buffer_ops
@dsl_loc_tracing
def buffer_load(
    rsrc: ir.Value,
    offset: ir.Value,
    vec_width: int = 4,
    dtype=None,
    mask: ir.Value | None = None,
    cache_modifier: int = 0,
    soffset_bytes: int | ir.Value | None = None,
    is_scalar: bool = False,
) -> ir.Value:
    """AMD buffer load operation.

    Load data from global memory using buffer descriptor and offset.
    Uses hardware-level bounds checking and vectorization.

    Args:
        rsrc: Buffer resource descriptor (!llvm.ptr<8>)
        offset: Offset in elements (i32 type)
        vec_width: Vector width (1, 2, or 4)
        dtype: Element data type (None for f32, or ir.F32Type, etc.)
        mask: Optional mask for predicated load (i1 type)
        cache_modifier: Cache control flags (0 for default)
        soffset_bytes: Optional scalar offset (in BYTES) added by the buffer instruction (soffset).
                      Use this to fold small constant deltas into the instruction instead of emitting
                      extra VGPR address arithmetic.
        is_scalar: Emit a uniform/SGPR scalar load (llvm.amdgcn.s.buffer.load) instead of the
                      vector buffer load. Use only for wave-uniform addresses to route through the
                      SMEM cache and land the result directly in SGPRs. Restricted to vec_width 1, 4, 8 or 16;
                      dtype is forced to i32 (the result is raw i32 dwords). mask and soffset_bytes
                      are not supported in this mode and raise ValueError if provided.

    Returns:
        Loaded data (scalar or vector depending on vec_width)

    Example:
        >>> # Load 4xf32
        >>> data = buffer_load(rsrc, offset, vec_width=4)
        >>>
        >>> # Load with mask
        >>> data = buffer_load(rsrc, offset, vec_width=4, mask=valid)
    """
    if is_scalar:
        if vec_width not in (1, 4, 8, 16):
            raise ValueError(
                f"buffer_load(is_scalar=True): unsupported vec_width={vec_width}"
            )
        if mask is not None or soffset_bytes is not None:
            raise ValueError(
                "buffer_load(is_scalar=True) does not support mask or soffset_bytes"
            )
        dtype = T_buffer_ops.i32()
    elif dtype is None:
        dtype = T_buffer_ops.f32()
    elif hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type
    offset = fx.Int32(_unwrap_value(offset))
    element_bytes = dtype.width // 8
    offset = offset * element_bytes
    if mask is not None:
        offset = fx.Boolean(_unwrap_value(mask)).select(offset, 2147483647)
    if vec_width == 1:
        result_type = dtype
    else:
        result_type = ir.VectorType.get([vec_width], dtype)
    if is_scalar:
        rsrc_v4 = _ptr8_to_v4i32(rsrc)
        cache_policy = fx.Int32(cache_modifier).ir_value()
        suffix = "i32" if vec_width == 1 else f"v{vec_width}i32"
        return llvm.call_intrinsic(
            result_type,
            f"llvm.amdgcn.s.buffer.load.{suffix}",
            [rsrc_v4, offset.ir_value(), cache_policy],
            [],
            [],
        )
    soffset = fx.Int32(
        0 if soffset_bytes is None else _unwrap_value(soffset_bytes)
    ).ir_value()
    aux = (
        ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_modifier)
        if _RAW_PTR_BUFFER_AUX_IS_ATTRIBUTE
        else fx.Int32(cache_modifier).ir_value()
    )
    return fx.rocdl.raw_ptr_buffer_load(
        result_type, rsrc, offset.ir_value(), soffset, aux=aux
    )


# from aiter.ops.flydsl.kernels.buffer_ops
@dsl_loc_tracing
def buffer_store(
    data: ir.Value,
    rsrc: ir.Value,
    offset: ir.Value,
    mask: ir.Value | None = None,
    cache_modifier: int = 0,
    *,
    soffset_bytes: int | ir.Value | None = None,
    offset_is_bytes: bool = False,
):
    """AMD buffer store operation.

    Store data to global memory using buffer descriptor and offset.

    Args:
        data: Data to store (scalar or vector)
        rsrc: Buffer resource descriptor (!llvm.ptr<8>)
        offset: Offset in elements (i32 type)
        mask: Optional mask for predicated store (i1 type)
        cache_modifier: Cache control flags (0 for default)

    Example:
        >>> buffer_store(data, rsrc, offset)
        >>>
        >>> # Store with mask
        >>> buffer_store(data, rsrc, offset, mask=valid)
    """
    data = _unwrap_value(data)
    rsrc = _unwrap_value(rsrc)
    offset = fx.Int32(_unwrap_value(offset))

    # IMPORTANT: RawPtrBufferStoreOp offset is in BYTES.
    # For backward compat, `buffer_store()` accepts element offsets by default
    # and scales them to bytes. Set `offset_is_bytes=True` to skip scaling.
    if not offset_is_bytes:
        # Get element size from data type
        data_type = data.type
        if hasattr(data_type, "element_type"):  # Vector type
            element_type = data_type.element_type
        else:  # Scalar type
            element_type = data_type
        element_bytes = element_type.width // 8
        offset = offset * element_bytes

    # Apply mask by setting invalid offsets to max
    if mask is not None:
        offset = fx.Boolean(_unwrap_value(mask)).select(offset, 0x7FFFFFFF)

    # Create instruction offset (soffset) and aux flags
    soffset = fx.Int32(
        0 if soffset_bytes is None else _unwrap_value(soffset_bytes)
    ).ir_value()
    aux = (
        ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_modifier)
        if _RAW_PTR_BUFFER_AUX_IS_ATTRIBUTE
        else fx.Int32(cache_modifier).ir_value()
    )

    # Emit buffer store
    fx.rocdl.raw_ptr_buffer_store(
        data,
        rsrc,
        offset.ir_value(),
        soffset,
        aux=aux,
    )
