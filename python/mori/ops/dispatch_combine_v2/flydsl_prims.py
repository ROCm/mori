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
"""Shared FlyDSL device primitives for the cco-LSA dispatch/combine kernels.

Thin wrappers over flydsl._mlir.dialects.llvm for system-scope atomics / ordered
stores / fences, plus uncached buffer load/store and a spin-wait helper — the
building blocks both the dispatch and combine kernels need on top of cco's
peer pointers (cco.Window(h).lsa_ptr(pe, off)).
"""
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl._mlir.dialects import rocdl as _rocdl_d
from flydsl._mlir.dialects import scf
from flydsl.expr import arith
from flydsl.expr.typing import T
import flydsl.expr as fx


def _I64():
    return ir.IntegerType.get_signless(64)


def _I32():
    return ir.IntegerType.get_signless(32)


def _V3I32():
    return ir.VectorType.get([3], _I32())


def _NUW():
    return ir.Attribute.parse("#llvm.overflow<none>")


def _gptr(addr_i64):
    return _llvm_d.IntToPtrOp(
        _llvm_d.PointerType.get(address_space=1), arith.unwrap(addr_i64)
    ).result


def _addr_plus(base_i64, offset, elem_bytes):
    """base + offset*elem_bytes as an i64 SSA value (offset may be i32 or i64)."""
    base = arith.unwrap(base_i64)
    off = arith.unwrap(offset)
    off64 = _llvm_d.ZExtOp(_I64(), off).res if off.type == _I32() else off
    eb = _llvm_d.ConstantOp(_I64(), ir.IntegerAttr.get(_I64(), elem_bytes)).result
    byte_off = _llvm_d.MulOp(off64, eb, _NUW()).result
    return _llvm_d.AddOp(base, byte_off, _NUW()).result


def lsa_ptr_global(window_i64, peer_i32, offset_i64):
    """Compute a CCO peer VA while loading window metadata from addrspace(1).

    ccoWindowDevice starts with {i64 winBase, i32 stride4G}. Loading it as
    vector<3xi32> keeps the 12-byte access vectorized and lets AMDGPU select
    global_load_dwordx3 instead of the generic-address flat_load_dwordx3 emitted
    by the linked cco_lsa_ptr extern.
    """
    meta = _llvm_d.LoadOp(
        _V3I32(), _gptr(window_i64), alignment=8
    ).res
    indices = [
        _llvm_d.ConstantOp(_I64(), ir.IntegerAttr.get(_I64(), i)).result
        for i in range(3)
    ]
    win_lo = _llvm_d.ExtractElementOp(meta, indices[0]).res
    win_hi = _llvm_d.ExtractElementOp(meta, indices[1]).res
    stride = _llvm_d.ExtractElementOp(meta, indices[2]).res
    win_lo64 = _llvm_d.ZExtOp(_I64(), win_lo).res
    win_hi64 = _llvm_d.ZExtOp(_I64(), win_hi).res
    stride64 = _llvm_d.ZExtOp(_I64(), stride).res
    peer = arith.unwrap(peer_i32)
    peer64 = _llvm_d.ZExtOp(_I64(), peer).res if peer.type == _I32() else peer
    offset = arith.unwrap(offset_i64)
    offset64 = (
        _llvm_d.ZExtOp(_I64(), offset).res if offset.type == _I32() else offset
    )
    shift32 = _llvm_d.ConstantOp(
        _I64(), ir.IntegerAttr.get(_I64(), 32)
    ).result
    win_hi_shifted = _llvm_d.ShlOp(win_hi64, shift32, _NUW()).res
    win_base = _llvm_d.OrOp(win_lo64, win_hi_shifted).res
    peer_stride = _llvm_d.MulOp(peer64, stride64, _NUW()).res
    peer_stride = _llvm_d.ShlOp(peer_stride, shift32, _NUW()).res
    peer_base = _llvm_d.AddOp(win_base, peer_stride, _NUW()).res
    return fx.Int64(_llvm_d.AddOp(peer_base, offset64, _NUW()).res)


def atomic_add_global(addr_i64, val):
    """Monotonic remote global fetch-and-add at addr_i64; returns old (i64/i32 per val)."""
    return _llvm_d.AtomicRMWOp(
        _llvm_d.AtomicBinOp.add,
        _gptr(addr_i64),
        arith.unwrap(val),
        _llvm_d.AtomicOrdering.monotonic,
    ).res


def store_i32_system(addr_i64, offset, val):
    """System-release i32 store at addr + offset*4."""
    addr = _addr_plus(addr_i64, offset, 4)
    gptr = _llvm_d.IntToPtrOp(_llvm_d.PointerType.get(address_space=1), addr).result
    _llvm_d.StoreOp(
        arith.unwrap(val),
        gptr,
        alignment=4,
        ordering=_llvm_d.AtomicOrdering.release,
        syncscope="one-as",
    )


def store_i64_system(addr_i64, offset, val):
    """System-release i64 store at addr + offset*8."""
    addr = _addr_plus(addr_i64, offset, 8)
    gptr = _llvm_d.IntToPtrOp(_llvm_d.PointerType.get(address_space=1), addr).result
    _llvm_d.StoreOp(
        arith.unwrap(val),
        gptr,
        alignment=8,
        ordering=_llvm_d.AtomicOrdering.release,
        syncscope="one-as",
    )


def _is_gfx12():
    try:
        from mori.jit.config import detect_gpu_arch

        return detect_gpu_arch().startswith("gfx12")
    except Exception:
        return False


def waitcnt_all():
    """Drain all outstanding memory counters (no cache management, unlike a
    release fence). gpu.barrier/s_barrier only syncs wavefronts and does NOT
    wait for in-flight memory ops, so this must precede a grid barrier when the
    stores before it need to be complete. gfx12/gfx1250 split the legacy
    s_waitcnt into per-kind counters."""
    if _is_gfx12():
        _rocdl_d.s_wait_storecnt(0)
        _rocdl_d.s_wait_loadcnt(0)
    else:
        _rocdl_d.s_waitcnt(0)


def fence_system_acquire():
    _llvm_d.FenceOp(_llvm_d.AtomicOrdering.acquire, syncscope="one-as")


def fence_system_release():
    _llvm_d.FenceOp(_llvm_d.AtomicOrdering.release, syncscope="one-as")


def _unwrap(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def load_i32_acquire(addr_i64):
    """Volatile monotonic i32 load at addr_i64 (global addrspace 1).

    Volatile + atomic ordering keeps the spin re-read from being hoisted/CSE'd
    out of the wait loop — a plain (even uncached) buffer_load can be lifted by
    LICM, making the loop spin forever on a stale value.
    """
    gptr = _gptr(addr_i64)
    return _llvm_d.LoadOp(
        _I32(),
        gptr,
        alignment=4,
        volatile_=True,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope="one-as",
    ).res


def load_i64_acquire(addr_i64):
    """Volatile monotonic i64 load at addr_i64 (global addrspace 1)."""
    gptr = _gptr(addr_i64)
    return _llvm_d.LoadOp(
        _I64(),
        gptr,
        alignment=8,
        volatile_=True,
        ordering=_llvm_d.AtomicOrdering.monotonic,
        syncscope="one-as",
    ).res


def load_i32_nt(base_i64, offset):
    """Non-temporal global i32 load at base + offset*4 (addrspace 1). A raw global
    load (VGPR address) avoids the per-expert buffer-descriptor waterfall, keeping
    the K combine loads in flight. Caller must ensure the address is in-bounds."""
    addr = _addr_plus(base_i64, offset, 4)
    gptr = _llvm_d.IntToPtrOp(_llvm_d.PointerType.get(address_space=1), addr).result
    return _llvm_d.LoadOp(_I32(), gptr, alignment=4, nontemporal=True).res


def _V4I32():
    return ir.VectorType.get([4], _I32())


def load_v4i32_nt(base_i64, offset):
    """Non-temporal global vec4 i32 load at base + offset*4 (addrspace 1).
    Returns vector<4xi32> (16 bytes, codegen: global_load_dwordx4). offset is in
    i32 units; alignment=4 since the caller's per-warp offset is only 4B-aligned
    (AMDGPU emits dwordx4 for a 4B-aligned global vector load)."""
    addr = _addr_plus(base_i64, offset, 4)
    gptr = _llvm_d.IntToPtrOp(_llvm_d.PointerType.get(address_space=1), addr).result
    return _llvm_d.LoadOp(_V4I32(), gptr, alignment=4, nontemporal=True).res


def _spin(addr_i64, keep_waiting):
    """Spin on a volatile/atomic i32 load at addr_i64; keep_waiting(cur_fx_i32)
    returns the fx-bool "should keep spinning". Returns the awaited fx.Int32.

    Self-contained — mori.ir.flydsl's wait_until_* are tied to ShmemStates and
    cannot run on a cco-only stack (they assert without ShmemInit).
    """
    i32 = T.i32
    first = load_i32_acquire(addr_i64)
    loop = scf.WhileOp([i32], [_unwrap(first)])
    cond = ir.Block.create_at_start(loop.before, [i32])
    body = ir.Block.create_at_start(loop.after, [i32])
    with ir.InsertionPoint(cond):
        cur = fx.Int32(cond.arguments[0])
        scf.ConditionOp(_unwrap(keep_waiting(cur)), [cond.arguments[0]])
    with ir.InsertionPoint(body):
        nxt = load_i32_acquire(addr_i64)
        scf.YieldOp([_unwrap(nxt)])
    return fx.Int32(loop.results[0])


def _spin64(addr_i64, keep_waiting):
    """i64 analogue of _spin (volatile monotonic loads)."""
    i64 = T.i64
    first = load_i64_acquire(addr_i64)
    loop = scf.WhileOp([i64], [_unwrap(first)])
    cond = ir.Block.create_at_start(loop.before, [i64])
    body = ir.Block.create_at_start(loop.after, [i64])
    with ir.InsertionPoint(cond):
        cur = fx.Int64(cond.arguments[0])
        scf.ConditionOp(_unwrap(keep_waiting(cur)), [cond.arguments[0]])
    with ir.InsertionPoint(body):
        nxt = load_i64_acquire(addr_i64)
        scf.YieldOp([_unwrap(nxt)])
    return fx.Int64(loop.results[0])


def spin_until_eq_i64(addr_i64, val):
    """Spin until *addr (i64) == val."""
    return _spin64(addr_i64, lambda cur: cur != fx.Int64(val))


def spin_until_ge_i64(addr_i64, val):
    """Spin until *addr (i64) >= val (signed).

    For a MONOTONIC cross-device flag this is deadlock-safe under lapping: if a
    faster peer overwrites the slot with a higher call count before a slow rank
    reads it, `>=` still passes (whereas `==` would spin forever). It only
    releases early when the peer is *ahead* — the safe direction for a
    wait-for-peer barrier.
    """
    return _spin64(addr_i64, lambda cur: cur < fx.Int64(val))


def spin_until_eq_i32(addr_i64, val):
    """Spin until *addr == val."""
    return _spin(addr_i64, lambda cur: cur != fx.Int32(val))


def spin_until_gt_i32(addr_i64, val):
    """Spin until *addr > val (signed); return the value seen."""
    return _spin(addr_i64, lambda cur: cur <= fx.Int32(val))
