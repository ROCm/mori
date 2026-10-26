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
"""u32 signal-slot accessors for FlyDSL kernels, on FlyDSL 0.2.x through 0.4.x.

Peer-visible flags (a slot a peer writes and we spin on, or one we publish) and
rank-private counters, addressed by a raw i64 device address. Shared by
``mori.ops.gemm_ar`` and the cco examples, so neither depends on the other.

These are **atomic**, not plain buffer accesses with a cache modifier, for two
independent reasons -- both found by reading the emitted ISA:

1. A ``rocdl.raw.ptr.buffer.load`` is side-effect-free, so the spin loop that
   reloads it was dead-code-eliminated outright: the barrier compiled down to
   a signal store and nothing else. An atomic is never removed.
2. The syncscope tells the AMDGPU backend which cache levels to bypass, so we
   stop hand-encoding ``aux`` bits whose meaning changes between CDNA3 and
   CDNA4 (see the CM_* note in :mod:`mori.cco.device.flydsl.buffer`).

Scopes mirror aiter's start_sync/end_sync: publish at **system** scope (the
store must cross xGMI), poll at **agent** scope (we only read our own HBM).
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm_d

from .buffer import _dtype

_SYSTEM_SCOPE = ""  # LLVM AMDGPU: the empty syncscope is system-wide
_AGENT_SCOPE = "agent"

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


# For a slot only this rank reads and writes (a per-block flag, an epoch): no
# atomic and no scope needed, and making it atomic serialises every lane on one
# address. ``volatile`` is still required so the load is not hoisted above the
# barrier that publishes the previous value.
def local_load_u32(ptr):
    return _llvm_d.load(_dtype("i32"), ptr, alignment=4, volatile_=True)


def local_store_u32(ptr, value):
    _llvm_d.store(fx.Int32(value).ir_value(), ptr, alignment=4, volatile_=True)


def i32_type():
    return _dtype("i32")


__all__ = [
    "signal_ptr",
    "signal_store_u32",
    "signal_load_u32",
    "local_load_u32",
    "local_store_u32",
    "i32_type",
]
