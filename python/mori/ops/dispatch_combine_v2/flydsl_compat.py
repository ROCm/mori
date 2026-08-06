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
"""FlyDSL 0.2.x / 0.3.x compatibility for the EPv2 device kernels.

0.3.0 dropped `flydsl.expr.vector` and `flydsl.expr.buffer_ops`, and turned
`T.<dtype>` from a factory into a property. This module re-exposes all three
under the spellings the kernels already use. Everything else they touch (rocdl
wrappers, arith, the llvm/scf/vector dialects, flyc) is unchanged.

Which branch is taken is decided by probing for the modules themselves, so
forks and backports resolve correctly whatever they call themselves; the
reported version is only recorded, for logs and diagnostics.

The two probes differ in kind: `buffer_ops` is a submodule, so its absence is a
ModuleNotFoundError and anything else means a broken install that must surface
as itself. `vector` is only a name in `flydsl.expr` — which is already imported
below, so by then a plain ImportError can only mean the name is gone.
"""
import inspect
import logging

import flydsl
from flydsl._mlir import ir as _ir
from flydsl.expr import T as _T
from flydsl.expr import arith as _arith

logger = logging.getLogger(__name__)

FLYDSL_VERSION = getattr(flydsl, "__version__", "unknown")

try:  # flydsl <= 0.2.x
    from flydsl.expr import vector

    HAS_EXPR_VECTOR = True
except ImportError:  # flydsl >= 0.3.0 — the name, not a module, is gone
    HAS_EXPR_VECTOR = False
    from flydsl._mlir.dialects import vector as _vector

    class _VectorNamespace:
        """MLIR vector dialect; only `extract`'s argument order differs."""

        def __getattr__(self, name):
            return getattr(_vector, name)

        @staticmethod
        def extract(vector, static_position=None, dynamic_position=None):
            # dialect op is (source, dynamic_position, static_position), both required
            return _vector.extract(
                vector, dynamic_position or [], static_position or []
            )

    vector = _VectorNamespace()


class _DTypeNamespace:
    """`T` whose dtype attributes stay callable on every flydsl version."""

    def __getattr__(self, name):
        try:
            # getattr_static keeps 0.3.0's property unfired: it needs a live MLIR
            # context, so `_BALLOT_INT = T.i64` must not resolve at import time.
            raw = inspect.getattr_static(_T, name)
        except AttributeError:
            return getattr(_ir, name)  # 0.3.0 moved VectorType to the ir module
        if isinstance(raw, property):
            return lambda: getattr(_T, name)  # 0.3.x dtype accessor
        return getattr(_T, name)  # 0.2.x factory, or a class such as VectorType


T = _DTypeNamespace()

try:  # flydsl <= 0.2.x
    from flydsl.expr.buffer_ops import (  # noqa: F401
        buffer_load,
        buffer_store,
        create_buffer_resource_from_addr,
    )

    HAS_BUFFER_OPS = True
except ModuleNotFoundError:  # flydsl >= 0.3.0
    HAS_BUFFER_OPS = False

    from flydsl.expr.rocdl import make_buffer_ptr
    from flydsl.expr.typing import (
        AddressSpace,
        PointerType,
        inttoptr,
        ptr_load,
        ptr_store,
    )

    # `mask` / `cache_modifier` are kept for signature parity but ignored: 0.3.0
    # has no V# cache-policy equivalent. Only the scatter-combine Stage-3 read
    # passes one, where it is a hint rather than a correctness requirement.

    def create_buffer_resource_from_addr(addr_i64, *, num_records_bytes=None):
        """Raw i64 device address -> buffer-descriptor pointer."""
        # i32-typed: every EPv2 access indexes in 4-byte units, and opaque
        # pointers let each access pick its own load/store type.
        pty = PointerType.get(T.i32(), AddressSpace.Global)
        return make_buffer_ptr(
            inttoptr(pty, addr_i64), num_records_bytes=num_records_bytes
        )

    def buffer_load(rsrc, offset, vec_width=4, dtype=None, mask=None, cache_modifier=0):
        """Load `vec_width` x `dtype` at element `offset` of `rsrc`."""
        elem = dtype if dtype is not None else T.i32()
        ty = elem if vec_width == 1 else T.VectorType.get([vec_width], elem)
        # unwrap to an ArithValue, as 0.2.x returned: callers do arithmetic on
        # the result and also feed it to arith.* ops, which need an ir.Value.
        return _arith.unwrap(ptr_load(rsrc + offset, result_type=ty))

    def buffer_store(data, rsrc, offset, mask=None, cache_modifier=0):
        """Store `data` at element `offset` of `rsrc`."""
        return ptr_store(data, rsrc + offset)


logger.debug(
    "flydsl %s: expr.vector=%s expr.buffer_ops=%s",
    FLYDSL_VERSION,
    HAS_EXPR_VECTOR,
    HAS_BUFFER_OPS,
)
