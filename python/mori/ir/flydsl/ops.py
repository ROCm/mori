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
"""
FlyDSL @flyc.kernel wrappers for mori shmem device functions.

Auto-generated from ``mori.ir.ops.MORI_DEVICE_FUNCTIONS`` metadata.
Each function maps 1-to-1 to a C symbol in ``libmori_shmem_device.bc``
and is callable inside ``@flyc.kernel`` functions.

Usage::

    from mori.ir import flydsl as mori_shmem

    @flyc.kernel
    def my_kernel(buf: fx.Tensor):
        pe  = mori_shmem.my_pe()
        npe = mori_shmem.n_pes()
        mori_shmem.quiet_thread_pe(pe)

Note
----

The :class:`ExternFunction` instances this module exposes are
module-level singletons and are **not picklable**.  FlyDSL's
``CompiledArtifact`` on-disk cache round-trips compiled kernels by
serialising ``module_init_fn`` as a ``module:qualname`` string (and
re-importing it on cache hit), not by pickling the ExternFunction
itself — so as long as :func:`mori.shmem.shmem_module_init` stays a
top-level callable, the cache works correctly across runs.
"""

from mori.ir.ops import MORI_DEVICE_FUNCTIONS, SIGNAL_SET, SIGNAL_ADD

from .compile_helper import get_flydsl_compile_info


_all_ops = None


def _get_extern_cls():
    try:
        from flydsl.compiler.extern import ExternFunction
        return ExternFunction
    except ImportError as e:
        raise ImportError(
            "flydsl.compiler.extern not found. "
            "Make sure FlyDSL is installed and flydsl/compiler/extern.py exists."
        ) from e


def _build_all():
    """Create ExternFunction wrappers from MORI_DEVICE_FUNCTIONS.

    Each wrapper carries (bitcode_path, module_init_fn) sourced from
    :func:`compile_helper.get_flydsl_compile_info`, so the FlyDSL JIT
    pipeline receives everything it needs via ``CompilationContext``
    without ever importing mori from its compiler path.
    """
    ExternFunction = _get_extern_cls()
    info = get_flydsl_compile_info()

    ns = {}
    for name, meta in MORI_DEVICE_FUNCTIONS.items():
        ns[name] = ExternFunction(
            symbol=meta["symbol"],
            arg_types=meta["args"],
            ret_type=meta["ret"],
            # Forward mori's pure-function declaration (currently only
            # n_pes).  FlyDSL will lower this to llvm.func readnone /
            # willreturn attributes once extern-attribute support lands,
            # enabling CSE / LICM for the affected calls.
            is_pure=meta.get("pure", False),
            bitcode_path=info.bitcode_path,
            module_init_fn=info.module_init_fn,
        )
    return ns


def _ensure_ops():
    global _all_ops
    if _all_ops is None:
        _all_ops = _build_all()
        globals().update(_all_ops)
    return _all_ops


def __getattr__(name):
    """Lazy init: FlyDSL is only required when an op is first accessed."""
    ops = _ensure_ops()
    if name in ops:
        return ops[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(MORI_DEVICE_FUNCTIONS.keys()) + ["SIGNAL_SET", "SIGNAL_ADD"]
