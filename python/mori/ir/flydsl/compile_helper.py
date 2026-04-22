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
Mori-side compile metadata for the FlyDSL backend.

This module is the **single source of truth** for everything FlyDSL needs
to know at compile time in order to use mori shmem device functions:
the LLVM bitcode path (with the correct code-object version), the
post-load initialiser, and any future per-arch knobs.

The consumer is ``mori.ir.flydsl.ops`` which calls
:func:`get_flydsl_compile_info` once on first access and attaches the
metadata to every :class:`ExternFunction` wrapper.  FlyDSL's compiler
path is intentionally *not* expected to import this file — the metadata
reaches it through ``CompilationContext`` populated by ExternFunction.

Keeping this layer lets mori evolve its bitcode selection (per-arch COV
mapping, NIC variants, JIT fallback, …) without touching ``ops.py`` or
any FlyDSL code.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

# Signature contract for the post-load module initializer.  Matches
# ``mori.shmem.api.shmem_module_init(hip_module: int) -> None``; any
# future addition (stream handle, device id, …) must go through this
# alias so mismatches surface at import time instead of at first
# kernel launch.
ModuleInitFn = Callable[[int], None]

# gfx9xx currently all use COV 6 for the FlyDSL ABI.  When future archs
# diverge, replace this constant with an arch-keyed lookup inside
# :func:`get_flydsl_compile_info` — the ``arch`` parameter is already
# wired through so callers don't need to change.
_DEFAULT_FLYDSL_COV = 6


@dataclass(frozen=True)
class FlyDSLCompileInfo:
    """Everything FlyDSL needs in order to compile a kernel that uses
    mori shmem device functions."""

    bitcode_path: str
    cov: int
    module_init_fn: ModuleInitFn


@lru_cache(maxsize=None)
def get_flydsl_compile_info(arch: Optional[str] = None) -> FlyDSLCompileInfo:
    """Resolve bitcode + module-init callable for the FlyDSL backend.

    Args:
        arch: Optional GPU arch override (e.g. ``"gfx942"``).  Currently
            unused — all supported archs share ``cov=6`` — but the
            parameter is part of the stable API so future arch-keyed
            cov selection does not require touching callers in
            :mod:`mori.ir.flydsl.ops`.

    Results are cached by ``arch`` (via :func:`functools.lru_cache`),
    so repeated calls are free and tests can invalidate the cache with
    ``get_flydsl_compile_info.cache_clear()``.

    Raises:
        ImportError: if the ``mori.shmem`` C extension is unavailable.
        FileNotFoundError: from :func:`mori.ir.bitcode.find_bitcode` if
            no bitcode can be located or JIT-built.
    """
    from mori.ir.bitcode import find_bitcode

    try:
        import mori.shmem as ms
    except ImportError as exc:
        raise ImportError(
            "mori.shmem is unavailable (the C extension likely failed "
            "to build or is missing from this environment).  Reinstall "
            "mori on a host with HIP/RCCL available, e.g. "
            "`pip install -e mori/`."
        ) from exc

    cov = _DEFAULT_FLYDSL_COV
    return FlyDSLCompileInfo(
        bitcode_path=find_bitcode(cov=cov),
        cov=cov,
        module_init_fn=ms.shmem_module_init,
    )
