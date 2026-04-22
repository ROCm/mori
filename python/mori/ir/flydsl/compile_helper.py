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
# ``mori.shmem.shmem_module_init(hip_module: int) -> None``; routing
# future additions (stream handle, device id, …) through this alias
# surfaces mismatches at import time instead of at first kernel
# launch.
ModuleInitFn = Callable[[int], None]

# FlyDSL's mori-shmem bitcode ABI.  Every supported arch currently
# ships COV 6; if a future arch needs a different value, turn this
# constant into a dict keyed by arch and add the arch to
# ``_SUPPORTED_ARCHS`` below.
_DEFAULT_FLYDSL_COV = 6

# Archs this integration layer accepts.  Keep intentionally narrow:
# every entry here implies that ``libmori_shmem_device.bc`` is known
# to build and run correctly on that arch with ``_DEFAULT_FLYDSL_COV``.
_SUPPORTED_ARCHS: frozenset = frozenset({"gfx942", "gfx950"})


@dataclass(frozen=True)
class FlyDSLCompileInfo:
    """Everything FlyDSL needs in order to compile a kernel that uses
    mori shmem device functions."""

    bitcode_path: str
    module_init_fn: ModuleInitFn


@lru_cache(maxsize=None)
def get_flydsl_compile_info(arch: Optional[str] = None) -> FlyDSLCompileInfo:
    """Resolve bitcode + module-init callable for the FlyDSL backend.

    Args:
        arch: GPU arch.  Must be ``None`` or one of
            ``_SUPPORTED_ARCHS`` (currently ``{"gfx942", "gfx950"}``).
            All supported archs share ``cov=6`` today so ``arch`` is a
            runtime no-op; it exists so typos fail loudly and so
            per-arch dispatch can be wired in without changing call
            sites.

    Results are cached per ``arch``; tests can invalidate with
    ``get_flydsl_compile_info.cache_clear()``.

    Raises:
        ValueError: if ``arch`` is not in ``_SUPPORTED_ARCHS`` (and not ``None``).
        ImportError: if the ``mori.shmem`` C extension is unavailable.
        FileNotFoundError: from :func:`mori.ir.bitcode.find_bitcode` if
            no bitcode can be located or JIT-built.
    """
    if arch is not None and arch not in _SUPPORTED_ARCHS:
        raise ValueError(
            f"Unsupported arch {arch!r}; expected None or one of "
            f"{sorted(_SUPPORTED_ARCHS)}."
        )

    # Deferred import: keeps ``import mori.ir.flydsl`` free of heavy
    # subpackage loads (matches the lazy-init idiom of the package).
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

    return FlyDSLCompileInfo(
        bitcode_path=find_bitcode(cov=_DEFAULT_FLYDSL_COV),
        module_init_fn=ms.shmem_module_init,
    )
