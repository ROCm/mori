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
Compile helper for FlyDSL <-> Mori shmem integration.

Provides :func:`prepare_compile` as the single entry point for FlyDSL to
obtain bitcode paths and post-load processors without hard-coding any
Mori-internal details (symbol prefixes, COV versions, etc.).
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Set

# gfx9xx all use COV 6 for FlyDSL ABI
_DEFAULT_FLYDSL_COV = 6


def _build_known_symbols() -> frozenset:
    from mori.ir.ops import MORI_DEVICE_FUNCTIONS
    return frozenset(m["symbol"] for m in MORI_DEVICE_FUNCTIONS.values())


_known_symbols: Optional[frozenset] = None


def _get_known_symbols() -> frozenset:
    global _known_symbols
    if _known_symbols is None:
        _known_symbols = _build_known_symbols()
    return _known_symbols


@dataclass
class ShmemCompileInfo:
    """Information returned by :func:`prepare_compile`."""
    link_libs: List[str]
    module_init_fn: Callable


def prepare_compile(
    extern_symbols: Set[str],
    arch: str = "gfx942",
) -> Optional[ShmemCompileInfo]:
    """Determine whether *extern_symbols* require mori shmem support.

    Returns a :class:`ShmemCompileInfo` with the bitcode path and a
    module-init callable if shmem symbols are detected, or ``None``
    otherwise.  FlyDSL should pass ``link_libs`` to the MLIR compiler
    and invoke ``module_init_fn(hip_module)`` on each loaded GPU module.
    """
    known = _get_known_symbols()
    if not extern_symbols & known:
        return None

    from mori.ir.bitcode import find_bitcode
    bc_path = find_bitcode(cov=_DEFAULT_FLYDSL_COV)

    try:
        import mori.shmem as ms
    except ImportError as exc:
        raise ImportError(
            "Kernel uses mori shmem symbols but 'mori' is not installed.\n"
            "  pip install: git clone https://github.com/ROCm/mori && cd mori && pip install -e ."
        ) from exc

    return ShmemCompileInfo(
        link_libs=[bc_path],
        module_init_fn=ms.shmem_module_init,
    )


def restore_processors() -> List[Callable]:
    """Re-create post-load processors after deserialization."""
    try:
        import mori.shmem as ms
        return [ms.shmem_module_init]
    except ImportError:
        return []
