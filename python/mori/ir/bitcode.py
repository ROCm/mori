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
Locate the mori shmem device bitcode (libmori_shmem_device.bc).

Search order:
  1. ``MORI_SHMEM_BC`` environment variable (explicit override)
  2. JIT cache (NIC-specific bitcode compiled for the runtime hardware)
  3. Alongside this Python file  (``python/mori/ir/``)
  4. ``<mori_repo>/lib/``
  5. ``<mori_repo>/build/lib/``
  6. JIT compile from source (if source tree is available and cache miss)
"""

import os
from pathlib import Path

_BC_FILENAME = "libmori_shmem_device.bc"
# Per-cov caches: {cov: path}
_cached_paths: dict[int | None, str] = {}


def _find_jit_cached_bitcode(*, cov: int | None = None) -> str | None:
    """Look for a previously JIT-compiled bitcode in the cache directory."""
    try:
        from mori.jit.config import (
            detect_build_config,
            detect_nic_type,
            get_mori_source_root,
        )

        cfg = detect_build_config()
        nic = detect_nic_type()
        mori_root = get_mori_source_root()
        if mori_root is None:
            return None

        from mori.jit.cache import get_cache_dir

        source_paths = [
            mori_root / "src" / "shmem" / "shmem_device_api_wrapper.cpp",
            mori_root / "include" / "mori" / "shmem",
            mori_root / "include" / "mori" / "core",
        ]
        cache_dir = get_cache_dir(cfg.arch, source_paths, nic, cov=cov)
        bc_path = cache_dir / _BC_FILENAME
        if bc_path.is_file():
            return str(bc_path)
    except Exception:
        pass
    return None


def find_bitcode(*, cov: int | None = None) -> str:
    """Return the absolute path to ``libmori_shmem_device.bc``.

    Args:
        cov: AMDGPU code object version.  ``None`` (default) uses the
             legacy search (pre-built bitcode or JIT with default cov=5,
             compatible with Triton).  Pass ``cov=6`` for FlyDSL which
             uses ABI 600.

    Prefers JIT-cached bitcode (compiled with the correct NIC branch for the
    runtime hardware) over pre-built bitcode that may lack NIC-specific symbols.

    Raises ``FileNotFoundError`` if the bitcode cannot be located or built.
    """
    cached = _cached_paths.get(cov)
    if cached is not None:
        return cached

    env = os.environ.get("MORI_SHMEM_BC")
    if env and os.path.isfile(env):
        _cached_paths[cov] = env
        return env

    jit_cached = _find_jit_cached_bitcode(cov=cov)
    if jit_cached:
        _cached_paths[cov] = jit_cached
        return jit_cached

    # Pre-built bitcode (only valid when cov is not specified)
    pre_built: list[str] = []
    if cov is None:
        here = Path(__file__).resolve().parent
        pre_built.append(str(here / _BC_FILENAME))

        mori_root = here.parent.parent.parent
        pre_built.append(str(mori_root / "lib" / _BC_FILENAME))
        pre_built.append(str(mori_root / "build" / "lib" / _BC_FILENAME))

    jit_disabled = os.environ.get("MORI_DISABLE_JIT", "").lower() in ("1", "true", "on")

    if not jit_disabled:
        try:
            from mori.jit.core import ensure_bitcode

            jit_cov = cov if cov is not None else 5
            path = ensure_bitcode(cov=jit_cov)
            _cached_paths[cov] = path
            return path
        except Exception:
            pass

    for p in pre_built:
        if os.path.isfile(p):
            _cached_paths[cov] = p
            return p

    raise FileNotFoundError(
        f"{_BC_FILENAME} not found (cov={cov}). Searched: {pre_built}\n"
        "Enable JIT compilation (unset MORI_DISABLE_JIT) or run:\n"
        "  MORI_PRECOMPILE=1 python -c 'import mori'"
    )


def get_bitcode_path(*, cov: int | None = None) -> str:
    """Alias for :func:`find_bitcode`."""
    return find_bitcode(cov=cov)
