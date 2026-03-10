# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
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
_cached_path: str | None = None


def _find_jit_cached_bitcode() -> str | None:
    """Look for a previously JIT-compiled bitcode in the cache directory."""
    try:
        from mori.jit.cache import get_cache_root
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
        cache_dir = get_cache_dir(cfg.arch, source_paths, nic)
        bc_path = cache_dir / _BC_FILENAME
        if bc_path.is_file():
            return str(bc_path)
    except Exception:
        pass
    return None


def find_bitcode() -> str:
    """Return the absolute path to ``libmori_shmem_device.bc``.

    Prefers JIT-cached bitcode (compiled with the correct NIC branch for the
    runtime hardware) over pre-built bitcode that may lack NIC-specific symbols.

    Raises ``FileNotFoundError`` if the bitcode cannot be located or built.
    """
    global _cached_path
    if _cached_path is not None:
        return _cached_path

    env = os.environ.get("MORI_SHMEM_BC")
    if env and os.path.isfile(env):
        _cached_path = env
        return env

    jit_cached = _find_jit_cached_bitcode()
    if jit_cached:
        _cached_path = jit_cached
        return jit_cached

    pre_built: list[str] = []
    here = Path(__file__).resolve().parent
    pre_built.append(str(here / _BC_FILENAME))

    mori_root = here.parent.parent.parent
    pre_built.append(str(mori_root / "lib" / _BC_FILENAME))
    pre_built.append(str(mori_root / "build" / "lib" / _BC_FILENAME))

    jit_disabled = os.environ.get("MORI_DISABLE_JIT", "").lower() in ("1", "true", "on")

    if not jit_disabled:
        try:
            from mori.jit.core import ensure_bitcode

            _cached_path = ensure_bitcode()
            return _cached_path
        except Exception:
            pass

    for p in pre_built:
        if os.path.isfile(p):
            _cached_path = p
            return p

    raise FileNotFoundError(
        f"{_BC_FILENAME} not found. Searched: {pre_built}\n"
        "Enable JIT compilation (unset MORI_DISABLE_JIT) or run:\n"
        "  MORI_PRECOMPILE=1 python -c 'import mori'"
    )


get_bitcode_path = find_bitcode
