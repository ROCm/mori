# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""
Locate the mori shmem device bitcode (libmori_shmem_device.bc).

Search order:
  1. ``MORI_SHMEM_BC`` environment variable
  2. Alongside this Python file  (``python/mori/ir/``)
  3. ``<mori_repo>/lib/``
  4. ``<mori_repo>/build/lib/``
  5. JIT compile from source (if source tree is available)
"""

import os
from pathlib import Path

_BC_FILENAME = "libmori_shmem_device.bc"
_cached_path: str | None = None


def find_bitcode() -> str:
    """Return the absolute path to ``libmori_shmem_device.bc``.

    Falls back to JIT compilation if no pre-built bitcode is found
    and the mori source tree is available.

    Raises ``FileNotFoundError`` if the bitcode cannot be located or built.
    """
    global _cached_path
    if _cached_path is not None:
        return _cached_path

    candidates: list[str] = []

    env = os.environ.get("MORI_SHMEM_BC")
    if env:
        candidates.append(env)

    here = Path(__file__).resolve().parent
    candidates.append(str(here / _BC_FILENAME))

    mori_root = here.parent.parent.parent
    candidates.append(str(mori_root / "lib" / _BC_FILENAME))
    candidates.append(str(mori_root / "build" / "lib" / _BC_FILENAME))

    for p in candidates:
        if os.path.isfile(p):
            _cached_path = p
            return p

    if os.environ.get("MORI_DISABLE_JIT", "").lower() not in ("1", "true", "on"):
        try:
            from mori.jit.core import ensure_bitcode
            _cached_path = ensure_bitcode()
            return _cached_path
        except Exception as e:
            raise FileNotFoundError(
                f"{_BC_FILENAME} not found (searched: {candidates}) "
                f"and JIT compilation failed: {e}\n"
                "Build it manually with: bash tools/build_shmem_bitcode.sh\n"
                "Or set MORI_SHMEM_BC to the path of the pre-built bitcode."
            ) from e

    raise FileNotFoundError(
        f"{_BC_FILENAME} not found. Searched: {candidates}\n"
        "Build it with: bash tools/build_shmem_bitcode.sh\n"
        "Or enable JIT compilation (unset MORI_DISABLE_JIT)."
    )


get_bitcode_path = find_bitcode
