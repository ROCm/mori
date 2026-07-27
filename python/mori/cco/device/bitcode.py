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
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""
Locate (or JIT-compile) the cco device bitcode (libmori_cco_device.bc).

The bitcode holds the ``extern "C"`` cco GDA/LSA device wrappers (``cco_gda_put``
/ ``cco_lsa_ptr`` / ``cco_devcomm_rank`` / ...) built from
``src/cco/device/cco_device_wrapper.cpp``. Self-contained — independent of the
shmem resolver in ``mori.ir.bitcode`` — but it *reuses* the shmem JIT machinery
in ``mori.jit`` (hipcc + cache), since cco has no global-state shim to link.

Resolution order:
  1. ``MORI_CCO_BC`` environment variable (explicit override)
  2. JIT compile for the runtime arch+NIC (cached), unless ``MORI_DISABLE_JIT``
  3. Pre-built: alongside this package / ``<repo>/lib`` / ``<repo>/build*/lib``
"""

import os
from pathlib import Path

_BC_FILENAME = "libmori_cco_device.bc"
_FLYDSL_COV = 6  # FlyDSL ROCm backend uses ABI 600 -> code object version 6
_cached_paths: dict[int, str] = {}


def _sdma_enabled() -> bool:
    """JIT the device wrapper with SDMA? env > baked _build_flags.py > OFF.

    The baked flag (written by setup.py) records what the host lib was built
    with, so `BUILD_CCO_SDMA=ON pip install .` works at runtime without re-setting
    the env; the env still overrides it.
    """
    env = os.environ.get("BUILD_CCO_SDMA")
    if env is not None:
        return env.strip().upper() in {"1", "ON", "TRUE", "YES"}
    try:
        from mori.cco.device._build_flags import BUILD_CCO_SDMA as _baked

        return bool(_baked)
    except Exception:
        return False


def _prebuilt_candidates() -> list[Path]:
    here = Path(__file__).resolve().parent
    mori_root = here.parents[3]  # python/mori/cco/device/ -> repo root
    out = [here / _BC_FILENAME, mori_root / "lib" / _BC_FILENAME]
    out += [p / "lib" / _BC_FILENAME for p in mori_root.glob("build*")]
    return out


def _jit_compile(cov: int) -> str | None:
    """Compile cco_device_wrapper.cpp to bitcode for the runtime arch+NIC (cached).

    Reuses mori.jit (same hipcc/opt/cache as shmem). Returns the path, or None if
    JIT is unavailable (no source tree / hipcc).
    """
    try:
        from mori.jit.config import (
            detect_build_config,
            detect_nic_type,
            get_mori_source_root,
        )
        from mori.jit.cache import get_cache_dir
        from mori.jit.core import (
            FileBaton,
            _collect_include_dirs,
            _hipcc_device_bc,
            _strip_lifetime_intrinsics,
        )

        mori_root = get_mori_source_root()
        if mori_root is None:
            return None
        wrapper = mori_root / "src" / "cco" / "device" / "cco_device_wrapper.cpp"
        if not wrapper.is_file():
            return None

        cfg = detect_build_config()
        nic = detect_nic_type()
        # SDMA path is compile-gated (BUILD_CCO_SDMA). Resolved from env or the
        # baked-in build flag so it matches the host libmori_cco.so. Folded into
        # the cache dir below so on/off bitcode never aliases.
        sdma_on = _sdma_enabled()
        source_paths = [
            wrapper,
            mori_root / "include" / "mori" / "cco",
            mori_root / "include" / "mori" / "core",
        ]
        cache_dir = get_cache_dir(cfg.arch, source_paths, nic, cov=cov)
        if sdma_on:
            cache_dir = cache_dir / "sdma"
            cache_dir.mkdir(parents=True, exist_ok=True)
        bc_path = cache_dir / _BC_FILENAME
        if bc_path.is_file():
            return str(bc_path)

        sdma_define = [f"-DBUILD_CCO_SDMA={1 if sdma_on else 0}"]
        lock_path = cache_dir / f".{_BC_FILENAME}.lock"
        with FileBaton(lock_path, wait_for=str(bc_path)) as baton:
            if baton.skipped or bc_path.is_file():
                return str(bc_path)
            import tempfile

            include_dirs = _collect_include_dirs(mori_root)
            with tempfile.TemporaryDirectory() as td:
                raw = Path(td) / "cco_wrapper.bc"
                _hipcc_device_bc(
                    cfg, wrapper, include_dirs, raw, cov=cov, extra_defines=sdma_define
                )
                _strip_lifetime_intrinsics(cfg, raw, bc_path)
        return str(bc_path)
    except Exception:
        return None


def find_cco_bitcode(cov: int = _FLYDSL_COV) -> str:
    """Return the absolute path to ``libmori_cco_device.bc`` (cov-specific)."""
    cached = _cached_paths.get(cov)
    if cached is not None:
        return cached

    env = os.environ.get("MORI_CCO_BC")
    if env and os.path.isfile(env):
        _cached_paths[cov] = env
        return env

    jit_disabled = os.environ.get("MORI_DISABLE_JIT", "").lower() in ("1", "true", "on")
    if not jit_disabled:
        p = _jit_compile(cov)
        if p:
            _cached_paths[cov] = p
            return p

    for p in _prebuilt_candidates():
        if p.is_file():
            _cached_paths[cov] = str(p)
            return str(p)

    raise FileNotFoundError(
        f"{_BC_FILENAME} not found and JIT unavailable.\n"
        f"Searched: {[str(c) for c in _prebuilt_candidates()]}\n"
        "Build it with `bash tools/build_cco_bitcode.sh` (-> lib/), set "
        "MORI_CCO_BC=/path/to/libmori_cco_device.bc, or enable JIT "
        "(unset MORI_DISABLE_JIT, needs a source/editable install)."
    )


def get_bitcode_path(cov: int = _FLYDSL_COV) -> str:
    """Alias for :func:`find_cco_bitcode`."""
    return find_cco_bitcode(cov)
