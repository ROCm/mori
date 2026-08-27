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
"""Select the ROCm runtime before loading MORI native extensions."""

from __future__ import annotations

import importlib
import os

try:
    from ._rocm_build_info import ROCM_BUILD_VERSION, ROCM_VERSION_PATTERN
except ImportError:
    ROCM_BUILD_VERSION = None
    ROCM_VERSION_PATTERN = None

_PRELOAD_LIBRARIES = ("amd_comgr", "amdhip64")
_TRUE_VALUES = {"1", "on", "true", "yes"}
_bootstrap_complete = False


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _loaded_libamdhip64_paths() -> tuple[str, ...]:
    """Return unique mapped HIP runtime paths in the current linker namespace."""

    paths: set[str] = set()
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            for line in maps:
                fields = line.rstrip("\n").split(maxsplit=5)
                if len(fields) < 6:
                    continue
                path = fields[5]
                if path.endswith(" (deleted)"):
                    path = path[: -len(" (deleted)")]
                if path.startswith("/") and os.path.basename(path).startswith(
                    "libamdhip64.so"
                ):
                    paths.add(os.path.realpath(path))
    except OSError:
        pass
    return tuple(sorted(paths))


def ensure_rocm_runtime() -> tuple[str, ...]:
    """Preload TheRock's HIP runtime before a MORI extension can load another.

    If HIP is already mapped (for example because Torch was imported first), it
    is reused. If ``rocm_sdk`` is unavailable, MORI's existing RUNPATH-based
    system ROCm loading remains in effect.
    """

    global _bootstrap_complete

    if _bootstrap_complete:
        return _loaded_libamdhip64_paths()

    loaded_paths = _loaded_libamdhip64_paths()
    if loaded_paths or _env_enabled("MORI_DISABLE_ROCM_SDK"):
        _bootstrap_complete = True
        return loaded_paths

    try:
        rocm_sdk = importlib.import_module("rocm_sdk")
    except ModuleNotFoundError as exc:
        if exc.name == "rocm_sdk":
            _bootstrap_complete = True
            return ()
        raise RuntimeError("Installed rocm_sdk package is incomplete") from exc
    except Exception as exc:
        raise RuntimeError("Failed to import the installed rocm_sdk package") from exc

    init_kwargs = {
        "preload_shortnames": list(_PRELOAD_LIBRARIES),
        "fail_on_version_mismatch": False,
    }
    if ROCM_VERSION_PATTERN is not None:
        init_kwargs["check_version"] = ROCM_VERSION_PATTERN

    try:
        rocm_sdk.initialize_process(**init_kwargs)
    except Exception as exc:
        version_note = (
            f" built against ROCm {ROCM_BUILD_VERSION}"
            if ROCM_BUILD_VERSION is not None
            else ""
        )
        raise RuntimeError(
            "rocm_sdk is installed but could not initialize the HIP runtime for MORI"
            f"{version_note}. Repair the ROCm Python packages or set "
            "MORI_DISABLE_ROCM_SDK=1 to explicitly use system ROCm."
        ) from exc

    loaded_paths = _loaded_libamdhip64_paths()
    if not loaded_paths:
        raise RuntimeError(
            "rocm_sdk initialization completed without loading libamdhip64"
        )

    _bootstrap_complete = True
    return loaded_paths


__all__ = ["ensure_rocm_runtime"]
