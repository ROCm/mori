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
"""Build inputs for mori_torch_symm, shared by setup.py (prebuild) and the JIT loader.

setup.py loads this file by path before mori is installed, so it must not import mori.
"""

from __future__ import annotations

import glob
import hashlib
import os


def torch_fingerprint(torch) -> str:
    """Hex token for the torch build an extension is compiled against.

    The SymmetricMemory vtable moves between torch builds, so a module built for one
    torch must never be loaded into another; the version string alone does not tell
    source builds apart, hence the git hash.
    """
    parts = (
        torch.__version__,
        str(getattr(torch.version, "git_version", "")),
        str(getattr(torch.version, "hip", "")),
        str(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", "")),
    )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def rocm_home(cpp_extension) -> str:
    """The ROCm torch builds extensions against (pip SDK or system), if it has HIP headers.

    torch derives ROCM_HOME from the first hipcc on PATH, which can be a wrapper script in
    a venv with no headers next to it; fall back to ROCM_PATH and /opt/rocm then.
    """
    candidates = [
        getattr(cpp_extension, "ROCM_HOME", None),
        os.environ.get("ROCM_PATH"),
        "/opt/rocm",
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "include", "hip", "hip_runtime_api.h")):
            return c
    return next((c for c in candidates if c), "/opt/rocm")


def _mapped_hip_runtimes() -> list[str]:
    paths = set()
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            for line in maps:
                fields = line.split(maxsplit=5)
                if len(fields) < 6:
                    continue
                path = fields[5].strip()
                if path.startswith("/") and os.path.basename(path).startswith(
                    "libamdhip64.so"
                ):
                    paths.add(path)
    except OSError:
        pass
    return sorted(paths)


def hip_runtime_library(rocm: str) -> str | None:
    """Path of the HIP runtime to link by file, so the extension records its SONAME.

    The runtime torch already mapped wins. pip-installed ROCm ships only the versioned
    libamdhip64.so.N, so a plain -lamdhip64 cannot find it there.
    """
    mapped = _mapped_hip_runtimes()
    if mapped:
        return mapped[0]
    candidates = sorted(
        glob.glob(os.path.join(rocm, "lib", "libamdhip64.so*")), key=len
    )
    return candidates[0] if candidates else None
