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
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""JIT cache directory management and content hashing."""

import hashlib
import os
from pathlib import Path


def get_cache_root() -> Path:
    """Return the JIT cache root directory.

    Default: ``~/.mori/jit/``.  Override with ``MORI_JIT_CACHE_DIR``.
    """
    env = os.environ.get("MORI_JIT_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".mori" / "jit"


def _hash_tree(paths: list[Path]) -> str:
    """Compute a short content hash over files and directories.

    For directories, all ``.hpp``, ``.h``, ``.cpp``, and ``.hip`` files are
    included, so ``#include``d translation units invalidate the cache too.
    """
    h = hashlib.sha256()
    for p in sorted(paths):
        if p.is_file():
            h.update(p.read_bytes())
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix in (".hpp", ".h", ".cpp", ".hip"):
                    h.update(f.read_bytes())
    return h.hexdigest()[:12]


def get_cache_dir(
    arch: str,
    source_paths: list[Path],
    nic: str = "mlx5",
    profiler: bool = False,
    *,
    cov: int | None = None,
    ccqe: bool = False,
) -> Path:
    """Return the cache directory for a specific arch + NIC + content combo.

    Structure: <cache_root>/<arch>_<nic>[_ccqe][_profiler][_cov<N>]/<content_hash>/

    Args:
        profiler: When True, appends '_profiler' to the directory name so that
                  kernels compiled with ENABLE_PROFILER are cached separately.
        cov: AMDGPU code object version. When specified, the version is
             included in the directory name to separate bitcode compiled
             with different ABI versions (e.g. cov5 for Triton, cov6 for
             FlyDSL).  None omits the suffix for backward compatibility.
        ccqe: When True, appends '_ccqe' so CCQE and non-CCQE kernels are
              cached separately (they differ by -DIONIC_CCQE compile flag).
    """
    content_hash = _hash_tree(source_paths)
    ccqe_suffix = "_ccqe" if ccqe else ""
    profiler_suffix = "_profiler" if profiler else ""
    cov_suffix = f"_cov{cov}" if cov is not None else ""
    # Experimental TDM dispatch adds -DMORI_DISP_TDM to the JIT compile; cache it
    # separately so toggling MORI_DISP_TDM never reuses the other variant's .hsaco.
    tdm_suffix = (
        "_disptdm"
        if os.environ.get("MORI_DISP_TDM", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    # MORI_DISP_NOTIFY / MORI_DISP_NOTIFY_CNT2 no longer affect the compiled code
    # (both dispatch kernels are always built; host selects at runtime), so they are
    # intentionally NOT part of the cache key — one .hsaco serves both paths.
    timing_suffix = (
        "_disptiming"
        if os.environ.get("MORI_DISP_TIMING", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    # -DMORI_DISP_BAREB swaps in the debug bare-B batch body (metadata stripped);
    # cache separately so it never reuses the clean 980 .hsaco.
    bareb_suffix = (
        "_dispbareb"
        if os.environ.get("MORI_DISP_BAREB", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    # -DMORI_DISP_TILE2D reshapes the TDM payload tile to a 2D rectangle; cache it
    # separately so toggling it never reuses the 1xN-tile .hsaco.
    tile2d_suffix = (
        "_disptile2d"
        if os.environ.get("MORI_DISP_TILE2D", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    splitmeta_suffix = (
        "_dispsplitmeta"
        if os.environ.get("MORI_DISP_SPLITMETA", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    nometa_suffix = (
        "_dispnometa"
        if os.environ.get("MORI_DISP_NOMETA", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    metaphase_suffix = (
        "_dispmetaphase"
        if os.environ.get("MORI_DISP_METAPHASE", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    _comp = "".join(
        s
        for v, s in (
            ("MORI_DISP_NOSCALES", "_noscales"),
            ("MORI_DISP_NOIDXW", "_noidxw"),
            ("MORI_DISP_NOSRCMAP", "_nosrcmap"),
            ("MORI_DISP_METAONLY", "_metaonly"),
            ("MORI_DISP_METAWIDE", "_metawide"),
        )
        if os.environ.get(v, "").lower() in ("1", "true", "on", "yes")
    )
    if os.environ.get("MORI_DISP_CUSPLIT", "").lower() in ("1", "true", "on", "yes"):
        _comp += "_cusplit"
        _pb = os.environ.get("MORI_DISP_PAYLOAD_BLOCKS", "").strip()
        if _pb.isdigit():
            _comp += f"pb{int(_pb)}"
    cntstep_suffix = (
        f"_cntstep{os.environ['MORI_CNT_STEP'].strip()}"
        if os.environ.get("MORI_CNT_STEP", "").strip().isdigit()
        else ""
    )
    d = (
        get_cache_root()
        / f"{arch}_{nic}{ccqe_suffix}{profiler_suffix}{cov_suffix}{tdm_suffix}{timing_suffix}{bareb_suffix}{tile2d_suffix}{splitmeta_suffix}{nometa_suffix}{metaphase_suffix}{_comp}{cntstep_suffix}"
        / content_hash
    )
    d.mkdir(parents=True, exist_ok=True)
    return d
