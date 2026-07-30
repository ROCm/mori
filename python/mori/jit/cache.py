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
    cntstep_suffix = (
        f"_cntstep{os.environ['MORI_CNT_STEP'].strip()}"
        if os.environ.get("MORI_CNT_STEP", "").strip().isdigit()
        else ""
    )
    # -DMORI_DISP_CLEAN builds the legacy clean dispatch body instead of the default one.
    clean_suffix = (
        "_dispclean"
        if os.environ.get("MORI_DISP_CLEAN", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    complbackoff_suffix = (
        f"_cbo{os.environ['MORI_DISP_COMPL_BACKOFF'].strip()}"
        if os.environ.get("MORI_DISP_COMPL_BACKOFF", "").strip().isdigit()
        else ""
    )
    _msp = os.environ.get("MORI_DISP_METASPLIT", "").strip()
    metasplit_suffix = f"_ms{int(_msp)}" if _msp.isdigit() and int(_msp) >= 1 else ""
    # The remaining metadata-phase gates each change the meta send's code. Same source tree,
    # different binary -> each MUST be part of the key or an A/B silently reuses one .hsaco.
    # The lane-parallel FINALIZE and the whole-run meta tile are now unconditional, so they no longer
    # need a suffix; content_hash separates them from the pre-flip binaries that used the old keys.
    metavec_suffix = (
        "_metavec"
        if os.environ.get("MORI_DISP_METAVEC", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    metafield_suffix = (
        "_metafield"
        if os.environ.get("MORI_DISP_METAFIELD", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    paydyn_suffix = (
        "_paydyn"
        if os.environ.get("MORI_DISP_PAYDYN", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    dbl_suffix = "".join(
        f"_{n.lower()}"
        for n in ("DBLCOUNT", "DBLRESERVE")
        if os.environ.get(f"MORI_DISP_{n}", "").lower() in ("1", "true", "on", "yes")
    )
    gridflag_suffix = (
        "_gflag"
        if os.environ.get("MORI_DISP_GRIDFLAG", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    _ps = os.environ.get("MORI_DISP_PAYSPLIT", "").strip()
    paysplit_suffix = f"_ps{int(_ps)}" if _ps.isdigit() and int(_ps) > 1 else ""
    srcvec_suffix = (
        "_srcvec"
        if os.environ.get("MORI_DISP_SRCVEC", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    metalds_suffix = (
        "_metalds"
        if os.environ.get("MORI_DISP_METALDS", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    metafuse_suffix = (
        "_metafuse"
        if os.environ.get("MORI_DISP_METAFUSE", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    nostg_suffix = (
        "_nostg"
        if os.environ.get("MORI_DISP_NOSTG", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    nometa_suffix = (
        "_nometa"
        if os.environ.get("MORI_DISP_NOMETA", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    nopay_suffix = (
        "_nopay"
        if os.environ.get("MORI_DISP_NOPAY", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    metadiag_suffix = (
        "_metadiag"
        if os.environ.get("MORI_DISP_METADIAG", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    d = (
        get_cache_root()
        / f"{arch}_{nic}{ccqe_suffix}{profiler_suffix}{cov_suffix}{tdm_suffix}{timing_suffix}{clean_suffix}{complbackoff_suffix}{metasplit_suffix}{metavec_suffix}{metafield_suffix}{paydyn_suffix}{dbl_suffix}{gridflag_suffix}{paysplit_suffix}{srcvec_suffix}{metalds_suffix}{metafuse_suffix}{nostg_suffix}{nometa_suffix}{nopay_suffix}{metadiag_suffix}{cntstep_suffix}"
        / content_hash
    )
    d.mkdir(parents=True, exist_ok=True)
    return d
