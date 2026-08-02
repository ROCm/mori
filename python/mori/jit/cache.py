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
    # There is no dispatch-TDM suffix: which dispatch body compiles is decided by the arch macros
    # alone now, and `arch` is already the first component of this key.
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
    # Every remaining dispatch gate changes the compiled code. Same source tree, different binary ->
    # each MUST be part of the key or an A/B silently reuses one .hsaco.
    fastdedup_suffix = (
        "_fastdd"
        if os.environ.get("MORI_DISP_FASTDEDUP", "").lower() in ("1", "true", "on", "yes")
        else ""
    )
    # Every combine gate below is read through the resolvers in core.py rather than off the
    # environment, because those now carry ARCH-DEPENDENT DEFAULTS: the same empty environment
    # produces different -D flags on gfx1250 than on gfx942. Reading os.environ here would key both
    # builds the same and hand one of them the other's .hsaco.
    from .core import (
        _comb_barsleep_defines,
        _comb_barspread_defines,
        _comb_tdm_chunks,
        _COMB_OPT_IN_FLAGS,
        _COMB_QUAD_DEFAULT_FLAGS,
        _comb_flag,
    )

    # -DMORI_COMB_TDM=N changes the combine push path AND its chunk count, so N is part of the key.
    combtdm_suffix = f"_combtdm{_comb_tdm_chunks()}" if _comb_tdm_chunks() else ""
    # -DMORI_COMB_BARSLEEP=N changes the barrier's poll backoff, so N is part of the key.
    barsleep_suffix = "".join(
        "_barsl" + d.rsplit("=", 1)[1] for d in _comb_barsleep_defines()
    )
    # -DMORI_COMB_BARSPREAD=N is the per-block line stride of the fanout barrier, so N is in the key.
    barspread_suffix = "".join(
        "_barsp" + d.rsplit("=", 1)[1] for d in _comb_barspread_defines()
    )
    # -DMORI_COMB_PIPE=N is the PULL gather's buffer count, so N is part of the key.
    _pipe = os.environ.get("MORI_COMB_PIPE", "").strip().lower()
    _pipe_n = int(_pipe) if _pipe.isdigit() else (2 if _pipe in ("true", "on", "yes") else 0)
    if _pipe_n == 1:
        _pipe_n = 2
    pipe_suffix = f"_pipe{_pipe_n}" if _pipe_n >= 2 else ""
    # QUAD's depth and split both change the emitted -D, so both are part of the key.
    from .core import _comb_quad_depth, _comb_quad_split

    _quad = _comb_quad_depth()
    if _quad >= 2:
        pipe_suffix += f"_quad{_quad}x{_comb_quad_split()}"
    for _g in _COMB_QUAD_DEFAULT_FLAGS:
        if _comb_flag(f"MORI_COMB_{_g}", bool(_quad >= 2)):
            pipe_suffix += "_" + _g.lower()
    for _g in _COMB_OPT_IN_FLAGS:
        if _comb_flag(f"MORI_COMB_{_g}", False):
            pipe_suffix += "_" + _g.lower()
    from .core import _comb_qloc, _comb_qob, _comb_qtst

    if _comb_qloc():
        pipe_suffix += f"_qloc{_comb_qloc()}"
    for _g in ("THLD", "THST", "SCLD", "SCST"):
        _v = os.environ.get(f"MORI_COMB_{_g}", "").strip()
        if _v.isdigit() and int(_v) > 0:
            pipe_suffix += f"_{_g.lower()}{int(_v)}"
    if _comb_qtst():
        pipe_suffix += f"_qtst{_comb_qtst()}"
        if _comb_qob():
            pipe_suffix += f"o{_comb_qob()}"
    _qred = os.environ.get("MORI_COMB_QRED", "").strip()
    if _qred.isdigit():
        pipe_suffix += f"_qred{int(_qred)}"
    _lb = os.environ.get("MORI_COMB_LB", "").strip()
    if _lb.isdigit() and int(_lb) > 0:
        pipe_suffix += f"_lb{int(_lb)}"
    # Combine's [CSPLIT] bucket print and its deletion diagnostics (all but TIMING are wrong on
    # purpose). NOPUSH/PUSHONLY/NOWEIGHT each delete a different part of the push path, so each is its
    # own binary; leaving them out of the key is what made the earlier deletion A/B compare a build
    # with itself.
    comb_diag_suffix = "".join(
        f"_{n.lower()}"
        for n in ("TIMING", "NOREDUCE", "NOPUSH", "NOGATHER", "PUSHONLY", "NOWEIGHT", "NOROUTE", "SPREAD", "RUNRR", "RUNRRQ", "DUMPCNT", "BARRIER2", "BARNOFENCE", "BARFAN", "NOBAR")
        if os.environ.get(f"MORI_COMB_{n}", "").lower() in ("1", "true", "on", "yes")
    )
    # Deletion diagnostics (wrong results on purpose) and the meta shape histogram.
    diag_suffix = "".join(
        f"_{n.lower()}"
        for n in ("NOSTG", "NOMETA", "NOPAY", "METADIAG", "PAYRAW", "NOLOAD", "NOSEND", "PREBASE")
        if os.environ.get(f"MORI_DISP_{n}", "").lower() in ("1", "true", "on", "yes")
    )
    # -DMORI_DISP_PAY2D=D0 reshapes the payload TDM descriptor, so D0 is part of the key.
    _pay2d = os.environ.get("MORI_DISP_PAY2D", "").strip()
    pay2d_suffix = f"_pay2d{int(_pay2d)}" if _pay2d.isdigit() and int(_pay2d) > 1 else ""
    d = (
        get_cache_root()
        / f"{arch}_{nic}{ccqe_suffix}{profiler_suffix}{cov_suffix}{timing_suffix}{fastdedup_suffix}{combtdm_suffix}{barsleep_suffix}{barspread_suffix}{pipe_suffix}{comb_diag_suffix}{diag_suffix}{pay2d_suffix}{cntstep_suffix}"
        / content_hash
    )
    d.mkdir(parents=True, exist_ok=True)
    return d
