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
"""Core JIT compilation: hipcc invocation, bitcode linking, and process-safe locking."""

from __future__ import annotations

import functools
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from mori.jit.cache import get_cache_dir, get_cache_root
from mori.jit.config import (
    BuildConfig,
    detect_build_config,
    detect_nic_type,
    find_mpi_include,
    get_mori_source_root,
    is_debuginfo_enabled,
    is_profiler_enabled,
)

logger = logging.getLogger(__name__)

_BC_FILENAME = "libmori_shmem_device.bc"

_GLOBAL_GPU_STATES_SHIM = """\
#include "mori/shmem/internal.hpp"

namespace mori {
namespace shmem {
__device__ __attribute__((visibility("default"))) GpuStates globalGpuStates;
}
}
"""


class FileBaton:
    """File-based lock for multi-process build safety.

    When *wait_for* is provided, waiters that see the target file appear
    will return immediately **without** acquiring the lock, setting
    ``self.skipped = True``.  The caller should check this flag and skip
    the build if it is set.
    """

    def __init__(self, lock_path: str | Path, wait_for: str | Path | None = None):
        self._lock_path = str(lock_path)
        self._wait_for = str(wait_for) if wait_for else None
        self.skipped = False

    def __enter__(self):
        while True:
            try:
                fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return self
            except FileExistsError:
                if self._wait_for and os.path.isfile(self._wait_for):
                    self.skipped = True
                    return self
                time.sleep(0.5)

    def __exit__(self, *exc):
        if self.skipped:
            return
        try:
            os.remove(self._lock_path)
        except OSError:
            pass


def _hipcc_device_bc(
    cfg: BuildConfig,
    source: Path,
    include_dirs: list[Path],
    output: Path,
    *,
    cov: int = 5,
) -> None:
    """Compile a single source file to device-only bitcode."""
    cmd = [
        cfg.hipcc,
        "-c",
        "--cuda-device-only",
        "-emit-llvm",
        f"--offload-arch={cfg.arch}",
        "-fgpu-rdc",
        f"-mcode-object-version={cov}",
        "-std=c++17",
        "-O2",
        "-D__HIP_PLATFORM_AMD__",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
        *_nic_defines(),
        *_ccqe_defines(),
        *_profiler_defines(),
    ]
    for d in include_dirs:
        cmd.extend(["-I", str(d)])
    cmd.extend([str(source), "-o", str(output)])

    subprocess.check_call(cmd, stderr=subprocess.STDOUT)


def _llvm_link(cfg: BuildConfig, inputs: list[Path], output: Path) -> None:
    """Link multiple .bc files into one."""
    cmd = [cfg.llvm_link] + [str(p) for p in inputs] + ["-o", str(output)]
    subprocess.check_call(cmd, stderr=subprocess.STDOUT)


def _strip_lifetime_intrinsics(cfg: BuildConfig, bc_in: Path, bc_out: Path) -> None:
    """Remove llvm.lifetime intrinsics for Triton LLVM compatibility."""
    ll_path = bc_in.with_suffix(".ll")

    subprocess.check_call(
        [cfg.opt, "-S", str(bc_in), "-o", str(ll_path)],
        stderr=subprocess.STDOUT,
    )

    ll_text = ll_path.read_text()
    ll_text = re.sub(r"^.*llvm\.lifetime\..*$", "", ll_text, flags=re.MULTILINE)
    ll_path.write_text(ll_text)

    subprocess.check_call(
        [cfg.opt, str(ll_path), "-o", str(bc_out)],
        stderr=subprocess.STDOUT,
    )


def _verify_bitcode(cfg: BuildConfig, bc_path: Path) -> None:
    """Verify that globalGpuStates symbol exists in the bitcode."""
    result = subprocess.run(
        [cfg.opt, "-S", str(bc_path), "-o", "-"],
        capture_output=True,
        text=True,
    )
    if "_ZN4mori5shmem15globalGpuStatesE" not in result.stdout:
        raise RuntimeError(
            "JIT compilation succeeded but globalGpuStates symbol not found in bitcode. "
            "This is a bug in the JIT compiler."
        )


def _lib_has_ionic_ccqe() -> bool:
    """Check whether the ionic driver supports CCQE by probing the runtime library symbol."""
    import ctypes
    import ctypes.util

    lib_name = ctypes.util.find_library("ionic")
    if lib_name is None:
        return False
    try:
        lib = ctypes.CDLL(lib_name)
        return hasattr(lib, "ionic_dv_create_cq_ex")
    except OSError:
        return False


_CCQE_MIN_FW_VERSION = (1, 117, 5, 58)


def _parse_ionic_fw_version(fw_ver: str) -> tuple[int, ...] | None:
    """Parse '1.117.5-a-58' → (1, 117, 5, 58). Returns None if unparseable."""
    if not fw_ver:
        return None
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)-a-?(\d+)$", fw_ver)
    if not m:
        return None
    return tuple(int(x) for x in m.groups())


def _is_firmware_support_ccqe(fw_ver: str) -> bool:
    """Return True if the firmware version >= 1.117.5-a-58."""
    ver = _parse_ionic_fw_version(fw_ver)
    return ver is not None and ver >= _CCQE_MIN_FW_VERSION


def _get_ionic_fw_versions() -> list[str]:
    """Return fw_ver strings for every ionic IB device found in sysfs."""
    ib_dir = "/sys/class/infiniband"
    versions: list[str] = []
    try:
        for dev in os.listdir(ib_dir):
            dev_path = os.path.join(ib_dir, dev)
            driver_link = os.path.join(dev_path, "device", "driver")
            try:
                driver_name = os.path.basename(os.readlink(driver_link))
            except OSError:
                continue
            if driver_name not in ("ionic_rdma", "ionic"):
                continue
            fw_path = os.path.join(dev_path, "fw_ver")
            try:
                fw_ver = Path(fw_path).read_text().strip()
                versions.append(fw_ver)
            except OSError:
                pass
    except OSError:
        pass
    return versions


def _is_all_ionic_support_ccqe() -> bool:
    """Return True only when every ionic device has the same fw version and that version >= 58."""
    versions = _get_ionic_fw_versions()
    if not versions:
        return False
    if len(set(versions)) != 1:
        return False

    logger.debug("ionic ver: %s", versions[-1])

    for ver in versions:
        if not _is_firmware_support_ccqe(ver):
            return False

    return True


@functools.cache
def is_ccqe_enabled() -> bool:
    """Return True if CCQE should be enabled (cached after first call)."""
    if os.environ.get("MORI_DISABLE_IONIC_CCQE", "").lower() in (
        "1",
        "true",
        "on",
        "yes",
    ):
        logger.info("Ionic _ccqe_enabled: False (disabled by MORI_DISABLE_IONIC_CCQE)")
        return False
    lib_support = _lib_has_ionic_ccqe()
    nic_support = _is_all_ionic_support_ccqe()
    enabled = lib_support and nic_support
    logger.info(
        "Ionic _ccqe_enabled: %s lib_support %s nic_support: %s",
        enabled,
        lib_support,
        nic_support,
    )
    return enabled


def _ccqe_defines() -> list[str]:
    return ["-DIONIC_CCQE"] if is_ccqe_enabled() else []


def _nic_defines() -> list[str]:
    """Return compiler -D flags for the detected NIC type (device-side macros)."""
    nic = detect_nic_type()
    if nic == "bnxt":
        return ["-DMORI_DEVICE_NIC_BNXT"]
    elif nic == "ionic":
        return ["-DMORI_DEVICE_NIC_IONIC"]
    return []


def _profiler_defines() -> list[str]:
    """Return -DENABLE_PROFILER if mori was built with profiler support."""
    return ["-DENABLE_PROFILER"] if is_profiler_enabled() else []


# Prefix rather than an enumeration: this tracks the kernel-side #if, which tests the gfx125x
# arch macros, and a new member of that family should not need a second edit here to be recognised.
_FASTPATH_ARCH_PREFIX = "gfx125"
_arch_cache: str | None = None
_fastpath_cache: bool | None = None


def _target_arch() -> str:
    """The arch the JIT will compile for, or "" when it cannot be determined.

    detect_gpu_arch() rather than get_build_config(): the latter also insists on hipcc and the LLVM
    tools, and this is called from the LAUNCH path (the LDS budget in ops/dispatch_combine.py) where
    a missing compiler must not raise. Any failure degrades to "", i.e. no fast path, never a crash.
    """
    global _arch_cache
    if _arch_cache is None:
        try:
            from .config import detect_gpu_arch

            _arch_cache = detect_gpu_arch(os.environ.get("ROCM_PATH", "/opt/rocm"))
        except Exception:
            _arch_cache = ""
    return _arch_cache


def _comb_fastpath() -> bool:
    """Whether combine's TDM/QUAD transport is on when the caller has set no gates.

    This is the single place that decides it. Every MORI_COMB_* reader below falls back to it, the
    cache key in cache.py derives from the same readers, and the LDS budget in
    ops/dispatch_combine.py calls them too -- so the emitted -D flags, the .hsaco identity and the
    shared-memory reservation cannot drift apart.

    The gates it turns on are the configuration measured at 64x8 ZC=1 on gfx1250. Checked (rc=0 on
    the bench's per-element combine check) with NOTHING in the environment: 168.6us / 1201 GB/s for
    202.47 MB, matching the 168.5us the same gates reach when set by hand. The same run with
    MORI_COMB_FASTPATH=0 is 2311.5us / 88 GB/s -- the vector-load gather this replaces, at a
    geometry it was never tuned for -- and is also rc=0, so the fallback stays correct.

    They are only defaults: an explicit env value always wins, including an explicit 0, and
    MORI_COMB_FASTPATH=0 turns the whole set off in one move.

    Restricted to gfx125x because that is where the TDM engine exists at all -- the kernel-side #if
    tests the same arch family, so enabling these elsewhere would emit -D flags that compile to
    nothing while still forking the cache key.
    """
    global _fastpath_cache
    if _fastpath_cache is None:
        val = os.environ.get("MORI_COMB_FASTPATH", "").strip().lower()
        if val in ("0", "false", "off", "no"):
            _fastpath_cache = False
        elif val in ("1", "true", "on", "yes"):
            _fastpath_cache = True
        else:
            _fastpath_cache = _target_arch().startswith(_FASTPATH_ARCH_PREFIX)
    return _fastpath_cache


def _comb_env_set(name: str) -> bool:
    """Whether the caller named this gate at all, as opposed to inheriting the arch default.

    The LDS budget needs the distinction: a geometry too wide for a gate the ARCH turned on should
    quietly fall back to a transport that fits, but the same geometry against a gate the CALLER
    asked for has to say so instead of silently ignoring them.
    """
    return os.environ.get(name, "").strip() != ""


def _comb_flag(name: str, default: bool) -> bool:
    """An on/off MORI_COMB_* gate, where "unset" is distinct from "explicitly off"."""
    val = os.environ.get(name, "").strip().lower()
    if val == "":
        return default
    return val in ("1", "true", "on", "yes")


# On/off combine gates that ride with the QUAD fold, and the ones that never default on.
# cache.py walks the same two lists so the .hsaco name and the -D flags cannot disagree.
_COMB_QUAD_DEFAULT_FLAGS = ("QBAR", "QU4", "QCVT")
_COMB_OPT_IN_FLAGS = (
    "QNOXFER",
    "QFLAG",
    "QST16",
    "QTLATE",
    "QNOOP",
    "QNOSYNC",
    "QNOPF",
)


def _comb_tdm_defines() -> list[str]:
    """-DMORI_COMB_TDM=N sends the COMBINE token push through the gfx1250 TDM engine instead of
    per-lane cross-card WarpCopy, reusing the shape the dispatch payload phase already proved (one
    TDM load into a per-warp LDS tile, one TDM store into the peer's slot).

    N is the number of chunks a token is split into, because the tile has to fit LDS: combine's best
    geometry is warp_per_block=32, where 32 full 14KB tiles (hidden 7168 bf16) would want 458KB
    against gfx1250's 320KB budget. N=2 gives 7KB/warp -> 229KB and is the default when the env value
    is not a number. _combine_shared_mem() in ops/dispatch_combine.py must size the tile with the
    same formula, so keep the two in sync.

    Both combine transports are wired to this one gate so they can be compared later:
      * `_nop2p` (UseP2PRead=false, use_external_inp_buf=True i.e. --zero-copy 0) -> PUSH: one TDM
        load into a per-warp tile, one TDM store into the peer's slot. One tile per warp.
      * `_p2p` (UseP2PRead=true, --zero-copy 1) -> PULL: the gather reads become topk TDM loads from
        the peers into one tile per source, all issued before a single wait, then fp32 accumulate out
        of LDS. Needs topk tiles per warp, so it wants a much larger N than the push path.
    PUSH cannot run in the zero-copy layout (there the combine input buffer IS the peer's staging
    buffer, so pushing would clobber the peer's own input), which is why the transport follows the
    same flag that picks the kernel rather than being independently selectable. Defaults to 2 where
    _comb_fastpath() says so, off everywhere else."""
    return [f"-DMORI_COMB_TDM={_comb_tdm_chunks()}"] if _comb_tdm_chunks() else []


def _comb_tdm_chunks() -> int:
    """MORI_COMB_TDM's resolved chunk count, 0 when the TDM transport is off."""
    val = os.environ.get("MORI_COMB_TDM", "").strip().lower()
    if val == "":
        return 2 if _comb_fastpath() else 0
    if val in ("0", "false", "off", "no"):
        return 0
    return int(val) if val.isdigit() and int(val) > 0 else 2


def _comb_barsleep_defines() -> list[str]:
    """-DMORI_COMB_BARSLEEP=N sets the backoff between two polls of the combine barrier's
    cross-device flag, in s_sleep units of ~64 clocks (default 1).

    The barrier costs 69.6us at 128 blocks and 15.0us when only block 0 polls and fences, so 54.6us
    of it is per-block work on flags that live in hipDeviceMallocUncached memory -- they must, or a
    peer's write would never be seen -- which makes every poll from every block a real fabric
    transaction aimed at the same 32 bytes. Backing off trades a bounded amount of exit latency for
    a proportional cut in that traffic, and unlike moving the poll it cannot change what any block
    observes. Measured, it is worth 15.2us and no more: 1 -> 8 -> 32 -> 127 gives barrier 69.3, 65.8,
    61.4, 58.5 and full combine 251.6, 246.0, 240.3, 236.1. The other ~44us does not respond to poll
    RATE at all, so it is one fixed cost per block -- the acquire itself, or the 128 uncached reads
    that all arrive the instant the flag flips and serialise on one line. MORI_COMB_BARNOFENCE is
    the gate that tells those two apart.

    Those numbers are all PUSH, where the wait is 58.5us and long enough to hide the oversleep. The
    PULL/QUAD wait is 7.6us, so the same 127 overshoots it and turns into a net loss: at 64x8 ZC=1
    bf16 EP4 with the check armed, RUNRR alone reads 168.9us / 1199 GB/s against RUNRR+127 at
    171.1 / 1183. Anything that pins this value instead of taking the default therefore reports
    every PULL geometry ~2us slow, which is exactly how tools/_ct_nobar.sh manufactured a phantom
    2% regression against a baseline that had been recorded at the default."""
    val = os.environ.get("MORI_COMB_BARSLEEP", "").strip()
    if val == "":
        # 15 was the bottom of the sweep at 64x8: 1 thrashes the flag line (+1.2us), the old 127
        # oversleeps. Only worth setting where the barrier is being paid for at all.
        return ["-DMORI_COMB_BARSLEEP=15"] if _comb_fastpath() else []
    if not val.isdigit() or int(val) <= 0:
        return []
    return [f"-DMORI_COMB_BARSLEEP={int(val)}"]


def _comb_barspread_defines() -> list[str]:
    """-DMORI_COMB_BARSPREAD=N makes block 0 poll the cross-device flags alone and republish the
    epoch into one line PER BLOCK, N uint32 words apart (32 = 128B). Correctness-preserving,
    default OFF.

    Priced by MORI_COMB_NOBAR, which is the only honest measurement of the wait (full minus the
    wait, in a complete kernel): 6.9us at 32 blocks, 16.8 at 64, 44.4 at 128, 150.4 at 256. Growing
    faster than the block count is what says the cost is contention on the single flag line rather
    than the unavoidable wait for the slowest peer, which the 32-block figure bounds at <= 6.9us.
    BARFAN already showed that moving that line into cache changes nothing (58.5 -> 110.8), so the
    variable left is how many blocks read the SAME line, which is what N spreads apart."""
    val = os.environ.get("MORI_COMB_BARSPREAD", "").strip()
    if val == "":
        return ["-DMORI_COMB_BARSPREAD=16"] if _comb_fastpath() else []
    if not val.isdigit() or int(val) <= 0:
        return []
    return [f"-DMORI_COMB_BARSPREAD={int(val)}"]


def _comb_quad_depth() -> int:
    """Tile buffers per warp for the QUAD gather, 0 when it is off. The legacy spelling
    MORI_COMB_QUAD=1 means the original double buffer, so it reads as 2. The LDS budget in
    ops/dispatch_combine.py and the cache key both have to agree with this."""
    val = os.environ.get("MORI_COMB_QUAD", "").strip().lower()
    if val == "":
        # Follows the resolved TDM chunk count rather than _comb_fastpath() directly: with
        # MORI_COMB_TDM=0 the kernel's #if compiles the whole QUAD body out, and a QUAD depth that
        # only the host believes in would have it reserve tiles nothing ever writes.
        return 2 if (_comb_fastpath() and _comb_tdm_chunks()) else 0
    if val.isdigit():
        n = int(val)
        return 2 if n == 1 else (n if n >= 2 else 0)
    return 2 if val in ("true", "on", "yes") else 0


def _comb_quad_split() -> int:
    """Parts each QUAD tile is cut into. 1 is the whole-token read."""
    val = os.environ.get("MORI_COMB_QSPLIT", "").strip()
    return int(val) if val.isdigit() and int(val) >= 1 else 1


def _comb_qtst() -> int:
    """How the QUAD fold's output leaves LDS: 0 = the warp's own vector stores, 1 = one TDM store
    per warp of its own slice, 2 = one whole-token TDM store per group, 3 = one TDM store per BLOCK
    covering its groups' consecutive tokens. The LDS budget in ops/dispatch_combine.py and the
    cache key both have to agree with this."""
    val = os.environ.get("MORI_COMB_QTST", "").strip().lower()
    if val == "":
        # 2 (one whole-token store per group) is the shape that was measured fastest; it only exists
        # inside the QUAD fold, so it defaults on exactly when QUAD does.
        return 2 if _comb_quad_depth() else 0
    if val.isdigit():
        n = int(val)
        return n if n in (1, 2, 3) else 0
    return 1 if val in ("true", "on", "yes") else 0


def _comb_qloc() -> int:
    """Read this rank's own copy of a token with vector loads during the fold instead of pulling it
    over the TDM engine: 1 = load it inside the fold, 2 = stage it in registers before the barrier.
    Costs two more ints per (warp, buffer) of LDS for the pointer ring."""
    val = os.environ.get("MORI_COMB_QLOC", "").strip().lower()
    if val.isdigit():
        n = int(val)
        return n if n in (1, 2, 3) else 0
    return 1 if val in ("true", "on", "yes") else 0


def _comb_qob() -> int:
    """Slots in the QUAD fold's output ring, 0 when it should follow the tile ring's depth."""
    val = os.environ.get("MORI_COMB_QOB", "").strip()
    return int(val) if val.isdigit() and int(val) >= 2 else 0


def _comb_pipe_defines() -> list[str]:
    """-DMORI_COMB_PIPE=1 double-buffers the PULL gather: chunk k+1's peer reads are issued before
    chunk k is folded, so the fabric is not idle for the fp32 add and the output store.
    Correctness-preserving, default OFF, and it doubles the PULL tile budget in
    _combine_shared_mem().

    What it attacks, measured at 64x8 ZC=1 MORI_COMB_TDM=2 (noTIMING, deletion pricing):
    combine 255.7us, of which 27.0 is barrier+launch, leaving 228.7us of gather+fold for 202.47MB
    of peer reads = 885 GB/s against a 1.40 TB/s P2P read ceiling. The unpipelined loop has nothing
    in flight for the whole fold half, which is the only structural reason for a gap that size."""
    val = os.environ.get("MORI_COMB_PIPE", "").strip()
    out = []
    if val.isdigit() and int(val) >= 2:
        out.append(f"-DMORI_COMB_PIPE={int(val)}")
    elif val.lower() in ("1", "true", "on", "yes"):
        out.append("-DMORI_COMB_PIPE=2")
    # MORI_COMB_QUAD=N: one warp per SOURCE, whole-token peer reads, group of worldSize warps folds
    # one token cooperatively, with N tile buffers per warp (1 and 2 both mean 2).
    # Correctness-preserving. It exists because the peer-read ceiling at grid 64 is set by the TDM
    # read SIZE (tools/_ct_epsim.sh mode9: 4864 B -> 801 GB/s, 14336 B -> 1395 GB/s) and the chunked
    # gather cannot afford whole-token reads in LDS. MORI_COMB_QSPLIT=S cuts each tile into S parts,
    # which is how depth beyond 2 is paid for: S halves the tile so N can double for the same LDS.
    _quad = _comb_quad_depth()
    if _quad >= 2:
        out.append(f"-DMORI_COMB_QUAD={_quad}")
        out.append(f"-DMORI_COMB_QSPLIT={_comb_quad_split()}")
    # QBAR/QU4/QCVT are the three correctness-preserving wins that only exist inside the QUAD fold,
    # so they default on with it. The rest are diagnostics and stay opt-in.
    for _g in _COMB_QUAD_DEFAULT_FLAGS:
        if _comb_flag(f"MORI_COMB_{_g}", bool(_quad >= 2)):
            out.append(f"-DMORI_COMB_{_g}=1")
    for _g in _COMB_OPT_IN_FLAGS:
        if _comb_flag(f"MORI_COMB_{_g}", False):
            out.append(f"-DMORI_COMB_{_g}=1")
    if _comb_qloc():
        out.append(f"-DMORI_COMB_QLOC={_comb_qloc()}")
    # GROUP0's temporal hint / scope trait on combine's QUAD peer read and output store.
    for _g in ("THLD", "THST", "SCLD", "SCST"):
        _v = os.environ.get(f"MORI_COMB_{_g}", "").strip()
        if _v.isdigit() and int(_v) > 0:
            out.append(f"-DMORI_COMB_{_g}={int(_v)}")
    # QTST has two shapes, not just on/off: 1 = one store per warp of its own slice, 2 = one
    # whole-token store per group. See the note at _qTG in intranode.hpp.
    _qtst = _comb_qtst()
    if _qtst:
        out.append(f"-DMORI_COMB_QTST={_qtst}")
        if _comb_qob():
            out.append(f"-DMORI_COMB_QOB={_comb_qob()}")
    _qred = os.environ.get("MORI_COMB_QRED", "").strip()
    if _qred.isdigit():
        out.append(f"-DMORI_COMB_QRED={int(_qred)}")
    # -DMORI_COMB_LB=N is the combine kernel's __launch_bounds__. Must be >= the launched block
    # size or the launch fails; set it to combine_warp_per_block * 32, since gfx1250 is wave32.
    # PAIR IT WITH MORI_COMB_WPEU or it does nothing measurable: the one-argument __launch_bounds__
    # only sets amdgpu_flat_work_group_size, and the VGPR budget follows amdgpu_waves_per_eu, which
    # the second argument carries. LB=256 alone gave a byte-identical code object (128 VGPRs, 29
    # spills, 192 B scratch) and 314.8us against 314.6. See the macro in intranode.hpp.
    _lb = os.environ.get("MORI_COMB_LB", "").strip()
    if _lb.isdigit() and int(_lb) > 0:
        out.append(f"-DMORI_COMB_LB={int(_lb)}")
    _wpeu = os.environ.get("MORI_COMB_WPEU", "").strip()
    if _wpeu.isdigit() and int(_wpeu) > 0:
        out.append(f"-DMORI_COMB_WPEU={int(_wpeu)}")
    out.append(f"-DMORI_COMB_QSTGU={_comb_qstgu()}")
    out.append(f"-DMORI_COMB_QWIDE={_comb_qwide()}")
    out.append(f"-DMORI_COMB_RELFENCE={_comb_relfence()}")
    out.append(f"-DMORI_COMB_QSCW={_comb_qscw()}")
    out.append(f"-DMORI_COMB_SCPRE={_comb_scpre()}")
    return out


def _comb_scpre() -> int:
    """Prefetch a blockwise source's whole scale row into registers once per token, instead of
    reading one scale per source per vector out of the peer's uncached allocation.

    Correctness-preserving; default ON. 0 exists so the prefetch can be PRICED in one binary-honest
    A/B rather than across two commits, which is the only kind of before/after this tree accepts --
    and the closest thing available before it existed was MORI_COMB_QNOSC, which deletes the scales
    entirely and is wrong by construction, so it could only bracket the answer.
    """
    val = os.environ.get("MORI_COMB_SCPRE", "").strip().lower()
    if val in ("0", "false", "off", "no"):
        return 0
    return 1


def _comb_qscw() -> int:
    """How the blockwise quantise pass writes the per-block scales.

    The scales are 56 floats per token against 7168 bytes of fp8, so 3% of the bytes -- but not 3%
    of the stores. In the exact-fit path one subwarp owns one scale block, so the scale write is
    one 4-byte store from one lane in each of 16, i.e. 8 live bytes per store instruction, and
    there are as many of those instructions per token as there are 256-byte fp8 stores. dstScales
    is hipDeviceMallocUncached, where a store instruction costs a transaction whether it carries 8
    bytes or 256.

        0  one store per subwarp per block. What the pass has always done.
        1  gather each group of scales into consecutive lanes with __shfl and store the group in
           one instruction. Same bytes to the same addresses, and no LDS: the shuffle pattern is
           fixed per unrolled step, so the register indices stay static.
        2  DIAGNOSTIC, WRONG RESULTS: do not write the scales at all. This is the upper bound on
           what 1 could ever be worth, and it is the row to read first -- if 2 buys nothing then 1
           cannot either, and the idea is dead without building it.
    """
    val = os.environ.get("MORI_COMB_QSCW", "").strip()
    return int(val) if val in ("0", "1", "2") else 0


def _comb_relfence() -> int:
    """Whether each block fences to system scope before combine's cross-device barrier when it has
    just staged a caller-owned buffer that peers will read. 1 = yes (default), 0 = the old behaviour.

    This is a CORRECTNESS default, not a tuning one. The barrier's release side fences on block 0's
    first worldSize threads only, so another block's stores can still be in flight when the peer flag
    goes up. MEASURED at 64x8 EP4 bf16 ZC=0 MORI_COMB_PULL=kernel with the check armed: without the
    fence 3 of 4 ranks are wrong, with it rc=0. 0 is kept only so the failure can be reproduced.

    Note what hid this: the gate used to be a plain on/off diagnostic name, and the build cache key
    did not include it, so the "with fence" run reused the "without fence" binary and reported rc=1.
    That reads as "the fence does not fix it" and sent the whole question down the wrong path.
    """
    val = os.environ.get("MORI_COMB_RELFENCE", "").strip().lower()
    if val in ("0", "false", "off", "no"):
        return 0
    return 1


def _comb_qwide() -> int:
    """Whether combine's PULL gather describes its peer reads in the widest element type the byte
    count allows: 0 = off, 1 = only for 1-byte tokens (fp8/fp4 blockwise), 2 = for every token type.

    The descriptor's dataSize is not free even though it carries nothing a contiguous copy needs.
    MEASURED at 64x8 EP4 on the chunked PULL gather with MORI_COMB_NOQUANT holding the quantise pass
    out, same code and the same 3584 elements per descriptor in both rows: bf16 at dataSize 1 moves
    212 MB in 247.7us (857 GB/s), fp8 at dataSize 0 moves 106 MB in 493.2us (215 GB/s). See
    TdmShapeWide in intranode.hpp.

    MEASURED NULL, WHICH IS WHY IT DEFAULTS OFF. The theory above was that a dataSize of 0 is what
    makes the fp8 gather slow, since bf16 moves twice the bytes through the same code in 247.7us
    while fp8 takes 493.2. Describing the identical run in 4-byte elements reads 630.0us against
    631.1 for the whole combine, and 1349.4 against 1348.8 with the quantise pass live -- nothing.
    The gate is kept, at 0, so the next person does not have to rebuild it to re-ask.
    """
    val = os.environ.get("MORI_COMB_QWIDE", "").strip()
    return int(val) if val.isdigit() and int(val) in (0, 1, 2) else 0


def _comb_qstgu() -> int:
    """Scale blocks the blockwise quantise/stage pass keeps in flight per subwarp.

    This is the pass that turns the caller's bf16 tensor into the fp8 + scales the peers will pull,
    and it runs ONLY under blockwise quant with a caller-owned input buffer -- bf16 zero-copy has no
    analogue of it. As written it was one dependent load per scale block, and MEASURED at 64x8 EP4
    that is 778us of a 1409.5us combine (full 1409.5, MORI_COMB_NOQUANT 631.1), against 15.8us for
    the launch and the cross-device barrier together. See WarpQuantizeBf16ToFp8BlockwiseVec.

    1 is the old behaviour. Correctness-preserving at every value: only the order of the loads
    changes, not which bytes are read, reduced or stored. MEASURED at 64x8 EP4, check armed, with
    the gather (630.0us) subtracted off to leave this pass alone:
        QSTGU 1  718.8us      QSTGU 4  444.9us      QSTGU 7  410.5us
    7 is where it flattens, and 56 scale blocks over 2 subwarps makes 7 an exact fit with no tail.
    """
    val = os.environ.get("MORI_COMB_QSTGU", "").strip()
    return int(val) if val.isdigit() and int(val) >= 1 else 7


def _comb_diag_defines() -> list[str]:
    """Two combine diagnostics, both compile gates, both default OFF.

    MORI_COMB_TIMING adds the [CSPLIT] print splitting the combine token loop into per-token routing,
    TDM load issue, the wait on those loads, and the fp32 fold out of LDS. Combine has only ever been
    one number, so there was no way to tell whether a faster transport could still help it. The
    kernel skips args.replayMode launches, so this prints from an eager run -- read it off
    tools/ep4_acc_check.py, not the CUDA-graph bench, which only ever replays.

    MORI_COMB_NOREDUCE gives WRONG RESULTS ON PURPOSE, the same family as MORI_DISP_NOMETA/NOPAY: the
    TDM pull path still issues and waits on every peer load, so the cross-card traffic is byte-for-byte
    unchanged, but the lanes fold one tile instead of topk. The bandwidth gap against a full build is
    therefore the fold alone, i.e. what combine would reach if the reduction were free. Never gate
    correctness on it.

    MORI_COMB_NOQUANT deletes the PULL side's local stage-and-quantise pass (intranode.hpp:2234),
    the loop that reads the caller's bf16 and writes fp8 + scales into this rank's own combineInp
    before the peers read it. Transport and fold are untouched, so full - NOQUANT is that pass on
    its own. It is needed because this phase has NO counterpart in the bf16 PULL reference: bf16
    runs zero-copy, where useExternalInpBuffer is false and the loop does not execute at all, so
    the 168.4us baseline cannot bound it. WRONG RESULTS, needs MORI_BENCH_SKIPCHECK.

    The three PUSH-side gates below were read by the kernel but never passed here, so setting them did
    nothing AND did not change the cache key -- an A/B against them silently compared a binary with
    itself and read "this phase is free". They are the combine counterparts of MORI_DISP_NOPAY:
    NOPUSH zeroes the push loop's trip count, leaving geometry, LDS and the whole gather side
    byte-identical, so kernel(full) - kernel(NOPUSH) is the send's unbiased marginal cost; NOGATHER
    is its mirror on the reduce loop, and the pair is the only way to price the barrier that sits
    between them (NOPUSH+NOGATHER leaves the barrier and the launch and nothing else); PUSHONLY
    returns right after the push and the barrier, so it prices everything downstream of them;
    NOWEIGHT drops only the per-token 32B cross-card weight write inside the send loop. NOROUTE
    replaces the send loop's localSrcMap lookup with arithmetic, which also moves where the token
    lands. All of them give wrong results and need MORI_BENCH_SKIPCHECK.

    NOWEIGHT only prices anything when weights are actually passed: the bench calls combine() with
    weights=None, which leaves args.weightsBuf null and the guarded copy unreachable, so on that
    workload it deletes dead code and reads as free.

    NOROUTE is worth 101.1us at 64x8 PUSH/TDM but prices the DESTINATION, not the lookup: two
    correctness-preserving attacks on the lookup itself (batching it per warp, hoisting the peer
    base pointers) moved 3.0us between them and were deleted -- see the push loop's comment.

    MORI_COMB_BARNOFENCE and MORI_COMB_BARFAN are the two barrier gates. NOFENCE is a pricing gate
    and WRONG BY CONSTRUCTION: every block still polls the cross-device flags, only the per-block
    system-scope acquire is dropped off blocks other than 0. It says the fence is free (58.6 -> 58.3)
    and, run without MORI_BENCH_SKIPCHECK, that it is nonetheless load-bearing (rc=1 against the same
    build's 236.9 with it). Together with the backoff sweep that pins the barrier's 54.6us per-block
    cost on the uncached reads themselves, which no backoff can reach. BARFAN has block 0 poll alone
    and fan the release out through cached memory while every block keeps its own fence; it needs
    worldSize >= 2, since it parks the epoch in combineGridBarrier[1]. It is CORRECT but SLOWER
    (barrier 58.5 -> 110.8, full 236.9 -> 309.2) and stays off: swapping 128 uncached reads of one
    line for 127 device-scope reads of one line buys nothing and serialises them behind block 0.

    MORI_COMB_NOWAIT deletes the s_wait_tensorcnt the PUSH fold does on its own TDM load, so
    full - NOWAIT is the time a warp spends stalled waiting for its tile and nothing else. It exists
    because double buffering is gated on UseP2PRead: on the PUSH path the fold aliases the send tile,
    so _cPullBufs is 1 and every token is a serial issue -> wait -> fold with no overlap, which is
    the leading suspect for the ~60us by which this fold exceeds tdm_redsim's. WRONG RESULTS -- the
    lanes fold a half-written tile -- but in bounds, since the LDS and the addresses are unchanged.

    MORI_COMB_NOBAR is the barrier's honest price: it deletes the cross-device WAIT from an
    otherwise complete kernel, keeping the arrival count, the flag stores and the flag increment,
    which the next replay needs. Every earlier barrier number came from the opposite deletion,
    NOPUSH+PUSHONLY, which prices a barrier that no longer resembles the real one -- nothing
    staggers the blocks' arrival, no peer is still pushing, and the launch cannot be separated out.

    Two entries are CORRECTNESS-PRESERVING and are here only because they need the same -D and
    cache-key plumbing:

      MORI_COMB_FOLDVEC  routes PUSH's fold through the 16B lane-load gather instead of the TDM tile
                         path. A SETTLED NEGATIVE -- dropping the tile costs 596.1us against 311.5.
      MORI_COMB_FOLDU    reads every source into registers before accumulating any of them, so the
                         ds_read_b128s issue back to back. Worth 314.6 -> 309.3us at 64x8, i.e.
                         1.8% -- real and repeatable, but a third of what the microbenchmark
                         predicted.
      MORI_COMB_FOLDB    implies FOLDU and drops the per-source skip from the READ loop, which is
                         why FOLDU alone bought so little: each skip puts its ds_load in its own
                         exec-masked block and the compiler opens every one with
                         s_wait_loadcnt_dscnt 0x0, so the four loads stay serialised however the
                         source orders them. A dead slot reads row 0 and is discarded in the
                         accumulate. Worth 314.7 -> 287.9us with the check armed, and 160.6 ->
                         133.6 on the fold measured on its own. All three are documented where
                         they act, at _cPullOk and _cRedSrcMax in intranode.hpp.

    Every other flag in this list is a DELETION or a diagnostic, which is what makes "off" the right
    default for all of them. The push loop's token ORDER used to be selected from here too --
    MORI_COMB_SPREAD (prime-step bijection), MORI_COMB_RUNRR (bucket by peer, flattened take) and
    MORI_COMB_RUNRRQ (bucket by peer, queued take) -- and that was a category error: reordering is
    not a deletion, all three produce correct results, and which one is fastest is decided by the
    peer count and the warp count, both of which the kernel knows at launch. The queued bucket order
    won at every measured point (64x8 bf16 PUSH, check armed: EP4 319.4us against 417.6 unordered
    and 325.8 flattened, EP2 196.6 against 203.4 and 216.2), so it is now the only push order and
    the three flags are gone."""
    return [
        f"-D{name}"
        for name in (
            "MORI_COMB_TIMING",
            "MORI_COMB_NOREDUCE",
            "MORI_COMB_NOQUANT",
            "MORI_COMB_NOPUSH",
            "MORI_COMB_NOGATHER",
            "MORI_COMB_NOWAIT",
            "MORI_COMB_BARRIER2",
            "MORI_COMB_BARNOFENCE",
            "MORI_COMB_BARFAN",
            "MORI_COMB_NOBAR",
            "MORI_COMB_PUSHONLY",
            "MORI_COMB_NOWEIGHT",
            "MORI_COMB_NOROUTE",
            "MORI_COMB_FOLDB",
            "MORI_COMB_FOLDVEC",
            "MORI_COMB_FOLDU",
            "MORI_COMB_DUMPCNT",
            "MORI_COMB_QNOSC",
        )
        if os.environ.get(name, "").strip().lower() in ("1", "true", "on", "yes")
    ]


def _disp_metadiag_defines() -> list[str]:
    """Diagnostic: -DMORI_DISP_METADIAG prints a [METASHAPE] histogram of the metadata idx run's
    length (cc), tile kind and 128B start phase. Not folded into MORI_DISP_TIMING because these are
    data-flow values -- identical in the shipping untimed build -- and reading them without the
    clock64 probes avoids the distortion those probes were measured to add to the meta phase.
    Gated by MORI_DISP_METADIAG env; default OFF."""
    return (
        ["-DMORI_DISP_METADIAG"]
        if os.environ.get("MORI_DISP_METADIAG", "").lower() in ("1", "true", "on", "yes")
        else []
    )


# Emitters for gates whose kernel bodies were removed from src/ops/dispatch_combine/intranode.hpp are
# deleted along with them: a -D that no #if reads is silently accepted and does nothing, and it is not
# in cache.py's key either, so setting one would quietly land on another config's cached .hsaco.
#
# What each of them measured is recorded next to the code it would have replaced -- the ALREADY
# REJECTED table in intranode.hpp (METAFUSE, METAVEC, METALDS, GRIDFLAG, PAYSPLIT, METAFIELD,
# METASPLIT, SRCVEC, PAYBUF), with the full reasoning in tools/HANDOFF-F01-2.md §8. Deliberately not
# repeated here; two copies of the same numbers drift.
#
# Three that table does not mention, because they were diagnostics rather than candidates:
#   DBLCOUNT / DBLRESERVE  isolate a phase's cost by DOUBLING it instead of deleting it, so ACC stays
#                          PASS and the measurement is known to have run on a working kernel. This is
#                          how gather = 5.45us and meta = 6.95us were derived.
#   COMPL_BACKOFF          s_sleep(N) in the completion spin instead of the tight spin.
#   PAYDYN                 warps claim payload work on demand (LDS atomic) rather than statically.
#                          Built and run with ACC=1, but no number for it survives anywhere in the
#                          repo, so it is the one whose result is unknown rather than negative.



def _disp_nophase_defines() -> list[str]:
    """DIAGNOSTIC, WRONG RESULTS ON PURPOSE. -DMORI_DISP_NOMETA / -DMORI_DISP_NOPAY /
    -DMORI_DISP_NOSTG compile away the meta send / the payload send / FINALIZE's staging gather while
    leaving launch geometry, LDS reservation and occupancy alone, so kernel(full) - kernel(NOX) gives
    phase X's real cost. This exists because MORI_DISP_TIMING's clock64() probes sit inside the
    per-token loops and inflate the phases they measure (the timed build reports ~87us of non-payload
    against a ~33us noTIMING budget), which made the timed split useless for deciding where the last
    3us to 1.3TB/s should come from. Against a full 166.0us kernel: NOSTG 160.55us and NOMETA
    159.05us, i.e. the gather is 5.45us and the meta phase 6.95us for 2.9MB.

    NOSTG was originally only correct alongside METAFUSE, which took over feeding the peer's meta
    buffers once staging was gone. METAFUSE measured 462.6 GB/s and its body has been removed, so
    NOSTG now leaves nothing feeding them and is diagnostic-only like the other two.

    Three more split the payload loop itself, to find why it reaches 1192 GB/s where the pure-TDM
    a2a probe reaches 1664 at the same tile size, grid and warp count. PAYRAW keeps the TDM traffic
    and drops only the routing (map read, shfl, slot arithmetic); NOLOAD keeps the stores and drops
    the load; NOSEND keeps the load and drops the stores.

    Never enable with ACC=1; the dispatch output is deliberately incomplete.
    """
    out: list[str] = []
    for name in ("NOMETA", "NOPAY", "NOSTG", "PAYRAW", "NOLOAD", "NOSEND", "PREBASE"):
        if os.environ.get(f"MORI_DISP_{name}", "").lower() in ("1", "true", "on", "yes"):
            out.append(f"-DMORI_DISP_{name}")
    out.extend(_disp_pay2d_defines())
    return out


def _disp_pay2d_defines() -> list[str]:
    """-DMORI_DISP_PAY2D=D0 reshapes the payload's TDM descriptor from the 1 x hiddenDim wedge to a
    D0 x (hiddenDim/D0) 2D tile. gfx1250 wants both tensor dims >= 2 (TdmShape2D in intranode.hpp
    records this), and the payload descriptor was the only one still sending tensorDim1 == 1 while
    the meta path and the pure-TDM a2a probe both send 2D. D0 is the fast dim in ELEMENTS and must
    divide hiddenDim; the kernel falls back to the 1D shape when it does not. The 128B minimum row
    means D0 >= 64 for bf16.

    MEASURED NULL at 64x8 PUSH/TDM + SPREAD, hiddenDim 7168 -- D0=128 and D0=256 are within noise of
    the 1xN wedge on both phases and D0=64 is worse. Descriptor shape is not what limits either
    payload phase; see TdmShape in intranode.hpp for the numbers."""
    v = os.environ.get("MORI_DISP_PAY2D", "").strip()
    return [f"-DMORI_DISP_PAY2D={int(v)}"] if v.isdigit() and int(v) > 1 else []


def _disp_timing_defines() -> list[str]:
    """Diagnostic: -DMORI_DISP_TIMING enables the in-kernel wall_clock64 phase breakdown of the EP
    IntraNode dispatch ([CUSPLIT]/[GEOM]/[DIAG] for the default body, [BPHASE] for the clean one).
    Gated by the MORI_DISP_TIMING env so normal builds are unperturbed."""
    val = os.environ.get("MORI_DISP_TIMING", "")
    return ["-DMORI_DISP_TIMING"] if val.lower() in ("1", "true", "on", "yes") else []


def _ocp_fp_defines(arch: str) -> list[str]:
    """Enable the native gfx950 OCP FP4/FP8 conversion instructions (cvt_scalef32_pk_*) used by
    the fp4_blockwise combine's E2M1 quant/dequant helpers. Without this the helpers fall back to
    slow software bit-manipulation. Only relevant on gfx950; a no-op elsewhere."""
    return ["-DHIP_ENABLE_GFX950_OCP_BUILTINS=1"] if "gfx950" in str(arch) else []


def _debuginfo_flags() -> list[str]:
    """Return hipcc debug flags if MORI_DEBUG_INFO is enabled."""
    return ["-g", "-ggdb"] if is_debuginfo_enabled() else []


def _ensure_generated_include(mori_root: Path) -> Path:
    """Run generate_profiler_bindings.py into the JIT cache and return the include root.

    Always runs the generator (which is idempotent via write_if_changed) so that
    profiler slot changes in source are picked up without invalidating the JIT cache.
    Returns ``<cache_root>/generated/include/``, which must be passed as ``-I`` to hipcc.

    Raises FileNotFoundError if the generator script is not present (e.g. wheel
    install without the full source tree) — ENABLE_PROFILER requires the source tree.
    """
    gen_script = mori_root / "tools" / "profiler" / "generate_profiler_bindings.py"

    out_include = get_cache_root() / "generated" / "include"
    profiler_include_dir = out_include / "mori" / "profiler"
    pybind_out = get_cache_root() / "generated" / "profiler_bindings.cpp"

    if not gen_script.is_file():
        raise FileNotFoundError(
            f"Profiler binding generator not found: {gen_script}\n"
            "JIT compilation with ENABLE_PROFILER requires the mori source tree."
        )

    subprocess.check_call(
        [
            sys.executable,
            str(gen_script),
            str(mori_root),
            str(mori_root / "src"),
            str(profiler_include_dir),
            str(pybind_out),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    return out_include


def _collect_include_dirs(mori_root: Path) -> list[Path]:
    """Gather all include directories needed for device bitcode compilation."""
    dirs = [mori_root, mori_root / "include", mori_root / "src"]

    for subdir in ["spdlog/include", "msgpack-c/include"]:
        p = mori_root / "3rdparty" / subdir
        if p.is_dir():
            dirs.append(p)

    mpi_inc = find_mpi_include()
    if mpi_inc:
        dirs.append(Path(mpi_inc))

    # Profiler slot headers are only needed when ENABLE_PROFILER is set;
    # kernel sources wrap profiler calls with IF_ENABLE_PROFILER() so the
    # generated header is not required when profiling is disabled.
    if is_profiler_enabled():
        dirs.append(_ensure_generated_include(mori_root))

    return dirs


def _build_bitcode(
    cfg: BuildConfig, mori_root: Path, output: Path, *, cov: int = 5
) -> None:
    """Full bitcode build pipeline: compile → link → strip → verify."""
    include_dirs = _collect_include_dirs(mori_root)
    wrapper_src = mori_root / "src" / "shmem" / "shmem_device_api_wrapper.cpp"

    if not wrapper_src.is_file():
        raise FileNotFoundError(
            f"Source file not found: {wrapper_src}\n"
            "JIT compilation requires the mori source tree."
        )

    with tempfile.TemporaryDirectory(prefix="mori_jit_") as tmp:
        tmp_dir = Path(tmp)

        shim_src = tmp_dir / "globalGpuStates.hip"
        shim_src.write_text(_GLOBAL_GPU_STATES_SHIM)

        nic = detect_nic_type()
        print(
            f"[mori-jit] Compiling shmem device bitcode for {cfg.arch} (nic={nic}, cov={cov}) ..."
        )

        wrapper_bc = tmp_dir / "wrapper.bc"
        _hipcc_device_bc(cfg, wrapper_src, include_dirs, wrapper_bc, cov=cov)

        shim_bc = tmp_dir / "shim.bc"
        _hipcc_device_bc(cfg, shim_src, include_dirs, shim_bc, cov=cov)

        linked_bc = tmp_dir / "linked.bc"
        _llvm_link(cfg, [wrapper_bc, shim_bc], linked_bc)

        stripped_bc = tmp_dir / _BC_FILENAME
        _strip_lifetime_intrinsics(cfg, linked_bc, stripped_bc)

        _verify_bitcode(cfg, stripped_bc)

        output.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(stripped_bc, output)

    print(f"[mori-jit] Cached: {output}")


def _tunable_defines() -> list[str]:
    """Every -D whose value depends on the environment or the target arch.

    One list, used both to compile and (through cache.get_cache_dir) to key the result. That is the
    point of it existing: these two used to be written out separately, and the separate copy in
    cache.py was missing MORI_COMB_NOQUANT. A NOQUANT run therefore compiled nothing, loaded the
    full build's .hsaco and reported the full build's time -- which reads as "deleting the local
    quantise pass costs 0us" and is a statement about the cache, not the kernel. Two gates added
    later had the same hole. Anything constant across runs (platform, NIC, arch fp traits) stays
    out; it cannot make two runs differ.
    """
    return [
        *_comb_tdm_defines(),
        *_comb_barsleep_defines(),
        *_comb_barspread_defines(),
        *_comb_pipe_defines(),
        *_comb_diag_defines(),
        *_disp_timing_defines(),
        *_disp_nophase_defines(),
        *_disp_metadiag_defines(),
    ]


def _hipcc_genco(
    cfg: BuildConfig,
    source: Path,
    include_dirs: list[Path],
    output: Path,
) -> None:
    """Compile a .hip source to a device code object (.hsaco) via --genco."""
    cmd = [
        cfg.hipcc,
        "--genco",
        f"--offload-arch={cfg.arch}",
        "-std=c++17",
        "-O2",
        *_debuginfo_flags(),
        "-D__HIP_PLATFORM_AMD__",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
        *_nic_defines(),
        *_ccqe_defines(),
        *_profiler_defines(),
        *_ocp_fp_defines(cfg.arch),
        *_tunable_defines(),
    ]

    for d in include_dirs:
        cmd.extend(["-I", str(d)])
    cmd.extend([str(source), "-o", str(output)])

    subprocess.check_call(cmd, stderr=subprocess.STDOUT)


_PARALLEL_KERNEL_GROUPS: dict[str, list[str]] = {
    # Parallel compilation disabled — multi-module loading causes issues
    # with multiprocessing workers (concurrent ShmemModuleInit).
    # "dispatch_combine_kernels": ["ep_dispatch_kernels", "ep_combine_kernels"],
}


def _compile_one_genco(args: tuple) -> str:
    """Worker for parallel genco compilation."""
    kernel_name, arch, rocm_path, hipcc, include_dirs_str, output_path = args
    cfg_local = BuildConfig(
        arch=arch,
        rocm_path=rocm_path,
        hipcc=hipcc,
        llvm_link="",
        opt="",
    )
    mori_root = get_mori_source_root()
    source = mori_root / "src" / "ops" / "kernels" / f"{kernel_name}.hip"
    include_dirs = [Path(p) for p in include_dirs_str]
    _hipcc_genco(cfg_local, source, include_dirs, Path(output_path))
    return output_path


def _update_latest_symlink(hsaco_path: Path) -> None:
    """Maintain a latest/ directory with symlinks to the most recent .hsaco files.

    Structure: <arch>_<nic>/latest/<kernel>.hsaco -> ../<hash>/<kernel>.hsaco
    This allows C++ AutoLoad to find JIT-compiled kernels without knowing the hash.
    """
    try:
        latest_dir = hsaco_path.parent.parent / "latest"
        latest_dir.mkdir(exist_ok=True)
        link = latest_dir / hsaco_path.name
        target = os.path.relpath(hsaco_path, latest_dir)
        link.unlink(missing_ok=True)
        link.symlink_to(target)
    except OSError:
        pass


def compile_genco(
    kernel_name: str, source_dir: str = "src/ops/kernels"
) -> str | list[str]:
    """JIT compile kernel .hip source(s) to .hsaco via --genco. Returns cached path(s).

    Args:
        kernel_name: Name of the kernel (without .hip extension).
        source_dir: Directory relative to mori source root containing the .hip file.
            Defaults to "src/ops/kernels" for ops kernels.

    If the kernel has parallel sub-groups (e.g. dispatch_combine_kernels splits
    into ep_dispatch_kernels + ep_combine_kernels), compiles them in parallel
    and returns a list of paths.
    """
    mori_root = get_mori_source_root()
    if mori_root is None:
        raise FileNotFoundError(
            "Cannot JIT compile: mori source tree not found.\n"
            "JIT requires a source/editable install (pip install -e .)."
        )

    cfg = detect_build_config()
    nic = detect_nic_type()
    profiler = is_profiler_enabled()
    ccqe = is_ccqe_enabled()
    include_dirs = _collect_include_dirs(mori_root)

    sub_kernels = _PARALLEL_KERNEL_GROUPS.get(kernel_name)
    if sub_kernels:
        source_paths = [
            mori_root / "src" / "ops",
            mori_root / "include" / "mori",
        ]
        cache_dir = get_cache_dir(
            cfg.arch, source_paths, nic, profiler=profiler, ccqe=ccqe
        )

        hsaco_paths = [cache_dir / f"{k}.hsaco" for k in sub_kernels]
        if all(p.is_file() for p in hsaco_paths):
            return [str(p) for p in hsaco_paths]

        lock_path = cache_dir / f".{kernel_name}.lock"
        last_hsaco = str(hsaco_paths[-1])
        with FileBaton(lock_path, wait_for=last_hsaco) as baton:
            if baton.skipped or all(p.is_file() for p in hsaco_paths):
                return [str(p) for p in hsaco_paths]

            print(
                f"[mori-jit] Compiling {kernel_name} for {cfg.arch} (nic={nic}, "
                f"{len(sub_kernels)} files in parallel) ..."
            )

            include_strs = [str(d) for d in include_dirs]
            tasks = [
                (
                    k,
                    cfg.arch,
                    cfg.rocm_path,
                    cfg.hipcc,
                    include_strs,
                    str(cache_dir / f"{k}.hsaco"),
                )
                for k in sub_kernels
            ]

            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=len(sub_kernels)) as pool:
                list(pool.map(_compile_one_genco, tasks))

            for p in hsaco_paths:
                print(f"[mori-jit]   Cached: {p}")

        return [str(p) for p in hsaco_paths]

    source = mori_root / source_dir / f"{kernel_name}.hip"
    if not source.is_file():
        raise FileNotFoundError(f"Kernel source not found: {source}")

    # The .hip translation unit #includes sibling sources from its subsystem
    # (e.g. ops kernels pull in src/ops/dispatch_combine/*), so hash the whole
    # subsystem source tree; hashing only the top-level .hip reuses a stale
    # .hsaco when an included file changes.
    source_paths = [(mori_root / source_dir).parent, mori_root / "include" / "mori"]
    cache_dir = get_cache_dir(cfg.arch, source_paths, nic, profiler=profiler, ccqe=ccqe)
    hsaco_path = cache_dir / f"{kernel_name}.hsaco"

    if hsaco_path.is_file():
        _update_latest_symlink(hsaco_path)
        return str(hsaco_path)

    lock_path = cache_dir / f".{kernel_name}.hsaco.lock"
    with FileBaton(lock_path, wait_for=str(hsaco_path)) as baton:
        if baton.skipped or hsaco_path.is_file():
            _update_latest_symlink(hsaco_path)
            return str(hsaco_path)

        nic = detect_nic_type()
        print(
            f"[mori-jit] Compiling {kernel_name} for {cfg.arch} "
            f"(nic={nic}, ccqe={ccqe}, profiler={profiler}) ..."
        )
        _hipcc_genco(cfg, source, include_dirs, hsaco_path)
        print(f"[mori-jit]   Cached: {hsaco_path}")
        _update_latest_symlink(hsaco_path)

    return str(hsaco_path)


def ensure_bitcode(*, cov: int = 5) -> str:
    """Ensure the shmem device bitcode is compiled and cached. Returns the path.

    Args:
        cov: AMDGPU code object version (5 for Triton, 6 for FlyDSL).

    Thread/process safe: uses a file-based lock to prevent concurrent builds.
    """
    mori_root = get_mori_source_root()
    if mori_root is None:
        raise FileNotFoundError(
            "Cannot JIT compile: mori source tree not found.\n"
            "JIT requires a source/editable install (pip install -e .)."
        )

    cfg = detect_build_config()

    nic = detect_nic_type()
    profiler = is_profiler_enabled()
    ccqe = is_ccqe_enabled()
    source_paths = [
        mori_root / "src" / "shmem" / "shmem_device_api_wrapper.cpp",
        mori_root / "include" / "mori" / "shmem",
        mori_root / "include" / "mori" / "core",
    ]
    cache_dir = get_cache_dir(
        cfg.arch, source_paths, nic, profiler=profiler, cov=cov, ccqe=ccqe
    )
    bc_path = cache_dir / _BC_FILENAME

    if bc_path.is_file():
        return str(bc_path)

    lock_path = cache_dir / f".{_BC_FILENAME}.lock"
    with FileBaton(lock_path, wait_for=str(bc_path)) as baton:
        if baton.skipped or bc_path.is_file():
            return str(bc_path)
        _build_bitcode(cfg, mori_root, bc_path, cov=cov)

    return str(bc_path)
