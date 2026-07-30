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


def _disp_tdm_defines() -> list[str]:
    """Experimental: -DMORI_DISP_TDM routes the EP dispatch token payload through
    the gfx1250 TDM engine (see src/ops/dispatch_combine/intranode.hpp). Gated by
    the MORI_DISP_TDM env so the JIT kernel matches the host launch build."""
    val = os.environ.get("MORI_DISP_TDM", "")
    return ["-DMORI_DISP_TDM"] if val.lower() in ("1", "true", "on", "yes") else []


def _disp_clean_defines() -> list[str]:
    """Kernel body selector: -DMORI_DISP_CLEAN builds the legacy clean IntraNode dispatch body
    (EpDispatchIntraNodeKernel_clean_body, default geometry 256 blocks x 16 warps) instead of the
    default EpDispatchIntraNodeKernel_body (64 x 8). Gated by MORI_DISP_CLEAN env; default OFF."""
    return (
        ["-DMORI_DISP_CLEAN"]
        if os.environ.get("MORI_DISP_CLEAN", "").lower() in ("1", "true", "on", "yes")
        else []
    )


def _disp_complbackoff_defines() -> list[str]:
    """Diagnostic: -DMORI_DISP_COMPL_BACKOFF=N throttles dispatch's completion spins with
    s_sleep(N) instead of the backoff-free tight spin the shmem WaitUntil* primitives use
    (shmem_device_api.hpp). CrossDeviceBarrierIntraNodeKernel already documents that the
    unthrottled form livelocks the cco/xGMI fabric so a peer's flag write is never re-observed,
    and fixes it with s_sleep -- dispatch's completion never got that fix. Gated by
    MORI_DISP_COMPL_BACKOFF env (integer sleep arg); default OFF (tight spin, unchanged)."""
    val = os.environ.get("MORI_DISP_COMPL_BACKOFF", "").strip()
    if not val.isdigit():
        return []
    return [f"-DMORI_DISP_COMPL_BACKOFF={int(val)}"]


def _disp_metasplit_defines() -> list[str]:
    """-DMORI_DISP_METASPLIT=N sets how many sub-ranges each (block,peer) metadata run is cut into
    (default warpNum/npes). Every sub-range carries its own 128B head/tail remainder, and those go
    out as per-lane remote 4B stores instead of TDM. Integer env; default OFF (unchanged)."""
    val = os.environ.get("MORI_DISP_METASPLIT", "").strip()
    if not val.isdigit() or int(val) < 1:
        return []
    return [f"-DMORI_DISP_METASPLIT={int(val)}"]


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


def _disp_metavec_defines() -> list[str]:
    """Experiment: -DMORI_DISP_METAVEC sends the metadata staging straight to the peer as a coalesced
    load-batched vector copy instead of bouncing it through an LDS tile with TDM. Meta is ~196B per
    token, so the TDM path spends the engine on 32 small ops per block (8 warps x 4 fields) and then
    has to s_wait_tensorcnt(0) to hand the LDS tile back to the payload phase. Plain stores own no
    tile and no engine slot, so nothing waits for them here and they drain across the payload phase,
    made visible by a __threadfence_system() after it.

    MEASURED A BIG LOSS, kept only so nobody retries it: EP4-4K metasend 44.4 -> 475.7us and
    dispatch 1055.7 -> 317.2 GB/s (acc still PASS). Cross-GPU plain vector stores move meta at
    ~6 GB/s against TDM's ~97, so the TDM engine was never the problem -- the same conclusion the
    payload path already reached. Gated by MORI_DISP_METAVEC env; default OFF."""
    return (
        ["-DMORI_DISP_METAVEC"]
        if os.environ.get("MORI_DISP_METAVEC", "").lower() in ("1", "true", "on", "yes")
        else []
    )



def _disp_paydyn_defines() -> list[str]:
    """-DMORI_DISP_PAYDYN lets a block's warps claim payload work on demand instead of statically.

    The kernel ends when the slowest block ends, and a block ends when its slowest warp ends. Warps get
    equal token COUNTS, but a token goes to 1..4 destinations depending on routing, so equal counts are
    not equal bytes and the unluckiest of the 8 warps holds its block back while the other 7 sit
    drained.

    This does not violate the COUNT/FINALIZE/payload partition rule: what that rule protects is which
    BLOCK owns which tokens (a block reads back the dispDestTokIdMap entries it wrote itself, which is
    why a plain __syncthreads() can stand in for a grid barrier). Which warp inside the block takes a
    token is free -- the map is indexed by token. The block's token set is unchanged; only unit->warp
    assignment inside the block becomes dynamic, through an LDS atomic claimed 16 times per block at
    DBN=64/wpb=8.

    Aimed at the largest remaining item at DBN=64/wpb=8: payload is 125.0us of the 166.0us kernel, and
    load imbalance inside it is not addressable by any of the geometry-preserving changes tried so far
    (PAYSPLIT, GRIDFLAG, SRCVEC all measured negative).
    """
    return (
        ["-DMORI_DISP_PAYDYN"]
        if os.environ.get("MORI_DISP_PAYDYN", "").lower() in ("1", "true", "on", "yes")
        else []
    )


def _disp_double_defines() -> list[str]:
    """Cost isolation by DOUBLING a phase: kernel(2x) - kernel(1x) is that phase's real cost.

    The deletion gates (MORI_DISP_NOMETA / NOPAY / NOSTG) cannot reach COUNT, RESERVE or COMPLETION:
    deleting COUNT feeds RESERVE a garbage s_N and lets the payload store outside its reserved slots
    (corruption, not just a wrong answer), and skipping the grid barrier or the peer wait breaks the
    signal-clearing invariant, so the next replay waits forever. Doubling has neither problem and,
    unlike deletion, keeps the result CORRECT -- ACC must still PASS, which is what proves the
    measurement ran on a working kernel rather than a broken one.

    MORI_DISP_DBLCOUNT   -- COUNT runs twice, second pass' atomicAdd redirected to a scratch LDS
                            histogram (its other store writes the same sentinel, already idempotent).
    MORI_DISP_DBLRESERVE -- RESERVE does a second remote atomic on the same counter adding 0: same
                            cross-GPU latency and same all-blocks-on-one-counter contention, no effect
                            on slot allocation.

    COMPLETION cannot be doubled (it is a protocol, not work), so it comes out by subtraction from the
    41.0us non-payload total once COUNT, RESERVE, meta (6.95) and gather (5.45) are known.
    """
    out = []
    for name in ("DBLCOUNT", "DBLRESERVE"):
        if os.environ.get(f"MORI_DISP_{name}", "").lower() in ("1", "true", "on", "yes"):
            out.append(f"-DMORI_DISP_{name}")
    return out


def _disp_gridflag_defines() -> list[str]:
    """-DMORI_DISP_GRIDFLAG makes the dispatch grid barrier gridDim.x flags instead of one counter.

    The counter form has every block's thread 0 atomicAdd one shared address while block 0's warp
    spins reading that same cacheline, so the increments serialize and reader and writers ping-pong
    the line. GRIDFLAG gives each block its own 128B-separated flag and polls gridDim.x of them with
    32 lanes at a time (2 rounds at DBN=64).

    Aimed at the completion bucket, the largest un-dissected part of the 41.0us non-payload cost at
    DBN=64/wpb=8 (kernel 166.0us, payload marginal 125.0us, of which meta 6.95 and gather 5.45 are
    already proven incompressible).
    """
    return (
        ["-DMORI_DISP_GRIDFLAG"]
        if os.environ.get("MORI_DISP_GRIDFLAG", "").lower() in ("1", "true", "on", "yes")
        else []
    )


def _disp_paysplit_defines() -> list[str]:
    """-DMORI_DISP_PAYSPLIT=N issues each token's payload as N TDM segments of the SAME LDS tile.

    The payload phase is bounded by in-flight TDM operations, not bytes. Geometry sweep evidence:
    64x8 (512 warps) = 1278 GB/s, while 64x16 and 128x8 both land on 1366 -- identical ceiling from
    an identical 1024-warp in-flight count, one bought with LDS (blockDim 512, 229KB) and the other
    with CUs (128 blocks). Both spend more physical resource, which is exactly what is not allowed
    here, so this buys the same in-flight count in code instead: per warp, N loads and N x ~3.6
    stores in flight rather than 1 and ~3.6, with the tile still hiddenDim elements (14336B per warp,
    114KB per block, 64 blocks -- unchanged).

    Cost is N x the TDM operations for the same bytes, so it only pays while a segment stays far
    above the 128B minimum row: at N=2 a segment is 7168B, at N=4 it is 3584B. Falls back to the
    single-segment form when hiddenDim does not divide evenly or a segment would drop under 128B.

    Set MORI_DISP_PAYSPLIT to the segment count (2, 4, ...); unset or 1 disables.
    """
    v = os.environ.get("MORI_DISP_PAYSPLIT", "").strip()
    if not v.isdigit() or int(v) <= 1:
        return []
    return [f"-DMORI_DISP_PAYSPLIT={int(v)}"]


def _disp_srcvec_defines() -> list[str]:
    """-DMORI_DISP_SRCVEC copies srcmap's cross-GPU run 16B at a time instead of 4B at a time.

    srcmap is the only meta field that never gets a TDM body: 1 element per slot means a (block,peer)
    run is cc ~= 58 elements at DBN=64/wpb=8, below the 64 a legal tile needs (TdmCheapDim1's 32x2),
    and TdmAlignSplit128's aligned remainder reaches 1 row where 2 are required. So the whole run
    goes through the scalar peel as cc separate 4B stores to the peer. Cross-GPU writes are priced
    per transaction (~54 GB/s for plain stores, see MORI_DISP_METAVEC), so that is 4x the
    transactions needed. Measured before: htSrc = 20.1us of a 36.7us metasend, with htIdx/htWt/htSc
    at 0.0/0.0/0.1 (TIMING build, DBN=64/wpb=8).

    Gated by MORI_DISP_SRCVEC env; default OFF. Falls back per run when the two bases do not share a
    16B phase.
    """
    return (
        ["-DMORI_DISP_SRCVEC"]
        if os.environ.get("MORI_DISP_SRCVEC", "").lower() in ("1", "true", "on", "yes")
        else []
    )


def _disp_metalds_defines() -> list[str]:
    """-DMORI_DISP_METALDS has FINALIZE gather the four metadata fields straight into the LDS tile the
    meta TDM store sends from, instead of into the global staging arrays that store then has to TDM
    load back. Same 2.9MB, one HBM round trip fewer.

    Measured cost of what this removes, by差分 (kernel time from [DISPBW], noTIMING):
      full 166.0us | MORI_DISP_NOSTG (no gather) 160.55us | MORI_DISP_NOMETA (no meta phase) 159.05us
    i.e. the gather is 5.45us and the meta phase 6.95us, against ~1.8us of engine time for 2.9MB.

    Cross-GPU vector stores are not an alternative: MORI_DISP_METAVEC measures 995.5 GB/s (213.4us,
    ~54 GB/s of effective fabric write bandwidth) and per-destination stores fused into the payload
    loop (MORI_DISP_METAFUSE) measure 462.6 GB/s. Meta must stay on TDM, and TDM sources LDS only.

    The LDS region is shared by the whole block, so this drains the meta stores just BEFORE the
    barrier that hands the tile to the payload phase; the default deferral of that drain past that
    barrier is only valid for the per-warp-private tile and is bypassed here.

    MEASURED A LOSS and kept only so nobody retries it: 1253.8 GB/s against a 1279.4 baseline at
    DBN=64 (-2.0%), ACC PASS. Splitting the LDS and global paths so neither pointer loses its address
    space (the first form let them share one variable and degrade to flat accesses) recovered only
    +0.15%, which rules that out as the cause. What the 5.45us of MORI_DISP_NOSTG actually measures is
    mostly the gather READING the original input from HBM, and that read is unchanged here -- so the
    savings are just the staging write plus the TDM load, and they do not cover the LDS bank conflicts
    on the gather, the extra thread-0 layout pass and barrier, or the loss of the deferred meta drain
    (this path has to drain before the barrier, being block-shared rather than per-warp).

    Gated by MORI_DISP_METALDS env; default OFF. Falls back to staging per block, at runtime, when
    the block's own s_N does not fit the tile (s_mOk).
    """
    return (
        ["-DMORI_DISP_METALDS"]
        if os.environ.get("MORI_DISP_METALDS", "").lower() in ("1", "true", "on", "yes")
        else []
    )


def _disp_metafuse_defines() -> list[str]:
    """-DMORI_DISP_METAFUSE folds the metadata send into the payload loop and deletes the separate
    meta phase. Each (token, destination) pair the payload loop already visits carries its own 196B
    of meta (indices, weights, scale, srcmap) as plain cross-GPU stores, issued right after that
    token's payload TDM store -- cycles the warp otherwise spends spinning in s_wait_tensorcnt.

    Why: the diagnostic gates below measured the meta phase at 6.95us of kernel time (166.0 ->
    159.05us with MORI_DISP_NOMETA) for 2.9MB, i.e. ~1.8us of engine time and ~5us of pure staging
    round trip / LDS bounce / TDM latency. Fusing removes the latency AND the staging buffers: the
    payload loop's destPe/destTokId are exactly what the staging gather keyed on, and the source is
    the original input, so -DMORI_DISP_NOSTG then drops FINALIZE's gather as dead work.

    MEASURED A LOSS, badly, and kept only so nobody retries it: 462.6 GB/s against a 1279.4 baseline
    (kernel 166.0 -> 459us), ACC PASS. Cross-GPU meta cannot leave TDM. Per destination this issues
    four short bursts (32B/32B/128B/4B) whose addresses depend on each other, so a warp serialises
    ~116 fabric write round trips over its ~29 destinations, and every other warp is simultaneously
    stalled in s_wait_tensorcnt with nothing to hide them behind. MORI_DISP_METAVEC bounds the batched
    version of the same idea at 995.5 GB/s (~54 GB/s of fabric write bandwidth), so the fine-grained
    form was never going to reach TDM's ~1600.

    Gated by MORI_DISP_METAFUSE / MORI_DISP_NOSTG env; default OFF. NOSTG is only correct together
    with METAFUSE (nothing else feeds the peer's meta buffers once staging is gone).
    """
    out: list[str] = []
    if os.environ.get("MORI_DISP_METAFUSE", "").lower() in ("1", "true", "on", "yes"):
        out.append("-DMORI_DISP_METAFUSE")
    if os.environ.get("MORI_DISP_NOSTG", "").lower() in ("1", "true", "on", "yes"):
        out.append("-DMORI_DISP_NOSTG")
    return out


def _disp_nophase_defines() -> list[str]:
    """DIAGNOSTIC, WRONG RESULTS ON PURPOSE. -DMORI_DISP_NOMETA / -DMORI_DISP_NOPAY compile away the
    meta send / the payload send while leaving launch geometry, LDS reservation and occupancy alone,
    so kernel(full) - kernel(NOX) gives phase X's real cost. This exists because MORI_DISP_TIMING's
    clock64() probes sit inside the per-token loops and inflate the phases they measure (the timed
    build reports ~87us of non-payload against a ~33us noTIMING budget), which made the timed split
    useless for deciding where the last 3us to 1.3TB/s should come from.

    Never enable with ACC=1; the dispatch output is deliberately incomplete.
    """
    out: list[str] = []
    if os.environ.get("MORI_DISP_NOMETA", "").lower() in ("1", "true", "on", "yes"):
        out.append("-DMORI_DISP_NOMETA")
    if os.environ.get("MORI_DISP_NOPAY", "").lower() in ("1", "true", "on", "yes"):
        out.append("-DMORI_DISP_NOPAY")
    return out


def _disp_metafield_defines() -> list[str]:
    """Experiment: -DMORI_DISP_METAFIELD gives each warp one (peer, field) item instead of one
    (peer, sub-range) run carrying all four fields. That halves the block's meta TDM op count
    (npes*4 = 16 items over 8 warps, 2 stores per warp instead of 4) without idling any warp, and
    each item covers the peer's whole run, so it carries the fewest possible 128B remainders.

    The premise was that op count is the lever: with the deferred meta drain, mSt is 21.4us of store ISSUE
    alone (mDrain 0.0), so the meta phase looked like it was queueing on the TDM engine.

    MEASURED A LOSS, kept only so nobody retries it: EP4-4K 1276.8 -> 1222.2 GB/s (-4.3%), acc still
    PASS. Together with METASPLIT=1 (also halves the op count, also lost: -1.5%) this says the meta
    TDM cost tracks BYTES, not op count -- so the premise was wrong and there is nothing to win by
    regrouping the same bytes into fewer, larger ops. Two costs this form adds outweigh the halved op
    count: srcmap is the only field with no TDM body (its run is cc 4B elements, under the 128B row
    floor), so per-field assignment concentrates ALL of its per-lane remote stores into npes warps
    while the other warps go idle -- and the __syncthreads() before the payload phase waits for those
    stragglers; and a warp's two items share one tile, so the second pays a drain the per-run form
    never had. The shipping per-run form keeps all 8 warps carrying an equal mix of every field.
    Gated by MORI_DISP_METAFIELD env; default OFF."""
    return (
        ["-DMORI_DISP_METAFIELD"]
        if os.environ.get("MORI_DISP_METAFIELD", "").lower() in ("1", "true", "on", "yes")
        else []
    )


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
        *_disp_tdm_defines(),
        *_disp_timing_defines(),
        *_disp_clean_defines(),
        *_disp_complbackoff_defines(),
        *_disp_metasplit_defines(),
        *_disp_metavec_defines(),
        *_disp_metafield_defines(),
        *_disp_paydyn_defines(),
        *_disp_double_defines(),
        *_disp_gridflag_defines(),
        *_disp_paysplit_defines(),
        *_disp_srcvec_defines(),
        *_disp_metalds_defines(),
        *_disp_metafuse_defines(),
        *_disp_nophase_defines(),
        *_disp_metadiag_defines(),
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
