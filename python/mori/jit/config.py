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
"""Runtime build configuration: GPU arch detection and compiler tool paths."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

_SUPPORTED_ARCHS = ["gfx942", "gfx950"]


@dataclass(frozen=True)
class BuildConfig:
    arch: str
    rocm_path: str
    hipcc: str
    llvm_link: str
    opt: str


_cached_config: BuildConfig | None = None


def detect_gpu_arch(rocm_path: str = "/opt/rocm") -> str:
    """Detect the GPU architecture on the current machine.

    Raises RuntimeError if detection fails and no env override is set.
    """
    env_arch = os.environ.get("MORI_GPU_ARCHS")
    if env_arch:
        for arch in _SUPPORTED_ARCHS:
            if arch in env_arch:
                return arch

    enumerator = os.path.join(rocm_path, "bin", "rocm_agent_enumerator")
    if os.path.isfile(enumerator):
        try:
            out = subprocess.check_output(
                [enumerator], text=True, stderr=subprocess.DEVNULL
            )
            for line in out.strip().split("\n"):
                line = line.strip()
                if line in _SUPPORTED_ARCHS:
                    return line
        except subprocess.CalledProcessError:
            pass

    try:
        out = subprocess.check_output(
            ["rocminfo"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.split("\n"):
            for arch in _SUPPORTED_ARCHS:
                if arch in line:
                    return arch
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    env_amdgpu = os.environ.get("AMDGPU_TARGETS")
    if env_amdgpu:
        for arch in _SUPPORTED_ARCHS:
            if arch in env_amdgpu:
                return arch

    raise RuntimeError(
        f"Cannot detect GPU architecture. "
        f"Set MORI_GPU_ARCHS to one of {_SUPPORTED_ARCHS}"
    )


def _find_tool(rocm_path: str, name: str) -> str:
    """Locate a ROCm LLVM tool, raising FileNotFoundError if missing."""
    candidates = [
        os.path.join(rocm_path, "lib", "llvm", "bin", name),
        os.path.join(rocm_path, "bin", name),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"{name} not found. Searched: {candidates}. Is ROCm installed?"
    )


def find_mpi_include() -> str | None:
    """Locate the MPI include directory containing mpi.h."""
    candidates = [
        "/usr/lib/x86_64-linux-gnu/openmpi/include",
        "/usr/include/mpi",
        "/usr/include/openmpi-x86_64",
        "/usr/include/mpich-x86_64",
        "/usr/local/include",
    ]
    for d in candidates:
        if os.path.isfile(os.path.join(d, "mpi.h")):
            return d

    try:
        out = subprocess.check_output(
            ["mpicc", "--showme:compile"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        for token in out.split():
            if token.startswith("-I"):
                path = token[2:]
                if os.path.isfile(os.path.join(path, "mpi.h")):
                    return path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


_DRIVER_TO_NIC = {
    "bnxt_re": "bnxt",
    "bnxt_en": "bnxt",
    "mlx5_core": "mlx5",
    "mlx5_ib": "mlx5",
    "ionic_rdma": "ionic",
    "ionic": "ionic",
}

_NIC_PCI_VENDORS = {
    "14e4": "bnxt",  # Broadcom BCM576xx / BCM578xx
    "1dd8": "ionic",  # AMD/Pensando
    "15b3": "mlx5",  # Mellanox/NVIDIA ConnectX
}

_LIB_SEARCH_PATHS = [
    "/usr/local/lib",
    "/usr/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/lib/x86_64-linux-gnu",
]

_NIC_LIB_NAMES: dict[str, str] = {
    "mlx5": "libmlx5.so",
    "bnxt": "libbnxt_re.so",
    "ionic": "libionic.so",
}



def _has_nic_lib(nic: str) -> bool:
    """Check whether the user-space RDMA verbs provider library is installed.

    bnxt headers (bnxt_re_dv.h, bnxt_re_hsi.h) are bundled in the mori source
    tree, so only the shared library needs to be present on the system.
    """
    lib_name = _NIC_LIB_NAMES.get(nic)
    if not lib_name:
        return False
    return any(os.path.exists(os.path.join(d, lib_name)) for d in _LIB_SEARCH_PATHS)


def _classify_ib_device(dev_path: str) -> str | None:
    """Identify the NIC type for a single /sys/class/infiniband/<dev> entry.

    Reads the kernel driver symlink which works regardless of device naming
    convention (bnxt_re_0, rdma0, etc.).
    """
    driver_link = os.path.join(dev_path, "device", "driver")
    try:
        driver_name = os.path.basename(os.readlink(driver_link))
        return _DRIVER_TO_NIC.get(driver_name)
    except OSError:
        return None


def detect_nic_type() -> str:
    """Detect the RDMA NIC type for device-side IBGDA dispatch.

    Detection priority:
      1. MORI_DEVICE_NIC env var (explicit override, same as CMake)
      2. /sys/class/infiniband/ — pick NIC with most devices + verify library
      3. lspci PCI vendor ID
      4. User-space library fallback
      5. Default: mlx5

    Returns ``"bnxt"``, ``"ionic"``, or ``"mlx5"``.
    """
    env_device_nic = os.environ.get("MORI_DEVICE_NIC", "").lower()
    if env_device_nic in ("bnxt", "ionic", "mlx5"):
        return env_device_nic

    ib_dir = "/sys/class/infiniband"
    if os.path.isdir(ib_dir):
        try:
            devices = os.listdir(ib_dir)
            counts: dict[str, int] = {"mlx5": 0, "bnxt": 0, "ionic": 0}

            for dev in devices:
                if dev.startswith("bnxt_re"):
                    counts["bnxt"] += 1
                elif dev.startswith("ionic"):
                    counts["ionic"] += 1
                elif dev.startswith("mlx5"):
                    counts["mlx5"] += 1
                else:
                    nic = _classify_ib_device(os.path.join(ib_dir, dev))
                    if nic and nic in counts:
                        counts[nic] += 1

            for nic, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                if cnt > 0 and _has_nic_lib(nic):
                    return nic
        except OSError:
            pass

    try:
        lspci_out = subprocess.check_output(
            ["lspci", "-nn", "-d", "::0200"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        vendor_counts: dict[str, int] = {}
        for line in lspci_out.strip().split("\n"):
            for vid, nic in _NIC_PCI_VENDORS.items():
                if vid in line:
                    vendor_counts[nic] = vendor_counts.get(nic, 0) + 1
        if vendor_counts:
            for nic, _ in sorted(
                vendor_counts.items(), key=lambda x: x[1], reverse=True
            ):
                if _has_nic_lib(nic):
                    return nic
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    for nic, lib_name in _NIC_LIB_NAMES.items():
        if any(os.path.exists(os.path.join(d, lib_name)) for d in _LIB_SEARCH_PATHS):
            return nic

    return "mlx5"


def detect_build_config() -> BuildConfig:
    """Auto-detect the full build config (cached after first call)."""
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    arch = detect_gpu_arch(rocm_path)
    hipcc = os.path.join(rocm_path, "bin", "hipcc")
    if not os.path.isfile(hipcc):
        raise FileNotFoundError(f"hipcc not found at {hipcc}")

    _cached_config = BuildConfig(
        arch=arch,
        rocm_path=rocm_path,
        hipcc=hipcc,
        llvm_link=_find_tool(rocm_path, "llvm-link"),
        opt=_find_tool(rocm_path, "opt"),
    )
    return _cached_config


def get_mori_source_root() -> Path | None:
    """Locate the mori source tree root for JIT compilation.

    Search order:
      1. Development / editable install: repo root (3 levels up from this file)
      2. Wheel install: _jit-sources/ packaged inside the mori package
    """
    here = Path(__file__).resolve().parent  # python/mori/jit/

    candidate = here.parent.parent.parent  # <repo>/
    if (candidate / "include" / "mori").is_dir() and (
        candidate / "src" / "shmem"
    ).is_dir():
        return candidate

    packaged = here.parent / "_jit-sources"  # mori/_jit-sources/
    if (packaged / "include" / "mori").is_dir():
        return packaged

    return None
