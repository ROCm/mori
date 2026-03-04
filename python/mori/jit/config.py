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
    "14e4": "bnxt",   # Broadcom BCM576xx / BCM578xx
    "1dd8": "ionic",  # AMD/Pensando
    "15b3": "mlx5",   # Mellanox/NVIDIA ConnectX
}

_LIB_SEARCH_PATHS = [
    "/usr/local/lib", "/usr/lib", "/usr/lib/x86_64-linux-gnu",
    "/lib/x86_64-linux-gnu",
]


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
    """Detect the RDMA NIC type on the current machine.

    Detection priority:
      1. Environment variable (USE_BNXT=ON or USE_IONIC=ON)
      2. /sys/class/infiniband/ — device name prefix, then driver symlink
      3. lspci PCI vendor ID
      4. User-space library fallback (libbnxt_re.so / libionic.so)
      5. Default: mlx5

    Returns ``"bnxt"``, ``"ionic"``, or ``"mlx5"``.
    """
    env_bnxt = os.environ.get("USE_BNXT", "").upper()
    env_ionic = os.environ.get("USE_IONIC", "").upper()
    if env_bnxt == "ON":
        return "bnxt"
    if env_ionic == "ON":
        return "ionic"

    ib_dir = "/sys/class/infiniband"
    if os.path.isdir(ib_dir):
        try:
            devices = os.listdir(ib_dir)
            counts: dict[str, int] = {"bnxt": 0, "ionic": 0, "mlx5": 0}

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

            if counts["bnxt"] > 0 and counts["bnxt"] >= counts["mlx5"]:
                return "bnxt"
            if counts["ionic"] > 0 and counts["ionic"] >= counts["mlx5"]:
                return "ionic"
            if counts["mlx5"] > 0:
                return "mlx5"
        except OSError:
            pass

    try:
        lspci_out = subprocess.check_output(
            ["lspci", "-nn", "-d", "::0200"],
            text=True, stderr=subprocess.DEVNULL,
        )
        vendor_counts: dict[str, int] = {}
        for line in lspci_out.strip().split("\n"):
            for vid, nic in _NIC_PCI_VENDORS.items():
                if vid in line:
                    vendor_counts[nic] = vendor_counts.get(nic, 0) + 1
        if vendor_counts:
            return max(vendor_counts, key=vendor_counts.get)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    _NIC_LIBS = [
        ("libbnxt_re.so", "bnxt"),
        ("libionic.so", "ionic"),
        ("libmlx5.so", "mlx5"),
    ]
    for lib_name, nic in _NIC_LIBS:
        for d in _LIB_SEARCH_PATHS:
            if os.path.exists(os.path.join(d, lib_name)):
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
      2. Wheel install: _jit_sources/ packaged inside the mori package
    """
    here = Path(__file__).resolve().parent          # python/mori/jit/

    candidate = here.parent.parent.parent           # <repo>/
    if (candidate / "include" / "mori").is_dir() and (candidate / "src" / "shmem").is_dir():
        return candidate

    packaged = here.parent / "_jit_sources"         # mori/_jit_sources/
    if (packaged / "include" / "mori").is_dir():
        return packaged

    return None
