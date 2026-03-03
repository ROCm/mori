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


def detect_nic_type() -> str:
    """Detect the RDMA NIC type on the current machine.

    Returns ``"bnxt"``, ``"ionic"``, or ``"mlx5"`` (default).
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
            bnxt_count = sum(1 for d in devices if d.startswith("bnxt_re"))
            ionic_count = sum(1 for d in devices if d.startswith("ionic"))
            mlx5_count = sum(1 for d in devices if d.startswith("mlx5"))
            if bnxt_count > 0 and bnxt_count >= mlx5_count:
                return "bnxt"
            if ionic_count > 0 and ionic_count >= mlx5_count:
                return "ionic"
        except OSError:
            pass

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
    """Locate the mori source tree root (for development/source installs).

    Returns None when running from a wheel without source tree.
    """
    here = Path(__file__).resolve().parent          # python/mori/jit/
    candidate = here.parent.parent.parent           # <repo>/
    if (candidate / "include" / "mori").is_dir() and (candidate / "src" / "shmem").is_dir():
        return candidate
    return None
