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

import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from mori.jit.cache import get_cache_dir
from mori.jit.config import (
    BuildConfig,
    detect_build_config,
    detect_nic_type,
    find_mpi_include,
    get_mori_source_root,
)

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
) -> None:
    """Compile a single source file to device-only bitcode."""
    cmd = [
        cfg.hipcc,
        "-c",
        "--cuda-device-only",
        "-emit-llvm",
        f"--offload-arch={cfg.arch}",
        "-fgpu-rdc",
        "-mcode-object-version=5",
        "-std=c++17",
        "-O2",
        "-D__HIP_PLATFORM_AMD__",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
        *_nic_defines(),
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


def _nic_defines() -> list[str]:
    """Return compiler -D flags for the detected NIC type (device-side macros)."""
    nic = detect_nic_type()
    if nic == "bnxt":
        return ["-DMORI_DEVICE_NIC_BNXT"]
    elif nic == "ionic":
        return ["-DMORI_DEVICE_NIC_IONIC"]
    return []


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

    return dirs


def _build_bitcode(cfg: BuildConfig, mori_root: Path, output: Path) -> None:
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
            f"[mori-jit] Compiling shmem device bitcode for {cfg.arch} (nic={nic}) ..."
        )

        wrapper_bc = tmp_dir / "wrapper.bc"
        _hipcc_device_bc(cfg, wrapper_src, include_dirs, wrapper_bc)

        shim_bc = tmp_dir / "shim.bc"
        _hipcc_device_bc(cfg, shim_src, include_dirs, shim_bc)

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
        "-D__HIP_PLATFORM_AMD__",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
        *_nic_defines(),
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


def compile_genco(kernel_name: str) -> str | list[str]:
    """JIT compile kernel .hip source(s) to .hsaco via --genco. Returns cached path(s).

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
    include_dirs = _collect_include_dirs(mori_root)

    sub_kernels = _PARALLEL_KERNEL_GROUPS.get(kernel_name)
    if sub_kernels:
        source_paths = [
            mori_root / "src" / "ops" / "kernels",
            mori_root / "include" / "mori",
        ]
        cache_dir = get_cache_dir(cfg.arch, source_paths, nic)

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

    source = mori_root / "src" / "ops" / "kernels" / f"{kernel_name}.hip"
    if not source.is_file():
        raise FileNotFoundError(f"Kernel source not found: {source}")

    source_paths = [source, mori_root / "include" / "mori"]
    cache_dir = get_cache_dir(cfg.arch, source_paths, nic)
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
        print(f"[mori-jit] Compiling {kernel_name} for {cfg.arch} (nic={nic}) ...")
        _hipcc_genco(cfg, source, include_dirs, hsaco_path)
        print(f"[mori-jit]   Cached: {hsaco_path}")
        _update_latest_symlink(hsaco_path)

    return str(hsaco_path)


def ensure_bitcode() -> str:
    """Ensure the shmem device bitcode is compiled and cached. Returns the path.

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
    source_paths = [
        mori_root / "src" / "shmem" / "shmem_device_api_wrapper.cpp",
        mori_root / "include" / "mori" / "shmem",
        mori_root / "include" / "mori" / "core",
    ]
    cache_dir = get_cache_dir(cfg.arch, source_paths, nic)
    bc_path = cache_dir / _BC_FILENAME

    if bc_path.is_file():
        return str(bc_path)

    lock_path = cache_dir / f".{_BC_FILENAME}.lock"
    with FileBaton(lock_path, wait_for=str(bc_path)) as baton:
        if baton.skipped or bc_path.is_file():
            return str(bc_path)
        _build_bitcode(cfg, mori_root, bc_path)

    return str(bc_path)
