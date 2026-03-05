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
import os
import subprocess
import sys
from pathlib import Path
import shutil

from setuptools import Extension, find_packages, setup
from setuptools.command.build import build as _build
from setuptools.command.build_ext import build_ext


_supported_arch_list = ["gfx942", "gfx950"]


def _detect_local_gpu_arch() -> str | None:
    """Auto-detect the GPU architecture on the current machine."""
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    enumerator = os.path.join(rocm_path, "bin", "rocm_agent_enumerator")
    if os.path.isfile(enumerator):
        try:
            out = subprocess.check_output([enumerator], text=True)
            for line in out.strip().split("\n"):
                line = line.strip()
                if line.startswith("gfx") and line != "gfx000" and line in _supported_arch_list:
                    return line
        except subprocess.CalledProcessError:
            pass
    return None


def _get_gpu_archs() -> str:
    """Determine GPU target architectures for compilation.

    Priority: MORI_GPU_ARCHS > local GPU > PYTORCH_ROCM_ARCH / GPU_ARCHS > fat binary default.
    """
    mori_gpu_archs = os.environ.get("MORI_GPU_ARCHS", None)
    if mori_gpu_archs:
        return mori_gpu_archs

    local_arch = _detect_local_gpu_arch()
    if local_arch:
        return local_arch

    archs = os.environ.get("PYTORCH_ROCM_ARCH", None)

    gpu_archs = os.environ.get("GPU_ARCHS", None)
    if gpu_archs:
        archs = gpu_archs

    if archs:
        arch_list = archs.replace(" ", ";").split(";")
        valid_arch_list = list(set(_supported_arch_list) & set(arch_list))
        if valid_arch_list:
            return ";".join(valid_arch_list)

    print(f"[mori] No GPU arch specified — building fat binary for {_supported_arch_list}")
    return ";".join(_supported_arch_list)


def _detect_nic_type() -> dict:
    """Auto-detect RDMA NIC type for IBGDA provider selection.

    Detection priority:
      1. Environment variable (USE_BNXT=ON or USE_IONIC=ON) — explicit override
      2. /sys/class/infiniband/ — active RDMA devices registered with the kernel
      3. lspci PCI vendor ID — hardware present but driver may not be loaded
      4. User-space library (libbnxt_re.so / libionic.so) — fallback
      5. Default: MLX5 (Mellanox ConnectX)
    """
    result = {"USE_BNXT": "OFF", "USE_IONIC": "OFF"}

    # 1. Explicit environment variable
    env_bnxt = os.environ.get("USE_BNXT", "").upper()
    env_ionic = os.environ.get("USE_IONIC", "").upper()
    if env_bnxt == "ON":
        result["USE_BNXT"] = "ON"
        return result
    if env_ionic == "ON":
        result["USE_IONIC"] = "ON"
        return result
    if env_bnxt or env_ionic:
        return result

    # 2. Active RDMA devices (most accurate — only shows devices with loaded drivers)
    # First try device name prefix (bnxt_re_*, mlx5_*, ionic_*), then fall back
    # to reading the kernel driver symlink for generically-named devices (rdma0, etc.).
    _driver_to_nic = {
        "bnxt_re": "BNXT", "bnxt_en": "BNXT",
        "mlx5_core": "MLX5", "mlx5_ib": "MLX5",
        "ionic_rdma": "IONIC", "ionic": "IONIC",
    }
    ib_dir = "/sys/class/infiniband"
    if os.path.isdir(ib_dir):
        try:
            devices = os.listdir(ib_dir)
            counts = {"BNXT": 0, "IONIC": 0, "MLX5": 0}

            for dev in devices:
                if dev.startswith("bnxt_re"):
                    counts["BNXT"] += 1
                elif dev.startswith("ionic"):
                    counts["IONIC"] += 1
                elif dev.startswith("mlx5"):
                    counts["MLX5"] += 1
                else:
                    driver_link = os.path.join(ib_dir, dev, "device", "driver")
                    try:
                        driver_name = os.path.basename(os.readlink(driver_link))
                        nic = _driver_to_nic.get(driver_name)
                        if nic:
                            counts[nic] += 1
                    except OSError:
                        pass

            if counts["BNXT"] > 0 and counts["BNXT"] >= counts["MLX5"]:
                result["USE_BNXT"] = "ON"
                print(f"[mori] Found {counts['BNXT']} BNXT RDMA device(s) in /sys/class/infiniband/")
                return result
            if counts["IONIC"] > 0 and counts["IONIC"] >= counts["MLX5"]:
                result["USE_IONIC"] = "ON"
                print(f"[mori] Found {counts['IONIC']} IONIC RDMA device(s) in /sys/class/infiniband/")
                return result
            if counts["MLX5"] > 0:
                print(f"[mori] Found {counts['MLX5']} MLX5 RDMA device(s) in /sys/class/infiniband/")
                return result
        except OSError:
            pass

    # 3. lspci hardware detection
    _NIC_PCI_VENDORS = {
        "14e4": "BNXT",   # Broadcom BCM576xx / BCM578xx
        "1dd8": "IONIC",  # AMD/Pensando
        "15b3": "MLX5",   # Mellanox/NVIDIA ConnectX
    }
    try:
        lspci_out = subprocess.check_output(
            ["lspci", "-nn", "-d", "::0200"],
            text=True, stderr=subprocess.DEVNULL,
        )
        vendor_counts = {}
        for line in lspci_out.strip().split("\n"):
            for vid, nic in _NIC_PCI_VENDORS.items():
                if vid in line:
                    vendor_counts[nic] = vendor_counts.get(nic, 0) + 1

        if vendor_counts:
            dominant = max(vendor_counts, key=vendor_counts.get)
            if dominant == "BNXT":
                result["USE_BNXT"] = "ON"
            elif dominant == "IONIC":
                result["USE_IONIC"] = "ON"
            print(f"[mori] lspci detected: {vendor_counts}")
            return result
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 4. Library file fallback
    _LIB_SEARCH_PATHS = ["/usr/local/lib", "/usr/lib", "/usr/lib/x86_64-linux-gnu",
                         "/lib/x86_64-linux-gnu"]
    _NIC_LIBS = [
        ("libbnxt_re.so", "USE_BNXT"),
        ("libionic.so", "USE_IONIC"),
    ]
    for lib_name, flag in _NIC_LIBS:
        for d in _LIB_SEARCH_PATHS:
            if os.path.exists(os.path.join(d, lib_name)):
                result[flag] = "ON"
                print(f"[mori] Found {lib_name} in {d}")
                return result

    for d in _LIB_SEARCH_PATHS:
        if os.path.exists(os.path.join(d, "libmlx5.so")):
            print(f"[mori] Found libmlx5.so in {d}")
            return result

    # 5. Default: MLX5
    return result


def _copy_jit_sources(root_dir: Path) -> None:
    """Copy JIT-required source files into the package for wheel distribution.

    This creates python/mori/_jit_sources/ with the same directory structure
    as the repo root, so that get_mori_source_root() can use it as a drop-in
    replacement when the original source tree is not available.
    """
    jit_dir = root_dir / "python" / "mori" / "_jit_sources"
    if jit_dir.exists():
        shutil.rmtree(jit_dir)

    def _copytree(src, dst, **kw):
        shutil.copytree(src, dst, dirs_exist_ok=True, **kw)

    _copytree(root_dir / "include", jit_dir / "include")

    _copytree(root_dir / "src" / "ops" / "kernels", jit_dir / "src" / "ops" / "kernels")
    _copytree(root_dir / "src" / "ops" / "dispatch_combine",
              jit_dir / "src" / "ops" / "dispatch_combine")

    shmem_dst = jit_dir / "src" / "shmem"
    shmem_dst.mkdir(parents=True, exist_ok=True)
    for name in ["shmem_device_api_wrapper.cpp"]:
        src_file = root_dir / "src" / "shmem" / name
        if src_file.is_file():
            shutil.copy2(src_file, shmem_dst / name)

    for subdir in ["spdlog/include", "msgpack-c/include"]:
        src = root_dir / "3rdparty" / subdir
        if src.is_dir():
            _copytree(src, jit_dir / "3rdparty" / subdir)


class CMakeBuild(build_ext):
    def run(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as exn:
            raise RuntimeError(
                f"CMake must be installed to build the following extensions: {', '.join(e.name for e in self.extensions)}"
            ) from exn
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: Extension) -> None:
        build_lib = Path(self.build_lib)
        build_lib.mkdir(parents=True, exist_ok=True)

        root_dir = Path(__file__).parent

        if (root_dir / ".gitmodules").is_file():
            try:
                subprocess.check_call(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=str(root_dir), stdout=subprocess.DEVNULL,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        build_dir = root_dir / os.environ.get("MORI_PYBUILD_DIR", "build")
        build_dir.mkdir(parents=True, exist_ok=True)

        build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release")
        unroll_value = os.environ.get("WARP_ACCUM_UNROLL", "1")
        build_shmem_device_wrapper = os.environ.get("BUILD_SHMEM_DEVICE_WRAPPER", "ON")
        enable_profiler = os.environ.get("ENABLE_PROFILER", "OFF")
        enable_debug_printf = os.environ.get("ENABLE_DEBUG_PRINTF", "OFF")

        nic = _detect_nic_type()
        use_bnxt = nic["USE_BNXT"]
        use_ionic = nic["USE_IONIC"]
        nic_name = "BNXT" if use_bnxt == "ON" else ("IONIC" if use_ionic == "ON" else "MLX5")
        print(f"[mori] NIC auto-detection: {nic_name} (USE_BNXT={use_bnxt}, USE_IONIC={use_ionic}, USE_MLX5={'ON' if nic_name == 'MLX5' else 'OFF'})")
        enable_standard_moe_adapt = os.environ.get("ENABLE_STANDARD_MOE_ADAPT", "OFF")
        gpu_archs = _get_gpu_archs()
        print(f"[mori] GPU architecture: {gpu_archs}")
        build_examples = os.environ.get("BUILD_EXAMPLES", "OFF")
        build_tests = os.environ.get("BUILD_TESTS", "OFF")

        cmake_args = [
            "cmake",
            "-DUSE_ROCM=ON",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DWARP_ACCUM_UNROLL={unroll_value}",
            f"-DUSE_BNXT={use_bnxt}",
            f"-DBUILD_SHMEM_DEVICE_WRAPPER={build_shmem_device_wrapper}",
            f"-DUSE_IONIC={use_ionic}",
            f"-DENABLE_DEBUG_PRINTF={enable_debug_printf}",
            f"-DENABLE_STANDARD_MOE_ADAPT={enable_standard_moe_adapt}",
            f"-DGPU_TARGETS={gpu_archs}",
            f"-DENABLE_PROFILER={enable_profiler}",
            f"-DBUILD_EXAMPLES={build_examples}",
            f"-DBUILD_TESTS={build_tests}",
            "-DBUILD_TORCH_BOOTSTRAP=OFF",
            "-B",
            str(build_dir),
            "-S",
            str(root_dir),
        ]

        if shutil.which("ninja"):
            cmake_args.insert(1, "-G")
            cmake_args.insert(2, "Ninja")

        if shutil.which("ccache"):
            cmake_args.append("-DCMAKE_C_COMPILER_LAUNCHER=ccache")
            cmake_args.append("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache")

        subprocess.check_call(cmake_args)
        subprocess.check_call(
            ["cmake", "--build", ".", "-j", f"{os.cpu_count()}"], cwd=str(build_dir)
        )

        files_to_copy = [
            (
                build_dir / "src/pybind/libmori_pybinds.so",
                root_dir / "python/mori/libmori_pybinds.so",
            ),
            (
                build_dir / "src/application/libmori_application.so",
                root_dir / "python/mori/libmori_application.so",
            ),
            (
                build_dir / "src/io/libmori_io.so",
                root_dir / "python/mori/libmori_io.so",
            ),
        ]
        for src_path, dst_path in files_to_copy:
            shutil.copyfile(src_path, dst_path)

        _copy_jit_sources(root_dir)

        if os.environ.get("MORI_SKIP_PRECOMPILE", "").lower() not in ("1", "true", "on"):
            _try_precompile(root_dir)


def _try_precompile(root_dir: Path) -> None:
    """Precompile JIT kernels in the background if a GPU is detected.

    Launches a detached subprocess that compiles all .hsaco kernels and shmem
    bitcode into ~/.mori/jit/. The subprocess is fire-and-forget — pip install
    returns immediately without waiting.

    If the user starts using kernels before precompilation finishes, the JIT
    framework handles the race safely via FileBaton file locks: the user process
    either waits for the background compile to finish, or compiles the kernel
    itself (the background process will skip already-compiled kernels).
    """
    if _detect_local_gpu_arch() is None:
        print("[mori] No GPU detected — skipping kernel precompilation")
        return
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    hipcc = os.path.join(rocm_path, "bin", "hipcc")
    if not os.path.isfile(hipcc):
        print(f"[mori] hipcc not found at {hipcc} — skipping kernel precompilation")
        return
    try:
        env = os.environ.copy()
        env["MORI_PRECOMPILE"] = "1"
        env["PYTHONPATH"] = str(root_dir / "python") + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.Popen(
            [sys.executable, "-c", "import mori"],
            env=env, cwd=str(root_dir),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("[mori] Kernel precompilation started in background")
    except Exception as e:
        print(f"[mori] Precompilation skipped: {e}")


class CustomBuild(_build):
    def run(self) -> None:
        self.run_command("build_ext")
        super().run()


extensions = [
    Extension(
        "mori",
        sources=[],
        # extra_compile_args=['-ggdb', '-O0'],
        # extra_link_args=['-g'],
    ),
]

setup(
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data={
        "mori": [
            "libmori_pybinds.so",
            "libmori_io.so",
            "libmori_application.so",
            "_jit_sources/include/**/*.hpp",
            "_jit_sources/include/**/*.h",
            "_jit_sources/src/**/*.hip",
            "_jit_sources/src/**/*.hpp",
            "_jit_sources/src/**/*.cpp",
            "_jit_sources/src/**/*.h",
            "_jit_sources/3rdparty/**/*.h",
            "_jit_sources/3rdparty/**/*.hpp",
        ],
        "mori.ir": ["*.bc"],
    },
    exclude_package_data={
        "mori": ["*.a"],
    },
    cmdclass={
        "build_ext": CMakeBuild,
        "build": CustomBuild,
    },
    ext_modules=extensions,
    include_package_data=True,
)
