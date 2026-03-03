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
from pathlib import Path
import shutil

from setuptools import Extension, find_packages, setup
from setuptools.command.build import build as _build
from setuptools.command.build_ext import build_ext


def _get_torch_cmake_prefix_path() -> str:
    import torch

    return torch.utils.cmake_prefix_path


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
    mori_gpu_archs = os.environ.get("MORI_GPU_ARCHS", None)
    if mori_gpu_archs:
        return mori_gpu_archs

    local_arch = _detect_local_gpu_arch()
    if local_arch:
        return local_arch

    archs = os.environ.get("PYTORCH_ROCM_ARCH", None)
    if not archs:
        import torch

        archs = torch._C._cuda_getArchFlags()

    gpu_archs = os.environ.get("GPU_ARCHS", None)
    if gpu_archs:
        archs = gpu_archs

    arch_list = archs.replace(" ", ";").split(";")

    # filter out supported architectures
    valid_arch_list = list(set(_supported_arch_list) & set(arch_list))
    if len(valid_arch_list) == 0:
        raise ValueError(
            f"no supported archs found, supported {_supported_arch_list}, got {arch_list}"
        )
    return ";".join(valid_arch_list)


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
    # Device name prefix directly maps to driver: bnxt_re_*, mlx5_*, ionic_rdma_*
    ib_dir = "/sys/class/infiniband"
    if os.path.isdir(ib_dir):
        try:
            devices = os.listdir(ib_dir)
            bnxt_count = sum(1 for d in devices if d.startswith("bnxt_re"))
            ionic_count = sum(1 for d in devices if d.startswith("ionic"))
            mlx5_count = sum(1 for d in devices if d.startswith("mlx5"))

            if bnxt_count > 0 and bnxt_count >= mlx5_count:
                result["USE_BNXT"] = "ON"
                print(f"[mori] Found {bnxt_count} BNXT RDMA device(s) in /sys/class/infiniband/")
                return result
            if ionic_count > 0 and ionic_count >= mlx5_count:
                result["USE_IONIC"] = "ON"
                print(f"[mori] Found {ionic_count} IONIC RDMA device(s) in /sys/class/infiniband/")
                return result
            if mlx5_count > 0:
                print(f"[mori] Found {mlx5_count} MLX5 RDMA device(s) in /sys/class/infiniband/")
                return result
        except OSError:
            pass

    # 3. lspci hardware detection
    _NIC_PCI_VENDORS = {
        "14e4": "BNXT",   # Broadcom BCM576xx / BCM578xx
        "1dd8": "IONIC",  # AMD/Pensando
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
    for d in _LIB_SEARCH_PATHS:
        if os.path.exists(os.path.join(d, "libbnxt_re.so")):
            result["USE_BNXT"] = "ON"
            print(f"[mori] Found libbnxt_re.so in {d}")
            return result
    for d in _LIB_SEARCH_PATHS:
        if os.path.exists(os.path.join(d, "libionic.so")):
            result["USE_IONIC"] = "ON"
            print(f"[mori] Found libionic.so in {d}")
            return result

    # 5. Default: MLX5
    return result


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
            "-B",
            str(build_dir),
            "-S",
            str(root_dir),
            f"-DCMAKE_PREFIX_PATH={_get_torch_cmake_prefix_path()}",
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
            (
                build_dir / "src/shmem/libmori_shmem.a",
                root_dir / "python/mori/libmori_shmem.a",
            ),
            (
                build_dir / "src/ops/libmori_ops.a",
                root_dir / "python/mori/libmori_ops.a",
            ),
        ]
        for src_path, dst_path in files_to_copy:
            shutil.copyfile(src_path, dst_path)

        if build_shmem_device_wrapper.upper() == "ON":
            bc_script = root_dir / "tools" / "build_shmem_bitcode.sh"
            if bc_script.exists():
                subprocess.check_call(["bash", str(bc_script)], cwd=str(root_dir))
            bc_path = root_dir / "lib" / "libmori_shmem_device.bc"
            if bc_path.exists():
                ir_dir = root_dir / "python" / "mori" / "ir"
                ir_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(bc_path, ir_dir / "libmori_shmem_device.bc")


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
    name="mori",
    use_scm_version=True,
    description="Modular RDMA Interface",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data={
        "mori": ["libmori_pybinds.so", "libmori_io.so", "libmori_application.so", "libmori_shmem.a", "libmori_ops.a"],
        "mori.ir": ["*.bc"],
    },
    cmdclass={
        "build_ext": CMakeBuild,
        "build": CustomBuild,
    },
    setup_requires=["setuptools_scm"],
    python_requires=">=3.10",
    ext_modules=extensions,
    include_package_data=True,
)
