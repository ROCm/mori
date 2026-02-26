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


def _get_gpu_archs() -> str:
    archs = os.environ.get("PYTORCH_ROCM_ARCH", None)
    if not archs:
        import torch

        archs = torch._C._cuda_getArchFlags()

    gpu_archs = os.environ.get("GPU_ARCHS", None)
    if gpu_archs:
        archs = gpu_archs

    mori_gpu_archs = os.environ.get("MORI_GPU_ARCHS", None)
    if mori_gpu_archs:
        archs = mori_gpu_archs

    arch_list = archs.replace(" ", ";").split(";")

    # filter out supported architectures
    valid_arch_list = list(set(_supported_arch_list) & set(arch_list))
    if len(valid_arch_list) == 0:
        raise ValueError(
            f"no supported archs found, supported {_supported_arch_list}, got {arch_list}"
        )
    return ";".join(valid_arch_list)


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
        use_bnxt = os.environ.get("USE_BNXT", "OFF")
        use_ionic = os.environ.get("USE_IONIC", "OFF")
        enable_profiler = os.environ.get("ENABLE_PROFILER", "OFF")
        enable_debug_printf = os.environ.get("ENABLE_DEBUG_PRINTF", "OFF")
        enable_standard_moe_adapt = os.environ.get("ENABLE_STANDARD_MOE_ADAPT", "OFF")
        gpu_archs = _get_gpu_archs()
        subprocess.check_call(
            [
                "cmake",
                "-DUSE_ROCM=ON",
                f"-DCMAKE_BUILD_TYPE={build_type}",
                f"-DWARP_ACCUM_UNROLL={unroll_value}",
                f"-DUSE_BNXT={use_bnxt}",
                f"-DUSE_IONIC={use_ionic}",
                f"-DENABLE_DEBUG_PRINTF={enable_debug_printf}",
                f"-DENABLE_STANDARD_MOE_ADAPT={enable_standard_moe_adapt}",
                f"-DGPU_TARGETS={gpu_archs}",
                f"-DENABLE_PROFILER={enable_profiler}",
                "-B",
                str(build_dir),
                "-S",
                str(root_dir),
                f"-DCMAKE_PREFIX_PATH={_get_torch_cmake_prefix_path()}",
            ]
        )
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
