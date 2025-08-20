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
        build_dir = root_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        unroll_value = os.environ.get("WARP_ACCUM_UNROLL", "1")
        subprocess.check_call(
            [
                "cmake",
                "-DUSE_ROCM=ON",
                "-DWARP_ACCUM_UNROLL=" + unroll_value,
                "-B",
                str(build_dir),
                "-S",
                str(root_dir),
                "-DCMAKE_PREFIX_PATH=" + _get_torch_cmake_prefix_path(),
            ]
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "-j", f"{os.cpu_count()}"], cwd=str(build_dir)
        )

        shutil.copyfile(
            build_dir / "src/pybind/libmori_pybinds.so",
            root_dir / "python/mori/libmori_pybinds.so",
        )


class CustomBuild(_build):
    def run(self) -> None:
        self.run_command("build_ext")
        super().run()


extensions = [
    Extension(
        "mori",
        sources=[],
        # extra_compile_args=['-g', '-O0'],
        # extra_link_args=['-g'],
    ),
]

setup(
    name="mori",
    version="0.0.0",
    description="Modular RDMA Interface",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data={"mori": ["libmori_pybinds.so"]},
    cmdclass={
        "build_ext": CMakeBuild,
        "build": CustomBuild,
    },
    install_requires=["torch", "pytest-assume"],
    python_requires=">=3.10",
    ext_modules=extensions,
    include_package_data=True,
)
