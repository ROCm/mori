import os
import subprocess
from pathlib import Path
import shutil

from setuptools import Command, Extension, find_packages, setup
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

        subprocess.check_call(
            [
                "cmake",
                "-DUSE_ROCM=ON",
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
    install_requires=["torch"],
    python_requires=">=3.10",
    ext_modules=extensions,
    include_package_data=True,
)
