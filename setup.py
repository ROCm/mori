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

_REQUIRED_SYSTEM_DEPS: list = []

_MPI_SYSTEM_DEPS = [
    (
        "mpicc",
        ("libopenmpi-dev", "openmpi-devel"),
        "MPI compiler wrapper (needed by CMake)",
    ),
    ("mpirun", ("openmpi-bin", "openmpi"), "MPI runtime (needed at runtime)"),
]

_REQUIRED_HEADERS = [
    (
        ["/usr/include/pci/pci.h", "/usr/include/x86_64-linux-gnu/pci/pci.h"],
        ("libpci-dev", "pciutils-devel"),
        "PCI library headers (needed for topology detection)",
    ),
    (
        ["/usr/include/infiniband/verbs.h"],
        ("libibverbs-dev", "rdma-core-devel"),
        "InfiniBand verbs headers (needed for RDMA transport)",
    ),
]


def _env_flag(name: str, default: str = "OFF") -> bool:
    return os.environ.get(name, default).strip().upper() in {
        "1",
        "ON",
        "TRUE",
        "YES",
    }


def _detect_pkg_manager() -> str:
    """Detect the system package manager."""
    if shutil.which("apt-get"):
        return "apt"
    if shutil.which("dnf"):
        return "dnf"
    if shutil.which("yum"):
        return "yum"
    return "unknown"


def _check_system_deps() -> None:
    """Verify required system packages are installed; print install hints if not."""
    missing = []

    for binary, pkgs, desc in _REQUIRED_SYSTEM_DEPS:
        if not shutil.which(binary):
            missing.append((pkgs, desc))

    for paths, pkgs, desc in _REQUIRED_HEADERS:
        if not any(os.path.isfile(p) for p in paths):
            missing.append((pkgs, desc))

    if not missing:
        return

    pm = _detect_pkg_manager()
    pkg_idx = 0 if pm == "apt" else 1

    lines = ["", "=" * 70, "[mori] Missing system dependencies:"]
    for pkgs, desc in missing:
        pkg_name = pkgs[pkg_idx] if isinstance(pkgs, tuple) else pkgs
        lines.append(f"  - {pkg_name:24s}  {desc}")
    lines.append("")

    pkg_names = [(p[pkg_idx] if isinstance(p, tuple) else p) for p, _ in missing]
    if pm == "apt":
        lines.append("  Install (Ubuntu/Debian):")
        lines.append(
            f"    sudo apt-get update && sudo apt-get install -y {' '.join(pkg_names)}"
        )
    elif pm in ("dnf", "yum"):
        lines.append("  Install (RHEL/CentOS/Fedora):")
        lines.append(f"    sudo {pm} install -y {' '.join(pkg_names)}")
    else:
        lines.append("  Install the equivalent packages for your distribution:")
        lines.append(
            f"    Ubuntu/Debian: sudo apt-get install {' '.join(p[0] if isinstance(p, tuple) else p for p, _ in missing)}"
        )
        lines.append(
            f"    RHEL/Fedora:   sudo dnf install {' '.join(p[1] if isinstance(p, tuple) else p for p, _ in missing)}"
        )
    lines.append("=" * 70)
    print("\n".join(lines), file=sys.stderr)
    raise RuntimeError(
        f"Missing system packages: {', '.join(pkg_names)}. "
        "See messages above for install instructions."
    )


def _invalidate_cmake_cache_if_changed(cmake_cache: "Path", cmake_args: list) -> None:
    """Clear CMake cache if any -DKEY=VALUE arg differs from the cached value."""
    if not cmake_cache.is_file():
        return

    # Parse -DKEY=VALUE args (normalize booleans to uppercase)
    _BOOL_MAP = {
        "1": "ON",
        "TRUE": "ON",
        "YES": "ON",
        "0": "OFF",
        "FALSE": "OFF",
        "NO": "OFF",
    }

    def _normalize(v: str) -> str:
        return _BOOL_MAP.get(v.upper(), v)

    new_opts: dict[str, str] = {}
    for arg in cmake_args:
        if arg.startswith("-D") and "=" in arg:
            key, val = arg[2:].split("=", 1)
            new_opts[key] = _normalize(val)

    # Parse CMakeCache.txt: lines like KEY:TYPE=VALUE
    cached_opts: dict[str, str] = {}
    for line in cmake_cache.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or line.startswith("//") or "=" not in line:
            continue
        key_type, val = line.split("=", 1)
        key = key_type.split(":")[0]
        cached_opts[key] = _normalize(val)

    changed = [
        k for k, v in new_opts.items() if k in cached_opts and cached_opts[k] != v
    ]

    # Also check stale CMAKE_MAKE_PROGRAM path
    make_prog = cached_opts.get("CMAKE_MAKE_PROGRAM", "")
    if make_prog and not os.path.isfile(make_prog):
        changed.append("CMAKE_MAKE_PROGRAM (no longer exists)")

    if changed:
        print(f"[mori] CMake options changed ({', '.join(changed)}), clearing cache.")
        cmake_cache.unlink()
        cmake_files = cmake_cache.parent / "CMakeFiles"
        if cmake_files.is_dir():
            shutil.rmtree(cmake_files)


def _detect_local_gpu_arch() -> str | None:
    """Auto-detect the GPU architecture on the current machine."""
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    enumerator = os.path.join(rocm_path, "bin", "rocm_agent_enumerator")
    if os.path.isfile(enumerator):
        try:
            out = subprocess.check_output([enumerator], text=True)
            for line in out.strip().split("\n"):
                line = line.strip()
                if (
                    line.startswith("gfx")
                    and line != "gfx000"
                    and line in _supported_arch_list
                ):
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

    print(
        f"[mori] No GPU arch specified — building fat binary for {_supported_arch_list}"
    )
    return ";".join(_supported_arch_list)


def _copy_jit_sources(root_dir: Path) -> None:
    """Copy JIT-required source files into the package for wheel distribution.

    This creates python/mori/_jit-sources/ with the same directory structure
    as the repo root, so that get_mori_source_root() can use it as a drop-in
    replacement when the original source tree is not available.
    """
    jit_dir = root_dir / "python" / "mori" / "_jit-sources"
    if jit_dir.exists():
        shutil.rmtree(jit_dir)

    def _copytree(src, dst, **kw):
        shutil.copytree(src, dst, dirs_exist_ok=True, **kw)

    _copytree(root_dir / "include", jit_dir / "include")

    _copytree(root_dir / "src" / "ops" / "kernels", jit_dir / "src" / "ops" / "kernels")
    _copytree(
        root_dir / "src" / "ops" / "dispatch_combine",
        jit_dir / "src" / "ops" / "dispatch_combine",
    )

    io_kernels_src = root_dir / "src" / "io" / "kernels"
    if io_kernels_src.is_dir():
        _copytree(io_kernels_src, jit_dir / "src" / "io" / "kernels")

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

    profiler_tools_src = root_dir / "tools" / "profiler"
    if profiler_tools_src.is_dir():
        _copytree(profiler_tools_src, jit_dir / "tools" / "profiler")


_3RDPARTY_DIRS = ["3rdparty/spdlog", "3rdparty/msgpack-c"]
_3RDPARTY_DIRS_UMBP = ["3rdparty/spdk"]


def _ensure_3rdparty(root_dir: Path, extra_dirs: list[str] | None = None) -> None:
    """Ensure 3rdparty submodule directories exist via git submodule update.

    Only the submodules in *required_dirs* are initialised.  Pass extra_dirs to
    opt-in to optional submodules (e.g. ``_3RDPARTY_DIRS_UMBP`` when
    BUILD_UMBP=ON).  SPDK is intentionally excluded from the default set
    because it is large and only needed for UMBP builds.
    """
    required_dirs = _3RDPARTY_DIRS + (extra_dirs or [])
    missing = [
        d
        for d in required_dirs
        if not (root_dir / d).is_dir() or not any((root_dir / d).iterdir())
    ]
    if not missing:
        return

    for d in missing:
        (root_dir / d).mkdir(parents=True, exist_ok=True)

    try:
        subprocess.check_call(
            ["git", "config", "--global", "--add", "safe.directory", str(root_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "--recursive"] + missing,
            cwd=str(root_dir),
            stdout=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    still_missing = [d for d in required_dirs if not any((root_dir / d).iterdir())]
    if still_missing:
        raise RuntimeError(
            f"Missing 3rdparty dependencies: {still_missing}. "
            "Run 'git submodule update --init --recursive' in the source directory."
        )


class CMakeBuild(build_ext):
    def run(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as exn:
            raise RuntimeError(
                "CMake is required. Install via: pip install cmake  OR  sudo apt-get install cmake"
            ) from exn
        mpi_enabled = (
            os.environ.get("BUILD_EXAMPLES", "OFF").upper() == "ON"
            or os.environ.get("MORI_WITH_MPI", "OFF").upper() == "ON"
        )
        if mpi_enabled:
            _REQUIRED_SYSTEM_DEPS.extend(_MPI_SYSTEM_DEPS)
        _check_system_deps()
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: Extension) -> None:
        build_lib = Path(self.build_lib)
        build_lib.mkdir(parents=True, exist_ok=True)

        root_dir = Path(__file__).parent

        extra = _3RDPARTY_DIRS_UMBP if _env_flag("BUILD_UMBP", "OFF") else []
        _ensure_3rdparty(root_dir, extra_dirs=extra)
        build_dir = root_dir / os.environ.get("MORI_PYBUILD_DIR", "build")
        build_dir.mkdir(parents=True, exist_ok=True)

        cmake_cache = build_dir / "CMakeCache.txt"

        build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release")
        unroll_value = os.environ.get("WARP_ACCUM_UNROLL", "1")
        build_shmem_device_wrapper = os.environ.get("BUILD_SHMEM_DEVICE_WRAPPER", "ON")
        enable_profiler = os.environ.get("ENABLE_PROFILER", "OFF")
        enable_debug_printf = os.environ.get("ENABLE_DEBUG_PRINTF", "OFF")

        enable_standard_moe_adapt = os.environ.get("ENABLE_STANDARD_MOE_ADAPT", "OFF")
        gpu_archs = _get_gpu_archs()
        print(f"[mori] GPU architecture: {gpu_archs}")
        build_examples = os.environ.get("BUILD_EXAMPLES", "OFF")
        build_benchmark = os.environ.get("BUILD_BENCHMARK", "OFF")
        build_tests = os.environ.get("BUILD_TESTS", "OFF")
        build_umbp_enabled = _env_flag("BUILD_UMBP", "OFF")
        build_umbp = "ON" if build_umbp_enabled else "OFF"
        build_xla_ffi_ops = os.environ.get("BUILD_XLA_FFI_OPS", "OFF")
        with_mpi = (
            "ON"
            if (
                build_examples.upper() == "ON"
                or os.environ.get("MORI_WITH_MPI", "OFF").upper() == "ON"
            )
            else "OFF"
        )
        build_ops_device = (
            "ON"
            if build_xla_ffi_ops.upper() == "ON"
            else os.environ.get("BUILD_OPS_DEVICE", "OFF")
        )

        cmake_args = [
            "cmake",
            "-DUSE_ROCM=ON",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DWARP_ACCUM_UNROLL={unroll_value}",
            f"-DBUILD_SHMEM_DEVICE_WRAPPER={build_shmem_device_wrapper}",
            f"-DENABLE_DEBUG_PRINTF={enable_debug_printf}",
            f"-DENABLE_STANDARD_MOE_ADAPT={enable_standard_moe_adapt}",
            f"-DGPU_TARGETS={gpu_archs}",
            f"-DENABLE_PROFILER={enable_profiler}",
            f"-DBUILD_EXAMPLES={build_examples}",
            f"-DBUILD_BENCHMARK={build_benchmark}",
            f"-DBUILD_TESTS={build_tests}",
            f"-DBUILD_UMBP={build_umbp}",
            f"-DWITH_MPI={with_mpi}",
            "-DBUILD_TORCH_BOOTSTRAP=OFF",
            f"-DBUILD_XLA_FFI_OPS={build_xla_ffi_ops}",
            f"-DBUILD_OPS_DEVICE={build_ops_device}",
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

        _invalidate_cmake_cache_if_changed(cmake_cache, cmake_args)
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
                build_dir / "src/shmem/libmori_shmem.so",
                root_dir / "python/mori/libmori_shmem.so",
            ),
            (
                build_dir / "src/ops/libmori_ops.so",
                root_dir / "python/mori/libmori_ops.so",
            ),
            (
                build_dir / "src/io/libmori_io.so",
                root_dir / "python/mori/libmori_io.so",
            ),
            (
                build_dir / "src/collective/libmori_collective.so",
                root_dir / "python/mori/libmori_collective.so",
            ),
        ]
        for src_path, dst_path in files_to_copy:
            shutil.copyfile(src_path, dst_path)

        # UMBP bindings are compiled into libmori_pybinds.so when BUILD_UMBP=ON
        # (no separate .so to copy)
        spdk_proxy_src = build_dir / "src/umbp/spdk_proxy"
        spdk_proxy_dst = root_dir / "python/mori/spdk_proxy"
        if build_umbp_enabled and spdk_proxy_src.exists():
            shutil.copyfile(spdk_proxy_src, spdk_proxy_dst)
            os.chmod(spdk_proxy_dst, 0o755)
        elif spdk_proxy_dst.exists():
            spdk_proxy_dst.unlink()

        umbp_master_src = build_dir / "src/umbp/umbp_master"
        umbp_master_dst = root_dir / "python/mori/umbp_master"
        if umbp_master_src.exists():
            shutil.copyfile(umbp_master_src, umbp_master_dst)
            os.chmod(umbp_master_dst, 0o755)
        elif umbp_master_dst.exists():
            umbp_master_dst.unlink()

        _copy_jit_sources(root_dir)

        if os.environ.get("MORI_SKIP_PRECOMPILE", "").lower() not in (
            "1",
            "true",
            "on",
        ):
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
        target_python = os.environ.get(
            "MORI_PYTHON",
            shutil.which("python3") or shutil.which("python") or sys.executable,
        )
        env = os.environ.copy()
        env["MORI_PRECOMPILE"] = "1"
        env.pop("PYTHONPATH", None)
        subprocess.Popen(
            [target_python, "-c", "import time; time.sleep(3); import mori"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[mori] Kernel precompilation started in background")
    except Exception as e:
        print(f"[mori] Precompilation skipped: {e}")


_BUNDLED_SCRIPTS = ("env_check.sh", "env_setup.sh", "diagnose_env.sh")


def _sync_bundled_scripts() -> None:
    """Copy ``tools/*.sh`` into ``python/mori/scripts/`` so they ship in the wheel.

    Keeps a single source of truth (``tools/``) while still letting the
    installed package expose them via the ``mori`` console script.
    """
    here = Path(__file__).resolve().parent
    src_dir = here / "tools"
    dst_dir = here / "python" / "mori" / "tools"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in _BUNDLED_SCRIPTS:
        src = src_dir / name
        if not src.is_file():
            continue
        dst = dst_dir / name
        try:
            if not dst.is_file() or dst.read_bytes() != src.read_bytes():
                shutil.copy2(src, dst)
            os.chmod(dst, os.stat(dst).st_mode | 0o111)
        except OSError as exc:
            print(f"[mori] WARN: failed to bundle {name}: {exc}")


_sync_bundled_scripts()


class CustomBuild(_build):
    def run(self) -> None:
        _sync_bundled_scripts()
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

mori_package_data = [
    "libmori_pybinds.so",
    "libmori_shmem.so",
    "libmori_ops.so",
    "libmori_io.so",
    "libmori_application.so",
    "libmori_collective.so",
    "spdk_proxy",
    "umbp_master",
    "_jit-sources/include/**/*.hpp",
    "_jit-sources/include/**/*.h",
    "_jit-sources/src/**/*.hip",
    "_jit-sources/src/**/*.hpp",
    "_jit-sources/src/**/*.cpp",
    "_jit-sources/src/**/*.h",
    "_jit-sources/3rdparty/**/*.h",
    "_jit-sources/3rdparty/**/*.hpp",
    "_jit-sources/tools/**/*.py",
    "ops/tuning_configs/*.json",
    "tools/*.sh",
]
if _env_flag("BUILD_UMBP", "OFF"):
    mori_package_data.append("spdk_proxy")

setup(
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    package_data={
        "mori": mori_package_data,
        "mori.ir": ["*.bc"],
        "mori.tools": ["*.sh"],
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
    entry_points={
        "console_scripts": [
            "mori = mori.cli:main",
        ],
    },
)
