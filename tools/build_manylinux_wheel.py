#!/usr/bin/env python3
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
"""Build a mori wheel on the manylinux_2_28 baseline and repair it for PyPI.

Run inside the docker/Dockerfile.manylinux image, from a mori checkout:

    docker build -t mori-manylinux -f docker/Dockerfile.manylinux .
    docker run --rm -v $PWD:$PWD -w $PWD mori-manylinux \\
        python3 tools/build_manylinux_wheel.py --python 3.12 --out-dir dist

The wheel is built against glibc 2.28 and gcc-toolset's libstdc++. ``auditwheel
repair`` then checks that it needs nothing newer, vendors UMBP's gRPC stack into
it and tags it manylinux_2_28, so it installs on any distro at or above glibc
2.28. setup.py's environment variables (SETUPTOOLS_SCM_PRETEND_VERSION,
MORI_GPU_ARCHS, ...) pass through.
"""

import argparse
import base64
import csv
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

PLAT = "manylinux_2_28_x86_64"

# Left to the host, never vendored. The ROCm runtime has to be the one the process
# already uses -- rocm_sdk's, or the system's that mori finds through its RUNPATH
# (see python/mori/_rocm_bootstrap.py) -- and libdrm/libnuma back both it and the
# hsakmt mori links statically.
EXCLUDE = (
    "libamdhip64.so*",
    "libhsa-runtime64.so*",
    "libdrm.so*",
    "libdrm_amdgpu.so*",
    "libnuma.so*",
)


def _run(cmd, **kwargs):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def _interpreter_bindir(version: str) -> str:
    tag = "cp" + version.replace(".", "")
    bindir = f"/opt/python/{tag}-{tag}/bin"
    if not os.path.isfile(os.path.join(bindir, "python")):
        sys.exit(
            f"build_manylinux_wheel: no Python {version} at {bindir}. "
            "Run this inside the docker/Dockerfile.manylinux image."
        )
    return bindir


def _build(bindir: str, repo: str, out_dir: str) -> str:
    env = dict(os.environ)
    # setup.py does not hand CMake an interpreter: src/pybind/CMakeLists.txt takes
    # the first python3 on PATH, and the image has one per CPython version.
    env["PATH"] = bindir + os.pathsep + env["PATH"]
    # A CMake cache remembers the Python it was configured with.
    env.setdefault(
        "MORI_PYBUILD_DIR",
        "build/manylinux-" + os.path.basename(os.path.dirname(bindir)),
    )
    env.setdefault("MORI_SKIP_PRECOMPILE", "1")
    # The checkout is usually owned by someone else than the container's root,
    # and setuptools_scm silently falls back to version 0.1.0 when git refuses it.
    count = int(env.get("GIT_CONFIG_COUNT", "0"))
    env["GIT_CONFIG_COUNT"] = str(count + 1)
    env[f"GIT_CONFIG_KEY_{count}"] = "safe.directory"
    env[f"GIT_CONFIG_VALUE_{count}"] = repo
    _run(
        [
            os.path.join(bindir, "python"),
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "-w",
            out_dir,
            ".",
        ],
        cwd=repo,
        env=env,
    )
    (wheel,) = [f for f in os.listdir(out_dir) if f.endswith(".whl")]
    return os.path.join(out_dir, wheel)


def _runpaths(wheel: str, workdir: str) -> dict:
    """RUNPATH entries of every ELF member of `wheel`, keyed by member name."""
    result = {}
    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            with zf.open(name) as member:
                if member.read(4) != b"\x7fELF":
                    continue
            path = zf.extract(name, workdir)
            out = subprocess.run(
                ["patchelf", "--print-rpath", path],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            result[name] = [p for p in out.split(":") if p]
    return result


def _record_hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + digest.decode()


def _replace_members(wheel: str, replace: dict) -> None:
    """Rewrite `wheel` in place with the members in `replace` swapped in."""
    tmp = wheel + ".tmp"
    with zipfile.ZipFile(wheel) as zin, zipfile.ZipFile(tmp, "w") as zout:
        record = next(n for n in zin.namelist() if n.endswith(".dist-info/RECORD"))
        rows = list(csv.reader(io.StringIO(zin.read(record).decode())))
        for row in rows:
            if row and row[0] in replace:
                row[1:] = [_record_hash(replace[row[0]]), str(len(replace[row[0]]))]
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows(rows)
        for info in zin.infolist():
            if info.filename == record:
                data = buf.getvalue().encode()
            else:
                data = replace.get(info.filename) or zin.read(info.filename)
            zout.writestr(info, data)
    os.replace(tmp, wheel)


def _restore_runpaths(built: str, repaired: str, workdir: str) -> list:
    """Put back the RUNPATH entries auditwheel dropped from the objects it patched.

    auditwheel keeps only the entries that point inside the wheel, which takes out
    ${ROCM_PATH}/lib -- the entry a process without rocm_sdk finds ROCm through.
    """
    before = _runpaths(built, os.path.join(workdir, "built"))
    after_dir = os.path.join(workdir, "repaired")
    replace = {}
    for name, entries in _runpaths(repaired, after_dir).items():
        lost = [p for p in before.get(name, []) if p not in entries]
        if not lost:
            continue
        path = os.path.join(after_dir, name)
        _run(["patchelf", "--set-rpath", ":".join(entries + lost), path])
        with open(path, "rb") as f:
            replace[name] = f.read()
    if replace:
        _replace_members(repaired, replace)
    return sorted(replace)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--python", default="3.12", help="CPython version (default: %(default)s)"
    )
    parser.add_argument(
        "--out-dir",
        default="dist",
        help="where the repaired wheel goes (default: %(default)s)",
    )
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bindir = _interpreter_bindir(args.python)
    os.makedirs(args.out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        built = _build(bindir, repo, os.path.join(tmp, "built"))
        repaired_dir = os.path.join(tmp, "repaired")
        # --only-plat: otherwise auditwheel also adds every older tag the wheel happens
        # to meet (manylinux_2_27 today), which changes with the symbols the code uses.
        cmd = [
            "auditwheel",
            "repair",
            "--plat",
            PLAT,
            "--only-plat",
            "-w",
            repaired_dir,
        ]
        for pattern in EXCLUDE:
            cmd += ["--exclude", pattern]
        _run(cmd + [built])
        (name,) = os.listdir(repaired_dir)
        repaired = os.path.join(repaired_dir, name)
        with zipfile.ZipFile(repaired) as zf:
            vendored = [
                os.path.basename(n) for n in zf.namelist() if ".libs/libmori" in n
            ]
        if vendored:
            sys.exit(
                f"build_manylinux_wheel: auditwheel vendored a second copy of {vendored}: "
                "a packaged library's RUNPATH points into the build tree instead of $ORIGIN."
            )
        for member in _restore_runpaths(built, repaired, os.path.join(tmp, "elf")):
            print(
                f"build_manylinux_wheel: restored the RUNPATH auditwheel trimmed from {member}"
            )
        dest = os.path.join(args.out_dir, name)
        shutil.move(repaired, dest)
    print(f"build_manylinux_wheel: {dest}")


if __name__ == "__main__":
    main()
