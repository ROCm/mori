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
2.28; the vendored code's licenses go into the wheel's dist-info. setup.py's
environment variables (SETUPTOOLS_SCM_PRETEND_VERSION, MORI_GPU_ARCHS, ...) pass
through.
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

# Everything auditwheel may vendor comes from the gRPC build in
# docker/Dockerfile.manylinux, which collects its license texts here. A vendored
# library outside these prefixes fails the build rather than ship unlicensed.
VENDORED_PREFIXES = (
    "libabsl_",
    "libaddress_sorting",
    "libgpr",
    "libgrpc",
    "libprotobuf",
    "libre2",
    "libupb",
)
VENDORED_LICENSES = "/usr/local/share/licenses/vendored"
# gRPC and abseil: Apache-2.0. protobuf, re2, upb, address_sorting: BSD-3-Clause.
# xxhash: BSD-2-Clause. c-ares, utf8_range: MIT.
VENDORED_SPDX = ("Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "MIT")


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


def _build(bindir: str, repo: str, work: str) -> str:
    env = dict(os.environ)
    # A CMake cache remembers the Python it was configured with.
    env.setdefault(
        "MORI_PYBUILD_DIR",
        "build/manylinux-" + os.path.basename(os.path.dirname(bindir)),
    )
    env.setdefault("MORI_SKIP_PRECOMPILE", "1")
    # setuptools stages the package in build/lib.* and never empties it, so a
    # checkout that saw a `pip install .` could leak stale extensions into the wheel.
    cfg = os.path.join(work, "setuptools.cfg")
    with open(cfg, "w") as f:
        f.write(f"[build]\nbuild_base = {os.path.join(work, 'setuptools')}\n")
    env["DIST_EXTRA_CONFIG"] = cfg
    # The checkout is usually owned by someone else than the container's root,
    # and setuptools_scm silently falls back to version 0.1.0 when git refuses it.
    count = int(env.get("GIT_CONFIG_COUNT", "0"))
    env["GIT_CONFIG_COUNT"] = str(count + 1)
    env[f"GIT_CONFIG_KEY_{count}"] = "safe.directory"
    env[f"GIT_CONFIG_VALUE_{count}"] = repo
    out_dir = os.path.join(work, "built")
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


def _rewrite_wheel(wheel: str, replace=None, add=None) -> None:
    """Rewrite `wheel` in place with `replace` swapped in and `add` appended."""
    replace, add = replace or {}, add or {}
    tmp = wheel + ".tmp"
    with zipfile.ZipFile(wheel) as zin, zipfile.ZipFile(tmp, "w") as zout:
        record = next(n for n in zin.namelist() if n.endswith(".dist-info/RECORD"))
        rows = [r for r in csv.reader(io.StringIO(zin.read(record).decode())) if r]
        for row in rows:
            data = replace.get(row[0])
            if data is not None:
                row[1:] = [_record_hash(data), str(len(data))]
        at = next(i for i, row in enumerate(rows) if row[0] == record)
        rows[at:at] = [[n, _record_hash(d), str(len(d))] for n, d in add.items()]
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows(rows)

        for info in zin.infolist():
            if info.filename != record:
                data = replace.get(info.filename)
                zout.writestr(info, data if data is not None else zin.read(info))
        stamp = zin.getinfo(record).date_time
        for name, data in add.items():
            info = zipfile.ZipInfo(name, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zout.writestr(info, data)
        zout.writestr(zin.getinfo(record), buf.getvalue().encode())
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
        _rewrite_wheel(repaired, replace=replace)
    return sorted(replace)


def _add_vendored_licenses(wheel: str) -> list:
    """Ship the license texts of the code auditwheel vendored into `wheel`.

    Their licenses require the notices to travel with every binary copy. They go
    under <dist-info>/licenses/third_party/, listed by License-File, and
    License-Expression grows to cover them.
    """
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        metadata = zf.read(meta_name).decode()
    libs = [os.path.basename(n) for n in names if ".libs/" in n and n[-1] != "/"]
    if not libs:
        return []
    unknown = [lib for lib in libs if not lib.startswith(VENDORED_PREFIXES)]
    if unknown:
        sys.exit(
            f"build_manylinux_wheel: no license on file for vendored {unknown}. Collect "
            "it in docker/Dockerfile.manylinux and extend VENDORED_PREFIXES."
        )

    licenses_dir = os.path.dirname(meta_name) + "/licenses/"
    add = {}
    for root, _, files in os.walk(VENDORED_LICENSES):
        for f in files:
            path = os.path.join(root, f)
            rel = os.path.relpath(path, VENDORED_LICENSES)
            with open(path, "rb") as fh:
                add[licenses_dir + "third_party/" + rel] = fh.read()
    if not add:
        sys.exit(f"build_manylinux_wheel: no license texts under {VENDORED_LICENSES}")
    listed = sorted(n[len(licenses_dir) :] for n in add)

    head, sep, body = metadata.partition("\n\n")
    lines = head.split("\n")
    expr = next(
        i for i, line in enumerate(lines) if line.startswith("License-Expression:")
    )
    ids = [p.strip() for p in lines[expr].split(":", 1)[1].split(" AND ")]
    lines[expr] = "License-Expression: " + " AND ".join(
        ids + [i for i in VENDORED_SPDX if i not in ids]
    )
    files = [i for i, line in enumerate(lines) if line.startswith("License-File:")]
    at = (files[-1] if files else expr) + 1
    lines[at:at] = ["License-File: " + f for f in listed]
    replace = {meta_name: ("\n".join(lines) + sep + body).encode()}

    _rewrite_wheel(wheel, replace=replace, add=add)
    return listed


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
        built = _build(bindir, repo, tmp)
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
        licenses = _add_vendored_licenses(repaired)
        if licenses:
            print(f"build_manylinux_wheel: added {len(licenses)} third-party licenses")
        dest = os.path.join(args.out_dir, name)
        shutil.move(repaired, dest)
    print(f"build_manylinux_wheel: {dest}")


if __name__ == "__main__":
    main()
