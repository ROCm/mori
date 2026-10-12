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
"""Assert a mori install -- or a built wheel -- is complete, not merely importable.

``import mori`` says almost nothing about a build. setup.py only *warns* when a
native extension's build dependency is missing and then reports success, so an
install that quietly lost mori.cco.cco -- and with it the flydsl EP backend --
imports perfectly and passes a bare smoke check. That is exactly how a build
with no Cython shipped ``available_backends() == ['hip']``.

Two modes, because the two catch different things:

``--wheel PATH``  inspects the artifact that actually ships, before anything
                  installs it. No GPU and no ROCm needed, so it can gate a
                  publish step on the machine that built it.
(no arguments)    inspects the installed package -- regular or editable --
                  which also covers a wheel that was complete but installed
                  wrong.
"""

import argparse
import importlib.util
import os
import struct
import sys
import tempfile
import zipfile
from typing import NoReturn

# Extensions every configuration must ship. Deliberately only the unconditional
# ones: anything behind a BUILD_* flag would make this fail on a legitimate
# partial build. mori.cco.cco qualifies -- setup.py gates it on Cython being
# importable, never on a build flag.
_REQUIRED_EXTENSIONS = ("mori/cco/cco",)

# libmori_pybinds backs mori.ops/io/shmem, so it must not need a library a plain
# ROCm host lacks. gRPC/Protobuf belong to UMBP's own module (libmori_umbp_pybinds).
_PYBINDS = "mori/libmori_pybinds.so"
_UMBP_PYBINDS = "mori/libmori_umbp_pybinds.so"
_PYBINDS_FORBIDDEN_DEPS = ("libgrpc", "libgpr", "libprotobuf")


def _fail(msg: str) -> NoReturn:
    print(f"verify_install: FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _elf_needed(data: bytes) -> list:
    """DT_NEEDED entries of an ELF64 little-endian shared object."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return []
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
        for i in range(shnum)
    ]
    needed = []
    for sh in sections:
        if sh[1] != 6:  # SHT_DYNAMIC
            continue
        strtab_off = sections[sh[6]][4]
        for off in range(sh[4], sh[4] + sh[5], 16):
            tag, val = struct.unpack_from("<qQ", data, off)
            if tag == 0:  # DT_NULL
                break
            if tag == 1:  # DT_NEEDED
                end = data.index(b"\0", strtab_off + val)
                needed.append(data[strtab_off + val : end].decode())
    return needed


def check_wheel(path: str) -> None:
    """Assert the wheel carries every required compiled extension."""
    name = os.path.basename(path)
    with zipfile.ZipFile(path) as zf:
        members = zf.namelist()
        pybinds = zf.read(_PYBINDS) if _PYBINDS in members else b""

    for prefix in _REQUIRED_EXTENSIONS:
        if not any(m.startswith(prefix) and m.endswith(".so") for m in members):
            _fail(
                f"{name} ships no {prefix}*.so. The extension was skipped at "
                "build time -- check the build log for 'Cython not found' -- and "
                "the wheel is incomplete even though the build reported success."
            )

    bad = [d for d in _elf_needed(pybinds) if d.startswith(_PYBINDS_FORBIDDEN_DEPS)]
    if bad:
        _fail(
            f"{name}: {_PYBINDS} needs {bad}, so mori.ops/io/shmem would not import "
            "on a host without them. UMBP's bindings belong in libmori_umbp_pybinds."
        )
    # umbp_master ships only with BUILD_UMBP=ON, and then UMBP's module must too.
    if "mori/umbp_master" in members and _UMBP_PYBINDS not in members:
        _fail(f"{name} ships UMBP but no {_UMBP_PYBINDS}; check setup.py package_data")
    print(f"verify_install: {name} carries every required extension")


def _drop_accidental_paths() -> str:
    """Drop the paths that let a checkout answer ``import mori`` by accident.

    The cwd, ``tools/`` and the repo root are never where an installer puts the
    package, so anything resolving through them is the source tree standing in
    for an install.

    ``<repo>/python`` is deliberately kept: only ``pip install -e .`` puts it on
    the path, through its own .pth file, so dropping it would fail this check on
    every editable tree -- the layout README.md recommends for development, and
    the one the gfx1250 images ship.

    Returns the repo root so the caller can report which layout it found.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    # Subtracted, not just omitted: the editable root has to survive even when
    # it is also the cwd, and by then the two are the same string.
    blocked = {os.getcwd(), here, repo} - {os.path.join(repo, "python")}
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) not in blocked]
    os.chdir(tempfile.gettempdir())
    return repo


def check_install() -> None:
    """Assert the installed package exposes its extensions and EP backends."""
    repo = _drop_accidental_paths()

    try:
        import mori
    except ImportError as exc:
        _fail(
            f"mori is not importable ({exc}). Nothing is installed -- the repo "
            "itself is deliberately off sys.path here, so a checkout cannot "
            "stand in for an install."
        )

    # An editable install resolves inside the checkout by design, so report the
    # layout rather than treat it as suspect. Nothing below gets weaker: a
    # source tree cannot produce the compiled extension either way.
    editable = os.path.abspath(mori.__file__).startswith(repo + os.sep)
    layout = "editable" if editable else "installed"
    print(f"verify_install: mori {mori.__version__} ({layout}) from {mori.__file__}")

    # The package `mori.cco` imports cleanly from source; the submodule of the
    # same name is the compiled extension, and only it proves the build ran.
    try:
        import mori.cco.cco  # noqa: F401
    except ImportError as exc:
        _fail(
            f"mori.cco.cco is missing ({exc}). The install shipped cco.pyx but no "
            "compiled extension, so the build skipped it -- check the build log "
            "for 'Cython not found'. The flydsl EP backend goes with it."
        )

    print("verify_install: mori.cco.cco (Cython extension) imported")

    from mori.ops.dispatch_combine_v2.dispatch_combine_op import EpDispatchCombineOp

    backends = EpDispatchCombineOp.available_backends()
    print(f"verify_install: available_backends() = {backends}")
    if "hip" not in backends:
        _fail("the hip EP backend did not register")
    # flydsl is optional (docker/Dockerfile.dev --build-arg WITH_FLYDSL=1), so
    # demand the backend only where the package is actually installed. It routes
    # through mori.cco, which is what made it vanish alongside the extension.
    if importlib.util.find_spec("flydsl") and "flydsl" not in backends:
        _fail("flydsl is installed but registered no EP backend")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        metavar="PATH",
        help="check a built wheel's contents instead of the installed package",
    )
    args = parser.parse_args()

    if args.wheel:
        check_wheel(args.wheel)
    else:
        check_install()
    print("verify_install: OK")


if __name__ == "__main__":
    main()
