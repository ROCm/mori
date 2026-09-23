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
(no arguments)    inspects the installed package, which also covers a wheel that
                  was complete but installed wrong.
"""

import argparse
import importlib.util
import os
import sys
import tempfile
import zipfile
from typing import NoReturn

# Extensions every configuration must ship. Deliberately only the unconditional
# ones: anything behind a BUILD_* flag would make this fail on a legitimate
# partial build. mori.cco.cco qualifies -- setup.py gates it on Cython being
# importable, never on a build flag.
_REQUIRED_EXTENSIONS = ("mori/cco/cco",)


def _fail(msg: str) -> NoReturn:
    print(f"verify_install: FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def check_wheel(path: str) -> None:
    """Assert the wheel carries every required compiled extension."""
    name = os.path.basename(path)
    with zipfile.ZipFile(path) as zf:
        members = zf.namelist()

    for prefix in _REQUIRED_EXTENSIONS:
        if not any(m.startswith(prefix) and m.endswith(".so") for m in members):
            _fail(
                f"{name} ships no {prefix}*.so. The extension was skipped at "
                "build time -- check the build log for 'Cython not found' -- and "
                "the wheel is incomplete even though the build reported success."
            )
    print(f"verify_install: {name} carries every required extension")


def _leave_the_checkout() -> None:
    """Drop the repo from sys.path and cwd before importing anything.

    Inside a checkout ``python/mori/`` can answer these imports on its own, so a
    packaging miss would import fine and read as green.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    blocked = {os.getcwd(), here, repo, os.path.join(repo, "python")}
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) not in blocked]
    os.chdir(tempfile.gettempdir())


def check_install() -> None:
    """Assert the installed package exposes its extensions and EP backends."""
    _leave_the_checkout()

    import mori

    print(f"verify_install: mori {mori.__version__} from {mori.__file__}")

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
