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
"""Command-line entry point for the ``mori`` package.

Exposes the bundled environment helper shell scripts (originally under
``tools/`` in the source repo) as subcommands so users can run e.g.::

    mori check [peer_ip]
    source $(mori setup --path)

after a plain ``pip install amd_mori``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional


_SCRIPTS_DIR = Path(__file__).resolve().parent / "tools"

_SCRIPT_FILES = {
    "check": "env_check.sh",
    "setup": "env_setup.sh",
    "diagnose": "diagnose_env.sh",
}


def _script_path(name: str) -> Path:
    fname = _SCRIPT_FILES[name]
    p = _SCRIPTS_DIR / fname
    if not p.is_file():
        raise FileNotFoundError(
            f"bundled script '{fname}' not found at {p}. "
            "The mori package may have been installed incorrectly."
        )
    return p


def _ensure_bash() -> str:
    bash = shutil.which("bash")
    if not bash:
        print(
            "[mori] bash not found in PATH; cannot run shell helpers.", file=sys.stderr
        )
        sys.exit(127)
    return bash


def _run_script(name: str, args: List[str]) -> int:
    path = _script_path(name)
    # Make sure it's executable for the user; harmless if already is.
    try:
        os.chmod(path, os.stat(path).st_mode | 0o111)
    except OSError:
        pass
    bash = _ensure_bash()
    os.execv(bash, [bash, str(path), *args])
    return 0  # unreachable


def _cmd_check(rest: List[str]) -> int:
    return _run_script("check", rest)


def _cmd_diagnose(rest: List[str]) -> int:
    return _run_script("diagnose", rest)


def _cmd_setup(rest: List[str]) -> int:
    """Handle ``mori setup``.

    ``env_setup.sh`` does two things:

    1. Configures PFC / DSCP / DCQCN on the ionic NICs via ``sudo
       nicctl``. These are system-side changes that persist beyond the
       calling process — running the script as a normal subprocess is
       sufficient.
    2. Exports ``MORI_RDMA_SL`` / ``MORI_RDMA_TC`` for the calling
       shell. A Python entry point cannot mutate its parent shell's
       environment, so getting these into your current shell requires
       *sourcing* the script (e.g. ``source $(mori setup --path)``).

    Default behavior (``mori setup``): execute the script as a
    subprocess so the NIC configuration is applied. The exported
    variables will be lost once the script returns; if you need them in
    your shell, re-run via ``source $(mori setup --path)``.

    ``mori setup --path``: print only the absolute path, suitable for
    shell substitution: ``source $(mori setup --path)``.
    """
    parser = argparse.ArgumentParser(
        prog="mori setup",
        description=(
            "Run env_setup.sh to configure PFC / DSCP / DCQCN on ionic NICs. "
            "To also pick up the exported MORI_RDMA_SL / MORI_RDMA_TC variables "
            "in your current shell, source the script instead: "
            "`source $(mori setup --path)`."
        ),
        epilog=(
            "Examples:\n"
            "    mori setup                   # apply NIC config (env vars do NOT persist)\n"
            "    source $(mori setup --path)  # apply NIC config AND export MORI_RDMA_SL/TC\n"
            "    mori setup --path            # print only the absolute script path\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--path",
        action="store_true",
        help=(
            "print only the absolute path of env_setup.sh and exit "
            "(use with `source $(mori setup --path)` to export env vars)"
        ),
    )
    ns = parser.parse_args(rest)

    path = _script_path("setup")
    if ns.path:
        print(path)
        return 0

    # Run the script as a subprocess so NIC-side config (sudo nicctl ...)
    # is applied. Note the exported MORI_RDMA_SL / MORI_RDMA_TC will not
    # propagate back to the caller's shell — for that, source the script.
    print(
        "[mori] running env_setup.sh (NIC config will be applied; "
        "to also export MORI_RDMA_SL/TC into your shell run: "
        "source $(mori setup --path))",
        file=sys.stderr,
    )
    return _run_script("setup", [])


def _cmd_path(rest: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="mori path",
        description="Print the absolute path of a bundled mori script.",
    )
    parser.add_argument(
        "name",
        choices=sorted(_SCRIPT_FILES.keys()),
        help="bundled script name",
    )
    ns = parser.parse_args(rest)
    print(_script_path(ns.name))
    return 0


def _cmd_version(_rest: List[str]) -> int:
    try:
        from . import __version__  # type: ignore[attr-defined]
    except Exception:
        try:
            from ._version import __version__  # type: ignore
        except Exception:
            __version__ = "unknown"
    print(f"mori {__version__}")
    return 0


_COMMANDS = {
    "check": (_cmd_check, "run the environment check script (env_check.sh)"),
    "setup": (
        _cmd_setup,
        "run env_setup.sh (use `source $(mori setup --path)` to export env vars)",
    ),
    "diagnose": (_cmd_diagnose, "run diagnose_env.sh"),
    "path": (_cmd_path, "print the absolute path of a bundled script"),
    "version": (_cmd_version, "print mori version"),
}


def _print_usage() -> None:
    lines = [
        "usage: mori <command> [args...]",
        "",
        "Commands:",
    ]
    width = max(len(c) for c in _COMMANDS)
    for name, (_fn, desc) in _COMMANDS.items():
        lines.append(f"  {name:<{width}}  {desc}")
    lines.extend(
        [
            "",
            "Examples:",
            "  mori check 10.0.0.2          # run env_check.sh against a peer",
            "  mori setup                   # configure NICs (env vars do NOT persist in shell)",
            "  source $(mori setup --path)  # configure NICs AND export MORI_RDMA_SL/TC",
            "  mori path check              # print path to env_check.sh",
        ]
    )
    print("\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in {"-h", "--help", "help"}:
        _print_usage()
        return 0

    cmd, *rest = args
    entry = _COMMANDS.get(cmd)
    if entry is None:
        print(f"mori: unknown command '{cmd}'\n", file=sys.stderr)
        _print_usage()
        return 2

    fn, _desc = entry
    return fn(rest) or 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
