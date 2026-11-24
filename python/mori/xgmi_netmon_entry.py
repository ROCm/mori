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
"""Entry point for xgmi-netmon, the XGMI bandwidth recorder (``tools/xgmi_netmon``).

The recorder is a standalone C++ binary shipped as ``mori/bin/xgmi-netmon``; this shim only
locates and execs it, so it behaves exactly like the binary built from tools/xgmi_netmon. It
needs no root. ``record``, ``import`` and ``list`` are subcommands of the binary.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional


def binary_path() -> Path:
    return Path(__file__).resolve().parent / "bin" / "xgmi-netmon"


def run(args: List[str]) -> int:
    exe = binary_path()
    if not exe.is_file():
        print(
            f"[mori] xgmi-netmon is not installed at {exe} "
            "(was mori built with BUILD_XGMI_NETMON=OFF?)",
            file=sys.stderr,
        )
        return 127
    os.execv(str(exe), [str(exe), *args])
    return 0  # unreachable


def main(argv: Optional[List[str]] = None) -> int:
    return run(list(sys.argv[1:] if argv is None else argv))
