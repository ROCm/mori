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
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""
mori.ir — Mori device IR layer.

Provides framework-agnostic access to the shmem device bitcode and
function ABI metadata, plus framework-specific integration sub-packages
(``mori.ir.triton``, and in the future ``mori.ir.flydsl``, etc.).

Quick start (no framework dependency)::

    from mori.ir import find_bitcode
    bc = find_bitcode()   # path to libmori_shmem_device.bc
"""

from .bitcode import find_bitcode, get_bitcode_path
from .ops import MORI_DEVICE_FUNCTIONS, SIGNAL_SET, SIGNAL_ADD

__all__ = [
    "find_bitcode",
    "get_bitcode_path",
    "MORI_DEVICE_FUNCTIONS",
    "SIGNAL_SET",
    "SIGNAL_ADD",
]
