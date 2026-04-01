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
"""JIT compilation for the XGMI scatter/gather copy kernel.

Compiles src/io/kernels/scatter_gather.hip via hipcc --genco on first use.
The resulting .hsaco is loaded by C++ XgmiBackend via hipModuleLoad.
"""

from __future__ import annotations

_hsaco_path: str | None = None


def ensure_scatter_gather_kernel() -> str:
    """JIT compile the scatter/gather kernel and return the .hsaco path."""
    global _hsaco_path
    if _hsaco_path is None:
        from mori.jit.core import compile_genco

        _hsaco_path = compile_genco("scatter_gather", source_dir="src/io/kernels")
    return _hsaco_path
