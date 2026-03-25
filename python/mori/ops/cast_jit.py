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
"""JIT-compiled cast kernel launcher (float -> fp4_e2m1).

Uses hipcc --genco to compile the kernel at first call,
then hipModuleLaunchKernel to launch from Python — no pybind11 needed.
"""

from __future__ import annotations

from mori.jit.core import compile_genco
from mori.jit.hip_driver import HipModule

_module: HipModule | None = None


def _get_module() -> HipModule:
    global _module
    if _module is None:
        hsaco = compile_genco("cast_kernel")
        _module = HipModule(hsaco)
    return _module


def cast_float_to_fp4(
    src_ptr: int,
    dst_ptr: int,
    nelems: int,
    stream: int = 0,
) -> None:
    """Launch the float->fp4_e2m1 cast kernel via HIP driver API.

    Args:
        src_ptr: Device pointer to float input.
        dst_ptr: Device pointer to fp4_e2m1 output.
        nelems: Number of elements.
        stream: HIP stream handle (as int).
    """
    mod = _get_module()
    func = mod.get_function("mori_cast_float_to_fp4")

    vec_size = 4
    threads = 256
    elems_per_block = threads * vec_size
    blocks = max(1, nelems // elems_per_block)

    func.launch(
        (blocks,),
        (threads,),
        0,
        stream,
        src_ptr,
        dst_ptr,
        nelems,
    )
