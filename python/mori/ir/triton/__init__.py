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
mori.ir.triton — Triton integration for mori shmem device API.

Provides ready-to-use ``@core.extern`` device functions and runtime helpers
so that Triton kernels can call mori shmem operations with minimal boilerplate.

Quick start::

    from mori.ir import triton as mori_shmem_device
    from mori.ir.triton import get_extern_libs, install_hook

    install_hook()

    @triton.jit
    def my_kernel(buf_ptr, N, BLOCK: tl.constexpr):
        pe = mori_shmem_device.my_pe()
        mori_shmem_device.putmem_nbi_block(buf_ptr, buf_ptr, N * 2, pe, 0)
        mori_shmem_device.quiet_thread()

    my_kernel[(grid,)](buf, N, BLOCK=1024, extern_libs=get_extern_libs())
"""

from .ops import *  # noqa: F401,F403 — export all device functions at package level
from .ops import __all__ as _ops_all
from .runtime import get_extern_libs, install_hook

__all__ = _ops_all + ["get_extern_libs", "install_hook"]
