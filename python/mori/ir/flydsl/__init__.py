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
"""
mori.ir.flydsl — FlyDSL integration for mori shmem device API.

Usage::

    from mori.ir import flydsl as mori_shmem

    @flyc.kernel
    def my_kernel(buf: fx.Tensor):
        pe = mori_shmem.my_pe()
        mori_shmem.putmem_nbi_warp(buf, buf, 64, (pe + 1) % 2, 0)
        mori_shmem.quiet_thread_pe((pe + 1) % 2)

FlyDSL automatically detects mori shmem symbols and uses
``compile_helper.prepare_compile()`` to obtain bitcode paths and
post-load processors.  No manual hook installation required.
"""

from .ops import *  # noqa: F401,F403
from .ops import __all__ as _ops_all

__all__ = list(_ops_all)
