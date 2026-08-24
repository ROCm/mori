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
"""Triton bindings for the CCO LSA, SDMA, and GDA device APIs.

Example::

    from mori.ir.triton import cco

    @triton.jit
    def kernel(dev_comm, window):
        rank = cco.devcomm_lsa_rank(dev_comm)
        peer_ptr = cco.lsa_ptr(window, rank, 0)

    kernel[(1,)](dc.ptr, win.handle, extern_libs=cco.get_extern_libs())
"""

from .ops import *  # noqa: F401,F403
from .ops import __all__ as _ops_all
from .runtime import get_extern_libs, get_cco_extern_libs

__all__ = _ops_all + ["get_extern_libs", "get_cco_extern_libs"]
