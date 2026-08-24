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
"""One cco symmetric window carved into named sub-regions.

Backend-agnostic on purpose: the FlyDSL op and the C++/JIT op take offsets from
the same arena, so a kernel from either can read a peer's region. Kept out of
``flydsl_backend`` because that module imports flydsl at top level and the
C++ backend must not depend on it.
"""

import torch

from mori.tensor_utils import from_gpu_ptr


def align_up(x, a):
    return (x + a - 1) // a * a


class SymmArena:
    """One cco symmetric window carved into named, aligned sub-regions. A kernel
    reaches peer pe's copy of region R via cco.Window(handle).lsa_ptr(pe, off_R)."""

    # Every region starts here. It is not free to lower: EpScaleAlign in
    # ep_cfg.hpp is 128, and the scale transport's whole reason for padding its
    # rows is that a run of them starts on a 128 B boundary -- which only holds
    # if the region does too. Dropping this below 128 would give that back
    # silently, as a ~1.7x dispatch regression rather than as a failure.
    _ALIGN = 256

    def __init__(self, comm, regions):
        self._comm = comm
        self._offsets = {}
        self._sizes = {}
        off = 0
        for name, nbytes in regions:
            off = align_up(off, self._ALIGN)
            self._offsets[name] = off
            self._sizes[name] = nbytes
            off += nbytes
        self._total = max(align_up(off, self._ALIGN), self._ALIGN)
        self._mem = comm.alloc_mem(self._total)
        self._win = comm.register_window(self._mem.ptr, self._total)

    @property
    def handle(self):
        return self._win.handle

    @property
    def total_bytes(self):
        return self._total

    def offset(self, name):
        return self._offsets[name]

    def size(self, name):
        return self._sizes[name]

    def local_ptr(self, name):
        return self._win.local_ptr + self._offsets[name]

    def zero(self, name=None):
        """Zero the whole window, or just region `name` if given. Wraps the raw
        pointer as a zero-copy int8 torch view (borrowed via
        __cuda_array_interface__ -- no ownership taken) and memsets it."""
        if name is None:
            ptr, nbytes = self._win.local_ptr, self._total
        else:
            ptr, nbytes = self.local_ptr(name), self._sizes[name]
        from_gpu_ptr(ptr, (nbytes,), torch.int8).zero_()

    def close(self):
        """Free the symmetric window (deregister before freeing the backing mem)."""
        self._win.close()
        self._mem.close()
