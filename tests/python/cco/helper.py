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
"""Host<->device memcpy helpers for the cco python tests.

cco windows are raw device pointers in cco's own VMM (``win.local_ptr``), so
filling/reading a window goes through hipMemcpy via ``mori.jit.hip_driver``
(ctypes over libamdhip64) — dependency-free and AMD-portable.

Owned by the tests so the suite does not depend on examples/ (which is demo code
free to change); mirrors the fill/zero/read/sync helpers used by the cco
examples.
"""

import ctypes

from mori.jit.hip_driver import _get_hip_lib, _check

_H2D, _D2H = 1, 2


def sync() -> None:
    _check(_get_hip_lib().hipDeviceSynchronize(), "hipDeviceSynchronize")


def _copy(dst, src, nbytes, kind) -> None:
    _check(
        _get_hip_lib().hipMemcpy(dst, src, ctypes.c_size_t(nbytes), ctypes.c_int(kind)),
        "hipMemcpy",
    )


def fill(dev_ptr: int, values, ctype=ctypes.c_uint64) -> None:
    """Host -> window: write `values` (a sequence) into device memory at dev_ptr."""
    arr = (ctype * len(values))(*values)
    _copy(ctypes.c_void_p(dev_ptr), arr, ctypes.sizeof(arr), _H2D)


def zero(dev_ptr: int, nbytes: int) -> None:
    _copy(ctypes.c_void_p(dev_ptr), (ctypes.c_uint8 * nbytes)(), nbytes, _H2D)


def read(dev_ptr: int, count: int, ctype=ctypes.c_uint64):
    """Window -> host: read `count` elements of `ctype` from device memory."""
    arr = (ctype * count)()
    _copy(arr, ctypes.c_void_p(dev_ptr), ctypes.sizeof(arr), _D2H)
    return arr
