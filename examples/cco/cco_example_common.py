# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""Shared host-side helpers for the cco FlyDSL examples.

cco windows are raw device pointers in cco's own VMM (``win.local_ptr``), and cco
only registers memory it allocated — so there's no clean way to back a window with
a third-party GPU array object. Fill/read therefore goes through hipMemcpy:
dependency-free and AMD-portable via ``mori.jit.hip_driver`` (ctypes over
libamdhip64).
"""

import ctypes
import os

from mori.jit.hip_driver import _get_hip_lib, _check

_H2D, _D2H = 1, 2

U64 = ctypes.c_uint64
F32 = ctypes.c_float


def set_device(rank: int) -> None:
    """Select the GPU for this rank (override with CCO_GPU, e.g. for the 2-node launcher)."""
    hip = _get_hip_lib()
    n = ctypes.c_int(0)
    hip.hipGetDeviceCount(ctypes.byref(n))
    gpu = int(os.environ.get("CCO_GPU", rank % n.value))
    _check(hip.hipSetDevice(ctypes.c_int(gpu)), "hipSetDevice")


def sync() -> None:
    _check(_get_hip_lib().hipDeviceSynchronize(), "hipDeviceSynchronize")


def _copy(dst, src, nbytes, kind):
    _check(_get_hip_lib().hipMemcpy(dst, src, ctypes.c_size_t(nbytes), ctypes.c_int(kind)), "hipMemcpy")


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
