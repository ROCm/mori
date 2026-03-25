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
"""Minimal ctypes wrapper around HIP driver API for loading and launching JIT-compiled kernels."""

from __future__ import annotations

import ctypes
import os
from ctypes import c_char_p, c_uint, c_void_p, POINTER, byref

_hip: ctypes.CDLL | None = None


def _get_hip_lib() -> ctypes.CDLL:
    global _hip
    if _hip is not None:
        return _hip

    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    candidates = [
        os.path.join(rocm_path, "lib", "libamdhip64.so"),
        "libamdhip64.so",
    ]
    for path in candidates:
        try:
            _hip = ctypes.CDLL(path)
            return _hip
        except OSError:
            continue

    raise OSError(
        f"libamdhip64.so not found. Is ROCm installed? Searched: {candidates}"
    )


def _check(err: int, msg: str = "") -> None:
    if err != 0:
        raise RuntimeError(f"HIP error {err}: {msg}")


class HipModule:
    """A loaded HIP code object (.hsaco) with kernel function lookup."""

    def __init__(self, hsaco_path: str):
        hip = _get_hip_lib()
        self._module = c_void_p()
        err = hip.hipModuleLoad(byref(self._module), c_char_p(hsaco_path.encode()))
        _check(err, f"hipModuleLoad({hsaco_path})")
        self._functions: dict[str, HipFunction] = {}
        self._path = hsaco_path

    def get_function(self, name: str) -> HipFunction:
        if name not in self._functions:
            hip = _get_hip_lib()
            func = c_void_p()
            err = hip.hipModuleGetFunction(
                byref(func), self._module, c_char_p(name.encode())
            )
            _check(err, f"hipModuleGetFunction({name})")
            self._functions[name] = HipFunction(func, name)
        return self._functions[name]

    def __del__(self):
        if self._module and _hip is not None:
            try:
                _hip.hipModuleUnload(self._module)
            except Exception:
                pass


class HipFunction:
    """A handle to a device kernel function, launchable via hipModuleLaunchKernel."""

    def __init__(self, func_handle: c_void_p, name: str):
        self._func = func_handle
        self._name = name

    def launch(
        self,
        grid: tuple[int, ...],
        block: tuple[int, ...],
        shared_mem: int,
        stream: int,
        *args: int | float,
    ) -> None:
        """Launch the kernel.

        ``args`` must be integers (device pointers or scalar values).
        Each arg is packed as a ``c_void_p`` (8 bytes) in a ``void**`` array.
        """
        hip = _get_hip_lib()

        n = len(args)
        ArgArray = c_void_p * n
        arg_ptrs = ArgArray()
        arg_values = []
        for i, val in enumerate(args):
            if isinstance(val, float):
                c_val = ctypes.c_double(val)
            else:
                c_val = c_void_p(val)
            arg_values.append(c_val)
            arg_ptrs[i] = ctypes.cast(ctypes.pointer(c_val), c_void_p)

        gx = grid[0] if len(grid) > 0 else 1
        gy = grid[1] if len(grid) > 1 else 1
        gz = grid[2] if len(grid) > 2 else 1
        bx = block[0] if len(block) > 0 else 1
        by = block[1] if len(block) > 1 else 1
        bz = block[2] if len(block) > 2 else 1

        err = hip.hipModuleLaunchKernel(
            self._func,
            c_uint(gx),
            c_uint(gy),
            c_uint(gz),
            c_uint(bx),
            c_uint(by),
            c_uint(bz),
            c_uint(shared_mem),
            c_void_p(stream),
            ctypes.cast(arg_ptrs, POINTER(c_void_p)),
            c_void_p(0),
        )
        _check(err, f"hipModuleLaunchKernel({self._name})")

    def launch_struct(
        self,
        grid: tuple[int, ...],
        block: tuple[int, ...],
        shared_mem: int,
        stream: int,
        struct_ptr: int,
    ) -> None:
        """Launch the kernel with a single struct argument passed by value.

        ``struct_ptr`` is a host pointer to the argument struct.
        hipModuleLaunchKernel reads the struct data from this address.
        """
        hip = _get_hip_lib()

        # kernelParams[0] must be the address of the struct data.
        # By storing struct_ptr in a c_void_p array, &params[0] == struct_ptr.
        params = (c_void_p * 1)(c_void_p(struct_ptr))

        gx = grid[0] if len(grid) > 0 else 1
        gy = grid[1] if len(grid) > 1 else 1
        gz = grid[2] if len(grid) > 2 else 1
        bx = block[0] if len(block) > 0 else 1
        by = block[1] if len(block) > 1 else 1
        bz = block[2] if len(block) > 2 else 1

        err = hip.hipModuleLaunchKernel(
            self._func,
            c_uint(gx),
            c_uint(gy),
            c_uint(gz),
            c_uint(bx),
            c_uint(by),
            c_uint(bz),
            c_uint(shared_mem),
            c_void_p(stream),
            ctypes.cast(params, POINTER(c_void_p)),
            c_void_p(0),
        )
        _check(err, f"hipModuleLaunchKernel({self._name})")


def launch_multi(
    funcs: list[c_void_p],
    grids: list[int],
    blocks: list[int],
    shared_mems: list[int],
    stream: int,
    struct_ptr: int,
) -> None:
    """Launch multiple kernels with the same struct arg in a tight loop.

    All kernels share the same stream and struct_ptr. Each entry uses a
    1-D grid and 1-D block for simplicity (covers all dispatch/combine cases).
    Minimises per-launch Python overhead by hoisting lib lookup, params
    allocation, and ctypes constants out of the loop.
    """
    hip = _get_hip_lib()
    params = (c_void_p * 1)(c_void_p(struct_ptr))
    c_params = ctypes.cast(params, POINTER(c_void_p))
    c_stream = c_void_p(stream)
    c_null = c_void_p(0)
    c_one = c_uint(1)
    launch = hip.hipModuleLaunchKernel

    for i in range(len(funcs)):
        err = launch(
            funcs[i],
            c_uint(grids[i]),
            c_one,
            c_one,
            c_uint(blocks[i]),
            c_one,
            c_one,
            c_uint(shared_mems[i]),
            c_stream,
            c_params,
            c_null,
        )
        if err != 0:
            raise RuntimeError(
                f"HIP error {err}: hipModuleLaunchKernel (batch index {i})"
            )
