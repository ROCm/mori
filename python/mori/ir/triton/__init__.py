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
