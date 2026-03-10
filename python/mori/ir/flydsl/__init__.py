# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""
mori.ir.flydsl — FlyDSL integration for mori shmem device API.

Quick start::

    from mori.ir import flydsl as mori_shmem
    from mori.ir.flydsl import get_bitcode_path, install_hook

    install_hook()

    @flyc.kernel
    def my_kernel(buf: fx.Tensor):
        pe = mori_shmem.my_pe()
        mori_shmem.putmem_nbi_warp(buf, buf, 64, (pe + 1) % 2, 0)
        mori_shmem.quiet_thread_pe((pe + 1) % 2)

Usage (with shmem compile helper)::

    kernel_callable = compile_shmem_kernel(my_kernel, dummy_args, chip="gfx942")
"""

from .ops import *  # noqa: F401,F403
from .ops import __all__ as _ops_all
from .runtime import get_bitcode_path, install_hook

__all__ = _ops_all + ["get_bitcode_path", "install_hook"]
