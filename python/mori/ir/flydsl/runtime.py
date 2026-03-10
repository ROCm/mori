# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""
FlyDSL-specific runtime helpers for mori shmem integration.

  - ``get_bitcode_path()``  — returns the path to libmori_shmem_device.bc
  - ``install_hook()``      — installs the FlyDSL post-compile hook that calls
                              ``shmem_module_init`` to inject ``globalGpuStates``
"""

from mori.ir.bitcode import find_bitcode


def get_bitcode_path() -> str:
    """Return the path to libmori_shmem_device.bc.

    Usage::

        from mori.ir.flydsl import get_bitcode_path
        bc = get_bitcode_path()
    """
    return find_bitcode()


def install_hook() -> None:
    """Install FlyDSL post-compile hook for mori shmem.

    The hook calls ``mori.shmem.shmem_module_init(hip_module)`` after each
    shmem kernel compilation so that the ``globalGpuStates`` device symbol is
    properly initialized inside the compiled GPU module.

    Call once before any shmem kernel launch::

        from mori.ir.flydsl import install_hook
        install_hook()
    """
    try:
        from flydsl.compiler import shmem_compile as sc
    except ImportError:
        raise ImportError(
            "flydsl.compiler.shmem_compile not found. "
            "Make sure FlyDSL is installed with shmem support."
        )

    def _hook(hip_module: int) -> None:
        import mori.shmem as ms
        ms.shmem_module_init(hip_module)

    sc._shmem_post_compile_hook = _hook
