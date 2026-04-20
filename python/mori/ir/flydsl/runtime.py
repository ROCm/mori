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
FlyDSL-specific runtime helpers for mori shmem integration.

  - ``get_bitcode_path()``  — returns the path to libmori_shmem_device.bc
  - ``install_hook()``      — installs the FlyDSL post-compile hook that calls
                              ``shmem_module_init`` to inject ``globalGpuStates``
"""

from mori.ir.bitcode import find_bitcode


def get_bitcode_path() -> str:
    """Return the path to libmori_shmem_device.bc (compiled with cov=6 for FlyDSL ABI).

    Usage::

        from mori.ir.flydsl import get_bitcode_path
        bc = get_bitcode_path()
    """
    return find_bitcode(cov=6)


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


def install_jit_hook() -> None:
    """Install FlyDSL JIT module-load hook for mori shmem.

    When called, registers a callback in libfly_jit_runtime.so so that
    every GPU module loaded by flyc.jit automatically gets its
    ``globalGpuStates`` initialized via ``shmem_module_init``.

    Call once before any shmem kernel launch via ``flyc.jit``::

        from mori.ir.flydsl.runtime import install_jit_hook
        install_jit_hook()

    Note: If using ``@flyc.jit`` with ``ExternFunction`` that declares
    ``mori_shmem_*`` symbols, the hook is installed automatically.
    This function provides an explicit entry point for manual control.
    """
    from flydsl.compiler.jit_executor import _ensure_shmem_hook

    _ensure_shmem_hook()
