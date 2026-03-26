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
"""
Triton-specific runtime helpers for mori shmem integration.

  - ``get_extern_libs()``  — returns the dict to pass as ``extern_libs=`` kwarg
  - ``install_hook()``     — installs the Triton post-compile hook that calls
                             ``shmem_module_init`` to inject ``globalGpuStates``
"""

from mori.ir.bitcode import find_bitcode

_LIB_KEY = "mori_shmem"


def get_extern_libs() -> dict[str, str]:
    """Return ``extern_libs`` dict for ``@triton.jit`` kernel launch.

    Usage::

        kernel[(grid,)](..., extern_libs=get_extern_libs())
    """
    return {_LIB_KEY: find_bitcode()}


def install_hook() -> None:
    """Install Triton ``jit_post_compile_hook`` for mori shmem.

    The hook calls ``mori.shmem.shmem_module_init(module)`` after each kernel
    compilation so that the ``globalGpuStates`` device symbol is properly
    initialized inside the compiled GPU module.

    Call once before any kernel launch::

        from mori.ir.triton.runtime import install_hook
        install_hook()
    """
    from triton import knobs

    def _hook(*args, **kwargs):
        key = kwargs["key"]
        jit_function = kwargs["fn"].jit_function
        device = kwargs["compile"]["device"]
        kernel = jit_function.device_caches[device][0].get(key)
        if kernel is None:
            return
        kernel._init_handles()
        import mori.shmem as ms

        ms.shmem_module_init(kernel.module)

    knobs.runtime.jit_post_compile_hook = _hook
