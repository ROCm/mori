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
  - ``shmem_module_init()`` — initializes ``globalGpuStates`` for a loaded
                              FlyDSL HIP module
"""

from mori.ir.bitcode import find_bitcode


def get_bitcode_path() -> str:
    """Return the path to libmori_shmem_device.bc (compiled with cov=6 for FlyDSL ABI).

    Usage::

        from mori.ir.flydsl import get_bitcode_path
        bc = get_bitcode_path()
    """
    return find_bitcode(cov=6)


def shmem_module_init(hip_module: int):
    """Initialize globalGpuStates in a FlyDSL-loaded HIP module."""
    import mori.shmem as ms

    return ms.shmem_module_init(hip_module)


def install_hook() -> None:
    """Compatibility no-op.

    Modern FlyDSL integration attaches ``shmem_module_init`` directly through
    ``link_extern(..., module_init_fn=...)`` when constructing the extern
    wrappers, so no global hook installation is required.
    """
    return None


def install_jit_hook() -> None:
    """Compatibility alias for :func:`install_hook`."""
    return install_hook()
