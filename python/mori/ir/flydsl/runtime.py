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

For compile-time integration, prefer :func:`mori.ir.flydsl.compile_helper.prepare_compile`
which returns bitcode paths and post-load processors in a single call.
"""

import warnings

from mori.ir.bitcode import find_bitcode


def get_bitcode_path() -> str:
    """Return the path to libmori_shmem_device.bc (compiled with cov=6 for FlyDSL ABI).

    Usage::

        from mori.ir.flydsl import get_bitcode_path
        bc = get_bitcode_path()
    """
    return find_bitcode(cov=6)


def install_hook() -> None:
    """Deprecated: use ``mori.ir.flydsl.compile_helper.prepare_compile()`` instead.

    FlyDSL now uses a post-load processor model and no longer requires
    a global hook.  This function is retained for backward compatibility
    but is a no-op.
    """
    warnings.warn(
        "install_hook() is deprecated. FlyDSL now uses post-load processors "
        "via mori.ir.flydsl.compile_helper.prepare_compile().",
        DeprecationWarning,
        stacklevel=2,
    )


def install_jit_hook() -> None:
    """Deprecated: use ``mori.ir.flydsl.compile_helper.prepare_compile()`` instead.

    FlyDSL now uses a post-load processor model that initializes
    ``globalGpuStates`` per-artifact instead of via a global hook.
    This function is retained for backward compatibility but is a no-op.
    """
    warnings.warn(
        "install_jit_hook() is deprecated. FlyDSL now uses post-load processors "
        "via mori.ir.flydsl.compile_helper.prepare_compile().",
        DeprecationWarning,
        stacklevel=2,
    )
