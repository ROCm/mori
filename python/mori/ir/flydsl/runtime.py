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
Deprecated FlyDSL runtime helpers — retained only for backward compatibility.

All functionality has moved to :mod:`mori.ir.flydsl.compile_helper`.
Use :func:`~mori.ir.flydsl.compile_helper.prepare_compile` instead.
"""

import warnings


def get_bitcode_path() -> str:
    """Deprecated: use ``compile_helper.prepare_compile()`` instead."""
    warnings.warn(
        "get_bitcode_path() is deprecated. Use "
        "mori.ir.flydsl.compile_helper.prepare_compile() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from mori.ir.bitcode import find_bitcode
    return find_bitcode(cov=6)


def install_hook() -> None:
    """Deprecated no-op. FlyDSL uses post-load processors automatically."""
    warnings.warn(
        "install_hook() is deprecated and is a no-op.",
        DeprecationWarning,
        stacklevel=2,
    )


def install_jit_hook() -> None:
    """Deprecated no-op. FlyDSL uses post-load processors automatically."""
    warnings.warn(
        "install_jit_hook() is deprecated and is a no-op.",
        DeprecationWarning,
        stacklevel=2,
    )
