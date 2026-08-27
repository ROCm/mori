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
"""Runtime helpers for linking CCO device bitcode into Triton kernels."""

from mori.cco.device.bitcode import find_cco_bitcode

# Triton's AMD backend links an extern library only when this key prefixes an
# unresolved symbol.  CCO wrapper symbols are named ``cco_*``.
_LIB_KEY = "cco"
_TRITON_COV = 5


def get_extern_libs() -> dict[str, str]:
    """Return the CCO bitcode mapping for a Triton kernel launch.

    CCO carries all device state through explicit ``dev_comm`` and window
    handles, so unlike shmem this integration does not install a post-compile
    module initialization hook.
    """

    return {_LIB_KEY: find_cco_bitcode(cov=_TRITON_COV)}


get_cco_extern_libs = get_extern_libs

__all__ = ["get_extern_libs", "get_cco_extern_libs"]
