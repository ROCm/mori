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
FlyDSL @flyc.kernel wrappers for mori shmem device functions.

Auto-generated from ``mori.ir.ops.MORI_DEVICE_FUNCTIONS`` metadata.
Each function maps 1-to-1 to a C symbol in ``libmori_shmem_device.bc``
and is callable inside ``@flyc.kernel`` functions.

Usage::

    from mori.ir import flydsl as mori_shmem

    @flyc.kernel
    def my_kernel(buf: fx.Tensor):
        pe  = mori_shmem.my_pe()
        npe = mori_shmem.n_pes()
        mori_shmem.quiet_thread_pe(pe)
"""

from mori.ir.ops import MORI_DEVICE_FUNCTIONS, SIGNAL_SET, SIGNAL_ADD


# ExternFunction is provided by FlyDSL (Part B).
# We use a lazy import so mori.ir.flydsl can be imported without FlyDSL installed;
# ExternFunction is only needed when building a @flyc.kernel.
def _get_extern_cls():
    try:
        from flydsl.compiler.extern import ExternFunction

        return ExternFunction
    except ImportError as e:
        raise ImportError(
            "flydsl.compiler.extern not found. "
            "Make sure FlyDSL is installed and flydsl/compiler/extern.py exists."
        ) from e


def _build_all():
    """Populate module globals from MORI_DEVICE_FUNCTIONS."""
    ExternFunction = _get_extern_cls()
    ns = {}
    for name, meta in MORI_DEVICE_FUNCTIONS.items():
        ns[name] = ExternFunction(
            symbol=meta["symbol"],
            arg_types=meta["args"],
            ret_type=meta["ret"],
            is_pure=meta.get("pure", False),
        )
    return ns


_all_ops = _build_all()
globals().update(_all_ops)

__all__ = list(_all_ops.keys()) + ["SIGNAL_SET", "SIGNAL_ADD"]
