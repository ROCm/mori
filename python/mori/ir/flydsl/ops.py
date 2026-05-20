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
from mori.ir.flydsl.runtime import get_bitcode_path, shmem_module_init


def _get_flydsl_extern_helpers():
    """Load FlyDSL extern helpers lazily so importing mori does not require FlyDSL."""
    try:
        from flydsl.expr.extern import ffi
        from flydsl.compiler.extern_link import link_extern

        return ffi, link_extern
    except ImportError as e:
        raise ImportError(
            "FlyDSL extern helpers not found. Make sure FlyDSL provides "
            "flydsl.expr.extern.ffi and flydsl.compiler.extern_link.link_extern."
        ) from e


def _build_all():
    """Populate module globals from MORI_DEVICE_FUNCTIONS."""
    ffi, link_extern = _get_flydsl_extern_helpers()
    bitcode_path = get_bitcode_path()
    ns = {}
    for name, meta in MORI_DEVICE_FUNCTIONS.items():
        ns[name] = link_extern(
            ffi(
                meta["symbol"],
                meta["args"],
                meta["ret"],
                is_pure=meta.get("pure", False),
            ),
            bitcode_path=bitcode_path,
            module_init_fn=shmem_module_init,
        )
    return ns


_all_ops = _build_all()
globals().update(_all_ops)

__all__ = list(_all_ops.keys()) + ["SIGNAL_SET", "SIGNAL_ADD"]
