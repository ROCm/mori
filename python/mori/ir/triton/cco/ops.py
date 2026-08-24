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
"""Triton ``@core.extern`` wrappers for the CCO device API.

Functions are generated from :mod:`mori.cco.device.ops`.  Handles cross the
boundary as ``uint64`` values:

* ``dev_comm`` is :attr:`mori.cco.DevCommHandle.ptr`.
* a window is :attr:`mori.cco.RegisteredWindow.handle`.

The C++ wrapper monomorphizes CCO template axes into symbol names.  For example,
``gda_put_iw_inc`` calls ``cco_gda_put__iw__inc`` directly.
"""

import triton.language as tl
from triton.language import core

from mori.cco.device.ops import (
    CCO_DEVICE_FUNCTIONS,
    CoopScope,
    SignalOp,
    ThreadMode,
    SdmaOptFlags,
)

_LIB_NAME = "libmori_cco_device"

_TYPE_MAP = {
    "int32": tl.int32,
    "uint64": tl.uint64,
}


def _set_name(fn, name, meta):
    fn.__name__ = name
    fn.__qualname__ = name
    fn.__doc__ = f"Triton extern for ``{meta['symbol']}``."
    if hasattr(fn, "fn"):
        fn.fn.__name__ = name
        fn.fn.__qualname__ = name
        fn.fn.__doc__ = fn.__doc__


def _make_extern(name: str, meta: dict):
    symbol = meta["symbol"]
    arg_types = tuple(_TYPE_MAP[arg] for arg in meta["args"])
    ret_type = _TYPE_MAP[meta["ret"]]
    is_pure = meta.get("pure", False)

    @core.extern
    def _fn(
        *args,
        _semantic=None,
        _sym=symbol,
        _at=arg_types,
        _rt=ret_type,
        _pure=is_pure,
    ):
        cast_args = [tl.cast(arg, typ, _semantic=_semantic) for arg, typ in zip(args, _at)]
        return core.extern_elementwise(
            _LIB_NAME,
            "",
            cast_args,
            {_at: (_sym, _rt)},
            is_pure=_pure,
            _semantic=_semantic,
        )

    _set_name(_fn, name, meta)
    return _fn


def _make_extern_noargs(name: str, meta: dict):
    symbol = meta["symbol"]
    ret_type = _TYPE_MAP[meta["ret"]]
    is_pure = meta.get("pure", False)

    @core.extern
    def _fn(_semantic=None, _sym=symbol, _rt=ret_type, _pure=is_pure):
        return core.extern_elementwise(
            _LIB_NAME,
            "",
            [],
            {(): (_sym, _rt)},
            is_pure=_pure,
            _semantic=_semantic,
        )

    _set_name(_fn, name, meta)
    return _fn


def _build_all():
    return {
        name: (
            _make_extern_noargs(name, meta)
            if not meta["args"]
            else _make_extern(name, meta)
        )
        for name, meta in CCO_DEVICE_FUNCTIONS.items()
    }


_all_ops = _build_all()
globals().update(_all_ops)

__all__ = list(_all_ops) + [
    "CoopScope",
    "SignalOp",
    "ThreadMode",
    "SdmaOptFlags",
]
