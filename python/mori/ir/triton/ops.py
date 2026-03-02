# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""
Triton @core.extern wrappers for mori shmem device functions.

Auto-generated from ``mori.ir.ops.MORI_DEVICE_FUNCTIONS`` metadata.
Each function maps 1-to-1 to a C symbol in ``libmori_shmem_device.bc``
and is callable inside ``@triton.jit`` kernels.

Usage::

    from mori.ir.triton import ops as mt

    @triton.jit
    def my_kernel(...):
        pe = mt.my_pe()
        mt.putmem_nbi_block(dest, src, nbytes, pe, 0)
        mt.quiet_thread()
"""

import triton.language as tl
from triton.language import core

from mori.ir.ops import MORI_DEVICE_FUNCTIONS, SIGNAL_SET, SIGNAL_ADD

_LIB_NAME = "libmori_shmem_device"

_TYPE_MAP = {
    "int32": tl.int32,
    "uint32": tl.uint32,
    "int64": tl.int64,
    "uint64": tl.uint64,
    "float32": tl.float32,
    "float64": tl.float64,
}


def _set_name(fn, name):
    """Set __name__ / __qualname__ on the extern function and its inner fn."""
    fn.__name__ = name
    fn.__qualname__ = name
    if hasattr(fn, "fn"):
        fn.fn.__name__ = name
        fn.fn.__qualname__ = name


def _make_extern(name: str, meta: dict):
    """Create a ``@core.extern`` function from ABI metadata."""
    symbol = meta["symbol"]
    arg_types = tuple(_TYPE_MAP[a] for a in meta["args"])
    ret_type = _TYPE_MAP[meta["ret"]]
    is_pure = meta.get("pure", False)

    @core.extern
    def _fn(*args, _semantic=None, _sym=symbol, _at=arg_types, _rt=ret_type,
            _pure=is_pure):
        cast_args = [
            tl.cast(a, t, _semantic=_semantic) for a, t in zip(args, _at)
        ]
        return core.extern_elementwise(
            _LIB_NAME, "", cast_args,
            {_at: (_sym, _rt)},
            is_pure=_pure, _semantic=_semantic,
        )

    _set_name(_fn, name)
    return _fn


def _make_extern_noargs(name: str, meta: dict):
    """Special case for zero-argument functions (my_pe, quiet, …)."""
    symbol = meta["symbol"]
    ret_type = _TYPE_MAP[meta["ret"]]
    is_pure = meta.get("pure", False)

    @core.extern
    def _fn(_semantic=None, _sym=symbol, _rt=ret_type, _pure=is_pure):
        return core.extern_elementwise(
            _LIB_NAME, "", [],
            {(): (_sym, _rt)},
            is_pure=_pure, _semantic=_semantic,
        )

    _set_name(_fn, name)
    return _fn


def _build_all():
    """Populate module globals from MORI_DEVICE_FUNCTIONS."""
    ns = {}
    for name, meta in MORI_DEVICE_FUNCTIONS.items():
        if len(meta["args"]) == 0:
            ns[name] = _make_extern_noargs(name, meta)
        else:
            ns[name] = _make_extern(name, meta)
    return ns


_all_ops = _build_all()
globals().update(_all_ops)

__all__ = list(_all_ops.keys()) + ["SIGNAL_SET", "SIGNAL_ADD"]
