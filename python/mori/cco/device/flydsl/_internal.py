# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""Internal plumbing for the cco FlyDSL bindings.

Two pieces:

* :func:`_ffi` — pairs a ``flydsl.expr.extern.ffi`` prototype with the cco device
  bitcode via ``link_extern`` so every binding auto-links
  ``libmori_cco_device.bc``. Construction is lazy: the bitcode is located only on
  first call (the ``.bc`` need not exist at import). cco has no global singleton
  state, so ``module_init_fn`` is always ``None``.

* :func:`cco_struct` — a method-carrying FlyDSL struct decorator. ``@fx.struct``
  rebuilds a data-only class from the field annotations (dropping methods);
  ``cco_struct`` re-attaches the user's methods/properties. Since the class
  implements ``__construct_from_ir_values__``, scf.if/for reconstruct the value
  as the *same* (method-augmented) class, so the handle keeps its behavior across
  control flow.

FlyDSL is required at import (the handles build ``fx.struct`` types).
"""

try:
    import flydsl.expr as fx
except ImportError as e:  # optional dependency
    raise ImportError(
        "cco FlyDSL bindings require FlyDSL. Install it with: pip install flydsl"
    ) from e

from mori.cco.device.bitcode import find_cco_bitcode


# ── FFI factory ──

def _load_flydsl():
    try:
        from flydsl.expr.extern import ffi
        from flydsl.compiler.extern_link import link_extern

        return ffi, link_extern
    except ImportError as e:
        raise ImportError(
            "FlyDSL not found. cco FlyDSL bindings require flydsl.expr.extern.ffi "
            "and flydsl.compiler.extern_link.link_extern."
        ) from e


class _LazyExtern:
    """Builds the linked extern on first call (defers bitcode lookup)."""

    def __init__(self, symbol, args, ret, pure=False):
        self._symbol, self._args, self._ret, self._pure = symbol, list(args), ret, pure
        self._fn = None

    def __call__(self, *args):
        if self._fn is None:
            ffi, link_extern = _load_flydsl()
            self._fn = link_extern(
                ffi(self._symbol, self._args, self._ret, is_pure=self._pure),
                bitcode_path=find_cco_bitcode(),
                module_init_fn=None,
            )
        return self._fn(*args)


def _ffi(symbol, args, ret, pure=False):
    """Lazy ``link_extern(ffi(...), bitcode_path=libmori_cco_device.bc)``."""
    return _LazyExtern(symbol, args, ret, pure=pure)


# ── method-carrying struct decorator ──

def cco_struct(klass):
    methods = {
        k: v
        for k, v in list(vars(klass).items())
        if not k.startswith("__") and (callable(v) or isinstance(v, property))
    }
    cls = fx.struct(klass)  # rebuilds a pure data class from field annotations
    for k, v in methods.items():
        setattr(cls, k, v)
    return cls
