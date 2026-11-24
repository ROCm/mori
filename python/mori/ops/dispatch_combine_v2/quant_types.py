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
"""The combine quant types a config accepts, and the wire format each one names.

A quantized combine sends each expert row as fp8 or fp4 elements plus per-group
scales. The explicit names say the element and the group:

  fp8_blockwise_1x32   fp8 elements, one E8M0 scale per 32 elements (MXFP8)
  fp4_blockwise_1x32   e2m1 elements, one E8M0 scale per 32 elements (MXFP4)
  fp8_blockwise_1x128  fp8 elements, one f32 scale per 128 elements
  fp4_blockwise_1x128  e2m1 elements, one f32 scale per 128 elements

fp8 is the arch's own fp8 (e4m3fnuz on gfx942, e4m3fn on gfx950 and gfx125x).

``fp8_blockwise`` and ``fp4_blockwise`` name the arch's default group: 1x32 on
gfx125x and 1x128 elsewhere, which is what each meant before the explicit names
existed. ``fp4_blockwise_fp32`` is the old name of fp4_blockwise_1x128.

Light on purpose (no torch, no JIT): the config and the tuning tables both resolve
names through it.
"""

from __future__ import annotations

import struct

# Explicit name -> (element, group size). 1x32 groups carry an E8M0 scale, 1x128
# blocks an f32 one.
QUANT_FORMATS = {
    "fp8_blockwise_1x32": ("fp8", 32),
    "fp4_blockwise_1x32": ("fp4", 32),
    "fp8_blockwise_1x128": ("fp8", 128),
    "fp4_blockwise_1x128": ("fp4", 128),
}
# Names that always mean one explicit format.
_ALIASES = {"fp4_blockwise_fp32": "fp4_blockwise_1x128"}
# Names whose group follows the arch.
_ARCH_DEFAULT = {"fp8_blockwise": "fp8", "fp4_blockwise": "fp4"}

QUANT_TYPES = ("none", *_ARCH_DEFAULT, *QUANT_FORMATS, *_ALIASES)


def default_group(gfx125x: bool) -> int:
    """The group fp8_blockwise / fp4_blockwise resolve to on this arch."""
    return 32 if gfx125x else 128


def resolve_quant_type(name: str, gfx125x: bool | None) -> str:
    """The explicit name (or "none") that ``name`` means on this arch; ``gfx125x``
    is None when the arch is unknown, which only an arch-dependent name minds."""
    if name == "none" or name in QUANT_FORMATS:
        return name
    if name in _ALIASES:
        return _ALIASES[name]
    if name in _ARCH_DEFAULT:
        if gfx125x is None:
            raise ValueError(
                f"quant_type={name!r} means {name}_1x32 on gfx125x and {name}_1x128 "
                "on gfx942/gfx950, and the GPU arch could not be read; pass the "
                "explicit name"
            )
        return f"{name}_1x{default_group(gfx125x)}"
    raise ValueError(f"quant_type must be one of {QUANT_TYPES}, got {name!r}")


def names_for(formats, gfx125x: bool) -> dict[str, str]:
    """Every accepted name that resolves to one of ``formats`` (plus "none"),
    mapped to what it resolves to on this arch."""
    served = set(formats) | {"none"}
    return {
        name: resolve_quant_type(name, gfx125x)
        for name in QUANT_TYPES
        if resolve_quant_type(name, gfx125x) in served
    }


def quant_element(name: str) -> str | None:
    """ "fp8" / "fp4" for an explicit name, None for "none"."""
    return QUANT_FORMATS[name][0] if name in QUANT_FORMATS else None


def quant_group(name: str) -> int:
    """Elements per scale for an explicit name, 0 for "none"."""
    return QUANT_FORMATS[name][1] if name in QUANT_FORMATS else 0


def fp8_max(arch: str) -> float:
    """Largest finite fp8 on `arch`: 448 for OCP e4m3fn (gfx950, gfx125x), 240 for
    gfx942's e4m3fnuz and for an unknown arch, where 240 is safe on both."""
    return 448.0 if arch.startswith(("gfx95", "gfx12")) else 240.0


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def e8m0_round_up(amax: float, fmax: float) -> int:
    """Biased exponent e of a 1x32 group's E8M0 scale 2^(e-127): the smallest power
    of two with amax / scale <= fmax, clamped to [1, 254]. Bit-exact with the
    kernels, which compute it in f32 as t = amax * (1/fmax) and round t's exponent
    up past any mantissa bits."""
    t = _f32(amax * _f32(1.0 / fmax))  # an f32 product is exact in double
    bits = struct.unpack("<I", struct.pack("<f", t))[0]
    e = ((bits & 0x7FFFFFFF) + 0x7FFFFF) >> 23
    return min(max(e, 1), 254)


def is_gfx125x_arch(arch: str) -> bool:
    """Whether ``arch`` (a gcnArchName, features allowed) is gfx125x: the one test
    every arch check in this package uses, matching the JIT's fast-path prefix."""
    return arch.split(":")[0].startswith("gfx125")


def is_gfx125x() -> bool | None:
    """Whether the JIT targets gfx125x; None when the arch cannot be read."""
    try:
        from mori.jit.core import _target_arch

        arch = _target_arch()
    except Exception:
        return None
    return is_gfx125x_arch(arch) if arch else None
