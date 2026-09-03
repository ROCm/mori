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
"""Input generation for the EP v2 tests and ubench: pick a distribution, pin a seed.

The names and semantics deliberately mirror aiter's ``aiter/test_common.py``
(``make_generator`` / ``fill`` / ``fill_fp8`` / ``fill_fp4``, dists
``zero|constant|uniform|norm``) so a mori number and an aiter number describe the
same input. It is a copy, not an import: mori does not depend on aiter, and
aiter's fp4 path pulls in triton.

Two things differ from aiter's, both because these tests are multi-rank:

* the generator is on the CPU, not the device. The same seed gives a different
  stream on cpu and cuda, and every mori number ever recorded came from a cpu
  generator; moving it would silently invalidate comparisons with all of them.
* seeds are per-rank (``seed_for(seed, rank)``). One seed for every rank makes
  each rank route identically, which collapses the all2all into a symmetric
  pattern and stops measuring the load imbalance that sets the dispatch time.

PAYLOAD AND ROUTING DRAW FROM SEPARATE STREAMS. The tests used to take both from
one generator in sequence, so anything that changed how much randomness the
payload consumed -- its dtype, its shape, or now its distribution -- silently
moved the routing too. That is not hypothetical: a geometry sweep read 15% apart
at one token count purely because a different SWEEP list changed the payload
shape, and so the routing. Splitting the streams makes ``DATA_INIT`` orthogonal
to routing, at the cost of the routing draw differing from pre-split runs (same
distribution, different sample).
"""

from __future__ import annotations

import os

import torch

DATA_DISTS = ("zero", "constant", "uniform", "norm")
DATA_UNIFORM = (-1.0, 1.0)
FP8_E4M3 = torch.float8_e4m3fn
FP8_UNIFORM = (-6.0, 6.0)
FP4_UNIFORM = (-3.0, 3.0)  # e2m1 tops out at 6.0; leave headroom
# Offset between the payload and routing streams. Any fixed odd number does.
_ROUTING_STREAM = 0x9E37

# OCP e2m1 magnitudes by code. Sign is bit 3, so code | 8 is the negative.
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def make_generator(seed, device="cpu"):
    """Seeded generator. Same seed, same device -> bit-identical buffers."""
    return torch.Generator(device=device).manual_seed(int(seed))


def seed_for(seed, rank, *, routing=False):
    """Per-rank seed. Ranks MUST differ or the all2all becomes symmetric."""
    return int(seed) + int(rank) + (_ROUTING_STREAM if routing else 0)


def env_config(default_init="norm", default_seed=1234):
    """(init, seed, const) from DATA_INIT / SEED / CONST_VAL.

    Defaults reproduce what these tests did before this module existed: N(0,1)
    payload, seed 1234.
    """
    init = os.environ.get("DATA_INIT", default_init)
    if init == "gaussian":
        init = "norm"
    if init not in DATA_DISTS:
        raise ValueError(f"DATA_INIT must be one of {DATA_DISTS}, got {init!r}")
    return init, int(os.environ.get("SEED", default_seed)), float(
        os.environ.get("CONST_VAL", 1.0)
    )


def verifies_nothing(init):
    """True when the distribution makes an identity-expert check vacuous.

    An all-zero payload reduces ``combine[t] == U[t] * input[t]`` to 0 == 0, which
    holds however wrong the kernel is. Callers must say so in their output rather
    than print a check that cannot fail.
    """
    return init == "zero"


def _sample_f32(shape, dist, gen, lo, hi):
    out = torch.empty(shape, dtype=torch.float32)
    if dist == "uniform":
        return out.uniform_(lo, hi, generator=gen)
    if dist == "norm":
        return out.normal_(0.0, 1.0, generator=gen)
    raise ValueError(f"{dist!r} is not a sampled distribution")


def fill(shape, dist, gen, *, dtype=torch.float32, uniform=DATA_UNIFORM, constant=1.0):
    """A DATA tensor of ``shape`` and ``dtype``.

    ``zero`` and ``constant`` ignore ``gen`` and draw no randomness -- which is
    exactly why routing must come from its own stream.
    """
    if dist == "zero":
        return torch.zeros(shape, dtype=dtype)
    if dist == "constant":
        return torch.full(shape, constant, dtype=torch.float32).to(dtype)
    return _sample_f32(shape, dist, gen, *uniform).to(dtype)


def fill_fp8(shape, dist, gen, *, uniform=FP8_UNIFORM, constant=0.5):
    """An e4m3 tensor. Sampled in a range that does not saturate the format."""
    if dist == "zero":
        return torch.zeros(shape, dtype=torch.float32).to(FP8_E4M3)
    if dist == "constant":
        return torch.full(shape, constant, dtype=torch.float32).to(FP8_E4M3)
    return _sample_f32(shape, dist, gen, *uniform).to(FP8_E4M3)


def _f32_to_e2m1_nibble(v):
    """Round f32 to the nearest OCP e2m1 code (0-7 magnitude, bit 3 = sign)."""
    levels = torch.tensor(_E2M1_LEVELS, dtype=torch.float32)
    mag = v.abs().clamp_(max=_E2M1_LEVELS[-1])
    # Midpoints between adjacent levels -> bucketize gives round-to-nearest.
    edges = (levels[1:] + levels[:-1]) / 2
    code = torch.bucketize(mag, edges).to(torch.uint8)
    return code | ((v < 0).to(torch.uint8) << 3)


def fill_fp4(shape, dist, gen, *, uniform=FP4_UNIFORM, constant=0.0):
    """MXFP4 on-wire: packed e2m1, shape ``(rows, cols // 2)`` as uint8.

    ``shape`` is the LOGICAL ``(rows, cols)``; cols must be even. Low nibble is
    the even element, matching torch's ``float4_e2m1fn_x2`` packing.

    The tests used to build fp4 with ``randint(0, 256)`` and reinterpret the
    bytes, which is a uniform draw over BIT PATTERNS, not over values -- the
    resulting magnitudes cluster wherever the e2m1 code space happens to. This
    sampling in float and converting gives a distribution you asked for.
    """
    rows, cols = shape
    if cols % 2:
        raise ValueError(f"fp4 needs an even column count, got {cols}")
    packed = (rows, cols // 2)
    if dist == "zero":
        return torch.zeros(packed, dtype=torch.uint8)
    if dist == "constant":
        n = _f32_to_e2m1_nibble(torch.full((1,), constant, dtype=torch.float32))
        return torch.full(packed, int(n[0]) | (int(n[0]) << 4), dtype=torch.uint8)
    v = _sample_f32((rows, cols), dist, gen, *uniform)
    nib = _f32_to_e2m1_nibble(v)
    return (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()


def make_payload(shape, dist, gen, wire_dtype, *, constant=1.0):
    """Dispatch payload in whatever ``wire_dtype`` the op transports.

    Returns a tensor already viewed as ``wire_dtype``; fp4 comes back as
    ``float4_e2m1fn_x2`` over the packed bytes.
    """
    if wire_dtype is torch.float4_e2m1fn_x2:
        return fill_fp4(shape, dist, gen, constant=constant).view(wire_dtype)
    if wire_dtype is FP8_E4M3:
        return fill_fp8(shape, dist, gen, constant=constant)
    return fill(shape, dist, gen, dtype=wire_dtype, constant=constant)
