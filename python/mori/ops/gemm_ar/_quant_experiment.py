# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Quantization helpers kept outside the mutually dependent kernel factories."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import range_constexpr


def _raw_value(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _bpermute_f32(value, src_lane):
    """``ds_bpermute`` on an f32, through its i32 bit pattern. Byte-addressed."""
    i32 = ir.IntegerType.get_signless(32)
    raw = fx.rocdl.ds_bpermute(
        i32,
        _raw_value(fx.Int32(src_lane) * fx.Int32(4)),
        _raw_value(value.bitcast(fx.Int32)),
    )
    return fx.Int32(raw).bitcast(fx.Float32)


def _pack8_fp8(v8):
    """Eight scaled f32 -> eight e4m3 bytes, as a ``vector<2xi32>``.

    ``v_cvt_pk_fp8_f32`` rather than the one-shot ``pk8`` form: the latter is
    gfx1250, and this has to run on gfx950. Four instructions instead of one,
    which is nothing in a kernel this memory-bound. Note that
    ``arith.truncf`` to fp8 is *not* an option -- it has no LLVM lowering and
    fails in the translation pass, not at trace time.

    ``old`` threads the two halves of a dword through one register: word_sel=0
    writes the low 16 bits, word_sel=1 the high.
    """
    i32 = ir.IntegerType.get_signless(32)
    poison = fx.Int32(0)
    words = []
    for half in range_constexpr(2):
        acc = _raw_value(poison)
        for pair in range_constexpr(2):
            i = half * 4 + pair * 2
            acc = fx.rocdl.cvt_pk_fp8_f32(
                i32,
                _raw_value(v8[i]),
                _raw_value(v8[i + 1]),
                acc,
                pair == 1,
            )
        words.append(fx.Int32(acc))
    return fx.Vector.from_elements(words, fx.Int32)


def _unpack8_fp8(v2i32):
    """Eight e4m3 bytes in a ``vector<2xi32>`` -> eight f32. The inverse."""
    f32x2 = ir.VectorType.get([2], ir.F32Type.get())
    out = []
    for half in range_constexpr(2):
        word = _raw_value(v2i32[half])
        for sel in range_constexpr(2):
            pair = fx.Vector(fx.rocdl.cvt_pk_f32_fp8(f32x2, word, sel == 1))
            out.append(pair[0])
            out.append(pair[1])
    return out
