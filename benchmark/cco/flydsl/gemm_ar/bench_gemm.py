#!/usr/bin/env python3
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
"""mori's mxfp8/blockscale GEMM, at either scope, optionally against SGLang.

One entry point for the two questions that are *not* about the collective, and
they are different questions rather than two views of one:

``--scope kernel``
    The operands are already quantised and already padded. This is the multiply
    and nothing else, which is what a tile or a scale layout is chosen on.

``--scope linear``
    bf16 in, bf16 out: quantisation, padding and the op's own tile dispatch are
    all inside the measurement. This is what a linear layer actually costs, and
    the only scope in which comparing against SGLang means anything, because
    SGLang's route pays its own quantisation too.

``--impl`` picks what runs. ``gemm256`` and ``gemm128`` pin mori's N tile so the
dispatch itself can be measured rather than trusted; ``auto`` leaves the op to
choose. ``sglang`` is the baseline and is **optional** -- mori's own numbers
need nothing installed but mori.

    python bench_gemm.py --shape wq_b -m 4096 --scope kernel
    python bench_gemm.py --shape wq_b -m 4096 --scope linear --impl auto,sglang
    python bench_gemm.py -n 5120 -k 2048 -m 4096 --quant blockscale --scope kernel
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import flydsl.expr as fx
import torch

sys.path.insert(0, str(Path(__file__).parent))
import timing  # noqa: E402

MXFP8_BK = 32
SCALE_BK = 128

#: Every fp8 linear DeepSeek-V4.1-Flash has, read off the checkpoint and split
#: by the parallelism each is declared with. `shared_down_tp4` (5120x576) is
#: absent because SGLang's own route refuses it -- K is not a multiple of 128 --
#: so there is nothing to compare against.
SHAPES = {
    "wq_b": (8192, 1280),
    "wo_b": (5120, 2048),
    "wq_a_tp4": (1280, 5120),
    "wkv_tp4": (512, 5120),
    "wqkv_a_tp4": (1792, 5120),
    "wo_a_tp4": (2048, 4096),
    "shared_gate_up_tp4": (1152, 5120),
    "wq_b_tp8": (4096, 1280),
    "wo_b_tp8": (5120, 1024),
    "wo_a_tp8": (1024, 4096),
    "wq_b_tp1": (32768, 1280),
    "wo_b_tp1": (5120, 8192),
}

IMPLS = ("auto", "gemm256", "gemm128", "sglang")


# --------------------------------------------------------------------------
# operands
# --------------------------------------------------------------------------


class _Layer:
    """Just enough of an SGLang linear for its native route to read."""

    def __init__(self, weight, weight_scale_mx_e8m0, weight_bf16):
        self.weight = weight
        self.weight_scale_mx_e8m0 = weight_scale_mx_e8m0
        self.weight_bf16 = weight_bf16
        self.mxfp8_native_ready = True


def build_mxfp8(n, k, seed=1234, want_sglang=False):
    """One weight, in every layout either side needs, from the same bytes."""
    from mori.ops.gemm_ar import preshuffle_b

    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    eb = torch.randint(
        120,
        123,
        (n // MXFP8_BK, k // MXFP8_BK),
        generator=g,
        device="cuda",
        dtype=torch.int32,
    )
    out = {
        "w_raw": w,
        "mori_w": preshuffle_b(w),
        # The GEMM wants the B scale K-block major as dwords; the GEMV wants
        # the checkpoint's own [N/32, K/32] bytes.
        "b_scale": eb.to(torch.uint8).t().contiguous().to(torch.int32).reshape(-1),
        "w_exps": eb.to(torch.uint8),
        "layer": None,
    }
    if want_sglang:
        from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
            prepare_mxfp8_native_weight,
        )

        shuffled, scale_e8m0, weight_bf16 = prepare_mxfp8_native_weight(
            w, torch.exp2(eb.float() - 127.0), (32, 32)
        )
        out["layer"] = _Layer(
            shuffled.view(torch.float8_e4m3fn), scale_e8m0, weight_bf16
        )
    return out


def build_blockscale(n, k, m, seed=1234):
    """mori's other operand contract: A 1x128, B 128x128, fp32 scales."""
    from mori.ops.gemm_ar import preshuffle_b

    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(m, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    w = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    kb = k // SCALE_BK
    sa = (
        torch.rand(m, kb, generator=g, device="cuda", dtype=torch.float32) * 0.01 + 0.01
    )
    sb = (
        torch.rand(n // SCALE_BK, kb, generator=g, device="cuda", dtype=torch.float32)
        * 0.01
        + 0.01
    )
    return a, w, preshuffle_b(w), sa.t().contiguous().t(), sb


# --------------------------------------------------------------------------
# implementations
# --------------------------------------------------------------------------


class _Unsupported(Exception):
    """mori declines this (shape, tile) by construction, not by accident."""


def forced_gemm_op(n, k, block_n):
    """A `Mxfp8GemmOp` pinned to one N tile.

    Two things have to be got around, and both are the op behaving correctly.
    `wants_narrow_n` *is* the dispatch under test, so it is overridden rather
    than consulted; and the narrow tile is not reachable through `block_n`,
    because on the wide path `block_n` also turns on the permlane store, which
    pairs two N tiles and needs 256. 128 lives behind the dispatch as
    `NARROW_BLOCK_N`.

    The constructor also requires N to divide the *wide* tile even when only the
    narrow one will run, which is what disqualifies `shared_gate_up` (N=1152).
    Whether the narrow tile could serve it is a thing worth measuring, so for
    that case the op is built field by field. Benchmark-only: a caller has no
    business doing this.
    """
    from mori.ops.gemm_ar import Mxfp8GemmOp
    from mori.ops.gemm_ar.gemm import NARROW_BLOCK_N, _gemm_shape_constraint
    from mori.ops.gemm_ar.op import DEFAULT_BLOCK_N, MXFP8_BLOCK_M

    if block_n is None:  # 'auto': let the op dispatch
        try:
            return Mxfp8GemmOp(n=n, k=k)
        except ValueError as err:
            raise _Unsupported(str(err)) from None

    narrow = block_n == NARROW_BLOCK_N
    why = _gemm_shape_constraint(n, k, block_n)
    if why is not None:
        raise _Unsupported(why)
    if n % DEFAULT_BLOCK_N == 0:
        op = Mxfp8GemmOp(n=n, k=k)
    else:
        op = Mxfp8GemmOp.__new__(Mxfp8GemmOp)
        op.n, op.k = n, k
        op.block_m, op.block_n = MXFP8_BLOCK_M, DEFAULT_BLOCK_N
        op._launch, op._pad_in = {}, None
    op.wants_narrow_n = lambda m, _v=narrow: _v
    return op


def mori_kernel_call(ops, n, k, m, block_n, quant):
    """Pre-quantised, pre-padded operands. The multiply and nothing else.

    Rotates the weight only, because that is the only operand large enough for
    residency to matter: A is M x K and the scales are kilobytes.
    """
    if quant == "blockscale":
        from mori.ops.gemm_ar import layout
        from mori.ops.gemm_ar.kernels_fused import compile_fused_gemm_scatter

        a, _w, w_shuf, sa, sb = ops
        gemm = compile_fused_gemm_scatter(
            layout.ArConfig(world_size=2, m=128, n=n),
            0,
            K=k,
            BLOCK_M=128,
            BLOCK_N=block_n or 256,
            b_preshuffled=True,
            fuse=False,
            swap_ab=True,
            permlane=True,
            lane_transpose=True,
            quant="blockscale",
        )
        y = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
        a_i8 = a.contiguous().view(torch.int8).view(-1)
        sa_arg, sb_arg = sa.t().reshape(-1).contiguous(), sb.reshape(-1).contiguous()

        def call(picked):
            gemm(
                a_i8,
                picked[0],
                y.view(-1),
                sa_arg,
                sb_arg,
                m,
                n,
                0,
                0,
                stream=fx.Stream(torch.cuda.current_stream()),
            )
            return y

        return call, [w_shuf.contiguous().view(torch.int8).view(-1)]

    from mori.ops.gemm_ar import preshuffle_a_scale

    op = forced_gemm_op(n, k, block_n)
    g = torch.Generator(device="cuda").manual_seed(99)
    m_pad = op.padded_m(m)
    a = (torch.randn(m_pad, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    ea = torch.randint(
        120, 123, (m_pad, k // MXFP8_BK), generator=g, device="cuda", dtype=torch.int32
    )
    a_scale = preshuffle_a_scale(ea)

    def call(picked):
        return op(a, picked[0], a_scale, ops["b_scale"])[:m]

    return call, [ops["mori_w"]]


def mori_linear_call(ops, n, k, m, block_n, x_bf16):
    """bf16 in, bf16 out, through mori: quantise, pad, dispatch, multiply."""
    from sglang.srt.layers.mori_mxfp8_common import quantize_packed

    op = forced_gemm_op(n, k, block_n)
    m_pad = op.padded_m(m)

    def call(picked):
        x_in = x_bf16 if m_pad == m else op.pad_rows(x_bf16, m_pad)
        a_fp8, a_scale = quantize_packed(x_in)
        return op(a_fp8, picked[0], a_scale, ops["b_scale"])[:m]

    return call, [ops["mori_w"]]


def sglang_linear_call(layer, x_bf16):
    """SGLang's native mxfp8 linear.

    **Every weight the chosen route may read is rotated**, not just the fp8
    one. `native_route_plan` picks `hipblaslt_bf16` for most of these shapes,
    and that route reads `weight_bf16` -- which is twice the size of the fp8
    weight and therefore the tensor that decides residency. Rotating only the
    fp8 copy left the baseline hot while mori was cold, on 76 of 120 points,
    and every one of those comparisons flattered mori.
    """
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        mxfp8_native_blockscaled_linear,
    )

    weights = [layer.weight]
    if layer.weight_bf16 is not None:
        weights.append(layer.weight_bf16)

    def call(picked):
        return mxfp8_native_blockscaled_linear(
            x_bf16,
            picked[0].view(torch.uint8),
            layer.weight_scale_mx_e8m0,
            weight_bf16=(picked[1] if len(picked) > 1 else None),
        )

    return call, weights


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def rel_l2(got, ref):
    if got is None or ref is None:
        return None
    d = torch.linalg.vector_norm(got.float() - ref.float())
    return (d / torch.linalg.vector_norm(ref.float()).clamp_min(1e-30)).item()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", choices=sorted(SHAPES), default=None)
    p.add_argument("-n", type=int, default=None)
    p.add_argument("-k", type=int, default=None)
    p.add_argument("-m", type=int, required=True)
    p.add_argument("--scope", choices=("kernel", "linear"), default="linear")
    p.add_argument("--quant", choices=("mxfp8", "blockscale"), default="mxfp8")
    p.add_argument(
        "--impl",
        default="auto,gemm256,gemm128,sglang",
        help="comma-separated: " + ", ".join(IMPLS),
    )
    p.add_argument("--reps", type=int, default=32)
    p.add_argument(
        "--tol",
        type=float,
        default=2.4e-3,
        help="rel_l2 above this marks the row invalid",
    )
    p.add_argument("--json-out", default="gemm.jsonl")
    args = p.parse_args()

    if args.shape is not None:
        n, k = SHAPES[args.shape]
    elif args.n and args.k:
        n, k = args.n, args.k
        args.shape = f"{n}x{k}"
    else:
        p.error("pass --shape, or -n and -k")
    m = args.m
    impls = [i.strip() for i in args.impl.split(",") if i.strip()]
    for i in impls:
        if i not in IMPLS:
            p.error(f"unknown --impl {i!r}; pick from {IMPLS}")

    if args.quant == "blockscale" and ("sglang" in impls or args.scope == "linear"):
        p.error("--quant blockscale is mori-only and kernel-scope only")

    vram_before = timing.vram_used()
    if args.quant == "mxfp8":
        ops = build_mxfp8(n, k, want_sglang="sglang" in impls)
    else:
        ops = build_blockscale(n, k, m)
    x = (torch.randn(m, k, device="cuda") / 8).to(torch.bfloat16)

    common = {
        "bench": "gemm",
        "scope": args.scope,
        "quant": args.quant,
        "shape": args.shape,
        "n": n,
        "k": k,
        "m": m,
        "input": "bf16" if args.scope == "linear" else "fp8",
        "includes_quant": args.scope == "linear",
        "timing": "amortized-graph-cold-hot",
    }
    rows, ref, failures = [], None, 0

    # SGLang first when present, so it is the reference the rest are scored on.
    order = [i for i in impls if i == "sglang"] + [i for i in impls if i != "sglang"]
    for impl in order:
        row = dict(common, impl=impl)
        try:
            if impl == "sglang":
                if ops["layer"] is None:
                    raise RuntimeError("--impl sglang needs the sglang build path")
                from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
                    native_route_plan,
                )

                row["route"] = native_route_plan(
                    m, n, k, ops["layer"].weight_bf16 is not None, False
                )
                call, weights = sglang_linear_call(ops["layer"], x)
            else:
                block_n = {"auto": None, "gemm256": 256, "gemm128": 128}[impl]
                row["route"] = impl
                if args.scope == "kernel":
                    call, weights = mori_kernel_call(ops, n, k, m, block_n, args.quant)
                else:
                    call, weights = mori_linear_call(ops, n, k, m, block_n, x)

            got = call(weights)
            torch.cuda.synchronize()
            if ref is None:
                ref = got.float().clone()
            row["rel_l2"] = rel_l2(got, ref)
            row.update(timing.cold_hot_us(call, weights, reps=args.reps))
            row["validated"] = row["rel_l2"] is None or row["rel_l2"] <= args.tol
        except _Unsupported as err:
            # The op declining a shape it documents as out of range is a
            # *result*, not a crash: `shared_gate_up` is N=1152, which is 4.5
            # tiles of 256, so the wide tile cannot be built for it at all.
            # Counting that as a failure makes a sweep of known-good shapes
            # exit non-zero and buries a real break in the noise.
            row.update(validated=None, supported=False, reason=str(err))
            print(f"  {impl:<10} unsupported: {err}", flush=True)
        except Exception as err:  # noqa: BLE001 - one impl must not stop the rest
            row.update(validated=False, error=f"{type(err).__name__}: {err}")
            failures += 1
            print(f"  {impl:<10} FAILED {row['error']}", flush=True)
            traceback.print_exc(limit=2)
        else:
            flag = "" if row["validated"] else "  !! rel_l2 over tol"
            print(
                f"  {impl:<10} hot {row['hot_us']:8.2f}  cold {row['cold_us']:8.2f}"
                f"  relL2 {row['rel_l2']:.2e}{flag}",
                flush=True,
            )
            if not row["validated"]:
                failures += 1
        rows.append(row)

    with open(args.json_out, "a") as f:
        for r in rows:
            f.write(
                json.dumps(
                    dict(r, vram_before=vram_before, vram_after=timing.vram_used())
                )
                + "\n"
            )
    # A benchmark that fails and exits 0 is how a broken sweep looks green.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
