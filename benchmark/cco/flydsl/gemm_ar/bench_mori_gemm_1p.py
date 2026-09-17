#!/usr/bin/env python3
"""mori's GEMM alone, single process, to match how sglang and aiter are timed.

``bench_gemm_ar.py`` cannot run below world_size=2 -- ``ArConfig`` rejects it --
so every mori number so far is a max over 2-8 ranks, while bench_sglang_wob.py
and bench_aiter_blockscale.py are one process. The gap is small (world 2 vs 8
differs 0-4%) but it is a real methodology difference and it favours the
competition, so this removes it.

The GEMM does not touch the symmetric window when the epilogue tail is compiled
out (``fuse=False``), which is why the unit tests already call it this way: a
config is still needed for the shape, but the device-comm and window handles are
passed as 0.

    python bench_mori_gemm_1p.py --quant mxfp8 -m 4096,8192,12288,16384
"""

from __future__ import annotations

import argparse
import json
import statistics

import flydsl.expr as fx
import torch
from mori.ops.gemm_ar import compile_fused_gemm_scatter, layout, preshuffle_b

SCALE_BK = 128
MXFP8_BK = 32


def median_us(fn, warmup: int, iters: int) -> float:
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    return statistics.median(ts)


def build(m, n, k, quant):
    """Operands and the fp32 reference, matching bench_gemm_ar.py's recipe."""
    g = torch.Generator(device="cuda").manual_seed(1234)
    a = (torch.randn(m, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    af, bf = a.float(), b.float()

    if quant == "mxfp8":
        bk = MXFP8_BK
        ea = torch.randint(
            120, 123, (m, k // bk), generator=g, device="cuda", dtype=torch.int32
        )
        eb = torch.randint(
            120, 123, (n // bk, k // bk), generator=g, device="cuda", dtype=torch.int32
        )
        sav, sbv = torch.exp2(ea.float() - 127.0), torch.exp2(eb.float() - 127.0)
        # K-block major, the coalesced layout the kernel indexes
        sa_arg = ea.t().reshape(-1).contiguous()
        sb_arg = eb.t().reshape(-1).contiguous()
    else:
        bk = SCALE_BK
        sav = (
            torch.rand(m, k // bk, generator=g, device="cuda", dtype=torch.float32)
            * 0.01
            + 0.01
        )
        sbv = (
            torch.rand(
                n // bk, k // bk, generator=g, device="cuda", dtype=torch.float32
            )
            * 0.01
            + 0.01
        )
        sav = sav.t().contiguous().t()  # A scale column-major
        sa_arg = sav.t().reshape(-1).contiguous()
        sb_arg = sbv.reshape(-1).contiguous()

    ref = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    for i in range(k // bk):
        ks = slice(i * bk, (i + 1) * bk)
        ref += (
            (af[:, ks] @ bf[:, ks].T)
            * sav[:, i][:, None]
            * sbv[:, i].repeat_interleave(bk)[None, :]
        )
    return a, preshuffle_b(b), sa_arg, sb_arg, ref


def make_call(gemm, a_i8, b_i8, y, sa_arg, sb_arg, m, n):
    """Bind one shape's operands into a nullary callable.

    A factory rather than a closure written inline in the loop: the loop rebinds
    every one of these names each iteration, so an inline closure would read
    whatever the last iteration left behind if it ever outlived the iteration
    that made it. It does not today, but that is a property of the call site,
    not of the closure.
    """

    def call():
        gemm(
            a_i8,
            b_i8,
            y.view(-1),
            sa_arg,
            sb_arg,
            m,
            n,
            0,
            0,
            stream=fx.Stream(torch.cuda.current_stream()),
        )

    return call


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quant", choices=("mxfp8", "blockscale"), default="mxfp8")
    p.add_argument("-m", default="4096,8192,12288,16384")
    p.add_argument("-n", type=int, default=5120)
    p.add_argument("-k", type=int, default=2048)
    p.add_argument("--block-m", type=int, default=None)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=31)
    p.add_argument("--json-out", default="/workspace/dsv41/mori_1p.jsonl")
    args = p.parse_args()

    block_m = args.block_m or (128 if args.quant == "blockscale" else 256)
    print(
        f"mori {args.quant} single process  N={args.n} K={args.k} BLOCK_M={block_m}",
        flush=True,
    )
    with open(args.json_out, "a") as out:
        for m in (int(v) for v in args.m.split(",")):
            a, b_shuf, sa_arg, sb_arg, ref = build(m, args.n, args.k, args.quant)
            cfg = layout.ArConfig(world_size=2, m=m, n=args.n)
            gemm = compile_fused_gemm_scatter(
                cfg,
                0,
                K=args.k,
                BLOCK_M=block_m,
                BLOCK_N=256,
                b_preshuffled=True,
                fuse=False,
                swap_ab=True,
                permlane=True,
                lane_transpose=True,
                quant=args.quant,
            )
            y = torch.zeros(m, args.n, device="cuda", dtype=torch.bfloat16)
            a_i8 = a.contiguous().view(torch.int8).view(-1)
            b_i8 = b_shuf.contiguous().view(torch.int8).view(-1)

            call = make_call(gemm, a_i8, b_i8, y, sa_arg, sb_arg, m, args.n)
            call()
            torch.cuda.synchronize()
            rel = ((y.float() - ref).norm() / ref.norm()).item()
            t = median_us(call, args.warmup, args.iters)
            print(f"  M={m:>6}  {t:8.1f}us   relL2 {rel:.2e}", flush=True)
            out.write(
                json.dumps(
                    {
                        "route": f"mori-{args.quant}-1p",
                        "m": m,
                        "n": args.n,
                        "k": args.k,
                        "us": t,
                        "rel_l2": rel,
                    }
                )
                + "\n"
            )
            out.flush()


if __name__ == "__main__":
    main()
