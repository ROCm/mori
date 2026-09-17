#!/usr/bin/env python3
"""aiter's fp8 dense GEMM on gfx950, at wo_b's shape, for the comparison table.

aiter has no 32-wide MXFP8 dense GEMM for gfx950 -- every mxfp8 entry point
(gemm_a8w8_mxfp8, gemm_a8w8_mxfp8_128_bpreshuffle_flydsl, gemm_a8w4_mxfp8) is
gfx1250, verified by calling them here and reading the refusals, and the one
called "mxscale" is a 128x128 block scale despite the name. So the comparable
thing aiter does offer is the 128-wide block scale route, which is also what the
model dispatches today for DSV4-Pro.

Operand layouts are mori's ``--quant blockscale`` verbatim -- sa logically
[M, K/128] but **column-major**, sb [N/128, K/128] row-major, B preshuffled --
because that path was built against this kernel in the first place. So
aiter-blockscale against mori-blockscale is apples to apples, and both against
mori-mxfp8 answers what the 32-wide format is worth.

    python bench_aiter_blockscale.py -m 4096,8192,12288,16384
"""

from __future__ import annotations

import argparse
import json
import statistics

import aiter
import torch
from aiter.ops.shuffle import shuffle_weight

BK = 128


def median_us(fn, warmup: int, iters: int) -> float:
    """Graph-replay median: the timing the mori bench and bench_sglang_wob use."""
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


def run(m: int, n: int, k: int, args, out):
    g = torch.Generator(device="cuda").manual_seed(1234)
    a = (torch.randn(m, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    sa = (
        torch.rand(m, k // BK, generator=g, device="cuda", dtype=torch.float32) * 0.01
        + 0.01
    )
    sb = (
        torch.rand(n // BK, k // BK, generator=g, device="cuda", dtype=torch.float32)
        * 0.01
        + 0.01
    )
    # sa column-major, exactly what mori's make_operands hands its kernel.
    sa = sa.t().contiguous().t()
    b_shuf = shuffle_weight(b, layout=(16, 16))

    af, bf = a.float(), b.float()
    ref = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    for i in range(k // BK):
        ks = slice(i * BK, (i + 1) * BK)
        ref += (
            (af[:, ks] @ bf[:, ks].T)
            * sa[:, i][:, None]
            * sb[:, i].repeat_interleave(BK)[None, :]
        )
    rn = ref.norm()

    y = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")

    def call():
        aiter.gemm_a8w8_blockscale_bpreshuffle(a, b_shuf, sa, sb, out=y)

    call()
    rel = ((y.float() - ref).norm() / rn).item()
    t = median_us(call, args.warmup, args.iters)
    print(f"  M={m:>6}  {t:8.1f}us   relL2 {rel:.2e}", flush=True)
    out.write(
        json.dumps(
            {
                "route": "aiter-blockscale",
                "m": m,
                "n": n,
                "k": k,
                "us": t,
                "rel_l2": rel,
            }
        )
        + "\n"
    )
    out.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", default="4096,8192,12288,16384")
    p.add_argument("-n", type=int, default=5120)
    p.add_argument("-k", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=31)
    p.add_argument("--json-out", default="/workspace/dsv41/aiter_blockscale.jsonl")
    args = p.parse_args()

    print(f"aiter gemm_a8w8_blockscale_bpreshuffle  N={args.n} K={args.k}", flush=True)
    with open(args.json_out, "a") as out:
        for m in (int(v) for v in args.m.split(",")):
            run(m, args.n, args.k, args, out)


if __name__ == "__main__":
    main()
