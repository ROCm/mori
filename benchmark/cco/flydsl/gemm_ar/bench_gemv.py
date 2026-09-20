#!/usr/bin/env python3
"""mori's mxfp8 GEMV against sglang's, and a config sweep for mori's.

Both sides get the same raw weight and the same ue8m0 scales, each in its own
shuffle, so the only difference is the kernel. Timed with `timing.py`: amortised
(a single-call capture has a 13.4us floor on this box, and sglang's kernel is
2-5us, i.e. entirely inside it) and cold as well as hot, because a decode step
reads each layer's weight exactly once and a hot loop reads it out of LLC at
1.7x the bandwidth.

The SGLang baseline is optional (`--baseline none`); mori itself does not
depend on it.

    python bench_gemv.py --shape wq_b -m 1 --sweep
    python bench_gemv.py --shape wq_b -m 1 --config w8s2r16t16k
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import flydsl.expr as fx
import torch
from mori.ops.gemm_ar import preshuffle_b
from mori.ops.gemm_ar.kernels_gemv import compile_mxfp8_gemv

sys.path.insert(0, str(Path(__file__).parent))
import timing  # noqa: E402

MXFP8_BK = 32
SHAPES = {"wq_b": (8192, 1280), "wo_b": (5120, 2048)}

#: The space sglang tunes over, minus the configs that cannot serve the M.
WAVES = (4, 8, 16)
STEPS = (1, 2, 4)
ROWS = (16, 32)
TOKENS = (16, 32)


def configs_for(m: int):
    for w in WAVES:
        for s in STEPS:
            for r in ROWS:
                for t in TOKENS:
                    if t < m:
                        continue
                    for ks in (True, False):
                        yield {
                            "waves": w,
                            "steps": s,
                            "rows": r,
                            "tokens": t,
                            "ksplit": ks,
                        }


def key_of(c) -> str:
    return (
        f"w{c['waves']}s{c['steps']}r{c['rows']}t{c['tokens']}"
        f"{'k' if c['ksplit'] else 'n'}"
    )


def parse_key(key: str):
    import re

    g = re.fullmatch(r"w(\d+)s(\d+)r(\d+)t(\d+)([kn])", key)
    if not g:
        raise ValueError(f"bad config key {key!r}")
    return {
        "waves": int(g[1]),
        "steps": int(g[2]),
        "rows": int(g[3]),
        "tokens": int(g[4]),
        "ksplit": g[5] == "k",
    }


def build(m_max, n, k, seed=1234):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(m_max, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    w = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    ex = torch.randint(
        120, 123, (m_max, k // MXFP8_BK), generator=g, device="cuda", dtype=torch.int32
    ).to(torch.uint8)
    ew = torch.randint(
        120,
        123,
        (n // MXFP8_BK, k // MXFP8_BK),
        generator=g,
        device="cuda",
        dtype=torch.int32,
    ).to(torch.uint8)
    return x, w, ex, ew


def mori_call(cfg, n, k, m, m_max, x, w, ex, ew, out):
    gemv = compile_mxfp8_gemv(n=n, k=k, m_max=m_max, **cfg)
    w_shuf = preshuffle_b(w).contiguous().view(torch.int32).view(-1)
    ws = ew.contiguous().view(torch.int32).view(-1)
    xs = ex.contiguous().view(torch.int32).view(-1)
    xi = x.contiguous().view(torch.int32).view(-1)

    def call(picked):
        gemv(
            picked[0],
            ws,
            xi,
            xs,
            out.view(-1),
            m,
            n,
            stream=fx.Stream(torch.cuda.current_stream()),
        )

    return call, w_shuf


def sglang_call(n, k, m, x, w, ex, ew, out):
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        mxfp8_gemv,
        shuffle_mxfp8_weight,
    )

    w_shuf = shuffle_mxfp8_weight(w).contiguous()
    xv, exv = x[:m].contiguous(), ex[:m].contiguous()

    def call(picked):
        mxfp8_gemv(xv, picked[0], ew, x_scale=exv, out=out[:m])

    return call, w_shuf


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", choices=sorted(SHAPES), required=True)
    p.add_argument("-m", type=int, required=True)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--config", default=None)
    p.add_argument("--reps", type=int, default=64)
    p.add_argument("--baseline", choices=("sglang", "none"), default="sglang",
                   help="'none' drops the SGLang import; mori's own numbers "
                        "need nothing but mori")
    p.add_argument("--tol", type=float, default=2.4e-3)
    p.add_argument("--json-out", default="gemv.jsonl")
    args = p.parse_args()

    n, k = SHAPES[args.shape]
    m = args.m
    m_max = 32
    x, w, ex, ew = build(m_max, n, k)
    out = torch.zeros(m_max, n, device="cuda", dtype=torch.bfloat16)
    vram_before = timing.vram_used()

    common = {
        "bench": "gemv", "scope": "kernel", "quant": "mxfp8",
        "shape": args.shape, "n": n, "k": k, "m": m,
        "input": "fp8", "includes_quant": False,
        "timing": "amortized-graph-cold-hot",
    }
    rows, base, ref, failures = [], None, None, 0

    if args.baseline == "sglang":
        call, wt = sglang_call(n, k, m, x, w, ex, ew, out)
        base = timing.cold_hot_us(call, [wt], reps=args.reps)
        ref = out[:m].float().clone()
        print(
            f"{args.shape} M={m}  sglang    hot {base['hot_us']:6.2f}  "
            f"cold {base['cold_us']:6.2f}",
            flush=True,
        )
        rows.append(dict(common, impl="sglang", route="sglang-gemv",
                         rel_l2=0.0, validated=True, **base))

    if args.config:
        cfgs = [parse_key(args.config)]
    elif args.sweep:
        cfgs = list(configs_for(m))
    else:
        cfgs = [{"waves": 8, "steps": 2, "rows": 16, "tokens": 32, "ksplit": True}]

    for cfg in cfgs:
        try:
            cfg_m_max = cfg["tokens"]
            call, wt = mori_call(cfg, n, k, m, cfg_m_max, x, w, ex, ew, out)
            out.zero_()
            call([wt])
            torch.cuda.synchronize()
            rel = (
                ((out[:m].float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()
                if ref is not None else None
            )
            res = timing.cold_hot_us(call, [wt], reps=args.reps)
        except Exception as err:  # noqa: BLE001 - a bad config must not stop the sweep
            print(f"  {key_of(cfg):<12} FAILED {type(err).__name__}: {err}", flush=True)
            traceback.print_exc(limit=3)
            rows.append(dict(common, impl="mori", config=key_of(cfg),
                             validated=False,
                             error=f"{type(err).__name__}: {err}"))
            failures += 1
            continue
        ok = rel is None or rel <= args.tol
        failures += 0 if ok else 1
        vs = (f"  {(res['cold_us'] / base['cold_us'] - 1) * 100:+6.1f}%"
              if base else " " * 8)
        rel_s = "   n/a  " if rel is None else f"  relL2 {rel:.2e}"
        print(
            f"  {key_of(cfg):<12} hot {res['hot_us']:6.2f}  cold {res['cold_us']:6.2f}"
            f"{vs}{rel_s}{'' if ok else '  !! over tol'}",
            flush=True,
        )
        rows.append(dict(common, impl="mori", config=key_of(cfg),
                         route=f"mori-gemv-{key_of(cfg)}",
                         rel_l2=rel, validated=ok, **res))

    with open(args.json_out, "a") as f:
        for r in rows:
            f.write(json.dumps(dict(
                r, vram_before=vram_before, vram_after=timing.vram_used()
            )) + "\n")
    # A benchmark that fails and exits 0 is how a broken sweep looks green.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
