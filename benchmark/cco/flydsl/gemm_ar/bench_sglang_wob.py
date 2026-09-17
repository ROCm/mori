#!/usr/bin/env python3
"""sglang's two real routes for DeepSeek-V4.1-Flash's ``wo_b``, at mori's shapes.

The mori ``gemm_ar`` benchmark answers "how fast is our GEMM"; this answers
"against what". Two routes matter, and they are not the same thing:

  prod-bf16    what the model runs today. A trace of the running server shows
               ``_fake_quant_fp8_kernel`` (21.6us) then a hipBLASLt
               ``Cijk_..._BBS_...`` (212.5us) -- i.e. the activation is rounded
               onto the fp8 grid and the GEMM itself is **bf16**, over a weight
               dequantised back from fp8 at load. Nothing runs in fp8.
  native-mxfp8 the route sglang has but wo_b does not take: fp8 e4m3 x fp8 e4m3
               with per-32 ue8m0 scales on both sides, over the shuffled weight.
               ``bf16-in`` quantises the activation inside the call; ``fp8-in``
               takes it pre-quantised, which is the operand form mori's kernel
               gets and therefore the like-for-like column.

Single process: this is a dense GEMM, no collective, so there is nothing to
distribute. Timing matches the mori bench -- CUDA graph replay, median of N --
so the two sets of numbers can go in one table.

    python bench_sglang_wob.py --shapes 5120x2048 -m 4096,8192,11617,16384
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys

import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    bf16_dequant_blockscaled_linear,
    dequant_block_fp8_weight_to_bf16,
    fake_quant_fp8_activation,
    mxfp8_e4m3_quantize,
)
from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
    mxfp8_native_blockscaled_linear,
    native_route_supports,
    prepare_mxfp8_native_weight,
)

BLOCK = (32, 32)


def gpu_occupancy():
    """Held VRAM per card, so a contended measurement can be thrown out later."""
    try:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except Exception:
        return None
    used = []
    for line in out.splitlines()[1:]:
        f = line.split(",")
        if len(f) >= 3 and f[0].startswith("card"):
            try:
                used.append(round(int(f[2]) / 2**30, 1))
            except ValueError:
                pass
    return used


def median_us(fn, warmup: int, iters: int) -> float:
    """Graph-replay median, the same shape of measurement the mori bench takes."""
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


def make_weight(n: int, k: int, g):
    """fp8 e4m3 ``[N, K]`` with power-of-two fp32 block scales ``[N/32, K/32]``.

    The scales have to be exact powers of two: that is what ue8m0 means, and
    ``ue8m0_weight_scale`` asserts it.
    """
    w = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    e = torch.randint(
        120, 123, (n // 32, k // 32), generator=g, device="cuda", dtype=torch.int32
    )
    return w, torch.exp2(e.float() - 127.0)


def reference(x_bf16, w, ws):
    """fp32 reference from the dequantised operands, block by block."""
    wf = w.float() * ws.repeat_interleave(32, 0).repeat_interleave(32, 1)
    return x_bf16.float() @ wf.T


def run_shape(m: int, n: int, k: int, args, out):
    g = torch.Generator(device="cuda").manual_seed(7)
    x = (torch.randn(m, k, generator=g, device="cuda") / 8).to(torch.bfloat16)
    w, ws = make_weight(n, k, g)

    ref = reference(x, w, ws)
    rn = ref.norm()

    def rel(y):
        return ((y.float() - ref).norm() / rn).item()

    w_bf16 = dequant_block_fp8_weight_to_bf16(w, ws, BLOCK)
    shuffled, e8m0, w_bf16_copy = prepare_mxfp8_native_weight(w, ws, BLOCK)
    xq, xs = mxfp8_e4m3_quantize(x)

    # The production route, split the way the trace shows it: a standalone
    # fake-quant kernel and then the bf16 GEMM.
    x_grid = fake_quant_fp8_activation(x)

    cases = {
        "prod-bf16(fake-quant+gemm)": lambda: bf16_dequant_blockscaled_linear(
            x, w_bf16, input_on_fp8_grid=False
        ),
        "prod-bf16(gemm only)": lambda: F.linear(x_grid, w_bf16),
        "native-mxfp8(bf16-in)": lambda: mxfp8_native_blockscaled_linear(
            x, shuffled, e8m0, w_bf16_copy
        ),
        "native-mxfp8(fp8-in)": lambda: mxfp8_native_blockscaled_linear(
            xq, shuffled, e8m0, w_bf16_copy, input_scale=xs
        ),
    }

    # relL2 against the fp32 reference is dominated by quantising the bf16
    # activation, which the reference does not do, so it is the same ~2.7e-2 for
    # every route and only says "this is an fp8 GEMM". What actually
    # distinguishes them is whether they agree with each other, so pin that too.
    outs = {}
    for name, fn in cases.items():
        try:
            outs[name] = fn().float()
        except Exception as exc:  # pragma: no cover - route availability varies
            print(f"  {name:<28} FAILED: {type(exc).__name__}: {exc}", flush=True)
    if len(outs) > 1:
        base_name, base = next(iter(outs.items()))
        bn = base.norm()
        spread = {
            nm: ((v - base).norm() / bn).item()
            for nm, v in outs.items()
            if nm != base_name
        }
        print(
            "  agreement vs "
            + base_name
            + ": "
            + ", ".join(f"{nm.split('(')[0]} {d:.1e}" for nm, d in spread.items()),
            flush=True,
        )

    for name, fn in cases.items():
        if name not in outs:
            continue
        try:
            r = rel(outs[name])
            t = median_us(fn, args.warmup, args.iters)
        except Exception as exc:  # pragma: no cover - route availability varies
            print(f"  {name:<28} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"  {name:<28} {t:8.1f}us  relL2 {r:.2e}", flush=True)
        out.write(
            json.dumps(
                {
                    "route": name,
                    "m": m,
                    "n": n,
                    "k": k,
                    "us": t,
                    "rel_l2": r,
                    "vram_before": args._vram_before,
                    "vram_after": gpu_occupancy(),
                }
            )
            + "\n"
        )
        out.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="5120x2048", help="comma-separated NxK")
    p.add_argument("-m", default="4096,8192,11617,16384")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=31)
    p.add_argument("--json-out", default="/workspace/dsv41/sglang_wob.jsonl")
    args = p.parse_args()
    args._vram_before = gpu_occupancy()

    print(f"VRAM held per card before: {args._vram_before}", flush=True)
    with open(args.json_out, "a") as out:
        for shape in args.shapes.split(","):
            n, k = (int(v) for v in shape.split("x"))
            if not native_route_supports(n, k):
                print(f"native route does not support N={n} K={k}", file=sys.stderr)
            for m in (int(v) for v in args.m.split(",")):
                print(f"\n=== M={m} N={n} K={k} ===", flush=True)
                run_shape(m, n, k, args, out)
    print(f"\nVRAM held per card after: {gpu_occupancy()}", flush=True)


if __name__ == "__main__":
    main()
