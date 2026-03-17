#!/usr/bin/env python3
"""
GEMM standalone benchmark — measure baseline latency without any CCL overlap.
Runs on a single GPU (default GPU 0).
"""

import argparse
import torch
import numpy as np

try:
    import aiter
    HAS_AITER = True
except ImportError:
    HAS_AITER = False


def bench_gemm(M, N, K, iterations, warmup, device_id=0):
    device = torch.device(f"cuda:{device_id}")
    A_q = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=device)
    B_q = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=device)
    A_scale = torch.randn(M, dtype=torch.float32, device=device)
    B_scale = torch.randn(N, dtype=torch.float32, device=device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=device)

    stream = torch.cuda.Stream(device=device)
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    times = []
    total_iters = warmup + iterations

    for i in range(total_iters):
        torch.cuda.synchronize(device)
        start_ev.record(stream)
        with torch.cuda.stream(stream):
            _ = aiter.gemm_a8w8_CK(A_q, B_q, A_scale, B_scale, bias, torch.bfloat16)
        end_ev.record(stream)
        stream.synchronize()

        t = start_ev.elapsed_time(end_ev)  # ms
        if i >= warmup:
            times.append(t)

    return times


def main():
    parser = argparse.ArgumentParser(description="GEMM standalone benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 8192, 16384],
                        help="List of M=N=K sizes to benchmark")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", type=int, default=0, help="GPU device id")
    args = parser.parse_args()

    if not HAS_AITER:
        print("ERROR: aiter is required for GEMM benchmark")
        return

    print(f"GEMM Standalone Benchmark (INT8 A8W8, aiter.gemm_a8w8_CK)")
    print(f"  Device: cuda:{args.device}")
    print(f"  Iterations: {args.iterations}, Warmup: {args.warmup}")
    print(f"  Sizes: {args.sizes}")
    print("=" * 60)

    results = {}
    for size in args.sizes:
        M = N = K = size
        print(f"\nBenchmarking GEMM M=N=K={size} ...")
        times = bench_gemm(M, N, K, args.iterations, args.warmup, args.device)

        avg = np.mean(times)
        mn = np.min(times)
        mx = np.max(times)
        std = np.std(times)
        results[size] = {"avg": avg, "min": mn, "max": mx, "std": std}

        print(f"  Min: {mn:.4f} ms")
        print(f"  Avg: {avg:.4f} ms")
        print(f"  Max: {mx:.4f} ms")
        print(f"  Std: {std:.4f} ms")
        print(f"  All times (ms): {[f'{t:.4f}' for t in times]}")

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"| {'GEMM Size':>10} | {'Min (ms)':>10} | {'Avg (ms)':>10} | {'Max (ms)':>10} | {'Std (ms)':>10} |")
    print(f"|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|")
    for size in args.sizes:
        r = results[size]
        print(f"| {size:>10} | {r['min']:>10.4f} | {r['avg']:>10.4f} | {r['max']:>10.4f} | {r['std']:>10.4f} |")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
