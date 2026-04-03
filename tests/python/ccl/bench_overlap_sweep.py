#!/usr/bin/env python3
"""
GEMM Overlap Sweep Benchmark — SDMA (copy / no-copy) vs RCCL.

Measures overlap wall time for multiple allreduce data sizes (default 1 MB → 1 GB)
and generates comparison charts (saved as PNG + CSV).

Usage:
    python -m tests.python.ccl.bench_overlap_sweep [options]

    --sizes       Comma-separated MB list, e.g. "1,4,16,64,256,512,1024"
    --output-dir  Directory for result files (default: bench_overlap_results)
    --iterations  Measurement iterations per size (default: 10)
    --warmup      Warmup iterations per size (default: 5)
    --gemm-m/n/k  GEMM dimensions (default: 4096 each)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Allow running as `python tests/python/ccl/bench_overlap_sweep.py` from project root
_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import torch.distributed as dist
import mori.shmem as shmem
from mori.ccl import AllreduceSdma
from tests.python.utils import TorchDistContext, get_free_port

_RCCL_DTYPE_MAP = {
    torch.uint32: torch.int32,
    torch.int32: torch.int32,
    torch.float16: torch.float16,
    torch.bfloat16: torch.bfloat16,
    torch.float32: torch.float32,
}

_DEFAULT_SIZES_MB = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def _reduce_scalar_list(vals, rank, npes, device):
    t = torch.tensor(
        [min(vals), max(vals), sum(vals) / len(vals)],
        dtype=torch.float64, device=device,
    )
    mn, mx, avg = t[0].clone(), t[1].clone(), t[2].clone()
    dist.all_reduce(mn, op=dist.ReduceOp.MIN)
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    dist.all_reduce(avg, op=dist.ReduceOp.SUM)
    return mn.item(), mx.item(), avg.item() / npes


def _bench_overlap_one(launch_ar, prep_fn, run_gemm, stream_ar, stream_gemm,
                       iterations, warmup, time_ar_wall):
    ev_ar_s = torch.cuda.Event(enable_timing=True)
    ev_ar_e = torch.cuda.Event(enable_timing=True)
    ev_g_s = torch.cuda.Event(enable_timing=True)
    ev_g_e = torch.cuda.Event(enable_timing=True)
    ov_s = torch.cuda.Event(enable_timing=True)
    ov_e = torch.cuda.Event(enable_timing=True)
    total = warmup + iterations

    seq_ar, seq_gemm, overlap = [], [], []

    for i in range(total):
        torch.cuda.synchronize()
        prep_fn()
        torch.cuda.synchronize()
        if time_ar_wall:
            t0 = time.perf_counter()
            launch_ar()
            torch.cuda.synchronize()
            t_ar = time.perf_counter() - t0
        else:
            ev_ar_s.record(stream_ar)
            with torch.cuda.stream(stream_ar):
                launch_ar()
            ev_ar_e.record(stream_ar)
            stream_ar.synchronize()
            t_ar = ev_ar_s.elapsed_time(ev_ar_e) / 1000.0
        if i >= warmup:
            seq_ar.append(t_ar)

    for i in range(total):
        torch.cuda.synchronize()
        ev_g_s.record(stream_gemm)
        with torch.cuda.stream(stream_gemm):
            run_gemm()
        ev_g_e.record(stream_gemm)
        stream_gemm.synchronize()
        t_g = ev_g_s.elapsed_time(ev_g_e) / 1000.0
        if i >= warmup:
            seq_gemm.append(t_g)

    for i in range(total):
        torch.cuda.synchronize()
        prep_fn()
        torch.cuda.synchronize()
        ov_s.record()
        with torch.cuda.stream(stream_ar):
            launch_ar()
        with torch.cuda.stream(stream_gemm):
            run_gemm()
        stream_ar.synchronize()
        stream_gemm.synchronize()
        ov_e.record()
        torch.cuda.synchronize()
        t_ov = ov_s.elapsed_time(ov_e) / 1000.0
        if i >= warmup:
            overlap.append(t_ov)

    return seq_ar, seq_gemm, overlap


def _worker(rank, world_size, port, sizes_mb, iterations, warmup,
            gemm_m, gemm_n, gemm_k, output_dir):
    dtype = torch.uint32
    elem_size = 4

    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        device = torch.device(f"cuda:{rank}")
        fill_value = (my_pe + 1) * 1000

        rccl_dtype = _RCCL_DTYPE_MAP.get(dtype, torch.float32)

        stream_ar = torch.cuda.Stream(device=device)
        stream_gemm = torch.cuda.Stream(device=device)

        A = torch.randn(gemm_m, gemm_k, dtype=torch.float32, device=device)
        B = torch.randn(gemm_k, gemm_n, dtype=torch.float32, device=device)

        def run_gemm():
            return torch.matmul(A, B)

        # warmup GEMM
        for _ in range(3):
            with torch.cuda.stream(stream_gemm):
                run_gemm()
        stream_gemm.synchronize()

        results = []

        for size_mb in sizes_mb:
            data_bytes = size_mb * 1024 * 1024
            elems = data_bytes // elem_size
            output_buf_size = npes * (elems // npes + 64) * elem_size

            if rank == 0:
                print(f"\n--- Sweep: {size_mb} MB ({elems:,} elems) ---")

            row = {"size_mb": size_mb, "elems": elems}

            # --- SDMA copy=True ---
            try:
                ar_copy = AllreduceSdma(
                    my_pe, npes,
                    input_buffer_size=data_bytes,
                    output_buffer_size=output_buf_size,
                    copy_output_to_user=True, dtype=dtype,
                )
                inp = torch.full((elems,), fill_value, dtype=dtype, device=device)
                out = torch.zeros(elems, dtype=dtype, device=device)
                torch.cuda.synchronize()
                dist.barrier()
                stream_ar.synchronize()
                ok = ar_copy(inp, out, elems, stream_ar)
                stream_ar.synchronize()

                if ok:
                    s_ar, s_gm, ov = _bench_overlap_one(
                        lambda: ar_copy(inp, out, elems, stream_ar),
                        lambda: inp.fill_(fill_value),
                        run_gemm, stream_ar, stream_gemm,
                        iterations, warmup, False,
                    )
                    g_ar = _reduce_scalar_list(s_ar, rank, npes, device)
                    g_gm = _reduce_scalar_list(s_gm, rank, npes, device)
                    g_ov = _reduce_scalar_list(ov, rank, npes, device)
                    row["sdma_copy_ar_avg"] = g_ar[2]
                    row["sdma_copy_gemm_avg"] = g_gm[2]
                    row["sdma_copy_overlap_avg"] = g_ov[2]
                    row["sdma_copy_overlap_min"] = g_ov[0]
                    if rank == 0:
                        print(f"  SDMA copy    : overlap {g_ov[2]*1000:.3f} ms  "
                              f"(ar {g_ar[2]*1000:.3f}, gemm {g_gm[2]*1000:.3f})")
                del ar_copy
            except Exception as e:
                if rank == 0:
                    print(f"  SDMA copy    : FAILED ({e})")

            torch.cuda.synchronize()
            dist.barrier()

            # --- SDMA copy=False ---
            try:
                ar_nocopy = AllreduceSdma(
                    my_pe, npes,
                    input_buffer_size=data_bytes,
                    output_buffer_size=output_buf_size,
                    copy_output_to_user=False, dtype=dtype,
                )
                inp = torch.full((elems,), fill_value, dtype=dtype, device=device)
                out = torch.zeros(elems, dtype=dtype, device=device)
                torch.cuda.synchronize()
                dist.barrier()
                stream_ar.synchronize()
                ok = ar_nocopy(inp, out, elems, stream_ar)
                stream_ar.synchronize()

                if ok:
                    s_ar, s_gm, ov = _bench_overlap_one(
                        lambda: ar_nocopy(inp, out, elems, stream_ar),
                        lambda: inp.fill_(fill_value),
                        run_gemm, stream_ar, stream_gemm,
                        iterations, warmup, False,
                    )
                    g_ar = _reduce_scalar_list(s_ar, rank, npes, device)
                    g_gm = _reduce_scalar_list(s_gm, rank, npes, device)
                    g_ov = _reduce_scalar_list(ov, rank, npes, device)
                    row["sdma_nocopy_ar_avg"] = g_ar[2]
                    row["sdma_nocopy_gemm_avg"] = g_gm[2]
                    row["sdma_nocopy_overlap_avg"] = g_ov[2]
                    row["sdma_nocopy_overlap_min"] = g_ov[0]
                    if rank == 0:
                        print(f"  SDMA no-copy : overlap {g_ov[2]*1000:.3f} ms  "
                              f"(ar {g_ar[2]*1000:.3f}, gemm {g_gm[2]*1000:.3f})")
                del ar_nocopy
            except Exception as e:
                if rank == 0:
                    print(f"  SDMA no-copy : FAILED ({e})")

            torch.cuda.synchronize()
            dist.barrier()

            # --- RCCL ---
            try:
                buf = torch.full((elems,), fill_value, dtype=rccl_dtype, device=device)
                dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize()

                s_ar, s_gm, ov = _bench_overlap_one(
                    lambda: dist.all_reduce(buf, op=dist.ReduceOp.SUM),
                    lambda: buf.fill_(fill_value),
                    run_gemm, stream_ar, stream_gemm,
                    iterations, warmup, True,
                )
                g_ar = _reduce_scalar_list(s_ar, rank, npes, device)
                g_gm = _reduce_scalar_list(s_gm, rank, npes, device)
                g_ov = _reduce_scalar_list(ov, rank, npes, device)
                row["rccl_ar_avg"] = g_ar[2]
                row["rccl_gemm_avg"] = g_gm[2]
                row["rccl_overlap_avg"] = g_ov[2]
                row["rccl_overlap_min"] = g_ov[0]
                if rank == 0:
                    print(f"  RCCL         : overlap {g_ov[2]*1000:.3f} ms  "
                          f"(ar {g_ar[2]*1000:.3f}, gemm {g_gm[2]*1000:.3f})")
                del buf
            except Exception as e:
                if rank == 0:
                    print(f"  RCCL         : FAILED ({e})")

            torch.cuda.synchronize()
            dist.barrier()
            results.append(row)

        # rank 0 saves results and generates charts
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, "overlap_sweep.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {json_path}")

            _save_csv(results, output_dir)
            _generate_charts(results, output_dir, gemm_m, gemm_n, gemm_k)

        torch.cuda.synchronize()
        dist.barrier()
        shmem.shmem_finalize()


def _save_csv(results, output_dir):
    import csv
    csv_path = os.path.join(output_dir, "overlap_sweep.csv")
    fields = [
        "size_mb",
        "sdma_copy_overlap_avg", "sdma_copy_overlap_min",
        "sdma_nocopy_overlap_avg", "sdma_nocopy_overlap_min",
        "rccl_overlap_avg", "rccl_overlap_min",
        "sdma_copy_ar_avg", "sdma_nocopy_ar_avg", "rccl_ar_avg",
        "sdma_copy_gemm_avg", "sdma_nocopy_gemm_avg", "rccl_gemm_avg",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"CSV saved to {csv_path}")


def _generate_charts(results, output_dir, gemm_m, gemm_n, gemm_k):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping chart generation")
        return

    sizes = [r["size_mb"] for r in results]

    def _get(key):
        return [r.get(key, float("nan")) * 1000 for r in results]

    sdma_copy_ov = _get("sdma_copy_overlap_avg")
    sdma_nocopy_ov = _get("sdma_nocopy_overlap_avg")
    rccl_ov = _get("rccl_overlap_avg")

    sdma_copy_ar = _get("sdma_copy_ar_avg")
    sdma_nocopy_ar = _get("sdma_nocopy_ar_avg")
    rccl_ar = _get("rccl_ar_avg")

    sdma_copy_gemm = _get("sdma_copy_gemm_avg")
    sdma_nocopy_gemm = _get("sdma_nocopy_gemm_avg")
    rccl_gemm = _get("rccl_gemm_avg")

    subtitle = f"GEMM {gemm_m}x{gemm_k}x{gemm_n}, 8×MI300X"

    # --- Chart 1: Overlap wall time ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sizes, sdma_copy_ov, "o-", label="SDMA copy", linewidth=2, markersize=6)
    ax.plot(sizes, sdma_nocopy_ov, "s-", label="SDMA no-copy", linewidth=2, markersize=6)
    ax.plot(sizes, rccl_ov, "^-", label="RCCL", linewidth=2, markersize=6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Allreduce Data Size (MB per rank)")
    ax.set_ylabel("Overlap Wall Time (ms)")
    ax.set_title(f"GEMM Overlap Wall Time vs Data Size\n{subtitle}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    fig.tight_layout()
    path1 = os.path.join(output_dir, "overlap_wall_time.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"Chart saved: {path1}")

    # --- Chart 2: Overlap ratio ---
    def _ratio(ov_list, ar_list, gemm_list):
        out = []
        for o, a, g in zip(ov_list, ar_list, gemm_list):
            s = a + g
            out.append(o / s if s > 0 else float("nan"))
        return out

    sdma_copy_ratio = _ratio(sdma_copy_ov, sdma_copy_ar, sdma_copy_gemm)
    sdma_nocopy_ratio = _ratio(sdma_nocopy_ov, sdma_nocopy_ar, sdma_nocopy_gemm)
    rccl_ratio = _ratio(rccl_ov, rccl_ar, rccl_gemm)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sizes, sdma_copy_ratio, "o-", label="SDMA copy", linewidth=2, markersize=6)
    ax.plot(sizes, sdma_nocopy_ratio, "s-", label="SDMA no-copy", linewidth=2, markersize=6)
    ax.plot(sizes, rccl_ratio, "^-", label="RCCL", linewidth=2, markersize=6)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Ideal (0.5x)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Allreduce Data Size (MB per rank)")
    ax.set_ylabel("Overlap Ratio (lower = better)")
    ax.set_title(f"Overlap Ratio vs Data Size\n{subtitle}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    fig.tight_layout()
    path2 = os.path.join(output_dir, "overlap_ratio.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"Chart saved: {path2}")

    # --- Chart 3: Speedup over RCCL ---
    def _speedup(rccl_list, sdma_list):
        return [
            (r - s) / r * 100 if r > 0 else float("nan")
            for r, s in zip(rccl_list, sdma_list)
        ]

    speedup_copy = _speedup(rccl_ov, sdma_copy_ov)
    speedup_nocopy = _speedup(rccl_ov, sdma_nocopy_ov)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([s - 0.15 * s for s in sizes], speedup_copy, width=[0.25 * s for s in sizes],
           label="SDMA copy", alpha=0.8)
    ax.bar([s + 0.15 * s for s in sizes], speedup_nocopy, width=[0.25 * s for s in sizes],
           label="SDMA no-copy", alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Allreduce Data Size (MB per rank)")
    ax.set_ylabel("Speedup over RCCL (%)")
    ax.set_title(f"SDMA Overlap Speedup over RCCL\n{subtitle}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    fig.tight_layout()
    path3 = os.path.join(output_dir, "speedup_vs_rccl.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"Chart saved: {path3}")

    # --- Chart 4: Sequential allreduce time ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sizes, sdma_copy_ar, "o-", label="SDMA copy", linewidth=2, markersize=6)
    ax.plot(sizes, sdma_nocopy_ar, "s-", label="SDMA no-copy", linewidth=2, markersize=6)
    ax.plot(sizes, rccl_ar, "^-", label="RCCL", linewidth=2, markersize=6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Allreduce Data Size (MB per rank)")
    ax.set_ylabel("Sequential Allreduce Time (ms)")
    ax.set_title(f"Sequential Allreduce Time vs Data Size\n{subtitle}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], rotation=45)
    fig.tight_layout()
    path4 = os.path.join(output_dir, "seq_allreduce_time.png")
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    print(f"Chart saved: {path4}")


def main():
    parser = argparse.ArgumentParser(
        description="GEMM Overlap Sweep: SDMA vs RCCL across data sizes",
    )
    parser.add_argument(
        "--sizes", type=str, default=None,
        help="Comma-separated MB sizes (default: 1,2,4,8,16,32,64,128,256,512,1024)",
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--gemm-m", type=int, default=4096)
    parser.add_argument("--gemm-n", type=int, default=4096)
    parser.add_argument("--gemm-k", type=int, default=4096)
    parser.add_argument("--output-dir", type=str, default="bench_overlap_results")
    parser.add_argument("--enable-sdma", type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    os.environ["MORI_ENABLE_SDMA"] = str(args.enable_sdma)

    if args.sizes:
        sizes_mb = [int(s.strip()) for s in args.sizes.split(",")]
    else:
        sizes_mb = _DEFAULT_SIZES_MB

    print("=" * 60)
    print("GEMM Overlap Sweep Benchmark")
    print(f"  Sizes (MB)   : {sizes_mb}")
    print(f"  World size   : {args.world_size}")
    print(f"  Iterations   : {args.iterations} (warmup: {args.warmup})")
    print(f"  GEMM         : {args.gemm_m}x{args.gemm_k}x{args.gemm_n}")
    print(f"  Output dir   : {args.output_dir}")
    print("=" * 60)

    port = get_free_port()
    torch.multiprocessing.spawn(
        _worker,
        args=(
            args.world_size, port, sizes_mb,
            args.iterations, args.warmup,
            args.gemm_m, args.gemm_n, args.gemm_k,
            args.output_dir,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
