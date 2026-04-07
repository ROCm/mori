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
        dist.barrier()
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
        dist.barrier()
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

        max_size_mb = max(sizes_mb)
        max_data_bytes = max_size_mb * 1024 * 1024
        max_elems = max_data_bytes // elem_size
        max_output_buf_size = npes * (max_elems // npes + 64) * elem_size

        # Create SDMA objects once with max size, reuse for all sizes
        ar_copy = None
        ar_nocopy = None
        try:
            ar_copy = AllreduceSdma(
                my_pe, npes,
                input_buffer_size=max_data_bytes,
                output_buffer_size=max_output_buf_size,
                copy_output_to_user=True, dtype=dtype,
            )
        except Exception as e:
            if rank == 0:
                print(f"  SDMA copy init FAILED: {e}")
        try:
            ar_nocopy = AllreduceSdma(
                my_pe, npes,
                input_buffer_size=max_data_bytes,
                output_buffer_size=max_output_buf_size,
                copy_output_to_user=False, dtype=dtype,
            )
        except Exception as e:
            if rank == 0:
                print(f"  SDMA no-copy init FAILED: {e}")

        # Warmup SDMA objects with max size
        inp_max = torch.full((max_elems,), fill_value, dtype=dtype, device=device)
        out_max = torch.zeros(max_elems, dtype=dtype, device=device)
        torch.cuda.synchronize()
        dist.barrier()
        if ar_copy is not None:
            ar_copy(inp_max, out_max, max_elems, stream_ar)
            stream_ar.synchronize()
        dist.barrier()
        if ar_nocopy is not None:
            ar_nocopy(inp_max, out_max, max_elems, stream_ar)
            stream_ar.synchronize()
        dist.barrier()
        del inp_max, out_max

        for size_mb in sizes_mb:
            data_bytes = size_mb * 1024 * 1024
            elems = data_bytes // elem_size

            if rank == 0:
                print(f"\n--- Sweep: {size_mb} MB ({elems:,} elems) ---")

            row = {"size_mb": size_mb, "elems": elems}
            inp = torch.full((elems,), fill_value, dtype=dtype, device=device)
            out = torch.zeros(elems, dtype=dtype, device=device)

            # --- SDMA copy=True ---
            if ar_copy is not None:
                try:
                    cur_elems = elems

                    def launch_copy():
                        return ar_copy(inp, out, cur_elems, stream_ar)

                    def prep_copy():
                        inp.fill_(fill_value)

                    torch.cuda.synchronize()
                    dist.barrier()
                    ok = ar_copy(inp, out, elems, stream_ar)
                    stream_ar.synchronize()
                    if ok:
                        s_ar, s_gm, ov = _bench_overlap_one(
                            launch_copy, prep_copy,
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
                except Exception as e:
                    if rank == 0:
                        print(f"  SDMA copy    : FAILED ({e})")

            torch.cuda.synchronize()
            dist.barrier()

            # --- SDMA copy=False ---
            if ar_nocopy is not None:
                try:
                    cur_elems = elems

                    def launch_nocopy():
                        return ar_nocopy(inp, out, cur_elems, stream_ar)

                    def prep_nocopy():
                        inp.fill_(fill_value)

                    torch.cuda.synchronize()
                    dist.barrier()
                    ok = ar_nocopy(inp, out, elems, stream_ar)
                    stream_ar.synchronize()
                    if ok:
                        s_ar, s_gm, ov = _bench_overlap_one(
                            launch_nocopy, prep_nocopy,
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

                def launch_rccl():
                    dist.all_reduce(buf, op=dist.ReduceOp.SUM)

                def prep_rccl():
                    buf.fill_(fill_value)

                s_ar, s_gm, ov = _bench_overlap_one(
                    launch_rccl, prep_rccl,
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

        # rank 0 saves results and prints tables before cleanup
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, "overlap_sweep.json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {json_path}")

            _save_csv(results, output_dir)
            _print_summary_tables(results, gemm_m, gemm_n, gemm_k)

        torch.cuda.synchronize()
        dist.barrier()
        del ar_copy, ar_nocopy
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


def _print_summary_tables(results, gemm_m, gemm_n, gemm_k):
    def _ms(r, key):
        v = r.get(key)
        return f"{v * 1000:8.3f}" if v is not None else "     N/A"

    def _slowdown(r, ov_key, gemm_key):
        ov = r.get(ov_key)
        gm = r.get(gemm_key)
        if ov is not None and gm is not None and gm > 0:
            return f"{ov / gm:8.3f}"
        return "     N/A"

    def _speedup(r, rccl_key, sdma_key):
        rc = r.get(rccl_key)
        sd = r.get(sdma_key)
        if rc is not None and sd is not None and rc > 0:
            return f"{(rc - sd) / rc * 100:+7.1f}%"
        return "     N/A"

    sep = "-" * 120

    # Table 1: Overlap Wall Time (ms)
    print(f"\n{'=' * 120}")
    print(f"  GEMM Overlap Summary — GEMM {gemm_m}x{gemm_k}x{gemm_n}")
    print(f"{'=' * 120}")
    print(f"\n  Table 1: Overlap Wall Time (ms, avg)")
    print(f"  {sep}")
    print(f"  {'Size':>8s} | {'SDMA copy':>10s} | {'SDMA no-copy':>12s} | {'RCCL':>10s} |"
          f" {'copy vs RCCL':>12s} | {'no-copy vs RCCL':>15s}")
    print(f"  {sep}")
    for r in results:
        sz = f"{r['size_mb']:>6d} MB"
        c = _ms(r, "sdma_copy_overlap_avg")
        nc = _ms(r, "sdma_nocopy_overlap_avg")
        rc = _ms(r, "rccl_overlap_avg")
        sp_c = _speedup(r, "rccl_overlap_avg", "sdma_copy_overlap_avg")
        sp_nc = _speedup(r, "rccl_overlap_avg", "sdma_nocopy_overlap_avg")
        print(f"  {sz:>8s} | {c:>10s} | {nc:>12s} | {rc:>10s} | {sp_c:>12s} | {sp_nc:>15s}")
    print(f"  {sep}")

    # Table 2: GEMM Slowdown (overlap_wall / seq_gemm, 1.0 = perfect hiding)
    print(f"\n  Table 2: GEMM Slowdown (overlap_wall / seq_gemm, 1.0 = perfect hiding)")
    print(f"  {sep}")
    print(f"  {'Size':>8s} | {'SDMA copy':>10s} | {'SDMA no-copy':>12s} | {'RCCL':>10s}")
    print(f"  {sep}")
    for r in results:
        sz = f"{r['size_mb']:>6d} MB"
        c = _slowdown(r, "sdma_copy_overlap_avg", "sdma_copy_gemm_avg")
        nc = _slowdown(r, "sdma_nocopy_overlap_avg", "sdma_nocopy_gemm_avg")
        rc = _slowdown(r, "rccl_overlap_avg", "rccl_gemm_avg")
        print(f"  {sz:>8s} | {c:>10s} | {nc:>12s} | {rc:>10s}")
    print(f"  {sep}")

    # Table 3: Sequential Allreduce Time (ms)
    print(f"\n  Table 3: Sequential Allreduce Time (ms, avg)")
    print(f"  {sep}")
    print(f"  {'Size':>8s} | {'SDMA copy':>10s} | {'SDMA no-copy':>12s} | {'RCCL':>10s}")
    print(f"  {sep}")
    for r in results:
        sz = f"{r['size_mb']:>6d} MB"
        c = _ms(r, "sdma_copy_ar_avg")
        nc = _ms(r, "sdma_nocopy_ar_avg")
        rc = _ms(r, "rccl_ar_avg")
        print(f"  {sz:>8s} | {c:>10s} | {nc:>12s} | {rc:>10s}")
    print(f"  {sep}")

    # Table 4: Sequential GEMM Time (ms)
    print(f"\n  Table 4: Sequential GEMM Time (ms, avg)")
    print(f"  {sep}")
    print(f"  {'Size':>8s} | {'SDMA copy':>10s} | {'SDMA no-copy':>12s} | {'RCCL':>10s}")
    print(f"  {sep}")
    for r in results:
        sz = f"{r['size_mb']:>6d} MB"
        c = _ms(r, "sdma_copy_gemm_avg")
        nc = _ms(r, "sdma_nocopy_gemm_avg")
        rc = _ms(r, "rccl_gemm_avg")
        print(f"  {sz:>8s} | {c:>10s} | {nc:>12s} | {rc:>10s}")
    print(f"  {sep}")
    print()


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
    parser.add_argument("--gemm-k", type=int, default=8192)
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
