#!/usr/bin/env python3
"""
AllReduce Comprehensive Benchmark — sweep data sizes × modes × copy settings.

Modes:
  sync      = SDMA operator() (pipeline kernel, synchronous)
  async     = SDMA start_async + wait_async
  rccl      = torch.distributed.all_reduce (NCCL/RCCL)

Copy:
  no-copy   = copy_output_to_user=False, result stays in transit buffer
  copy      = copy_output_to_user=True (or allreduce_inplace), result copied to user buffer

Output: one compact table per rank-0 with all combinations.

Usage:
  python tests/python/ccl/bench_allreduce.py
  python tests/python/ccl/bench_allreduce.py --world-size 8 --warmup 5 --iterations 10
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.distributed as dist
import mori.shmem as shmem
from mori.ccl import AllreduceSdma
from tests.python.utils import TorchDistContext, get_free_port


DATA_SIZES_MB = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

_RCCL_DTYPE_MAP = {
    torch.uint32: torch.int32,
    torch.int32: torch.int32,
    torch.float16: torch.float16,
    torch.bfloat16: torch.bfloat16,
    torch.float32: torch.float32,
}


def _stream_to_int(stream) -> int:
    if stream is None:
        return 0
    if isinstance(stream, int):
        return stream
    return stream.cuda_stream


def _measure(fn, setup_fn, stream, warmup, iterations):
    """Return list of per-iteration times (seconds) measured by CUDA events."""
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    times = []
    for i in range(warmup + iterations):
        if setup_fn:
            setup_fn()
        torch.cuda.synchronize()
        dist.barrier()
        ev_s.record(stream)
        fn()
        ev_e.record(stream)
        stream.synchronize()
        t = ev_s.elapsed_time(ev_e) / 1000.0
        if i >= warmup:
            times.append(t)
    return times


def _measure_rccl(fn, setup_fn, warmup, iterations):
    """RCCL uses its own stream; measure with wall-clock + cuda sync."""
    times = []
    for i in range(warmup + iterations):
        if setup_fn:
            setup_fn()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        if i >= warmup:
            times.append(t)
    return times


def _avg_across_pes(local_times, npes):
    """Return global average time (seconds) across all PEs."""
    avg = float(np.mean(local_times)) if local_times else 0.0
    t = torch.tensor([avg], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / npes


def _bench_worker(rank, world_size, port, warmup, iterations, dtype):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")

        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        device = torch.device(f"cuda:{rank}")
        stream = torch.cuda.Stream(device=device)
        elem_size = torch.tensor([], dtype=dtype).element_size()
        rccl_dtype = _RCCL_DTYPE_MAP.get(dtype, torch.float32)

        rows = []

        for mb in DATA_SIZES_MB:
            data_bytes = mb * 1024 * 1024
            elems = data_bytes // elem_size
            fill_val = (my_pe + 1) * 1000
            output_buf_size = npes * (elems // npes + 64) * elem_size

            in_buf = torch.full((elems,), fill_val, dtype=dtype, device=device)
            out_buf = torch.zeros(elems, dtype=dtype, device=device)

            row = {"mb": mb}

            # --- sync no-copy ---
            ar = AllreduceSdma(my_pe, npes, input_buffer_size=data_bytes,
                               output_buffer_size=output_buf_size,
                               copy_output_to_user=False, dtype=dtype)
            ts = _measure(
                lambda: ar(in_buf, out_buf, elems, stream),
                None, stream, warmup, iterations)
            row["sync_nocopy"] = _avg_across_pes(ts, npes)
            del ar

            # --- sync copy (in-place) ---
            ar = AllreduceSdma(my_pe, npes, input_buffer_size=data_bytes,
                               output_buffer_size=output_buf_size,
                               copy_output_to_user=False, dtype=dtype)
            ip_buf = torch.full((elems,), fill_val, dtype=dtype, device=device)

            def _ip_setup(buf=ip_buf, v=fill_val):
                buf.fill_(v)

            ts = _measure(
                lambda: ar.allreduce_inplace(ip_buf, elems, stream),
                _ip_setup, stream, warmup, iterations)
            row["sync_copy"] = _avg_across_pes(ts, npes)
            del ar, ip_buf

            # --- async no-copy ---
            ar = AllreduceSdma(my_pe, npes, input_buffer_size=data_bytes,
                               output_buffer_size=output_buf_size,
                               copy_output_to_user=False, dtype=dtype)

            def _async_fn(h=ar, ib=in_buf, ob=out_buf, n=elems, s=stream):
                h.start_async(ib, ob, n, s)
                h.wait_async(s)

            ts = _measure(_async_fn, None, stream, warmup, iterations)
            row["async_nocopy"] = _avg_across_pes(ts, npes)
            del ar

            # --- async copy (in-place via start_async + manual copy) ---
            ar = AllreduceSdma(my_pe, npes, input_buffer_size=data_bytes,
                               output_buffer_size=output_buf_size,
                               copy_output_to_user=True, dtype=dtype)

            def _async_copy_fn(h=ar, ib=in_buf, ob=out_buf, n=elems, s=stream):
                h.start_async(ib, ob, n, s)
                h.wait_async(s)

            ts = _measure(_async_copy_fn, None, stream, warmup, iterations)
            row["async_copy"] = _avg_across_pes(ts, npes)
            del ar

            # --- RCCL out-of-place (copy input → output, then allreduce output) ---
            rccl_fill = float(fill_val) if rccl_dtype in (torch.float16, torch.bfloat16) else fill_val
            rccl_in = torch.full((elems,), rccl_fill, dtype=rccl_dtype, device=device)
            rccl_out = torch.zeros(elems, dtype=rccl_dtype, device=device)

            def _rccl_op_fn(src=rccl_in, dst=rccl_out):
                dst.copy_(src)
                dist.all_reduce(dst, op=dist.ReduceOp.SUM)

            ts = _measure_rccl(_rccl_op_fn, None, warmup, iterations)
            row["rccl_nocopy"] = _avg_across_pes(ts, npes)
            del rccl_in, rccl_out

            # --- RCCL in-place ---
            rccl_buf = torch.full((elems,), rccl_fill, dtype=rccl_dtype, device=device)

            def _rccl_setup(buf=rccl_buf, v=rccl_fill):
                buf.fill_(v)

            ts = _measure_rccl(
                lambda: dist.all_reduce(rccl_buf, op=dist.ReduceOp.SUM),
                _rccl_setup, warmup, iterations)
            row["rccl_copy"] = _avg_across_pes(ts, npes)
            del rccl_buf

            rows.append(row)
            del in_buf, out_buf

            if rank == 0:
                import sys
                print(f"  {mb} MB done", file=sys.stderr, flush=True)

        # --- Print table (rank 0 only) ---
        if rank == 0:
            _print_table(rows, npes, elem_size, dtype)

        torch.cuda.synchronize()
        dist.barrier()


def _bw(data_bytes, t):
    if t <= 0:
        return 0.0
    return data_bytes / t / (1024.0 ** 3)


def _print_table(rows, npes, elem_size, dtype):
    dtype_name = str(dtype).split(".")[-1]
    sep = "-" * 120
    print()
    print(sep)
    print(f"  AllReduce Benchmark  npes={npes}  dtype={dtype_name}")
    print(f"  Columns: Algo Bandwidth (GB/s)")
    print(f"  sync=pipeline  async=RS+AG(async)  rccl=torch.distributed")
    print(sep)
    print()

    hdr = (
        f"{'MB/PE':>6} | {'sync':>8} {'sync':>8} | {'async':>8} {'async':>8} | "
        f"{'RCCL':>8} {'RCCL':>8} | {'best':>14}"
    )
    sub = (
        f"{'':>6} | {'no-copy':>8} {'copy':>8} | {'no-copy':>8} {'copy':>8} | "
        f"{'outplace':>8} {'inplace':>8} | {'':>14}"
    )
    print(hdr)
    print(sub)
    print(sep)

    for r in rows:
        mb = r["mb"]
        db = mb * 1024 * 1024

        bw_sn = _bw(db, r["sync_nocopy"])
        bw_sc = _bw(db, r["sync_copy"])
        bw_an = _bw(db, r["async_nocopy"])
        bw_ac = _bw(db, r["async_copy"])
        bw_rn = _bw(db, r["rccl_nocopy"])
        bw_rc = _bw(db, r["rccl_copy"])

        all_bw = {"sync-no-copy": bw_sn, "sync-copy": bw_sc,
                  "async-no-copy": bw_an, "async-copy": bw_ac,
                  "rccl-outplace": bw_rn, "rccl-inplace": bw_rc}
        best_name = max(all_bw, key=all_bw.get)

        print(
            f"{mb:>6} | {bw_sn:>8.1f} {bw_sc:>8.1f} | "
            f"{bw_an:>8.1f} {bw_ac:>8.1f} | "
            f"{bw_rn:>8.1f} {bw_rc:>8.1f} | {best_name:>14}"
        )

    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="AllReduce comprehensive benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="uint32",
                        choices=["uint32", "fp16", "bf16"])
    args = parser.parse_args()

    os.environ.setdefault("MORI_ENABLE_SDMA", "1")

    dtype_map = {"uint32": torch.uint32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    print(f"AllReduce Benchmark: world_size={args.world_size} warmup={args.warmup} "
          f"iters={args.iterations} dtype={args.dtype}")

    port = get_free_port()
    torch.multiprocessing.spawn(
        _bench_worker,
        args=(args.world_size, port, args.warmup, args.iterations, dtype),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
