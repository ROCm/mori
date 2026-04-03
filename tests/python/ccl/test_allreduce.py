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
"""
AllReduce SDMA Test using torch.distributed and multiprocessing.

Tests four modes (always):
  1. SDMA out-of-place (copy_output_to_user=False, read from transit buffer)
  2. SDMA in-place     (allreduce_inplace)
  3. RCCL out-of-place (torch.distributed.all_reduce, copy + reduce)
  4. RCCL in-place     (torch.distributed.all_reduce, refill + reduce)

Optional (--test-gemm-overlap): overlap wall time vs torch.matmul GEMM on a second
stream, comparing SDMA allreduce (copy_output_to_user True and False) vs RCCL.
Correctness is not checked in this phase (Tests 1–4 already cover it); only timing.
"""

import os
import time
import numpy as np
import torch
import torch.distributed as dist
import mori.shmem as shmem
from mori.ccl import AllreduceSdma
from tests.python.utils import TorchDistContext, get_free_port


def _to_numpy(tensor):
    """Convert a CPU tensor to numpy, handling dtypes that don't support .numpy() directly."""
    if tensor.dtype in (torch.bfloat16, torch.float16):
        return tensor.float().numpy()
    return tensor.numpy()


def _verify_allreduce_result(
    data_cpu, elems, my_pe, npes, label="", dtype=torch.uint32
):
    """Verify allreduce result: each element should equal sum of (pe+1)*1000 across all PEs."""
    expected_value = sum((pe + 1) * 1000 for pe in range(npes))
    first_n = min(elems, len(data_cpu))
    chunk = data_cpu[:first_n]

    if dtype in (torch.float16, torch.bfloat16):
        expected_arr = np.full(first_n, expected_value, dtype=np.float32)
        chunk_f = chunk.astype(np.float32)
        if np.allclose(chunk_f, expected_arr, rtol=1e-2, atol=1.0):
            return True
        else:
            mismatches = np.where(
                ~np.isclose(chunk_f, expected_arr, rtol=1e-2, atol=1.0)
            )[0]
            print(
                f"  PE {my_pe} [{label}]: FAILED! Expected ~{expected_value}, "
                f"got {len(mismatches)} mismatches in first {first_n} elements. "
                f"First mismatch at idx {mismatches[0]}: {chunk[mismatches[0]]}"
            )
            return False
    else:
        if np.all(chunk == expected_value):
            return True
        else:
            mismatches = np.where(chunk != expected_value)[0]
            print(
                f"  PE {my_pe} [{label}]: FAILED! Expected {expected_value}, "
                f"got {len(mismatches)} mismatches in first {first_n} elements. "
                f"First mismatch at idx {mismatches[0]}: {chunk[mismatches[0]]}"
            )
            return False


def _run_benchmark(allreduce_fn, iterations, warmup, stream, rank):
    """Run warmup + measurement iterations, return list of measured times."""
    exec_times = []
    total_iters = warmup + iterations

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    for i in range(total_iters):
        ev_start.record(stream)
        success = allreduce_fn()
        ev_end.record(stream)
        stream.synchronize()

        if not success:
            print(f"PE {rank}: AllReduce failed at iteration {i}")
            break

        elapsed = ev_start.elapsed_time(ev_end) / 1000.0
        if i >= warmup:
            exec_times.append(elapsed)
        elif rank == 0:
            print(f"  Warmup {i + 1}/{warmup}: {elapsed * 1000:.3f} ms")

    return exec_times


def _print_stats(exec_times, data_bytes, npes, rank, label):
    """Aggregate and print performance statistics."""
    if len(exec_times) > 0:
        avg_time = np.mean(exec_times)
        min_time = np.min(exec_times)
        max_time = np.max(exec_times)
    else:
        avg_time = min_time = max_time = 0.0

    min_t = torch.tensor([min_time], dtype=torch.float64)
    max_t = torch.tensor([max_time], dtype=torch.float64)
    avg_t = torch.tensor([avg_time], dtype=torch.float64)

    dist.all_reduce(min_t, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_t, op=dist.ReduceOp.MAX)
    dist.all_reduce(avg_t, op=dist.ReduceOp.SUM)

    if rank == 0:
        g_min = min_t.item()
        g_max = max_t.item()
        g_avg = avg_t.item() / npes
        algo_bw = data_bytes / g_avg / (1024.0**3) if g_avg > 0 else 0
        bus_bw = algo_bw * 2 * (npes - 1) / npes if npes > 1 else algo_bw

        print(f"\n--- {label} Performance ---")
        print(f"  Min time : {g_min * 1000:.3f} ms")
        print(f"  Max time : {g_max * 1000:.3f} ms")
        print(f"  Avg time : {g_avg * 1000:.3f} ms")
        print(f"  Algo BW  : {algo_bw:.2f} GB/s")
        print(f"  Bus BW   : {bus_bw:.2f} GB/s")
        print(f"  Data size: {data_bytes / (1024.0 ** 2):.2f} MB per rank")

    return avg_time


# RCCL/NCCL doesn't support uint32; map to int32 for the baseline benchmark.
_RCCL_DTYPE_MAP = {
    torch.uint32: torch.int32,
    torch.int32: torch.int32,
    torch.float16: torch.float16,
    torch.bfloat16: torch.bfloat16,
    torch.float32: torch.float32,
}


# ---------------------------------------------------------------------------
#  Individual test helpers
# ---------------------------------------------------------------------------


def _test_outplace(
    rank,
    my_pe,
    npes,
    elems,
    data_bytes,
    output_buf_size,
    fill_value,
    dtype,
    dtype_name,
    device,
    stream,
    iterations,
    warmup,
):
    """Test 1: out-of-place (copy_output_to_user=False)."""
    if rank == 0:
        print(f"\n>>> Test 1: Out-of-place (copy_output_to_user=False, {dtype_name})")

    ar = AllreduceSdma(
        my_pe,
        npes,
        input_buffer_size=data_bytes,
        output_buffer_size=output_buf_size,
        copy_output_to_user=False,
        dtype=dtype,
    )

    input_tensor = torch.full((elems,), fill_value, dtype=dtype, device=device)
    output_tensor = torch.zeros(elems, dtype=dtype, device=device)

    torch.cuda.synchronize()
    dist.barrier()

    allreduce = ar

    def _bench_outplace():
        return allreduce(input_tensor, output_tensor, elems, stream)

    times = _run_benchmark(
        _bench_outplace,
        iterations,
        warmup,
        stream,
        rank,
    )

    transit_buf = ar.get_output_transit_buffer(dtype=input_tensor.dtype)
    transit_cpu = _to_numpy(transit_buf.cpu())
    ok = _verify_allreduce_result(
        transit_cpu, elems, my_pe, npes, f"outplace/{dtype_name}", dtype=dtype
    )

    out_cpu = _to_numpy(output_tensor.cpu())
    zero_check = (
        np.allclose(out_cpu, 0)
        if dtype not in (torch.uint32, torch.int32)
        else np.all(out_cpu == 0)
    )
    if zero_check and rank == 0:
        print(f"  PE {rank}: output_tensor correctly untouched (all zeros)")

    dist.barrier()
    _print_stats(times, data_bytes, npes, rank, f"Out-of-place ({dtype_name})")

    del ar
    return ok


def _test_inplace(
    rank,
    my_pe,
    npes,
    elems,
    data_bytes,
    output_buf_size,
    fill_value,
    dtype,
    dtype_name,
    device,
    stream,
    iterations,
    warmup,
):
    """Test 2: in-place."""
    if rank == 0:
        print(f"\n>>> Test 2: In-place (allreduce_inplace, {dtype_name})")

    ar = AllreduceSdma(
        my_pe,
        npes,
        input_buffer_size=data_bytes,
        output_buffer_size=output_buf_size,
        copy_output_to_user=False,
        dtype=dtype,
    )

    inplace_tensor = torch.full((elems,), fill_value, dtype=dtype, device=device)

    torch.cuda.synchronize()
    dist.barrier()

    # Single-shot correctness verification
    inplace_tensor.fill_(fill_value)
    stream.synchronize()
    ar.allreduce_inplace(inplace_tensor, elems, stream)
    stream.synchronize()

    inp_cpu = _to_numpy(inplace_tensor.cpu())
    ok = _verify_allreduce_result(
        inp_cpu, elems, my_pe, npes, f"inplace/{dtype_name}", dtype=dtype
    )
    if rank == 0 and ok:
        expected = sum((pe + 1) * 1000 for pe in range(npes))
        print(f"  PE {rank}: inplace result verified (all values ~ {expected})")

    dist.barrier()

    allreduce = ar

    def _inplace_with_refill():
        inplace_tensor.fill_(fill_value)
        return allreduce.allreduce_inplace(inplace_tensor, elems, stream)

    times = _run_benchmark(_inplace_with_refill, iterations, warmup, stream, rank)

    dist.barrier()
    _print_stats(times, data_bytes, npes, rank, f"In-place ({dtype_name})")

    del ar
    return ok


def _rccl_verify(cpu_data, elems, npes, rccl_dtype, rank, label):
    """Check RCCL allreduce result."""
    expected_value = sum((pe + 1) * 1000 for pe in range(npes))
    first_n = min(elems, len(cpu_data))
    if rccl_dtype in (torch.float16, torch.bfloat16):
        ok = np.allclose(
            cpu_data[:first_n].astype(np.float32), expected_value, rtol=1e-2, atol=1.0
        )
    else:
        ok = np.all(cpu_data[:first_n] == expected_value)
    if rank == 0:
        print(f"  PE {rank}: RCCL {label} correctness {'PASSED' if ok else 'FAILED'}")
    return ok


def _test_rccl_outplace(
    rank,
    my_pe,
    npes,
    elems,
    data_bytes,
    fill_value,
    dtype,
    dtype_name,
    device,
    stream,
    iterations,
    warmup,
):
    """Test 3: RCCL out-of-place — allreduce into a separate output tensor."""
    rccl_dtype = _RCCL_DTYPE_MAP.get(dtype, torch.float32)
    rccl_dtype_name = str(rccl_dtype).split(".")[-1]
    if rank == 0:
        print(
            f"\n>>> Test 3: RCCL out-of-place (torch.distributed, "
            f"{dtype_name}→{rccl_dtype_name})"
        )

    rccl_fill = (
        float(fill_value)
        if rccl_dtype in (torch.float16, torch.bfloat16)
        else fill_value
    )

    input_tensor = torch.full((elems,), rccl_fill, dtype=rccl_dtype, device=device)
    output_tensor = torch.zeros(elems, dtype=rccl_dtype, device=device)

    torch.cuda.synchronize()
    dist.barrier()

    # Correctness check
    output_tensor.copy_(input_tensor)
    dist.all_reduce(output_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    ok = _rccl_verify(
        _to_numpy(output_tensor.cpu()), elems, npes, rccl_dtype, rank, "outplace"
    )

    dist.barrier()

    # Benchmark — dist.all_reduce runs on RCCL's internal stream, so use
    # torch.cuda.synchronize() (all-stream barrier) + wall-clock timing.
    exec_times = []
    for i in range(warmup + iterations):
        output_tensor.copy_(input_tensor)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(output_tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if i >= warmup:
            exec_times.append(elapsed)
        elif rank == 0:
            print(f"  Warmup {i + 1}/{warmup}: {elapsed * 1000:.3f} ms")

    dist.barrier()
    _print_stats(
        exec_times, data_bytes, npes, rank, f"RCCL out-of-place ({rccl_dtype_name})"
    )
    return ok


def _test_rccl_inplace(
    rank,
    my_pe,
    npes,
    elems,
    data_bytes,
    fill_value,
    dtype,
    dtype_name,
    device,
    stream,
    iterations,
    warmup,
):
    """Test 4: RCCL in-place — allreduce directly on the tensor."""
    rccl_dtype = _RCCL_DTYPE_MAP.get(dtype, torch.float32)
    rccl_dtype_name = str(rccl_dtype).split(".")[-1]
    if rank == 0:
        print(
            f"\n>>> Test 4: RCCL in-place (torch.distributed, "
            f"{dtype_name}→{rccl_dtype_name})"
        )

    rccl_fill = (
        float(fill_value)
        if rccl_dtype in (torch.float16, torch.bfloat16)
        else fill_value
    )

    inplace_tensor = torch.full((elems,), rccl_fill, dtype=rccl_dtype, device=device)

    torch.cuda.synchronize()
    dist.barrier()

    # Correctness check
    inplace_tensor.fill_(rccl_fill)
    dist.all_reduce(inplace_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    ok = _rccl_verify(
        _to_numpy(inplace_tensor.cpu()), elems, npes, rccl_dtype, rank, "inplace"
    )

    dist.barrier()

    # Benchmark — same wall-clock approach as outplace.
    exec_times = []
    for i in range(warmup + iterations):
        inplace_tensor.fill_(rccl_fill)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(inplace_tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if i >= warmup:
            exec_times.append(elapsed)
        elif rank == 0:
            print(f"  Warmup {i + 1}/{warmup}: {elapsed * 1000:.3f} ms")

    dist.barrier()
    _print_stats(
        exec_times, data_bytes, npes, rank, f"RCCL in-place ({rccl_dtype_name})"
    )
    return ok


# GEMM size for overlap benchmarks (torch.matmul only; no aiter).
_GEMM_M_DEFAULT = 4096
_GEMM_N_DEFAULT = 4096
_GEMM_K_DEFAULT = 4096


def _print_overlap_summary(
    rank, npes, label, seq_ar, seq_gemm, overlap_times, data_bytes
):
    """Reduce timing lists across ranks; print on rank 0 (seconds)."""
    if len(seq_ar) == 0 or len(seq_gemm) == 0 or len(overlap_times) == 0:
        if rank == 0:
            print(f"  [{label}] insufficient samples for summary")
        return

    def _reduce_scalars(vals):
        t = torch.tensor(
            [min(vals), max(vals), sum(vals) / len(vals)],
            dtype=torch.float64,
            device=torch.device("cuda", rank),
        )
        mn = t[0].clone()
        mx = t[1].clone()
        avg = t[2].clone()
        dist.all_reduce(mn, op=dist.ReduceOp.MIN)
        dist.all_reduce(mx, op=dist.ReduceOp.MAX)
        dist.all_reduce(avg, op=dist.ReduceOp.SUM)
        return mn.item(), mx.item(), avg.item() / npes

    g_ar = _reduce_scalars(seq_ar)
    g_gm = _reduce_scalars(seq_gemm)
    g_ov = _reduce_scalars(overlap_times)

    if rank == 0:
        seq_sum = g_ar[2] + g_gm[2]
        print(f"\n--- GEMM overlap: {label} ---")
        print(f"  Data (allreduce): {data_bytes / (1024**2):.2f} MB per rank")
        print(
            f"  Sequential allreduce avg : {g_ar[2] * 1000:.3f} ms (min {g_ar[0]*1000:.3f})"
        )
        print(
            f"  Sequential GEMM avg      : {g_gm[2] * 1000:.3f} ms (min {g_gm[0]*1000:.3f})"
        )
        print(f"  Sum (sequential bound)   : {seq_sum * 1000:.3f} ms")
        print(
            f"  Overlap wall avg         : {g_ov[2] * 1000:.3f} ms (min {g_ov[0]*1000:.3f})"
        )
        if seq_sum > 0:
            print(f"  Overlap vs sequential sum: {g_ov[2] / seq_sum:.3f}x")


def _test_gemm_overlap_comparison(
    rank,
    my_pe,
    npes,
    elems,
    data_bytes,
    output_buf_size,
    fill_value,
    dtype,
    dtype_name,
    device,
    iterations,
    warmup,
    gemm_m=_GEMM_M_DEFAULT,
    gemm_n=_GEMM_N_DEFAULT,
    gemm_k=_GEMM_K_DEFAULT,
    num_stages=2,
):
    """SDMA (copy True/False) vs RCCL allreduce overlapped with torch.matmul.

    No payload correctness checks here (Tests 1–4); only launch success and timing.
    """
    stream_ar = torch.cuda.Stream(device=device)
    stream_gemm = torch.cuda.Stream(device=device)
    rccl_dtype = _RCCL_DTYPE_MAP.get(dtype, torch.float32)
    rccl_fill = (
        float(fill_value)
        if rccl_dtype in (torch.float16, torch.bfloat16)
        else fill_value
    )
    total_iters = warmup + iterations

    A = torch.randn(gemm_m, gemm_k, dtype=torch.float32, device=device)
    B = torch.randn(gemm_k, gemm_n, dtype=torch.float32, device=device)

    def _run_gemm():
        return torch.matmul(A, B)

    ev_ar_s = torch.cuda.Event(enable_timing=True)
    ev_ar_e = torch.cuda.Event(enable_timing=True)
    ev_g_s = torch.cuda.Event(enable_timing=True)
    ev_g_e = torch.cuda.Event(enable_timing=True)
    ov_s = torch.cuda.Event(enable_timing=True)
    ov_e = torch.cuda.Event(enable_timing=True)

    all_ok = True

    def bench_one(label, setup_verify_and_launch_ar):
        """setup returns (ok, launch, prep, time_ar_with_wall).

        prep runs each iteration before the timed section (untimed), e.g.
        inp/buffer fill_. launch is timed only: SDMA ar(...) or RCCL all_reduce.
        time_ar_with_wall: True -> perf_counter around launch (RCCL); False ->
        CUDA events on stream_ar (SDMA).
        """
        nonlocal all_ok
        setup_ret = setup_verify_and_launch_ar()
        if len(setup_ret) != 4:
            raise TypeError(
                f"{label}: setup must return (ok, launch, prep, time_ar_with_wall)"
            )
        ok_c, launch_ar, prep_ar, time_ar_with_wall = setup_ret

        ok_pe = torch.tensor([1 if ok_c else 0], dtype=torch.int32, device=device)
        dist.all_reduce(ok_pe, op=dist.ReduceOp.MIN)
        ok_c = ok_pe.item() == 1
        all_ok = all_ok and ok_c
        if not ok_c:
            if rank == 0:
                print(f"  [{label}] setup failed on some PE, skipping bench")
            dist.barrier()
            return

        seq_ar, seq_gemm, overlap_times = [], [], []

        for i in range(total_iters):
            torch.cuda.synchronize()
            prep_ar()
            torch.cuda.synchronize()
            if time_ar_with_wall:
                t0 = time.perf_counter()
                for _ in range(num_stages):
                    launch_ar()
                torch.cuda.synchronize()
                t_ar = time.perf_counter() - t0
            else:
                ev_ar_s.record(stream_ar)
                with torch.cuda.stream(stream_ar):
                    for _ in range(num_stages):
                        launch_ar()
                ev_ar_e.record(stream_ar)
                stream_ar.synchronize()
                t_ar = ev_ar_s.elapsed_time(ev_ar_e) / 1000.0
            if i >= warmup:
                seq_ar.append(t_ar)

        for i in range(total_iters):
            torch.cuda.synchronize()
            ev_g_s.record(stream_gemm)
            with torch.cuda.stream(stream_gemm):
                for _ in range(num_stages):
                    C = _run_gemm()
            ev_g_e.record(stream_gemm)
            stream_gemm.synchronize()
            t_g = ev_g_s.elapsed_time(ev_g_e) / 1000.0
            if i >= warmup:
                seq_gemm.append(t_g)
            elif rank == 0:
                _ = C.sum().item()

        # Per-stage events for overlap timing and cross-stream dependencies
        ev_g_s_list = [torch.cuda.Event(enable_timing=True) for _ in range(num_stages)]
        ev_g_e_list = [torch.cuda.Event(enable_timing=True) for _ in range(num_stages)]
        ev_ar_s_list = [torch.cuda.Event(enable_timing=True) for _ in range(num_stages)]
        ev_ar_e_list = [torch.cuda.Event(enable_timing=True) for _ in range(num_stages)]

        for i in range(total_iters):
            torch.cuda.synchronize()
            prep_ar()
            torch.cuda.synchronize()
            ov_s.record()

            # Enqueue all gemms on stream_gemm back-to-back
            for s in range(num_stages):
                ev_g_s_list[s].record(stream_gemm)
                with torch.cuda.stream(stream_gemm):
                    C = _run_gemm()
                ev_g_e_list[s].record(stream_gemm)

            # Enqueue all ars on stream_ar: each ar[s] waits for gemm[s]
            for s in range(num_stages):
                stream_ar.wait_event(ev_g_e_list[s])
                ev_ar_s_list[s].record(stream_ar)
                with torch.cuda.stream(stream_ar):
                    launch_ar()
                ev_ar_e_list[s].record(stream_ar)

            stream_ar.synchronize()
            stream_gemm.synchronize()
            ov_e.record()
            torch.cuda.synchronize()
            t_ov = ov_s.elapsed_time(ov_e) / 1000.0
            if i >= warmup:
                overlap_times.append(t_ov)
            elif rank == 0:
                _ = C.sum().item()

        dist.barrier()
        _print_overlap_summary(
            rank, npes, label, seq_ar, seq_gemm, overlap_times, data_bytes
        )

    if rank == 0:
        print(f"\n>>> Test 5: GEMM overlap (torch.matmul {gemm_m}x{gemm_k}x{gemm_n})")
        print(
            "    Compare SDMA allreduce vs RCCL with concurrent GEMM on another stream"
        )

    torch.cuda.synchronize()
    dist.barrier()

    # --- SDMA copy_output_to_user=True ---
    def setup_sdma_copy_true():
        ar = AllreduceSdma(
            my_pe,
            npes,
            input_buffer_size=data_bytes,
            output_buffer_size=output_buf_size,
            copy_output_to_user=True,
            dtype=dtype,
        )
        inp = torch.full((elems,), fill_value, dtype=dtype, device=device)
        out = torch.zeros(elems, dtype=dtype, device=device)

        stream_ar.synchronize()
        ok = ar(inp, out, elems, stream_ar)
        stream_ar.synchronize()

        def prep():
            inp.fill_(fill_value)

        def launch():
            return ar(inp, out, elems, stream_ar)

        if not ok and rank == 0:
            print("  SDMA copy=True setup ar() failed")
        return ok, launch, prep, False

    bench_one(
        f"SDMA allreduce copy_output_to_user=True ({dtype_name}; prep fill_ untimed)",
        setup_sdma_copy_true,
    )

    # --- SDMA copy_output_to_user=False ---
    def setup_sdma_copy_false():
        ar = AllreduceSdma(
            my_pe,
            npes,
            input_buffer_size=data_bytes,
            output_buffer_size=output_buf_size,
            copy_output_to_user=False,
            dtype=dtype,
        )
        inp = torch.full((elems,), fill_value, dtype=dtype, device=device)
        out = torch.zeros(elems, dtype=dtype, device=device)

        stream_ar.synchronize()
        ok = ar(inp, out, elems, stream_ar)
        stream_ar.synchronize()

        def prep():
            inp.fill_(fill_value)

        def launch():
            return ar(inp, out, elems, stream_ar)

        if not ok and rank == 0:
            print("  SDMA copy=False setup ar() failed")
        return ok, launch, prep, False

    bench_one(
        f"SDMA allreduce copy_output_to_user=False ({dtype_name}; prep fill_ untimed)",
        setup_sdma_copy_false,
    )

    # --- RCCL: timed section is only dist.all_reduce; fill_ runs before each iter (outside events / wall slice). ---
    def setup_rccl():
        buf = torch.full((elems,), rccl_fill, dtype=rccl_dtype, device=device)

        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        def prep():
            buf.fill_(rccl_fill)

        def launch():
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            return True

        return True, launch, prep, True

    rccl_name = str(rccl_dtype).split(".")[-1]
    bench_one(
        f"RCCL all_reduce only ({dtype_name}→{rccl_name}; prep fill_ untimed)",
        setup_rccl,
    )

    torch.cuda.synchronize()
    dist.barrier()
    return all_ok


# ---------------------------------------------------------------------------
#  Main worker
# ---------------------------------------------------------------------------


def _test_allreduce(
    rank,
    world_size,
    port,
    elems,
    iterations,
    warmup,
    dtype=torch.uint32,
    test_gemm_overlap=False,
    gemm_m=_GEMM_M_DEFAULT,
    gemm_n=_GEMM_N_DEFAULT,
    gemm_k=_GEMM_K_DEFAULT,
    num_stages=2,
):
    """Worker function for each process."""

    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")

        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        assert my_pe == rank
        assert npes == world_size

        elem_size = torch.tensor([], dtype=dtype).element_size()
        data_bytes = elems * elem_size
        output_buf_size = npes * (elems // npes + 64) * elem_size
        dtype_name = str(dtype).split(".")[-1]

        if rank == 0:
            print(f"\n{'=' * 70}")
            print(f"AllReduce SDMA Test (dtype={dtype_name})")
            print(f"  World size      : {world_size}")
            print(f"  Elements per PE : {elems:,}")
            print(f"  Data size       : {data_bytes / (1024 ** 2):.2f} MB per rank")
            print(f"  Iterations      : {iterations} (warmup: {warmup})")
            print(f"{'=' * 70}")

        device = torch.device(f"cuda:{rank}")
        stream = torch.cuda.Stream(device=device)
        fill_value = (my_pe + 1) * 1000

        ok1 = _test_outplace(
            rank,
            my_pe,
            npes,
            elems,
            data_bytes,
            output_buf_size,
            fill_value,
            dtype,
            dtype_name,
            device,
            stream,
            iterations,
            warmup,
        )
        ok2 = _test_inplace(
            rank,
            my_pe,
            npes,
            elems,
            data_bytes,
            output_buf_size,
            fill_value,
            dtype,
            dtype_name,
            device,
            stream,
            iterations,
            warmup,
        )
        ok3 = _test_rccl_outplace(
            rank,
            my_pe,
            npes,
            elems,
            data_bytes,
            fill_value,
            dtype,
            dtype_name,
            device,
            stream,
            iterations,
            warmup,
        )
        ok4 = _test_rccl_inplace(
            rank,
            my_pe,
            npes,
            elems,
            data_bytes,
            fill_value,
            dtype,
            dtype_name,
            device,
            stream,
            iterations,
            warmup,
        )

        all_ok = ok1 and ok2 and ok3 and ok4

        if test_gemm_overlap:
            ok5 = _test_gemm_overlap_comparison(
                rank,
                my_pe,
                npes,
                elems,
                data_bytes,
                output_buf_size,
                fill_value,
                dtype,
                dtype_name,
                device,
                iterations,
                warmup,
                gemm_m=gemm_m,
                gemm_n=gemm_n,
                gemm_k=gemm_k,
                num_stages=num_stages,
            )
            all_ok = all_ok and ok5

        # --- Final summary ---
        ok_tensor = torch.tensor([1 if all_ok else 0], dtype=torch.int32)
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            passed = ok_tensor.item()
            print(f"\n{'=' * 70}")
            print(f"PEs passed ({dtype_name}): {passed}/{npes}")
            if passed == npes:
                print(f"=== All Tests PASSED ({dtype_name}) ===")
            else:
                print(f"=== SOME Tests FAILED ({dtype_name}) ===")
            print(f"{'=' * 70}\n")

        torch.cuda.synchronize()
        dist.barrier()
        shmem.shmem_finalize()

        if not all_ok:
            raise AssertionError(
                f"PE {rank}: AllReduce verification failed ({dtype_name})"
            )


_DTYPE_MAP = {
    "uint32": torch.uint32,
    "int32": torch.int32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def test_allreduce(
    elems=67108864,
    world_size=8,
    iterations=10,
    warmup=10,
    dtype=torch.uint32,
    test_gemm_overlap=False,
    gemm_m=_GEMM_M_DEFAULT,
    gemm_n=_GEMM_N_DEFAULT,
    gemm_k=_GEMM_K_DEFAULT,
    num_stages=2,
):
    """Run AllReduce SDMA test."""
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    port = get_free_port()
    torch.multiprocessing.spawn(
        _test_allreduce,
        args=(
            world_size,
            port,
            elems,
            iterations,
            warmup,
            dtype,
            test_gemm_overlap,
            gemm_m,
            gemm_n,
            gemm_k,
            num_stages,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test AllReduce SDMA (correctness + bandwidth)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--elems", type=int, default=67108864, help="Elements per PE")
    parser.add_argument("--world-size", type=int, default=8, help="Number of processes")
    parser.add_argument(
        "--iterations", type=int, default=10, help="Measurement iterations"
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument(
        "--enable-sdma", type=int, default=1, choices=[0, 1], help="Enable SDMA"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint32",
        choices=list(_DTYPE_MAP.keys()),
        help="Data type (uint32, fp16, bf16)",
    )
    parser.add_argument(
        "--test-gemm-overlap",
        action="store_true",
        help="Run SDMA vs RCCL allreduce overlap vs torch.matmul (extra benchmark)",
    )
    parser.add_argument(
        "--gemm-m",
        type=int,
        default=_GEMM_M_DEFAULT,
        help="GEMM M dimension for overlap test",
    )
    parser.add_argument(
        "--gemm-n",
        type=int,
        default=_GEMM_N_DEFAULT,
        help="GEMM N dimension for overlap test",
    )
    parser.add_argument(
        "--gemm-k",
        type=int,
        default=_GEMM_K_DEFAULT,
        help="GEMM K dimension for overlap test",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=2,
        help="Number of pipelined gemm+ar stages for overlap test",
    )
    args = parser.parse_args()
    os.environ["MORI_ENABLE_SDMA"] = str(args.enable_sdma)

    dtype = _DTYPE_MAP[args.dtype]
    dtype_name = str(dtype).split(".")[-1]

    print("AllReduce SDMA Test")
    print(f"  Elements per PE : {args.elems:,}")
    print(f"  World size      : {args.world_size}")
    print(f"  Iterations      : {args.iterations}")
    print(f"  Warmup          : {args.warmup}")
    print(f"  Dtype           : {dtype_name}")
    print(f"  GEMM overlap    : {args.test_gemm_overlap}")
    if args.test_gemm_overlap:
        print(
            f"  GEMM size       : {args.gemm_m}x{args.gemm_k}x{args.gemm_n} (torch.matmul)"
        )
        print(f"  Num stages      : {args.num_stages}")
    print("-" * 60)

    test_allreduce(
        args.elems,
        args.world_size,
        args.iterations,
        args.warmup,
        dtype,
        test_gemm_overlap=args.test_gemm_overlap,
        gemm_m=args.gemm_m,
        gemm_n=args.gemm_n,
        gemm_k=args.gemm_k,
        num_stages=args.num_stages,
    )
