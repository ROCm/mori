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
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so "tests.python.utils" resolves
# when running this script directly (python tests/python/ccl/test_allreduce.py).
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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


def _run_benchmark(allreduce_fn, iterations, warmup, stream, rank, setup_fn=None):
    """Run warmup + measurement iterations, return list of measured times."""
    exec_times = []
    total_iters = warmup + iterations

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)

    for i in range(total_iters):
        if setup_fn:
            setup_fn()
        torch.cuda.synchronize()
        dist.barrier()
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


def _test_copy_to_user_verify(
    rank, my_pe, npes, elems, data_bytes, output_buf_size, fill_value,
    dtype, dtype_name, device, stream, iterations, warmup,
):
    """Test 1b: copy_output_to_user=True correctness.

    Covers the D' fast path when MORI_REGISTER_USER_OUTPUT=1: the user
    output tensor is shmem-registered, AR kernel writes directly into it,
    external hipMemcpyAsync is skipped. Result is read from the USER'S
    output tensor and validated against expected allreduce sum.
    """
    if rank == 0:
        print(f"\n>>> Test 1b: Copy-to-user (copy_output_to_user=True, "
              f"+ MORI_REGISTER_USER_OUTPUT if set, {dtype_name})")

    ar = AllreduceSdma(
        my_pe, npes,
        input_buffer_size=data_bytes,
        output_buffer_size=output_buf_size,
        copy_output_to_user=True,
        dtype=dtype,
    )
    if os.environ.get("MORI_REGISTER_USER_OUTPUT", "0") == "1":
        try:
            ar.enable_register_user_output(True)
        except Exception as e:
            if rank == 0:
                print(f"  [warn] enable_register_user_output failed: {e}",
                      flush=True)
    # Direction θ: multi-qId AG correctness check
    if os.environ.get("MORI_AG_MULTI_Q", "0") == "1":
        try:
            ar.enable_ag_multi_q(True)
        except Exception as e:
            if rank == 0:
                print(f"  [warn] enable_ag_multi_q failed: {e}", flush=True)

    input_tensor = torch.full((elems,), fill_value, dtype=dtype, device=device)
    output_tensor = torch.zeros(elems, dtype=dtype, device=device)

    # Pre-register (collective) so fast path is hit on every call
    if os.environ.get("MORI_REGISTER_USER_OUTPUT", "0") == "1":
        try:
            ar.register_user_output(output_tensor.data_ptr(),
                                    output_tensor.numel() * output_tensor.element_size())
        except Exception as e:
            if rank == 0:
                print(f"  [warn] register_user_output failed: {e}",
                      flush=True)

    torch.cuda.synchronize(); dist.barrier()

    def _bench():
        return ar(input_tensor, output_tensor, elems, stream)

    times = _run_benchmark(_bench, iterations, warmup, stream, rank)

    out_cpu = _to_numpy(output_tensor.cpu())
    ok = _verify_allreduce_result(
        out_cpu, elems, my_pe, npes, f"copy_to_user/{dtype_name}", dtype=dtype
    )
    if rank == 0 and ok:
        hits = ar._handle.cache_hits()
        misses = ar._handle.cache_misses()
        print(f"  PE {rank}: copy-to-user correctness PASSED  "
              f"(cache hits={hits}, misses={misses})", flush=True)

    dist.barrier()
    _print_stats(times, data_bytes, npes, rank, f"Copy-to-user ({dtype_name})")

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

    # ---------- Single-shot correctness verification ----------
    expected_val = sum((pe + 1) * 1000 for pe in range(npes))

    # Test C: inplace as the FIRST call on this AR (no prior outplace)
    inplace_tensor.fill_(fill_value)
    torch.cuda.synchronize()
    ar.allreduce_inplace(inplace_tensor, elems, stream)
    stream.synchronize()

    transit_buf = ar.get_output_transit_buffer(dtype=dtype)
    transit_cpu = _to_numpy(transit_buf.cpu())
    transit_ok = _verify_allreduce_result(
        transit_cpu, elems, my_pe, npes, f"inplace-transit/{dtype_name}", dtype=dtype
    )

    inp_cpu = _to_numpy(inplace_tensor.cpu())
    ok = _verify_allreduce_result(
        inp_cpu, elems, my_pe, npes, f"inplace/{dtype_name}", dtype=dtype
    )

    if rank == 0:
        print(f"  [diag] inplace (1st call): transit={'PASS' if transit_ok else 'FAIL'}, "
              f"data={'PASS' if ok else 'FAIL'}")
        if not transit_ok:
            for shard in range(npes):
                s = shard * (elems // npes)
                print(f"    shard {shard} [idx {s}]: transit={transit_cpu[s]}, data={inp_cpu[s]}")

    # Test D: inplace as the SECOND call (transit has stale reduce result)
    dist.barrier()
    if rank == 0:
        print(f"  --- testing 2nd call (transit has stale reduce from 1st call) ---")
    inplace_tensor.fill_(fill_value)
    torch.cuda.synchronize()
    ar.allreduce_inplace(inplace_tensor, elems, stream)
    stream.synchronize()

    transit_buf2 = ar.get_output_transit_buffer(dtype=dtype)
    transit_cpu2 = _to_numpy(transit_buf2.cpu())
    transit_ok2 = _verify_allreduce_result(
        transit_cpu2, elems, my_pe, npes, f"inplace2-transit/{dtype_name}", dtype=dtype
    )
    inp_cpu2 = _to_numpy(inplace_tensor.cpu())
    ok2 = _verify_allreduce_result(
        inp_cpu2, elems, my_pe, npes, f"inplace2/{dtype_name}", dtype=dtype
    )
    if rank == 0:
        print(f"  [diag] inplace (2nd call): transit={'PASS' if transit_ok2 else 'FAIL'}, "
              f"data={'PASS' if ok2 else 'FAIL'}")
        if not transit_ok2:
            for shard in range(npes):
                s = shard * (elems // npes)
                print(f"    shard {shard} [idx {s}]: transit={transit_cpu2[s]}, data={inp_cpu2[s]}")

    ok = ok and ok2
    if rank == 0 and ok:
        print(f"  PE {rank}: inplace result verified (all values ~ {expected_val})")

    dist.barrier()

    allreduce = ar

    def _inplace_setup():
        inplace_tensor.fill_(fill_value)

    def _inplace_fn():
        return allreduce.allreduce_inplace(inplace_tensor, elems, stream)

    times = _run_benchmark(
        _inplace_fn, iterations, warmup, stream, rank, setup_fn=_inplace_setup
    )

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
        dist.barrier()
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
        dist.barrier()
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
        print(f"  Sequential allreduce avg : {g_ar[2] * 1000:.3f} ms (min {g_ar[0]*1000:.3f})")
        print(f"  Sequential GEMM avg      : {g_gm[2] * 1000:.3f} ms (min {g_gm[0]*1000:.3f})")
        print(f"  Sum (sequential bound)   : {seq_sum * 1000:.3f} ms")
        print(f"  Overlap wall avg         : {g_ov[2] * 1000:.3f} ms (min {g_ov[0]*1000:.3f})")
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

        for i in range(total_iters):
            torch.cuda.synchronize()
            ev_g_s.record(stream_gemm)
            with torch.cuda.stream(stream_gemm):
                C = _run_gemm()
            ev_g_e.record(stream_gemm)
            stream_gemm.synchronize()
            t_g = ev_g_s.elapsed_time(ev_g_e) / 1000.0
            if i >= warmup:
                seq_gemm.append(t_g)
            elif rank == 0:
                _ = C.sum().item()

        for i in range(total_iters):
            torch.cuda.synchronize()
            prep_ar()
            torch.cuda.synchronize()
            ov_s.record()
            ev_ar_s.record(stream_ar)
            with torch.cuda.stream(stream_ar):
                launch_ar()
            ev_ar_e.record(stream_ar)
            ev_g_s.record(stream_gemm)
            with torch.cuda.stream(stream_gemm):
                C = _run_gemm()
            ev_g_e.record(stream_gemm)
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
        print("    Compare SDMA allreduce vs RCCL with concurrent GEMM on another stream")

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

        torch.cuda.synchronize()
        dist.barrier()
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

        torch.cuda.synchronize()
        dist.barrier()
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
#  Test 6: Multi-stage pipelined GEMM + AllReduce overlap (with size sweep)
# ---------------------------------------------------------------------------

_SWEEP_SIZES_MB = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def _bench_overlap_one_size(
    rank, my_pe, npes, elems, fill_value, dtype, device,
    stream_ar, stream_gemm, _run_gemm,
    ev_ar_s, ev_ar_e, ev_g_s, ev_g_e, ov_s, ov_e,
    num_stages, total_iters, warmup,
    setup_fn,
    timeline_dump=False,
    timeline_label="",
    ar_phase_timing=False,
):
    """Benchmark one (mode, size) combo.  Returns dict with timing or None."""
    _dbg = (rank == 0)
    setup_ret = setup_fn()
    ok_c = setup_ret[0] if len(setup_ret) >= 4 else False
    ok_pe = torch.tensor([1 if ok_c else 0], dtype=torch.int32, device=device)
    dist.all_reduce(ok_pe, op=dist.ReduceOp.MIN)
    if ok_pe.item() != 1:
        dist.barrier()
        return None
    _, launch_ar, prep_ar, time_ar_with_wall = setup_ret[:4]
    copy_stream = setup_ret[4] if len(setup_ret) > 4 else None
    # Optional: AllreduceSdma object, used for --ar-phase-timing. None for RCCL.
    ar_obj = setup_ret[5] if len(setup_ret) > 5 else None

    seq_ar, seq_gemm, overlap_times = [], [], []
    timeline_samples = []  # populated when timeline_dump=True; one dict per measure iter

    if _dbg: print(f" seq_ar", end="", flush=True)
    for i in range(total_iters):
        torch.cuda.synchronize()
        prep_ar()
        torch.cuda.synchronize()
        if time_ar_with_wall:
            t0 = time.perf_counter()
            for _ in range(num_stages):
                launch_ar()
            if copy_stream:
                copy_stream.synchronize()
            torch.cuda.synchronize()
            t_ar = time.perf_counter() - t0
        else:
            ev_ar_s.record(stream_ar)
            with torch.cuda.stream(stream_ar):
                for _ in range(num_stages):
                    launch_ar()
            ev_ar_e.record(stream_ar)
            stream_ar.synchronize()
            if copy_stream:
                copy_stream.synchronize()
            t_ar = ev_ar_s.elapsed_time(ev_ar_e) / 1000.0
        if i >= warmup:
            seq_ar.append(t_ar)
        if _dbg and i < 3: print(f"[{i}]", end="", flush=True)

    torch.cuda.synchronize()
    dist.barrier()

    if _dbg: print(f" seq_gemm", end="", flush=True)
    for i in range(total_iters):
        torch.cuda.synchronize()
        ev_g_s.record(stream_gemm)
        with torch.cuda.stream(stream_gemm):
            for _ in range(num_stages):
                _run_gemm()
        ev_g_e.record(stream_gemm)
        stream_gemm.synchronize()
        t_g = ev_g_s.elapsed_time(ev_g_e) / 1000.0
        if i >= warmup:
            seq_gemm.append(t_g)

    torch.cuda.synchronize()
    dist.barrier()

    if _dbg: print(f" overlap", end="", flush=True)
    ev_g_s_list = [torch.cuda.Event(enable_timing=True) for _ in range(num_stages)]
    ev_g_e_list = [torch.cuda.Event(enable_timing=True) for _ in range(num_stages)]
    ev_ar_s_list = [torch.cuda.Event(enable_timing=True) for _ in range(num_stages)]
    ev_ar_e_list = [torch.cuda.Event(enable_timing=True) for _ in range(num_stages)]

    elem_bytes = elems * torch.tensor([], dtype=dtype).element_size()
    data_mb = elem_bytes / (1024 * 1024)
    use_overlap = data_mb >= 128
    for i in range(total_iters):
        torch.cuda.synchronize()
        prep_ar()
        torch.cuda.synchronize()
        ov_s.record()
        for s in range(num_stages):
            ev_g_s_list[s].record(stream_gemm)
            with torch.cuda.stream(stream_gemm):
                _run_gemm()
            ev_g_e_list[s].record(stream_gemm)
            if use_overlap:
                stream_ar.wait_event(ev_g_e_list[s])
            else:
                stream_gemm.synchronize()
            ev_ar_s_list[s].record(stream_ar)
            with torch.cuda.stream(stream_ar):
                launch_ar()
            ev_ar_e_list[s].record(stream_ar)
            if not use_overlap:
                stream_ar.synchronize()
                if copy_stream:
                    copy_stream.synchronize()
        if use_overlap:
            stream_ar.synchronize()
            if copy_stream:
                copy_stream.synchronize()
            stream_gemm.synchronize()
        ov_e.record()
        torch.cuda.synchronize()
        t_ov = ov_s.elapsed_time(ov_e) / 1000.0
        if i >= warmup:
            overlap_times.append(t_ov)
            if timeline_dump and rank == 0:
                # Per-stage offsets relative to ov_s (ms), median across measure iters
                iter_sample = {
                    "wall": ov_s.elapsed_time(ov_e),
                    "stages": [
                        {
                            "g_start": ov_s.elapsed_time(ev_g_s_list[s]),
                            "g_end": ov_s.elapsed_time(ev_g_e_list[s]),
                            "a_start": ov_s.elapsed_time(ev_ar_s_list[s]),
                            "a_end": ov_s.elapsed_time(ev_ar_e_list[s]),
                        }
                        for s in range(num_stages)
                    ],
                }
                timeline_samples.append(iter_sample)

    if timeline_dump and rank == 0 and len(timeline_samples) > 0:
        import statistics as _stats
        n = len(timeline_samples)

        def _med(values):
            return _stats.median(values)

        print(f"\n  === Per-Stage Timeline [{timeline_label}]  "
              f"N={num_stages}  size={data_mb:.0f} MB  "
              f"use_overlap={use_overlap}  samples={n} (median ms from ov_s) ===")
        print(f"  {'stage':>5s} | "
              f"{'GEMM start':>10s} {'GEMM end':>10s} {'GEMM dur':>9s} | "
              f"{'AR start':>9s} {'AR end':>9s} {'AR dur':>8s}")
        print(f"  {'-' * 5} | {'-' * 10} {'-' * 10} {'-' * 9} | "
              f"{'-' * 9} {'-' * 9} {'-' * 8}")
        stage_med = []
        for s in range(num_stages):
            g_st = _med([x["stages"][s]["g_start"] for x in timeline_samples])
            g_en = _med([x["stages"][s]["g_end"] for x in timeline_samples])
            a_st = _med([x["stages"][s]["a_start"] for x in timeline_samples])
            a_en = _med([x["stages"][s]["a_end"] for x in timeline_samples])
            stage_med.append((g_st, g_en, a_st, a_en))
            print(f"  {s:>5d} | "
                  f"{g_st:>10.3f} {g_en:>10.3f} {g_en - g_st:>9.3f} | "
                  f"{a_st:>9.3f} {a_en:>9.3f} {a_en - a_st:>8.3f}")

        # Pair-wise overlap: AR[s] vs GEMM[s+1]
        print(f"\n  === Overlap Pairs: AR[s] vs GEMM[s+1] ===")
        print(f"  {'pair':>12s} | {'overlap ms':>10s} | {'AR dur':>8s} | "
              f"{'GEMM dur':>9s} | {'hide % of min':>14s}")
        for s in range(num_stages - 1):
            _, _, a_st, a_en = stage_med[s]
            g_st, g_en, _, _ = stage_med[s + 1]
            ov_lo = max(a_st, g_st)
            ov_hi = min(a_en, g_en)
            ov_len = max(0.0, ov_hi - ov_lo)
            a_len = a_en - a_st
            g_len = g_en - g_st
            denom = min(a_len, g_len)
            pct = (ov_len / denom * 100.0) if denom > 0 else 0.0
            print(f"  AR[{s}]-GEMM[{s+1}] | "
                  f"{ov_len:>10.3f} | {a_len:>8.3f} | "
                  f"{g_len:>9.3f} | {pct:>13.1f}%")

        wall_med = _med([x["wall"] for x in timeline_samples])
        print(f"\n  === Totals ===")
        print(f"  median wall:        {wall_med:.3f} ms")
        # Edge (non-overlap) time estimate: GEMM[0] solo + AR[N-1] solo (if bottleneck)
        g0_len = stage_med[0][1] - stage_med[0][0]
        ar_last_len = stage_med[-1][3] - stage_med[-1][2]
        print(f"  GEMM[0] duration:   {g0_len:.3f} ms   (first GEMM, runs before any AR)")
        print(f"  AR[{num_stages-1}] duration:      {ar_last_len:.3f} ms   "
              f"(last AR, runs after all GEMMs)")
        print()

    # ------------------------------------------------------------------
    # AR-kernel phase-level breakdown (SDMA only; requires rebuilt kernel)
    #
    # COLLECTIVE: every rank must participate in the AR launch; only rank 0
    # reads/prints its own phase_ts buffer. Without collective participation
    # the AR kernel will hang forever on scatter/AG signal waits because
    # peers never submit their corresponding SDMA ops.
    # ------------------------------------------------------------------
    # phase_target_stage controls WHICH AR kernel gets its phase timestamps
    # read. default 0 (AR[0], cold path); set via --ar-phase-stage=N to
    # instrument AR[N] and compare (e.g. AR[2] as "warm/no-GEMM-contention").
    import os as _os_mod
    phase_target_stage = int(_os_mod.environ.get("MORI_PHASE_TARGET_STAGE", "0"))
    if ar_phase_timing and ar_obj is not None and use_overlap:
        try:
            n_samples = 5
            all_phase_deltas = []     # block-0 phases, only populated on rank 0
            all_cb_phase_deltas = []  # compute-block-1 phases, only populated on rank 0
            all_copy_timings = []     # copy-path timings, only populated on rank 0
            stage_med_ar0 = []        # only populated on rank 0

            # Turn on copy-path timing too. We only parse it for AR[0] (same
            # gating logic as phase timing): enable before AR[0] submit,
            # disable right after so AR[1..N-1] don't overwrite the events.
            try:
                ar_obj.enable_copy_timing(True)
            except Exception:
                pass

            if rank == 0 and phase_target_stage != 0:
                print(f"  [phase-timing] targeting AR[{phase_target_stage}] "
                      f"(set via MORI_PHASE_TARGET_STAGE)", flush=True)

            for _sample in range(n_samples):
                # Sync all ranks before each sample so cold path is
                # measured from the same starting point on every PE.
                torch.cuda.synchronize()
                prep_ar()
                torch.cuda.synchronize()
                dist.barrier()

                tgt_ar_s = torch.cuda.Event(enable_timing=True)
                tgt_ar_e = torch.cuda.Event(enable_timing=True)

                # Iterate over all stages; only enable phase+copy timing
                # exactly at stage == phase_target_stage. Pattern matches
                # the main overlap loop so scheduling state at the target
                # stage reproduces the real-run condition.
                for s in range(num_stages):
                    if s == phase_target_stage:
                        ar_obj.enable_phase_timing(True)
                        try:
                            ar_obj.enable_copy_timing(True)
                        except Exception:
                            pass

                    tmp_g_ev = torch.cuda.Event()
                    with torch.cuda.stream(stream_gemm):
                        _run_gemm()
                    tmp_g_ev.record(stream_gemm)
                    stream_ar.wait_event(tmp_g_ev)

                    if s == phase_target_stage:
                        tgt_ar_s.record(stream_ar)
                    with torch.cuda.stream(stream_ar):
                        launch_ar()
                    if s == phase_target_stage:
                        tgt_ar_e.record(stream_ar)
                        ar_obj.enable_phase_timing(False)
                        try:
                            ar_obj.enable_copy_timing(False)
                        except Exception:
                            pass

                # Sync everything, then read target stage's phase timestamps.
                stream_ar.synchronize()
                stream_gemm.synchronize()
                dist.barrier()
                ts = ar_obj.get_phase_timestamps()
                last_nc = ar_obj.get_last_num_chunks()

                # Read copy timing for target stage.
                try:
                    ct = ar_obj.get_copy_timing_ms()
                    if len(ct) == 2 and rank == 0:
                        all_copy_timings.append(ct)
                except Exception:
                    pass

                if rank == 0:
                    # Convert raw cycles to ms using target AR event duration.
                    ar0_dur_ms = tgt_ar_s.elapsed_time(tgt_ar_e)
                    stage_med_ar0.append(ar0_dur_ms)
                    entry_cy = ts[0]
                    exit_cy = ts[3 + 3 * last_nc]
                    total_cy = exit_cy - entry_cy if exit_cy > entry_cy else 1
                    cy_to_ms = ar0_dur_ms / float(total_cy) if total_cy > 0 else 0.0

                    phases = {}
                    phases["entry→scatter_done"] = (ts[1] - ts[0]) * cy_to_ms
                    for c in range(last_nc):
                        prev = ts[1] if c == 0 else ts[2 + 3 * (c - 1) + 2]
                        phases[f"c{c}:scatter→compute-wait"] = (ts[2 + 3 * c + 0] - prev) * cy_to_ms
                        phases[f"c{c}:compute-wait→barrier"] = (ts[2 + 3 * c + 1] - ts[2 + 3 * c + 0]) * cy_to_ms
                        phases[f"c{c}:barrier→AG-submit"] = (ts[2 + 3 * c + 2] - ts[2 + 3 * c + 1]) * cy_to_ms
                    phases[f"AG-submit→AG-wait-done"] = (ts[2 + 3 * last_nc] - ts[2 + 3 * (last_nc - 1) + 2]) * cy_to_ms
                    phases["AG-wait-done→exit"] = (ts[3 + 3 * last_nc] - ts[2 + 3 * last_nc]) * cy_to_ms

                    # Stage 2b-0 leftover: per-chunk AG done timestamps
                    # (slot 20+c). Shows when each chunk's AG signals arrived
                    # at block 0. Delta between c0→c1 AG done reveals whether
                    # the SDMA queue saturates bandwidth (large delta =
                    # serial, small delta = parallel / queue saturated
                    # already).
                    for c in range(last_nc):
                        if (20 + c) < len(ts) and ts[20 + c] != 0:
                            # since AG for chunk c is submitted at ts[2+3c+2],
                            # delta = AG done - AG submit = physical SDMA
                            # transfer time for chunk c
                            submit_t = ts[2 + 3 * c + 2]
                            done_t   = ts[20 + c]
                            phases[f"c{c}:AG-submit→AG-done (per-chunk)"] = (done_t - submit_t) * cy_to_ms
                    all_phase_deltas.append(phases)

                    # Compute-block-1 phase breakdown (slots 10..11+3*nc).
                    # Only present if kernel was built with compute-block
                    # instrumentation; slots 10+ will be 0 otherwise.
                    if ts[10] != 0 and ts[11 + 3 * last_nc] != 0:
                        cb_events = [(10, "entry")]
                        for c in range(last_nc):
                            cb_events.append((11 + 3 * c + 0, f"c{c}-loop"))
                            cb_events.append((11 + 3 * c + 1, f"c{c}-sct-poll-done"))
                            cb_events.append((11 + 3 * c + 2, f"c{c}-reduce-done"))
                        cb_events.append((11 + 3 * last_nc, "cb-exit"))
                        cb_phases = {}
                        for i in range(1, len(cb_events)):
                            label = f"{cb_events[i-1][1]}→{cb_events[i][1]}"
                            cb_phases[label] = (ts[cb_events[i][0]] - ts[cb_events[i-1][0]]) * cy_to_ms
                        all_cb_phase_deltas.append(cb_phases)

            if rank == 0 and len(all_phase_deltas) > 0:
                import statistics as _st
                phase_keys = list(all_phase_deltas[0].keys())
                med = {k: _st.median([d[k] for d in all_phase_deltas]) for k in phase_keys}
                ar0_med = _st.median(stage_med_ar0)
                print(f"\n  === AR[{phase_target_stage}] Phase Breakdown [{timeline_label}]  "
                      f"N={num_stages}  numChunks={last_nc}  "
                      f"(median of {n_samples} samples, ms) ===")
                print(f"  --- Block 0 (scatter/barrier/AG-submit orchestration) ---")
                print(f"  AR[{phase_target_stage}] total (event):                  {ar0_med:7.3f} ms")
                print(f"  sum of block-0 phase deltas:          "
                      f"{sum(med.values()):7.3f} ms")
                for k, v in med.items():
                    print(f"    {k:<35s} {v:7.3f} ms")

                if len(all_cb_phase_deltas) > 0:
                    cb_keys = list(all_cb_phase_deltas[0].keys())
                    cb_med = {k: _st.median([d[k] for d in all_cb_phase_deltas]) for k in cb_keys}
                    print(f"  --- Block 1 (first compute block: dispatch / scatter-poll / reduce) ---")
                    print(f"  sum of compute-block phase deltas:    "
                          f"{sum(cb_med.values()):7.3f} ms")
                    for k, v in cb_med.items():
                        print(f"    {k:<35s} {v:7.3f} ms")
                else:
                    print(f"  --- Block 1 (compute block) timestamps all zero: kernel was "
                          f"not built with compute-block instrumentation ---")

                if len(all_copy_timings) > 0:
                    host_us_vals = [v[0] for v in all_copy_timings]
                    gpu_ms_vals = [v[1] for v in all_copy_timings]
                    host_us_med = _st.median(host_us_vals)
                    gpu_ms_med = _st.median(gpu_ms_vals)
                    print(f"  --- Copy-path (baseline: single hipMemcpyAsync in copy_output_to_user) ---")
                    print(f"    host-side hipMemcpyAsync() call:      {host_us_med:7.2f} us")
                    print(f"    gpu-side copy kernel wall:            {gpu_ms_med:7.3f} ms")
                else:
                    print(f"  --- Copy-path timing unavailable "
                          f"(copy_output_to_user path not hit or api unavailable) ---")
                print()
        except Exception as e:
            if rank == 0:
                print(f"  [WARN] phase-timing failed: {e}")
            try:
                ar_obj.enable_phase_timing(False)
            except Exception:
                pass

    dist.barrier()

    def _reduce(vals):
        import statistics
        sorted_v = sorted(vals)
        med = statistics.median(sorted_v)
        t = torch.tensor(
            [min(vals), max(vals), sum(vals) / len(vals), med],
            dtype=torch.float64, device=device,
        )
        mn = t[0].clone(); mx = t[1].clone()
        avg = t[2].clone(); med_t = t[3].clone()
        dist.all_reduce(mn, op=dist.ReduceOp.MIN)
        dist.all_reduce(mx, op=dist.ReduceOp.MAX)
        dist.all_reduce(avg, op=dist.ReduceOp.SUM)
        dist.all_reduce(med_t, op=dist.ReduceOp.SUM)
        return mn.item(), mx.item(), avg.item() / npes, med_t.item() / npes

    g_ar = _reduce(seq_ar)
    g_gm = _reduce(seq_gemm)
    g_ov = _reduce(overlap_times)
    return {
        "seq_ar": g_ar[3] * 1000, "seq_gemm": g_gm[3] * 1000,
        "overlap": g_ov[3] * 1000, "overlap_min": g_ov[0] * 1000,
        "overlap_avg": g_ov[2] * 1000,
    }


def _test_multi_stage_overlap(
    rank, my_pe, npes, elems, data_bytes, output_buf_size, fill_value,
    dtype, dtype_name, device, iterations, warmup,
    gemm_m=_GEMM_M_DEFAULT, gemm_n=_GEMM_N_DEFAULT, gemm_k=_GEMM_K_DEFAULT,
    num_stages=2, sweep=False, timeline_dump=False, ar_phase_timing=False,
    ar_priority=0, gemm_priority=0,
):
    """Multi-stage pipelined GEMM+AllReduce overlap benchmark."""
    rccl_dtype = _RCCL_DTYPE_MAP.get(dtype, torch.float32)
    rccl_fill = (
        float(fill_value)
        if rccl_dtype in (torch.float16, torch.bfloat16)
        else fill_value
    )

    # Query device's supported priority range (HIP: [low, high], low = most-negative = highest priority).
    # torch.cuda.Stream(priority=...) takes negative values for higher priority.
    if rank == 0 and (ar_priority != 0 or gemm_priority != 0):
        try:
            low, high = torch.cuda.get_stream_priority_range()
            print(f"  [priority] stream priority range supported: [{low}(highest), {high}(lowest)]"
                  f"; using ar_priority={ar_priority} gemm_priority={gemm_priority}",
                  flush=True)
        except Exception as e:
            print(f"  [priority] get_stream_priority_range failed: {e}; "
                  f"using ar_priority={ar_priority} gemm_priority={gemm_priority}",
                  flush=True)

    stream_ar = torch.cuda.Stream(device=device, priority=ar_priority)
    stream_gemm = torch.cuda.Stream(device=device, priority=gemm_priority)
    total_iters = warmup + iterations
    elem_size = torch.tensor([], dtype=dtype).element_size()

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

    shared = dict(
        rank=rank, my_pe=my_pe, npes=npes, fill_value=fill_value,
        dtype=dtype, device=device, stream_ar=stream_ar,
        stream_gemm=stream_gemm, _run_gemm=_run_gemm,
        ev_ar_s=ev_ar_s, ev_ar_e=ev_ar_e, ev_g_s=ev_g_s,
        ev_g_e=ev_g_e, ov_s=ov_s, ov_e=ov_e,
        num_stages=num_stages, total_iters=total_iters, warmup=warmup,
        timeline_dump=timeline_dump,
        ar_phase_timing=ar_phase_timing,
    )

    single_mb = max(1, data_bytes // (1024 * 1024))
    sizes_mb = _SWEEP_SIZES_MB if sweep else [single_mb]

    if rank == 0:
        sz_str = ", ".join(f"{s}MB" for s in sizes_mb)
        print(
            f"\n>>> Test 6: Multi-stage pipelined overlap "
            f"({num_stages} stages, GEMM {gemm_m}x{gemm_k}x{gemm_n})"
        )
        print(f"    Sizes: {sz_str}")

    torch.cuda.synchronize()
    dist.barrier()

    results = {}  # results[(size_mb, mode)] = dict

    max_elems = max(sizes_mb) * 1024 * 1024 // elem_size
    max_bytes = max_elems * elem_size
    max_out_size = npes * (max_elems // npes + 64) * elem_size

    for mode in ["SDMA copy", "SDMA no-copy", "RCCL"]:
        ar_obj = None
        if mode in ("SDMA copy", "SDMA no-copy"):
            _copy_user = (mode == "SDMA copy")
            if rank == 0:
                print(f"  [{mode}] creating AR (max buf {max(sizes_mb)} MB) ...",
                      end="", flush=True)
            ar_obj = AllreduceSdma(my_pe, npes, input_buffer_size=max_bytes,
                                   output_buffer_size=max_out_size,
                                   copy_output_to_user=_copy_user,
                                   dtype=dtype)
            # Stage 1 E' prototype: compute blocks stay alive during AG wait.
            # Enable via env MORI_POST_AG_WAIT=1 (off by default).
            if os.environ.get("MORI_POST_AG_WAIT", "0") == "1":
                try:
                    ar_obj.enable_post_ag_wait(True)
                except Exception as e:
                    if rank == 0:
                        print(f"  [warn] enable_post_ag_wait failed: {e}",
                              flush=True)
            # Direction θ: multi-qId AG. Must be before first pipelined() call.
            if os.environ.get("MORI_AG_MULTI_Q", "0") == "1":
                try:
                    ar_obj.enable_ag_multi_q(True)
                except Exception as e:
                    if rank == 0:
                        print(f"  [warn] enable_ag_multi_q failed: {e}",
                              flush=True)
            torch.cuda.synchronize(); dist.barrier()
            if rank == 0:
                print(" ok", flush=True)

        for size_mb in sizes_mb:
            cur_elems = size_mb * 1024 * 1024 // elem_size
            cur_bytes = cur_elems * elem_size

            ov_tag = "overlap" if size_mb >= 128 else "serial"
            if rank == 0:
                print(f"  {mode:14s} | {size_mb:6d} MB [{ov_tag}] ...",
                      end="", flush=True)

            if mode in ("SDMA copy", "SDMA no-copy"):
                def make_setup(e=cur_elems, _ar=ar_obj):
                    def setup():
                        inp = torch.full((e,), fill_value, dtype=dtype, device=device)
                        out = torch.zeros(e, dtype=dtype, device=device)
                        torch.cuda.synchronize(); dist.barrier()
                        stream_ar.synchronize()
                        ok = _ar(inp, out, e, stream_ar); stream_ar.synchronize()
                        def prep(): inp.fill_(fill_value)
                        def launch(): return _ar(inp, out, e, stream_ar)
                        # Return tuple slot 5 (copy_stream) = None,
                        # slot 6 = ar_obj so _bench can call enable_phase_timing
                        return ok, launch, prep, False, None, _ar
                    return setup
                setup_fn = make_setup()
            else:
                def make_setup(e=cur_elems):
                    def setup():
                        buf = torch.full((e,), rccl_fill, dtype=rccl_dtype, device=device)
                        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                        torch.cuda.synchronize()
                        def prep(): buf.fill_(rccl_fill)
                        def launch():
                            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                            return True
                        return True, launch, prep, True
                    return setup
                setup_fn = make_setup()

            r = _bench_overlap_one_size(
                elems=cur_elems, setup_fn=setup_fn,
                timeline_label=f"{mode} {size_mb}MB",
                **shared,
            )
            results[(size_mb, mode)] = r
            if rank == 0:
                if r:
                    print(f" overlap {r['overlap']:.3f} ms  "
                          f"(ar {r['seq_ar']:.3f}, gemm {r['seq_gemm']:.3f})",
                          flush=True)
                else:
                    print(" FAILED", flush=True)

            torch.cuda.synchronize()
            dist.barrier()

        if ar_obj is not None:
            del ar_obj
        import gc; gc.collect()
        torch.cuda.synchronize()
        dist.barrier()

    # ---- Summary tables (rank 0 only) ----
    if rank == 0:
        modes = ["SDMA copy", "SDMA no-copy", "RCCL"]
        w = 120
        print(f"\n{'=' * w}")
        print(f"  Multi-stage Overlap Summary — {num_stages} stages, "
              f"GEMM {gemm_m}x{gemm_k}x{gemm_n}")
        print(f"  (sizes >= 128 MB use true GEMM+AR overlap; "
              f"smaller sizes run GEMM then AR serially)")
        print(f"{'=' * w}")

        def _val(sm, key):
            r = results.get((sm, key))
            return r if r else None

        def _fmt(v, key="overlap"):
            if v is None: return "     N/A"
            return f"{v[key]:8.3f}"

        def _pct(base_v, ref_v, key="overlap"):
            if base_v is None or ref_v is None: return "       N/A"
            b = base_v[key]; r = ref_v[key]
            if r == 0: return "       N/A"
            pct = (r - b) / r * 100
            return f"    {pct:+.1f}%"

        # Table 1: Overlap Wall Time
        print(f"\n  Table 1: Overlap Wall Time (ms, median)")
        hdr = (f"  {'Size':>8s} | {'SDMA copy':>10s} | {'SDMA no-copy':>12s} | "
               f"{'RCCL':>10s} | {'copy vs RCCL':>12s} | {'no-copy vs RCCL':>15s}")
        print(f"  {'-' * (len(hdr) - 2)}")
        print(hdr)
        print(f"  {'-' * (len(hdr) - 2)}")
        for sm in sizes_mb:
            rc = _val(sm, "SDMA copy")
            rn = _val(sm, "SDMA no-copy")
            rr = _val(sm, "RCCL")
            print(f"  {sm:>6d} MB | {_fmt(rc):>10s} | {_fmt(rn):>12s} | "
                  f"{_fmt(rr):>10s} | {_pct(rc, rr):>12s} | {_pct(rn, rr):>15s}")
        print(f"  {'-' * (len(hdr) - 2)}")

        # Table 2: GEMM Slowdown
        print(f"\n  Table 2: GEMM Slowdown (overlap / seq_gemm, 1.0 = perfect hiding, median)")
        hdr2 = f"  {'Size':>8s} | {'SDMA copy':>10s} | {'SDMA no-copy':>12s} | {'RCCL':>10s}"
        print(f"  {'-' * (len(hdr2) - 2)}")
        print(hdr2)
        print(f"  {'-' * (len(hdr2) - 2)}")
        for sm in sizes_mb:
            vals = []
            for m in modes:
                r = _val(sm, m)
                if r and r["seq_gemm"] > 0:
                    vals.append(f"{r['overlap'] / r['seq_gemm']:8.3f}")
                else:
                    vals.append("     N/A")
            print(f"  {sm:>6d} MB | {vals[0]:>10s} | {vals[1]:>12s} | {vals[2]:>10s}")
        print(f"  {'-' * (len(hdr2) - 2)}")

        # Table 3: Sequential AllReduce Time
        print(f"\n  Table 3: Sequential AllReduce Time (ms, median, {num_stages} stages)")
        print(f"  {'-' * (len(hdr2) - 2)}")
        print(hdr2)
        print(f"  {'-' * (len(hdr2) - 2)}")
        for sm in sizes_mb:
            vals = []
            for m in modes:
                r = _val(sm, m)
                vals.append(f"{r['seq_ar']:8.3f}" if r else "     N/A")
            print(f"  {sm:>6d} MB | {vals[0]:>10s} | {vals[1]:>12s} | {vals[2]:>10s}")
        print(f"  {'-' * (len(hdr2) - 2)}")

        # Table 4: Sequential GEMM Time
        print(f"\n  Table 4: Sequential GEMM Time (ms, median, {num_stages} stages)")
        print(f"  {'-' * (len(hdr2) - 2)}")
        print(hdr2)
        print(f"  {'-' * (len(hdr2) - 2)}")
        for sm in sizes_mb:
            vals = []
            for m in modes:
                r = _val(sm, m)
                vals.append(f"{r['seq_gemm']:8.3f}" if r else "     N/A")
            print(f"  {sm:>6d} MB | {vals[0]:>10s} | {vals[1]:>12s} | {vals[2]:>10s}")
        print(f"  {'-' * (len(hdr2) - 2)}")

    torch.cuda.synchronize()
    dist.barrier()
    return True


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
    num_stages=0,
    sweep=False,
    timeline_dump=False,
    ar_phase_timing=False,
    ar_priority=0,
    gemm_priority=0,
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
        ok1b = _test_copy_to_user_verify(
            rank, my_pe, npes, elems, data_bytes, output_buf_size, fill_value,
            dtype, dtype_name, device, stream, iterations, warmup,
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

        all_ok = ok1 and ok1b and ok2 and ok3 and ok4

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
            )
            all_ok = all_ok and ok5

        if num_stages > 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            dist.barrier()
            if rank == 0:
                print(f"\n  [debug] MORI_PIPELINE_CU={os.environ.get('MORI_PIPELINE_CU', 'unset')}"
                      f"  MORI_PIPELINE_FUSED={os.environ.get('MORI_PIPELINE_FUSED', 'unset')}",
                      flush=True)
            ok6 = _test_multi_stage_overlap(
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
                sweep=sweep,
                timeline_dump=timeline_dump,
                ar_phase_timing=ar_phase_timing,
                ar_priority=ar_priority,
                gemm_priority=gemm_priority,
            )
            all_ok = all_ok and ok6

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
    num_stages=0,
    sweep=False,
    timeline_dump=False,
    ar_phase_timing=False,
    ar_priority=0,
    gemm_priority=0,
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
            sweep,
            timeline_dump,
            ar_phase_timing,
            ar_priority,
            gemm_priority,
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
        default=0,
        help="Number of pipelined GEMM+AR stages (0=skip multi-stage test)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep multiple data sizes (1MB..1GB) in multi-stage test and print summary table",
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Dump per-stage GEMM/AR timeline (median of measure iters) for each "
             "(mode, size) combo in multi-stage test. Useful for overlap analysis.",
    )
    parser.add_argument(
        "--ar-phase-timing",
        action="store_true",
        help="Enable kernel-internal phase timestamps for SDMA AR[0] (requires "
             "rebuilt mori with instrumented kernel). Prints per-phase ms "
             "breakdown: entry→scatter_done, compute-wait, cross-PE-barrier, "
             "AG-submit, AG-wait. Use together with --timeline and a single size "
             "(e.g. --elems 67108864) for clear output.",
    )
    parser.add_argument(
        "--ar-priority",
        type=int,
        default=0,
        help="Set stream_ar priority (HIP supports -1=high, 0=normal, range "
             "queryable via hipDeviceGetStreamPriorityRange). Set -1 to test "
             "whether giving AR stream higher priority reduces GEMM-AR CU "
             "contention (exploratory).",
    )
    parser.add_argument(
        "--gemm-priority",
        type=int,
        default=0,
        help="Set stream_gemm priority. Set 0 (normal) while --ar-priority=-1 "
             "to let AR preempt GEMM for CU resources.",
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
    if args.num_stages > 0:
        print(f"  Multi-stage     : {args.num_stages} stages (pipelined GEMM+AR)")
        if args.sweep:
            print(f"  Size sweep      : {', '.join(f'{s}MB' for s in _SWEEP_SIZES_MB)}")
    if args.num_stages > 0 or args.test_gemm_overlap:
        print(f"  PIPELINE_CU     : {os.environ.get('MORI_PIPELINE_CU', 'not set (default=all)')}")
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
        sweep=args.sweep,
        timeline_dump=args.timeline,
        ar_phase_timing=args.ar_phase_timing,
        ar_priority=args.ar_priority,
        gemm_priority=args.gemm_priority,
    )
