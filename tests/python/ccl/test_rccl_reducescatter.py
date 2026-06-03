#!/usr/bin/env python3
"""
ReduceScatter RCCL Test using torch.distributed and multiprocessing
"""

import os
import numpy as np
import torch
import torch.distributed as dist
from tests.python.utils import TorchDistContext, get_free_port

try:
    import aiter
    HAS_AITER = True
except ImportError:
    HAS_AITER = False
    print("Warning: aiter not available, gemm timing will be disabled")


def _test_reducescatter(rank, world_size, port, elems, iterations, warmup,
                        use_custom_stream, test_gemm_overlap,
                        gemm_m=4096, gemm_n=4096, gemm_k=4096):
    """Worker function for each process"""

    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        npes = world_size

        elems_per_pe = elems
        total_elems = elems_per_pe * npes
        input_bytes = total_elems * 4
        output_bytes = elems_per_pe * 4

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"ReduceScatter RCCL Test")
            print(f"World size: {world_size}")
            print(f"Elements per PE (output): {elems_per_pe:,}")
            print(f"Total elements per PE (input): {total_elems:,}")
            print(f"Data size: {input_bytes / (1024**2):.2f} MB input, {output_bytes / (1024**2):.2f} MB output per PE")
            print(f"Iterations: {iterations}" + (f" (warmup: {warmup})" if warmup > 0 else ""))
            print(f"Custom Stream: {'Yes' if use_custom_stream else 'No (default stream)'}")
            print(f"{'='*60}\n")

        print(f"PE {rank}/{world_size}: Initialized")

        device = torch.device(f"cuda:{rank}")

        # Each PE has total_elems input elements, divided into npes chunks.
        # Chunk[i] = (rank + 1) * 1000 + i for verification.
        input_tensor = torch.zeros(total_elems, dtype=torch.int32, device=device)
        for i in range(npes):
            start = i * elems_per_pe
            end = (i + 1) * elems_per_pe
            input_tensor[start:end] = (rank + 1) * 1000 + i

        # Output: elems_per_pe elements (the reduced chunk for this rank)
        output_tensor = torch.zeros(elems_per_pe, dtype=torch.int32, device=device)

        # For dist.reduce_scatter, input is a list of tensors (one per PE's contribution)
        input_list = list(input_tensor.chunk(npes))

        if rank == 0:
            print(f"\n=== Data Pattern ===")
            print(f"Each PE contributes {npes} chunks of {elems_per_pe:,} elements")
            print(f"PE r, chunk i has value: (r+1)*1000 + i")
            print(f"\nAfter ReduceScatter, PE r gets reduced chunk r:")
            for r in range(npes):
                expected = sum((pe + 1) * 1000 + r for pe in range(npes))
                print(f"  PE {r} output = sum over all PEs of chunk[{r}] = {expected}")
            print()

        # Prepare GEMM test data if testing overlap
        A_q = B_q = A_scale = B_scale = bias = None
        if test_gemm_overlap and HAS_AITER:
            M, N, K = gemm_m, gemm_n, gemm_k
            A_q = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=device)
            B_q = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=device)
            A_scale = torch.randn(M, dtype=torch.float32, device=device)
            B_scale = torch.randn(N, dtype=torch.float32, device=device)
            bias = torch.randn(N, dtype=torch.bfloat16, device=device)
            if rank == 0:
                print(f"PE {rank}: Prepared GEMM test data (M={M}, N={N}, K={K})")

        stream_gemm = None
        if use_custom_stream:
            stream = torch.cuda.Stream(device=device)
            if test_gemm_overlap and HAS_AITER:
                stream_gemm = torch.cuda.Stream(device=device)
                if rank == 0:
                    print(f"PE {rank}: Created separate CUDA streams for RS and GEMM")
            else:
                if rank == 0:
                    print(f"PE {rank}: Created custom CUDA stream")
        else:
            stream = None
            if rank == 0:
                print(f"PE {rank}: Using default CUDA stream")

        torch.cuda.synchronize()
        dist.barrier()

        exec_times = []
        gemm_times = []
        overlap_times = []
        sequential_rs_times = []
        sequential_gemm_times = []
        total_iters = warmup + iterations

        rs_start = torch.cuda.Event(enable_timing=True)
        rs_end = torch.cuda.Event(enable_timing=True)

        if test_gemm_overlap and HAS_AITER and stream_gemm is not None:
            gemm_start = torch.cuda.Event(enable_timing=True)
            gemm_end = torch.cuda.Event(enable_timing=True)
            overlap_start = torch.cuda.Event(enable_timing=True)
            overlap_end = torch.cuda.Event(enable_timing=True)

        # Step 1: Sequential baseline (if overlap test)
        if use_custom_stream and test_gemm_overlap and HAS_AITER and stream_gemm is not None:
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Step 1: Sequential Baseline Tests")
                print(f"{'='*60}")

            if rank == 0:
                print(f"\nTesting ReduceScatter sequentially (baseline)...")
            for iter_idx in range(total_iters):
                torch.cuda.synchronize()
                if use_custom_stream:
                    rs_start.record(stream)
                    with torch.cuda.stream(stream):
                        dist.reduce_scatter(output_tensor, input_list)
                    rs_end.record(stream)
                    stream.synchronize()
                else:
                    rs_start.record()
                    dist.reduce_scatter(output_tensor, input_list)
                    rs_end.record()
                    torch.cuda.synchronize()
                rs_time = rs_start.elapsed_time(rs_end) / 1000.0
                if iter_idx >= warmup:
                    sequential_rs_times.append(rs_time)
                elif rank == 0:
                    print(f"  Warmup {iter_idx + 1}/{warmup}: {rs_time:.6f}s")

            if rank == 0:
                print(f"\nTesting GEMM sequentially (baseline)...")
            for iter_idx in range(total_iters):
                torch.cuda.synchronize()
                gemm_start.record(stream_gemm)
                with torch.cuda.stream(stream_gemm):
                    _ = aiter.gemm_a8w8_CK(A_q, B_q, A_scale, B_scale, bias, torch.bfloat16)
                gemm_end.record(stream_gemm)
                stream_gemm.synchronize()
                gemm_time = gemm_start.elapsed_time(gemm_end) / 1000.0
                if iter_idx >= warmup:
                    sequential_gemm_times.append(gemm_time)
                elif rank == 0:
                    print(f"  Warmup {iter_idx + 1}/{warmup}: {gemm_time:.6f}s")

            if rank == 0:
                seq_rs_avg = np.mean(sequential_rs_times)
                seq_gemm_avg = np.mean(sequential_gemm_times)
                print(f"\nSequential Baseline Results:")
                print(f"  ReduceScatter: Min={np.min(sequential_rs_times):.6f}s, Avg={seq_rs_avg:.6f}s, Max={np.max(sequential_rs_times):.6f}s")
                print(f"  GEMM: Min={np.min(sequential_gemm_times):.6f}s, Avg={seq_gemm_avg:.6f}s, Max={np.max(sequential_gemm_times):.6f}s")
                print(f"  Total sequential: {seq_rs_avg + seq_gemm_avg:.6f}s")
                print(f"\n{'='*60}")
                print(f"Step 2: Concurrent Overlap Tests")
                print(f"{'='*60}\n")

        # Step 2: Main test loop
        for iter_idx in range(total_iters):
            if use_custom_stream and test_gemm_overlap and HAS_AITER and stream_gemm is not None:
                torch.cuda.synchronize()
                overlap_start.record()
                rs_start.record(stream)
                gemm_start.record(stream_gemm)

                with torch.cuda.stream(stream):
                    dist.reduce_scatter(output_tensor, input_list)
                with torch.cuda.stream(stream_gemm):
                    _ = aiter.gemm_a8w8_CK(A_q, B_q, A_scale, B_scale, bias, torch.bfloat16)

                rs_end.record(stream)
                gemm_end.record(stream_gemm)
                stream.synchronize()
                stream_gemm.synchronize()
                overlap_end.record()
                torch.cuda.synchronize()

                rs_time = rs_start.elapsed_time(rs_end) / 1000.0
                gemm_time = gemm_start.elapsed_time(gemm_end) / 1000.0
                overlap_time = overlap_start.elapsed_time(overlap_end) / 1000.0

                if iter_idx >= warmup:
                    exec_times.append(rs_time)
                    gemm_times.append(gemm_time)
                    overlap_times.append(overlap_time)
                elif rank == 0:
                    print(f"Warmup {iter_idx+1}/{warmup}: RS={rs_time:.6f}s, GEMM={gemm_time:.6f}s, Overlap={overlap_time:.6f}s")
            else:
                if use_custom_stream:
                    rs_start.record(stream)
                    with torch.cuda.stream(stream):
                        dist.reduce_scatter(output_tensor, input_list)
                    rs_end.record(stream)
                    stream.synchronize()
                else:
                    rs_start.record()
                    dist.reduce_scatter(output_tensor, input_list)
                    rs_end.record()
                    torch.cuda.synchronize()
                rs_time = rs_start.elapsed_time(rs_end) / 1000.0
                if iter_idx >= warmup:
                    exec_times.append(rs_time)
                elif rank == 0:
                    print(f"Warmup {iter_idx+1}/{warmup}: {rs_time:.6f}s")

        if use_custom_stream and stream:
            stream.synchronize()
        torch.cuda.synchronize()

        # Stats
        avg_time = np.mean(exec_times) if exec_times else 0.0
        min_time = np.min(exec_times) if exec_times else 0.0
        max_time = np.max(exec_times) if exec_times else 0.0
        gemm_avg_time = np.mean(gemm_times) if gemm_times else 0.0
        overlap_avg_time = np.mean(overlap_times) if overlap_times else 0.0
        seq_rs_avg = np.mean(sequential_rs_times) if sequential_rs_times else 0.0
        seq_gemm_avg = np.mean(sequential_gemm_times) if sequential_gemm_times else 0.0

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Performance Statistics")
            print(f"{'='*60}")
            print(f"ReduceScatter Times:")
            print(f"  Min time: {min_time:.6f}s")
            print(f"  Max time: {max_time:.6f}s")
            print(f"  Avg time: {avg_time:.6f}s")
            bw = input_bytes / avg_time / (1024**3) if avg_time > 0 else 0
            print(f"  Bandwidth: {bw:.2f} GB/s")

            if gemm_times:
                print(f"\nSequential Baseline:")
                print(f"  ReduceScatter avg: {seq_rs_avg:.6f}s")
                print(f"  GEMM avg: {seq_gemm_avg:.6f}s")
                print(f"  Sequential total: {seq_rs_avg + seq_gemm_avg:.6f}s")
                print(f"\nConcurrent:")
                print(f"  ReduceScatter avg: {avg_time:.6f}s")
                print(f"  GEMM avg: {gemm_avg_time:.6f}s")

            if overlap_times:
                ideal = max(avg_time, gemm_avg_time)
                seq_total = seq_rs_avg + seq_gemm_avg
                speedup = seq_total / overlap_avg_time if overlap_avg_time > 0 else 0
                efficiency = (ideal / overlap_avg_time * 100) if overlap_avg_time > 0 else 0
                print(f"\nOverlap Analysis:")
                print(f"  Overlap time (measured): {overlap_avg_time:.6f}s")
                print(f"  Theoretical best: {ideal:.6f}s")
                print(f"  Sequential baseline: {seq_total:.6f}s")
                print(f"  Time saved: {seq_total - overlap_avg_time:.6f}s")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Concurrency efficiency: {efficiency:.2f}%")

        # Verify
        output_cpu = output_tensor.cpu().numpy()
        expected = sum((pe + 1) * 1000 + rank for pe in range(npes))
        success = np.all(output_cpu == expected)
        if success:
            print(f"PE {rank}: Verification PASSED (all values = {expected})")
        else:
            print(f"PE {rank}: Verification FAILED! Expected {expected}, got unique: {np.unique(output_cpu)}")

        torch.cuda.synchronize()
        dist.barrier()

        # Global stats
        min_t = torch.tensor([min_time], dtype=torch.float64)
        max_t = torch.tensor([max_time], dtype=torch.float64)
        avg_t = torch.tensor([avg_time], dtype=torch.float64)
        success_t = torch.tensor([1 if success else 0], dtype=torch.int32)

        dist.all_reduce(min_t, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_t, op=dist.ReduceOp.MAX)
        dist.all_reduce(avg_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(success_t, op=dist.ReduceOp.SUM)

        if rank == 0:
            g_avg = avg_t.item() / npes
            g_bw = input_bytes / g_avg / (1024**3) if g_avg > 0 else 0
            print(f"\n{'='*60}")
            print(f"Global Results")
            print(f"{'='*60}")
            print(f"  Min: {min_t.item():.6f}s, Avg: {g_avg:.6f}s, Max: {max_t.item():.6f}s")
            print(f"  Bandwidth: {g_bw:.2f} GB/s")
            print(f"  PEs passed: {success_t.item()}/{npes}")
            print(f"\n=== Test {'PASSED' if success_t.item() == npes else 'FAILED'} ===")
            print(f"{'='*60}\n")

        torch.cuda.synchronize()
        dist.barrier()

        if not success:
            raise AssertionError(f"PE {rank}: ReduceScatter verification failed")


def test_reducescatter(elems=67108864, world_size=8, iterations=10, warmup=1,
                       use_custom_stream=False, test_gemm_overlap=False,
                       gemm_m=4096, gemm_n=4096, gemm_k=4096):
    port = get_free_port()
    torch.multiprocessing.spawn(
        _test_reducescatter,
        args=(world_size, port, elems, iterations, warmup, use_custom_stream,
              test_gemm_overlap, gemm_m, gemm_n, gemm_k),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test ReduceScatter RCCL (torch.distributed)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--elems", type=int, default=33554432, help="Output elements per PE")
    parser.add_argument("--world-size", type=int, default=8, help="Number of processes")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--use-custom-stream", action="store_true")
    parser.add_argument("--test-gemm-overlap", action="store_true")
    parser.add_argument("--gemm-m", type=int, default=4096, help="GEMM M dimension")
    parser.add_argument("--gemm-n", type=int, default=4096, help="GEMM N dimension")
    parser.add_argument("--gemm-k", type=int, default=4096, help="GEMM K dimension")
    args = parser.parse_args()

    print(f"ReduceScatter RCCL Test")
    print(f"  Output elements per PE: {args.elems:,}")
    print(f"  World size: {args.world_size}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Warmup: {args.warmup}")
    if args.test_gemm_overlap:
        print(f"  GEMM Dimensions: M={args.gemm_m}, N={args.gemm_n}, K={args.gemm_k}")
        if not HAS_AITER:
            print(f"  WARNING: aiter not available")
    print("-" * 60)

    test_reducescatter(args.elems, args.world_size, args.iterations, args.warmup,
                       args.use_custom_stream, args.test_gemm_overlap,
                       args.gemm_m, args.gemm_n, args.gemm_k)
