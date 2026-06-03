#!/usr/bin/env python3
"""
ReduceScatter SDMA Test using MORI ReduceScatterSdma and multiprocessing
"""

import os
import numpy as np
import torch
import torch.distributed as dist
import mori.shmem as shmem
from mori.ccl import ReduceScatterSdma
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
        shmem.shmem_torch_process_group_init("default")

        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()

        assert my_pe == rank
        assert npes == world_size

        # ReduceScatter: input = elems_per_pe * npes, output = elems_per_pe
        elems_per_pe = elems
        total_elems = elems_per_pe * npes
        input_bytes = total_elems * 4
        output_bytes = elems_per_pe * 4
        # Transit buffer needs to hold the gather buffer = total_elems * sizeof(T)
        transit_buffer_bytes = total_elems * 4

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"ReduceScatter SDMA Test")
            print(f"World size: {world_size}")
            print(f"Elements per PE (output): {elems_per_pe:,}")
            print(f"Total elements per PE (input): {total_elems:,}")
            print(f"Data size: {input_bytes / (1024**2):.2f} MB input, {output_bytes / (1024**2):.2f} MB output per PE")
            print(f"Transit buffer: {transit_buffer_bytes / (1024**2):.2f} MB")
            print(f"Iterations: {iterations}" + (f" (warmup: {warmup})" if warmup > 0 else ""))
            print(f"Custom Stream: {'Yes' if use_custom_stream else 'No (default stream)'}")
            print(f"{'='*60}\n")

        print(f"PE {rank}/{world_size}: SHMEM initialized, myPe={my_pe}, npes={npes}")

        # Create ReduceScatter object
        rs = ReduceScatterSdma(my_pe, npes,
                               input_buffer_size=input_bytes,
                               output_buffer_size=transit_buffer_bytes)
        print(f"PE {rank}: Created ReduceScatterSdma object")

        device = torch.device(f"cuda:{rank}")

        # Input: total_elems elements. Chunk[i] = (rank+1)*1000 + i
        input_tensor = torch.zeros(total_elems, dtype=torch.uint32, device=device)
        for i in range(npes):
            start = i * elems_per_pe
            end = (i + 1) * elems_per_pe
            val = (my_pe + 1) * 1000 + i
            input_data = np.full(elems_per_pe, val, dtype=np.uint32)
            input_tensor[start:end] = torch.from_numpy(input_data).to(device)

        # Output: elems_per_pe elements
        output_tensor = torch.zeros(elems_per_pe, dtype=torch.uint32, device=device)

        if rank == 0:
            print(f"\n=== Data Pattern ===")
            print(f"Each PE contributes {npes} chunks of {elems_per_pe:,} elements")
            print(f"PE r, chunk i has value: (r+1)*1000 + i")
            print(f"\nAfter ReduceScatter, PE r gets reduced chunk r:")
            for r in range(npes):
                expected = sum((pe + 1) * 1000 + r for pe in range(npes))
                print(f"  PE {r} output = {expected}")
            print()

        # GEMM setup
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

        # Step 1: Sequential baseline
        if use_custom_stream and test_gemm_overlap and HAS_AITER and stream_gemm is not None:
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Step 1: Sequential Baseline Tests")
                print(f"{'='*60}")
                print(f"\nTesting ReduceScatter sequentially (baseline)...")

            for iter_idx in range(total_iters):
                torch.cuda.synchronize()
                if use_custom_stream:
                    rs_start.record(stream)
                    with torch.cuda.stream(stream):
                        success = rs(input_tensor, output_tensor, total_elems)
                    rs_end.record(stream)
                    stream.synchronize()
                else:
                    rs_start.record()
                    success = rs(input_tensor, output_tensor, total_elems)
                    rs_end.record()
                    torch.cuda.synchronize()
                rs_time = rs_start.elapsed_time(rs_end) / 1000.0
                if iter_idx >= warmup:
                    sequential_rs_times.append(rs_time)
                elif rank == 0:
                    print(f"  Warmup {iter_idx+1}/{warmup}: {rs_time:.6f}s")

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
                    print(f"  Warmup {iter_idx+1}/{warmup}: {gemm_time:.6f}s")

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
            op_success = True
            if use_custom_stream and test_gemm_overlap and HAS_AITER and stream_gemm is not None:
                torch.cuda.synchronize()
                overlap_start.record()
                rs_start.record(stream)
                gemm_start.record(stream_gemm)

                with torch.cuda.stream(stream):
                    op_success = rs(input_tensor, output_tensor, total_elems)
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
                        op_success = rs(input_tensor, output_tensor, total_elems)
                    rs_end.record(stream)
                    stream.synchronize()
                else:
                    rs_start.record()
                    op_success = rs(input_tensor, output_tensor, total_elems)
                    rs_end.record()
                    torch.cuda.synchronize()
                rs_time = rs_start.elapsed_time(rs_end) / 1000.0
                if iter_idx >= warmup:
                    exec_times.append(rs_time)
                elif rank == 0:
                    print(f"Warmup {iter_idx+1}/{warmup}: {rs_time:.6f}s")

            if not op_success:
                print(f"PE {rank}: ReduceScatter failed at iteration {iter_idx}")
                break

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
            bw = input_bytes / avg_time / (1024**3) if avg_time > 0 else 0
            print(f"\n{'='*60}")
            print(f"Performance Statistics")
            print(f"{'='*60}")
            print(f"ReduceScatter Times:")
            print(f"  Min: {min_time:.6f}s, Avg: {avg_time:.6f}s, Max: {max_time:.6f}s")
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
            print(f"{'='*60}")

        # Verify
        output_cpu = output_tensor.cpu().numpy()
        expected = sum((pe + 1) * 1000 + rank for pe in range(npes))
        # For uint32, cast expected
        expected_val = np.uint32(expected)
        success = np.all(output_cpu == expected_val)
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

        # Cleanup
        torch.cuda.synchronize()
        dist.barrier()
        del rs
        dist.barrier()
        shmem.shmem_finalize()

        if not success:
            raise AssertionError(f"PE {rank}: ReduceScatter verification failed")


def test_reducescatter(elems=33554432, world_size=8, iterations=10, warmup=1,
                       use_custom_stream=False, test_gemm_overlap=False,
                       gemm_m=4096, gemm_n=4096, gemm_k=4096):
    os.environ.setdefault('MORI_ENABLE_SDMA', '1')
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
        description="Test ReduceScatter SDMA (MORI)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--elems", type=int, default=33554432, help="Output elements per PE")
    parser.add_argument("--world-size", type=int, default=8, help="Number of processes")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--enable-sdma", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use-custom-stream", action="store_true")
    parser.add_argument("--test-gemm-overlap", action="store_true")
    parser.add_argument("--gemm-m", type=int, default=4096, help="GEMM M dimension")
    parser.add_argument("--gemm-n", type=int, default=4096, help="GEMM N dimension")
    parser.add_argument("--gemm-k", type=int, default=4096, help="GEMM K dimension")
    args = parser.parse_args()
    os.environ['MORI_ENABLE_SDMA'] = str(args.enable_sdma)

    print(f"ReduceScatter SDMA Test")
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
