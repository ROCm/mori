#!/usr/bin/env python3
"""
Allgather SDMA Test using torch.distributed and multiprocessing
"""

import os
import numpy as np
import torch
import torch.distributed as dist
import mori.shmem as shmem
from mori.ccl import AllgatherSdma
from tests.python.utils import TorchDistContext, get_free_port

try:
    import aiter
    from aiter.ops.dtypes import dtypes
    HAS_AITER = True
except ImportError:
    HAS_AITER = False
    print("Warning: aiter not available, gemm timing will be disabled")


def _test_allgather(rank, world_size, port, elems, iterations, warmup, use_custom_stream, test_gemm_overlap):
    """Worker function for each process"""
    
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        
        assert my_pe == rank, f"PE mismatch: {my_pe} != {rank}"
        assert npes == world_size, f"npes mismatch: {npes} != {world_size}"
        
        # Match C++ naming and logic
        elems_per_pe = elems  # Elements each PE contributes (like C++ elemsPerPe)
        bytes_per_pe = elems_per_pe * 4  # Bytes per PE contribution (like C++ bytesPerPe)
        total_bytes = bytes_per_pe * npes  # Total bytes after gathering from all PEs (like C++ totalBytes)
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Allgather SDMA Test")
            print(f"World size: {world_size}")
            print(f"Elements per PE: {elems_per_pe:,}")
            print(f"Data size: {bytes_per_pe / (1024**2):.2f} MB per PE (input), {total_bytes / (1024**2):.2f} MB total (output)")
            print(f"Iterations: {iterations}" + (f" (warmup: {warmup})" if warmup > 0 else ""))
            print(f"Custom Stream: {'Yes' if use_custom_stream else 'No (default stream)'}")
            print(f"{'='*60}\n")
        
        print(f"PE {rank}/{world_size}: SHMEM initialized, myPe={my_pe}, npes={npes}")

        # Create Allgather object with sufficient buffer size
        allgather = AllgatherSdma(my_pe, npes, 
                                  input_buffer_size=bytes_per_pe,
                                  output_buffer_size=total_bytes)
        print(f"PE {rank}: Created AllgatherSdma object")

        # Allocate GPU memory
        # Note: Using torch.uint32 to match C++ AllgatherSdma<uint32_t>
        device = torch.device(f"cuda:{rank}")
        input_tensor = torch.zeros(elems_per_pe, dtype=torch.uint32, device=device)
        output_tensor = torch.zeros(elems_per_pe * npes, dtype=torch.uint32, device=device)

        matrix_size=8192
        A = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.bfloat16)
        B = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.bfloat16)

        # 量化输入数据
        A_q, A_scale = aiter.pertoken_quant(A, quant_dtype=dtypes.fp8)
        B_q, B_scale = aiter.pertoken_quant(B, quant_dtype=dtypes.fp8)
        bias = None
        
        # Prepare data: Each PE has unique value = (myPe + 1) * 1000
        value = (my_pe + 1) * 1000
        input_data_cpu = np.full(elems_per_pe, value, dtype=np.uint32)
        
        # Copy to GPU
        input_tensor.copy_(torch.from_numpy(input_data_cpu))

        if rank == 0:
            print(f"\n=== Data Pattern ===")
            print(f"Each PE contributes unique data:")
            for pe in range(npes):
                pe_value = (pe + 1) * 1000
                print(f"  PE {pe} contributes: {pe_value}")
            print(f"\nAfter Allgather, all PEs should have:")
            for pe in range(npes):
                pe_value = (pe + 1) * 1000
                print(f"  Chunk {pe} (from PE {pe}): {pe_value}")
            print()

        print(f"PE {rank}: Prepared input data with value: {value}")

        # Prepare GEMM test data if testing overlap
        A_q = B_q = A_scale = B_scale = bias = None
        if test_gemm_overlap and HAS_AITER:
            # Create sample GEMM matrices for testing overlap
            M, N, K = 4096, 4096, 4096
            A_q = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=device)
            B_q = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=device)
            A_scale = torch.randn(M, dtype=torch.float32, device=device)
            B_scale = torch.randn(N, dtype=torch.float32, device=device)
            bias = torch.randn(N, dtype=torch.bfloat16, device=device)
            if rank == 0:
                print(f"PE {rank}: Prepared GEMM test data (M={M}, N={N}, K={K})")

        # Create CUDA streams for allgather and gemm operations (if requested)
        stream_gemm = None
        if use_custom_stream:
            stream = torch.cuda.Stream(device=device)
            if test_gemm_overlap and HAS_AITER:
                stream_gemm = torch.cuda.Stream(device=device)
                if rank == 0:
                    print(f"PE {rank}: Created separate CUDA streams for allgather and gemm")
            else:
                if rank == 0:
                    print(f"PE {rank}: Created custom CUDA stream for allgather operations")
        else:
            stream = None  # Use default stream
            if rank == 0:
                print(f"PE {rank}: Using default CUDA stream (None)")

        torch.cuda.synchronize()
        dist.barrier()

        # Execute Allgather multiple times
        exec_times = []
        gemm_times = []
        overlap_times = []  # Total time for concurrent execution
        total_iters = warmup + iterations
        use_async = False  # Use async mode to match C++ test
        
        if not use_async:
            # Synchronous mode (single SDMA queue)
            # Create CUDA events for timing allgather
            allgather_start = torch.cuda.Event(enable_timing=True)
            allgather_end = torch.cuda.Event(enable_timing=True)
            
            # Create CUDA events for timing gemm (if testing overlap)
            if test_gemm_overlap and HAS_AITER and stream_gemm is not None:
                gemm_start = torch.cuda.Event(enable_timing=True)
                gemm_end = torch.cuda.Event(enable_timing=True)
                # Create events for measuring total overlap time (from start to both complete)
                overlap_start = torch.cuda.Event(enable_timing=True)
                overlap_end = torch.cuda.Event(enable_timing=True)
            
            for iter_idx in range(total_iters):
                success = True  # Initialize success flag
                
                # Execute allgather and gemm concurrently on different streams
                if use_custom_stream and test_gemm_overlap and HAS_AITER and stream_gemm is not None:
                    # Synchronize all streams before starting to get accurate baseline
                    torch.cuda.synchronize()
                    
                    # Record overall start time on default stream (after sync, before any launch)
                    overlap_start.record()
                    
                    # Record individual start events on their respective streams
                    allgather_start.record(stream)
                    gemm_start.record(stream_gemm)
                    
                    # Launch allgather on its stream
                    with torch.cuda.stream(stream):
                        success = allgather(input_tensor, output_tensor, elems_per_pe)
                    
                    # Launch gemm on separate stream (concurrent execution)
                    with torch.cuda.stream(stream_gemm):
                        _ = aiter.gemm_a8w8_CK(A_q, B_q, A_scale, B_scale, bias, dtypes.bf16)
                    
                    # Record individual end events
                    allgather_end.record(stream)
                    gemm_end.record(stream_gemm)
                    
                    # Wait for both operations to complete
                    allgather_end.synchronize()
                    gemm_end.synchronize()
                    
                    # Record overall end time on default stream (after both complete)
                    overlap_end.record()
                    overlap_end.synchronize()
                    
                    # Calculate elapsed times
                    allgather_time = allgather_start.elapsed_time(allgather_end) / 1000.0
                    gemm_time = gemm_start.elapsed_time(gemm_end) / 1000.0
                    overlap_time = overlap_start.elapsed_time(overlap_end) / 1000.0
                    
                    # Also calculate theoretical overlap time (should be close to max of the two)
                    # This helps verify the measurement accuracy
                    theoretical_overlap = max(allgather_time, gemm_time)
                    
                    if iter_idx >= warmup:
                        exec_times.append(allgather_time)
                        gemm_times.append(gemm_time)
                        overlap_times.append(overlap_time)
                    elif rank == 0:
                        measurement_accuracy = (overlap_time / theoretical_overlap) if theoretical_overlap > 0 else 1.0
                        print(f"Warmup iteration {iter_idx + 1}/{warmup}: AllGather={allgather_time:.6f}s, GEMM={gemm_time:.6f}s, Overlap={overlap_time:.6f}s (theoretical={theoretical_overlap:.6f}s, accuracy={(measurement_accuracy*100):.1f}%)")
                else:
                    # Sequential execution on single stream
                    if use_custom_stream:
                        allgather_start.record(stream)
                    else:
                        allgather_start.record()
                    
                    if use_custom_stream:
                        with torch.cuda.stream(stream):
                            success = allgather(input_tensor, output_tensor, elems_per_pe)
                    else:
                        success = allgather(input_tensor, output_tensor, elems_per_pe)
                    
                    if use_custom_stream:
                        allgather_end.record(stream)
                    else:
                        allgather_end.record()
                    
                    allgather_end.synchronize()
                    allgather_time = allgather_start.elapsed_time(allgather_end) / 1000.0
                    
                    if iter_idx >= warmup:
                        exec_times.append(allgather_time)
                    elif rank == 0:
                        print(f"Warmup iteration {iter_idx + 1}/{warmup}: {allgather_time:.6f}s")
                
                if not success:
                    print(f"PE {rank}: Allgather operation failed at iteration {iter_idx}")
                    break
        else:
            # Asynchronous mode (multiple SDMA queues, matches C++ test)
            if rank == 0:
                print(f"Using ASYNC mode (start_async + wait_async) to match C++ test")
                if warmup > 0:
                    print(f"Warmup iterations: {warmup}, Measurement iterations: {iterations}\n")
            
            for iter_idx in range(total_iters):
                if rank == 0 and (iter_idx == 0 or iter_idx == warmup):
                    stage = "Warmup" if iter_idx < warmup else "Measurement"
                    print(f"\n--- {stage} Iteration {iter_idx + 1} ---")
                
                dist.barrier()
                
                # Start async operation using context manager style
                if use_custom_stream:
                    with torch.cuda.stream(stream):
                        started = allgather.start_async(input_tensor, output_tensor, elems_per_pe)
                else:
                    started = allgather.start_async(input_tensor, output_tensor, elems_per_pe)
                
                if not started:
                    print(f"PE {rank}: Failed to start async operation")
                    break
                
                # Wait for completion (using the same stream)
                if use_custom_stream:
                    with torch.cuda.stream(stream):
                        exec_time = allgather.wait_async()
                else:
                    exec_time = allgather.wait_async()
                
                if exec_time < 0:
                    print(f"PE {rank}: Async operation failed")
                    break
                
                # Collect times after warmup
                if iter_idx >= warmup:
                    exec_times.append(exec_time)
                    if rank == 0 and len(exec_times) == 1:
                        print(f"PE {rank}: First measurement iteration: {exec_time:.6f}s")
                
                dist.barrier()
        
        # Synchronize stream before verification
        if use_custom_stream:
            stream.synchronize()
        torch.cuda.synchronize()

        # Calculate statistics from post-warmup iterations
        if len(exec_times) > 0:
            avg_time = np.mean(exec_times)
            min_time = np.min(exec_times)
            max_time = np.max(exec_times)
        else:
            avg_time = min_time = max_time = 0.0
        
        # Calculate GEMM statistics if available
        if len(gemm_times) > 0:
            gemm_avg_time = np.mean(gemm_times)
            gemm_min_time = np.min(gemm_times)
            gemm_max_time = np.max(gemm_times)
        else:
            gemm_avg_time = gemm_min_time = gemm_max_time = 0.0
        
        # Calculate overlap statistics if available
        if len(overlap_times) > 0:
            overlap_avg_time = np.mean(overlap_times)
            overlap_min_time = np.min(overlap_times)
            overlap_max_time = np.max(overlap_times)
        else:
            overlap_avg_time = overlap_min_time = overlap_max_time = 0.0
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"PE {rank} Local Performance Statistics")
            print(f"{'='*60}")
            print(f"AllGather Times:")
            print(f"  Min time: {min_time:.6f}s")
            print(f"  Max time: {max_time:.6f}s")
            print(f"  Avg time: {avg_time:.6f}s")
            
            if len(gemm_times) > 0:
                print(f"\nGEMM Times (concurrent on separate stream):")
                print(f"  Min time: {gemm_min_time:.6f}s")
                print(f"  Max time: {gemm_max_time:.6f}s")
                print(f"  Avg time: {gemm_avg_time:.6f}s")
                
                print(f"\nTotal Overlap Time (both operations):")
                print(f"  Min time: {overlap_min_time:.6f}s")
                print(f"  Max time: {overlap_max_time:.6f}s")
                print(f"  Avg time: {overlap_avg_time:.6f}s")
                
                # Theoretical overlap time (perfect concurrency = max of the two operations)
                theoretical_overlap = max(avg_time, gemm_avg_time)
                print(f"  Theoretical best (max of two): {theoretical_overlap:.6f}s")
                
                # Measurement accuracy
                measurement_overhead = overlap_avg_time - theoretical_overlap
                measurement_overhead_pct = (measurement_overhead / theoretical_overlap * 100) if theoretical_overlap > 0 else 0
                print(f"  Measurement overhead: {measurement_overhead:.6f}s ({measurement_overhead_pct:.2f}%)")
                
                print(f"\nOverlap Efficiency Analysis:")
                # Sequential time would be the sum of both operations
                sequential_time = avg_time + gemm_avg_time
                speedup = sequential_time / overlap_avg_time if overlap_avg_time > 0 else 0
                efficiency = (theoretical_overlap / overlap_avg_time * 100) if overlap_avg_time > 0 else 0
                print(f"  Sequential time (sum): {sequential_time:.6f}s")
                print(f"  Concurrent time (measured): {overlap_avg_time:.6f}s")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Concurrency efficiency: {efficiency:.2f}%")
            print(f"{'='*60}")

        # Verify results
        output_data_cpu = output_tensor.cpu().numpy()
        
        success = True
        for src_pe in range(npes):
            chunk = output_data_cpu[src_pe * elems_per_pe : (src_pe + 1) * elems_per_pe]
            expected_value = (src_pe + 1) * 1000
            
            if not np.all(chunk == expected_value):
                print(f"PE {rank}: Chunk from PE {src_pe} verification FAILED!")
                print(f"  Expected all values = {expected_value}")
                print(f"  Got first 10 values: {chunk[:10]}")
                print(f"  Got unique values: {np.unique(chunk)}")
                success = False
            else:
                print(f"PE {rank}: Chunk from PE {src_pe} verified (all values = {expected_value})")

        torch.cuda.synchronize()
        dist.barrier()
        
        # Gather global statistics
        min_time_tensor = torch.tensor([min_time], dtype=torch.float64)
        max_time_tensor = torch.tensor([max_time], dtype=torch.float64)
        avg_time_tensor = torch.tensor([avg_time], dtype=torch.float64)
        success_tensor = torch.tensor([1 if success else 0], dtype=torch.int32)
        
        dist.all_reduce(min_time_tensor, op=dist.ReduceOp.MIN)
        dist.all_reduce(max_time_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(avg_time_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(success_tensor, op=dist.ReduceOp.SUM)
        
        global_min = min_time_tensor.item()
        global_max = max_time_tensor.item()
        global_avg = avg_time_tensor.item() / npes
        passed_count = success_tensor.item()

        # Gather GEMM global statistics if available
        if len(gemm_times) > 0:
            gemm_min_tensor = torch.tensor([gemm_min_time], dtype=torch.float64)
            gemm_max_tensor = torch.tensor([gemm_max_time], dtype=torch.float64)
            gemm_avg_tensor = torch.tensor([gemm_avg_time], dtype=torch.float64)
            
            dist.all_reduce(gemm_min_tensor, op=dist.ReduceOp.MIN)
            dist.all_reduce(gemm_max_tensor, op=dist.ReduceOp.MAX)
            dist.all_reduce(gemm_avg_tensor, op=dist.ReduceOp.SUM)
            
            gemm_global_min = gemm_min_tensor.item()
            gemm_global_max = gemm_max_tensor.item()
            gemm_global_avg = gemm_avg_tensor.item() / npes
        
        # Gather overlap global statistics if available
        if len(overlap_times) > 0:
            overlap_min_tensor = torch.tensor([overlap_min_time], dtype=torch.float64)
            overlap_max_tensor = torch.tensor([overlap_max_time], dtype=torch.float64)
            overlap_avg_tensor = torch.tensor([overlap_avg_time], dtype=torch.float64)
            
            dist.all_reduce(overlap_min_tensor, op=dist.ReduceOp.MIN)
            dist.all_reduce(overlap_max_tensor, op=dist.ReduceOp.MAX)
            dist.all_reduce(overlap_avg_tensor, op=dist.ReduceOp.SUM)
            
            overlap_global_min = overlap_min_tensor.item()
            overlap_global_max = overlap_max_tensor.item()
            overlap_global_avg = overlap_avg_tensor.item() / npes

        if rank == 0:
            global_bandwidth = total_bytes / global_avg / (1024.0 * 1024.0 * 1024.0)
            
            print(f"\n{'='*60}")
            print(f"Global Performance Statistics")
            print(f"{'='*60}")
            print(f"AllGather Performance:")
            print(f"  Min time: {global_min:.6f}s")
            print(f"  Max time: {global_max:.6f}s")
            print(f"  Avg time: {global_avg:.6f}s")
            print(f"  Bandwidth: {global_bandwidth:.2f} GB/s")
            print(f"  Total data: {total_bytes / (1024.0 * 1024.0 * 1024.0):.3f} GB")
            
            if len(gemm_times) > 0:
                print(f"\nGEMM Performance (concurrent on separate stream):")
                print(f"  Min time: {gemm_global_min:.6f}s")
                print(f"  Max time: {gemm_global_max:.6f}s")
                print(f"  Avg time: {gemm_global_avg:.6f}s")
                
            if len(overlap_times) > 0:
                print(f"\nTotal Overlap Time (AllGather + GEMM concurrent):")
                print(f"  Min time: {overlap_global_min:.6f}s")
                print(f"  Max time: {overlap_global_max:.6f}s")
                print(f"  Avg time (measured): {overlap_global_avg:.6f}s")
                
                # Ideal overlap time would be max(allgather, gemm)
                ideal_overlap = max(global_avg, gemm_global_avg)
                measurement_overhead = overlap_global_avg - ideal_overlap
                measurement_overhead_pct = (measurement_overhead / ideal_overlap * 100) if ideal_overlap > 0 else 0
                print(f"  Theoretical best (max of two): {ideal_overlap:.6f}s")
                print(f"  Measurement overhead: {measurement_overhead:.6f}s ({measurement_overhead_pct:.2f}%)")
                
                print(f"\nConcurrency Analysis:")
                sequential_time = global_avg + gemm_global_avg
                speedup = sequential_time / overlap_global_avg if overlap_global_avg > 0 else 0
                time_saved = sequential_time - overlap_global_avg
                saved_percentage = (time_saved / sequential_time * 100) if sequential_time > 0 else 0
                overlap_efficiency = (ideal_overlap / overlap_global_avg * 100) if overlap_global_avg > 0 else 0
                
                print(f"  Sequential execution time: {sequential_time:.6f}s")
                print(f"  Concurrent execution time: {overlap_global_avg:.6f}s")
                print(f"  Time saved: {time_saved:.6f}s ({saved_percentage:.2f}%)")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Concurrency efficiency: {overlap_efficiency:.2f}%")
                
                if gemm_global_avg < global_avg:
                    print(f"  GEMM is {(global_avg/gemm_global_avg):.2f}x faster than AllGather")
                else:
                    print(f"  AllGather is {(gemm_global_avg/global_avg):.2f}x faster than GEMM")
            
            print(f"\nPEs passed: {passed_count}/{npes}")
            
            if passed_count == npes:
                print(f"\n=== Test PASSED ===")
            else:
                print(f"\n=== Test FAILED ===")
            print(f"{'='*60}\n")

        # Proper cleanup order to avoid race conditions
        torch.cuda.synchronize()  # 1. Ensure all GPU operations complete
        dist.barrier()             # 2. Synchronize all processes
        del allgather             # 3. Delete object (releases SHMEM buffers)
        dist.barrier()             # 4. Wait for all processes to finish cleanup
        shmem.shmem_finalize()    # 5. Finalize SHMEM (closes SDMA/HSA resources)
        
        if not success:
            raise AssertionError(f"PE {rank}: Allgather verification failed")


def test_allgather(elems=67108864, world_size=8, iterations=10, warmup=1, use_custom_stream=False, test_gemm_overlap=False):
    """Run Allgather SDMA test"""
    os.environ.setdefault('MORI_ENABLE_SDMA', '1')
    port = get_free_port()
    torch.multiprocessing.spawn(
        _test_allgather,
        args=(world_size, port, elems, iterations, warmup, use_custom_stream, test_gemm_overlap),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Allgather SDMA (similar to C++ example)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--elems", type=int, default=67108864, help="Elements per PE")
    parser.add_argument("--world-size", type=int, default=8, help="Number of processes")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--enable-sdma", type=int, default=1, choices=[0, 1], help="Enable SDMA")
    parser.add_argument("--use-custom-stream", action="store_true", help="Use custom CUDA stream instead of default stream")
    parser.add_argument("--test-gemm-overlap", action="store_true", help="Test GEMM and AllGather overlap on different streams")
    args = parser.parse_args()
    os.environ['MORI_ENABLE_SDMA'] = str(args.enable_sdma)
    
    print(f"Allgather SDMA Test")
    print(f"  Elements per PE: {args.elems:,}")
    print(f"  World size: {args.world_size}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Custom Stream: {args.use_custom_stream}")
    print(f"  Test GEMM Overlap: {args.test_gemm_overlap}")
    if args.test_gemm_overlap and not HAS_AITER:
        print(f"  WARNING: aiter not available, GEMM testing will be skipped")
    print("-" * 60)
    
    test_allgather(args.elems, args.world_size, args.iterations, args.warmup, args.use_custom_stream, args.test_gemm_overlap)
