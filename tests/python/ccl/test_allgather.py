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


def _test_allgather(rank, world_size, port, elems, iterations, warmup, use_custom_stream):
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

        # Create CUDA stream for allgather operations (if requested)
        if use_custom_stream:
            stream = torch.cuda.Stream(device=device)
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
        total_iters = warmup + iterations
        use_async = True  # Use async mode to match C++ test
        
        if not use_async:
            # Synchronous mode (single SDMA queue)
            for iter_idx in range(total_iters):
                exec_time = allgather(input_tensor, output_tensor, elems_per_pe, stream)
                
                if iter_idx >= warmup:
                    exec_times.append(exec_time)
                elif rank == 0:
                    print(f"Warmup iteration {iter_idx + 1}/{warmup}: {exec_time:.6f}s")
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
                
                # Start async operation
                started = allgather.start_async(input_tensor, output_tensor, elems_per_pe, stream)
                if not started:
                    print(f"PE {rank}: Failed to start async operation")
                    break
                
                # Wait for completion (using the same stream)
                exec_time = allgather.wait_async(stream)
                
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
        
        if rank == 0:
            print(f"\nPE {rank} local statistics:")
            print(f"  Min time: {min_time:.6f}s")
            print(f"  Max time: {max_time:.6f}s")
            print(f"  Avg time: {avg_time:.6f}s")

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

        if rank == 0:
            global_bandwidth = total_bytes / global_avg / (1024.0 * 1024.0 * 1024.0)
            
            print(f"\n=== Performance Statistics ===")
            print(f"Min time: {global_min:.6f}s")
            print(f"Max time: {global_max:.6f}s")
            print(f"Avg time: {global_avg:.6f}s")
            print(f"Bandwidth: {global_bandwidth:.2f} GB/s")
            print(f"Total data: {total_bytes / (1024.0 * 1024.0 * 1024.0):.3f} GB")
            print(f"\nPEs passed: {passed_count}/{npes}")
            
            if passed_count == npes:
                print(f"\n=== Test PASSED ===\n")
            else:
                print(f"\n=== Test FAILED ===\n")

        # Proper cleanup order to avoid race conditions
        torch.cuda.synchronize()  # 1. Ensure all GPU operations complete
        dist.barrier()             # 2. Synchronize all processes
        del allgather             # 3. Delete object (releases SHMEM buffers)
        dist.barrier()             # 4. Wait for all processes to finish cleanup
        shmem.shmem_finalize()    # 5. Finalize SHMEM (closes SDMA/HSA resources)
        
        if not success:
            raise AssertionError(f"PE {rank}: Allgather verification failed")


def test_allgather(elems=67108864, world_size=8, iterations=10, warmup=1, use_custom_stream=False):
    """Run Allgather SDMA test"""
    os.environ.setdefault('MORI_ENABLE_SDMA', '1')
    port = get_free_port()
    torch.multiprocessing.spawn(
        _test_allgather,
        args=(world_size, port, elems, iterations, warmup, use_custom_stream),
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
    args = parser.parse_args()
    os.environ['MORI_ENABLE_SDMA'] = str(args.enable_sdma)
    
    print(f"Allgather SDMA Test")
    print(f"  Elements per PE: {args.elems:,}")
    print(f"  World size: {args.world_size}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Warmup: {args.warmup}")
    print(f"  Custom Stream: {args.use_custom_stream}")
    print("-" * 60)
    
    test_allgather(args.elems, args.world_size, args.iterations, args.warmup, args.use_custom_stream)
