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


def _test_allgather(rank, world_size, port, elems, iterations, warmup):
    """Worker function for each process"""
    
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        
        assert my_pe == rank, f"PE mismatch: {my_pe} != {rank}"
        assert npes == world_size, f"npes mismatch: {npes} != {world_size}"
        
        bytes_per_pe = elems * 4  # Input: only local data
        total_bytes = bytes_per_pe * npes  # Output: all chunks combined
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Allgather SDMA Test")
            print(f"World size: {world_size}")
            print(f"Elements per PE: {elems:,}")
            print(f"Data size: {bytes_per_pe / (1024**2):.2f} MB per PE (input), {total_bytes / (1024**2):.2f} MB total (output)")
            print(f"Iterations: {iterations}" + (f" (warmup: {warmup})" if warmup > 0 else ""))
            print(f"{'='*60}\n")
        
        print(f"PE {rank}/{world_size}: SHMEM initialized, myPe={my_pe}, npes={npes}")

        # Create Allgather object
        # Input buffer size: only needs to hold this PE's data
        # Output buffer size: needs to hold all PEs' data
        allgather = AllgatherSdma(my_pe, npes, 
                                  input_buffer_size=bytes_per_pe,
                                  output_buffer_size=total_bytes)
        print(f"PE {rank}: Created AllgatherSdma object")

        # Allocate GPU memory
        device = torch.device(f"cuda:{rank}")
        # Input tensor: only holds this PE's data (size = elems)
        input_tensor = torch.zeros(elems, dtype=torch.int32, device=device)
        # Output tensor: receives all PEs' data (size = elems * npes)
        output_tensor = torch.zeros(elems * npes, dtype=torch.int32, device=device)
        
        # === UPDATED: Prepare data for Allgather ===
        # Each PE has its own unique data that will be sent to all other PEs
        # Format: value = (my_pe + 1) * 1000 (constant for all elements)
        # This creates unique but consistent data for each PE
        # Example for 2 PEs:
        #   PE 0 data: 1000, 1000, 1000, ...
        #   PE 1 data: 2000, 2000, 2000, ...
        value = (my_pe + 1) * 1000
        input_data_cpu = np.full(elems, value, dtype=np.uint32)
        
        # Copy to GPU
        input_tensor.copy_(torch.from_numpy(input_data_cpu))

        if rank == 0:
            print(f"\nData Pattern Explanation:")
            print(f"  Each PE sends the same data to all other PEs")
            print(f"  Format: value = (source_pe + 1) * 1000")
            print(f"  After Allgather, every PE will have identical output buffers")
        
        print(f"PE {rank}: Prepared input data: value = {value} (all {elems} elements)")
        
        if rank == 0:
            print(f"\nExpected results after Allgather:")
            for src_pe in range(npes):
                expected_value = (src_pe + 1) * 1000
                print(f"  Chunk from PE {src_pe}: all values = {expected_value}")

        torch.cuda.synchronize()
        dist.barrier()

        # Execute Allgather multiple times
        exec_times = []
        total_iters = warmup + iterations
        
        for iter_idx in range(total_iters):
            exec_time = allgather(input_tensor, output_tensor, elems)
            
            if iter_idx >= warmup:
                exec_times.append(exec_time)
            elif rank == 0 and warmup > 0:
                print(f"Warmup iteration {iter_idx + 1}/{warmup}: {exec_time:.6f}s")

        avg_time = np.mean(exec_times)
        min_time = np.min(exec_times)
        max_time = np.max(exec_times)
        
        if rank == 0:
            print(f"\nPE {rank} local statistics:")
            print(f"  Min time: {min_time:.6f}s")
            print(f"  Max time: {max_time:.6f}s")
            print(f"  Avg time: {avg_time:.6f}s")

        # === UPDATED: Verify results ===
        output_data_cpu = output_tensor.cpu().numpy()
        
        success = True
        for src_pe in range(npes):
            chunk = output_data_cpu[src_pe * elems : (src_pe + 1) * elems]
            expected_value = (src_pe + 1) * 1000  # Each PE's unique value
            
            if not np.all(chunk == expected_value):
                print(f"\nPE {rank}: Chunk from PE {src_pe} verification FAILED!")
                print(f"  Expected all values = {expected_value}")
                print(f"  Got first 10 values: {chunk[:10]}")
                print(f"  Got unique values: {np.unique(chunk[:100])}")  # Check first 100 elements
                success = False
                break  # Stop checking on first failure
            elif rank == 0:
                print(f"PE {rank}: Chunk from PE {src_pe} verified (all values = {expected_value})")

        torch.cuda.synchronize()
        dist.barrier()
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
            global_bandwidth = total_bytes / global_max / (1024.0 * 1024.0 * 1024.0)
            
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

        del allgather
        torch.cuda.synchronize()
        dist.barrier()
        shmem.shmem_finalize()
        
        if not success:
            raise AssertionError(f"PE {rank}: Allgather verification failed")


def test_allgather(elems=67108864, world_size=8, iterations=10, warmup=0):
    """Run Allgather SDMA test"""
    os.environ.setdefault('MORI_ENABLE_SDMA', '1')
    port = get_free_port()
    torch.multiprocessing.spawn(
        _test_allgather,
        args=(world_size, port, elems, iterations, warmup),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Allgather SDMA (similar to C++ example)\n\n"
                    "Note: You may see 'ERROR code: 20 hsaKmtCloseKFD() failed' "
                    "during cleanup. This is a known issue with concurrent SDMA "
                    "resource cleanup in multi-process scenarios and can be ignored "
                    "if the test passes successfully.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--elems", type=int, default=67108864, help="Elements per PE")
    parser.add_argument("--world-size", type=int, default=8, help="Number of processes")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup iterations")
    parser.add_argument("--enable-sdma", type=int, default=1, choices=[0, 1], help="Enable SDMA")
    args = parser.parse_args()
    os.environ['MORI_ENABLE_SDMA'] = str(args.enable_sdma)
    
    print(f"Allgather SDMA Test")
    print(f"  Elements per PE: {args.elems:,}")
    print(f"  World size: {args.world_size}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Warmup: {args.warmup}")
    print("-" * 60)
    
    test_allgather(args.elems, args.world_size, args.iterations, args.warmup)