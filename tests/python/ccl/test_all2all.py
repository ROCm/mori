#!/usr/bin/env python3
"""
All2All SDMA Test using torch.distributed and multiprocessing
"""

import os
import numpy as np
import torch
import torch.distributed as dist
import mori.shmem as shmem
from mori.ccl import All2allSdma
from tests.python.utils import TorchDistContext, get_free_port


def _test_all2all(rank, world_size, port, elems, iterations, warmup):
    """Worker function for each process"""
    
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        
        my_pe = shmem.shmem_mype()
        npes = shmem.shmem_npes()
        
        assert my_pe == rank, f"PE mismatch: {my_pe} != {rank}"
        assert npes == world_size, f"npes mismatch: {npes} != {world_size}"
        
        # Match C++ naming and logic
        elems_per_pe = elems  # Elements each PE sends to each target PE (like C++ elemsPerPe)
        bytes_per_pe = elems_per_pe * 4  # Bytes per PE chunk (like C++ bytesPerPe)
        total_bytes = bytes_per_pe * npes  # Total bytes per PE: all chunks combined (like C++ totalBytes)
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"All2All SDMA Test")
            print(f"World size: {world_size}")
            print(f"Elements per PE: {elems_per_pe:,}")
            print(f"Data size: {total_bytes / (1024**2):.2f} MB per PE, {total_bytes * npes / (1024**2):.2f} MB total")
            print(f"Iterations: {iterations}" + (f" (warmup: {warmup})" if warmup > 0 else ""))
            print(f"{'='*60}\n")
        
        print(f"PE {rank}/{world_size}: SHMEM initialized, myPe={my_pe}, npes={npes}")

        # Create All2all object with sufficient buffer size
        all2all = All2allSdma(my_pe, npes, 
                             input_buffer_size=total_bytes,
                             output_buffer_size=total_bytes)
        print(f"PE {rank}: Created All2allSdma object")

        # Allocate GPU memory
        # Note: Using torch.uint32 to match C++ All2allSdma<uint32_t>
        device = torch.device(f"cuda:{rank}")
        input_tensor = torch.zeros(elems_per_pe * npes, dtype=torch.uint32, device=device)
        output_tensor = torch.zeros(elems_per_pe * npes, dtype=torch.uint32, device=device)
        
        # Prepare data: PE i sends value (i+1)*1000 + j to PE j
        input_data_cpu = np.zeros(elems_per_pe * npes, dtype=np.uint32)
        for dest_pe in range(npes):
            value = (my_pe + 1) * 1000 + dest_pe
            input_data_cpu[dest_pe * elems_per_pe : (dest_pe + 1) * elems_per_pe] = value
        
        # Copy to GPU
        input_tensor.copy_(torch.from_numpy(input_data_cpu))

        if rank == 0:
            print(f"PE {rank}: Prepared input data with pattern: (src_pe+1)*1000 + dest_pe")
            print(f"  Sending to PE 0: {input_tensor[0].item()} (expected: {(my_pe+1)*1000 + 0})")
            print(f"  Sending to PE 1: {input_tensor[elems_per_pe].item()} (expected: {(my_pe+1)*1000 + 1})")

        torch.cuda.synchronize()
        dist.barrier()

        # Execute All2All multiple times
        exec_times = []
        total_iters = warmup + iterations
        use_async = True  # Use async mode to match C++ test
        
        if not use_async:
            # Synchronous mode (single SDMA queue)
            for iter_idx in range(total_iters):
                exec_time = all2all(input_tensor, output_tensor, elems_per_pe)
                
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
                started = all2all.start_async(input_tensor, output_tensor, elems_per_pe)
                if not started:
                    print(f"PE {rank}: Failed to start async operation")
                    break
                
                # Wait for completion
                exec_time = all2all.wait_async()
                
                if exec_time < 0:
                    print(f"PE {rank}: Async operation failed")
                    break
                
                # Collect times after warmup
                if iter_idx >= warmup:
                    exec_times.append(exec_time)
                    if rank == 0 and len(exec_times) == 1:
                        print(f"PE {rank}: First measurement iteration: {exec_time:.6f}s")
                
                dist.barrier()

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
        
        # Get output transit buffer and verify
        # Pass the output_tensor to ensure the buffer is on the correct device
        output_transit_buffer = all2all.get_output_transit_buffer(device=output_tensor)
        output_transit_buffer_cpu = output_transit_buffer.cpu().numpy()
        
        if rank == 0:
            print(f"\nPE {rank}: Output transit buffer size: {output_transit_buffer.size(0)} elements")
            print(f"PE {rank}: Output transit buffer first 10 values: {output_transit_buffer_cpu[:10]}")
        
        success = True
        for src_pe in range(npes):
            chunk = output_data_cpu[src_pe * elems_per_pe : (src_pe + 1) * elems_per_pe]
            expected_value = (src_pe + 1) * 1000 + my_pe
            
            if not np.all(chunk == expected_value):
                print(f"PE {rank}: Chunk from PE {src_pe} verification FAILED!")
                print(f"  Expected all values = {expected_value}")
                print(f"  Got first 10 values: {chunk[:10]}")
                print(f"  Got unique values: {np.unique(chunk)}")
                success = False
            elif rank == 0:
                print(f"PE {rank}: Chunk from PE {src_pe} verified (all values = {expected_value})")
        
        # Verify output transit buffer contains the same data as output_tensor
        # The output transit buffer should contain the same data as output_tensor
        # (at least the first total_bytes / sizeof(uint32_t) elements)
        expected_elements = total_bytes // 4
        if output_transit_buffer_cpu.size >= expected_elements:
            transit_chunk = output_transit_buffer_cpu[:expected_elements]
            if np.array_equal(transit_chunk, output_data_cpu):
                if rank == 0:
                    print(f"PE {rank}: Output transit buffer matches output_tensor ✓")
            else:
                print(f"PE {rank}: Output transit buffer does NOT match output_tensor!")
                print(f"  First 10 values in transit buffer: {transit_chunk[:10]}")
                print(f"  First 10 values in output_tensor: {output_data_cpu[:10]}")
                success = False
        else:
            if rank == 0:
                print(f"PE {rank}: Output transit buffer size ({output_transit_buffer_cpu.size}) is smaller than expected ({expected_elements})")

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

        # Proper cleanup order to avoid race conditions
        torch.cuda.synchronize()  # 1. Ensure all GPU operations complete
        dist.barrier()             # 2. Synchronize all processes
        del all2all               # 3. Delete object (releases SHMEM buffers)
        dist.barrier()             # 4. Wait for all processes to finish cleanup
        shmem.shmem_finalize()    # 5. Finalize SHMEM (closes SDMA/HSA resources)
        
        if not success:
            raise AssertionError(f"PE {rank}: All2All verification failed")


def test_all2all(elems=67108864, world_size=8, iterations=10, warmup=1):
    """Run All2All SDMA test"""
    os.environ.setdefault('MORI_ENABLE_SDMA', '1')
    port = get_free_port()
    torch.multiprocessing.spawn(
        _test_all2all,
        args=(world_size, port, elems, iterations, warmup),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test All2All SDMA (similar to C++ example)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--elems", type=int, default=67108864, help="Elements per PE")
    parser.add_argument("--world-size", type=int, default=8, help="Number of processes")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--enable-sdma", type=int, default=1, choices=[0, 1], help="Enable SDMA")
    args = parser.parse_args()
    os.environ['MORI_ENABLE_SDMA'] = str(args.enable_sdma)
    
    print(f"All2All SDMA Test")
    print(f"  Elements per PE: {args.elems:,}")
    print(f"  World size: {args.world_size}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Warmup: {args.warmup}")
    print("-" * 60)
    
    test_all2all(args.elems, args.world_size, args.iterations, args.warmup)
