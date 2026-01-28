"""
Distributed All2All test for single machine with 8 GPUs.
Use the actual available function name.
Run with: mpirun -np 8 python test_all2all_distributed.py
"""

import sys
import torch
import numpy as np

# Try MPI
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    print("mpi4py not installed, running single process")
    HAS_MPI = False

# Import after potential MPI init
from mori import cpp as mori_cpp

# Check what functions are available
print("Available functions in mori_cpp:")
for attr in dir(mori_cpp):
    if not attr.startswith('_') and 'all2all' in attr.lower():
        print(f"  - {attr}")

# Determine which function to use
if hasattr(mori_cpp, 'all2all_sdma_int32'):
    ALL2ALL_FUNC = mori_cpp.all2all_sdma_int32
    DTYPE = torch.int32
    print("Using all2all_sdma_int32")
elif hasattr(mori_cpp, 'all2all_sdma'):
    ALL2ALL_FUNC = mori_cpp.all2all_sdma
    # all2all_sdma might expect float32, but let's try int32
    DTYPE = torch.float32  # Try float32 first
    print("Using all2all_sdma (expects float32)")
else:
    print("ERROR: No all2all function found!")
    sys.exit(1)


def distributed_test():
    """Distributed test matching C++ example."""

    # Get MPI info if available
    if HAS_MPI:
        comm = MPI.COMM_WORLD
        myPe = comm.Get_rank()
        npes = comm.Get_size()
    else:
        myPe = 0
        npes = 1
        comm = None

    print(f"PE {myPe} of {npes} started")

    # Check GPU
    if not torch.cuda.is_available():
        print(f"PE {myPe}: CUDA not available")
        return False

    # Set GPU for this process
    if HAS_MPI and torch.cuda.device_count() >= npes:
        gpu_id = myPe % torch.cuda.device_count()
        torch.cuda.set_device(gpu_id)
        print(f"PE {myPe}: Using GPU {gpu_id}")

    # Configuration - smaller for testing
    elemsPerPe = 1024  # Small for quick test
    bytesPerPe = elemsPerPe * 4  # Assuming 4 bytes per element
    totalBytes = bytesPerPe * npes

    if myPe == 0:
        print(f"\nConfiguration:")
        print(f"  Processes: {npes}")
        print(f"  Elements per process: {elemsPerPe}")
        print(f"  Total data: {totalBytes/1024/1024:.2f} MB")
        print(f"  Using dtype: {DTYPE}")

    try:
        # Allocate memory
        total_elements = elemsPerPe * npes
        input_tensor = torch.zeros(total_elements, dtype=DTYPE, device='cuda')
        output_tensor = torch.zeros(total_elements, dtype=DTYPE, device='cuda')

        # Initialize data - adjust pattern based on dtype
        if DTYPE == torch.int32:
            # Use integer pattern
            for k in range(npes):
                for i in range(elemsPerPe):
                    # C++ pattern: k + (myPe + 1) * 100
                    value = k + (myPe + 1) * 100
                    input_tensor[k * elemsPerPe + i] = value
        else:
            # Use float pattern
            for k in range(npes):
                for i in range(elemsPerPe):
                    value = float(k + (myPe + 1) * 100)
                    input_tensor[k * elemsPerPe + i] = value

        # Show sample
        print(f"PE {myPe}: Initial data (first 2 elements of each chunk):")
        for pe in range(min(3, npes)):
            start = pe * elemsPerPe
            values = input_tensor[start:start+2].cpu().numpy()
            print(f"  Chunk {pe}: {values}")

        # Ensure contiguous
        input_tensor = input_tensor.contiguous()
        output_tensor = output_tensor.contiguous()

        torch.cuda.synchronize()

        # Sync all processes
        if HAS_MPI:
            comm.Barrier()

        # Call All2All
        print(f"PE {myPe}: Calling All2All...")

        local_duration = ALL2ALL_FUNC(
            input_tensor.data_ptr(),
            output_tensor.data_ptr(),
            elemsPerPe,  # elements per PE
            0            # default stream
        )

        torch.cuda.synchronize()

        if HAS_MPI:
            comm.Barrier()

        print(f"PE {myPe}: Local time: {local_duration:.9f}")

        # Copy result back
        output_cpu = output_tensor.cpu().numpy()

        # Verify - check first few elements
        success = True
        for pe in range(npes):
            if DTYPE == torch.int32:
                expected_value = myPe + (pe + 1) * 100
            else:
                expected_value = float(myPe + (pe + 1) * 100)

            start_idx = pe * elemsPerPe

            # Check first 5 elements
            chunk = output_cpu[start_idx:start_idx+5]

            # Allow small tolerance for floats
            if DTYPE == torch.float32:
                if not np.allclose(chunk, expected_value, rtol=1e-5):
                    success = False
                    print(f"PE {myPe}: Chunk {pe} mismatch: {chunk} != {expected_value}")
                    break
            else:
                if not np.all(chunk == expected_value):
                    success = False
                    print(f"PE {myPe}: Chunk {pe} mismatch: {chunk} != {expected_value}")
                    break

        if success:
            print(f"PE {myPe}: Verification PASSED")
        else:
            print(f"PE {myPe}: Verification FAILED")

        # Calculate bandwidth
        if local_duration > 0:
            local_bandwidth = totalBytes / local_duration / (1024**3)
            print(f"PE {myPe}: Bandwidth: {local_bandwidth:.3f} GB/s")

        # Gather global stats
        if HAS_MPI:
            # Get max duration across all processes
            global_max_duration = comm.reduce(local_duration, op=MPI.MAX, root=0)

            if myPe == 0 and global_max_duration > 0:
                global_bandwidth = totalBytes / global_max_duration / (1024**3)
                print(f"\n=== Global Results ===")
                print(f"Global max time: {global_max_duration:.9f}")
                print(f"Global bandwidth: {global_bandwidth:.3f} GB/s")

        return success

    except Exception as e:
        print(f"PE {myPe}: Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("Distributed All2All Test")
    print("For single machine with multiple GPUs")

    # Check GPU count if using MPI
    if HAS_MPI:
        world_size = MPI.COMM_WORLD.Get_size()
        gpu_count = torch.cuda.device_count()

        if world_size > gpu_count:
            print(f"Warning: {world_size} processes but only {gpu_count} GPUs")
            print("Some processes will share GPUs")

    # Run test
    success = distributed_test()

    # Final sync
    if HAS_MPI:
        MPI.COMM_WORLD.Barrier()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("\n=== Test Completed ===")
    else:
        print("\n=== Test Completed ===")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
