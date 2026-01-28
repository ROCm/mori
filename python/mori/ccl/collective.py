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

from mori import cpp as mori_cpp
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import time


@dataclass
class All2allSdmaResult:
    """Result container for all2all_sdma operation."""
    output_tensor: torch.Tensor
    duration_seconds: float
    bandwidth_gbps: float


class All2allMode(Enum):
    """Operation mode for All2All."""
    SYNC = 0
    ASYNC = 1


def _get_cpp_all2all_func(dtype: torch.dtype):
    """Get the appropriate C++ function based on dtype."""
    # Map torch dtypes to C++ type-specific functions
    dtype_to_func = {
        torch.float32: "All2all_sdma_float",
        torch.float64: "All2all_sdma_double",
        torch.float16: "All2all_sdma_half",
        torch.bfloat16: "All2all_sdma_bfloat16",
        torch.int8: "All2all_sdma_int8",
        torch.uint8: "All2all_sdma_uint8",
        torch.int16: "All2all_sdma_int16",
        torch.uint16: "All2all_sdma_uint16",
        torch.int32: "All2all_sdma_int32",
        torch.uint32: "All2all_sdma_uint32",
        torch.int64: "All2all_sdma_int64",
        torch.uint64: "All2all_sdma_uint64",
    }
    
    if dtype not in dtype_to_func:
        # Fallback to generic function if type-specific not available
        return getattr(mori_cpp, "All2all_sdma", None)
    
    func_name = dtype_to_func[dtype]
    return getattr(mori_cpp, func_name, None)


def all2all_sdma(
    input_tensor: torch.Tensor,
    output_tensor: Optional[torch.Tensor] = None,
    stream: Optional[torch.cuda.Stream] = None,
    mode: All2allMode = All2allMode.SYNC,
    verify: bool = False,
) -> All2allSdmaResult:
    """
    Perform All2All operation using SDMA.
    
    This is a direct Python wrapper for the C++ template function:
    template <typename T> double All2all_sdma(T* input, T* output, size_t total_count, hipStream_t stream)
    
    Args:
        input_tensor: Input tensor containing local data.
                     Each rank should have tensor of same shape [local_count].
        output_tensor: Optional pre-allocated output tensor.
                      If None, will be allocated automatically.
                      Should be of size [world_size * local_count].
        stream: CUDA stream for asynchronous execution.
        mode: Synchronous or asynchronous mode.
        verify: If True, verify the operation results.
        
    Returns:
        All2allSdmaResult containing output tensor and performance metrics.
        
    Raises:
        ValueError: If input validation fails.
        RuntimeError: If operation fails.
    """
    # 1. Validate and prepare input
    if not input_tensor.is_cuda:
        raise ValueError("Input tensor must be on GPU")
    
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    
    # 2. Get distributed info
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')  # or 'mpi' depending on your setup
    
    world_size = dist.get_world_size()
    local_count = input_tensor.numel()
    
    # 3. Prepare or validate output tensor
    total_count = local_count * world_size
    
    if output_tensor is None:
        # Allocate output tensor
        output_tensor = torch.empty(
            total_count,
            dtype=input_tensor.dtype,
            device=input_tensor.device
        )
    else:
        # Validate existing output tensor
        if output_tensor.numel() != total_count:
            raise ValueError(f"Output tensor size mismatch. "
                           f"Expected {total_count}, got {output_tensor.numel()}")
        if output_tensor.dtype != input_tensor.dtype:
            raise ValueError("Output tensor dtype must match input tensor dtype")
        if not output_tensor.is_cuda:
            raise ValueError("Output tensor must be on GPU")
        if not output_tensor.is_contiguous():
            output_tensor = output_tensor.contiguous()
    
    # 4. Get C++ function for this dtype
    cpp_func = _get_cpp_all2all_func(input_tensor.dtype)
    if cpp_func is None:
        raise RuntimeError(f"No All2all_sdma implementation found for dtype {input_tensor.dtype}")
    
    # 5. Prepare stream
    if stream is None:
        if mode == All2allMode.ASYNC:
            stream = torch.cuda.Stream()
        else:
            stream = torch.cuda.current_stream()
    
    # Convert PyTorch stream to HIP stream
    hip_stream = stream.cuda_stream
    
    # 6. Perform All2All operation
    # Note: total_count parameter in C++ is the per-rank element count
    start_time = time.time()
    
    # Call the C++ function
    cpp_duration = cpp_func(
        input_tensor.data_ptr(),
        output_tensor.data_ptr(),
        local_count,  # This is total_count per rank
        hip_stream
    )
    
    # Wait for completion if synchronous
    if mode == All2allMode.SYNC:
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # 7. Calculate duration
    # Use C++ returned duration if valid, otherwise use Python timing
    if cpp_duration > 0:
        duration = cpp_duration
    else:
        duration = end_time - start_time
    
    # 8. Calculate bandwidth
    data_size_bytes = total_count * input_tensor.element_size()
    bandwidth_gbps = data_size_bytes / duration / (1024**3)
    
    # 9. Verify if requested
    if verify:
        _verify_all2all_result(input_tensor, output_tensor, world_size, local_count)
    
    return All2allSdmaResult(
        output_tensor=output_tensor,
        duration_seconds=duration,
        bandwidth_gbps=bandwidth_gbps
    )


def all2all_sdma_benchmark(
    local_count: int,
    dtype: torch.dtype = torch.float32,
    iterations: int = 10,
    warmup_iterations: int = 2,
    mode: All2allMode = All2allMode.SYNC,
) -> Dict[str, Any]:
    """
    Benchmark the All2All SDMA operation.
    
    Args:
        local_count: Number of elements per rank.
        dtype: Data type for benchmark.
        iterations: Number of benchmark iterations.
        warmup_iterations: Number of warmup iterations.
        mode: Synchronous or asynchronous mode.
        
    Returns:
        Dictionary containing benchmark results.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Create synthetic data similar to C++ example
    # Pattern: each PE's chunk has value = PE rank + (chunk index + 1) * 100
    local_data = torch.zeros(local_count * world_size, dtype=dtype, device='cuda')
    
    for k in range(world_size):
        base_value = (rank + 1) * 100
        for i in range(local_count):
            local_data[k * local_count + i] = base_value + k
    
    # Prepare output tensor
    total_count = local_count * world_size
    output_tensor = torch.empty(total_count, dtype=dtype, device='cuda')
    
    # Warmup
    for _ in range(warmup_iterations):
        all2all_sdma(
            local_data,
            output_tensor,
            mode=mode,
            verify=False
        )
    
    # Benchmark
    durations = []
    for i in range(iterations):
        result = all2all_sdma(
            local_data,
            output_tensor,
            mode=mode,
            verify=False
        )
        durations.append(result.duration_seconds)
        
        # Verify on first iteration
        if i == 0:
            _verify_all2all_result(local_data, output_tensor, world_size, local_count)
    
    # Calculate statistics
    avg_duration = sum(durations) / len(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    
    data_size_bytes = total_count * torch.tensor([], dtype=dtype).element_size()
    avg_bandwidth_gbps = data_size_bytes / avg_duration / (1024**3)
    
    # Gather global statistics
    if world_size > 1:
        global_durations = [torch.tensor(0.0, device='cuda') for _ in range(world_size)]
        avg_duration_tensor = torch.tensor(avg_duration, device='cuda')
        dist.all_gather(global_durations, avg_duration_tensor)
        
        global_max_duration = max(d.item() for d in global_durations)
        global_bandwidth_gbps = data_size_bytes / global_max_duration / (1024**3)
    else:
        global_max_duration = avg_duration
        global_bandwidth_gbps = avg_bandwidth_gbps
    
    return {
        'rank': rank,
        'world_size': world_size,
        'local_count': local_count,
        'total_count': total_count,
        'data_size_bytes': data_size_bytes,
        'dtype': str(dtype),
        'iterations': iterations,
        'avg_duration_seconds': avg_duration,
        'min_duration_seconds': min_duration,
        'max_duration_seconds': max_duration,
        'avg_bandwidth_gbps': avg_bandwidth_gbps,
        'global_max_duration_seconds': global_max_duration,
        'global_bandwidth_gbps': global_bandwidth_gbps,
        'mode': mode.name,
    }


def _verify_all2all_result(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    world_size: int,
    local_count: int
):
    """
    Verify All2All operation results.
    
    This follows the same verification logic as the C++ example.
    """
    torch.cuda.synchronize()
    
    # Reshape output for easier verification
    output_reshaped = output_tensor.view(world_size, local_count)
    
    # Get rank for verification pattern
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Check each chunk
    for chunk_idx in range(world_size):
        expected_base = rank + (chunk_idx + 1) * 100
        chunk_data = output_reshaped[chunk_idx]
        
        # Check if all elements in chunk have the expected value
        if not torch.all(chunk_data == expected_base):
            mismatch_mask = chunk_data != expected_base
            mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=True)[0]
            
            if len(mismatch_indices) > 0:
                first_idx = mismatch_indices[0].item()
                actual_value = chunk_data[first_idx].item()
                
                raise AssertionError(
                    f"Verification FAILED at rank {rank}, chunk {chunk_idx}, "
                    f"element {first_idx}: expected {expected_base}, got {actual_value}"
                )
    
    if rank == 0:
        print(f"All2All SDMA verification PASSED")


# Convenience functions for common data types
def all2all_sdma_float(*args, **kwargs):
    """Convenience function for float32 All2All."""
    return all2all_sdma(*args, **kwargs)


def all2all_sdma_half(*args, **kwargs):
    """Convenience function for float16 All2All."""
    return all2all_sdma(*args, **kwargs)


def all2all_sdma_bfloat16(*args, **kwargs):
    """Convenience function for bfloat16 All2All."""
    return all2all_sdma(*args, **kwargs)


def all2all_sdma_int32(*args, **kwargs):
    """Convenience function for int32 All2All."""
    return all2all_sdma(*args, **kwargs)


def all2all_sdma_uint32(*args, **kwargs):
    """Convenience function for uint32 All2All."""
    return all2all_sdma(*args, **kwargs)