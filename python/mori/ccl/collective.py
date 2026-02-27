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

import numpy as np
import torch
from mori import cpp as mori_cpp
from mori import shmem as mori_shme
from typing import Optional


def _cpp_all2all_factory(entity_name: str):
    """Factory function to get C++ entities from mori_cpp module"""
    return getattr(mori_cpp, entity_name)


class All2allSdma:
    """Python wrapper for All2allSdma C++ class"""

    def __init__(self, my_pe: int, npes: int, 
                 input_buffer_size: Optional[int] = None,
                 output_buffer_size: Optional[int] = None,
                 transit_buffer_size: Optional[int] = None,
                 copy_output_to_user: bool = True):
        """Initialize All2allSdma
        
        Args:
            my_pe: Current PE ID
            npes: Total number of PEs
            input_buffer_size: Input transit buffer size in bytes
            output_buffer_size: Output transit buffer size in bytes
            transit_buffer_size: Transit buffer size in bytes (split equally for input and output)
            copy_output_to_user: If True, copy output_transit_buffer to user output buffer (default True).
                                If False, user should directly use output_transit_buffer via get_output_transit_buffer()
        """
        self.my_pe = my_pe
        self.npes = npes
        handle_class = _cpp_all2all_factory("All2allSdmaHandle")
        
        if input_buffer_size is not None and output_buffer_size is not None:
            self._handle = handle_class(my_pe, npes, input_buffer_size, output_buffer_size, copy_output_to_user)
        elif transit_buffer_size is not None:
            self._handle = handle_class(my_pe, npes, transit_buffer_size, copy_output_to_user)
        else:
            self._handle = handle_class(my_pe, npes, 512 * 1024 * 1024, copy_output_to_user)

    def __call__(self, input_data, output_data, count: int, stream=None) -> float:
        """Execute All2All SDMA operation.
        
        Args:
            input_data: Input CUDA tensor (torch.int32, 1D, GPU memory)
            output_data: Output CUDA tensor (torch.int32, 1D, GPU memory)
            count: Number of elements per PE
            stream: Optional HIP stream
            
        Returns:
            Execution time in seconds
        """
        return self._handle(input_data, output_data, count, stream)

    def start_async(self, input_data, output_data, count: int, stream=None) -> bool:
        """Start asynchronous All2All SDMA operation (PUT phase).
        
        Args:
            input_data: Input CUDA tensor (torch.int32, 1D, GPU memory)
            output_data: Output CUDA tensor (torch.int32, 1D, GPU memory)
            count: Number of elements per PE
            stream: Optional HIP stream
            
        Returns:
            True if operation started successfully, False otherwise
        """
        return self._handle.start_async(input_data, output_data, count, stream)

    def wait_async(self, stream=None) -> float:
        """Wait for asynchronous All2All SDMA operation to complete (WAIT phase).
        
        Args:
            stream: Optional HIP stream
            
        Returns:
            Execution time in seconds, -1.0 if failed
        """
        return self._handle.wait_async(stream)

    def is_async_in_progress(self) -> bool:
        """Check if async operation is in progress.
        
        Returns:
            True if async operation is active
        """
        return self._handle.is_async_in_progress()

    def cancel_async(self):
        """Cancel ongoing async operation"""
        self._handle.cancel_async()

    def reset_flags(self):
        """Reset synchronization flags"""
        self._handle.reset_flags()

    def get_output_transit_buffer(self, device=None):
        """Get output transit buffer as a PyTorch tensor.
        
        Args:
            device: Optional device specification. Can be:
                - An int: device index (e.g., 0, 1)
                - A CUDA tensor: uses the device of that tensor
                - None: uses the current CUDA device
        
        Returns:
            torch.Tensor: Output transit buffer as a CUDA tensor (uint32, 1D)
            
        Note:
            The tensor is a view of the internal buffer. Do not modify the buffer
            while an async operation is in progress.
        """
        return self._handle.get_output_transit_buffer(device)

def _cpp_allgather_factory(entity_name: str):
    """Factory function to get C++ entities from mori_cpp module"""
    return getattr(mori_cpp, entity_name)


class AllgatherSdma:
    """Python wrapper for AllgatherSdma C++ class"""

    def __init__(self, my_pe: int, npes: int, 
                 input_buffer_size: Optional[int] = None,
                 output_buffer_size: Optional[int] = None,
                 transit_buffer_size: Optional[int] = None,
                 copy_output_to_user: bool = True):
        """Initialize AllgatherSdma
        
        Args:
            my_pe: Current PE ID
            npes: Total number of PEs
            input_buffer_size: Input transit buffer size in bytes
            output_buffer_size: Output transit buffer size in bytes
            transit_buffer_size: Transit buffer size in bytes (split equally for input and output)
            copy_output_to_user: If True, copy output_transit_buffer to user output buffer (default True).
                                If False, user should directly use output_transit_buffer via get_output_transit_buffer()
        """
        self.my_pe = my_pe
        self.npes = npes
        handle_class = _cpp_allgather_factory("AllgatherSdmaHandle")
        
        if input_buffer_size is not None and output_buffer_size is not None:
            self._handle = handle_class(my_pe, npes, input_buffer_size, output_buffer_size, copy_output_to_user)
        elif transit_buffer_size is not None:
            self._handle = handle_class(my_pe, npes, transit_buffer_size, copy_output_to_user)
        else:
            self._handle = handle_class(my_pe, npes, 512 * 1024 * 1024, copy_output_to_user)

    def __call__(self, input_data, output_data, count: int, stream=None) -> bool:
        """Execute AllGATHER SDMA operation.
        
        Args:
            input_data: Input CUDA tensor (torch.int32, 1D, GPU memory)
            output_data: Output CUDA tensor (torch.int32, 1D, GPU memory)
            count: Number of elements per PE
            stream: Optional HIP stream
            
        Returns:
            True if successful, False if failed
            
        Note:
            Caller must handle synchronization (stream.synchronize() or torch.cuda.synchronize())
        """
        return self._handle(input_data, output_data, count, stream)

    def start_async(self, input_data, output_data, count: int, stream=None) -> bool:
        """Start asynchronous AllGATHER SDMA operation (PUT phase).
        
        Args:
            input_data: Input CUDA tensor (torch.int32, 1D, GPU memory)
            output_data: Output CUDA tensor (torch.int32, 1D, GPU memory)
            count: Number of elements per PE
            stream: Optional HIP stream
            
        Returns:
            True if operation started successfully, False otherwise
        """
        return self._handle.start_async(input_data, output_data, count, stream)

    def wait_async(self, stream=None) -> float:
        """Wait for asynchronous AllGATHER SDMA operation to complete (WAIT phase).
        
        Args:
            stream: Optional HIP stream
            
        Returns:
            Execution time in seconds, -1.0 if failed
        """
        return self._handle.wait_async(stream)

    def is_async_in_progress(self) -> bool:
        """Check if async operation is in progress.
        
        Returns:
            True if async operation is active
        """
        return self._handle.is_async_in_progress()

    def cancel_async(self):
        """Cancel ongoing async operation"""
        self._handle.cancel_async()

    def reset_flags(self):
        """Reset synchronization flags"""
        self._handle.reset_flags()

    def get_output_transit_buffer(self, device=None):
        """Get output transit buffer as a PyTorch tensor.
        
        Args:
            device: Optional device specification. Can be:
                - An int: device index (e.g., 0, 1)
                - A CUDA tensor: uses the device of that tensor
                - None: uses the current CUDA device
        
        Returns:
            torch.Tensor: Output transit buffer as a CUDA tensor (uint32, 1D)
            
        Note:
            The tensor is a view of the internal buffer. Do not modify the buffer
            while an async operation is in progress.
        """
        return self._handle.get_output_transit_buffer(device)


def _cpp_allreduce_factory(entity_name: str):
    """Factory function to get C++ entities from mori_cpp module"""
    return getattr(mori_cpp, entity_name)


class AllreduceSdma:
    """Python wrapper for AllreduceSdma C++ class.
    
    Performs AllReduce in two stages:
      Stage 1: ReduceScatter — each rank reduces its shard across all peers
      Stage 2: AllGather — broadcast reduced shards to all peers via SDMA
    
    After the operation, every rank holds the same reduced result (elementwise sum
    of all ranks' inputs) in the output buffer.
    
    Supported dtypes: torch.uint32, torch.int32, torch.float16, torch.bfloat16
    
    Modes:
      - "eager": copies user input to a pre-registered transit buffer each call.
      - "graph": registers the user input pointer directly (cached on first call).
                 Skips copy_input_to_transit and copy_output_to_user.
                 User should pre-allocate a fixed-address input tensor and
                 read results from get_output_transit_buffer().
    """

    _HANDLE_MAP = {
        torch.uint32: "AllreduceSdmaHandle",
        torch.int32: "AllreduceSdmaHandle",
        torch.float32: "AllreduceSdmaHandle",
        torch.float16: "AllreduceSdmaHandleFp16",
        torch.bfloat16: "AllreduceSdmaHandleBf16",
    }

    def __init__(self, my_pe: int, npes: int,
                 input_buffer_size: Optional[int] = None,
                 output_buffer_size: Optional[int] = None,
                 transit_buffer_size: Optional[int] = None,
                 copy_output_to_user: bool = True,
                 dtype: torch.dtype = torch.uint32,
                 mode: str = "eager"):
        """Initialize AllreduceSdma
        
        Args:
            my_pe: Current PE ID
            npes: Total number of PEs
            input_buffer_size: Input transit buffer size in bytes
            output_buffer_size: Output transit buffer size in bytes (must hold npes * padded_shard_size elements)
            transit_buffer_size: Transit buffer size in bytes (split equally for input and output)
            copy_output_to_user: If True, copy output_transit_buffer to user output buffer (default True).
                                If False, user should directly use output_transit_buffer via get_output_transit_buffer()
            dtype: Data type for the allreduce operation (default torch.uint32).
                   Supported: torch.uint32, torch.int32, torch.float16, torch.bfloat16
            mode: "eager" (default) or "graph". Graph mode registers user input directly
                  and always skips copy_output_to_user for higher performance.
        """
        self.my_pe = my_pe
        self.npes = npes
        self.dtype = dtype
        self.mode = mode

        if mode not in ("eager", "graph"):
            raise ValueError(f"mode must be 'eager' or 'graph', got '{mode}'")
        use_graph_mode = (mode == "graph")

        handle_name = self._HANDLE_MAP.get(dtype)
        if handle_name is None:
            raise ValueError(
                f"Unsupported dtype {dtype}. Supported: {list(self._HANDLE_MAP.keys())}"
            )
        handle_class = _cpp_allreduce_factory(handle_name)

        if input_buffer_size is not None and output_buffer_size is not None:
            self._handle = handle_class(my_pe, npes, input_buffer_size, output_buffer_size,
                                        copy_output_to_user, use_graph_mode)
        elif transit_buffer_size is not None:
            self._handle = handle_class(my_pe, npes, transit_buffer_size,
                                        copy_output_to_user, use_graph_mode)
        else:
            self._handle = handle_class(my_pe, npes, 512 * 1024 * 1024,
                                        copy_output_to_user, use_graph_mode)

    def __call__(self, input_data, output_data, count: int, stream=None) -> bool:
        """Execute out-of-place AllReduce SDMA operation.
        
        Args:
            input_data: Input CUDA tensor (torch.int32 or torch.uint32, 1D, GPU memory).
                       Contains `count` elements on this rank.
            output_data: Output CUDA tensor (torch.int32 or torch.uint32, 1D, GPU memory).
                        Will contain `count` elements — the element-wise sum across all ranks.
            count: Number of elements per PE
            stream: Optional HIP stream
            
        Returns:
            True if successful, False if failed
            
        Note:
            Caller must handle synchronization (stream.synchronize() or torch.cuda.synchronize())
        """
        return self._handle(input_data, output_data, count, stream)

    def allreduce_inplace(self, data, count: int, stream=None) -> bool:
        """Execute in-place AllReduce SDMA operation (result overwrites input).
        
        Args:
            data: Input/output CUDA tensor (torch.int32 or torch.uint32, 1D, GPU memory).
                  Contains `count` elements on this rank. After the operation, holds the
                  element-wise sum across all ranks.
            count: Number of elements per PE
            stream: Optional HIP stream
            
        Returns:
            True if successful, False if failed
            
        Note:
            Caller must handle synchronization (stream.synchronize() or torch.cuda.synchronize())
        """
        return self._handle.allreduce_inplace(data, count, stream)

    def reset_flags(self):
        """Reset synchronization flags"""
        self._handle.reset_flags()

    def get_output_transit_buffer(self, device=None):
        """Get output transit buffer as a PyTorch tensor.
        
        Args:
            device: Optional device specification. Can be:
                - An int: device index (e.g., 0, 1)
                - A CUDA tensor: uses the device of that tensor
                - None: uses the current CUDA device
        
        Returns:
            torch.Tensor: Output transit buffer as a CUDA tensor (uint32, 1D)
            
        Note:
            The tensor is a view of the internal buffer. Do not modify the buffer
            while an operation is in progress.
        """
        return self._handle.get_output_transit_buffer(device)
