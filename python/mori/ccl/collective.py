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
                 transit_buffer_size: Optional[int] = None):
        """Initialize All2allSdma"""
        self.my_pe = my_pe
        self.npes = npes
        handle_class = _cpp_all2all_factory("All2allSdmaHandle")
        
        if input_buffer_size is not None and output_buffer_size is not None:
            self._handle = handle_class(my_pe, npes, input_buffer_size, output_buffer_size)
        elif transit_buffer_size is not None:
            self._handle = handle_class(my_pe, npes, transit_buffer_size)
        else:
            self._handle = handle_class(my_pe, npes, 512 * 1024 * 1024)

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

    def reset_flags(self):
        """Reset synchronization flags"""
        self._handle.reset_flags()
