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
import ctypes

class All2allSdma: 
    def __init__(self, my_pe: int, npes: int):
        self.my_pe = my_pe
        self.npes = npes
    
    def __call__(self, 
                 input_data: np.ndarray, 
                 output_data: np.ndarray,
                 count: int) -> float:
        """        
        Args:
            input_data: 输入数据 (uint32 numpy数组)
            output_data: 输出数据 (uint32 numpy数组) 
            count: 每个PE的元素数量
            
        Returns:
            执行时间(秒)
        """
        # 直接调用C++函数
        return mori_cpp.all2all_sdma(
            self.my_pe,
            self.npes,
            input_data,
            output_data,
            count
        )
