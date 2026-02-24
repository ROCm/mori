// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>

namespace mori {
namespace collective {

// Local reduce kernel: sum npes chunks element-wise in the gathered buffer.
// Layout: [PE0_data(elementCount) | PE1_data(elementCount) | ... | PE(npes-1)_data]
// Result: gathered[0 : elementCount] = sum of all chunks
template <typename T>
__global__ void AllReduceLocalSumKernel(T* gathered, size_t elementCount, int npes) {
  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t threadsPerGrid =
      static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);

  for (size_t i = threadLinearId; i < elementCount; i += threadsPerGrid) {
    T sum = gathered[i];
    for (int pe = 1; pe < npes; pe++) {
      sum += gathered[static_cast<size_t>(pe) * elementCount + i];
    }
    gathered[i] = sum;
  }
}

}  // namespace collective
}  // namespace mori
