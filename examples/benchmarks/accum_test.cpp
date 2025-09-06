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
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cassert>

#include "mori/application/utils/check.hpp"
#include "mori/core/transport/p2p/device_primitives.hpp"

template <typename T, int VecBytes>
__global__ void LocalAccumKernel(size_t size, T* __restrict__ src, T* __restrict__ dest) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int laneId = tid % 64;
  int thdNum = gridDim.x * blockDim.x;
  constexpr int VecSize = VecBytes / sizeof(T);
  using VecDType = mori::core::VecTypeSelector<VecBytes>::dataType;

  __shared__ T* __restrict__ srcPtrs[8];
  if (laneId < 8) {
    srcPtrs[laneId] = src + size * laneId;
  }
  __syncthreads();

  float accumVec[VecSize];

  for (int i = tid * VecSize; i < size; i += thdNum * VecSize) {
    VecDType srcVal[8];
    for (int j = 0; j < 8; j++) {
      srcVal[j] = *reinterpret_cast<VecDType*>(&srcPtrs[j][i]);
    }

    for (int j = 0; j < 8; j++) {
      for (int k = 0; k < VecSize; k++) {
        accumVec[k] += float(reinterpret_cast<const T*>(&srcVal[j])[k]);
      }
    }
    VecDType dstVal;
    for (int k = 0; k < VecSize; k++) {
      reinterpret_cast<T*>(&dstVal)[k] = T(accumVec[k]);
    }
    *reinterpret_cast<VecDType*>(&dest[i]) = dstVal;
  }
}

using DataType = hip_bfloat16;

int main() {
  constexpr int VecBytes = 16;
  int blockNum = 4;
  int warpNum = 4;
  size_t accumNum = 8;
  size_t size = blockNum * warpNum * warpSize * (VecBytes / sizeof(DataType));
  DataType* src;
  DataType* dest;
  HIP_RUNTIME_CHECK(hipExtMallocWithFlags((void**)&src, size * 8, hipDeviceMallocUncached));
  HIP_RUNTIME_CHECK(hipExtMallocWithFlags((void**)&dest, size, hipDeviceMallocUncached));

  LocalAccumKernel<DataType, VecBytes><<<blockNum, warpNum * warpSize>>>(size, src, dest);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
}
