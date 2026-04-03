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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License

#include <hip/hip_runtime.h>

#include <cstdint>

#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#define MORI_SHMEM_NO_STATIC_INIT
#include "mori/shmem/shmem.hpp"

using mori::moe::index_t;

extern "C" __global__ __attribute__((visibility("default"))) void LocalExpertCountKernel(
    const index_t* indices, const index_t* totalRecvTokenNum, int rank, int numExpertPerRank,
    int numExpertPerToken, int* localExpertCount) {
  const int expertBase = rank * numExpertPerRank;
  const int64_t totalAssignments =
      static_cast<int64_t>(totalRecvTokenNum[0]) * static_cast<int64_t>(numExpertPerToken);
  const int64_t globalThreadId = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t globalThreadNum = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t i = globalThreadId; i < totalAssignments; i += globalThreadNum) {
    const index_t expertId = indices[i];
    const int localExpert = static_cast<int>(expertId) - expertBase;
    if (static_cast<unsigned int>(localExpert) < static_cast<unsigned int>(numExpertPerRank)) {
      atomicAdd(localExpertCount + localExpert, 1);
    }
  }
}
