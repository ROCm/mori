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
#include <mpi.h>

#include <cassert>

#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

using T = hip_bfloat16;

__global__ void AccumPerfKernel(int myPe, int npes, const SymmMemObjPtr src,
                                const SymmMemObjPtr dest, int elementNum, int elementPerWarp) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;
  int globalWarpId = blockIdx.x * warpNum + warpId;

  __shared__ T* sharedMem[8 * 16];
  T** srcPtrs = sharedMem + warpId * npes;

  if (laneId < npes) {
    srcPtrs[laneId] = src->template GetAs<T*>(laneId) + globalWarpId * elementPerWarp;
  }

  mori::core::WarpAccum<T, 8>(dest->template GetAs<T*>() + globalWarpId * elementPerWarp, srcPtrs,
                              nullptr, npes,
                              std::min(elementPerWarp, elementNum - globalWarpId * elementPerWarp));
}

void AccumPerf() {
  int status;
  MPI_Init(NULL, NULL);
  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  size_t elementSize = sizeof(T);
  // size_t elementNum = 16 * 1000 * 1024;
  size_t elementNum = 4096 * 7168;
  size_t bufferSize = elementNum * elementSize;

  //   void* srcBuff = ShmemExtMallocWithFlags(bufferSize, hipDeviceMallocUncached);
  void* srcBuff = ShmemMalloc(bufferSize);
  HIP_RUNTIME_CHECK(hipMemset(reinterpret_cast<uint32_t*>(srcBuff), 0, bufferSize));
  SymmMemObjPtr srcBuffObj = ShmemQueryMemObjPtr(srcBuff);
  assert(srcBuffObj.IsValid());

  void* destBuff = ShmemExtMallocWithFlags(bufferSize, hipDeviceMallocUncached);
  HIP_RUNTIME_CHECK(hipMemset(reinterpret_cast<uint32_t*>(destBuff), 0, bufferSize));
  SymmMemObjPtr destBuffObj = ShmemQueryMemObjPtr(destBuff);
  assert(destBuffObj.IsValid());

  int blockNum = 80;
  int warpNum = 8;
  int threadNum = warpSize * warpNum;
  int totalWarpNum = blockNum * warpNum;

  size_t elementPerWarp = (elementNum + totalWarpNum - 1) / totalWarpNum;

  printf("elementPerWarp %zu\n", elementPerWarp);

  for (int i = 0; i < 3; i++)
    AccumPerfKernel<<<blockNum, threadNum>>>(myPe, npes, srcBuffObj, destBuffObj, elementNum,
                                             elementPerWarp);

  const int iters = 5;
  hipEvent_t start, stop;
  HIP_RUNTIME_CHECK(hipEventCreate(&start));
  HIP_RUNTIME_CHECK(hipEventCreate(&stop));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  HIP_RUNTIME_CHECK(hipEventRecord(start, 0));
  for (int i = 0; i < iters; i++) {
    AccumPerfKernel<<<blockNum, threadNum>>>(myPe, npes, srcBuffObj, destBuffObj, elementNum,
                                             elementPerWarp);
  }
  HIP_RUNTIME_CHECK(hipEventRecord(stop, 0));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  float elapsedTime;
  HIP_RUNTIME_CHECK(hipEventElapsedTime(&elapsedTime, start, stop));
  printf("rank %d time %f avgtime %f bw %f\n", myPe, elapsedTime, elapsedTime / iters,
         (bufferSize / 1.0E9) * npes / (elapsedTime / 1000.0f) / iters);
  MPI_Barrier(MPI_COMM_WORLD);

  ShmemFinalize();
}

int main() { AccumPerf(); }
