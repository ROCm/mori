#include <hip/hip_bfloat16.h>
#include <mpi.h>

#include <cassert>

#include "mori/application/utils/hip_check.hpp"
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

  mori::core::WarpAccum<T, 16>(
      dest->template GetAs<T*>() + globalWarpId * elementPerWarp, srcPtrs, nullptr, npes,
      std::min(elementPerWarp, elementNum - globalWarpId * elementPerWarp));
}

__global__ void AccumBaselineKernel(int myPe, int npes, const SymmMemObjPtr src,
                                    const SymmMemObjPtr dest, int elementNum, int elementPerWarp) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;
  int globalThdNum = gridDim.x * warpNum * warpSize;

  __shared__ T* sharedMem[8 * 16];
  T** srcPtrs = sharedMem + warpId * npes;

  if (laneId < npes) {
    srcPtrs[laneId] = src->template GetAs<T*>(laneId);
  }

  constexpr int vecSize = 16 / sizeof(T);
  for (int i = globalThdId * vecSize; i < elementNum; i += globalThdNum * vecSize) {
    float accumValFp32[vecSize] = {0};

    for (int j = 0; j < npes; j++) {
      T* srcPtr = srcPtrs[j];
      float srcScale = 1.0f;

      uint4 srcVals = reinterpret_cast<uint4*>(srcPtr + i)[0];

      if constexpr (vecSize > 0)
        accumValFp32[0] += float(reinterpret_cast<T*>(&srcVals)[0]) * srcScale;
      if constexpr (vecSize > 1)
        accumValFp32[1] += float(reinterpret_cast<T*>(&srcVals)[1]) * srcScale;
      if constexpr (vecSize > 2)
        accumValFp32[2] += float(reinterpret_cast<T*>(&srcVals)[2]) * srcScale;
      if constexpr (vecSize > 3)
        accumValFp32[3] += float(reinterpret_cast<T*>(&srcVals)[3]) * srcScale;
      if constexpr (vecSize > 4)
        accumValFp32[4] += float(reinterpret_cast<T*>(&srcVals)[4]) * srcScale;
      if constexpr (vecSize > 5)
        accumValFp32[5] += float(reinterpret_cast<T*>(&srcVals)[5]) * srcScale;
      if constexpr (vecSize > 6)
        accumValFp32[6] += float(reinterpret_cast<T*>(&srcVals)[6]) * srcScale;
      if constexpr (vecSize > 7)
        accumValFp32[7] += float(reinterpret_cast<T*>(&srcVals)[7]) * srcScale;
      if constexpr (vecSize > 8)
        accumValFp32[8] += float(reinterpret_cast<T*>(&srcVals)[8]) * srcScale;
      if constexpr (vecSize > 9)
        accumValFp32[9] += float(reinterpret_cast<T*>(&srcVals)[9]) * srcScale;
      if constexpr (vecSize > 10)
        accumValFp32[10] += float(reinterpret_cast<T*>(&srcVals)[10]) * srcScale;
      if constexpr (vecSize > 11)
        accumValFp32[11] += float(reinterpret_cast<T*>(&srcVals)[11]) * srcScale;
      if constexpr (vecSize > 12)
        accumValFp32[12] += float(reinterpret_cast<T*>(&srcVals)[12]) * srcScale;
      if constexpr (vecSize > 13)
        accumValFp32[13] += float(reinterpret_cast<T*>(&srcVals)[13]) * srcScale;
      if constexpr (vecSize > 14)
        accumValFp32[14] += float(reinterpret_cast<T*>(&srcVals)[14]) * srcScale;
      if constexpr (vecSize > 15)
        accumValFp32[15] += float(reinterpret_cast<T*>(&srcVals)[15]) * srcScale;
    }

    uint4 accumVals;
    if constexpr (vecSize > 0) reinterpret_cast<T*>(&accumVals)[0] = T(accumValFp32[0]);
    if constexpr (vecSize > 1) reinterpret_cast<T*>(&accumVals)[1] = T(accumValFp32[1]);
    if constexpr (vecSize > 2) reinterpret_cast<T*>(&accumVals)[2] = T(accumValFp32[2]);
    if constexpr (vecSize > 3) reinterpret_cast<T*>(&accumVals)[3] = T(accumValFp32[3]);
    if constexpr (vecSize > 4) reinterpret_cast<T*>(&accumVals)[4] = T(accumValFp32[4]);
    if constexpr (vecSize > 5) reinterpret_cast<T*>(&accumVals)[5] = T(accumValFp32[5]);
    if constexpr (vecSize > 6) reinterpret_cast<T*>(&accumVals)[6] = T(accumValFp32[6]);
    if constexpr (vecSize > 7) reinterpret_cast<T*>(&accumVals)[7] = T(accumValFp32[7]);
    if constexpr (vecSize > 8) reinterpret_cast<T*>(&accumVals)[8] = T(accumValFp32[8]);
    if constexpr (vecSize > 9) reinterpret_cast<T*>(&accumVals)[9] = T(accumValFp32[9]);
    if constexpr (vecSize > 10) reinterpret_cast<T*>(&accumVals)[10] = T(accumValFp32[10]);
    if constexpr (vecSize > 11) reinterpret_cast<T*>(&accumVals)[11] = T(accumValFp32[11]);
    if constexpr (vecSize > 12) reinterpret_cast<T*>(&accumVals)[12] = T(accumValFp32[12]);
    if constexpr (vecSize > 13) reinterpret_cast<T*>(&accumVals)[13] = T(accumValFp32[13]);
    if constexpr (vecSize > 14) reinterpret_cast<T*>(&accumVals)[14] = T(accumValFp32[14]);
    if constexpr (vecSize > 15) reinterpret_cast<T*>(&accumVals)[15] = T(accumValFp32[15]);

    reinterpret_cast<uint4*>(dest->template GetAs<T*>() + i)[0] = accumVals;
  }
}

void AccumPerf() {
  int status;
  MPI_Init(NULL, NULL);
  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  size_t elementSize = sizeof(T);
  size_t elementNum = 16 * 1000 * 1024;
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

  for (int i = 0; i < 100; i++)
    AccumPerfKernel<<<blockNum, threadNum>>>(myPe, npes, srcBuffObj, destBuffObj, elementNum,
                                             elementPerWarp);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  ShmemFinalize();
}

int main() { AccumPerf(); }