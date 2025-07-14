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

  mori::core::WarpAccum<T, 16>(
      dest->template GetAs<T*>() + globalWarpId * elementPerWarp, srcPtrs, nullptr, npes,
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