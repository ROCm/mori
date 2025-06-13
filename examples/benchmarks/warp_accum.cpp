#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>
#include <mpi.h>

#include <cassert>

#include "mori/application/utils/hip_check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

using T = hip_bfloat16;
// using T = __hip_fp8_e4m3_fnuz;

__global__ void WarpAccumBenchKernel(T* dest, SymmMemObjPtr memObj, size_t chunkSize,
                                     size_t chunkNum, int worldSize) {
  int laneId = threadIdx.x & (warpSize - 1);

  int warpId = threadIdx.x / warpSize;
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalThdNum = gridDim.x * blockDim.x;

  int globalWarpId = globalThdId / warpSize;
  int globalWarpNum = gridDim.x * blockDim.x / warpSize;

  __shared__ T* shmem[4 * 8];
  T** srcPtrs = &shmem[warpId * 8];

#pragma unroll
  for (int i = globalWarpId; i < chunkNum; i += globalWarpNum) {
    for (int j = laneId; j < worldSize; j += warpSize)
      srcPtrs[j] = memObj->GetAs<T*>(j) + i * chunkSize;

    WarpAccum(dest + i * chunkSize, srcPtrs, nullptr, worldSize, chunkSize);
  }
}

void WarpAccumTest() {
  int status;
  MPI_Init(NULL, NULL);

  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  // Assume in same node
  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  // Allocate buffer
  int chunkSize = 7168;
  int chunkNum = 512;
  int warpNum = 8;
  int blockNum = 160;

  for (int i = 0; i < 10; i++) {
    void* buff = ShmemExtMallocWithFlags(chunkSize * chunkNum * sizeof(T), hipDeviceMallocUncached);
    SymmMemObjPtr buffObj = ShmemQueryMemObjPtr(buff);
    assert(buffObj.IsValid());

    void* dest;
    HIP_RUNTIME_CHECK(hipMalloc(&dest, chunkSize * chunkNum * sizeof(T)));

    WarpAccumBenchKernel<<<blockNum, warpNum * warpSize>>>(reinterpret_cast<T*>(dest), buffObj,
                                                           chunkSize, chunkNum, npes);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    ShmemFree(buff);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Finalize
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  WarpAccumTest();
  return 0;
}