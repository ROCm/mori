#include <mpi.h>

#include <cassert>

#include "mori/application/utils/hip_check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

__global__ void P2PBandwidthTestKernel(SymmMemObjPtr memObj, size_t chunkSize, size_t chunkNum) {
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalThdNum = gridDim.x * blockDim.x;
  int globalWarpId = globalThdId / warpSize;
  int globalWarpNum = gridDim.x * blockDim.x / warpSize;

  uint8_t* src = memObj->GetAs<uint8_t*>();
  uint8_t* dest = memObj->GetAs<uint8_t*>(1);

#pragma unroll
  for (int i = globalWarpId; i < chunkNum; i += globalWarpNum) {
    WarpCopy(dest + i * chunkSize, src + i * chunkSize, chunkSize);
  }

  // #pragma unroll
  //   for (int i = globalThdId * 16; i < chunkNum * chunkSize; i += globalThdNum * 16) {
  //     reinterpret_cast<uint4*>(dest + i)[0] = reinterpret_cast<uint4*>(src + i)[0];
  //   }
}

void P2PBandwidthTest() {
  int status;
  MPI_Init(NULL, NULL);

  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  // Assume in same node
  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  // Allocate buffer
  int chunkSize = 7168;
  int chunkNum = 128;
  int warpNum = 8;
  int blockNum = 256;

  for (int i = 0; i < 10; i++) {
    void* buff = ShmemExtMallocWithFlags(chunkSize * chunkNum, hipDeviceMallocUncached);
    SymmMemObjPtr buffObj = ShmemQueryMemObjPtr(buff);
    assert(buffObj.IsValid());

    if (myPe == 0) {
      // HIP_RUNTIME_CHECK(hipMemset(buff, 9, chunkSize * chunkNum));
      HIP_RUNTIME_CHECK(hipDeviceSynchronize());
      P2PBandwidthTestKernel<<<blockNum, warpNum * warpSize>>>(buffObj, chunkSize, chunkNum);
    }
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // if (myPe == 1) {
    //   for (int i = 0; i < chunkSize * chunkNum; i++)
    //     if (reinterpret_cast<uint8_t*>(buff)[i] != 9) {
    //       printf("pos %d val %d\n", i, reinterpret_cast<uint8_t*>(buff)[i]);
    //       assert(reinterpret_cast<uint8_t*>(buff)[i] == 9);
    //     }
    // }

    ShmemFree(buff);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Finalize
  ShmemMpiFinalize();
}

int main(int argc, char* argv[]) {
  P2PBandwidthTest();
  return 0;
}