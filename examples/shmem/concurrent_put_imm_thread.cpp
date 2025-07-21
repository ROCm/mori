#include <mpi.h>

#include <cassert>

#include "mori/application/utils/hip_check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

__global__ void ConcurrentPutImmThreadKernel(int myPe, const SymmMemObjPtr memObj) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;
  uint32_t val = 42;
  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadOffset = globalTid * sizeof(uint32_t);

  if (myPe == sendPe) {
    MemoryRegion source = memObj->GetMemoryRegion(myPe);

    ShmemPutSizeImmNbiThread(memObj, threadOffset, &val, sizeof(uint32_t), recvPe);
    __threadfence_system();

    ShmemQuietThread();
    // __syncthreads();
  } else {
    while (atomicAdd(reinterpret_cast<uint32_t*>(memObj->localPtr) + globalTid, 0) != val) {
    }
  }
}

void ConcurrentPutImmThread() {
  int status;
  MPI_Init(NULL, NULL);

  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  // Assume in same node
  int myPe = ShmemMyPe();
  int npes = ShmemNPes();
  assert(npes == 2);

  constexpr int threadNum = 128;
  constexpr int blockNum = 3;

  // Allocate buffer
  int numEle = threadNum * blockNum;
  int buffSize = numEle * sizeof(uint32_t);

  void* buff = ShmemMalloc(buffSize);
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buff), myPe, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr buffObj = ShmemQueryMemObjPtr(buff);
  assert(buffObj.IsValid());

  // Run put
  ConcurrentPutImmThreadKernel<<<blockNum, threadNum>>>(myPe, buffObj);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  // Finalize
  ShmemFree(buff);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  ConcurrentPutImmThread();
  return 0;
}