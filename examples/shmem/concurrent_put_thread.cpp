#include <mpi.h>

#include <cassert>

#include "mori/application/utils/hip_check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

constexpr ProviderType PrvdType = ProviderType::MLX5;

__global__ void ConcurrentPutThreadKernel(int myPe, const SymmMemObjPtr memObj) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadOffset = globalTid * sizeof(uint32_t);

  if (myPe == sendPe) {
    MemoryRegion source = memObj->GetMemoryRegion(myPe);

    ShmemPutMemNbiThread<PrvdType>(memObj, threadOffset, source, threadOffset, sizeof(uint32_t),
                                   recvPe);
    __threadfence_system();

    if (globalTid == 0) ShmemQuietThread<PrvdType>();
    // __syncthreads();
  } else {
    while (atomicAdd(reinterpret_cast<uint32_t*>(memObj->localPtr) + globalTid, 0) != sendPe) {
    }
  }
}

void ConcurrentPutThread() {
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
  ConcurrentPutThreadKernel<<<blockNum, threadNum>>>(myPe, buffObj);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  // Finalize
  ShmemFree(buff);
  ShmemMpiFinalize();
  MPI_Finalize();
}

int main(int argc, char* argv[]) {
  ConcurrentPutThread();
  return 0;
}