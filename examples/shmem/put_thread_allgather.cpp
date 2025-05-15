#include <mpi.h>

#include <cassert>

#include "mori/application/utils/hip_check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

constexpr ProviderType PrvdType = ProviderType::MLX5;

__global__ void RingAllGatherWithPutMemAPIKernel(int myPe, int npes, const SymmMemObjPtr memObj) {
  int nextPeer = (myPe + 1) % npes;
  int peChunkSize = memObj->size / npes;

  MemoryRegion source;
  source.addr = reinterpret_cast<uintptr_t>(memObj->localPtr);
  source.lkey = memObj->lkey;

  for (int i = 0; i < npes - 1; i++) {
    int sendDataRank = ((myPe - i) + npes) % npes;
    int sendOffset = sendDataRank * peChunkSize;
    ShmemPutMemNbiThread<PrvdType>(memObj, sendOffset, source, sendOffset, peChunkSize, nextPeer);
    ShmemQuietThread<PrvdType>();

    int recvDataRank = ((sendDataRank - 1) + npes) % npes;
    int recvOffset = recvDataRank * peChunkSize;
    void* recvAddr = reinterpret_cast<char*>(memObj->localPtr) + recvOffset;

    // Wait until received
    printf("rank %d round %d recv rank %d sendoff %d recvoff %d\n", myPe, i, recvDataRank,
           sendOffset, recvOffset);

    while ((atomicAdd(reinterpret_cast<uint32_t*>(recvAddr), 0)) != (recvDataRank + 1)) {
    }
  }
}

void RingAllGatherWithPutMemAPI() {
  int status;
  MPI_Init(NULL, NULL);

  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  // Assume in same node
  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  // Allocate buffer
  int buffSize = npes * 1024 * sizeof(uint32_t);
  int peChunkSize = buffSize / npes / sizeof(uint32_t);

  void* buff = ShmemMalloc(buffSize);
  HIP_RUNTIME_CHECK(
      hipMemsetD32(reinterpret_cast<uint32_t*>(buff) + myPe * peChunkSize, myPe + 1, peChunkSize));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr buffObj = ShmemQueryMemObjPtr(buff);
  assert(buffObj.IsValid());

  for (int i = 0; i < npes; i++) {
    printf("Before rank %d, got %d on %dth chunk\n", myPe,
           reinterpret_cast<uint32_t*>(buff)[i * peChunkSize], i);
  }
  // Run put
  RingAllGatherWithPutMemAPIKernel<<<1, 1>>>(myPe, npes, buffObj);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < npes; i++) {
    printf("After rank %d, got %d on %dth chunk\n", myPe,
           reinterpret_cast<uint32_t*>(buff)[i * peChunkSize], i);
    for (int j = i * peChunkSize; j < ((i + 1) * peChunkSize); j++) {
      assert(reinterpret_cast<uint32_t*>(buff)[j] == i + 1);
    }
  }

  // Finalize
  ShmemFree(buff);
  ShmemMpiFinalize();
  MPI_Finalize();
}

int main(int argc, char* argv[]) {
  RingAllGatherWithPutMemAPI();
  return 0;
}