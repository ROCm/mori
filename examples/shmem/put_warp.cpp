#include <mpi.h>

#include <cassert>

#include "mori/application/utils/hip_check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

constexpr ProviderType PrvdType = ProviderType::MLX5;

__global__ void RingShmemPutMemNbiThread(int myPe, int npes, SymmMemObj* memObj) {
  int nextPeer = (myPe + 1) % npes;
  int peChunkSize = memObj->size / npes;

  MemoryRegion source;
  source.addr = reinterpret_cast<uintptr_t>(memObj->localPtr);
  source.lkey = memObj->lkey;

  for (int i = 0; i < npes - 1; i++) {
    int sendOffset = ((myPe - i) % npes) * peChunkSize;
    ShmemPutMemNbiThread<PrvdType>(memObj, sendOffset, source, sendOffset, peChunkSize, nextPeer);
    ShmemQuietThread<PrvdType>();
  }
}

void PutWarpExample() {
  int status;
  MPI_Init(NULL, NULL);

  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  // Assume in same node
  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  // Alloc memory
  int buffSize = 4096;
  assert((buffSize % npes) == 0);
  int peChunkSize = buffSize / npes;

  void* buff = ShmemMalloc(buffSize);
  HIP_RUNTIME_CHECK(
      hipMemset(reinterpret_cast<char*>(buff) + myPe * peChunkSize, myPe + 1, peChunkSize));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr memObjPtr = ShmemQueryMemObjPtr(buff);
  assert(memObjPtr.IsValid());

  for (int i = 0; i < npes; i++) {
    printf("Before rank %d, got %d on %dth chunk\n", myPe,
           reinterpret_cast<char*>(buff)[i * peChunkSize], i);
  }
  // Run put
  RingShmemPutMemNbiThread<<<1, 1>>>(myPe, npes, memObjPtr.gpu);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  for (int i = 0; i < npes; i++) {
    printf("After rank %d, got %d on %dth chunk\n", myPe,
           reinterpret_cast<char*>(buff)[i * peChunkSize], i);
    // for (int j = i * peChunkSize; j < ((i + 1) * peChunkSize); j++) {
    //   assert(reinterpret_cast<char*>(buff)[j] == i + 1);
    // }
  }

  // Finalize
  ShmemFree(buff);
  ShmemMpiFinalize();
  MPI_Finalize();
}

int main(int argc, char* argv[]) { PutWarpExample(); }