#include <mpi.h>

#include <cassert>

#include "mori/application/utils/hip_check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::shmem;

__global__ void TestShmemPutMemNbiThread() { printf("%p\n", GetEpsStartAddr()); }

void PutWarpExample() {
  int status;
  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  // Assume in same node
  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  // Alloc memory
  int buffSize = 2048;
  void* buff = ShmemMalloc(buffSize);

  // Run put
  TestShmemPutMemNbiThread<<<1, 1>>>();
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // Finalize
  ShmemFree(buff);
  MPI_Finalize();
}

int main(int argc, char* argv[]) { PutWarpExample(); }