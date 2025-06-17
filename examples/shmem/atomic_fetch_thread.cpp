#include <mpi.h>

#include <cassert>

#include "mori/application/utils/hip_check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

__global__ void memsetD64Kernel(unsigned long long* dst, unsigned long long value, size_t count) {  
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx < count) {  
        dst[idx] = value;  
    }  
}  
  
void myHipMemsetD64(void* dst, unsigned long long value, size_t count) {  
    const int blockSize = 256;  
    const int gridSize = (count + blockSize - 1) / blockSize;  
    memsetD64Kernel<<<gridSize, blockSize>>>(reinterpret_cast<unsigned long long*>(dst), value, count);  
}

template<typename T>
__global__ void AtomicFetchThreadKernel(int myPe, const SymmMemObjPtr memObj) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadOffset = globalTid * sizeof(T);

  if (myPe == sendPe) {
    MemoryRegion source = memObj->GetMemoryRegion(sendPe);
    
    ShmemAtomicTypeFetchThread<T>(memObj, threadOffset, source, threadOffset, sendPe, recvPe, recvPe, AMO_COMPARE_SWAP);
    __threadfence_system();

    if (globalTid == 0) ShmemQuietThread();
    // __syncthreads();
  } else {
    while (atomicAdd(reinterpret_cast<T*>(memObj->localPtr) + globalTid, 0) != sendPe) {
    }
    if (globalTid == 0){
        printf("atomic fetch is ok!~\n");
    }
  }
}

void testAtomicFetchThread() {
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
  int buffSize = numEle * sizeof(uint64_t);

  void* buff = ShmemMalloc(buffSize);
  myHipMemsetD64(buff, myPe, numEle);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  printf("before rank[%d] %lu %lu\n", myPe, *(reinterpret_cast<uint64_t*>(buff)),
         *(reinterpret_cast<uint64_t*>(buff) + numEle - 1));
  SymmMemObjPtr buffObj = ShmemQueryMemObjPtr(buff);
  assert(buffObj.IsValid());

  // Run uint64 atomic nonfetch
  AtomicFetchThreadKernel<uint64_t><<<blockNum, threadNum>>>(myPe, buffObj);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  printf("after rank[%d] %lu %lu\n", myPe, *(reinterpret_cast<uint64_t*>(buff)),
         *(reinterpret_cast<uint64_t*>(buff) + numEle - 1));

  buffSize = numEle * sizeof(uint32_t);
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buff), myPe, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  printf("before rank[%d] %u %u\n", myPe, *(reinterpret_cast<uint32_t*>(buff)),
         *(reinterpret_cast<uint32_t*>(buff) + numEle - 1));
  // Run uint32 atomic nonfetch
  AtomicFetchThreadKernel<uint32_t><<<blockNum, threadNum>>>(myPe, buffObj);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  printf("after rank[%d] %u %u\n", myPe, *(reinterpret_cast<uint32_t*>(buff)),
         *(reinterpret_cast<uint32_t*>(buff) + numEle - 1));

  // Finalize
  ShmemFree(buff);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  testAtomicFetchThread();
  return 0;
}