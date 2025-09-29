// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <mpi.h>

#include <cassert>

#include "mori/application/utils/check.hpp"
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
  memsetD64Kernel<<<gridSize, blockSize>>>(reinterpret_cast<unsigned long long*>(dst), value,
                                           count);
}

template <typename T>
__global__ void AtomicNonFetchThreadKernel(int myPe, const SymmMemObjPtr memObj) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = globalTid / warpSize;
  int threadOffset = globalTid * sizeof(T);

  if (myPe == sendPe) {
    RdmaMemoryRegion source = memObj->GetRdmaMemoryRegion(sendPe);

    // ShmemAtomicUint64NonFetchThread(memObj, threadOffset, sendPe, AMO_SET, recvPe);
    if (globalWarpId % 2 == 0) {
      ShmemAtomicTypeNonFetchThread<T>(memObj, 2 * sizeof(T), 1, AMO_ADD, recvPe);
    } else {
      ShmemAtomicTypeNonFetchThread<T>(memObj, 2 * sizeof(T), 1, AMO_ADD, recvPe, 1);
    }
    __threadfence_system();

    ShmemQuietThread();
    // __syncthreads();
  } else {
    while (AtomicLoadRelaxed(reinterpret_cast<T*>(memObj->localPtr) + 2) !=
           gridDim.x * blockDim.x + 1) {
    }
    if (globalTid == 0) {
      printf("atomic nonfetch is ok!~\n");
    }
  }
}

void testAtomicNonFetchThread() {
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

  void* buff = ShmemExtMallocWithFlags(buffSize, hipDeviceMallocUncached);
  myHipMemsetD64(buff, myPe, numEle);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  printf("before rank[%d] %lu %lu\n", myPe, *(reinterpret_cast<uint64_t*>(buff)),
         *(reinterpret_cast<uint64_t*>(buff)));
  SymmMemObjPtr buffObj = ShmemQueryMemObjPtr(buff);
  assert(buffObj.IsValid());

  // Run atomic operations 10 times
  for (int iteration = 0; iteration < 10; iteration++) {
    printf("========== Iteration %d ==========\n", iteration + 1);

    // Run uint64 atomic nonfetch
    myHipMemsetD64(buff, myPe, numEle);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    printf("before rank[%d] uint64: %lu %lu\n", myPe, *(reinterpret_cast<uint64_t*>(buff)),
           *(reinterpret_cast<uint64_t*>(buff)));
    AtomicNonFetchThreadKernel<uint64_t><<<blockNum, threadNum>>>(myPe, buffObj);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    printf("after rank[%d] uint64: %lu %lu\n", myPe, *(reinterpret_cast<uint64_t*>(buff)),
           *(reinterpret_cast<uint64_t*>(buff) + 2));

    // Test int64_t atomic nonfetch
    buffSize = numEle * sizeof(int64_t);
    myHipMemsetD64(buff, myPe, numEle);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    printf("before rank[%d] int64: %ld %ld\n", myPe, *(reinterpret_cast<int64_t*>(buff)),
           *(reinterpret_cast<int64_t*>(buff)));
    // Run int64 atomic nonfetch
    AtomicNonFetchThreadKernel<int64_t><<<blockNum, threadNum>>>(myPe, buffObj);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    printf("after rank[%d] int64: %ld %ld\n", myPe, *(reinterpret_cast<int64_t*>(buff)),
           *(reinterpret_cast<int64_t*>(buff) + 2));

    // Test uint32_t atomic nonfetch
    buffSize = numEle * sizeof(uint32_t);
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buff), myPe, numEle));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    printf("before rank[%d] uint32: %u %u\n", myPe, *(reinterpret_cast<uint32_t*>(buff)),
           *(reinterpret_cast<uint32_t*>(buff)));
    // Run uint32 atomic nonfetch
    AtomicNonFetchThreadKernel<uint32_t><<<blockNum, threadNum>>>(myPe, buffObj);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    printf("after rank[%d] uint32: %u %u\n", myPe, *(reinterpret_cast<uint32_t*>(buff)),
           *(reinterpret_cast<uint32_t*>(buff) + 2));

    // Test int32_t atomic nonfetch
    buffSize = numEle * sizeof(int32_t);
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<int32_t*>(buff), myPe, numEle));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    printf("before rank[%d] int32: %d %d\n", myPe, *(reinterpret_cast<int32_t*>(buff)),
           *(reinterpret_cast<int32_t*>(buff)));
    // Run int32 atomic nonfetch
    AtomicNonFetchThreadKernel<int32_t><<<blockNum, threadNum>>>(myPe, buffObj);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    printf("after rank[%d] int32: %d %d\n", myPe, *(reinterpret_cast<int32_t*>(buff)),
           *(reinterpret_cast<int32_t*>(buff) + 2));

    printf("Iteration %d completed\n", iteration + 1);
    MPI_Barrier(MPI_COMM_WORLD);  // Ensure all processes complete this iteration before next
  }

  // Finalize
  ShmemFree(buff);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  testAtomicNonFetchThread();
  return 0;
}
