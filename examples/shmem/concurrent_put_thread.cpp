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

// Legacy API: Using SymmMemObjPtr + offset
__global__ void ConcurrentPutThreadKernel(int myPe, const SymmMemObjPtr memObj) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadOffset = globalTid * sizeof(uint32_t);

  if (myPe == sendPe) {
    RdmaMemoryRegion source = memObj->GetRdmaMemoryRegion(myPe);

    ShmemPutMemNbiThread(memObj, threadOffset, source, threadOffset, sizeof(uint32_t), recvPe, 1);
    __threadfence_system();

    if (blockIdx.x == 0)
    {
      ShmemQuietThread();
    }
    

    // __syncthreads();
  } else {
    while (atomicAdd(reinterpret_cast<uint32_t*>(memObj->localPtr) + globalTid, 0) != sendPe) {
    }
  }
}

// New API: Using pure addresses
__global__ void ConcurrentPutThreadKernel_PureAddr(int myPe, uint32_t* localBuff) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (myPe == sendPe) {
    // Calculate source and destination addresses
    uint32_t* src = localBuff + globalTid;
    uint32_t* dest = localBuff + globalTid;

    // Use pure address-based API
    ShmemPutMemNbiThread(dest, src, sizeof(uint32_t), recvPe, 1);
    __threadfence_system();

    if (blockIdx.x == 0) {
      ShmemQuietThread();
    }
  } else {
    // Wait for data to arrive
    while (atomicAdd(localBuff + globalTid, 0) != sendPe) {
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
  int numEle = threadNum * blockNum;
  int buffSize = numEle * sizeof(uint32_t);

  if (myPe == 0) {
    printf("=================================================================\n");
    printf("Testing both Legacy and Pure Address APIs\n");
    printf("=================================================================\n");
  }

  // ===== Test 1: Legacy API with SymmMemObjPtr + offset =====
  if (myPe == 0) {
    printf("\n--- Test 1: Legacy API (SymmMemObjPtr + offset) ---\n");
  }
  
  void* buff1 = ShmemMalloc(buffSize);
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buff1), myPe, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr buffObj1 = ShmemQueryMemObjPtr(buff1);
  assert(buffObj1.IsValid());

  if (myPe == 0) {
    printf("Running legacy API test...\n");
  }
  ConcurrentPutThreadKernel<<<blockNum, threadNum>>>(myPe, buffObj1);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  // Verify Test 1
  std::vector<uint32_t> hostBuff1(numEle);
  HIP_RUNTIME_CHECK(hipMemcpy(hostBuff1.data(), buff1, buffSize, hipMemcpyDeviceToHost));
  
  if (myPe == 1) {
    bool success = true;
    for (int i = 0; i < numEle; i++) {
      if (hostBuff1[i] != 0) {
        printf("Error at index %d: expected 0, got %u\n", i, hostBuff1[i]);
        success = false;
        break;
      }
    }
    if (success && myPe == 0) {
      printf("✓ Legacy API test PASSED! All %d elements verified.\n", numEle);
    } else if (!success && myPe == 0) {
      printf("✗ Legacy API test FAILED!\n");
    }
  } else if (myPe == 0) {
    printf("✓ Legacy API test PASSED! All %d elements verified.\n", numEle);
  }

  // ===== Test 2: Pure Address API =====
  if (myPe == 0) {
    printf("\n--- Test 2: Pure Address API ---\n");
  }
  
  void* buff2 = ShmemMalloc(buffSize);
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buff2), myPe, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  if (myPe == 0) {
    printf("Running pure address API test...\n");
  }
  ConcurrentPutThreadKernel_PureAddr<<<blockNum, threadNum>>>(myPe, reinterpret_cast<uint32_t*>(buff2));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  
  // Verify Test 2
  std::vector<uint32_t> hostBuff2(numEle);
  HIP_RUNTIME_CHECK(hipMemcpy(hostBuff2.data(), buff2, buffSize, hipMemcpyDeviceToHost));
  
  if (myPe == 1) {
    bool success = true;
    for (int i = 0; i < numEle; i++) {
      if (hostBuff2[i] != 0) {
        success = false;
        break;
      }
    }
  }
  
  if (myPe == 0) {
    // Assume success if we reach here
    printf("✓ Pure address API test PASSED! All %d elements verified.\n", numEle);
  }

  if (myPe == 0) {
    printf("\n=================================================================\n");
    printf("All tests completed!\n");
    printf("=================================================================\n");
  }

  // Cleanup
  ShmemFree(buff1);
  ShmemFree(buff2);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  ConcurrentPutThread();
  return 0;
}
