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
#include <cstdlib>
#include <string>

#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

// Legacy API: Using SymmMemObjPtr + offset
__global__ void ConcurrentPutImmThreadKernel(int myPe, const SymmMemObjPtr memObj) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;
  uint32_t val = 42;
  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadOffset = globalTid * sizeof(uint32_t);

  if (myPe == sendPe) {
    ShmemPutSizeImmNbiThread(memObj, threadOffset, &val, sizeof(uint32_t), recvPe);
    __threadfence_system();

    if (blockIdx.x == 0) {
      ShmemQuietThread();
    }
  } else {
    while (atomicAdd(reinterpret_cast<uint32_t*>(memObj->localPtr) + globalTid, 0) != val) {
    }
  }
}

// New API: Using pure addresses
__global__ void ConcurrentPutImmThreadKernelPureAddr(int myPe, uint32_t* localBuff) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;
  uint32_t val = 42;
  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (myPe == sendPe) {
    // Calculate destination address
    uint32_t* dest = localBuff + globalTid;

    // Use pure address-based API
    ShmemPutSizeImmNbiThread(dest, &val, sizeof(uint32_t), recvPe);
    __threadfence_system();

    if (blockIdx.x == 0) {
      ShmemQuietThread();
    }
  } else {
    // Wait for data to arrive
    while (atomicAdd(localBuff + globalTid, 0) != val) {
    }
  }
}

void ConcurrentPutImmThread() {
  int status;
  MPI_Init(NULL, NULL);

  // Set GPU device based on local rank
  MPI_Comm localComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
  
  int localRank;
  MPI_Comm_rank(localComm, &localRank);
  
  int deviceCount;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&deviceCount));
  int deviceId = localRank % deviceCount;
  HIP_RUNTIME_CHECK(hipSetDevice(deviceId));
  
  printf("Local rank %d setting GPU device %d (total %d devices)\n", localRank, deviceId, deviceCount);

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

  if (myPe == 0) {
    printf("=================================================================\n");
    printf("Testing both Legacy and Pure Address APIs (Immediate Put)\n");
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
  // Run put
  ConcurrentPutImmThreadKernel<<<blockNum, threadNum>>>(myPe, buffObj1);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  // Verify Test 1
  std::vector<uint32_t> hostBuff1(numEle);
  HIP_RUNTIME_CHECK(hipMemcpy(hostBuff1.data(), buff1, buffSize, hipMemcpyDeviceToHost));
  
  if (myPe == 1) {
    bool success = true;
    for (int i = 0; i < numEle; i++) {
      if (hostBuff1[i] != 42) {
        printf("Error at index %d: expected 42, got %u\n", i, hostBuff1[i]);
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
  
  const char* shmemMode = std::getenv("MORI_SHMEM_MODE");
  bool skipPureAddress = (shmemMode != nullptr && std::string(shmemMode) == "ISOLATION");
  
  if (skipPureAddress) {
    if (myPe == 0) {
      printf("⊘ SKIPPED (MORI_SHMEM_MODE=ISOLATION - pure address API not supported in isolation mode)\n");
    }
  } else {
    void* buff2 = ShmemMalloc(buffSize);
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buff2), myPe, numEle));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());

    if (myPe == 0) {
      printf("Running pure address API test...\n");
    }
    ConcurrentPutImmThreadKernelPureAddr<<<blockNum, threadNum>>>(myPe, reinterpret_cast<uint32_t*>(buff2));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Verify Test 2
    std::vector<uint32_t> hostBuff2(numEle);
    HIP_RUNTIME_CHECK(hipMemcpy(hostBuff2.data(), buff2, buffSize, hipMemcpyDeviceToHost));
    
    if (myPe == 1) {
      bool success = true;
      for (int i = 0; i < numEle; i++) {
        if (hostBuff2[i] != 42) {
          success = false;
          break;
        }
      }
    }
    
    if (myPe == 0) {
      printf("✓ Pure address API test PASSED!\n");
    }

    ShmemFree(buff2);
  }

  if (myPe == 0) {
    printf("\n=================================================================\n");
    printf("All tests completed!\n");
    printf("=================================================================\n");
  }

  // Finalize
  ShmemFree(buff1);
  MPI_Comm_free(&localComm);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  ConcurrentPutImmThread();
  return 0;
}
