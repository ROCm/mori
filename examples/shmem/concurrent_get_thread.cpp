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
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

// ============================================================================
// Test Suite Overview
// ============================================================================
// Test 1: Legacy API GET with SymmMemObjPtr + offset (Thread scope)
// Test 1B: Legacy API GET with SymmMemObjPtr + offset (Block scope)
// Test 2: Pure address API GET (Thread scope)
// Test 2B: Pure address API GET (Block scope)
// Test 3: Large multi-chunk GET (>200MB spanning multiple 64MB chunks)
// ============================================================================

// ============================================================================
// GPU Kernels
// ============================================================================

// Legacy API: GET using SymmMemObjPtr + offset (Thread scope)
// PE 0 reads from PE 1's buffer into its own local buffer
__global__ void ConcurrentGetThreadKernel(int myPe, const SymmMemObjPtr srcMemObj,
                                          const SymmMemObjPtr dstMemObj) {
  constexpr int getPe = 0;
  constexpr int remotePe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadOffset = globalTid * sizeof(uint32_t);

  if (myPe == getPe) {
    ShmemGetMemNbiThread(dstMemObj, threadOffset, srcMemObj, threadOffset, sizeof(uint32_t),
                         remotePe, 1);
    ShmemQuietThread(remotePe, 1);
  }
}

// Legacy API: GET using SymmMemObjPtr + offset (Block scope)
__global__ void ConcurrentGetBlockKernel(int myPe, const SymmMemObjPtr srcMemObj,
                                         const SymmMemObjPtr dstMemObj, size_t numElements) {
  constexpr int getPe = 0;
  constexpr int remotePe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalTid >= static_cast<int>(numElements)) {
    return;
  }

  size_t blockElemOffset = static_cast<size_t>(blockIdx.x) * blockDim.x;
  size_t blockByteOffset = blockElemOffset * sizeof(uint32_t);
  size_t blockBytes = static_cast<size_t>(blockDim.x) * sizeof(uint32_t);

  if (myPe == getPe) {
    ShmemGetMemNbiBlock(dstMemObj, blockByteOffset, srcMemObj, blockByteOffset, blockBytes,
                        remotePe, 1);
    if (threadIdx.x == 0) {
      ShmemQuietThread(remotePe, 1);
    }
  }
}

// Pure address API: GET using raw pointers (Thread scope)
// PE 0 reads from PE 1's remoteBuff into its own localBuff
__global__ void ConcurrentGetThreadKernel_PureAddr(int myPe, uint32_t* localBuff,
                                                   uint32_t* remoteBuff, size_t numElements) {
  constexpr int getPe = 0;
  constexpr int remotePe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalTid >= numElements) {
    return;
  }

  if (myPe == getPe) {
    uint32_t* dest = localBuff + globalTid;
    uint32_t* src = remoteBuff + globalTid;

    ShmemGetMemNbiThread(dest, src, sizeof(uint32_t), remotePe, 1);
    ShmemQuietThread(remotePe, 1);
  }
}

// Pure address API: GET using raw pointers (Block scope)
__global__ void ConcurrentGetBlockKernel_PureAddr(int myPe, uint32_t* localBuff,
                                                  uint32_t* remoteBuff, size_t numElements) {
  constexpr int getPe = 0;
  constexpr int remotePe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalTid >= static_cast<int>(numElements)) {
    return;
  }

  size_t blockElemOffset = static_cast<size_t>(blockIdx.x) * blockDim.x;
  size_t blockBytes = static_cast<size_t>(blockDim.x) * sizeof(uint32_t);
  uint32_t* dest = localBuff + blockElemOffset;
  uint32_t* src = remoteBuff + blockElemOffset;

  if (myPe == getPe) {
    ShmemGetMemNbiBlock(dest, src, blockBytes, remotePe, 1);
    if (threadIdx.x == 0) {
      ShmemQuietThread(remotePe, 1);
    }
  }
}

// Blocking GET: Legacy API (Thread scope) — no separate quiet needed
__global__ void BlockingGetThreadKernel(int myPe, const SymmMemObjPtr srcMemObj,
                                        const SymmMemObjPtr dstMemObj) {
  constexpr int getPe = 0;
  constexpr int remotePe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadOffset = globalTid * sizeof(uint32_t);

  if (myPe == getPe) {
    ShmemGetMemThread(dstMemObj, threadOffset, srcMemObj, threadOffset, sizeof(uint32_t),
                      remotePe, 1);
  }
}

// Blocking GET: Pure address API (Thread scope)
__global__ void BlockingGetThreadKernel_PureAddr(int myPe, uint32_t* localBuff,
                                                 uint32_t* remoteBuff, size_t numElements) {
  constexpr int getPe = 0;
  constexpr int remotePe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalTid >= numElements) return;

  if (myPe == getPe) {
    ShmemGetMemThread(localBuff + globalTid, remoteBuff + globalTid, sizeof(uint32_t), remotePe, 1);
  }
}

// ============================================================================
// Test Functions
// ============================================================================

void Test1_LegacyAPI(int myPe) {
  if (myPe == 0) {
    printf("\n--- Test 1: Legacy API GET (SymmMemObjPtr + offset, Thread scope) ---\n");
  }

  constexpr int threadNum = 128;
  constexpr int blockNum = 3;
  int numEle = threadNum * blockNum;
  int buffSize = numEle * sizeof(uint32_t);

  // srcBuff: PE 1 fills with known pattern, PE 0 will GET from PE 1
  void* srcBuff = ShmemMalloc(buffSize);
  // dstBuff: PE 0 will receive GET data here
  void* dstBuff = ShmemMalloc(buffSize);

  // PE 1 fills source with its rank (1), PE 0 fills with 0xDEAD (sentinel)
  if (myPe == 1) {
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(srcBuff), 1, numEle));
  } else {
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(srcBuff), 0, numEle));
  }
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(dstBuff), 0xDEAD, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr srcObj = ShmemQueryMemObjPtr(srcBuff);
  SymmMemObjPtr dstObj = ShmemQueryMemObjPtr(dstBuff);
  assert(srcObj.IsValid());
  assert(dstObj.IsValid());

  if (myPe == 0) {
    printf("Running legacy API GET test (thread scope)...\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  ConcurrentGetThreadKernel<<<blockNum, threadNum>>>(myPe, srcObj, dstObj);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    std::vector<uint32_t> hostBuff(numEle);
    HIP_RUNTIME_CHECK(hipMemcpy(hostBuff.data(), dstBuff, buffSize, hipMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < numEle; i++) {
      if (hostBuff[i] != 1) {
        printf("Error at index %d: expected 1 (PE 1's data), got %u\n", i, hostBuff[i]);
        success = false;
        break;
      }
    }
    if (success) {
      printf("✓ Legacy API GET test PASSED! All %d elements verified.\n", numEle);
    } else {
      printf("✗ Legacy API GET test FAILED!\n");
    }
  }

  ShmemFree(srcBuff);
  ShmemFree(dstBuff);
}

void Test1_LegacyAPI_Block(int myPe) {
  if (myPe == 0) {
    printf("\n--- Test 1B: Legacy API GET Block Scope ---\n");
  }

  constexpr int threadNum = 128;
  constexpr int blockNum = 3;
  int numEle = threadNum * blockNum;
  int buffSize = numEle * sizeof(uint32_t);

  void* srcBuff = ShmemMalloc(buffSize);
  void* dstBuff = ShmemMalloc(buffSize);

  if (myPe == 1) {
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(srcBuff), 1, numEle));
  } else {
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(srcBuff), 0, numEle));
  }
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(dstBuff), 0xDEAD, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr srcObj = ShmemQueryMemObjPtr(srcBuff);
  SymmMemObjPtr dstObj = ShmemQueryMemObjPtr(dstBuff);
  assert(srcObj.IsValid());
  assert(dstObj.IsValid());

  if (myPe == 0) {
    printf("Running legacy API GET test (block scope)...\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  ConcurrentGetBlockKernel<<<blockNum, threadNum>>>(myPe, srcObj, dstObj, numEle);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    std::vector<uint32_t> hostBuff(numEle);
    HIP_RUNTIME_CHECK(hipMemcpy(hostBuff.data(), dstBuff, buffSize, hipMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < numEle; i++) {
      if (hostBuff[i] != 1) {
        printf("Error at index %d: expected 1, got %u\n", i, hostBuff[i]);
        success = false;
        break;
      }
    }
    if (success) {
      printf("✓ Legacy API GET block test PASSED! All %d elements verified.\n", numEle);
    } else {
      printf("✗ Legacy API GET block test FAILED!\n");
    }
  }

  ShmemFree(srcBuff);
  ShmemFree(dstBuff);
}

void Test2_PureAddressAPI(int myPe) {
  if (myPe == 0) {
    printf("\n--- Test 2: Pure Address API GET (Thread scope) ---\n");
  }

  const char* shmemMode = std::getenv("MORI_SHMEM_MODE");
  bool skipPureAddress = (shmemMode != nullptr && std::string(shmemMode) == "ISOLATION");

  if (skipPureAddress) {
    if (myPe == 0) {
      printf("⊘ SKIPPED (MORI_SHMEM_MODE=ISOLATION)\n");
    }
    return;
  }

  constexpr int threadNum = 128;
  constexpr int blockNum = 3;
  int numEle = threadNum * blockNum;
  int buffSize = numEle * sizeof(uint32_t);

  // remoteBuff: symmetric allocation, PE 1 fills with pattern
  void* remoteBuff = ShmemMalloc(buffSize);
  // localBuff: PE 0's destination for GET
  void* localBuff = ShmemMalloc(buffSize);

  if (myPe == 1) {
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(remoteBuff), 1, numEle));
  } else {
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(remoteBuff), 0, numEle));
  }
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(localBuff), 0xDEAD, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  if (myPe == 0) {
    printf("Running pure address API GET test (thread scope)...\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  ConcurrentGetThreadKernel_PureAddr<<<blockNum, threadNum>>>(
      myPe, reinterpret_cast<uint32_t*>(localBuff), reinterpret_cast<uint32_t*>(remoteBuff),
      numEle);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    std::vector<uint32_t> hostBuff(numEle);
    HIP_RUNTIME_CHECK(hipMemcpy(hostBuff.data(), localBuff, buffSize, hipMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < numEle; i++) {
      if (hostBuff[i] != 1) {
        printf("Error at index %d: expected 1, got %u\n", i, hostBuff[i]);
        success = false;
        break;
      }
    }
    if (success) {
      printf("✓ Pure address API GET test PASSED! All %d elements verified.\n", numEle);
    } else {
      printf("✗ Pure address API GET test FAILED!\n");
    }
  }

  ShmemFree(remoteBuff);
  ShmemFree(localBuff);
}

void Test2_PureAddressAPI_Block(int myPe) {
  if (myPe == 0) {
    printf("\n--- Test 2B: Pure Address API GET Block Scope ---\n");
  }

  const char* shmemMode = std::getenv("MORI_SHMEM_MODE");
  bool skipPureAddress = (shmemMode != nullptr && std::string(shmemMode) == "ISOLATION");

  if (skipPureAddress) {
    if (myPe == 0) {
      printf("⊘ SKIPPED (MORI_SHMEM_MODE=ISOLATION)\n");
    }
    return;
  }

  constexpr int threadNum = 128;
  constexpr int blockNum = 3;
  int numEle = threadNum * blockNum;
  int buffSize = numEle * sizeof(uint32_t);

  void* remoteBuff = ShmemMalloc(buffSize);
  void* localBuff = ShmemMalloc(buffSize);

  if (myPe == 1) {
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(remoteBuff), 1, numEle));
  } else {
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(remoteBuff), 0, numEle));
  }
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(localBuff), 0xDEAD, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  if (myPe == 0) {
    printf("Running pure address API GET test (block scope)...\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  ConcurrentGetBlockKernel_PureAddr<<<blockNum, threadNum>>>(
      myPe, reinterpret_cast<uint32_t*>(localBuff), reinterpret_cast<uint32_t*>(remoteBuff),
      numEle);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    std::vector<uint32_t> hostBuff(numEle);
    HIP_RUNTIME_CHECK(hipMemcpy(hostBuff.data(), localBuff, buffSize, hipMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < numEle; i++) {
      if (hostBuff[i] != 1) {
        printf("Error at index %d: expected 1, got %u\n", i, hostBuff[i]);
        success = false;
        break;
      }
    }
    if (success) {
      printf("✓ Pure address API GET block test PASSED! All %d elements verified.\n", numEle);
    } else {
      printf("✗ Pure address API GET block test FAILED!\n");
    }
  }

  ShmemFree(remoteBuff);
  ShmemFree(localBuff);
}

void Test3_LargeMultiChunk(int myPe) {
  if (myPe == 0) {
    printf("\n--- Test 3: Large Multi-Chunk GET (>200MB) ---\n");
  }

  const char* shmemMode = std::getenv("MORI_SHMEM_MODE");
  bool skipPureAddress = (shmemMode != nullptr && std::string(shmemMode) == "ISOLATION");

  if (skipPureAddress) {
    if (myPe == 0) {
      printf("⊘ SKIPPED (MORI_SHMEM_MODE=ISOLATION)\n");
    }
    return;
  }

  constexpr size_t largeSize = 200 * 1024 * 1024;
  constexpr size_t largeNumEle = largeSize / sizeof(uint32_t);

  if (myPe == 0) {
    printf("Allocating %zu MB (%zu elements)...\n", largeSize / (1024 * 1024), largeNumEle);
  }

  void* remoteBuff = ShmemMalloc(largeSize);
  void* localBuff = ShmemMalloc(largeSize);
  assert(remoteBuff != nullptr);
  assert(localBuff != nullptr);

  if (myPe == 1) {
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(remoteBuff), 1, largeNumEle));
  } else {
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(remoteBuff), 0, largeNumEle));
  }
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(localBuff), 0xDEAD, largeNumEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  if (myPe == 0) {
    printf("Testing large data GET with pure address API...\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int largeBlockNum = 1024;
  constexpr int largeThreadNum = 256;
  constexpr size_t testElements = largeBlockNum * largeThreadNum;

  ConcurrentGetThreadKernel_PureAddr<<<largeBlockNum, largeThreadNum>>>(
      myPe, reinterpret_cast<uint32_t*>(localBuff), reinterpret_cast<uint32_t*>(remoteBuff),
      testElements);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    std::vector<uint32_t> hostBuff(testElements);
    HIP_RUNTIME_CHECK(
        hipMemcpy(hostBuff.data(), localBuff, testElements * sizeof(uint32_t), hipMemcpyDeviceToHost));

    bool success = true;
    for (size_t i = 0; i < testElements; i++) {
      if (hostBuff[i] != 1) {
        printf("Error at index %zu: expected 1, got %u\n", i, hostBuff[i]);
        success = false;
        break;
      }
    }
    if (success) {
      printf("✓ Large multi-chunk GET test PASSED! Verified %zu elements.\n", testElements);
    } else {
      printf("✗ Large multi-chunk GET test FAILED!\n");
    }
  }

  ShmemFree(remoteBuff);
  ShmemFree(localBuff);
}

void Test4_BlockingLegacyAPI(int myPe) {
  if (myPe == 0) {
    printf("\n--- Test 4: Blocking GET Legacy API (Thread scope) ---\n");
  }

  constexpr int threadNum = 128;
  constexpr int blockNum = 3;
  int numEle = threadNum * blockNum;
  int buffSize = numEle * sizeof(uint32_t);

  void* srcBuff = ShmemMalloc(buffSize);
  void* dstBuff = ShmemMalloc(buffSize);

  if (myPe == 1) {
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(srcBuff), 1, numEle));
  } else {
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(srcBuff), 0, numEle));
  }
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(dstBuff), 0xDEAD, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr srcObj = ShmemQueryMemObjPtr(srcBuff);
  SymmMemObjPtr dstObj = ShmemQueryMemObjPtr(dstBuff);
  assert(srcObj.IsValid());
  assert(dstObj.IsValid());

  if (myPe == 0) {
    printf("Running blocking GET legacy API test...\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  BlockingGetThreadKernel<<<blockNum, threadNum>>>(myPe, srcObj, dstObj);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    std::vector<uint32_t> hostBuff(numEle);
    HIP_RUNTIME_CHECK(hipMemcpy(hostBuff.data(), dstBuff, buffSize, hipMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < numEle; i++) {
      if (hostBuff[i] != 1) {
        printf("Error at index %d: expected 1, got %u\n", i, hostBuff[i]);
        success = false;
        break;
      }
    }
    if (success) {
      printf("✓ Blocking GET legacy API test PASSED! All %d elements verified.\n", numEle);
    } else {
      printf("✗ Blocking GET legacy API test FAILED!\n");
    }
  }

  ShmemFree(srcBuff);
  ShmemFree(dstBuff);
}

void Test5_BlockingPureAddressAPI(int myPe) {
  if (myPe == 0) {
    printf("\n--- Test 5: Blocking GET Pure Address API (Thread scope) ---\n");
  }

  const char* shmemMode = std::getenv("MORI_SHMEM_MODE");
  bool skipPureAddress = (shmemMode != nullptr && std::string(shmemMode) == "ISOLATION");

  if (skipPureAddress) {
    if (myPe == 0) {
      printf("⊘ SKIPPED (MORI_SHMEM_MODE=ISOLATION)\n");
    }
    return;
  }

  constexpr int threadNum = 128;
  constexpr int blockNum = 3;
  int numEle = threadNum * blockNum;
  int buffSize = numEle * sizeof(uint32_t);

  void* remoteBuff = ShmemMalloc(buffSize);
  void* localBuff = ShmemMalloc(buffSize);

  if (myPe == 1) {
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(remoteBuff), 1, numEle));
  } else {
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(remoteBuff), 0, numEle));
  }
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(localBuff), 0xDEAD, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  if (myPe == 0) {
    printf("Running blocking GET pure address API test...\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  BlockingGetThreadKernel_PureAddr<<<blockNum, threadNum>>>(
      myPe, reinterpret_cast<uint32_t*>(localBuff), reinterpret_cast<uint32_t*>(remoteBuff),
      numEle);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    std::vector<uint32_t> hostBuff(numEle);
    HIP_RUNTIME_CHECK(hipMemcpy(hostBuff.data(), localBuff, buffSize, hipMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < numEle; i++) {
      if (hostBuff[i] != 1) {
        printf("Error at index %d: expected 1, got %u\n", i, hostBuff[i]);
        success = false;
        break;
      }
    }
    if (success) {
      printf("✓ Blocking GET pure address API test PASSED! All %d elements verified.\n", numEle);
    } else {
      printf("✗ Blocking GET pure address API test FAILED!\n");
    }
  }

  ShmemFree(remoteBuff);
  ShmemFree(localBuff);
}

// ============================================================================
// Main
// ============================================================================

void ConcurrentGetThread() {
  MPI_Init(NULL, NULL);

  MPI_Comm localComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);

  int localRank;
  MPI_Comm_rank(localComm, &localRank);

  int deviceCount;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&deviceCount));
  int deviceId = localRank % deviceCount;
  HIP_RUNTIME_CHECK(hipSetDevice(deviceId));

  printf("Local rank %d setting GPU device %d (total %d devices)\n", localRank, deviceId,
         deviceCount);

  int status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();
  assert(npes == 2);

  if (myPe == 0) {
    printf("=================================================================\n");
    printf("MORI SHMEM GET Test Suite\n");
    printf("=================================================================\n");
  }

  Test1_LegacyAPI(myPe);
  Test1_LegacyAPI_Block(myPe);
  Test2_PureAddressAPI(myPe);
  Test2_PureAddressAPI_Block(myPe);
  Test3_LargeMultiChunk(myPe);
  Test4_BlockingLegacyAPI(myPe);
  Test5_BlockingPureAddressAPI(myPe);

  if (myPe == 0) {
    printf("\n=================================================================\n");
    printf("All GET tests completed!\n");
    printf("Summary:\n");
    printf("  - Test 1:  Nbi GET Legacy API (Thread scope)\n");
    printf("  - Test 1B: Nbi GET Legacy API (Block scope)\n");
    printf("  - Test 2:  Nbi GET Pure Address API (Thread scope)\n");
    printf("  - Test 2B: Nbi GET Pure Address API (Block scope)\n");
    printf("  - Test 3:  Nbi GET Large Multi-Chunk (>200MB)\n");
    printf("  - Test 4:  Blocking GET Legacy API (Thread scope)\n");
    printf("  - Test 5:  Blocking GET Pure Address API (Thread scope)\n");
    printf("=================================================================\n");
  }

  MPI_Comm_free(&localComm);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  ConcurrentGetThread();
  return 0;
}
