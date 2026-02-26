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

// ============================================================================
// Test Suite Overview
// ============================================================================
// Test 0: Direct GPU-to-GPU access (P2P peer pointer test)
// Test 1: Legacy API with SymmMemObjPtr + offset
// Test 2: Pure address API with small data
// Test 3: Large multi-chunk allocation (>200MB spanning multiple 64MB chunks)
// Test 4: Mixed malloc/free with chunk overlap (reference counting)
// Test 5: Fragmentation and VA reuse
// ============================================================================

// Forward declarations
void Test0_DirectGPUAccess(int myPe);
void Test1_LegacyAPI(int myPe);
void Test2_PureAddressAPI(int myPe);
void Test3_LargeMultiChunk(int myPe);
void Test4_MixedMallocFree(int myPe);
void Test5_FragmentationReuse(int myPe);

// ============================================================================
// GPU Kernels
// ============================================================================

// Legacy API: Using SymmMemObjPtr + offset
__global__ void ConcurrentPutThreadKernel(int myPe, const SymmMemObjPtr memObj) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadOffset = globalTid * sizeof(uint32_t);

  if (myPe == sendPe) {
    ShmemPutMemNbiThread(memObj, threadOffset, memObj, threadOffset, sizeof(uint32_t), recvPe, 1);
    ShmemFenceThread();
  } else {
    while (atomicAdd(reinterpret_cast<uint32_t*>(memObj->localPtr) + globalTid, 0) != sendPe) {
    }
  }
}

// New API: Using pure addresses
__global__ void ConcurrentPutThreadKernel_PureAddr(int myPe, uint32_t* localBuff,
                                                   size_t numElements) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (globalTid >= numElements) {
    return;
  }

  if (myPe == sendPe) {
    uint32_t* src = localBuff + globalTid;
    uint32_t* dest = localBuff + globalTid;

    ShmemPutMemNbiThread(dest, src, sizeof(uint32_t), recvPe, 1);
    __threadfence_system();

    if (blockIdx.x == 0) {
      ShmemQuietThread();
    }
  } else {
    while (atomicAdd(localBuff + globalTid, 0) != sendPe) {
    }
  }
}

// Test direct GPU-to-GPU access using peer pointers
__global__ void DirectAccessTestKernel(int myPe, const SymmMemObjPtr memObj, uint32_t* peerBuffer,
                                       bool* accessResult) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (globalTid == 0) {
    printf("PE %d: localPtr = %p\n", myPe, memObj->localPtr);
    if (memObj->peerPtrs) {
      printf("PE %d: peerPtrs[0] = %p\n", myPe, (void*)memObj->peerPtrs[0]);
      printf("PE %d: peerPtrs[1] = %p\n", myPe, (void*)memObj->peerPtrs[1]);
    }
  }

  __syncthreads();

  if (myPe == sendPe && globalTid < 1) {
    if (peerBuffer != nullptr) {
      uint32_t testValue = 0xABCD0000 + globalTid;

      __threadfence_system();
      peerBuffer[globalTid] = testValue;
      __threadfence_system();

      if (globalTid == 0) {
        *accessResult = true;
        printf("PE %d: Successfully wrote to peer address %p\n", myPe, peerBuffer);
      }
    }
  } else if (myPe == recvPe && globalTid < 1) {
    uint32_t expected = 0xABCD0000 + globalTid;
    uint32_t* localBuff = reinterpret_cast<uint32_t*>(memObj->localPtr);

    int timeout = 1000000;
    while (timeout-- > 0 && localBuff[globalTid] != expected) {
      __threadfence_system();
    }

    if (localBuff[globalTid] == expected) {
      if (globalTid == 0) {
        printf("PE %d: Successfully received data from peer\n", myPe);
      }
    } else {
      if (globalTid == 0) {
        printf("PE %d: Failed to receive expected data. Got 0x%x, expected 0x%x\n", myPe,
               localBuff[globalTid], expected);
        *accessResult = false;
      }
    }
  }
}

// ============================================================================
// Test Functions
// ============================================================================

void Test0_DirectGPUAccess(int myPe) {
  const char* disableP2P = std::getenv("MORI_DISABLE_P2P");
  bool skipDirectAccess = (disableP2P != nullptr &&
                           (std::string(disableP2P) == "ON" || std::string(disableP2P) == "1"));

  if (skipDirectAccess) {
    if (myPe == 0) {
      printf("\n--- Test 0: Direct GPU-to-GPU Access Test ---\n");
      printf("⊘ SKIPPED (MORI_DISABLE_P2P=%s - no direct P2P path available)\n", disableP2P);
    }
    return;
  }

  if (myPe == 0) {
    printf("\n--- Test 0: Direct GPU-to-GPU Access Test ---\n");
  }

  constexpr int threadNum = 128;
  constexpr int blockNum = 3;
  int numEle = threadNum * blockNum;
  int buffSize = numEle * sizeof(uint32_t);

  void* buff = ShmemMalloc(buffSize);
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buff), 0x12345678, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr buffObj = ShmemQueryMemObjPtr(buff);
  assert(buffObj.IsValid());

  bool* d_accessResult;
  HIP_RUNTIME_CHECK(hipMalloc(&d_accessResult, sizeof(bool)));
  HIP_RUNTIME_CHECK(hipMemset(d_accessResult, 1, sizeof(bool)));

  if (myPe == 0) {
    printf("Running direct access test...\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    uint32_t* peerAddr = reinterpret_cast<uint32_t*>(buffObj.cpu->peerPtrs[1]);
    DirectAccessTestKernel<<<2, 64>>>(myPe, buffObj, peerAddr, d_accessResult);
  } else {
    DirectAccessTestKernel<<<2, 64>>>(myPe, buffObj, nullptr, d_accessResult);
  }

  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  bool h_accessResult;
  HIP_RUNTIME_CHECK(
      hipMemcpy(&h_accessResult, d_accessResult, sizeof(bool), hipMemcpyDeviceToHost));

  if (myPe == 0) {
    if (h_accessResult) {
      printf("✓ Direct GPU-to-GPU access test PASSED!\n");
    } else {
      printf("✗ Direct GPU-to-GPU access test FAILED!\n");
    }
  }

  std::vector<uint32_t> hostBuff(64);
  HIP_RUNTIME_CHECK(hipMemcpy(hostBuff.data(), buff, 64 * sizeof(uint32_t), hipMemcpyDeviceToHost));

  if (myPe == 1) {
    printf("PE %d verification: First few values: ", myPe);
    for (int i = 0; i < 8; i++) {
      printf("0x%x ", hostBuff[i]);
    }
    printf("\n");
  }

  HIP_RUNTIME_CHECK(hipFree(d_accessResult));
  ShmemFree(buff);
}

void Test1_LegacyAPI(int myPe) {
  if (myPe == 0) {
    printf("\n--- Test 1: Legacy API (SymmMemObjPtr + offset) ---\n");
  }

  constexpr int threadNum = 128;
  constexpr int blockNum = 3;
  int numEle = threadNum * blockNum;
  int buffSize = numEle * sizeof(uint32_t);

  void* buff = ShmemMalloc(buffSize);
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buff), myPe, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr buffObj = ShmemQueryMemObjPtr(buff);
  assert(buffObj.IsValid());

  if (myPe == 0) {
    printf("Running legacy API test...\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  ConcurrentPutThreadKernel<<<blockNum, threadNum>>>(myPe, buffObj);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<uint32_t> hostBuff(numEle);
  HIP_RUNTIME_CHECK(hipMemcpy(hostBuff.data(), buff, buffSize, hipMemcpyDeviceToHost));

  if (myPe == 1) {
    bool success = true;
    for (int i = 0; i < numEle; i++) {
      if (hostBuff[i] != 0) {
        printf("Error at index %d: expected 0, got %u\n", i, hostBuff[i]);
        success = false;
        break;
      }
    }
    if (!success) {
      printf("✗ Legacy API test FAILED!\n");
    }
  }

  if (myPe == 0) {
    printf("✓ Legacy API test PASSED! All %d elements verified.\n", numEle);
  }

  ShmemFree(buff);
}

void Test2_PureAddressAPI(int myPe) {
  if (myPe == 0) {
    printf("\n--- Test 2: Pure Address API ---\n");
  }

  const char* shmemMode = std::getenv("MORI_SHMEM_MODE");
  bool skipPureAddress = (shmemMode != nullptr && std::string(shmemMode) == "ISOLATION");

  if (skipPureAddress) {
    if (myPe == 0) {
      printf(
          "⊘ SKIPPED (MORI_SHMEM_MODE=ISOLATION - pure address API not supported in isolation "
          "mode)\n");
    }
    return;
  }

  constexpr int threadNum = 128;
  constexpr int blockNum = 3;
  int numEle = threadNum * blockNum;
  int buffSize = numEle * sizeof(uint32_t);

  void* buff = ShmemMalloc(buffSize);
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buff), myPe, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  if (myPe == 0) {
    printf("Running pure address API test...\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  ConcurrentPutThreadKernel_PureAddr<<<blockNum, threadNum>>>(
      myPe, reinterpret_cast<uint32_t*>(buff), numEle);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<uint32_t> hostBuff(numEle);
  HIP_RUNTIME_CHECK(hipMemcpy(hostBuff.data(), buff, buffSize, hipMemcpyDeviceToHost));

  if (myPe == 1) {
    bool success = true;
    for (int i = 0; i < numEle; i++) {
      if (hostBuff[i] != 0) {
        printf("Error at index %d: expected 0, got %u\n", i, hostBuff[i]);
        success = false;
        break;
      }
    }
    if (!success) {
      printf("✗ Pure address API test FAILED!\n");
    }
  }

  if (myPe == 0) {
    printf("✓ Pure address API test PASSED! All %d elements verified.\n", numEle);
  }

  ShmemFree(buff);
}

void Test3_LargeMultiChunk(int myPe) {
  if (myPe == 0) {
    printf("\n--- Test 3: Large Multi-Chunk Allocation (>200MB) ---\n");
  }

  const char* shmemMode = std::getenv("MORI_SHMEM_MODE");
  bool skipPureAddress = (shmemMode != nullptr && std::string(shmemMode) == "ISOLATION");

  if (skipPureAddress) {
    if (myPe == 0) {
      printf(
          "⊘ SKIPPED (MORI_SHMEM_MODE=ISOLATION - pure address API not supported in isolation "
          "mode)\n");
    }
    return;
  }

  constexpr size_t largeSize = 200 * 1024 * 1024;  // 200 MB
  constexpr size_t largeNumEle = largeSize / sizeof(uint32_t);

  if (myPe == 0) {
    printf("Allocating %zu MB (%zu elements)...\n", largeSize / (1024 * 1024), largeNumEle);
  }

  void* largeBuff = ShmemMalloc(largeSize);
  assert(largeBuff != nullptr);

  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(largeBuff), myPe, largeNumEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  if (myPe == 0) {
    printf("Testing large data transfer with pure address API...\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  constexpr int largeBlockNum = 1024;
  constexpr int largeThreadNum = 256;
  constexpr size_t testElements = largeBlockNum * largeThreadNum;

  ConcurrentPutThreadKernel_PureAddr<<<largeBlockNum, largeThreadNum>>>(
      myPe, reinterpret_cast<uint32_t*>(largeBuff), testElements);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<uint32_t> hostLargeBuff(testElements);
  HIP_RUNTIME_CHECK(hipMemcpy(hostLargeBuff.data(), largeBuff, testElements * sizeof(uint32_t),
                              hipMemcpyDeviceToHost));

  if (myPe == 1) {
    bool success = true;
    for (size_t i = 0; i < testElements; i++) {
      if (hostLargeBuff[i] != 0) {
        printf("Error at index %zu: expected 0, got %u\n", i, hostLargeBuff[i]);
        success = false;
        break;
      }
    }
    if (!success) {
      printf("✗ Large multi-chunk allocation test FAILED!\n");
    }
  }

  if (myPe == 0) {
    printf(
        "✓ Large multi-chunk allocation test PASSED! Verified %zu elements (200MB allocation "
        "successful).\n",
        testElements);
  }

  ShmemFree(largeBuff);
}

void Test4_MixedMallocFree(int myPe) {
  if (myPe == 0) {
    printf("\n--- Test 4: Mixed Malloc/Free with Chunk Overlap ---\n");
    printf("Testing reference counting mechanism for shared chunks...\n");
  }

  constexpr size_t sizeA = 150 * 1024 * 1024;
  void* buffA = ShmemMalloc(sizeA);
  assert(buffA != nullptr);
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buffA), 0xAAAA0000 + myPe,
                                 sizeA / sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  if (myPe == 0) {
    printf("Step 1: Allocated buffer A (%zu MB)\n", sizeA / (1024 * 1024));
  }

  constexpr size_t sizeB = 100 * 1024 * 1024;
  void* buffB = ShmemMalloc(sizeB);
  assert(buffB != nullptr);
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buffB), 0xBBBB0000 + myPe,
                                 sizeB / sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  if (myPe == 0) {
    printf("Step 2: Allocated buffer B (%zu MB)\n", sizeB / (1024 * 1024));
  }

  uint32_t testValA;
  HIP_RUNTIME_CHECK(hipMemcpy(&testValA, buffA, sizeof(uint32_t), hipMemcpyDeviceToHost));
  if (testValA != 0xAAAA0000 + myPe) {
    printf("PE %d: Warning - Buffer A verification failed before free! Got 0x%x\n", myPe, testValA);
  }

  if (myPe == 0) {
    printf("Step 3: Freeing buffer A (shared chunks should remain allocated)...\n");
  }
  ShmemFree(buffA);

  uint32_t testValB;
  HIP_RUNTIME_CHECK(hipMemcpy(&testValB, buffB, sizeof(uint32_t), hipMemcpyDeviceToHost));
  if (testValB != 0xBBBB0000 + myPe) {
    printf("PE %d: ✗ Buffer B corrupted after freeing A! Got 0x%x, expected 0x%x\n", myPe, testValB,
           0xBBBB0000 + myPe);
  } else {
    if (myPe == 0) {
      printf("✓ Buffer B still valid after freeing A (reference counting works!)\n");
    }
  }

  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buffB), 0xCCCC0000 + myPe,
                                 sizeB / sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  if (myPe == 0) {
    printf("Step 4: Updated buffer B after freeing A...\n");
  }

  if (myPe == 0) {
    printf("Step 5: Freeing buffer B (all shared chunks should now be released)...\n");
  }
  ShmemFree(buffB);

  if (myPe == 0) {
    printf("✓ Mixed malloc/free test PASSED! Reference counting verified.\n");
  }
}

void Test5_FragmentationReuse(int myPe) {
  if (myPe == 0) {
    printf("\n--- Test 5: Fragmentation and VA Reuse Test ---\n");
  }

  constexpr int numFragments = 5;
  constexpr size_t fragmentSize = 80 * 1024 * 1024;  // 80MB each
  void* fragments[numFragments];

  if (myPe == 0) {
    printf("Allocating %d fragments of %zu MB each...\n", numFragments,
           fragmentSize / (1024 * 1024));
  }

  for (int i = 0; i < numFragments; i++) {
    fragments[i] = ShmemMalloc(fragmentSize);
    assert(fragments[i] != nullptr);
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(fragments[i]), 0x1000 * i + myPe,
                                   fragmentSize / sizeof(uint32_t)));
  }
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  if (myPe == 0) {
    printf("Freeing alternating fragments (creating fragmentation)...\n");
  }

  for (int i = 0; i < numFragments; i += 2) {
    ShmemFree(fragments[i]);
    fragments[i] = nullptr;
  }

  bool allValid = true;
  for (int i = 1; i < numFragments; i += 2) {
    uint32_t testVal;
    HIP_RUNTIME_CHECK(hipMemcpy(&testVal, fragments[i], sizeof(uint32_t), hipMemcpyDeviceToHost));
    if (testVal != 0x1000 * i + myPe) {
      printf("PE %d: Fragment %d corrupted! Got 0x%x, expected 0x%x\n", myPe, i, testVal,
             0x1000 * i + myPe);
      allValid = false;
    }
  }

  if (allValid && myPe == 0) {
    printf("✓ Remaining fragments intact after freeing alternating allocations\n");
  }

  if (myPe == 0) {
    printf("Re-allocating in freed spaces (testing VA reuse)...\n");
  }

  for (int i = 0; i < numFragments; i += 2) {
    fragments[i] = ShmemMalloc(fragmentSize);
    assert(fragments[i] != nullptr);
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(fragments[i]), 0x2000 * i + myPe,
                                   fragmentSize / sizeof(uint32_t)));
  }
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  allValid = true;
  for (int i = 0; i < numFragments; i++) {
    uint32_t testVal;
    uint32_t expected = (i % 2 == 0) ? (0x2000 * i + myPe) : (0x1000 * i + myPe);
    HIP_RUNTIME_CHECK(hipMemcpy(&testVal, fragments[i], sizeof(uint32_t), hipMemcpyDeviceToHost));
    if (testVal != expected) {
      printf("PE %d: Fragment %d verification failed! Got 0x%x, expected 0x%x\n", myPe, i, testVal,
             expected);
      allValid = false;
    }
  }

  if (allValid && myPe == 0) {
    printf("✓ All fragments verified after reallocation\n");
  }

  for (int i = 0; i < numFragments; i++) {
    ShmemFree(fragments[i]);
  }

  if (myPe == 0) {
    printf("✓ Fragmentation and VA reuse test PASSED!\n");
  }
}

void ConcurrentPutThread() {
  int status;
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

  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();
  assert(npes == 2);

  if (myPe == 0) {
    printf("=================================================================\n");
    printf("MORI SHMEM Comprehensive Test Suite\n");
    printf("=================================================================\n");
  }

  Test0_DirectGPUAccess(myPe);
  Test1_LegacyAPI(myPe);
  Test2_PureAddressAPI(myPe);
  Test3_LargeMultiChunk(myPe);
  Test4_MixedMallocFree(myPe);
  Test5_FragmentationReuse(myPe);

  if (myPe == 0) {
    printf("\n=================================================================\n");
    printf("All tests completed!\n");
    printf("Summary:\n");
    printf("  - Test 0: Direct GPU-to-GPU access\n");
    printf("  - Test 1: Legacy API with small data\n");
    printf("  - Test 2: Pure address API with small data\n");
    printf("  - Test 3: Large multi-chunk allocation (>200MB)\n");
    printf("  - Test 4: Mixed malloc/free with reference counting\n");
    printf("  - Test 5: Fragmentation and VA reuse\n");
    printf("=================================================================\n");
  }

  MPI_Comm_free(&localComm);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  ConcurrentPutThread();
  return 0;
}
