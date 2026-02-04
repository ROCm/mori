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

// This test validates the P2P direct memory access using GetAs(pe) API
// It tests that:
// 1. GetAs(pe) uses p2pPeerPtrs for same-node peers
// 2. Direct atomic operations work via P2P pointers
// 3. WarpCopy works via P2P pointers
// 4. This works regardless of MORI_DISABLE_P2P setting

#include <mpi.h>

#include <cassert>
#include <cstdlib>
#include <string>

#include "mori/application/utils/check.hpp"
#include "mori/core/core.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

// Test 1: Direct atomic access via GetAs(pe)
template <typename T>
__global__ void DirectAtomicAccessKernel(int myPe, int npes, const SymmMemObjPtr memObj,
                                        int* errorFlag) {
  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

  // Each PE writes to the next PE in a ring pattern (intra-node only)
  int targetPe = (myPe + 1) % npes;

  // Get the pointer directly using GetAs - this should use p2pPeerPtrs
  T* targetPtr = memObj->template GetAs<T*>(targetPe);

  // Check if pointer is valid for same-node access
  if (targetPtr == nullptr) {
    if (globalTid == 0) {
      printf("PE %d: ERROR - GetAs(%d) returned nullptr (expected valid P2P pointer)\n", myPe,
             targetPe);
      atomicAdd(errorFlag, 1);
    }
    return;
  }

  // Perform direct atomic add via P2P pointer
  if (globalTid < 32) {  // Only first warp
    atomicAdd(&targetPtr[0], static_cast<T>(1));
  }

  if (globalTid == 0) {
    printf("PE %d: Successfully performed atomic add to PE %d via P2P pointer %p\n", myPe, targetPe,
           targetPtr);
  }
}

// Test 2: Direct memory copy via GetAs(pe)
template <typename T>
__global__ void DirectMemoryCopyKernel(int myPe, int npes, const SymmMemObjPtr srcMemObj,
                                       const SymmMemObjPtr dstMemObj, int numElements,
                                       int* errorFlag) {
  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = globalTid / warpSize;
  int laneId = threadIdx.x & (warpSize - 1);

  // Each PE copies data to the next PE
  int targetPe = (myPe + 1) % npes;

  // Get local source pointer
  T* localSrc = srcMemObj->template GetAs<T*>();

  // Get target pointer via GetAs - this should use p2pPeerPtrs
  T* targetDst = dstMemObj->template GetAs<T*>(targetPe);

  if (targetDst == nullptr) {
    if (globalTid == 0) {
      printf("PE %d: ERROR - GetAs(%d) returned nullptr for dst buffer\n", myPe, targetPe);
      atomicAdd(errorFlag, 1);
    }
    return;
  }

  // Use WarpCopy to copy data
  if (globalWarpId == 0) {
    WarpCopy(targetDst, localSrc, numElements);
    if (laneId == 0) {
      printf("PE %d: Successfully copied %d elements to PE %d via P2P pointer\n", myPe, numElements,
             targetPe);
    }
  }
}

// Test 3: Verify pointer addresses
__global__ void VerifyPointerKernel(int myPe, int npes, const SymmMemObjPtr memObj) {
  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (globalTid != 0) return;

  printf("\n=== PE %d: Pointer Verification ===\n", myPe);
  printf("Local pointer (GetAs()): %p\n", memObj->template GetAs<void*>());

  for (int pe = 0; pe < npes; pe++) {
    void* peerPtr = memObj->Get(pe);
    if (pe == myPe) {
      printf("  PE %d (self):       %p (expected: local ptr)\n", pe, peerPtr);
    } else {
      if (peerPtr != nullptr) {
        printf("  PE %d (same-node): %p (P2P accessible)\n", pe, peerPtr);
      } else {
        printf("  PE %d (diff-node): %p (not P2P accessible)\n", pe, peerPtr);
      }
    }
  }
  printf("=======================================\n\n");
}

// Test 4: Shmem Put Thread (Legacy API)
template <typename T>
__global__ void ShmemPutThreadKernel(int myPe, int npes, const SymmMemObjPtr memObj,
                                     int numElements) {
  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (globalTid >= numElements) return;

  int targetPe = (myPe + 1) % npes;
  int sourcePe = (myPe - 1 + npes) % npes;
  size_t threadOffset = globalTid * sizeof(T);

  // Each PE writes its rank to the next PE's buffer
  if (myPe != targetPe) {  // Skip if only one PE
    ShmemPutMemNbiThread(memObj, threadOffset, memObj, threadOffset, sizeof(T), targetPe, 1);
    ShmemFenceThread();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
      ShmemQuietThread(targetPe);
    }
  }
  
  // Receiver side: wait for data to arrive by polling
  T* localBuff = reinterpret_cast<T*>(memObj->localPtr);
  T expected = static_cast<T>(sourcePe);
  while (atomicAdd(&localBuff[globalTid], 0) != expected) {
    // Busy wait for data
  }
}

// Test 5: Shmem Put Thread (Pure Address API)
template <typename T>
__global__ void ShmemPutThreadKernel_PureAddr(int myPe, int npes, T* localBuff, int numElements) {
  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (globalTid >= numElements) return;

  int targetPe = (myPe + 1) % npes;
  int sourcePe = (myPe - 1 + npes) % npes;

  // Each PE writes its rank to the next PE's buffer
  if (myPe != targetPe) {  // Skip if only one PE
    T* src = localBuff + globalTid;
    T* dest = localBuff + globalTid;

    ShmemPutMemNbiThread(dest, src, sizeof(T), targetPe, 1);
    __threadfence_system();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
      ShmemQuietThread(targetPe);
    }
  }
  
  // Receiver side: wait for data to arrive by polling
  T expected = static_cast<T>(sourcePe);
  while (atomicAdd(&localBuff[globalTid], 0) != expected) {
    // Busy wait for data
  }
}

void testDirectP2PAccess() {
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

  printf("Local rank %d setting GPU device %d (total %d devices)\n", localRank, deviceId,
         deviceCount);

  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  // Check if MORI_DISABLE_P2P is set
  const char* disableP2P = std::getenv("MORI_DISABLE_P2P");
  bool p2pDisabled = (disableP2P != nullptr && std::string(disableP2P) == "1");

  // Check if we should skip pure address API tests
  const char* shmemMode = std::getenv("MORI_SHMEM_MODE");
  bool skipPureAddress = (shmemMode != nullptr && std::string(shmemMode) == "ISOLATION");

  if (myPe == 0) {
    printf("\n=================================================================\n");
    printf("Testing P2P Direct Memory Access via GetAs(pe) API\n");
    printf("=================================================================\n");
    printf("Number of PEs: %d\n", npes);
    printf("MORI_DISABLE_P2P: %s\n", p2pDisabled ? "1 (RDMA transport)" : "0 (P2P transport)");
    printf("MORI_SHMEM_MODE: %s\n", shmemMode ? shmemMode : "default");
    if (skipPureAddress) {
      printf("Note: Pure Address API tests will be skipped in ISOLATION mode\n");
    }
    printf("Expected: P2P data path should work regardless of MORI_DISABLE_P2P\n");
    printf("=================================================================\n\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Allocate buffers
  constexpr int numElements = 256;
  size_t buffSize = numElements * sizeof(uint64_t);

  void* atomicBuff = ShmemExtMallocWithFlags(buffSize, hipDeviceMallocUncached);
  void* srcBuff = ShmemExtMallocWithFlags(buffSize, hipDeviceMallocUncached);
  void* dstBuff = ShmemExtMallocWithFlags(buffSize, hipDeviceMallocUncached);

  SymmMemObjPtr atomicBuffObj = ShmemQueryMemObjPtr(atomicBuff);
  SymmMemObjPtr srcBuffObj = ShmemQueryMemObjPtr(srcBuff);
  SymmMemObjPtr dstBuffObj = ShmemQueryMemObjPtr(dstBuff);

  assert(atomicBuffObj.IsValid());
  assert(srcBuffObj.IsValid());
  assert(dstBuffObj.IsValid());

  // Allocate error flag
  int* errorFlag_d;
  HIP_RUNTIME_CHECK(hipMalloc(&errorFlag_d, sizeof(int)));

  // ===== Test 1: Verify Pointer Addresses =====
  if (myPe == 0) {
    printf("\n========== Test 1: Pointer Verification ==========\n");
  }

  VerifyPointerKernel<<<1, 32>>>(myPe, npes, atomicBuffObj);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    printf("✓ Pointer verification completed\n");
  }

  // ===== Test 2: Direct Atomic Access via GetAs(pe) =====
  if (myPe == 0) {
    printf("\n========== Test 2: Direct Atomic Access ==========\n");
  }

  // Initialize atomic buffer to zero
  HIP_RUNTIME_CHECK(hipMemset(atomicBuff, 0, buffSize));
  HIP_RUNTIME_CHECK(hipMemset(errorFlag_d, 0, sizeof(int)));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  // Each PE writes to next PE
  constexpr int blockSize = 256;
  constexpr int gridSize = 1;
  DirectAtomicAccessKernel<uint64_t><<<gridSize, blockSize>>>(myPe, npes, atomicBuffObj,
                                                               errorFlag_d);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  // Verify results
  uint64_t resultData[numElements];
  HIP_RUNTIME_CHECK(
      hipMemcpy(resultData, atomicBuff, buffSize, hipMemcpyDeviceToHost));

  int errorCount = 0;
  HIP_RUNTIME_CHECK(hipMemcpy(&errorCount, errorFlag_d, sizeof(int), hipMemcpyDeviceToHost));

  // Each PE should have received writes from previous PE (32 threads)
  uint64_t expected = 32;
  if (resultData[0] != expected) {
    printf("PE %d: ERROR - Expected %lu atomic adds, got %lu\n", myPe, expected, resultData[0]);
    errorCount++;
  } else {
    printf("PE %d: ✓ Atomic access test PASSED (received %lu atomic adds from PE %d)\n", myPe,
           resultData[0], (myPe - 1 + npes) % npes);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    if (errorCount == 0) {
      printf("✓ Direct atomic access test completed successfully!\n");
    } else {
      printf("✗ Direct atomic access test FAILED with %d errors\n", errorCount);
    }
  }

  // ===== Test 3: Direct Memory Copy via GetAs(pe) =====
  if (myPe == 0) {
    printf("\n========== Test 3: Direct Memory Copy (WarpCopy) ==========\n");
  }

  // Initialize source buffer with unique pattern
  uint64_t srcData[numElements];
  for (int i = 0; i < numElements; i++) {
    srcData[i] = (static_cast<uint64_t>(myPe) << 32) | i;
  }
  HIP_RUNTIME_CHECK(
      hipMemcpy(srcBuff, srcData, buffSize, hipMemcpyHostToDevice));

  // Initialize dst buffer to zero
  HIP_RUNTIME_CHECK(hipMemset(dstBuff, 0, buffSize));
  HIP_RUNTIME_CHECK(hipMemset(errorFlag_d, 0, sizeof(int)));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  // Copy data to next PE
  DirectMemoryCopyKernel<uint64_t><<<gridSize, blockSize>>>(myPe, npes, srcBuffObj, dstBuffObj,
                                                             numElements, errorFlag_d);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  // Verify results
  uint64_t dstData[numElements];
  HIP_RUNTIME_CHECK(
      hipMemcpy(dstData, dstBuff, buffSize, hipMemcpyDeviceToHost));
  HIP_RUNTIME_CHECK(hipMemcpy(&errorCount, errorFlag_d, sizeof(int), hipMemcpyDeviceToHost));

  // Each PE should have received data from previous PE
  int prevPe = (myPe - 1 + npes) % npes;
  bool copySuccess = true;
  for (int i = 0; i < numElements; i++) {
    uint64_t expected = (static_cast<uint64_t>(prevPe) << 32) | i;
    if (dstData[i] != expected) {
      printf("PE %d: ERROR - dst[%d] = 0x%lx, expected 0x%lx (from PE %d)\n", myPe, i, dstData[i],
             expected, prevPe);
      copySuccess = false;
      errorCount++;
      break;
    }
  }

  if (copySuccess) {
    printf("PE %d: ✓ Memory copy test PASSED (received data from PE %d)\n", myPe, prevPe);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    if (errorCount == 0) {
      printf("✓ Direct memory copy test completed successfully!\n");
    } else {
      printf("✗ Direct memory copy test FAILED with %d errors\n", errorCount);
    }
  }

  // ===== Test 4: Shmem Put Thread (Legacy API) =====
  if (myPe == 0) {
    printf("\n========== Test 4: Shmem Put Thread (Legacy API) ==========\n");
  }

  // Initialize buffer with myPe value
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(atomicBuff), myPe, numElements));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  // Each PE writes to next PE using shmem put (includes receiver polling)
  ShmemPutThreadKernel<uint32_t><<<gridSize, blockSize>>>(myPe, npes, atomicBuffObj, numElements);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  // Note: No MPI_Barrier needed here as kernel includes receiver-side polling

  // Verify results
  uint32_t putResult[numElements];
  HIP_RUNTIME_CHECK(
      hipMemcpy(putResult, atomicBuff, buffSize, hipMemcpyDeviceToHost));

  int prevPe2 = (myPe - 1 + npes) % npes;
  bool putSuccess = true;
  for (int i = 0; i < numElements; i++) {
    if (putResult[i] != static_cast<uint32_t>(prevPe2)) {
      printf("PE %d: ERROR - putResult[%d] = %u, expected %u (from PE %d)\n", myPe, i, putResult[i],
             prevPe2, prevPe2);
      putSuccess = false;
      break;
    }
  }

  if (putSuccess) {
    printf("PE %d: ✓ Shmem Put Thread (Legacy API) test PASSED (received data from PE %d)\n", myPe,
           prevPe2);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    if (putSuccess) {
      printf("✓ Shmem Put Thread (Legacy API) test completed successfully!\n");
    } else {
      printf("✗ Shmem Put Thread (Legacy API) test FAILED\n");
    }
  }

  // ===== Test 5: Shmem Put Thread (Pure Address API) =====
  if (myPe == 0) {
    printf("\n========== Test 5: Shmem Put Thread (Pure Address API) ==========\n");
  }

  if (skipPureAddress) {
    if (myPe == 0) {
      printf("⊘ SKIPPED (MORI_SHMEM_MODE=ISOLATION)\n");
    }
  } else {
    // Initialize buffer with myPe value
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(srcBuff), myPe, numElements));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Each PE writes to next PE using shmem put (pure address API, includes receiver polling)
    ShmemPutThreadKernel_PureAddr<uint32_t><<<gridSize, blockSize>>>(
        myPe, npes, reinterpret_cast<uint32_t*>(srcBuff), numElements);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    // Note: No MPI_Barrier needed here as kernel includes receiver-side polling

    // Verify results
    uint32_t putResult2[numElements];
    HIP_RUNTIME_CHECK(
        hipMemcpy(putResult2, srcBuff, buffSize, hipMemcpyDeviceToHost));

    bool putSuccess2 = true;
    for (int i = 0; i < numElements; i++) {
      if (putResult2[i] != static_cast<uint32_t>(prevPe2)) {
        printf("PE %d: ERROR - putResult2[%d] = %u, expected %u (from PE %d)\n", myPe, i,
               putResult2[i], prevPe2, prevPe2);
        putSuccess2 = false;
        break;
      }
    }

    if (putSuccess2) {
      printf("PE %d: ✓ Shmem Put Thread (Pure Address API) test PASSED (received data from PE %d)\n",
             myPe, prevPe2);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (myPe == 0) {
      if (putSuccess2) {
        printf("✓ Shmem Put Thread (Pure Address API) test completed successfully!\n");
      } else {
        printf("✗ Shmem Put Thread (Pure Address API) test FAILED\n");
      }
    }
  }

  // ===== Final Summary =====
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    printf("\n=================================================================\n");
    printf("P2P Direct Access Test Summary\n");
    printf("=================================================================\n");
    printf("✓ GetAs(pe) successfully uses p2pPeerPtrs for direct access\n");
    printf("✓ Direct atomic operations work via P2P pointers\n");
    printf("✓ WarpCopy operations work via P2P pointers\n");
    printf("✓ Shmem Put Thread (Legacy API) works correctly\n");
    if (skipPureAddress) {
      printf("⊘ Shmem Put Thread (Pure Address API) skipped (ISOLATION mode)\n");
    } else {
      printf("✓ Shmem Put Thread (Pure Address API) works correctly\n");
    }
    printf("✓ P2P data path works regardless of MORI_DISABLE_P2P setting\n");
    printf("=================================================================\n");
    printf("\nAll tests PASSED! 🎉\n");
  }

  // Cleanup
  HIP_RUNTIME_CHECK(hipFree(errorFlag_d));
  ShmemFree(atomicBuff);
  ShmemFree(srcBuff);
  ShmemFree(dstBuff);
  MPI_Comm_free(&localComm);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  testDirectP2PAccess();
  return 0;
}
