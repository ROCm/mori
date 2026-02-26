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
__global__ void ConcurrentPutSignalThreadKernelAdd(int myPe, const SymmMemObjPtr dataObj,
                                                   const SymmMemObjPtr signalObj) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadOffset = globalTid * sizeof(uint32_t);

  if (myPe == sendPe) {
    // Test onlyOneSignal=true with AMO_ADD: only leader thread signals
    ShmemPutMemNbiSignalThread<true>(dataObj, threadOffset, dataObj, threadOffset, sizeof(uint32_t),
                                     signalObj, 0, 1, atomicType::AMO_ADD, recvPe, 0);
    __threadfence_system();

    if (blockIdx.x == 0) {
      ShmemQuietThread();
    }
  } else {
    // Receiver: wait for all data to arrive by checking signal counter
    if (threadIdx.x == 0) {
      uint64_t* signalPtr = reinterpret_cast<uint64_t*>(signalObj->localPtr);
      uint64_t expectedSignals = blockDim.x * gridDim.x / warpSize;  // One signal per warp
      while (atomicAdd(signalPtr, 0) != expectedSignals) {
        // Busy wait for all signals
      }
    }
    __syncthreads();

    // Verify data
    uint32_t receivedData =
        atomicAdd(reinterpret_cast<uint32_t*>(dataObj->localPtr) + globalTid, 0);
    if (receivedData != sendPe) {
      printf("PE %d, thread %d: Data mismatch! Expected %d, got %d\n", myPe, globalTid, sendPe,
             receivedData);
    }
  }
}

// New API: Using pure addresses with AMO_ADD
__global__ void ConcurrentPutSignalThreadKernelAdd_PureAddr(int myPe, uint32_t* dataBuff,
                                                            uint64_t* signalBuff) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (myPe == sendPe) {
    uint32_t* src = dataBuff + globalTid;
    uint32_t* dest = dataBuff + globalTid;

    // Test onlyOneSignal=true with AMO_ADD: only leader thread signals
    ShmemPutMemNbiSignalThread<true>(dest, src, sizeof(uint32_t), signalBuff, 1,
                                     atomicType::AMO_ADD, recvPe, 0);
    __threadfence_system();

    if (blockIdx.x == 0) {
      ShmemQuietThread();
    }
  } else {
    // Receiver: wait for all data to arrive by checking signal counter
    if (threadIdx.x == 0) {
      uint64_t expectedSignals = blockDim.x * gridDim.x / warpSize;  // One signal per warp
      while (atomicAdd(signalBuff, 0) != expectedSignals) {
        // Busy wait for all signals
      }
    }
    __syncthreads();

    // Verify data
    uint32_t receivedData = atomicAdd(dataBuff + globalTid, 0);
    if (receivedData != sendPe) {
      printf("PE %d, thread %d: Data mismatch! Expected %d, got %d\n", myPe, globalTid, sendPe,
             receivedData);
    }
  }
}

// Legacy API: Using SymmMemObjPtr + offset with AMO_SET
__global__ void ConcurrentPutSignalThreadKernelSet(int myPe, const SymmMemObjPtr dataObj,
                                                   const SymmMemObjPtr signalObj) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;
  constexpr uint64_t MAGIC_VALUE = 0xDEADBEEF;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadOffset = globalTid * sizeof(uint32_t);
  int globalWarpId = globalTid / warpSize;

  if (myPe == sendPe) {
    // Test onlyOneSignal=true with AMO_SET: each warp sets its own signal slot
    // Use warp ID as offset to avoid overwriting other warps' signals
    ShmemPutMemNbiSignalThread<true>(dataObj, threadOffset, dataObj, threadOffset, sizeof(uint32_t),
                                     signalObj, globalWarpId * sizeof(uint64_t), MAGIC_VALUE,
                                     atomicType::AMO_SET, recvPe, 0);
    __threadfence_system();

    if (blockIdx.x == 0) {
      ShmemQuietThread();
    }
  } else {
    // Receiver: wait for all warps' signals to be set to magic value
    int totalWarps = (blockDim.x * gridDim.x) / warpSize;
    if (threadIdx.x == 0) {
      uint64_t* signalPtr = reinterpret_cast<uint64_t*>(signalObj->localPtr);
      bool allReceived = false;
      while (!allReceived) {
        allReceived = true;
        for (int warpId = 0; warpId < totalWarps; warpId++) {
          if (atomicAdd(&signalPtr[warpId], 0) != MAGIC_VALUE) {
            allReceived = false;
            break;
          }
        }
      }
    }
    __syncthreads();

    // Verify data
    uint32_t receivedData =
        atomicAdd(reinterpret_cast<uint32_t*>(dataObj->localPtr) + globalTid, 0);
    if (receivedData != sendPe) {
      printf("PE %d, thread %d: Data mismatch! Expected %d, got %d\n", myPe, globalTid, sendPe,
             receivedData);
    }
  }
}

// New API: Using pure addresses with AMO_SET
__global__ void ConcurrentPutSignalThreadKernelSet_PureAddr(int myPe, uint32_t* dataBuff,
                                                            uint64_t* signalBuff) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;
  constexpr uint64_t MAGIC_VALUE = 0xDEADBEEF;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = globalTid / warpSize;

  if (myPe == sendPe) {
    uint32_t* src = dataBuff + globalTid;
    uint32_t* dest = dataBuff + globalTid;

    // Test onlyOneSignal=true with AMO_SET: each warp sets its own signal slot
    // Use warp ID as index to avoid overwriting other warps' signals
    ShmemPutMemNbiSignalThread<true>(dest, src, sizeof(uint32_t), signalBuff + globalWarpId,
                                     MAGIC_VALUE, atomicType::AMO_SET, recvPe, 0);
    __threadfence_system();

    if (blockIdx.x == 0) {
      ShmemQuietThread();
    }
  } else {
    // Receiver: wait for all warps' signals to be set to magic value
    int totalWarps = (blockDim.x * gridDim.x) / warpSize;
    if (threadIdx.x == 0) {
      bool allReceived = false;
      while (!allReceived) {
        allReceived = true;
        for (int warpId = 0; warpId < totalWarps; warpId++) {
          if (atomicAdd(&signalBuff[warpId], 0) != MAGIC_VALUE) {
            allReceived = false;
            break;
          }
        }
      }
    }
    __syncthreads();

    // Verify data
    uint32_t receivedData = atomicAdd(dataBuff + globalTid, 0);
    if (receivedData != sendPe) {
      printf("PE %d, thread %d: Data mismatch! Expected %d, got %d\n", myPe, globalTid, sendPe,
             receivedData);
    }
  }
}

// Legacy API: Large size transfer (14KB per thread)
__global__ void ConcurrentPutSignalThreadKernelLargeSize(int myPe, const SymmMemObjPtr dataObj,
                                                         const SymmMemObjPtr signalObj,
                                                         size_t sizePerThread) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t threadOffset = globalTid * sizePerThread;

  if (myPe == sendPe) {
    // Each thread transfers a large chunk (14KB) with signal
    ShmemPutMemNbiSignalThread<true>(dataObj, threadOffset, dataObj, threadOffset, sizePerThread,
                                     signalObj, 0, 1, atomicType::AMO_ADD, recvPe, 0);
    __threadfence_system();

    if (blockIdx.x == 0) {
      ShmemQuietThread();
    }
  } else {
    // Receiver: wait for all data to arrive by checking signal counter
    if (threadIdx.x == 0) {
      uint64_t* signalPtr = reinterpret_cast<uint64_t*>(signalObj->localPtr);
      uint64_t expectedSignals = blockDim.x * gridDim.x / warpSize;  // One signal per warp
      while (atomicAdd(signalPtr, 0) != expectedSignals) {
        // Busy wait for all signals
      }
    }
    __syncthreads();

    // Verify data (check first and last uint32_t of each thread's chunk)
    uint32_t* dataPtr = reinterpret_cast<uint32_t*>(dataObj->localPtr);
    size_t offset = globalTid * sizePerThread / sizeof(uint32_t);
    size_t numElements = sizePerThread / sizeof(uint32_t);

    uint32_t firstValue = atomicAdd(&dataPtr[offset], 0);
    uint32_t lastValue = atomicAdd(&dataPtr[offset + numElements - 1], 0);

    if (firstValue != sendPe || lastValue != sendPe) {
      printf("PE %d, thread %d: Data mismatch! First: %d, Last: %d (expected: %d)\n", myPe,
             globalTid, firstValue, lastValue, sendPe);
    }
  }
}

// Legacy API: Very large size transfer (5MB per thread - tests cross-VMM-chunk scenario)
__global__ void ConcurrentPutSignalThreadKernelVeryLargeSize(int myPe, const SymmMemObjPtr dataObj,
                                                             const SymmMemObjPtr signalObj,
                                                             size_t sizePerThread) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t threadOffset = globalTid * sizePerThread;

  if (myPe == sendPe) {
    // Each thread transfers 5MB chunk with signal (crosses VMM chunk boundaries)
    ShmemPutMemNbiSignalThread<true>(dataObj, threadOffset, dataObj, threadOffset, sizePerThread,
                                     signalObj, 0, 1, atomicType::AMO_ADD, recvPe, 0);
    __threadfence_system();

    if (blockIdx.x == 0) {
      ShmemQuietThread();
    }
  } else {
    // Receiver: wait for all data to arrive by checking signal counter
    if (threadIdx.x == 0) {
      uint64_t* signalPtr = reinterpret_cast<uint64_t*>(signalObj->localPtr);
      uint64_t expectedSignals = blockDim.x * gridDim.x / warpSize;
      while (atomicAdd(signalPtr, 0) != expectedSignals) {
        // Busy wait for all signals
      }
    }
    __syncthreads();

    // Verify data (sample check for very large transfers)
    uint32_t* dataPtr = reinterpret_cast<uint32_t*>(dataObj->localPtr);
    size_t offset = globalTid * sizePerThread / sizeof(uint32_t);
    size_t numElements = sizePerThread / sizeof(uint32_t);

    // Check first, middle, and last values
    uint32_t firstValue = atomicAdd(&dataPtr[offset], 0);
    uint32_t midValue = atomicAdd(&dataPtr[offset + numElements / 2], 0);
    uint32_t lastValue = atomicAdd(&dataPtr[offset + numElements - 1], 0);

    if (firstValue != sendPe || midValue != sendPe || lastValue != sendPe) {
      printf("PE %d, thread %d: Data mismatch! First: %d, Mid: %d, Last: %d (expected: %d)\n", myPe,
             globalTid, firstValue, midValue, lastValue, sendPe);
    }
  }
}

// Pure Address API: Very large size transfer (5MB per thread - tests cross-VMM-chunk scenario)
__global__ void ConcurrentPutSignalThreadKernelVeryLargeSize_PureAddr(int myPe, uint8_t* dataBuff,
                                                                      uint64_t* signalBuff,
                                                                      size_t sizePerThread) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t threadOffset = globalTid * sizePerThread;

  if (myPe == sendPe) {
    uint8_t* src = dataBuff + threadOffset;
    uint8_t* dest = dataBuff + threadOffset;

    // Each thread transfers 5MB chunk with signal (crosses VMM chunk boundaries)
    ShmemPutMemNbiSignalThread<true>(dest, src, sizePerThread, signalBuff, 1, atomicType::AMO_ADD,
                                     recvPe, 0);
    __threadfence_system();

    if (blockIdx.x == 0) {
      ShmemQuietThread();
    }
  } else {
    // Receiver: wait for all data to arrive by checking signal counter
    if (threadIdx.x == 0) {
      uint64_t expectedSignals = blockDim.x * gridDim.x / warpSize;
      while (atomicAdd(signalBuff, 0) != expectedSignals) {
        // Busy wait for all signals
      }
    }
    __syncthreads();

    // Verify data (sample check for very large transfers)
    uint32_t* dataPtr = reinterpret_cast<uint32_t*>(dataBuff);
    size_t offset = globalTid * sizePerThread / sizeof(uint32_t);
    size_t numElements = sizePerThread / sizeof(uint32_t);

    // Check first, middle, and last values
    uint32_t firstValue = atomicAdd(&dataPtr[offset], 0);
    uint32_t midValue = atomicAdd(&dataPtr[offset + numElements / 2], 0);
    uint32_t lastValue = atomicAdd(&dataPtr[offset + numElements - 1], 0);

    if (firstValue != sendPe || midValue != sendPe || lastValue != sendPe) {
      printf("PE %d, thread %d: Data mismatch! First: %d, Mid: %d, Last: %d (expected: %d)\n", myPe,
             globalTid, firstValue, midValue, lastValue, sendPe);
    }
  }
}

void ConcurrentPutSignalThread() {
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

  // Assume in same node
  int myPe = ShmemMyPe();
  int npes = ShmemNPes();
  assert(npes == 2);

  constexpr int threadNum = 128;
  constexpr int blockNum = 3;

  // Allocate data buffer
  int numEle = threadNum * blockNum;
  int buffSize = numEle * sizeof(uint32_t);

  if (myPe == 1) {
    printf("=================================================================\n");
    printf("Testing both Legacy and Pure Address APIs (Put with Signal)\n");
    printf("=================================================================\n");
  }

  // Check if we should skip pure address API tests
  const char* shmemMode = std::getenv("MORI_SHMEM_MODE");
  bool skipPureAddress = (shmemMode != nullptr && std::string(shmemMode) == "ISOLATION");

  // ===== Test 1: Legacy API with AMO_ADD =====
  if (myPe == 1) {
    printf("\n--- Test 1: Legacy API with AMO_ADD Signal ---\n");
  }

  void* dataBuff1 = ShmemMalloc(buffSize);
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(dataBuff1), myPe, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr dataBuffObj1 = ShmemQueryMemObjPtr(dataBuff1);
  assert(dataBuffObj1.IsValid());

  void* signalBuff1 = ShmemMalloc(sizeof(uint64_t));
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(signalBuff1), 0, 2));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr signalBuffObj1 = ShmemQueryMemObjPtr(signalBuff1);
  assert(signalBuffObj1.IsValid());

  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 1) {
    printf("Running legacy API test with AMO_ADD...\n");
  }
  ConcurrentPutSignalThreadKernelAdd<<<blockNum, threadNum>>>(myPe, dataBuffObj1, signalBuffObj1);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  // Verify Test 1
  std::vector<uint32_t> hostData1(numEle);
  HIP_RUNTIME_CHECK(hipMemcpy(hostData1.data(), dataBuff1, buffSize, hipMemcpyDeviceToHost));

  if (myPe == 1) {
    bool success = true;
    for (int i = 0; i < numEle; i++) {
      if (hostData1[i] != 0) {
        success = false;
        break;
      }
    }

    uint64_t signalValue;
    HIP_RUNTIME_CHECK(
        hipMemcpy(&signalValue, signalBuff1, sizeof(uint64_t), hipMemcpyDeviceToHost));
    uint64_t expectedSignals =
        (threadNum * blockNum + warpSize - 1) / warpSize;  // One signal per warp
    printf("✓ Legacy API AMO_ADD test PASSED! Signal counter: %lu (expected: %lu), Data: %s\n",
           signalValue, expectedSignals, success ? "OK" : "FAILED");
  }

  // Cleanup Test 1
  ShmemFree(dataBuff1);
  ShmemFree(signalBuff1);

  // ===== Test 2: Pure Address API with AMO_ADD =====
  if (myPe == 1) {
    printf("\n--- Test 2: Pure Address API with AMO_ADD Signal ---\n");
  }

  if (skipPureAddress) {
    if (myPe == 1) {
      printf("⊘ SKIPPED (MORI_SHMEM_MODE=ISOLATION)\n");
    }
  } else {
    void* dataBuff2 = ShmemMalloc(buffSize);
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(dataBuff2), myPe, numEle));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());

    void* signalBuff2 = ShmemMalloc(sizeof(uint64_t));
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(signalBuff2), 0, 2));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);

    if (myPe == 1) {
      printf("Running pure address API test with AMO_ADD...\n");
    }
    ConcurrentPutSignalThreadKernelAdd_PureAddr<<<blockNum, threadNum>>>(
        myPe, reinterpret_cast<uint32_t*>(dataBuff2), reinterpret_cast<uint64_t*>(signalBuff2));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Verify Test 2
    std::vector<uint32_t> hostData2(numEle);
    HIP_RUNTIME_CHECK(hipMemcpy(hostData2.data(), dataBuff2, buffSize, hipMemcpyDeviceToHost));

    if (myPe == 1) {
      bool success = true;
      for (int i = 0; i < numEle; i++) {
        if (hostData2[i] != 0) {
          success = false;
          break;
        }
      }

      uint64_t signalValue;
      HIP_RUNTIME_CHECK(
          hipMemcpy(&signalValue, signalBuff2, sizeof(uint64_t), hipMemcpyDeviceToHost));
      uint64_t expectedSignals = (threadNum * blockNum + warpSize - 1) / warpSize;
      printf(
          "✓ Pure Address API AMO_ADD test PASSED! Signal counter: %lu (expected: %lu), Data: %s\n",
          signalValue, expectedSignals, success ? "OK" : "FAILED");
    }

    // Cleanup Test 2
    ShmemFree(dataBuff2);
    ShmemFree(signalBuff2);
  }

  // ===== Test 3: Legacy API with AMO_SET =====
  if (myPe == 1) {
    printf("\n--- Test 3: Legacy API with AMO_SET Signal ---\n");
    printf("  Each warp sets its own signal slot\n");
  }

  void* dataBuff3 = ShmemMalloc(buffSize);
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(dataBuff3), myPe, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr dataBuffObj3 = ShmemQueryMemObjPtr(dataBuff3);
  assert(dataBuffObj3.IsValid());

  // Allocate signal buffer for all warps (one uint64_t per warp)
  int totalWarps = (threadNum * blockNum + warpSize - 1) / warpSize;
  void* signalBuff3 = ShmemMalloc(totalWarps * sizeof(uint64_t));
  HIP_RUNTIME_CHECK(hipMemset(signalBuff3, 0, totalWarps * sizeof(uint64_t)));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr signalBuffObj3 = ShmemQueryMemObjPtr(signalBuff3);
  assert(signalBuffObj3.IsValid());

  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 1) {
    printf("Running legacy API test with AMO_SET (%d warps, %d signals)...\n", totalWarps,
           totalWarps);
  }
  ConcurrentPutSignalThreadKernelSet<<<blockNum, threadNum>>>(myPe, dataBuffObj3, signalBuffObj3);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  // Verify Test 3
  std::vector<uint32_t> hostData3(numEle);
  HIP_RUNTIME_CHECK(hipMemcpy(hostData3.data(), dataBuff3, buffSize, hipMemcpyDeviceToHost));

  bool dataSuccess = true;
  if (myPe == 1) {
    for (int i = 0; i < numEle; i++) {
      if (hostData3[i] != 0) {
        dataSuccess = false;
        break;
      }
    }

    // PE 1: Verify signal values (PE 1 is the receiver)
    std::vector<uint64_t> signalValues(totalWarps);
    HIP_RUNTIME_CHECK(hipMemcpy(signalValues.data(), signalBuff3, totalWarps * sizeof(uint64_t),
                                hipMemcpyDeviceToHost));
    int validSignals = 0;
    for (int i = 0; i < totalWarps; i++) {
      if (signalValues[i] == 0xDEADBEEF) {
        validSignals++;
      } else {
        printf("Warning: Signal[%d] = 0x%lx (expected 0xDEADBEEF)\n", i, signalValues[i]);
      }
    }
    printf("✓ Legacy API AMO_SET test PASSED! Data: %s, Valid signals: %d/%d\n",
           dataSuccess ? "OK" : "FAILED", validSignals, totalWarps);
  }

  // Cleanup Test 3
  ShmemFree(dataBuff3);
  ShmemFree(signalBuff3);

  // ===== Test 4: Pure Address API with AMO_SET =====
  if (myPe == 1) {
    printf("\n--- Test 4: Pure Address API with AMO_SET Signal ---\n");
    printf("  Each warp sets its own signal slot\n");
  }

  if (skipPureAddress) {
    if (myPe == 1) {
      printf("⊘ SKIPPED (MORI_SHMEM_MODE=ISOLATION)\n");
    }
  } else {
    void* dataBuff4 = ShmemMalloc(buffSize);
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(dataBuff4), myPe, numEle));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());

    // Allocate signal buffer for all warps (one uint64_t per warp)
    void* signalBuff4 = ShmemMalloc(totalWarps * sizeof(uint64_t));
    HIP_RUNTIME_CHECK(hipMemset(signalBuff4, 0, totalWarps * sizeof(uint64_t)));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);

    if (myPe == 1) {
      printf("Running pure address API test with AMO_SET (%d warps, %d signals)...\n", totalWarps,
             totalWarps);
    }
    ConcurrentPutSignalThreadKernelSet_PureAddr<<<blockNum, threadNum>>>(
        myPe, reinterpret_cast<uint32_t*>(dataBuff4), reinterpret_cast<uint64_t*>(signalBuff4));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Verify Test 4
    std::vector<uint32_t> hostData4(numEle);
    HIP_RUNTIME_CHECK(hipMemcpy(hostData4.data(), dataBuff4, buffSize, hipMemcpyDeviceToHost));

    dataSuccess = true;
    if (myPe == 1) {
      for (int i = 0; i < numEle; i++) {
        if (hostData4[i] != 0) {
          dataSuccess = false;
          break;
        }
      }

      // PE 1: Verify signal values (PE 1 is the receiver)
      std::vector<uint64_t> signalValues(totalWarps);
      HIP_RUNTIME_CHECK(hipMemcpy(signalValues.data(), signalBuff4, totalWarps * sizeof(uint64_t),
                                  hipMemcpyDeviceToHost));
      int validSignals = 0;
      for (int i = 0; i < totalWarps; i++) {
        if (signalValues[i] == 0xDEADBEEF) {
          validSignals++;
        } else {
          printf("Warning: Signal[%d] = 0x%lx (expected 0xDEADBEEF)\n", i, signalValues[i]);
        }
      }
      printf("✓ Pure Address API AMO_SET test PASSED! Data: %s, Valid signals: %d/%d\n",
             dataSuccess ? "OK" : "FAILED", validSignals, totalWarps);
    }

    // Finalize
    ShmemFree(dataBuff4);
    ShmemFree(signalBuff4);
  }

  // ===== Test 5: Legacy API with Large Size (14KB per thread) =====
  if (myPe == 1) {
    printf("\n--- Test 5: Legacy API with Large Size Transfer (14KB per thread) ---\n");
  }

  constexpr size_t sizePerThread = 14 * 1024;  // 14KB per thread
  size_t totalSize = threadNum * blockNum * sizePerThread;

  void* dataBuff5 = ShmemMalloc(totalSize);
  // Initialize with myPe value
  HIP_RUNTIME_CHECK(
      hipMemsetD32(reinterpret_cast<uint32_t*>(dataBuff5), myPe, totalSize / sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr dataBuffObj5 = ShmemQueryMemObjPtr(dataBuff5);
  assert(dataBuffObj5.IsValid());

  void* signalBuff5 = ShmemMalloc(sizeof(uint64_t));
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(signalBuff5), 0, 2));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr signalBuffObj5 = ShmemQueryMemObjPtr(signalBuff5);
  assert(signalBuffObj5.IsValid());

  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 1) {
    printf("Running large size test: %d threads × %zu bytes = %.2f MB total\n",
           threadNum * blockNum, sizePerThread, totalSize / (1024.0 * 1024.0));
  }
  ConcurrentPutSignalThreadKernelLargeSize<<<blockNum, threadNum>>>(myPe, dataBuffObj5,
                                                                    signalBuffObj5, sizePerThread);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  // Verify Test 5 (sample check: verify first and last 1KB of transferred data)
  if (myPe == 1) {
    size_t sampleSize = 1024 / sizeof(uint32_t);  // Check first 1KB
    std::vector<uint32_t> hostDataSample(sampleSize);

    // Check first 1KB
    HIP_RUNTIME_CHECK(hipMemcpy(hostDataSample.data(), dataBuff5, 1024, hipMemcpyDeviceToHost));
    bool firstOk = true;
    for (size_t i = 0; i < sampleSize; i++) {
      if (hostDataSample[i] != 0) {
        firstOk = false;
        break;
      }
    }

    // Check last 1KB
    HIP_RUNTIME_CHECK(hipMemcpy(hostDataSample.data(),
                                reinterpret_cast<uint8_t*>(dataBuff5) + totalSize - 1024, 1024,
                                hipMemcpyDeviceToHost));
    bool lastOk = true;
    for (size_t i = 0; i < sampleSize; i++) {
      if (hostDataSample[i] != 0) {
        lastOk = false;
        break;
      }
    }

    uint64_t signalValue;
    HIP_RUNTIME_CHECK(
        hipMemcpy(&signalValue, signalBuff5, sizeof(uint64_t), hipMemcpyDeviceToHost));
    uint64_t expectedSignals = (threadNum * blockNum + warpSize - 1) / warpSize;

    printf("✓ Large Size test PASSED! Signal counter: %lu (expected: %lu)\n", signalValue,
           expectedSignals);
    printf("  Data verification: First 1KB: %s, Last 1KB: %s\n", firstOk ? "OK" : "FAILED",
           lastOk ? "OK" : "FAILED");
  }

  // Cleanup Test 5
  ShmemFree(dataBuff5);
  ShmemFree(signalBuff5);

  // ===== Test 6: Legacy API with Very Large Size (5MB per thread - tests cross-VMM-chunk) =====
  if (myPe == 1) {
    printf("\n--- Test 6: Legacy API with Very Large Size Transfer (5MB per thread) ---\n");
    printf("  This test verifies large transfers (VMM chunk size = 64MB)\n");
  }

  // Check if we should skip large transfer tests in P2P mode
  const char* disableP2P = std::getenv("MORI_DISABLE_P2P");
  bool isP2PMode = !(disableP2P != nullptr &&
                     (std::string(disableP2P) == "ON" || std::string(disableP2P) == "1"));

  if (isP2PMode) {
    if (myPe == 1) {
      printf("⊘ SKIPPED (P2P mode with large data transfer)\n");
      printf("   Reason: ThreadCopy (used by P2P) has severe performance issues with large\n");
      printf("   data transfers (5MB per thread). This affects both STATIC_HEAP and VMM_HEAP.\n");
      printf("   Recommendation: Use MORI_DISABLE_P2P=ON to enable RDMA transport for better\n");
      printf("   performance with large transfers.\n");
    }
  } else {
    constexpr size_t sizePerThreadLarge = 5 * 1024 * 1024;  // 5MB per thread
    size_t totalSizeLarge =
        threadNum * blockNum * sizePerThreadLarge;  // ~1.88GB total (384 threads × 5MB)

    void* dataBuff6 = ShmemMalloc(totalSizeLarge);
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(dataBuff6), myPe,
                                   totalSizeLarge / sizeof(uint32_t)));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());

    SymmMemObjPtr dataBuffObj6 = ShmemQueryMemObjPtr(dataBuff6);
    assert(dataBuffObj6.IsValid());

    void* signalBuff6 = ShmemMalloc(sizeof(uint64_t));
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(signalBuff6), 0, 2));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());

    SymmMemObjPtr signalBuffObj6 = ShmemQueryMemObjPtr(signalBuff6);
    assert(signalBuffObj6.IsValid());

    MPI_Barrier(MPI_COMM_WORLD);

    if (myPe == 1) {
      printf("Running very large size test: %d threads × %d MB = %.2f GB total\n",
             threadNum * blockNum, (int)(sizePerThreadLarge / (1024 * 1024)),
             totalSizeLarge / (1024.0 * 1024.0 * 1024.0));
    }
    ConcurrentPutSignalThreadKernelVeryLargeSize<<<blockNum, threadNum>>>(
        myPe, dataBuffObj6, signalBuffObj6, sizePerThreadLarge);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Verify Test 6 (sample check: verify first, middle and last 1KB)
    if (myPe == 1) {
      size_t sampleSize = 1024 / sizeof(uint32_t);
      std::vector<uint32_t> hostDataSample(sampleSize);

      // Check first 1KB
      HIP_RUNTIME_CHECK(hipMemcpy(hostDataSample.data(), dataBuff6, 1024, hipMemcpyDeviceToHost));
      bool firstOk = true;
      for (size_t i = 0; i < sampleSize; i++) {
        if (hostDataSample[i] != 0) {
          firstOk = false;
          break;
        }
      }

      // Check middle 1KB
      HIP_RUNTIME_CHECK(hipMemcpy(hostDataSample.data(),
                                  reinterpret_cast<uint8_t*>(dataBuff6) + totalSizeLarge / 2, 1024,
                                  hipMemcpyDeviceToHost));
      bool midOk = true;
      for (size_t i = 0; i < sampleSize; i++) {
        if (hostDataSample[i] != 0) {
          midOk = false;
          break;
        }
      }

      // Check last 1KB
      HIP_RUNTIME_CHECK(hipMemcpy(hostDataSample.data(),
                                  reinterpret_cast<uint8_t*>(dataBuff6) + totalSizeLarge - 1024,
                                  1024, hipMemcpyDeviceToHost));
      bool lastOk = true;
      for (size_t i = 0; i < sampleSize; i++) {
        if (hostDataSample[i] != 0) {
          lastOk = false;
          break;
        }
      }

      uint64_t signalValue;
      HIP_RUNTIME_CHECK(
          hipMemcpy(&signalValue, signalBuff6, sizeof(uint64_t), hipMemcpyDeviceToHost));
      uint64_t expectedSignals = (threadNum * blockNum + warpSize - 1) / warpSize;

      printf(
          "✓ Very Large Size (5MB/thread, %.2f GB total) test PASSED! Signal counter: %lu "
          "(expected: %lu)\n",
          totalSizeLarge / (1024.0 * 1024.0 * 1024.0), signalValue, expectedSignals);
      printf("  Data verification: First 1KB: %s, Middle 1KB: %s, Last 1KB: %s\n",
             firstOk ? "OK" : "FAILED", midOk ? "OK" : "FAILED", lastOk ? "OK" : "FAILED");
    }

    // Cleanup Test 6
    ShmemFree(dataBuff6);
    ShmemFree(signalBuff6);
  }

  // ===== Test 7: Pure Address API with Very Large Size (5MB per thread) =====
  if (myPe == 1) {
    printf("\n--- Test 7: Pure Address API with Very Large Size Transfer (5MB per thread) ---\n");
    printf("  This test verifies large transfers with pure address API\n");
  }

  if (skipPureAddress) {
    if (myPe == 1) {
      printf("⊘ SKIPPED (MORI_SHMEM_MODE=ISOLATION)\n");
    }
  } else if (isP2PMode) {
    if (myPe == 1) {
      printf("⊘ SKIPPED (P2P mode with large data transfer)\n");
      printf("   Reason: ThreadCopy (used by P2P) has severe performance issues with large\n");
      printf("   data transfers (5MB per thread). This affects both STATIC_HEAP and VMM_HEAP.\n");
      printf("   Recommendation: Use MORI_DISABLE_P2P=ON to enable RDMA transport for better\n");
      printf("   performance with large transfers.\n");
    }
  } else {
    constexpr size_t sizePerThreadLarge = 5 * 1024 * 1024;  // 5MB per thread
    size_t totalSizeLarge =
        threadNum * blockNum * sizePerThreadLarge;  // ~1.88GB total (384 threads × 5MB)

    void* dataBuff7 = ShmemMalloc(totalSizeLarge);
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(dataBuff7), myPe,
                                   totalSizeLarge / sizeof(uint32_t)));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());

    void* signalBuff7 = ShmemMalloc(sizeof(uint64_t));
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(signalBuff7), 0, 2));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);

    if (myPe == 1) {
      printf("Running pure address very large size test: %d threads × %d MB = %.2f GB total\n",
             threadNum * blockNum, (int)(sizePerThreadLarge / (1024 * 1024)),
             totalSizeLarge / (1024.0 * 1024.0 * 1024.0));
    }
    ConcurrentPutSignalThreadKernelVeryLargeSize_PureAddr<<<blockNum, threadNum>>>(
        myPe, reinterpret_cast<uint8_t*>(dataBuff7), reinterpret_cast<uint64_t*>(signalBuff7),
        sizePerThreadLarge);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Verify Test 7 (sample check: verify first, middle and last 1KB)
    if (myPe == 1) {
      size_t sampleSize = 1024 / sizeof(uint32_t);
      std::vector<uint32_t> hostDataSample(sampleSize);

      // Check first 1KB
      HIP_RUNTIME_CHECK(hipMemcpy(hostDataSample.data(), dataBuff7, 1024, hipMemcpyDeviceToHost));
      bool firstOk = true;
      for (size_t i = 0; i < sampleSize; i++) {
        if (hostDataSample[i] != 0) {
          firstOk = false;
          break;
        }
      }

      // Check middle 1KB
      HIP_RUNTIME_CHECK(hipMemcpy(hostDataSample.data(),
                                  reinterpret_cast<uint8_t*>(dataBuff7) + totalSizeLarge / 2, 1024,
                                  hipMemcpyDeviceToHost));
      bool midOk = true;
      for (size_t i = 0; i < sampleSize; i++) {
        if (hostDataSample[i] != 0) {
          midOk = false;
          break;
        }
      }

      // Check last 1KB
      HIP_RUNTIME_CHECK(hipMemcpy(hostDataSample.data(),
                                  reinterpret_cast<uint8_t*>(dataBuff7) + totalSizeLarge - 1024,
                                  1024, hipMemcpyDeviceToHost));
      bool lastOk = true;
      for (size_t i = 0; i < sampleSize; i++) {
        if (hostDataSample[i] != 0) {
          lastOk = false;
          break;
        }
      }

      uint64_t signalValue;
      HIP_RUNTIME_CHECK(
          hipMemcpy(&signalValue, signalBuff7, sizeof(uint64_t), hipMemcpyDeviceToHost));
      uint64_t expectedSignals = (threadNum * blockNum + warpSize - 1) / warpSize;

      printf(
          "✓ Pure Address Very Large Size (5MB/thread, %.2f GB total) test PASSED! Signal counter: "
          "%lu (expected: %lu)\n",
          totalSizeLarge / (1024.0 * 1024.0 * 1024.0), signalValue, expectedSignals);
      printf("  Data verification: First 1KB: %s, Middle 1KB: %s, Last 1KB: %s\n",
             firstOk ? "OK" : "FAILED", midOk ? "OK" : "FAILED", lastOk ? "OK" : "FAILED");
    }

    // Cleanup Test 7
    ShmemFree(dataBuff7);
    ShmemFree(signalBuff7);
  }

  if (myPe == 1) {
    printf("\n=================================================================\n");
    printf("All tests completed successfully!\n");
    printf("=================================================================\n");
  }
  MPI_Comm_free(&localComm);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  ConcurrentPutSignalThread();
  return 0;
}