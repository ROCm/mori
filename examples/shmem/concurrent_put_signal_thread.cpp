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
__global__ void ConcurrentPutSignalThreadKernelAdd(int myPe, const SymmMemObjPtr dataObj,
                                                   const SymmMemObjPtr signalObj) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadOffset = globalTid * sizeof(uint32_t);

  if (myPe == sendPe) {
    RdmaMemoryRegion source = dataObj->GetRdmaMemoryRegion(myPe);

    // Test onlyOneSignal=true with AMO_ADD: only leader thread signals
    ShmemPutMemNbiSignalThread<true>(dataObj, threadOffset, source, threadOffset,
                                     sizeof(uint32_t), signalObj, 0, 1, atomicType::AMO_ADD,
                                     recvPe, 0);
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
    uint32_t receivedData = atomicAdd(reinterpret_cast<uint32_t*>(dataObj->localPtr) + globalTid, 0);
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
    RdmaMemoryRegion source = dataObj->GetRdmaMemoryRegion(myPe);

    // Test onlyOneSignal=true with AMO_SET: each warp sets its own signal slot
    // Use warp ID as offset to avoid overwriting other warps' signals
    ShmemPutMemNbiSignalThread<true>(dataObj, threadOffset, source, threadOffset,
                                     sizeof(uint32_t), signalObj, globalWarpId * sizeof(uint64_t), 
                                     MAGIC_VALUE, atomicType::AMO_SET, recvPe, 0);
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
    uint32_t receivedData = atomicAdd(reinterpret_cast<uint32_t*>(dataObj->localPtr) + globalTid, 0);
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

void ConcurrentPutSignalThread() {
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

  // Allocate data buffer
  int numEle = threadNum * blockNum;
  int buffSize = numEle * sizeof(uint32_t);

  if (myPe == 1) {
    printf("=================================================================\n");
    printf("Testing both Legacy and Pure Address APIs (Put with Signal)\n");
    printf("=================================================================\n");
  }

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
    HIP_RUNTIME_CHECK(hipMemcpy(&signalValue, signalBuff1, sizeof(uint64_t), hipMemcpyDeviceToHost));
    uint64_t expectedSignals = (threadNum * blockNum + warpSize - 1) / warpSize;  // One signal per warp
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
    HIP_RUNTIME_CHECK(hipMemcpy(&signalValue, signalBuff2, sizeof(uint64_t), hipMemcpyDeviceToHost));
    uint64_t expectedSignals = (threadNum * blockNum + warpSize - 1) / warpSize;
    printf("✓ Pure Address API AMO_ADD test PASSED! Signal counter: %lu (expected: %lu), Data: %s\n", 
           signalValue, expectedSignals, success ? "OK" : "FAILED");
  }

  // Cleanup Test 2
  ShmemFree(dataBuff2);
  ShmemFree(signalBuff2);

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
    printf("Running legacy API test with AMO_SET (%d warps, %d signals)...\n", 
           totalWarps, totalWarps);
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
    HIP_RUNTIME_CHECK(hipMemcpy(signalValues.data(), signalBuff3, 
                                totalWarps * sizeof(uint64_t), hipMemcpyDeviceToHost));
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

  void* dataBuff4 = ShmemMalloc(buffSize);
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(dataBuff4), myPe, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // Allocate signal buffer for all warps (one uint64_t per warp)
  void* signalBuff4 = ShmemMalloc(totalWarps * sizeof(uint64_t));
  HIP_RUNTIME_CHECK(hipMemset(signalBuff4, 0, totalWarps * sizeof(uint64_t)));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 1) {
    printf("Running pure address API test with AMO_SET (%d warps, %d signals)...\n", 
           totalWarps, totalWarps);
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
    HIP_RUNTIME_CHECK(hipMemcpy(signalValues.data(), signalBuff4, 
                                totalWarps * sizeof(uint64_t), hipMemcpyDeviceToHost));
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

  if (myPe == 1) {
    printf("\n=================================================================\n");
    printf("All tests completed successfully!\n");
    printf("=================================================================\n");
  }

  // Finalize
  ShmemFree(dataBuff4);
  ShmemFree(signalBuff4);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  ConcurrentPutSignalThread();
  return 0;
}