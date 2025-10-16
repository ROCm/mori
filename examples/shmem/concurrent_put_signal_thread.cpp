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

    ShmemQuietThread();
  } else {
    // Receiver: wait for all data to arrive by checking signal counter
    if (threadIdx.x == 0) {
      uint64_t* signalPtr = reinterpret_cast<uint64_t*>(signalObj->localPtr);
      uint64_t expectedSignals = blockDim.x * gridDim.x / warpSize;  // One signal per warp
      while (atomicAdd(signalPtr, 0) != expectedSignals) {
        // Busy wait for all signals
      }
      printf("PE %d: AMO_ADD test - Received all %lu signals!\n", myPe, expectedSignals);
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

__global__ void ConcurrentPutSignalThreadKernelSet(int myPe, const SymmMemObjPtr dataObj,
                                                   const SymmMemObjPtr signalObj) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;
  constexpr uint64_t MAGIC_VALUE = 0xDEADBEEF;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadOffset = globalTid * sizeof(uint32_t);

  if (myPe == sendPe) {
    RdmaMemoryRegion source = dataObj->GetRdmaMemoryRegion(myPe);

    // Test onlyOneSignal=true with AMO_SET: only leader thread signals
    ShmemPutMemNbiSignalThread<true>(dataObj, threadOffset, source, threadOffset,
                                     sizeof(uint32_t), signalObj, 0, MAGIC_VALUE,
                                     atomicType::AMO_SET, recvPe, 0);
    __threadfence_system();

    ShmemQuietThread();
  } else {
    // Receiver: wait for signal to be set to magic value
    if (threadIdx.x == 0) {
      uint64_t* signalPtr = reinterpret_cast<uint64_t*>(signalObj->localPtr);
      while (atomicAdd(signalPtr, 0) != MAGIC_VALUE) {
        // Busy wait for signal
      }
      printf("PE %d: AMO_SET test - Received magic signal value 0x%lx!\n", myPe, MAGIC_VALUE);
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

  void* dataBuff = ShmemMalloc(buffSize);
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(dataBuff), myPe, numEle));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr dataBuffObj = ShmemQueryMemObjPtr(dataBuff);
  assert(dataBuffObj.IsValid());

  // Allocate signal buffer
  void* signalBuff = ShmemMalloc(sizeof(uint64_t));
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(signalBuff), 0, 2));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  SymmMemObjPtr signalBuffObj = ShmemQueryMemObjPtr(signalBuff);
  assert(signalBuffObj.IsValid());

  MPI_Barrier(MPI_COMM_WORLD);

  // Test 1: AMO_ADD signal operation
  if (myPe == 0) {
    printf("\n=== Test 1: PutMemNbi with Signal (AMO_ADD) ===\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  ConcurrentPutSignalThreadKernelAdd<<<blockNum, threadNum>>>(myPe, dataBuffObj, signalBuffObj);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    printf("Test 1 completed successfully!\n");
  }

  // Reset buffers for next test
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(dataBuff), myPe, numEle));
  HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(signalBuff), 0, 2));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  // Test 2: AMO_SET signal operation
  if (myPe == 0) {
    printf("\n=== Test 2: PutMemNbi with Signal (AMO_SET) ===\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  ConcurrentPutSignalThreadKernelSet<<<blockNum, threadNum>>>(myPe, dataBuffObj, signalBuffObj);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    printf("Test 2 completed successfully!\n");
    printf("\n=== All PutMemNbi with Signal tests passed! ===\n");
  }

  // Finalize
  ShmemFree(dataBuff);
  ShmemFree(signalBuff);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  ConcurrentPutSignalThread();
  return 0;
}