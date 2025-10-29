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

// Legacy API: Using SymmMemObjPtr + offset
template <typename T>
__global__ void AtomicFetchThreadKernel(int myPe, const SymmMemObjPtr memObj) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
  int threadOffset = globalTid * sizeof(T);

  if (myPe == sendPe) {
    T ret = ShmemAtomicTypeFetchThread<T>(memObj, 2 * sizeof(T), 1, 0, AMO_FETCH_ADD, recvPe);
    __threadfence_system();
    if (ret == gridDim.x * blockDim.x) {
      printf("Legacy API: globalTid: %d ret = %lu atomic fetch is ok!~\n", globalTid, (uint64_t)ret);
    }
  } else {
    while (AtomicLoadRelaxed(reinterpret_cast<T*>(memObj->localPtr) + 2) !=
           gridDim.x * blockDim.x + 1) {
    }
    if (globalTid == 0) {
      printf("Legacy API: atomic fetch is ok!~\n");
    }
  }
}

// New API: Using pure addresses
template <typename T>
__global__ void AtomicFetchThreadKernel_PureAddr(int myPe, T* localBuff) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (myPe == sendPe) {
    T* dest = localBuff + 2;
    T ret = ShmemAtomicTypeFetchThread<T>(dest, 1, 0, AMO_FETCH_ADD, recvPe);
    __threadfence_system();
    if (ret == gridDim.x * blockDim.x) {
      printf("Pure Address API: globalTid: %d ret = %lu atomic fetch is ok!~\n", globalTid, (uint64_t)ret);
    }
  } else {
    while (AtomicLoadRelaxed(localBuff + 2) != gridDim.x * blockDim.x + 1) {
    }
    if (globalTid == 0) {
      printf("Pure Address API: atomic fetch is ok!~\n");
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

  if (myPe == 0) {
    printf("=================================================================\n");
    printf("Testing both Legacy and Pure Address APIs (Atomic Fetch)\n");
    printf("=================================================================\n");
  }

  void* buff = ShmemMalloc(buffSize);
  SymmMemObjPtr buffObj = ShmemQueryMemObjPtr(buff);
  assert(buffObj.IsValid());

  // Run atomic fetch operations for different types
  for (int iteration = 0; iteration < 3; iteration++) {
    if (myPe == 0) {
      printf("\n========== Iteration %d ==========\n", iteration + 1);
    }

    // ===== Test 1: Legacy API with uint64_t =====
    if (myPe == 0) {
      printf("\n--- Test 1: Legacy API (uint64_t) ---\n");
    }
    myHipMemsetD64(buff, myPe, numEle);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    AtomicFetchThreadKernel<uint64_t><<<blockNum, threadNum>>>(myPe, buffObj);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (myPe == 0) {
      uint64_t result = *(reinterpret_cast<uint64_t*>(buff) + 2);
      printf("✓ Legacy API uint64_t test completed. Result at index 2: %lu\n", result);
    }

    // ===== Test 2: Pure Address API with uint64_t =====
    if (myPe == 0) {
      printf("\n--- Test 2: Pure Address API (uint64_t) ---\n");
    }
    myHipMemsetD64(buff, myPe, numEle);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    AtomicFetchThreadKernel_PureAddr<uint64_t><<<blockNum, threadNum>>>(
        myPe, reinterpret_cast<uint64_t*>(buff));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (myPe == 0) {
      uint64_t result = *(reinterpret_cast<uint64_t*>(buff) + 2);
      printf("✓ Pure Address API uint64_t test completed. Result at index 2: %lu\n", result);
    }

    // ===== Test 3: Legacy API with int64_t =====
    if (myPe == 0) {
      printf("\n--- Test 3: Legacy API (int64_t) ---\n");
    }
    myHipMemsetD64(buff, myPe, numEle);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    AtomicFetchThreadKernel<int64_t><<<blockNum, threadNum>>>(myPe, buffObj);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (myPe == 0) {
      int64_t result = *(reinterpret_cast<int64_t*>(buff) + 2);
      printf("✓ Legacy API int64_t test completed. Result at index 2: %ld\n", result);
    }

    // ===== Test 4: Pure Address API with int64_t =====
    if (myPe == 0) {
      printf("\n--- Test 4: Pure Address API (int64_t) ---\n");
    }
    myHipMemsetD64(buff, myPe, numEle);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    AtomicFetchThreadKernel_PureAddr<int64_t><<<blockNum, threadNum>>>(
        myPe, reinterpret_cast<int64_t*>(buff));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (myPe == 0) {
      int64_t result = *(reinterpret_cast<int64_t*>(buff) + 2);
      printf("✓ Pure Address API int64_t test completed. Result at index 2: %ld\n", result);
    }

    // ===== Test 5: Legacy API with uint32_t =====
    if (myPe == 0) {
      printf("\n--- Test 5: Legacy API (uint32_t) ---\n");
    }
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buff), myPe, numEle));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    AtomicFetchThreadKernel<uint32_t><<<blockNum, threadNum>>>(myPe, buffObj);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (myPe == 0) {
      uint32_t result = *(reinterpret_cast<uint32_t*>(buff) + 2);
      printf("✓ Legacy API uint32_t test completed. Result at index 2: %u\n", result);
    }

    // ===== Test 6: Pure Address API with uint32_t =====
    if (myPe == 0) {
      printf("\n--- Test 6: Pure Address API (uint32_t) ---\n");
    }
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<uint32_t*>(buff), myPe, numEle));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    AtomicFetchThreadKernel_PureAddr<uint32_t><<<blockNum, threadNum>>>(
        myPe, reinterpret_cast<uint32_t*>(buff));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (myPe == 0) {
      uint32_t result = *(reinterpret_cast<uint32_t*>(buff) + 2);
      printf("✓ Pure Address API uint32_t test completed. Result at index 2: %u\n", result);
    }

    // ===== Test 7: Legacy API with int32_t =====
    if (myPe == 0) {
      printf("\n--- Test 7: Legacy API (int32_t) ---\n");
    }
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<int32_t*>(buff), myPe, numEle));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    AtomicFetchThreadKernel<int32_t><<<blockNum, threadNum>>>(myPe, buffObj);
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (myPe == 0) {
      int32_t result = *(reinterpret_cast<int32_t*>(buff) + 2);
      printf("✓ Legacy API int32_t test completed. Result at index 2: %d\n", result);
    }

    // ===== Test 8: Pure Address API with int32_t =====
    if (myPe == 0) {
      printf("\n--- Test 8: Pure Address API (int32_t) ---\n");
    }
    HIP_RUNTIME_CHECK(hipMemsetD32(reinterpret_cast<int32_t*>(buff), myPe, numEle));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    AtomicFetchThreadKernel_PureAddr<int32_t><<<blockNum, threadNum>>>(
        myPe, reinterpret_cast<int32_t*>(buff));
    HIP_RUNTIME_CHECK(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (myPe == 0) {
      int32_t result = *(reinterpret_cast<int32_t*>(buff) + 2);
      printf("✓ Pure Address API int32_t test completed. Result at index 2: %d\n", result);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (myPe == 0) {
      printf("\nIteration %d completed successfully!\n", iteration + 1);
    }
    sleep(1);
  }

  if (myPe == 0) {
    printf("\n=================================================================\n");
    printf("All Atomic Fetch tests completed!\n");
    printf("=================================================================\n");
  }

  // Finalize
  ShmemFree(buff);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  testAtomicFetchThread();
  return 0;
}
