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

// test commad:
// 1. mpirun --allow-run-as-root -np 2 ./examples/test_put_schar_nbi_thread
// 2. USE_SOCKET_BOOTSTRAP=1 mpirun --allow-run-as-root -np 2 ./examples/test_put_schar_nbi_thread
// Test kernel for ShmemPutScharNbiThread API and test socket bootstrap initialization

#include <mpi.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

__global__ void TestPutScharNbiThreadKernel(int myPe, signed char* localBuff, int numElements) {
  constexpr int sendPe = 0;
  constexpr int recvPe = 1;

  int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

  if (globalTid >= numElements) {
    return;
  }

  if (myPe == sendPe) {
    // Each thread sends its own signed char value
    signed char* src = localBuff + globalTid;
    signed char* dest = localBuff + globalTid;

    // Use ShmemPutScharNbiThread API
    ShmemPutScharNbiThread(dest, src, 1, recvPe, 1);

    // Ensure all operations are visible
    __threadfence_system();

    // Thread 0 in block 0 calls quiet to ensure all operations complete
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      ShmemQuietThread();
    }
  } else {
    // Receiver: wait for data to arrive
    // Expected value is (sendPe + globalTid) % 128 - 64, which ranges from -64 to 63
    signed char expected = static_cast<signed char>((sendPe + globalTid) % 128 - 64);

    // Use a simple busy-wait loop
    volatile signed char* volatileBuff = localBuff + globalTid;
    int waitCount = 0;
    while (*volatileBuff == 0 && waitCount < 10000000) {
      waitCount++;
    }
  }
}

void TestPutScharNbiThread() {
  // Check environment variable to decide initialization method
  const char* use_socket_env = std::getenv("USE_SOCKET_BOOTSTRAP");
  bool use_socket_bootstrap = (use_socket_env && std::string(use_socket_env) == "1");

  int status;
  int myRank = 0, nRanks = 1;

  if (use_socket_bootstrap) {
    // Socket Bootstrap mode: Initialize MPI first for UniqueId broadcast
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    printf("Rank %d: Using Socket Bootstrap Network for initialization (nranks=%d)\n", myRank,
           nRanks);

    mori_shmem_uniqueid_t uid;

    // Rank 0 generates the UniqueId
    if (myRank == 0) {
      status = ShmemGetUniqueId(&uid);
      if (status != 0) {
        fprintf(stderr, "Rank 0: ShmemGetUniqueId failed with status %d\n", status);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      printf("Rank 0: Generated UniqueId successfully\n");
    }

    // Broadcast UniqueId from rank 0 to all ranks using MPI
    MPI_Bcast(uid.data(), uid.size(), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (myRank != 0) {
      printf("Rank %d: Received UniqueId via MPI_Bcast\n", myRank);
    }

    // Set initialization attributes
    mori_shmem_init_attr_t attr;
    status = ShmemSetAttrUniqueIdArgs(myRank, nRanks, &uid, &attr);
    if (status != 0) {
      fprintf(stderr, "Rank %d: ShmemSetAttrUniqueIdArgs failed with status %d\n", myRank, status);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Initialize SHMEM with UniqueId (Socket Bootstrap)
    status = ShmemInitAttr(MORI_SHMEM_INIT_WITH_UNIQUEID, &attr);
    if (status != 0) {
      fprintf(stderr, "Rank %d: ShmemInitAttr failed with status %d\n", myRank, status);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    printf("Rank %d: SHMEM initialized successfully with Socket Bootstrap\n", myRank);

  } else {
    // Traditional MPI-based SHMEM initialization
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    printf("Rank %d: Using MPI-based SHMEM initialization\n", myRank);

    status = ShmemMpiInit(MPI_COMM_WORLD);
    if (status != 0) {
      fprintf(stderr, "Rank %d: ShmemMpiInit failed with status %d\n", myRank, status);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();
  assert(npes == 2);

  constexpr int threadNum = 128;
  constexpr int blockNum = 4;
  int numElements = threadNum * blockNum;
  size_t buffSize = numElements * sizeof(signed char);

  if (myPe == 0) {
    printf("=================================================================\n");
    printf("Testing ShmemPutScharNbiThread API\n");
    printf("=================================================================\n");
    printf("Number of PEs: %d\n", npes);
    printf("Number of elements: %d\n", numElements);
    printf("Threads per block: %d\n", threadNum);
    printf("Number of blocks: %d\n", blockNum);
    printf("=================================================================\n\n");
  }

  // Allocate symmetric memory
  void* buff = ShmemMalloc(buffSize);
  assert(buff != nullptr);

  // Initialize buffer
  std::vector<signed char> hostBuff(numElements);
  if (myPe == 0) {
    // Sender: initialize with pattern (value ranges from -64 to 63)
    for (int i = 0; i < numElements; i++) {
      hostBuff[i] = static_cast<signed char>((myPe + i) % 128 - 64);
    }
  } else {
    // Receiver: initialize with zeros
    for (int i = 0; i < numElements; i++) {
      hostBuff[i] = 0;
    }
  }

  HIP_RUNTIME_CHECK(hipMemcpy(buff, hostBuff.data(), buffSize, hipMemcpyHostToDevice));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  if (myPe == 0) {
    printf("--- Running test kernel ---\n");
  }

  // Launch kernel
  TestPutScharNbiThreadKernel<<<blockNum, threadNum>>>(myPe, reinterpret_cast<signed char*>(buff),
                                                       numElements);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // Barrier to ensure all operations complete
  MPI_Barrier(MPI_COMM_WORLD);

  // Verify results on host
  HIP_RUNTIME_CHECK(hipMemcpy(hostBuff.data(), buff, buffSize, hipMemcpyDeviceToHost));

  if (myPe == 1) {
    bool success = true;
    int errorCount = 0;
    const int maxErrorsToShow = 10;

    for (int i = 0; i < numElements; i++) {
      signed char expected = static_cast<signed char>((0 + i) % 128 - 64);  // sendPe = 0
      if (hostBuff[i] != expected) {
        if (errorCount < maxErrorsToShow) {
          printf("Error at index %d: expected %d, got %d\n", i, static_cast<int>(expected),
                 static_cast<int>(hostBuff[i]));
        }
        success = false;
        errorCount++;
      }
    }

    if (success) {
      printf("✓ Test PASSED! All %d signed char elements verified correctly.\n", numElements);
      printf("  Value range tested: -64 to 63\n");
    } else {
      printf("✗ Test FAILED! %d errors found out of %d elements.\n", errorCount, numElements);
    }
  }

  if (myPe == 0) {
    printf("--- Sender PE completed ---\n");
  }

  if (myPe == 0) {
    printf("\n=================================================================\n");
    printf("Test completed!\n");
    printf("=================================================================\n");
  }

  // Cleanup
  ShmemFree(buff);
  ShmemFinalize();

  // In Socket Bootstrap mode, we need to manually finalize MPI
  // because ShmemFinalize only finalizes the Socket Bootstrap network, not MPI
  const char* socket_env = std::getenv("USE_SOCKET_BOOTSTRAP");
  if (socket_env && std::string(socket_env) == "1") {
    MPI_Finalize();
  }
  // Note: In MPI mode, ShmemFinalize() calls MPI_Finalize() internally
}

int main(int argc, char* argv[]) {
  TestPutScharNbiThread();
  return 0;
}
