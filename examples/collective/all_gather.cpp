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

#include "mori/collective/device/all_gather.hpp"

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <cassert>
#include <cstdio>

#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::application;
using namespace mori::shmem;
using namespace mori::collective;

void testAllGatherRing() {
  int status;

  // Initialize SHMEM
  MPI_Init(NULL, NULL);
  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  printf("PE %d of %d started\n", myPe, npes);

  // Configuration
  // Each PE contributes a chunk of data; all PEs will have all chunks after AllGather
  const int elemsPerPe = 1024 * 1024;  // Number of elements each PE contributes
  const size_t bytesPerPe = elemsPerPe * sizeof(uint32_t);
  const size_t totalBytes = bytesPerPe * npes;  // Total buffer size

  // Allocate data buffer - each PE will fill its own chunk initially
  void* dataBuff = ShmemMalloc(totalBytes);
  assert(dataBuff != nullptr);

  // Initialize data buffer: each PE initializes only its own chunk
  uint32_t* hostData = new uint32_t[elemsPerPe * npes];
  memset(hostData, 0, totalBytes);

  // Each PE fills its own chunk with its PE ID
  for (int i = 0; i < elemsPerPe; i++) {
    hostData[myPe * elemsPerPe + i] = myPe + 100;  // Using PE_ID + 100 for clarity
  }

  // Copy initialized data to device
  HIP_RUNTIME_CHECK(hipMemcpy(dataBuff, hostData, totalBytes, hipMemcpyHostToDevice));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // Get symmetric memory object pointer
  SymmMemObjPtr dataBuffObj = ShmemQueryMemObjPtr(dataBuff);
  assert(dataBuffObj.IsValid());

  // Allocate flags buffer for synchronization
  const size_t flagsSize = npes * sizeof(uint64_t);
  void* flagsBuff = ShmemMalloc(flagsSize);
  assert(flagsBuff != nullptr);

  // Initialize flags to zero
  HIP_RUNTIME_CHECK(hipMemset(flagsBuff, 0, flagsSize));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // Get symmetric memory object pointer for flags
  SymmMemObjPtr flagsBuffObj = ShmemQueryMemObjPtr(flagsBuff);
  assert(flagsBuffObj.IsValid());

  // Print initial data (only this PE's chunk should be non-zero)
  HIP_RUNTIME_CHECK(hipMemcpy(hostData, dataBuff, totalBytes, hipMemcpyDeviceToHost));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  printf("PE %d: Initial data (showing first 4 elements of each chunk):\n", myPe);
  for (int pe = 0; pe < npes; pe++) {
    printf("  Chunk %d: ", pe);
    for (int i = 0; i < 4 && i < elemsPerPe; i++) {
      printf("%u ", hostData[pe * elemsPerPe + i]);
    }
    printf("...\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    printf("\n=== Starting AllGather Operation ===\n\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Launch AllGather kernel
  const int blockSize = 256;
  const int numBlocks = 1;
  AllGatherRingKernel<uint32_t><<<numBlocks, blockSize>>>(myPe, npes, dataBuffObj, flagsBuffObj);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    printf("=== AllGather Operation Completed ===\n\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Copy result back to host for verification
  HIP_RUNTIME_CHECK(hipMemcpy(hostData, dataBuff, totalBytes, hipMemcpyDeviceToHost));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  printf("PE %d: AllGather result (showing first 4 elements of each chunk):\n", myPe);
  for (int pe = 0; pe < npes; pe++) {
    printf("  Chunk %d: ", pe);
    for (int i = 0; i < 4 && i < elemsPerPe; i++) {
      printf("%u ", hostData[pe * elemsPerPe + i]);
    }
    printf("...\n");
  }

  // Verify the result
  bool success = true;
  for (int pe = 0; pe < npes; pe++) {
    uint32_t expectedValue = pe + 100;
    for (int i = 0; i < elemsPerPe; i++) {
      if (hostData[pe * elemsPerPe + i] != expectedValue) {
        printf("PE %d: Verification FAILED at chunk %d, element %d: expected %u, got %u\n", myPe,
               pe, i, expectedValue, hostData[pe * elemsPerPe + i]);
        success = false;
        break;
      }
    }
    if (!success) break;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (success) {
    printf("PE %d: Verification PASSED ✓\n", myPe);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    if (success) {
      printf("\n=== All-Gather Test Completed Successfully! ===\n");
    } else {
      printf("\n=== All-Gather Test FAILED! ===\n");
    }
  }

  // Cleanup
  ShmemFree(dataBuff);
  ShmemFree(flagsBuff);
  delete[] hostData;

  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  testAllGatherRing();
  return 0;
}
