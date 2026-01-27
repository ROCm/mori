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

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <cassert>
#include <cstdio>

#include "mori/application/utils/check.hpp"
#include "mori/collective/allgather/oneshot_sdma_kernel.hpp"
#include "mori/collective/allgather/sdma_allgather.hpp"
#include "mori/collective/allgather/oneshot_sdma_async_kernel.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::application;
using namespace mori::shmem;
using namespace mori::collective;

void testOneShotSdmaAllGather() {
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
  const int elemsPerPe = 1024 * 1024 * 128;  // Number of elements each PE contributes
  const size_t bytesPerPe = elemsPerPe * sizeof(uint32_t);
  const size_t totalBytes = bytesPerPe * npes;  // Total buffer size

  //void* outPutBuff = ShmemMalloc(totalBytes);
  uint32_t* outPutBuff = nullptr;
  HIP_RUNTIME_CHECK(hipExtMallocWithFlags((void**)&outPutBuff, totalBytes, hipDeviceMallocUncached));
  assert(outPutBuff != nullptr);

  // void* inPutBuff = ShmemMalloc(bytesPerPe);
  uint32_t* inPutBuff = nullptr;
  HIP_RUNTIME_CHECK(hipExtMallocWithFlags((void**)&inPutBuff, bytesPerPe, hipDeviceMallocUncached));
  assert(inPutBuff != nullptr);

  // Initialize data buffer: each PE initializes only its own chunk
  uint32_t* hostData = new uint32_t[elemsPerPe*npes];
  memset(hostData, 0, elemsPerPe);

  // Each PE fills its own chunk with its PE ID
  for (int i = 0; i < elemsPerPe; i++) {
    hostData[i] = myPe + 100;  // Using PE_ID + 100 for clarity
  }

  // Copy initialized data to device
  HIP_RUNTIME_CHECK(hipMemcpy(inPutBuff, hostData, bytesPerPe, hipMemcpyHostToDevice));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

#if 0
  // Get symmetric memory object pointer
  SymmMemObjPtr outPutBuffObj = ShmemQueryMemObjPtr(outPutBuff);
  assert(outPutBuffObj.IsValid());

  SymmMemObjPtr inPutBuffObj = ShmemQueryMemObjPtr(inPutBuff);
  assert(inPutBuffObj.IsValid());

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
  #endif

  // Print initial data (only this PE's chunk should be non-zero)
  HIP_RUNTIME_CHECK(hipMemcpy(hostData + myPe*elemsPerPe, inPutBuff, bytesPerPe, hipMemcpyDeviceToHost));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  printf("PE %d: Initial data (showing first 4 elements of each chunk):\n", myPe);
  for (int pe = 0; pe < npes; pe++) {
    printf("  Chunk %d: ", pe);
    for (int i = 0; i < 4 && i < elemsPerPe; i++) {
      printf("%u ", hostData[i + myPe*elemsPerPe]);
    }
    printf("...\n");
  }

  // Launch AllGather kernel
  const int blockSize = 256;
  const int numBlocks = 1;
  bool use_async = 1;
  hipStream_t stream;
  HIP_RUNTIME_CHECK(hipStreamCreate(&stream));

  MPI_Barrier(MPI_COMM_WORLD);
  if (myPe == 0) {
    printf("\n=== Starting AllGather Operation ===\n\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);

  double start = MPI_Wtime();
 // if(1)
  double local_duration = 0;
 for(int i = 0; i<20; i++){
   local_duration = AllGather_sdma<uint32_t>(inPutBuff, outPutBuff, elemsPerPe, stream);
}  //OneShotAllGatharSdmaKernel<uint32_t><<<numBlocks, blockSize>>>(myPe, npes, inPutBuffObj, outPutBuffObj, flagsBuffObj, elemsPerPe);

//  else{
//    OneShotAllGatharSdmaAsyncPutKernel<uint32_t><<<numBlocks, blockSize>>>(myPe, npes, inPutBuffObj, outPutBuffObj, flagsBuffObj, elemsPerPe);
//    OneShotAllGatharSdmaAsyncWaitKernel<<<numBlocks, blockSize>>>(myPe, npes,  outPutBuffObj, flagsBuffObj);
//  }
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  double end = MPI_Wtime();
//  double local_duration = end - start;

  double global_max_duration;
  MPI_Reduce(&local_duration, &globaql_max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    printf("=== AllGather Operation Completed ===\n\n");
    double global_bandwidth = totalBytes / global_max_duration;
    global_bandwidth /= (1024.0 * 1024.0 * 1024.0);
    printf("========global bandwidth (base on slowest progress): %.6f GB/s ======== \n", global_bandwidth);
    printf("========global max time %.9f ======== \n", global_max_duration);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Copy result back to host for verification
  HIP_RUNTIME_CHECK(hipMemcpy(hostData, outPutBuff, totalBytes, hipMemcpyDeviceToHost));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  #if 0
  printf("PE %d: AllGather result (showing first 4 elements of each chunk):\n", myPe);
  for (int pe = 0; pe < npes; pe++) {
    printf("  Chunk %d: ", pe);
    for (int i = 0; i < 4 && i < elemsPerPe; i++) {
      printf("%u ", hostData[pe * elemsPerPe + i]);
    }
    printf("...\n");
  }
  #endif

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
  //ShmemFree(outPutBuff);
  hipFree(outPutBuff);
  hipFree(inPutBuff);
  //ShmemFree(flagsBuff);
  delete[] hostData;

  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  testOneShotSdmaAllGather();
  return 0;
}