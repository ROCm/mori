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
#include "mori/collective/all2all/sdma_all2all.hpp"
#include "mori/collective/all2all/oneshot_all2all_sdma_class.hpp"  // 新的头文件
#include "mori/collective/all2all/oneshot_all2all_sdma_kernel.hpp"
#include "mori/collective/all2all/oneshot_all2all_sdma_async_kernel.hpp"
#include "mori/shmem/shmem.hpp"
 
using namespace mori::core;
using namespace mori::application;
using namespace mori::shmem;
using namespace mori::collective;
#if 0
void testOneShotSdmaAll2all() {
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
  const int elemsPerPe = 8 * 1024 * 1024;  // Number of elements each PE contributes
  const size_t bytesPerPe = elemsPerPe * sizeof(uint32_t);
  const size_t totalBytes = bytesPerPe * npes;  // Total buffer size
 
  #if 0
  // Allocate data buffer - each PE will fill its own chunk initially
  void* outPutBuff = ShmemMalloc(totalBytes);
  assert(outPutBuff != nullptr);
  void* inPutBuff = ShmemMalloc(totalBytes);
  assert(inPutBuff != nullptr);
  #endif
  //void* outPutBuff = ShmemMalloc(totalBytes);
  uint32_t* outPutBuff = nullptr;
  HIP_RUNTIME_CHECK(hipExtMallocWithFlags((void**)&outPutBuff, totalBytes, hipDeviceMallocUncached));
  assert(outPutBuff != nullptr);
  // void* inPutBuff = ShmemMalloc(bytesPerPe);
  uint32_t* inPutBuff = nullptr;
  HIP_RUNTIME_CHECK(hipExtMallocWithFlags((void**)&inPutBuff, totalBytes, hipDeviceMallocUncached));
  assert(inPutBuff != nullptr);
 
  // Initialize data buffer: each PE initializes only its own chunk
  uint32_t* hostData = new uint32_t[elemsPerPe*npes];
  memset(hostData, 0, totalBytes);
 
  // Each PE fills its own chunk with its PE ID
  for (int k = 0; k < npes; k++) {
    for (int i = 0; i < elemsPerPe; i++) {
      hostData[k*elemsPerPe + i] = k + (myPe + 1) * 100;  // Using PE_ID + 100 for clarity
    }
  }
 
  // Copy initialized data to device
  HIP_RUNTIME_CHECK(hipMemcpy(inPutBuff, hostData, totalBytes, hipMemcpyHostToDevice));
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
  //HIP_RUNTIME_CHECK(hipMemcpy(hostData + myPe*elemsPerPe, inPutBuff, bytesPerPe, hipMemcpyDeviceToHost));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  printf("PE %d: Initial data (showing first 4 elements of each chunk):\n", myPe);
  for (int pe = 0; pe < npes; pe++) {
    printf("  Chunk %d: ", pe);
    for (int i = 0; i < 4 && i < elemsPerPe; i++) {
      printf("%u ", hostData[i + pe*elemsPerPe]);
    }
    printf("...\n");
  }
 
  const int blockSize = 256;
  const int numBlocks = 1;
  bool use_async = 0;

  hipStream_t stream;
  HIP_RUNTIME_CHECK(hipStreamCreate(&stream));
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    printf("\n=== Starting All2all Operation ===\n\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  #if 1
  double local_duration;
  if(!use_async) {
    for (int i = 0; i < 10; i++) {
      local_duration = All2all_sdma<uint32_t>(inPutBuff, outPutBuff, elemsPerPe, stream);
    }
    //OneShotAllGatharSdmaKernel<uint32_t><<<numBlocks, blockSize>>>(myPe, npes, inPutBuffObj, outPutBuffObj, flagsBuffObj, elemsPerPe);
  } else {
      //OneShotAllGatharSdmaAsyncPutKernel<uint32_t><<<numBlocks, blockSize>>>(myPe, npes, inPutBuffObj, outPutBuffObj, flagsBuffObj, elemsPerPe);
      //OneShotAllGatharSdmaAsyncWaitKernel<<<numBlocks, blockSize>>>(myPe, npes,  outPutBuffObj, flagsBuffObj);
  }
  #endif

  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
 
  // Copy result back to host for verification
  HIP_RUNTIME_CHECK(hipMemcpy(hostData, outPutBuff, totalBytes, hipMemcpyDeviceToHost));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  printf("PE %d: All2all result (showing first 4 elements of each chunk):\n", myPe);
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
    uint32_t expectedValue = myPe + (pe + 1) * 100;
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
    printf("PE %d: Verification PASSED!\n", myPe);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  double global_max_duration;
  MPI_Reduce(&local_duration, &global_max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  double local_bandwidth = totalBytes / local_duration;
  local_bandwidth /= (1024.0 * 1024.0 * 1024.0);
  printf("========myPe:%u, totalBytes:%.9fGB, local time:%.9f, local_bandwidth:%.9f\n", myPe, totalBytes/(1024.0 * 1024.0 * 1024.0), local_duration, local_bandwidth);

  if (myPe == 0) {
    printf("=== All2all Operation Completed ===\n\n");
    double global_bandwidth = totalBytes / global_max_duration;
    global_bandwidth /= (1024.0 * 1024.0 * 1024.0);
    printf("========myPe:%u, totalBytes:%.9fGB, global time:%.9f, global_bandwidth:%.9f\n", myPe, totalBytes/(1024.0 * 1024.0 * 1024.0), global_max_duration, global_bandwidth);
  }
 
  if (myPe == 0) {
    if (success) {
      printf("\n=== All2all Test Completed Successfully! ===\n");
    } else {
      printf("\n=== All2all Test FAILED! ===\n");
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
#endif

void testOneShotSdmaAll2all() {
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
  const int elemsPerPe = 8 * 1024 * 1024;  // Number of elements each PE contributes
  const size_t bytesPerPe = elemsPerPe * sizeof(uint32_t);
  const size_t totalBytes = bytesPerPe * npes;  // Total buffer size
 
  // Allocate device memory using hipExtMallocWithFlags
  uint32_t* outPutBuff = nullptr;
  HIP_RUNTIME_CHECK(hipExtMallocWithFlags((void**)&outPutBuff, totalBytes, hipDeviceMallocUncached));
  assert(outPutBuff != nullptr);
  
  uint32_t* inPutBuff = nullptr;
  HIP_RUNTIME_CHECK(hipExtMallocWithFlags((void**)&inPutBuff, totalBytes, hipDeviceMallocUncached));
  assert(inPutBuff != nullptr);
 
  // Initialize data buffer on host
  uint32_t* hostData = new uint32_t[elemsPerPe * npes];
  memset(hostData, 0, totalBytes);
 
  // Each PE fills its own chunk with its PE ID
  for (int k = 0; k < npes; k++) {
    for (int i = 0; i < elemsPerPe; i++) {
      hostData[k * elemsPerPe + i] = k + (myPe + 1) * 100;  // Using PE_ID + 100 for clarity
    }
  }
 
  // Copy initialized data to device
  HIP_RUNTIME_CHECK(hipMemcpy(inPutBuff, hostData, totalBytes, hipMemcpyHostToDevice));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // Print initial data (only this PE's chunk should be non-zero)
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  printf("PE %d: Initial data (showing first 4 elements of each chunk):\n", myPe);
  for (int pe = 0; pe < npes; pe++) {
    printf("  Chunk %d: ", pe);
    for (int i = 0; i < 4 && i < elemsPerPe; i++) {
      printf("%u ", hostData[i + pe * elemsPerPe]);
    }
    printf("...\n");
  }
 
  const int blockSize = 256;
  const int numBlocks = 1;
  bool use_async = false;

  hipStream_t stream;
  HIP_RUNTIME_CHECK(hipStreamCreate(&stream));
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    printf("\n=== Starting All2all Operation ===\n\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Create All2allSdma object (flags initialized in constructor)
  std::unique_ptr<All2allSdma<uint32_t>> all2all_obj;
  try {
    all2all_obj = std::make_unique<All2allSdma<uint32_t>>(myPe, npes);
  } catch (const std::exception& e) {
    fprintf(stderr, "PE %d: Failed to create All2allSdma object: %s\n", myPe, e.what());
    ShmemFinalize();
    MPI_Finalize();
    exit(1);
  }

  #if 1
  double local_duration = 0.0;
  if (!use_async) {
    for (int i = 0; i < 10; i++) {
      // Use the All2allSdma object (flags reused across iterations)
      local_duration = (*all2all_obj)(inPutBuff, outPutBuff, elemsPerPe, stream);
      
      // Optional: print iteration progress
      if (i == 0 && myPe == 0) {
        printf("Completed iteration 0\n");
      }
    }
  } else {
    // For async version, you might need a different class implementation
    // OneShotAllGatharSdmaAsyncPutKernel<uint32_t><<<numBlocks, blockSize>>>(myPe, npes, inPutBuffObj, outPutBuffObj, flagsBuffObj, elemsPerPe);
    // OneShotAllGatharSdmaAsyncWaitKernel<<<numBlocks, blockSize>>>(myPe, npes, outPutBuffObj, flagsBuffObj);
  }
  #endif

  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
 
  // Copy result back to host for verification
  HIP_RUNTIME_CHECK(hipMemcpy(hostData, outPutBuff, totalBytes, hipMemcpyDeviceToHost));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  
  printf("PE %d: All2all result (showing first 4 elements of each chunk):\n", myPe);
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
    uint32_t expectedValue = myPe + (pe + 1) * 100;
    for (int i = 0; i < elemsPerPe; i++) {
      if (hostData[pe * elemsPerPe + i] != expectedValue) {
        printf("PE %d: Verification FAILED at chunk %d, element %d: expected %u, got %u\n", 
               myPe, pe, i, expectedValue, hostData[pe * elemsPerPe + i]);
        success = false;
        break;
      }
    }
    if (!success) break;
  }
 
  MPI_Barrier(MPI_COMM_WORLD);
 
  if (success) {
    printf("PE %d: Verification PASSED!\n", myPe);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  double global_max_duration;
  MPI_Reduce(&local_duration, &global_max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  
  double local_bandwidth = totalBytes / local_duration;
  local_bandwidth /= (1024.0 * 1024.0 * 1024.0);
  printf("======== myPe:%u, totalBytes:%.9fGB, local time:%.9f, local_bandwidth:%.9f GB/s ========\n", 
         myPe, totalBytes/(1024.0 * 1024.0 * 1024.0), local_duration, local_bandwidth);

  if (myPe == 0) {
    printf("=== All2all Operation Completed ===\n\n");
    double global_bandwidth = totalBytes / global_max_duration;
    global_bandwidth /= (1024.0 * 1024.0 * 1024.0);
    printf("======== myPe:%u, totalBytes:%.9fGB, global time:%.9f, global_bandwidth:%.9f GB/s ========\n", 
           myPe, totalBytes/(1024.0 * 1024.0 * 1024.0), global_max_duration, global_bandwidth);
  }
 
  if (myPe == 0) {
    if (success) {
      printf("\n=== All2all Test Completed Successfully! ===\n");
    } else {
      printf("\n=== All2all Test FAILED! ===\n");
    }
  }
 
  // Cleanup
  // Note: flags memory is automatically managed by All2allSdma destructor
  
  // Destroy All2allSdma object explicitly (optional, will be destroyed when out of scope)
  all2all_obj.reset();
  
  HIP_RUNTIME_CHECK(hipFree(outPutBuff));
  HIP_RUNTIME_CHECK(hipFree(inPutBuff));
  delete[] hostData;
  
  // Cleanup stream
  HIP_RUNTIME_CHECK(hipStreamDestroy(stream));
  
  ShmemFinalize();
  MPI_Finalize();
}

int main(int argc, char* argv[]) {
  testOneShotSdmaAll2all();
  return 0;
}
