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
#include <memory>
#include <iostream>
#include <vector>
#include <algorithm>

#include "mori/application/utils/check.hpp"
#include "mori/collective/all2all/oneshot_all2all_sdma_class.hpp"
#include "mori/collective/all2all/oneshot_all2all_sdma_kernel.hpp"
#include "mori/collective/all2all/oneshot_all2all_sdma_async_kernel.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::application;
using namespace mori::shmem;
using namespace mori::collective;

// Helper function: check HIP calls
#define CHECK_HIP(call) \
    do { \
        hipError_t err = (call); \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP Error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            throw std::runtime_error("HIP call failed"); \
        } \
    } while(0)

/**
 * @brief Initialize All2All test data (enhanced mode)
 * 
 * Each PE sends different data to different target PEs, facilitating route correctness verification
 * Format: value = (source_pe + 1) * 1000 + dest_pe
 * 
 * For 2 PEs case:
 * PE 0 sends: to PE 0: 1000, to PE 1: 1001
 * PE 1 sends: to PE 0: 2000, to PE 1: 2001
 */
void initializeAll2AllTestData(uint32_t* hostData, int myPe, int npes, 
                               size_t elemsPerPe, size_t totalElems) {
    // Clear data
    std::fill(hostData, hostData + totalElems, 0);
    
    // Each PE prepares different data for each target PE
    for (int destPe = 0; destPe < npes; destPe++) {
        // Generate unique value: identifies source PE and target PE
        uint32_t value = static_cast<uint32_t>((myPe + 1) * 1000 + destPe);
        
        // Fill entire data block
        for (size_t i = 0; i < elemsPerPe; i++) {
            size_t idx = destPe * elemsPerPe + i;
            hostData[idx] = value;
        }
    }
}

/**
 * @brief Verify All2All results
 * 
 * @param resultData Result data pointer
 * @param myPe Current PE ID
 * @param npes Total number of PEs
 * @param elemsPerPe Number of elements per PE
 * @param totalElems Total number of elements
 * @param verbose Whether to output detailed error information
 * @return true Verification passed
 * @return false Verification failed
 */
bool verifyAll2AllResult(const uint32_t* resultData, int myPe, int npes,
                         size_t elemsPerPe, size_t totalElems, 
                         bool verbose = true) {
    bool all_correct = true;
    int error_count = 0;
    const int max_errors_to_show = 5;
    
    if (verbose) {
        printf("\nPE %d: Verifying All2All results...\n", myPe);
    }
    
    // Verify each data block
    for (int sourcePe = 0; sourcePe < npes; sourcePe++) {
        // Data that should be received from source PE sourcePe
        uint32_t expected_value = static_cast<uint32_t>((sourcePe + 1) * 1000 + myPe);
        
        bool chunk_correct = true;
        int errors_in_chunk = 0;
        
        // Check first few elements
        size_t check_count = std::min(elemsPerPe, (size_t)4);
        for (size_t i = 0; i < check_count; i++) {
            size_t idx = sourcePe * elemsPerPe + i;
            uint32_t actual_value = resultData[idx];
            
            if (actual_value != expected_value) {
                chunk_correct = false;
                errors_in_chunk++;
                all_correct = false;
                
                if (verbose && errors_in_chunk <= max_errors_to_show) {
                    printf("  ERROR in chunk from PE %d[%zu]: expected %u, got %u\n",
                           sourcePe, i, expected_value, actual_value);
                    
                    // Try to decode received value
                    int got_source = actual_value / 1000 - 1;
                    int got_dest = actual_value % 1000;
                    printf("    Decoded as: source=PE%d, dest=PE%d\n", got_source, got_dest);
                }
            }
        }
        
        if (chunk_correct) {
            // If first few elements are correct, verify entire block
            bool full_chunk_ok = true;
            for (size_t i = 0; i < elemsPerPe; i++) {
                size_t idx = sourcePe * elemsPerPe + i;
                if (resultData[idx] != expected_value) {
                    full_chunk_ok = false;
                    all_correct = false;
                    break;
                }
            }
            
            if (verbose) {
                if (full_chunk_ok) {
                    printf("  From PE %d: CORRECT (all %zu values = %u)\n", 
                           sourcePe, elemsPerPe, expected_value);
                } else {
                    printf("  From PE %d: PARTIAL CORRECT (first %zu ok, but errors later)\n",
                           sourcePe, check_count);
                }
            }
        } else {
            if (verbose) {
                printf("  From PE %d: FAILED (%d errors in first %zu elements)\n",
                       sourcePe, errors_in_chunk, check_count);
            }
        }
    }
    
    if (verbose) {
        if (all_correct) {
            printf("PE %d: All2All verification PASSED! All %zu elements correct.\n",
                   myPe, totalElems);
        } else {
            printf("PE %d: All2All verification FAILED!\n", myPe);
            if (error_count > max_errors_to_show) {
                printf("  (showing first %d errors only)\n", max_errors_to_show);
            }
        }
    }
    
    return all_correct;
}

/**
 * @brief Print data pattern information
 */
void printDataPatternInfo(int myPe, int npes) {
    printf("\nPE %d: Data Pattern Explanation:\n", myPe);
    printf("  Format: value = (source_pe + 1) * 1000 + dest_pe\n");
    printf("  Example values:\n");
    
    for (int src = 0; src < npes; src++) {
        for (int dst = 0; dst < npes; dst++) {
            uint32_t value = static_cast<uint32_t>((src + 1) * 1000 + dst);
            printf("    PE%d -> PE%d: %u\n", src, dst, value);
        }
    }
    
    printf("\n  After All2All, PE %d should receive:\n", myPe);
    for (int src = 0; src < npes; src++) {
        uint32_t expected = static_cast<uint32_t>((src + 1) * 1000 + myPe);
        printf("    From PE %d: %u\n", src, expected);
    }
}

void testOneShotSdmaAll2all() {
  int status;

  // Initialize SHMEM
  MPI_Init(NULL, NULL);
  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  printf("PE %d of %d started\n", myPe, npes);

  // Configuration - can adjust size as needed
  const int elemsPerPe = 64 * 1024 * 1024;  // 64*1M elements
  const size_t bytesPerPe = elemsPerPe * sizeof(uint32_t);
  const size_t totalBytes = bytesPerPe * npes;

  // Allocate device memory
  uint32_t* outPutBuff = nullptr;
  CHECK_HIP(hipMalloc(&outPutBuff, totalBytes));

  uint32_t* inPutBuff = nullptr;
  CHECK_HIP(hipMalloc(&inPutBuff, totalBytes));

  // Initialize test data: enhanced mode, each PE sends different data to different target PEs
  std::vector<uint32_t> hostData(elemsPerPe * npes, 0);
  
  printf("PE %d: Initializing test data with enhanced pattern...\n", myPe);
  initializeAll2AllTestData(hostData.data(), myPe, npes, elemsPerPe, elemsPerPe * npes);
  
  // Print data pattern information (only on PE 0, once)
  if (myPe == 0) {
    printDataPatternInfo(myPe, npes);
  }

  // Verify input data
  printf("\nPE %d: Verifying input data pattern...\n", myPe);
  for (int destPe = 0; destPe < npes; destPe++) {
    uint32_t expected = static_cast<uint32_t>((myPe + 1) * 1000 + destPe);
    uint32_t actual = hostData[destPe * elemsPerPe];  // First element of each block
    
    if (actual == expected) {
      printf("  Sending to PE %d: %u (correct)\n", destPe, actual);
    } else {
      printf("  ERROR: To PE %d: expected %u, got %u\n", destPe, expected, actual);
    }
  }

  // Copy initialized data to device
  CHECK_HIP(hipMemcpy(inPutBuff, hostData.data(), totalBytes, hipMemcpyHostToDevice));
  CHECK_HIP(hipDeviceSynchronize());

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    printf("\n=== Starting All2all Operation ===\n");
    printf("Data size: %.2f MB per PE, %.2f MB total\n",
           bytesPerPe / (1024.0 * 1024.0), totalBytes / (1024.0 * 1024.0));
    printf("Using enhanced verification pattern\n\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Create All2allSdma object
  std::unique_ptr<All2allSdma<uint32_t>> all2all_obj;

  // Allocate sufficiently large buffer
  size_t input_buffer_size = totalBytes;  // Need to accommodate data layout from all PEs
  size_t output_buffer_size = totalBytes; // Need to accommodate results from all PEs

  printf("PE %d: Creating All2allSdma...\n", myPe);
  all2all_obj = std::make_unique<All2allSdma<uint32_t>>(
        myPe, npes, input_buffer_size, output_buffer_size);

  printf("PE %d: All2allSdma created successfully\n", myPe);
  // Execute All2All operation
  printf("PE %d: Executing All2All...\n", myPe);
  double execution_time;
  for (int i = 0; i < 10; i++)
    execution_time = (*all2all_obj)(inPutBuff, outPutBuff, elemsPerPe, stream);
 
  //CHECK_HIP(hipStreamSynchronize(stream));

  if (execution_time < 0) {
    fprintf(stderr, "PE %d: All2All operation failed\n", myPe);
  } else {
    printf("PE %d: All2All completed in %.6f seconds\n", myPe, execution_time);
  }

  // Copy results back to host
  std::vector<uint32_t> resultData(elemsPerPe * npes);
  CHECK_HIP(hipMemcpy(resultData.data(), outPutBuff, totalBytes, hipMemcpyDeviceToHost));
  CHECK_HIP(hipDeviceSynchronize());

  // Verify All2All results
  bool success = verifyAll2AllResult(resultData.data(), myPe, npes, 
                                     elemsPerPe, elemsPerPe * npes, true);

  MPI_Barrier(MPI_COMM_WORLD);

  if (success) {
    printf("PE %d: Verification PASSED!\n", myPe);
  } else {
    printf("PE %d: Verification FAILED!\n", myPe);
  }

  // Global statistics
  double global_max_time = 0.0;
  double global_min_time = 0.0;
  double global_avg_time = 0.0;
  
  MPI_Reduce(&execution_time, &global_max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&execution_time, &global_min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  
  double global_sum_time = 0.0;
  MPI_Reduce(&execution_time, &global_sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  // Global verification results
  int local_success_int = success ? 1 : 0;
  int global_success_sum = 0;
  MPI_Reduce(&local_success_int, &global_success_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (myPe == 0) {
    global_avg_time = global_sum_time / npes;
    
    printf("\n=== Global Performance Statistics ===\n");
    printf("Min time across PEs: %.6f seconds\n", global_min_time);
    printf("Max time across PEs: %.6f seconds\n", global_max_time);
    printf("Avg time across PEs: %.6f seconds\n", global_avg_time);
    
    // Calculate global bandwidth based on slowest PE
    double global_bandwidth = totalBytes / global_max_time / (1024.0 * 1024.0 * 1024.0);
    printf("Global bandwidth (based on slowest PE): %.2f GB/s\n", global_bandwidth);
    printf("Total data transferred: %.3f GB\n", totalBytes / (1024.0 * 1024.0 * 1024.0));
    
    printf("\n=== Global Verification Results ===\n");
    printf("PEs passed: %d/%d\n", global_success_sum, npes);
    
    if (global_success_sum == npes) {
      printf("\n=== All2all Test Completed Successfully! ===\n");
    } else {
      printf("\n=== All2all Test FAILED! ===\n");
    }
  }

  // Cleanup
  all2all_obj.reset();
  CHECK_HIP(hipFree(outPutBuff));
  CHECK_HIP(hipFree(inPutBuff));
  CHECK_HIP(hipStreamDestroy(stream));

  MPI_Barrier(MPI_COMM_WORLD);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
    testOneShotSdmaAll2all();
  return 0;
}