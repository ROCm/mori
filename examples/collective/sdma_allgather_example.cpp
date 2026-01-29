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
#include "mori/collective/allgather/oneshot_allgather_sdma_class.hpp"

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
 * @brief Initialize allgather test data (enhanced mode)
 *
 * Each PE sends different data to different target PEs, facilitating route correctness verification
 * Format: value = (source_pe + 1) * 1000 + dest_pe
 * This creates unique but consistent data for each PE
 *
 * For 2 PEs case:
 * PE 0 data: 1000, 1000, 1000, ...
 * PE 1 data: 2000, 2000, 2000, ...
 */
void initializeAllgatherTestData(uint32_t* hostData, int myPe, size_t elemsPerPe) {
    // Generate unique value for this PE
    uint32_t value = static_cast<uint32_t>((myPe + 1) * 1000);

    // Fill entire data block with same value
    std::fill(hostData, hostData + elemsPerPe, value);
}

/**
 * @brief Verify Allgather results
 *
 * @param resultData Result data pointer (contains all chunks)
 * @param myPe Current PE ID
 * @param npes Total number of PEs
 * @param elemsPerPe Number of elements per PE
 * @param verbose Whether to output detailed error information
 * @return true Verification passed
 * @return false Verification failed
 */
bool verifyAllgatherResult(const uint32_t* resultData, int myPe, int npes,
                         size_t elemsPerPe, bool verbose = true) {
    bool all_correct = true;
    int error_count = 0;
    const int max_errors_to_show = 5;

    if (verbose) {
        printf("\nPE %d: Verifying Allgather results...\n", myPe);
    }

    // Verify each PE's data chunk
    for (int sourcePe = 0; sourcePe < npes; sourcePe++) {
        // Data that should be received from source PE
        // Each PE sends the same data to everyone
        uint32_t expected_value = static_cast<uint32_t>((sourcePe + 1) * 1000 + 000);

        bool chunk_correct = true;
        int errors_in_chunk = 0;

        // Check first few elements of this chunk
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
                    if (actual_value >= 1000) {
                        int got_source = actual_value / 1000 - 1;
                        int remainder = actual_value % 1000;
                        printf("    Decoded as: source=PE%d, remainder=%d\n", got_source, remainder);
                    } else {
                        printf("    Value too small to decode\n");
                    }
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
            printf("PE %d: Allgather verification PASSED! All %zu elements correct.\n",
                   myPe, elemsPerPe * npes);
        } else {
            printf("PE %d: Allgather verification FAILED!\n", myPe);
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
    printf("\nPE %d: Allgather Data Pattern Explanation:\n", myPe);
    printf("  Each PE sends the same data to all other PEs\n");
    printf("  Format: value = (source_pe + 1) * 1000 + 000\n");
    printf("  Example data values:\n");

    for (int src = 0; src < npes; src++) {
        uint32_t value = static_cast<uint32_t>((src + 1) * 1000 + 000);
        printf("    PE %d data: %u (sent to all PEs)\n", src, value);
    }

    printf("\n  After Allgather, every PE will have:\n");
    for (int src = 0; src < npes; src++) {
        uint32_t expected = static_cast<uint32_t>((src + 1) * 1000 + 000);
        printf("    From PE %d: %u\n", src, expected);
    }
    printf("  All PEs will have identical output buffers\n");
}

/**
 * @brief Display input data for debugging
 */
void displayInputData(const uint32_t* hostData, int myPe, size_t elemsPerPe) {
    printf("\nPE %d: My Input Data (first 4 elements):\n", myPe);
    for (size_t i = 0; i < 4 && i < elemsPerPe; i++) {
        printf("  [%zu] = %u\n", i, hostData[i]);
    }

    // Decode the value
    if (elemsPerPe > 0) {
        uint32_t value = hostData[0];
        int src_pe = value / 1000 - 1;
        int remainder = value % 1000;
        printf("  Decoded: PE %d data, remainder=%d\n", src_pe, remainder);
    }
}

void testOneShotSdmaAllgather() {
  int status;

  // Initialize SHMEM
  MPI_Init(NULL, NULL);
  status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  printf("PE %d of %d started\n", myPe, npes);

  // Configuration - can be adjusted as needed
  const int elemsPerPe = 64 * 1024 * 1024;  // 64*1M elements
  const size_t bytesPerPe = elemsPerPe * sizeof(uint32_t);
  const size_t totalBytes = bytesPerPe * npes;

  // Allocate device memory
  uint32_t* outPutBuff = nullptr;
  CHECK_HIP(hipMalloc(&outPutBuff, totalBytes));  // Output: receives all chunks

  // For allgather, input buffer only needs to hold this PE's data
  uint32_t* inPutBuff = nullptr;
  CHECK_HIP(hipMalloc(&inPutBuff, bytesPerPe));  // Input: only local chunk

  // Initialize test data: each PE has its own unique data
  std::vector<uint32_t> hostData(elemsPerPe);

  printf("PE %d: Initializing my data...\n", myPe);
  initializeAllgatherTestData(hostData.data(), myPe, elemsPerPe);

  // Print data pattern information (only on PE 0)
  if (myPe == 0) {
    printDataPatternInfo(myPe, npes);
  }

  // Display input data for debugging
  displayInputData(hostData.data(), myPe, elemsPerPe);

  // Copy my data to device
  CHECK_HIP(hipMemcpy(inPutBuff, hostData.data(), bytesPerPe, hipMemcpyHostToDevice));
  CHECK_HIP(hipDeviceSynchronize());

  printf("PE %d: My data copied to device successfully\n", myPe);

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));
  MPI_Barrier(MPI_COMM_WORLD);

  if (myPe == 0) {
    printf("\n=== Starting Allgather Operation ===\n");
    printf("Data size: %.2f MB per PE (input), %.2f MB total (output)\n",
           bytesPerPe / (1024.0 * 1024.0), totalBytes / (1024.0 * 1024.0));
    printf("Each PE sends its unique data to all other PEs\n\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Create AllgatherSdma object
  std::unique_ptr<AllgatherSdma<uint32_t>> allgather_obj;

  // Allocate buffers - important: input buffer is smaller for allgather!
  size_t input_buffer_size = bytesPerPe;       // Input: only local chunk
  size_t output_buffer_size = totalBytes;      // Output: all chunks

  printf("PE %d: Creating AllgatherSdma...\n", myPe);
  allgather_obj = std::make_unique<AllgatherSdma<uint32_t>>(
        myPe, npes, input_buffer_size, output_buffer_size);

  printf("PE %d: AllgatherSdma created successfully\n", myPe);

  // Execute Allgather operation
  printf("PE %d: Executing Allgather...\n", myPe);
  double execution_time = -1.0;
  bool use_async = 1;  // Set to 1 for async mode, 0 for sync mode

  if (use_async == 0) {
    // Synchronous mode (original logic)
    for (int i = 0; i < 10; i++) {
      execution_time = (*allgather_obj)(inPutBuff, outPutBuff, elemsPerPe, stream);
      if (execution_time < 0) {
        fprintf(stderr, "PE %d: Iteration %d failed\n", myPe, i);
        break;
      }
    }
  } else {
    // Asynchronous mode
    printf("PE %d: Using ASYNC mode (start_async + wait_async)\n", myPe);

    // Test multiple async operations
    for (int i = 0; i < 10; i++) {
      if (myPe == 0) {
        printf("\n--- Iteration %d ---\n", i + 1);
      }

      MPI_Barrier(MPI_COMM_WORLD);

      #if 0
      // Check if there's an async operation already in progress
      if (allgather_obj->is_async_in_progress()) {
        printf("PE %d: Warning: Async operation already in progress, cancelling...\n", myPe);
        allgather_obj->cancel_async();
      }
      #endif

      // Step 1: Start async operation (PUT phase)
      bool started = allgather_obj->start_async(inPutBuff, outPutBuff, elemsPerPe, stream);
      if (!started) {
        fprintf(stderr, "PE %d: Failed to start async operation\n", myPe);
        break;
      }

      printf("PE %d: Async operation started successfully\n", myPe);

      // Perform other computations while waiting (simulated)
      if (myPe == 0) {
        printf("PE 0: Performing other computations while Allgather is in progress...\n");
        // Add other GPU computations here
      }

      // Step 2: Wait for async operation to complete (WAIT phase)
      execution_time = allgather_obj->wait_async(stream);

      if (execution_time < 0) {
        fprintf(stderr, "PE %d: Async operation failed\n", myPe);
        break;
      }

      printf("PE %d: Async iteration %d completed in %.6f seconds\n", myPe, i + 1, execution_time);

      // Add synchronization between iterations to ensure all PEs complete
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  //CHECK_HIP(hipStreamSynchronize(stream));

  if (execution_time < 0) {
    fprintf(stderr, "PE %d: Allgather operation failed\n", myPe);
  } else {
    printf("PE %d: Allgather completed in %.6f seconds\n", myPe, execution_time);
  }

  // Copy results back to host
  std::vector<uint32_t> resultData(elemsPerPe * npes);
  CHECK_HIP(hipMemcpy(resultData.data(), outPutBuff, totalBytes, hipMemcpyDeviceToHost));
  CHECK_HIP(hipDeviceSynchronize());

  // Display allgather results
  printf("\nPE %d: Allgather Result Pattern:\n", myPe);
  for (int pe = 0; pe < npes; pe++) {
    uint32_t* chunkData = resultData.data() + pe * elemsPerPe;
    printf("  Chunk from PE %d: [%u, %u, %u, %u, ...]\n",
           pe, chunkData[0], chunkData[1], chunkData[2], chunkData[3]);
  }

  // Verify Allgather results
  bool success = verifyAllgatherResult(resultData.data(), myPe, npes, elemsPerPe, true);

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
      printf("\n=== Allgather Test Completed Successfully! ===\n");
    } else {
      printf("\n=== Allgather Test FAILED! ===\n");
    }
  }

  // Cleanup
  allgather_obj.reset();
  CHECK_HIP(hipFree(outPutBuff));
  CHECK_HIP(hipFree(inPutBuff));
  CHECK_HIP(hipStreamDestroy(stream));

  MPI_Barrier(MPI_COMM_WORLD);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  testOneShotSdmaAllgather();
  return 0;
}
