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
#include "mori/collective/allreduce/twoshot_allreduce_sdma_class.hpp"

#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::application;
using namespace mori::shmem;
using namespace mori::collective;

#define CHECK_HIP(call) \
    do { \
        hipError_t err = (call); \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP Error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            throw std::runtime_error("HIP call failed"); \
        } \
    } while(0)

/**
 * @brief Initialize allreduce test data
 *
 * PE i fills all elements with value (i + 1).
 * After allreduce (sum), every element should equal npes*(npes+1)/2.
 */
void initializeAllreduceTestData(uint32_t* hostData, int myPe, size_t elemCount) {
    uint32_t value = static_cast<uint32_t>(myPe + 1);
    std::fill(hostData, hostData + elemCount, value);
}

/**
 * @brief Verify AllReduce results
 *
 * Expected: every element = npes*(npes+1)/2
 */
bool verifyAllreduceResult(const uint32_t* resultData, int myPe, int npes,
                           size_t elemCount, bool verbose = true) {
    bool all_correct = true;
    int error_count = 0;
    const int max_errors_to_show = 10;

    uint32_t expected_value = static_cast<uint32_t>(npes * (npes + 1) / 2);

    if (verbose) {
        printf("\nPE %d: Verifying AllReduce results (expected all values = %u)...\n",
               myPe, expected_value);
    }

    // Quick-check first few elements
    size_t check_count = std::min(elemCount, static_cast<size_t>(8));
    for (size_t i = 0; i < check_count; i++) {
        if (resultData[i] != expected_value) {
            all_correct = false;
            error_count++;
            if (verbose && error_count <= max_errors_to_show) {
                printf("  ERROR at [%zu]: expected %u, got %u\n",
                       i, expected_value, resultData[i]);
            }
        }
    }

    // If first elements are correct, verify entire buffer
    if (all_correct) {
        for (size_t i = 0; i < elemCount; i++) {
            if (resultData[i] != expected_value) {
                all_correct = false;
                error_count++;
                if (verbose && error_count <= max_errors_to_show) {
                    printf("  ERROR at [%zu]: expected %u, got %u\n",
                           i, expected_value, resultData[i]);
                }
                if (error_count > max_errors_to_show) break;
            }
        }
    }

    if (verbose) {
        if (all_correct) {
            printf("PE %d: AllReduce verification PASSED! All %zu elements = %u\n",
                   myPe, elemCount, expected_value);
        } else {
            printf("PE %d: AllReduce verification FAILED! (%d errors found)\n",
                   myPe, error_count);
        }
    }

    return all_correct;
}

void printDataPatternInfo(int myPe, int npes) {
    printf("\nPE %d: AllReduce Data Pattern:\n", myPe);
    printf("  Each PE fills all elements with value = (PE_id + 1)\n");
    printf("  After AllReduce (sum), expected = sum(1..%d) = %d\n",
           npes, npes * (npes + 1) / 2);
    printf("  PE contributions:\n");
    for (int pe = 0; pe < npes; pe++) {
        printf("    PE %d: all elements = %d\n", pe, pe + 1);
    }
    printf("\n");
}

void displayInputData(const uint32_t* hostData, int myPe, size_t elemCount) {
    printf("PE %d: Input data (first 4 elements): [%u, %u, %u, %u]\n",
           myPe,
           elemCount > 0 ? hostData[0] : 0,
           elemCount > 1 ? hostData[1] : 0,
           elemCount > 2 ? hostData[2] : 0,
           elemCount > 3 ? hostData[3] : 0);
}

void testAllreduceSdma() {
    int status;

    MPI_Init(NULL, NULL);
    status = ShmemMpiInit(MPI_COMM_WORLD);
    assert(!status);

    int myPe = ShmemMyPe();
    int npes = ShmemNPes();

    printf("PE %d of %d started\n", myPe, npes);

    const int elemsPerPe = 64 * 1024 * 1024;  // 64M elements
    const size_t bytesPerPe = elemsPerPe * sizeof(uint32_t);
    const size_t totalBytes = bytesPerPe * npes;  // for allgather intermediate

    // Allreduce: input and output are both elemsPerPe elements
    uint32_t* inPutBuff = nullptr;
    CHECK_HIP(hipMalloc(&inPutBuff, bytesPerPe));

    uint32_t* outPutBuff = nullptr;
    CHECK_HIP(hipMalloc(&outPutBuff, bytesPerPe));

    // Initialize test data
    std::vector<uint32_t> hostData(elemsPerPe);

    printf("PE %d: Initializing data...\n", myPe);
    initializeAllreduceTestData(hostData.data(), myPe, elemsPerPe);

    if (myPe == 0) {
        printDataPatternInfo(myPe, npes);
    }

    displayInputData(hostData.data(), myPe, elemsPerPe);

    CHECK_HIP(hipMemcpy(inPutBuff, hostData.data(), bytesPerPe, hipMemcpyHostToDevice));
    CHECK_HIP(hipDeviceSynchronize());

    printf("PE %d: Data copied to device\n", myPe);

    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));
    MPI_Barrier(MPI_COMM_WORLD);

    if (myPe == 0) {
        printf("\n=== Starting AllReduce Operation ===\n");
        printf("Data size: %.2f MB per PE\n", bytesPerPe / (1024.0 * 1024.0));
        printf("Transit buffer: %.2f MB (for allgather intermediate)\n",
               totalBytes / (1024.0 * 1024.0));
        printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Create AllreduceSdma object
    // input_buffer_size = bytesPerPe (local data)
    // output_buffer_size = totalBytes (allgather intermediate needs npes * bytesPerPe)
    std::unique_ptr<AllreduceSdma<uint32_t>> allreduce_obj;

    printf("PE %d: Creating AllreduceSdma...\n", myPe);
    allreduce_obj = std::make_unique<AllreduceSdma<uint32_t>>(
        myPe, npes, bytesPerPe, totalBytes);

    printf("PE %d: AllreduceSdma created successfully\n", myPe);

    // Execute AllReduce with async mode
    printf("PE %d: Using ASYNC mode (start_async + wait_async)\n", myPe);

    double execution_time = -1.0;
    std::vector<double> exec_times;
    const int num_iterations = 10;
    const int warmup_iterations = 10;

    for (int i = 0; i < num_iterations + warmup_iterations; i++) {
        if (myPe == 0 && (i == 0 || i == warmup_iterations)) {
            const char* stage = (i < warmup_iterations) ? "Warmup" : "Measurement";
            printf("\n--- %s Iteration %d ---\n", stage, i + 1);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        bool started = allreduce_obj->start_async(inPutBuff, outPutBuff, elemsPerPe, stream);
        if (!started) {
            fprintf(stderr, "PE %d: Failed to start async operation\n", myPe);
            break;
        }

        execution_time = allreduce_obj->wait_async(stream);

        if (execution_time < 0) {
            fprintf(stderr, "PE %d: Async operation failed\n", myPe);
            break;
        }

        if (i >= warmup_iterations) {
            exec_times.push_back(execution_time);
            if (myPe == 0 && exec_times.size() == 1) {
                printf("PE %d: First measurement iteration: %.6f seconds\n", myPe, execution_time);
            }
        } else if (myPe == 0) {
            printf("PE %d: Warmup iteration %d: %.6f seconds\n", myPe, i + 1, execution_time);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Calculate local statistics
    double avg_time = 0.0, min_time = 0.0, max_time = 0.0;
    if (!exec_times.empty()) {
        double sum_time = 0.0;
        min_time = exec_times[0];
        max_time = exec_times[0];
        for (double t : exec_times) {
            sum_time += t;
            if (t < min_time) min_time = t;
            if (t > max_time) max_time = t;
        }
        avg_time = sum_time / exec_times.size();

        if (myPe == 0) {
            printf("\nPE %d local statistics (from %zu iterations):\n", myPe, exec_times.size());
            printf("  Min time: %.6f seconds\n", min_time);
            printf("  Max time: %.6f seconds\n", max_time);
            printf("  Avg time: %.6f seconds\n", avg_time);
        }
    }

    // Copy results back to host and verify
    std::vector<uint32_t> resultData(elemsPerPe);
    CHECK_HIP(hipMemcpy(resultData.data(), outPutBuff, bytesPerPe, hipMemcpyDeviceToHost));
    CHECK_HIP(hipDeviceSynchronize());

    printf("\nPE %d: Result (first 4 elements): [%u, %u, %u, %u]\n",
           myPe, resultData[0], resultData[1], resultData[2], resultData[3]);

    bool success = verifyAllreduceResult(resultData.data(), myPe, npes, elemsPerPe, true);

    MPI_Barrier(MPI_COMM_WORLD);

    // Global statistics
    double global_max_time = 0.0;
    double global_min_time = 0.0;
    double global_sum_time = 0.0;

    MPI_Reduce(&avg_time, &global_max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&avg_time, &global_min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&avg_time, &global_sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    int local_success_int = success ? 1 : 0;
    int global_success_sum = 0;
    MPI_Reduce(&local_success_int, &global_success_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myPe == 0) {
        double global_avg_time = global_sum_time / npes;

        printf("\n=== Global Performance Statistics ===\n");
        printf("Min avg time across PEs: %.6f seconds\n", global_min_time);
        printf("Max avg time across PEs: %.6f seconds\n", global_max_time);
        printf("Avg time across PEs:     %.6f seconds\n", global_avg_time);

        double algo_bandwidth = bytesPerPe / global_avg_time / (1024.0 * 1024.0 * 1024.0);
        double bus_bandwidth = algo_bandwidth * 2.0 * (npes - 1) / npes;
        printf("Algo bandwidth: %.2f GB/s (data size: %.3f GB)\n",
               algo_bandwidth, bytesPerPe / (1024.0 * 1024.0 * 1024.0));
        printf("Bus  bandwidth: %.2f GB/s (factor: 2*(N-1)/N = %.2f)\n",
               bus_bandwidth, 2.0 * (npes - 1) / npes);

        printf("\n=== Global Verification Results ===\n");
        printf("PEs passed: %d/%d\n", global_success_sum, npes);

        if (global_success_sum == npes) {
            printf("\n=== AllReduce Test Completed Successfully! ===\n");
        } else {
            printf("\n=== AllReduce Test FAILED! ===\n");
        }
    }

    // Cleanup
    allreduce_obj.reset();
    CHECK_HIP(hipFree(outPutBuff));
    CHECK_HIP(hipFree(inPutBuff));
    CHECK_HIP(hipStreamDestroy(stream));

    MPI_Barrier(MPI_COMM_WORLD);
    ShmemFinalize();
}

int main(int argc, char* argv[]) {
    testAllreduceSdma();
    return 0;
}
