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

void testAllreduceSdmaSync() {
    int status;

    MPI_Init(NULL, NULL);
    status = ShmemMpiInit(MPI_COMM_WORLD);
    assert(!status);

    int myPe = ShmemMyPe();
    int npes = ShmemNPes();

    printf("PE %d of %d started\n", myPe, npes);

    const int elemsPerPe = 8 * 1024 * 1024;
    const size_t bytesPerPe = elemsPerPe * sizeof(uint32_t);
    const size_t totalBytes = bytesPerPe * npes;

    uint32_t* inPutBuff = nullptr;
    CHECK_HIP(hipMalloc(&inPutBuff, bytesPerPe));

    uint32_t* outPutBuff = nullptr;
    CHECK_HIP(hipMalloc(&outPutBuff, bytesPerPe));

    // Data init: each PE fills all elements with (myPe + 1)
    std::vector<uint32_t> hostData(elemsPerPe);
    uint32_t fillValue = static_cast<uint32_t>(myPe + 1);
    std::fill(hostData.begin(), hostData.end(), fillValue);

    if (myPe == 0) {
        uint32_t expected = static_cast<uint32_t>(npes * (npes + 1) / 2);
        printf("\n=== AllReduce Sync Test ===\n");
        printf("  Elements per PE : %d\n", elemsPerPe);
        printf("  Data size       : %.2f MB per PE\n", bytesPerPe / (1024.0 * 1024.0));
        printf("  Each PE fills   : (PE_id + 1)\n");
        printf("  Expected result : %u\n\n", expected);
    }

    printf("PE %d: Input = all %u\n", myPe, fillValue);

    CHECK_HIP(hipMemcpy(inPutBuff, hostData.data(), bytesPerPe, hipMemcpyHostToDevice));
    CHECK_HIP(hipDeviceSynchronize());

    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));
    MPI_Barrier(MPI_COMM_WORLD);

    // Create AllreduceSdma object
    std::unique_ptr<AllreduceSdma<uint32_t>> allreduce_obj;
    allreduce_obj = std::make_unique<AllreduceSdma<uint32_t>>(
        myPe, npes, bytesPerPe, totalBytes);

    printf("PE %d: AllreduceSdma created\n", myPe);

    // Warmup + measurement using synchronous operator()
    const int num_iterations = 10;
    const int warmup_iterations = 10;
    std::vector<double> exec_times;

    if (myPe == 0) {
        printf("\nUsing SYNC mode (operator())\n");
        printf("Warmup: %d, Measurement: %d\n\n", warmup_iterations, num_iterations);
    }

    for (int i = 0; i < num_iterations + warmup_iterations; i++) {
        MPI_Barrier(MPI_COMM_WORLD);

        double start_time = MPI_Wtime();

        bool success = (*allreduce_obj)(inPutBuff, outPutBuff, elemsPerPe, stream);

        CHECK_HIP(hipStreamSynchronize(stream));

        double end_time = MPI_Wtime();
        double iter_time = end_time - start_time;

        if (!success) {
            fprintf(stderr, "PE %d: AllReduce failed at iteration %d\n", myPe, i);
            break;
        }

        if (i >= warmup_iterations) {
            exec_times.push_back(iter_time);
            if (myPe == 0 && exec_times.size() == 1) {
                printf("PE %d: First measurement: %.6f s\n", myPe, iter_time);
            }
        } else if (myPe == 0) {
            printf("PE %d: Warmup %d: %.6f s\n", myPe, i + 1, iter_time);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Local statistics
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
            printf("\nPE %d local statistics (%zu iterations):\n", myPe, exec_times.size());
            printf("  Min: %.6f s  Max: %.6f s  Avg: %.6f s\n", min_time, max_time, avg_time);
        }
    }

    // Verify
    std::vector<uint32_t> resultData(elemsPerPe);
    CHECK_HIP(hipMemcpy(resultData.data(), outPutBuff, bytesPerPe, hipMemcpyDeviceToHost));
    CHECK_HIP(hipDeviceSynchronize());

    uint32_t expected_value = static_cast<uint32_t>(npes * (npes + 1) / 2);
    bool success = true;
    for (size_t i = 0; i < static_cast<size_t>(elemsPerPe); i++) {
        if (resultData[i] != expected_value) {
            printf("PE %d: FAILED at [%zu]: expected %u, got %u\n",
                   myPe, i, expected_value, resultData[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("PE %d: Verification PASSED (all %d elements = %u)\n",
               myPe, elemsPerPe, expected_value);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Global statistics
    double global_max_time = 0.0, global_min_time = 0.0, global_sum_time = 0.0;
    MPI_Reduce(&avg_time, &global_max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&avg_time, &global_min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&avg_time, &global_sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    int local_ok = success ? 1 : 0;
    int global_ok = 0;
    MPI_Reduce(&local_ok, &global_ok, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myPe == 0) {
        double global_avg = global_sum_time / npes;
        double algo_bw = bytesPerPe / global_avg / (1024.0 * 1024.0 * 1024.0);
        double bus_bw = algo_bw * 2.0 * (npes - 1) / npes;

        printf("\n=== Global Performance Statistics ===\n");
        printf("Min avg time: %.6f s\n", global_min_time);
        printf("Max avg time: %.6f s\n", global_max_time);
        printf("Avg time:     %.6f s\n", global_avg);
        printf("Algo bandwidth: %.2f GB/s (data: %.3f GB)\n",
               algo_bw, bytesPerPe / (1024.0 * 1024.0 * 1024.0));
        printf("Bus  bandwidth: %.2f GB/s (factor: 2*(N-1)/N = %.2f)\n",
               bus_bw, 2.0 * (npes - 1) / npes);

        printf("\nPEs passed: %d/%d\n", global_ok, npes);
        if (global_ok == npes) {
            printf("\n=== AllReduce Sync Test PASSED ===\n");
        } else {
            printf("\n=== AllReduce Sync Test FAILED ===\n");
        }
    }

    allreduce_obj.reset();
    CHECK_HIP(hipFree(outPutBuff));
    CHECK_HIP(hipFree(inPutBuff));
    CHECK_HIP(hipStreamDestroy(stream));

    MPI_Barrier(MPI_COMM_WORLD);
    ShmemFinalize();
}

int main(int argc, char* argv[]) {
    testAllreduceSdmaSync();
    return 0;
}
