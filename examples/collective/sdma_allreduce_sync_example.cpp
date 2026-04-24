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

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <memory>
#include <numeric>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/collective/allreduce/twoshot_allreduce_sdma_class.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::application;
using namespace mori::shmem;
using namespace mori::collective;

#define CHECK_HIP(call)                                                                        \
  do {                                                                                         \
    hipError_t err = (call);                                                                   \
    if (err != hipSuccess) {                                                                   \
      fprintf(stderr, "HIP Error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
      throw std::runtime_error("HIP call failed");                                             \
    }                                                                                          \
  } while (0)

static uint32_t computeExpected(int npes) {
  uint32_t sum = 0;
  for (int pe = 0; pe < npes; pe++) sum += (pe + 1) * 1000;
  return sum;
}

static bool verifyResult(const uint32_t* data, size_t elems, uint32_t expected, int myPe) {
  for (size_t i = 0; i < elems; i++) {
    if (data[i] != expected) {
      printf("PE %d: FAILED at [%zu]: expected %u, got %u\n", myPe, i, expected, data[i]);
      return false;
    }
  }
  printf("PE %d: Verification PASSED (all %zu elements = %u)\n", myPe, elems, expected);
  return true;
}

static void printStats(const std::vector<double>& times, size_t bytesPerPe, int npes, int myPe) {
  double avg = 0, mn = times[0], mx = times[0];
  for (double t : times) {
    avg += t;
    mn = std::min(mn, t);
    mx = std::max(mx, t);
  }
  avg /= times.size();

  double g_max = 0, g_min = 0, g_sum = 0;
  MPI_Reduce(&avg, &g_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&avg, &g_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&avg, &g_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (myPe == 0) {
    double g_avg = g_sum / npes;
    double algo_bw = bytesPerPe / g_avg / (1024.0 * 1024.0 * 1024.0);
    double bus_bw = algo_bw * 2.0 * (npes - 1) / npes;
    printf("  Min time : %.3f ms\n", g_min * 1000);
    printf("  Max time : %.3f ms\n", g_max * 1000);
    printf("  Avg time : %.3f ms\n", g_avg * 1000);
    printf("  Algo BW  : %.2f GB/s\n", algo_bw);
    printf("  Bus BW   : %.2f GB/s\n", bus_bw);
    printf("  Data size: %.2f MB per rank\n", bytesPerPe / (1024.0 * 1024.0));
  }
}

// =========================================================================
// Test 1: Out-of-place (copy_output_to_user=False, read from transit buffer)
// =========================================================================
static bool testOutplace(int myPe, int npes, int elemsPerPe, size_t bytesPerPe,
                         size_t outputBufSize, uint32_t fillValue, hipStream_t stream,
                         int iterations, int warmup) {
  if (myPe == 0) printf("\n>>> Test 1: Out-of-place (copy_output_to_user=False)\n");

  auto ar = std::make_unique<AllreduceSdma<uint32_t>>(myPe, npes, bytesPerPe, outputBufSize, false);

  uint32_t* inBuf = nullptr;
  uint32_t* outBuf = nullptr;
  CHECK_HIP(hipMalloc(&inBuf, bytesPerPe));
  CHECK_HIP(hipMalloc(&outBuf, bytesPerPe));

  std::vector<uint32_t> hostIn(elemsPerPe, fillValue);
  CHECK_HIP(hipMemcpy(inBuf, hostIn.data(), bytesPerPe, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemset(outBuf, 0, bytesPerPe));
  CHECK_HIP(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<double> times;
  for (int i = 0; i < warmup + iterations; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    bool ok = (*ar)(inBuf, outBuf, elemsPerPe, stream);
    CHECK_HIP(hipStreamSynchronize(stream));
    double t1 = MPI_Wtime();
    if (!ok) {
      fprintf(stderr, "PE %d: failed at iter %d\n", myPe, i);
      break;
    }
    if (i >= warmup)
      times.push_back(t1 - t0);
    else if (myPe == 0)
      printf("  Warmup %d/%d: %.3f ms\n", i + 1, warmup, (t1 - t0) * 1000);
  }

  // Verify from transit buffer (outBuf should remain zeros)
  void* transitPtr = ar->getOutputTransitBuffer();
  std::vector<uint32_t> transitData(elemsPerPe);
  CHECK_HIP(hipMemcpy(transitData.data(), transitPtr, bytesPerPe, hipMemcpyDeviceToHost));
  bool ok = verifyResult(transitData.data(), elemsPerPe, computeExpected(npes), myPe);

  std::vector<uint32_t> outData(elemsPerPe);
  CHECK_HIP(hipMemcpy(outData.data(), outBuf, bytesPerPe, hipMemcpyDeviceToHost));
  bool allZero = std::all_of(outData.begin(), outData.end(), [](uint32_t v) { return v == 0; });
  if (allZero && myPe == 0)
    printf("  PE %d: output_tensor correctly untouched (all zeros)\n", myPe);

  MPI_Barrier(MPI_COMM_WORLD);
  if (myPe == 0) printf("\n--- Out-of-place Performance ---\n");
  if (!times.empty()) printStats(times, bytesPerPe, npes, myPe);

  ar.reset();
  CHECK_HIP(hipFree(inBuf));
  CHECK_HIP(hipFree(outBuf));
  return ok;
}

// =========================================================================
// Test 2: In-place (allreduce_inplace)
// =========================================================================
static bool testInplace(int myPe, int npes, int elemsPerPe, size_t bytesPerPe, size_t outputBufSize,
                        uint32_t fillValue, hipStream_t stream, int iterations, int warmup) {
  if (myPe == 0) printf("\n>>> Test 2: In-place (allreduce_inplace)\n");

  auto ar = std::make_unique<AllreduceSdma<uint32_t>>(myPe, npes, bytesPerPe, outputBufSize, false);

  uint32_t* buf = nullptr;
  CHECK_HIP(hipMalloc(&buf, bytesPerPe));

  std::vector<uint32_t> hostData(elemsPerPe, fillValue);
  CHECK_HIP(hipMemcpy(buf, hostData.data(), bytesPerPe, hipMemcpyHostToDevice));
  CHECK_HIP(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  // Single-shot correctness check
  ar->allreduce_inplace(buf, elemsPerPe, stream);
  CHECK_HIP(hipStreamSynchronize(stream));

  std::vector<uint32_t> result(elemsPerPe);
  CHECK_HIP(hipMemcpy(result.data(), buf, bytesPerPe, hipMemcpyDeviceToHost));
  bool ok = verifyResult(result.data(), elemsPerPe, computeExpected(npes), myPe);
  if (ok && myPe == 0) printf("  PE %d: inplace result verified\n", myPe);

  MPI_Barrier(MPI_COMM_WORLD);

  // Benchmark with refill each iteration
  std::vector<double> times;
  for (int i = 0; i < warmup + iterations; i++) {
    CHECK_HIP(hipMemcpy(buf, hostData.data(), bytesPerPe, hipMemcpyHostToDevice));
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    ar->allreduce_inplace(buf, elemsPerPe, stream);
    CHECK_HIP(hipStreamSynchronize(stream));
    double t1 = MPI_Wtime();
    if (i >= warmup)
      times.push_back(t1 - t0);
    else if (myPe == 0)
      printf("  Warmup %d/%d: %.3f ms\n", i + 1, warmup, (t1 - t0) * 1000);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (myPe == 0) printf("\n--- In-place Performance ---\n");
  if (!times.empty()) printStats(times, bytesPerPe, npes, myPe);

  ar.reset();
  CHECK_HIP(hipFree(buf));
  return ok;
}

void testAllreduceSdmaSync() {
  MPI_Init(NULL, NULL);
  int status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  const int elemsPerPe = 8 * 1024 * 1024;
  const size_t bytesPerPe = elemsPerPe * sizeof(uint32_t);
  const size_t outputBufSize =
      static_cast<size_t>(npes) * (elemsPerPe / npes + 64) * sizeof(uint32_t);
  const uint32_t fillValue = static_cast<uint32_t>((myPe + 1) * 1000);

  if (myPe == 0) {
    printf("\n======================================================================\n");
    printf("AllReduce Sync Test\n");
    printf("  World size      : %d\n", npes);
    printf("  Elements per PE : %d\n", elemsPerPe);
    printf("  Data size       : %.2f MB per rank\n", bytesPerPe / (1024.0 * 1024.0));
    printf("  Fill value      : (PE_id + 1) * 1000\n");
    printf("  Expected result : %u\n", computeExpected(npes));
    printf("======================================================================\n");
  }
  printf("PE %d: Input = all %u\n", myPe, fillValue);

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));
  MPI_Barrier(MPI_COMM_WORLD);

  const int iterations = 10;
  const int warmup = 10;

  bool ok1 = testOutplace(myPe, npes, elemsPerPe, bytesPerPe, outputBufSize, fillValue, stream,
                          iterations, warmup);
  bool ok2 = testInplace(myPe, npes, elemsPerPe, bytesPerPe, outputBufSize, fillValue, stream,
                         iterations, warmup);

  bool allOk = ok1 && ok2;
  int localOk = allOk ? 1 : 0, globalOk = 0;
  MPI_Reduce(&localOk, &globalOk, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (myPe == 0) {
    printf("\n======================================================================\n");
    printf("PEs passed: %d/%d\n", globalOk, npes);
    if (globalOk == npes)
      printf("=== All Tests PASSED ===\n");
    else
      printf("=== SOME Tests FAILED ===\n");
    printf("======================================================================\n\n");
  }

  CHECK_HIP(hipStreamDestroy(stream));
  MPI_Barrier(MPI_COMM_WORLD);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  testAllreduceSdmaSync();
  return 0;
}
