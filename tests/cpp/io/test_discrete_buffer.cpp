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

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "src/io/xgmi/scatter_gather_kernel.hpp"

#define HIP_CHECK(cmd)                                                                         \
  do {                                                                                         \
    hipError_t err = (cmd);                                                                    \
    if (err != hipSuccess) {                                                                   \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
      assert(false);                                                                           \
    }                                                                                          \
  } while (0)

// --------------------------------------------------------------------------
// Naive baseline: one hipMemcpyAsync per segment
// --------------------------------------------------------------------------
static void NaiveBatchCopy(char* dst, const char* src, const std::vector<size_t>& srcOffsets,
                           const std::vector<size_t>& dstOffsets, const std::vector<size_t>& sizes,
                           hipStream_t stream) {
  for (size_t i = 0; i < sizes.size(); ++i) {
    HIP_CHECK(hipMemcpyAsync(dst + dstOffsets[i], src + srcOffsets[i], sizes[i],
                             hipMemcpyDeviceToDevice, stream));
  }
}

// --------------------------------------------------------------------------
// Optimized: single scatter/gather kernel launch
// --------------------------------------------------------------------------
static void KernelBatchCopy(char* dst, const char* src, const std::vector<size_t>& srcOffsets,
                            const std::vector<size_t>& dstOffsets, const std::vector<size_t>& sizes,
                            size_t* dMeta, hipStream_t stream) {
  int n = static_cast<int>(sizes.size());
  size_t metaBytes = n * sizeof(size_t);

  std::vector<size_t> hostMeta(n * 3);
  std::copy(srcOffsets.begin(), srcOffsets.end(), hostMeta.begin());
  std::copy(dstOffsets.begin(), dstOffsets.end(), hostMeta.begin() + n);
  std::copy(sizes.begin(), sizes.end(), hostMeta.begin() + n * 2);

  HIP_CHECK(hipMemcpyAsync(dMeta, hostMeta.data(), metaBytes * 3, hipMemcpyHostToDevice, stream));

  size_t* dSrcOff = dMeta;
  size_t* dDstOff = dMeta + n;
  size_t* dSizes = dMeta + n * 2;

  int threads = 256;
  int blocks = std::min(n, 1024);
  mori::io::scatterGatherCopyKernel<<<blocks, threads, 0, stream>>>(src, dst, dSrcOff, dDstOff,
                                                                    dSizes, n);
  HIP_CHECK(hipGetLastError());
}

// --------------------------------------------------------------------------
// TestCorrectness: verify kernel produces identical results to hipMemcpy
// --------------------------------------------------------------------------
static void TestCorrectness() {
  printf("=== TestCorrectness ===\n");

  const size_t bufSize = 64 * 1024 * 1024;
  const int numSegments = 256;
  const size_t segSize = 4096;
  const size_t stride = segSize * 4;

  char *srcGpu, *dstNaive, *dstKernel;
  HIP_CHECK(hipMalloc(&srcGpu, bufSize));
  HIP_CHECK(hipMalloc(&dstNaive, bufSize));
  HIP_CHECK(hipMalloc(&dstKernel, bufSize));

  std::vector<char> srcHost(bufSize);
  for (size_t i = 0; i < bufSize; i++) srcHost[i] = static_cast<char>(i % 251);
  HIP_CHECK(hipMemcpy(srcGpu, srcHost.data(), bufSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dstNaive, 0, bufSize));
  HIP_CHECK(hipMemset(dstKernel, 0, bufSize));

  std::vector<size_t> srcOff(numSegments), dstOff(numSegments);
  std::vector<size_t> sizes(numSegments, segSize);
  for (int i = 0; i < numSegments; i++) {
    srcOff[i] = i * stride;
    dstOff[i] = i * stride;
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  NaiveBatchCopy(dstNaive, srcGpu, srcOff, dstOff, sizes, stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  size_t* dMeta = nullptr;
  HIP_CHECK(hipMalloc(&dMeta, numSegments * sizeof(size_t) * 3));
  KernelBatchCopy(dstKernel, srcGpu, srcOff, dstOff, sizes, dMeta, stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<char> naiveRes(bufSize), kernelRes(bufSize);
  HIP_CHECK(hipMemcpy(naiveRes.data(), dstNaive, bufSize, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(kernelRes.data(), dstKernel, bufSize, hipMemcpyDeviceToHost));

  for (int i = 0; i < numSegments; i++) {
    for (size_t j = 0; j < segSize; j++) {
      size_t off = dstOff[i] + j;
      if (naiveRes[off] != kernelRes[off]) {
        fprintf(stderr, "FAIL: mismatch at seg %d byte %zu (naive=%d kernel=%d)\n", i, j,
                (int)(unsigned char)naiveRes[off], (int)(unsigned char)kernelRes[off]);
        assert(false);
      }
    }
  }

  printf("  PASSED: %d discrete segments of %zu bytes verified\n", numSegments, segSize);

  HIP_CHECK(hipFree(srcGpu));
  HIP_CHECK(hipFree(dstNaive));
  HIP_CHECK(hipFree(dstKernel));
  HIP_CHECK(hipFree(dMeta));
  HIP_CHECK(hipStreamDestroy(stream));
}

// --------------------------------------------------------------------------
// TestCorrectnessUnaligned: verify with odd-sized, unaligned segments
// --------------------------------------------------------------------------
static void TestCorrectnessUnaligned() {
  printf("=== TestCorrectnessUnaligned ===\n");

  const size_t bufSize = 16 * 1024 * 1024;
  const int numSegments = 128;

  char *srcGpu, *dstNaive, *dstKernel;
  HIP_CHECK(hipMalloc(&srcGpu, bufSize));
  HIP_CHECK(hipMalloc(&dstNaive, bufSize));
  HIP_CHECK(hipMalloc(&dstKernel, bufSize));

  std::vector<char> srcHost(bufSize);
  std::mt19937 rng(42);
  for (size_t i = 0; i < bufSize; i++) srcHost[i] = static_cast<char>(rng() & 0xFF);
  HIP_CHECK(hipMemcpy(srcGpu, srcHost.data(), bufSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dstNaive, 0, bufSize));
  HIP_CHECK(hipMemset(dstKernel, 0, bufSize));

  std::vector<size_t> srcOff(numSegments), dstOff(numSegments), sizes(numSegments);
  std::uniform_int_distribution<size_t> sizeDist(1, 8191);
  size_t cursor = 0;
  for (int i = 0; i < numSegments; i++) {
    sizes[i] = sizeDist(rng);
    srcOff[i] = cursor;
    dstOff[i] = cursor + 37;
    cursor += sizes[i] + 127;
    if (cursor + sizes[i] + 37 >= bufSize) {
      sizes.resize(i);
      srcOff.resize(i);
      dstOff.resize(i);
      break;
    }
  }
  int actualSegs = static_cast<int>(sizes.size());

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  NaiveBatchCopy(dstNaive, srcGpu, srcOff, dstOff, sizes, stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  size_t* dMeta = nullptr;
  HIP_CHECK(hipMalloc(&dMeta, actualSegs * sizeof(size_t) * 3));
  KernelBatchCopy(dstKernel, srcGpu, srcOff, dstOff, sizes, dMeta, stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<char> naiveRes(bufSize), kernelRes(bufSize);
  HIP_CHECK(hipMemcpy(naiveRes.data(), dstNaive, bufSize, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(kernelRes.data(), dstKernel, bufSize, hipMemcpyDeviceToHost));

  for (int i = 0; i < actualSegs; i++) {
    for (size_t j = 0; j < sizes[i]; j++) {
      size_t off = dstOff[i] + j;
      if (naiveRes[off] != kernelRes[off]) {
        fprintf(stderr, "FAIL: unaligned mismatch at seg %d byte %zu (naive=%d kernel=%d)\n", i, j,
                (int)(unsigned char)naiveRes[off], (int)(unsigned char)kernelRes[off]);
        assert(false);
      }
    }
  }

  printf("  PASSED: %d unaligned segments verified\n", actualSegs);

  HIP_CHECK(hipFree(srcGpu));
  HIP_CHECK(hipFree(dstNaive));
  HIP_CHECK(hipFree(dstKernel));
  HIP_CHECK(hipFree(dMeta));
  HIP_CHECK(hipStreamDestroy(stream));
}

// --------------------------------------------------------------------------
// TestPerformance: benchmark naive vs kernel for various discrete patterns
// --------------------------------------------------------------------------
static void TestPerformance() {
  printf("\n=== TestPerformance (discrete buffer batch copy) ===\n");
  printf("%-30s %12s %12s %10s %6s\n", "TestCase", "Naive(us)", "Kernel(us)", "Speedup", "Result");
  printf(
      "----------------------------------------------------------------------"
      "------\n");

  struct TestCase {
    int numSegments;
    size_t segSize;
    const char* desc;
  };

  std::vector<TestCase> cases = {
      {32, 4096, "32 segs x 4KB"},     {64, 4096, "64 segs x 4KB"},
      {128, 4096, "128 segs x 4KB"},   {256, 4096, "256 segs x 4KB"},
      {512, 2048, "512 segs x 2KB"},   {1024, 1024, "1024 segs x 1KB"},
      {128, 16384, "128 segs x 16KB"}, {64, 65536, "64 segs x 64KB"},
  };

  bool allPassed = true;
  for (auto& tc : cases) {
    const size_t stride = tc.segSize * 4;
    const size_t bufSize = static_cast<size_t>(tc.numSegments) * stride + tc.segSize;

    char *srcGpu, *dstGpu;
    HIP_CHECK(hipMalloc(&srcGpu, bufSize));
    HIP_CHECK(hipMalloc(&dstGpu, bufSize));
    HIP_CHECK(hipMemset(srcGpu, 0xAB, bufSize));

    std::vector<size_t> srcOff(tc.numSegments), dstOff(tc.numSegments);
    std::vector<size_t> sizes(tc.numSegments, tc.segSize);
    for (int i = 0; i < tc.numSegments; i++) {
      srcOff[i] = i * stride;
      dstOff[i] = i * stride;
    }

    size_t* dMeta = nullptr;
    HIP_CHECK(hipMalloc(&dMeta, tc.numSegments * sizeof(size_t) * 3));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    const int warmup = 20;
    const int iterations = 200;

    // Warmup naive
    for (int w = 0; w < warmup; w++) {
      NaiveBatchCopy(dstGpu, srcGpu, srcOff, dstOff, sizes, stream);
      HIP_CHECK(hipStreamSynchronize(stream));
    }

    // Benchmark naive
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; it++) {
      NaiveBatchCopy(dstGpu, srcGpu, srcOff, dstOff, sizes, stream);
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double naiveUs = std::chrono::duration<double, std::micro>(t1 - t0).count() / iterations;

    // Warmup kernel
    for (int w = 0; w < warmup; w++) {
      KernelBatchCopy(dstGpu, srcGpu, srcOff, dstOff, sizes, dMeta, stream);
      HIP_CHECK(hipStreamSynchronize(stream));
    }

    // Benchmark kernel
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iterations; it++) {
      KernelBatchCopy(dstGpu, srcGpu, srcOff, dstOff, sizes, dMeta, stream);
      HIP_CHECK(hipStreamSynchronize(stream));
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double kernelUs = std::chrono::duration<double, std::micro>(t3 - t2).count() / iterations;

    double speedup = naiveUs / kernelUs;
    bool passed = speedup > 1.0;
    if (!passed) allPassed = false;

    printf("%-30s %12.2f %12.2f %9.2fx %6s\n", tc.desc, naiveUs, kernelUs, speedup,
           passed ? "PASS" : "FAIL");

    HIP_CHECK(hipFree(srcGpu));
    HIP_CHECK(hipFree(dstGpu));
    HIP_CHECK(hipFree(dMeta));
    HIP_CHECK(hipStreamDestroy(stream));
  }

  printf(
      "----------------------------------------------------------------------"
      "------\n");
  if (allPassed) {
    printf("All performance tests PASSED: kernel is faster for discrete buffers\n");
  } else {
    printf(
        "WARNING: some tests show kernel not faster - may be expected for "
        "few/large segments\n");
  }
}

// --------------------------------------------------------------------------
// TestSortMerge: verify that sorting improves merging for shuffled segments
// --------------------------------------------------------------------------
static void TestSortMerge() {
  printf("\n=== TestSortMerge ===\n");

  const int numSegments = 64;
  const size_t segSize = 4096;

  // Create segments that ARE contiguous (sequential) but arrive in random order
  std::vector<size_t> localOffsets(numSegments), remoteOffsets(numSegments);
  std::vector<size_t> sizes(numSegments, segSize);
  for (int i = 0; i < numSegments; i++) {
    localOffsets[i] = i * segSize;
    remoteOffsets[i] = i * segSize;
  }

  // Shuffle
  std::mt19937 rng(123);
  std::vector<size_t> shuffledIndices(numSegments);
  std::iota(shuffledIndices.begin(), shuffledIndices.end(), 0);
  std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), rng);

  std::vector<size_t> shuffledLocal(numSegments), shuffledRemote(numSegments);
  std::vector<size_t> shuffledSizes(numSegments);
  for (int i = 0; i < numSegments; i++) {
    shuffledLocal[i] = localOffsets[shuffledIndices[i]];
    shuffledRemote[i] = remoteOffsets[shuffledIndices[i]];
    shuffledSizes[i] = sizes[shuffledIndices[i]];
  }

  // Without sorting: count merged segments (old approach)
  int unsortedMerged = 0;
  {
    size_t lastLocalEnd = 0, lastRemoteEnd = 0;
    bool hasRun = false;
    for (int i = 0; i < numSegments; i++) {
      if (!hasRun) {
        unsortedMerged++;
        lastLocalEnd = shuffledLocal[i] + shuffledSizes[i];
        lastRemoteEnd = shuffledRemote[i] + shuffledSizes[i];
        hasRun = true;
        continue;
      }
      if (lastLocalEnd == shuffledLocal[i] && lastRemoteEnd == shuffledRemote[i]) {
        lastLocalEnd += shuffledSizes[i];
        lastRemoteEnd += shuffledSizes[i];
      } else {
        unsortedMerged++;
        lastLocalEnd = shuffledLocal[i] + shuffledSizes[i];
        lastRemoteEnd = shuffledRemote[i] + shuffledSizes[i];
      }
    }
  }

  // With sorting: sort by remote offset, then merge
  int sortedMerged = 0;
  {
    std::vector<size_t> idx(numSegments);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return shuffledRemote[a] < shuffledRemote[b]; });

    size_t lastLocalEnd = 0, lastRemoteEnd = 0;
    bool hasRun = false;
    for (int i = 0; i < numSegments; i++) {
      size_t ii = idx[i];
      if (!hasRun) {
        sortedMerged++;
        lastLocalEnd = shuffledLocal[ii] + shuffledSizes[ii];
        lastRemoteEnd = shuffledRemote[ii] + shuffledSizes[ii];
        hasRun = true;
        continue;
      }
      if (lastLocalEnd == shuffledLocal[ii] && lastRemoteEnd == shuffledRemote[ii]) {
        lastLocalEnd += shuffledSizes[ii];
        lastRemoteEnd += shuffledSizes[ii];
      } else {
        sortedMerged++;
        lastLocalEnd = shuffledLocal[ii] + shuffledSizes[ii];
        lastRemoteEnd = shuffledRemote[ii] + shuffledSizes[ii];
      }
    }
  }

  printf("  %d input segments, unsorted merge → %d, sorted merge → %d\n", numSegments,
         unsortedMerged, sortedMerged);
  assert(sortedMerged <= unsortedMerged);
  assert(sortedMerged == 1);
  printf("  PASSED: sorting reduces %d segments to %d (optimal: 1)\n", unsortedMerged,
         sortedMerged);
}

int main() {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    printf("No GPU devices found, skipping test\n");
    return 0;
  }

  HIP_CHECK(hipSetDevice(0));

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  printf("Device: %s\n\n", prop.name);

  TestCorrectness();
  TestCorrectnessUnaligned();
  TestSortMerge();
  TestPerformance();

  printf("\nAll tests completed.\n");
  return 0;
}
