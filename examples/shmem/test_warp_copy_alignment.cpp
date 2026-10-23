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

#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "mori/application/utils/check.hpp"
#include "mori/core/core.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::application;
using namespace mori::core;
using namespace mori::shmem;

namespace {

constexpr size_t kDefaultCopyBytes = 1ULL << 30;
constexpr int kGridSize = 256;
constexpr int kBlockSize = 256;

__host__ __device__ uint8_t ExpectedByte(size_t index, int sourcePe) {
  return static_cast<uint8_t>(
             ((index * 131U) ^ (static_cast<size_t>(sourcePe) * 17U) ^ (index >> 8)) & 0xFFU) |
         1U;
}

__global__ void InitPatternKernel(uint8_t* buffer, size_t bytes, int sourcePe) {
  const size_t threadId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t index = threadId; index < bytes; index += stride) {
    buffer[index] = ExpectedByte(index, sourcePe);
  }
}

__global__ void WarpCopyKernel(int myPe, int npes, const SymmMemObjPtr srcMemObj,
                               const SymmMemObjPtr dstMemObj, size_t srcOffset, size_t dstOffset,
                               size_t bytes, unsigned expectedMod16, int* configError) {
  const int laneId = threadIdx.x & (warpSize - 1);
  const int targetPe = (myPe + 1) % npes;

  const auto* src = srcMemObj->template GetAs<const uint8_t*>() + srcOffset;
  auto* dst = dstMemObj->template GetAs<uint8_t*>(targetPe);
  if (dst != nullptr) dst += dstOffset;

  const bool valid = dst != nullptr && (reinterpret_cast<uintptr_t>(src) & 0xFU) == expectedMod16 &&
                     (reinterpret_cast<uintptr_t>(dst) & 0xFU) == expectedMod16;
  if (!valid) {
    if (laneId == 0) atomicExch(configError, 1);
    return;
  }

  WarpCopy<uint8_t>(dst, src, bytes);
  __threadfence_system();
}

__global__ void CheckPatternKernel(const uint8_t* buffer, size_t bytes, int sourcePe,
                                   int* mismatch) {
  const size_t threadId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t index = threadId; index < bytes; index += stride) {
    const uint8_t expected = ExpectedByte(index, sourcePe);
    const uint8_t actual = buffer[index];
    if (actual != expected) {
      if (atomicCAS(mismatch, 0, 1) == 0) {
        printf("Mismatch at byte %llu: got 0x%02x, expected 0x%02x from PE %d\n",
               static_cast<unsigned long long>(index), static_cast<unsigned>(actual),
               static_cast<unsigned>(expected), sourcePe);
      }
      return;
    }
  }
}

size_t CopyBytes() {
  const char* value = std::getenv("MORI_WARPCOPY_TEST_BYTES");
  if (value == nullptr) return kDefaultCopyBytes;

  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0) {
    std::fprintf(stderr, "Invalid MORI_WARPCOPY_TEST_BYTES value: %s\n", value);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<size_t>(parsed);
}

struct CaseResult {
  float milliseconds;
  int errors;
};

CaseResult RunCase(const char* name, unsigned offset, size_t bytes, int myPe, int npes,
                   int warpSizeHost, uint8_t* srcBase, uint8_t* dstBase,
                   const SymmMemObjPtr srcMemObj, const SymmMemObjPtr dstMemObj, int* deviceError) {
  auto* src = srcBase + offset;
  auto* dst = dstBase + offset;
  const unsigned expectedMod16 = offset & 0xFU;

  InitPatternKernel<<<kGridSize, kBlockSize>>>(src, bytes, myPe);
  HIP_RUNTIME_CHECK(hipMemset(dstBase, 0, bytes + 16));
  HIP_RUNTIME_CHECK(hipMemset(deviceError, 0, sizeof(int)));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  hipEvent_t start;
  hipEvent_t stop;
  HIP_RUNTIME_CHECK(hipEventCreate(&start));
  HIP_RUNTIME_CHECK(hipEventCreate(&stop));
  HIP_RUNTIME_CHECK(hipEventRecord(start));
  WarpCopyKernel<<<1, warpSizeHost>>>(myPe, npes, srcMemObj, dstMemObj, offset, offset, bytes,
                                      expectedMod16, deviceError);
  HIP_RUNTIME_CHECK(hipGetLastError());
  HIP_RUNTIME_CHECK(hipEventRecord(stop));
  HIP_RUNTIME_CHECK(hipEventSynchronize(stop));

  float elapsedMs = 0.0f;
  HIP_RUNTIME_CHECK(hipEventElapsedTime(&elapsedMs, start, stop));
  HIP_RUNTIME_CHECK(hipEventDestroy(start));
  HIP_RUNTIME_CHECK(hipEventDestroy(stop));

  MPI_Barrier(MPI_COMM_WORLD);

  const int sourcePe = (myPe - 1 + npes) % npes;
  CheckPatternKernel<<<kGridSize, kBlockSize>>>(dst, bytes, sourcePe, deviceError);
  HIP_RUNTIME_CHECK(hipGetLastError());
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  int localErrors = 0;
  HIP_RUNTIME_CHECK(hipMemcpy(&localErrors, deviceError, sizeof(int), hipMemcpyDeviceToHost));
  int globalErrors = 0;
  MPI_Allreduce(&localErrors, &globalErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  float maxElapsedMs = 0.0f;
  MPI_Reduce(&elapsedMs, &maxElapsedMs, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
  if (myPe == 0) {
    const double bandwidthGbPerSec =
        static_cast<double>(bytes) / (static_cast<double>(maxElapsedMs) * 1.0e6);
    std::printf("%s: %.3f ms, %.3f GB/s, correctness %s\n", name, maxElapsedMs, bandwidthGbPerSec,
                globalErrors == 0 ? "PASS" : "FAIL");
    std::fflush(stdout);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return {maxElapsedMs, globalErrors};
}

}  // namespace

int main() {
  MPI_Init(nullptr, nullptr);

  MPI_Comm localComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
  int localRank = 0;
  MPI_Comm_rank(localComm, &localRank);

  int deviceCount = 0;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    std::fprintf(stderr, "WarpCopy alignment test requires at least two GPUs\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  const int deviceId = localRank % deviceCount;
  HIP_RUNTIME_CHECK(hipSetDevice(deviceId));
  if (ShmemMpiInit(MPI_COMM_WORLD) != 0) {
    std::fprintf(stderr, "ShmemMpiInit failed\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  const int myPe = ShmemMyPe();
  const int npes = ShmemNPes();
  if (npes != 2) {
    if (myPe == 0) std::fprintf(stderr, "WarpCopy alignment test requires exactly two PEs\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  hipDeviceProp_t properties{};
  HIP_RUNTIME_CHECK(hipGetDeviceProperties(&properties, deviceId));

  const size_t bytes = CopyBytes();
  constexpr size_t kAlignmentPadding = 16;
  auto* srcBase = static_cast<uint8_t*>(ShmemMalloc(bytes + kAlignmentPadding));
  auto* dstBase = static_cast<uint8_t*>(ShmemMalloc(bytes + kAlignmentPadding));
  if (srcBase == nullptr || dstBase == nullptr) {
    std::fprintf(stderr, "PE %d failed to allocate two %zu-byte symmetric buffers\n", myPe,
                 bytes + kAlignmentPadding);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  SymmMemObjPtr srcMemObj = ShmemQueryMemObjPtr(srcBase);
  SymmMemObjPtr dstMemObj = ShmemQueryMemObjPtr(dstBase);
  if (!srcMemObj.IsValid() || !dstMemObj.IsValid()) {
    std::fprintf(stderr, "PE %d failed to query symmetric memory objects\n", myPe);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  int* deviceError = nullptr;
  HIP_RUNTIME_CHECK(hipMalloc(&deviceError, sizeof(int)));

  if (myPe == 0) {
    std::printf("WarpCopy peer-copy alignment test: %zu bytes, %s, warp size %d\n", bytes,
                properties.name, properties.warpSize);
    std::fflush(stdout);
  }

  const CaseResult aligned = RunCase("16-byte aligned", 0, bytes, myPe, npes, properties.warpSize,
                                     srcBase, dstBase, srcMemObj, dstMemObj, deviceError);
  const CaseResult misaligned = RunCase("8-byte aligned", 8, bytes, myPe, npes, properties.warpSize,
                                        srcBase, dstBase, srcMemObj, dstMemObj, deviceError);

  if (myPe == 0 && aligned.milliseconds > 0.0f) {
    std::printf("8-byte / 16-byte runtime ratio: %.3fx\n",
                misaligned.milliseconds / aligned.milliseconds);
  }

  const int result = aligned.errors + misaligned.errors;
  HIP_RUNTIME_CHECK(hipFree(deviceError));
  ShmemFree(dstBase);
  ShmemFree(srcBase);
  MPI_Comm_free(&localComm);
  ShmemFinalize();
  return result == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
