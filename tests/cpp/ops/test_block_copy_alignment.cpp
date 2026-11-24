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
//
// Two-GPU BlockCopy alignment sweep over P2P symmetric memory. Requires MPI,
// shmem, and exactly two PEs. Not registered with ctest (needs two GPUs).
//
//   cmake -B build -DBUILD_TESTS=ON -DWITH_MPI=ON
//   cmake --build build --target test_block_copy_alignment
//   MORI_SHMEM_HEAP_SIZE=6G mpirun -np 2 ./build/tests/cpp/test_block_copy_alignment

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

constexpr size_t kDefaultCopyBytes = 2ULL << 30;
constexpr int kGridSize = 4;
constexpr int kBlockSize = 256;
constexpr size_t kCopyTileBytes = 1ULL << 20;
constexpr uint32_t kAlignments[] = {1, 2, 3, 4, 5, 6, 7, 8, 16};

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

// Test-local fixed-width copy. This is not mori::core::BlockCopy: every thread
// issues VecSize-wide load/store with a scalar tail, so the alignment cases
// below measure that access width rather than the library peel.
template <int VecSize, MemAddressSpace Space>
__device__ __forceinline__ void BlockCopy(uint8_t* __restrict__ dst,
                                          const uint8_t* __restrict__ src, size_t bytes) {
  constexpr size_t B = VecSize;
  using DataType = VecTypeT<B>;
  const int threadId = FlatBlockThreadId();
  const int blockSize = FlatBlockSize();
  const size_t bytesPerIteration = static_cast<size_t>(blockSize) * B;

  size_t offset = 0;
  const size_t iterations = bytes / bytesPerIteration;
  for (size_t iteration = 0; iteration < iterations; ++iteration) {
    const size_t threadOffset = offset + static_cast<size_t>(threadId) * B;
    const DataType value = load<B, Space>(src + threadOffset);
    store<B, Space>(dst + threadOffset, value);
    offset += bytesPerIteration;
  }

  offset += threadId;
  while (offset < bytes) {
    dst[offset] = src[offset];
    offset += blockSize;
  }
}

template <int VecSize, bool ForceFlatLdSt>
__global__ void BlockCopyKernel(int myPe, int npes, const SymmMemObjPtr srcMemObj,
                                const SymmMemObjPtr dstMemObj, size_t srcOffset, size_t dstOffset,
                                size_t bytes, uint32_t expectedMod16, int* configError) {
  const int targetPe = (myPe + 1) % npes;

  const auto* src = srcMemObj->template GetAs<const uint8_t*>() + srcOffset;
  auto* dst = dstMemObj->template GetAs<uint8_t*>(targetPe);
  if (dst != nullptr) dst += dstOffset;

  const bool valid = dst != nullptr && (reinterpret_cast<uintptr_t>(src) & 0xFU) == expectedMod16 &&
                     (reinterpret_cast<uintptr_t>(dst) & 0xFU) == expectedMod16;
  if (!valid) {
    if (FlatBlockThreadId() == 0) atomicExch(configError, 1);
    return;
  }

  // "Poison" src/dst pointers to force flat load/store.
  if (ForceFlatLdSt && bytes * npes == 0xACEBABEFEEDDEADBULL) {
    src = reinterpret_cast<const uint8_t*>(srcMemObj.cpu);
    dst = reinterpret_cast<uint8_t*>(dstMemObj.cpu);
  }

  const size_t gridStride = static_cast<size_t>(gridDim.x) * kCopyTileBytes;
  for (size_t tileOffset = static_cast<size_t>(blockIdx.x) * kCopyTileBytes; tileOffset < bytes;
       tileOffset += gridStride) {
    const size_t remaining = bytes - tileOffset;
    const size_t tileBytes = remaining < kCopyTileBytes ? remaining : kCopyTileBytes;
    if constexpr (ForceFlatLdSt) {
      BlockCopy<VecSize, EAddrAuto>(dst + tileOffset, src + tileOffset, tileBytes);
    } else {
      BlockCopy<VecSize, EAddrGlobal>(dst + tileOffset, src + tileOffset, tileBytes);
    }
  }
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
        printf("Mismatch at byte %zu: got 0x%02x, expected 0x%02x from PE %d\n", index,
               static_cast<uint32_t>(actual), static_cast<uint32_t>(expected), sourcePe);
      }
      return;
    }
  }
}

size_t CopyBytes() {
  const char* value = std::getenv("MORI_BLOCKCOPY_TEST_BYTES");
  if (value == nullptr) return kDefaultCopyBytes;

  char* end = nullptr;
  const uint64_t parsed = std::strtoull(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0) {
    std::fprintf(stderr, "Invalid MORI_BLOCKCOPY_TEST_BYTES value: %s\n", value);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<size_t>(parsed);
}

uint32_t AlignmentOffset(uint32_t alignment) { return alignment == 16 ? 0u : alignment; }

struct CaseResult {
  float milliseconds;
  int errors;
};

template <int VecSize, bool ForceFlatLdSt = false>
CaseResult RunCase(const char* name, uint32_t offset, size_t bytes, int myPe, int npes,
                   uint8_t* srcBase, uint8_t* dstBase, const SymmMemObjPtr srcMemObj,
                   const SymmMemObjPtr dstMemObj, int* deviceError) {
  auto* src = srcBase + offset;
  auto* dst = dstBase + offset;
  const uint32_t expectedMod16 = offset & 0xFU;

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
  const size_t tileCount = (bytes - 1) / kCopyTileBytes + 1;
  const uint32_t copyGridSize =
      tileCount < static_cast<size_t>(kGridSize) ? static_cast<uint32_t>(tileCount) : kGridSize;
  BlockCopyKernel<VecSize, ForceFlatLdSt><<<copyGridSize, kBlockSize>>>(
      myPe, npes, srcMemObj, dstMemObj, offset, offset, bytes, expectedMod16, deviceError);
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
    std::fprintf(stderr, "BlockCopy alignment test requires at least two GPUs\n");
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
    if (myPe == 0) std::fprintf(stderr, "BlockCopy alignment test requires exactly two PEs\n");
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
    std::printf("BlockCopy peer-copy alignment test: %zu bytes, %s, %d blocks x %d threads\n",
                bytes, properties.name, kGridSize, kBlockSize);
    std::fflush(stdout);
  }

  int errors = 0;
  float aligned16Ms = 0.0f;
  float align8Ms = 0.0f;
  float align1Ms = 0.0f;
  for (uint32_t alignment : kAlignments) {
    const uint32_t offset = AlignmentOffset(alignment);
    char name[64];
    std::snprintf(name, sizeof(name), "BlockCopy, %u-byte align", alignment);
    const CaseResult result = RunCase<16>(name, offset, bytes, myPe, npes, srcBase, dstBase,
                                          srcMemObj, dstMemObj, deviceError);
    errors += result.errors;
    if (alignment == 16) aligned16Ms = result.milliseconds;
    if (alignment == 8) align8Ms = result.milliseconds;
    if (alignment == 1) align1Ms = result.milliseconds;
  }

  const CaseResult misalignedB64 = RunCase<8>("ld/st 8b, 8b align", 8, bytes, myPe, npes, srcBase,
                                              dstBase, srcMemObj, dstMemObj, deviceError);
  const CaseResult misalignedB64flat =
      RunCase<8, true>("flat ld/st 8b, 8b align", 8, bytes, myPe, npes, srcBase, dstBase, srcMemObj,
                       dstMemObj, deviceError);
  errors += misalignedB64.errors + misalignedB64flat.errors;

  if (myPe == 0 && aligned16Ms > 0.0f) {
    if (align8Ms > 0.0f) {
      std::printf("8-byte/16-byte align BlockCopy runtime ratio: %.3fx\n", align8Ms / aligned16Ms);
    }
    if (align1Ms > 0.0f) {
      std::printf("1-byte/16-byte align BlockCopy runtime ratio: %.3fx\n", align1Ms / aligned16Ms);
    }
    if (misalignedB64.milliseconds > 0.0f && align8Ms > 0.0f) {
      std::printf("16-byte-op/8-byte-op runtime ratio at 8-byte alignment: %.3fx\n",
                  align8Ms / misalignedB64.milliseconds);
    }
  }

  HIP_RUNTIME_CHECK(hipFree(deviceError));
  ShmemFree(dstBase);
  ShmemFree(srcBase);
  MPI_Comm_free(&localComm);
  ShmemFinalize();
  return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
