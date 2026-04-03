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

// Straightforward allgather test using ShmemPutMemNbiBlock.
//
// Each of N PEs owns a chunk of data. After the allgather every PE holds
// all N chunks in a single contiguous output buffer:
//
//   output[ pe * chunkBytes .. (pe+1) * chunkBytes ) == PE pe's data
//
// Algorithm:
//   - Each PE fills its own slot in the symmetric buffer with a pattern.
//   - A GPU kernel launches N blocks. Block i puts myPe's chunk to PE i.
//   - After a global barrier every PE verifies the full buffer.
//
// Usage: mpirun --allow-run-as-root -np <N> ./allgather_test [chunk_bytes]
//   chunk_bytes defaults to 1 MB and must be a multiple of 4.

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>

#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

constexpr size_t DEFAULT_CHUNK_BYTES = 1 * 1024 * 1024;

#define XPUT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

// Fill buf[0..numElements) with pattern:  seed ^ element_index
__global__ void FillPatternKernel(uint32_t* buf, size_t numElements, uint32_t seed) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numElements;
       i += (size_t)gridDim.x * blockDim.x) {
    buf[i] = seed ^ static_cast<uint32_t>(i);
  }
}

// Block i puts myPe's chunk to PE i's output buffer at offset myPe*chunkBytes.
// Self-copy (destPe == myPe) is skipped because the data is already in place.
__global__ void AllGatherKernel(int myPe, int npes, const SymmMemObjPtr buf, size_t chunkBytes) {
  int destPe = blockIdx.x;
  if (destPe >= npes || destPe == myPe) return;

  size_t myOffset = static_cast<size_t>(myPe) * chunkBytes;

  ShmemPutMemNbiBlock(buf, myOffset, buf, myOffset, chunkBytes, destPe);
  if (threadIdx.x == 0) {
    ShmemQuietThread(destPe);
  }
}

// Check every element across all N chunks.
// Chunk for rank r should contain (r+1) ^ element_index.
__global__ void VerifyKernel(const uint32_t* buf, size_t elementsPerChunk, int npes,
                             uint32_t* errorCount) {
  size_t totalElements = static_cast<size_t>(npes) * elementsPerChunk;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < totalElements;
       i += (size_t)gridDim.x * blockDim.x) {
    int rank = static_cast<int>(i / elementsPerChunk);
    size_t localIdx = i % elementsPerChunk;
    uint32_t expected = static_cast<uint32_t>(rank + 1) ^ static_cast<uint32_t>(localIdx);
    if (buf[i] != expected) {
      atomicAdd(errorCount, 1);
    }
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  MPI_Init(NULL, NULL);

  MPI_Comm localComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);

  int localRank;
  MPI_Comm_rank(localComm, &localRank);

  int deviceCount;
  HIP_RUNTIME_CHECK(hipGetDeviceCount(&deviceCount));
  int deviceId = localRank % deviceCount;
  HIP_RUNTIME_CHECK(hipSetDevice(deviceId));

  XPUT("Local rank %d setting GPU device %d (total %d devices)", localRank, deviceId, deviceCount);

  int status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  size_t chunkBytes = DEFAULT_CHUNK_BYTES;
  if (argc > 1) chunkBytes = std::atol(argv[1]);
  assert(chunkBytes >= 4 && (chunkBytes % 4) == 0);

  size_t totalBytes = static_cast<size_t>(npes) * chunkBytes;
  size_t elementsPerChunk = chunkBytes / sizeof(uint32_t);

  if (myPe == 0) {
    printf("allgather_test: %d PEs, %zu bytes/PE (%zu KB), %zu bytes total\n", npes, chunkBytes,
           chunkBytes / 1024, totalBytes);
  }

  // Allocate symmetric buffer: N chunks, one per PE
  void* buf = ShmemMalloc(totalBytes);
  HIP_RUNTIME_CHECK(hipMemset(buf, 0, totalBytes));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // Each PE fills its own chunk with a rank-specific pattern
  uint32_t* myChunk = reinterpret_cast<uint32_t*>(buf) + myPe * elementsPerChunk;
  uint32_t seed = static_cast<uint32_t>(myPe + 1);
  constexpr int kThreads = 256;
  int fillBlocks =
      static_cast<int>(std::min<size_t>(1024, (elementsPerChunk + kThreads - 1) / kThreads));
  FillPatternKernel<<<fillBlocks, kThreads>>>(myChunk, elementsPerChunk, seed);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // Make sure every PE has its chunk ready before we start pushing
  MPI_Barrier(MPI_COMM_WORLD);

  SymmMemObjPtr bufObj = ShmemQueryMemObjPtr(buf);
  assert(bufObj.IsValid());

  // --- Allgather ---
  hipEvent_t tStart, tStop;
  HIP_RUNTIME_CHECK(hipEventCreate(&tStart));
  HIP_RUNTIME_CHECK(hipEventCreate(&tStop));
  HIP_RUNTIME_CHECK(hipEventRecord(tStart));

  AllGatherKernel<<<npes, kThreads>>>(myPe, npes, bufObj, chunkBytes);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  HIP_RUNTIME_CHECK(hipEventRecord(tStop));
  HIP_RUNTIME_CHECK(hipEventSynchronize(tStop));
  float elapsedMs = 0;
  HIP_RUNTIME_CHECK(hipEventElapsedTime(&elapsedMs, tStart, tStop));

  // Wait for all PEs to finish their puts before reading
  MPI_Barrier(MPI_COMM_WORLD);

  // --- Verify ---
  uint32_t* dErrors;
  HIP_RUNTIME_CHECK(hipMalloc(&dErrors, sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemset(dErrors, 0, sizeof(uint32_t)));

  size_t totalElements = static_cast<size_t>(npes) * elementsPerChunk;
  int vBlocks =
      static_cast<int>(std::min<size_t>(1024, (totalElements + kThreads - 1) / kThreads));
  VerifyKernel<<<vBlocks, kThreads>>>(reinterpret_cast<const uint32_t*>(buf), elementsPerChunk,
                                      npes, dErrors);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  uint32_t hErrors = 0;
  HIP_RUNTIME_CHECK(hipMemcpy(&hErrors, dErrors, sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_RUNTIME_CHECK(hipFree(dErrors));

  double bw = (totalBytes / 1e9) / (elapsedMs / 1e3);
  printf("Rank %d: %s (%u errors), %.2f ms, %.3f GB/s\n", myPe,
         hErrors == 0 ? "PASS" : "FAIL", hErrors, elapsedMs, bw);

  HIP_RUNTIME_CHECK(hipEventDestroy(tStart));
  HIP_RUNTIME_CHECK(hipEventDestroy(tStop));
  ShmemFree(buf);
  ShmemFinalize();
  return 0;
}
