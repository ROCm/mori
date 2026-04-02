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

// Ring allgather example with configurable transfer size.
// Each PE fills its chunk with a rank-specific pattern, then all PEs
// exchange chunks in a ring so every PE ends up with the full buffer.
//
// Usage: mpirun --allow-run-as-root -np <npes> ./put_allgather_large [bytes_per_pe]
//   bytes_per_pe defaults to 1MB. Must be a multiple of 4.

#include <mpi.h>

#include <cassert>
#include <cstdlib>
#include <cstring>

#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"

#define XPUT(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;

constexpr size_t DEFAULT_BYTES_PER_PE = 1 * 1024 * 1024;  // 1 MB

__global__ void FillPatternKernel(uint32_t* buf, size_t numElements, uint32_t rankSeed) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < numElements; i += gridDim.x * blockDim.x) {
    buf[i] = rankSeed ^ static_cast<uint32_t>(i);
  }
}

__global__ void RingAllGatherKernel(int myPe, int npes, const SymmMemObjPtr memObj,
                                    size_t peChunkBytes) {
  int nextPeer = (myPe + 1) % npes;

  auto peerPtr = ShmemPtrP2p(reinterpret_cast<uint64_t>(memObj->localPtr), myPe, nextPeer);

  printf("myPe: %d, npes: %d, peChunkSize: %d\n", myPe, npes, peChunkBytes);
  printf("localPtr: %p peerPtr %p\n", memObj->localPtr, peerPtr);

  for (int i = 0; i < npes - 1; i++) {
    int sendDataRank = ((myPe - i) + npes) % npes;
    size_t sendOffset = sendDataRank * peChunkBytes;

    ShmemPutMemNbiThread(memObj, sendOffset, memObj, sendOffset, peChunkBytes, nextPeer);
    ShmemQuietThread(nextPeer, memObj);

    int recvDataRank = ((sendDataRank - 1) + npes) % npes;
    size_t recvOffset = recvDataRank * peChunkBytes;
    uint32_t* recvAddr =
        reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(memObj->localPtr) + recvOffset);

    uint32_t expectedFirst = static_cast<uint32_t>(recvDataRank + 1) ^ 0u;
    while (atomicAdd(recvAddr, 0) != expectedFirst) {
    }
  }
}

__global__ void VerifyKernel(const uint32_t* buf, size_t elementsPerPe, int npes,
                             uint32_t* errorCount) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t totalElements = static_cast<size_t>(npes) * elementsPerPe;
  for (size_t i = idx; i < totalElements; i += gridDim.x * blockDim.x) {
    int rank = static_cast<int>(i / elementsPerPe);
    size_t localIdx = i % elementsPerPe;
    uint32_t expected = static_cast<uint32_t>(rank + 1) ^ static_cast<uint32_t>(localIdx);
    if (buf[i] != expected) {
      atomicAdd(errorCount, 1);
    }
  }
}

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

  size_t bytesPerPe = DEFAULT_BYTES_PER_PE;
  if (argc > 1) {
    bytesPerPe = std::atol(argv[1]);
  }
  assert(bytesPerPe >= 4 && (bytesPerPe % 4) == 0);

  size_t totalBytes = static_cast<size_t>(npes) * bytesPerPe;
  size_t elementsPerPe = bytesPerPe / sizeof(uint32_t);

  if (myPe == 0) {
    XPUT("Ring allgather: %d PEs, %zu bytes/PE (%zu KB), %zu bytes total", npes, bytesPerPe,
           bytesPerPe / 1024, totalBytes);
  }

  void* buff = ShmemMalloc(totalBytes);
  HIP_RUNTIME_CHECK(hipMemset(buff, 0, totalBytes));
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  // Fill this PE's chunk with a rank-specific pattern: (rank+1) ^ element_index
  uint32_t* myChunk = reinterpret_cast<uint32_t*>(buff) + myPe * elementsPerPe;
  uint32_t rankSeed = static_cast<uint32_t>(myPe + 1);
  int threads = 256;
  int blocks = std::min(static_cast<size_t>(1024), (elementsPerPe + threads - 1) / threads);
  FillPatternKernel<<<blocks, threads>>>(myChunk, elementsPerPe, rankSeed);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  SymmMemObjPtr buffObj = ShmemQueryMemObjPtr(buff);
  assert(buffObj.IsValid());

  // Run ring allgather
  hipEvent_t start, stop;
  HIP_RUNTIME_CHECK(hipEventCreate(&start));
  HIP_RUNTIME_CHECK(hipEventCreate(&stop));
  HIP_RUNTIME_CHECK(hipEventRecord(start));

  RingAllGatherKernel<<<1, 1>>>(myPe, npes, buffObj, bytesPerPe);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  HIP_RUNTIME_CHECK(hipEventRecord(stop));
  HIP_RUNTIME_CHECK(hipEventSynchronize(stop));
  float elapsedMs = 0;
  HIP_RUNTIME_CHECK(hipEventElapsedTime(&elapsedMs, start, stop));
  MPI_Barrier(MPI_COMM_WORLD);

  // Verify all chunks on GPU
  uint32_t* dErrorCount;
  HIP_RUNTIME_CHECK(hipMalloc(&dErrorCount, sizeof(uint32_t)));
  HIP_RUNTIME_CHECK(hipMemset(dErrorCount, 0, sizeof(uint32_t)));

  size_t totalElements = static_cast<size_t>(npes) * elementsPerPe;
  int vBlocks = std::min(static_cast<size_t>(1024), (totalElements + threads - 1) / threads);
  VerifyKernel<<<vBlocks, threads>>>(reinterpret_cast<const uint32_t*>(buff), elementsPerPe, npes,
                                     dErrorCount);
  HIP_RUNTIME_CHECK(hipDeviceSynchronize());

  uint32_t hErrorCount = 0;
  HIP_RUNTIME_CHECK(hipMemcpy(&hErrorCount, dErrorCount, sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_RUNTIME_CHECK(hipFree(dErrorCount));

  double bw = (totalBytes / 1e9) / (elapsedMs / 1e3);
  printf("Rank %d: %s (%u errors), %.2f ms, %.3f GB/s\n", myPe,
         hErrorCount == 0 ? "PASS" : "FAIL", hErrorCount, elapsedMs, bw);

  HIP_RUNTIME_CHECK(hipEventDestroy(start));
  HIP_RUNTIME_CHECK(hipEventDestroy(stop));
  ShmemFree(buff);
  ShmemFinalize();
  return 0;
}
