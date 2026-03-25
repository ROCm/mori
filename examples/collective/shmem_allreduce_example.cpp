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
// Intra-node 8-GPU AllReduce using ShmemPutMemNbiThread + ShmemQuietThread
//
// Two-shot algorithm: reduce-scatter + allgather
//   Phase 1: Scatter — each PE sends shard[destPe] of input to destPe
//   Phase 2: Local reduce — sum all PE contributions for my shard
//   Phase 3: AllGather — broadcast reduced shard to all PEs
//   Phase 4: Wait for allgather completion
//

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/core/transport/rdma/device_primitives.hpp"
#include "mori/core/transport/sdma/device_primitives.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::application;
using namespace mori::shmem;

#define CHECK_HIP(call)                                                                        \
  do {                                                                                         \
    hipError_t err = (call);                                                                   \
    if (err != hipSuccess) {                                                                   \
      fprintf(stderr, "HIP Error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
      throw std::runtime_error("HIP call failed");                                             \
    }                                                                                          \
  } while (0)

__global__ void ShmemAllReduceKernel(int myPe, int npes, const uint32_t* __restrict__ input,
                                     const SymmMemObjPtr gatherObj, const SymmMemObjPtr flagsObj,
                                     size_t elementCount) {
  if (elementCount == 0 || npes <= 0) return;

  const size_t elementCountPerRank = elementCount / npes;
  const size_t bytesPerElement = sizeof(uint32_t);
  const size_t chunkBytes = elementCountPerRank * bytesPerElement;

  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsObj->localPtr);

  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  int warpId = static_cast<int>(threadIdx.x) / warpSize;
  int laneId = static_cast<int>(threadIdx.x) % warpSize;

  // =========================================================================
  // Phase 1: Scatter — block 0, each warp's lane 0 sends one shard
  // Uses core::SdmaPutThread directly (ShmemPutMemNbiThread requires RDMA
  // registration which may not be available on all machines).
  // =========================================================================
  if (blockIdx.x == 0 && warpId < npes && laneId == 0) {
    int destPe = warpId;
    size_t srcOffset = static_cast<size_t>(destPe) * chunkBytes;
    size_t dstOffset = static_cast<size_t>(myPe) * chunkBytes;

    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(gatherObj->localPtr) + srcOffset;
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(gatherObj->peerPtrs[destPe]) + dstOffset;

    anvil::SdmaQueueDeviceHandle** dh =
        gatherObj->deviceHandles_d + destPe * gatherObj->sdmaNumQueue;
    HSAuint64* remoteSignal =
        gatherObj->peerSignalPtrs[destPe] + static_cast<size_t>(myPe) * gatherObj->sdmaNumQueue;
    HSAuint64* localExpected =
        gatherObj->expectSignalsPtr + static_cast<size_t>(destPe) * gatherObj->sdmaNumQueue;

    mori::core::SdmaPutThread(srcPtr, dstPtr, chunkBytes, dh, remoteSignal, localExpected,
                              gatherObj->sdmaNumQueue, 0);
  }

  if (blockIdx.x == 0 && warpId < npes && laneId == 0) {
    int destPe = warpId;
    mori::shmem::ShmemQuietThread(destPe, gatherObj);

    uint64_t flag_val = 1;
    mori::shmem::ShmemAtomicSizeNonFetchThread(
        flagsObj, static_cast<size_t>(myPe) * sizeof(uint64_t), &flag_val, 8,
        mori::core::atomicType::AMO_ADD, destPe);
  }

  if (blockIdx.x == 0) {
    __syncthreads();
    for (int sender = 0; sender < npes; ++sender) {
      if (sender == myPe) continue;
      if (threadIdx.x == 0) {
        while (mori::core::AtomicLoadRelaxed(flags + sender) == 0) {
        }
      }
      __syncthreads();
    }
    if (threadIdx.x < static_cast<unsigned>(npes)) {
      flags[threadIdx.x] = 0;
    }
  }
  __syncthreads();

  // =========================================================================
  // Phase 2: Local reduce — all blocks participate
  // =========================================================================
  uint32_t* __restrict__ gathered = reinterpret_cast<uint32_t*>(gatherObj->localPtr);
  uint32_t* __restrict__ myDst = gathered + static_cast<size_t>(myPe) * elementCountPerRank;

  const uint32_t* inputSlot = input + static_cast<size_t>(myPe) * elementCountPerRank;
  for (size_t k = tid; k < elementCountPerRank; k += stride) {
    myDst[k] = inputSlot[k];
  }

  for (size_t k = tid; k < elementCountPerRank; k += stride) {
    uint32_t sum = gathered[k];
    for (int pe = 1; pe < npes; pe++) {
      sum += gathered[static_cast<size_t>(pe) * elementCountPerRank + k];
    }
    myDst[k] = sum;
  }

#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("buffer_wbl2" ::: "memory");
  }
#endif
  __syncthreads();

  // =========================================================================
  // Phase 3: AllGather — block 0 sends reduced shard to all PEs
  // =========================================================================
  if (blockIdx.x == 0 && warpId < npes && laneId == 0) {
    int destPe = warpId;
    size_t offset = static_cast<size_t>(myPe) * chunkBytes;

    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(gatherObj->localPtr) + offset;
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(gatherObj->peerPtrs[destPe]) + offset;

    anvil::SdmaQueueDeviceHandle** dh =
        gatherObj->deviceHandles_d + destPe * gatherObj->sdmaNumQueue;
    HSAuint64* remoteSignal =
        gatherObj->peerSignalPtrs[destPe] + static_cast<size_t>(myPe) * gatherObj->sdmaNumQueue;
    HSAuint64* localExpected =
        gatherObj->expectSignalsPtr + static_cast<size_t>(destPe) * gatherObj->sdmaNumQueue;

    mori::core::SdmaPutThread(srcPtr, dstPtr, chunkBytes, dh, remoteSignal, localExpected,
                              gatherObj->sdmaNumQueue, 0);
  }

  if (blockIdx.x == 0 && warpId < npes && laneId == 0) {
    int destPe = warpId;
    mori::shmem::ShmemQuietThread(destPe, gatherObj);

    uint64_t flag_val = 1;
    mori::shmem::ShmemAtomicSizeNonFetchThread(
        flagsObj, static_cast<size_t>(myPe) * sizeof(uint64_t), &flag_val, 8,
        mori::core::atomicType::AMO_ADD, destPe);
  }

  // =========================================================================
  // Phase 4: Wait for allgather completion
  // =========================================================================
  if (blockIdx.x == 0) {
    __syncthreads();
    for (int sender = 0; sender < npes; ++sender) {
      if (sender == myPe) continue;
      if (threadIdx.x == 0) {
        while (mori::core::AtomicLoadRelaxed(flags + sender) == 0) {
        }
      }
      __syncthreads();
    }
    if (threadIdx.x < static_cast<unsigned>(npes)) {
      flags[threadIdx.x] = 0;
    }
  }
}

void testShmemAllReduce() {
  MPI_Init(NULL, NULL);
  int status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  printf("PE %d of %d started\n", myPe, npes);

  const int elemsPerPe = 8 * 1024 * 1024;
  const size_t bytesPerPe = elemsPerPe * sizeof(uint32_t);
  const size_t totalBytes = bytesPerPe * npes;

  uint32_t* inputBuf = nullptr;
  CHECK_HIP(hipMalloc(&inputBuf, bytesPerPe));

  uint32_t fillValue = static_cast<uint32_t>((myPe + 1) * 1000);
  std::vector<uint32_t> hostData(elemsPerPe, fillValue);
  CHECK_HIP(hipMemcpy(inputBuf, hostData.data(), bytesPerPe, hipMemcpyHostToDevice));

  void* gatherBuf = ShmemMalloc(totalBytes);
  assert(gatherBuf);
  CHECK_HIP(hipMemset(gatherBuf, 0, totalBytes));
  CHECK_HIP(hipMemcpy(gatherBuf, inputBuf, bytesPerPe, hipMemcpyDeviceToDevice));
  CHECK_HIP(hipDeviceSynchronize());

  SymmMemObjPtr gatherObj = ShmemQueryMemObjPtr(gatherBuf);
  assert(gatherObj.IsValid());

  size_t flagsSize = npes * sizeof(uint64_t);
  void* flagsBuf = ShmemMalloc(flagsSize);
  assert(flagsBuf);
  CHECK_HIP(hipMemset(flagsBuf, 0, flagsSize));
  SymmMemObjPtr flagsObj = ShmemQueryMemObjPtr(flagsBuf);
  assert(flagsObj.IsValid());

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));
  MPI_Barrier(MPI_COMM_WORLD);

  uint32_t expected = 0;
  for (int pe = 0; pe < npes; pe++) expected += (pe + 1) * 1000;

  if (myPe == 0) {
    printf("\n=== Shmem AllReduce Test ===\n");
    printf("  Elements per PE : %d\n", elemsPerPe);
    printf("  Data size       : %.2f MB per PE\n", bytesPerPe / (1024.0 * 1024.0));
    printf("  Fill value      : (PE_id + 1) * 1000\n");
    printf("  Expected result : %u\n\n", expected);
  }

  const int warmup = 10;
  const int iterations = 10;
  const int blocks = 80;
  const int threads = 512;
  std::vector<double> exec_times;

  for (int i = 0; i < warmup + iterations; i++) {
    CHECK_HIP(hipMemcpy(gatherBuf, inputBuf, bytesPerPe, hipMemcpyDeviceToDevice));
    CHECK_HIP(hipMemset(flagsBuf, 0, flagsSize));
    CHECK_HIP(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();
    ShmemAllReduceKernel<<<blocks, threads, 0, stream>>>(myPe, npes, inputBuf, gatherObj, flagsObj,
                                                         elemsPerPe);
    CHECK_HIP(hipStreamSynchronize(stream));
    double t1 = MPI_Wtime();

    if (i >= warmup) {
      exec_times.push_back(t1 - t0);
      if (myPe == 0 && exec_times.size() == 1)
        printf("PE %d: First measurement: %.3f ms\n", myPe, (t1 - t0) * 1000);
    } else if (myPe == 0) {
      printf("  Warmup %d: %.3f ms\n", i + 1, (t1 - t0) * 1000);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  std::vector<uint32_t> result(elemsPerPe);
  CHECK_HIP(hipMemcpy(result.data(), gatherBuf, bytesPerPe, hipMemcpyDeviceToHost));

  bool ok = true;
  for (size_t i = 0; i < static_cast<size_t>(elemsPerPe); i++) {
    if (result[i] != expected) {
      printf("PE %d: FAILED at [%zu]: expected %u, got %u\n", myPe, i, expected, result[i]);
      ok = false;
      break;
    }
  }
  if (ok) printf("PE %d: Verification PASSED (all %d elements = %u)\n", myPe, elemsPerPe, expected);

  double avg = 0, mn = exec_times[0], mx = exec_times[0];
  for (double t : exec_times) {
    avg += t;
    mn = std::min(mn, t);
    mx = std::max(mx, t);
  }
  avg /= exec_times.size();

  double g_max = 0, g_min = 0, g_sum = 0;
  MPI_Reduce(&avg, &g_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&avg, &g_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&avg, &g_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  int lok = ok ? 1 : 0, gok = 0;
  MPI_Reduce(&lok, &gok, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (myPe == 0) {
    double g_avg = g_sum / npes;
    double algo_bw = bytesPerPe / g_avg / (1024.0 * 1024.0 * 1024.0);
    double bus_bw = algo_bw * 2.0 * (npes - 1) / npes;

    printf("\n=== Performance ===\n");
    printf("  Min: %.3f ms  Max: %.3f ms  Avg: %.3f ms\n", g_min * 1000, g_max * 1000,
           g_avg * 1000);
    printf("  Algo BW: %.2f GB/s  Bus BW: %.2f GB/s\n", algo_bw, bus_bw);
    printf("  PEs passed: %d/%d\n", gok, npes);
    printf("  %s\n\n", gok == npes ? "=== PASSED ===" : "=== FAILED ===");
  }

  CHECK_HIP(hipStreamDestroy(stream));
  CHECK_HIP(hipFree(inputBuf));
  ShmemFree(gatherBuf);
  ShmemFree(flagsBuf);
  MPI_Barrier(MPI_COMM_WORLD);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  testShmemAllReduce();
  return 0;
}
