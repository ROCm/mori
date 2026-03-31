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
// SDMA self-copy (D2D on same GPU) performance test.
// Single process, single GPU. Measures SDMA copy bandwidth vs hipMemcpy D2D.

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

#include "mori/application/utils/check.hpp"
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

// Kernel: SDMA self-copy using multi-queue, then wait for completion
__global__ void SdmaSelfCopyKernel(const SymmMemObjPtr srcObj, void* dst, size_t totalBytes,
                                   int myPe) {
  const size_t threadLinearId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  uint32_t numQueues = srcObj->sdmaNumQueue;

  __shared__ HSAuint64 s_baseline[8];
  if (threadLinearId < numQueues) {
    s_baseline[threadLinearId] = mori::core::AtomicLoadRelaxed(
        srcObj->signalPtrs + static_cast<size_t>(myPe) * numQueues + threadLinearId);
  }
  __syncthreads();

  // Phase 1: multi-queue SDMA copy (self-to-self)
  if (threadLinearId < numQueues) {
    int qId = threadLinearId;
    const size_t chunkBase = totalBytes / numQueues;
    size_t offset = qId * chunkBase;
    size_t copyBytes =
        (qId == (int)(numQueues - 1)) ? (totalBytes - (numQueues - 1) * chunkBase) : chunkBase;

    anvil::SdmaQueueDeviceHandle** dh = srcObj->deviceHandles_d + myPe * numQueues;
    HSAuint64* remoteSignal = srcObj->peerSignalPtrs[myPe]
                              + static_cast<size_t>(myPe) * numQueues;

    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(srcObj->localPtr) + offset;
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dst) + offset;

    SdmaPutThread(srcPtr, dstPtr, copyBytes, dh, remoteSignal, numQueues, qId);
  }
  __syncthreads();

  // Phase 2: wait for all queues
  if (threadLinearId < numQueues) {
    int qId = threadLinearId;
    HSAuint64* signal = srcObj->signalPtrs + static_cast<size_t>(myPe) * numQueues;
    anvil::waitForSignal(signal + qId, s_baseline[qId] + 1);
  }
  __syncthreads();
}

void testSdmaSelfCopy() {
  MPI_Init(NULL, NULL);
  int status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  std::vector<size_t> testSizes = {1 * 1024 * 1024,  4 * 1024 * 1024,  16 * 1024 * 1024,
                                   32 * 1024 * 1024, 64 * 1024 * 1024, 128 * 1024 * 1024,
                                   256 * 1024 * 1024};

  const int warmup = 10;
  const int iterations = 20;

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  // ========================================================================
  // Test A: Same-GPU copy comparison (SDMA self vs hipMemcpy D2D)
  // ========================================================================
  if (myPe == 0) {
    printf("=== Test A: Same-GPU Copy (SDMA self vs hipMemcpy D2D) ===\n\n");
    printf("%12s %12s %12s %12s %12s\n", "Size(MB)", "SDMA(GB/s)", "hipMem(GB/s)", "SDMA(ms)",
           "hipMem(ms)");
    printf("%s\n", std::string(60, '-').c_str());
  }

  for (size_t totalBytes : testSizes) {
    void* srcBuf = ShmemMalloc(totalBytes);
    assert(srcBuf);
    CHECK_HIP(hipMemset(srcBuf, 0xAB, totalBytes));
    SymmMemObjPtr srcObj = ShmemQueryMemObjPtr(srcBuf);
    assert(srcObj.IsValid());

    void* dstBuf = nullptr;
    CHECK_HIP(hipMalloc(&dstBuf, totalBytes));
    CHECK_HIP(hipMemset(dstBuf, 0, totalBytes));
    CHECK_HIP(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // SDMA self-copy
    std::vector<double> sdma_times;
    for (int i = 0; i < warmup + iterations; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double t0 = MPI_Wtime();
      SdmaSelfCopyKernel<<<1, 64, 0, stream>>>(srcObj, dstBuf, totalBytes, myPe);
      CHECK_HIP(hipStreamSynchronize(stream));
      double t1 = MPI_Wtime();
      if (i >= warmup) sdma_times.push_back(t1 - t0);
    }

    // hipMemcpy D2D (same GPU)
    std::vector<double> hip_times;
    for (int i = 0; i < warmup + iterations; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double t0 = MPI_Wtime();
      CHECK_HIP(hipMemcpyAsync(dstBuf, srcBuf, totalBytes, hipMemcpyDeviceToDevice, stream));
      CHECK_HIP(hipStreamSynchronize(stream));
      double t1 = MPI_Wtime();
      if (i >= warmup) hip_times.push_back(t1 - t0);
    }

    double sdma_avg = 0, hip_avg = 0;
    for (double t : sdma_times) sdma_avg += t;
    sdma_avg /= sdma_times.size();
    for (double t : hip_times) hip_avg += t;
    hip_avg /= hip_times.size();

    if (myPe == 0) {
      printf("%10.1f %12.2f %12.2f %12.3f %12.3f\n", totalBytes / (1024.0 * 1024.0),
             totalBytes / sdma_avg / (1024.0 * 1024.0 * 1024.0),
             totalBytes / hip_avg / (1024.0 * 1024.0 * 1024.0), sdma_avg * 1000, hip_avg * 1000);
    }

    CHECK_HIP(hipFree(dstBuf));
    ShmemFree(srcBuf);
  }

  // ========================================================================
  // Test B: Cross-GPU copy comparison (SDMA PUT vs hipMemcpy P2P)
  // Requires npes >= 2
  // ========================================================================
  if (npes >= 2) {
    int remotePe = (myPe + 1) % npes;

    if (myPe == 0) {
      printf("\n=== Test B: Cross-GPU Copy (SDMA PUT vs hipMemcpy P2P) ===\n");
      printf("PE %d -> PE %d\n\n", myPe, remotePe);
      printf("%12s %12s %12s %12s %12s\n", "Size(MB)", "SDMA(GB/s)", "hipMem(GB/s)", "SDMA(ms)",
             "hipMem(ms)");
      printf("%s\n", std::string(60, '-').c_str());
    }

    for (size_t totalBytes : testSizes) {
      // Allocate symmetric memory (has peer mappings for both SDMA and hipMemcpy P2P)
      void* srcBuf = ShmemMalloc(totalBytes);
      assert(srcBuf);
      CHECK_HIP(hipMemset(srcBuf, myPe + 1, totalBytes));
      SymmMemObjPtr srcObj = ShmemQueryMemObjPtr(srcBuf);
      assert(srcObj.IsValid());

      CHECK_HIP(hipDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);

      // SDMA PUT to remote PE (kernel-based, multi-queue)
      std::vector<double> sdma_times;
      for (int i = 0; i < warmup + iterations; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        SdmaSelfCopyKernel<<<1, 64, 0, stream>>>(
            srcObj, reinterpret_cast<void*>(srcObj->peerPtrs[remotePe]), totalBytes, remotePe);
        CHECK_HIP(hipStreamSynchronize(stream));
        double t1 = MPI_Wtime();
        if (i >= warmup) sdma_times.push_back(t1 - t0);
      }

      // hipMemcpy P2P to remote PE
      std::vector<double> hip_times;
      for (int i = 0; i < warmup + iterations; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        CHECK_HIP(hipMemcpyAsync(reinterpret_cast<void*>(srcObj->peerPtrs[remotePe]), srcBuf,
                                 totalBytes, hipMemcpyDeviceToDevice, stream));
        CHECK_HIP(hipStreamSynchronize(stream));
        double t1 = MPI_Wtime();
        if (i >= warmup) hip_times.push_back(t1 - t0);
      }

      double sdma_avg = 0, hip_avg = 0;
      for (double t : sdma_times) sdma_avg += t;
      sdma_avg /= sdma_times.size();
      for (double t : hip_times) hip_avg += t;
      hip_avg /= hip_times.size();

      if (myPe == 0) {
        printf("%10.1f %12.2f %12.2f %12.3f %12.3f\n", totalBytes / (1024.0 * 1024.0),
               totalBytes / sdma_avg / (1024.0 * 1024.0 * 1024.0),
               totalBytes / hip_avg / (1024.0 * 1024.0 * 1024.0), sdma_avg * 1000, hip_avg * 1000);
      }

      ShmemFree(srcBuf);
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  CHECK_HIP(hipStreamDestroy(stream));
  MPI_Barrier(MPI_COMM_WORLD);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  testSdmaSelfCopy();
  return 0;
}
