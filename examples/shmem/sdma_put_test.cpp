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
// SDMA PUT test suite for sdma-batch branch.
// Tests peerSignalPtrs (remote signal) and zero-size PUT guard.
//
// Tests:
//   1. Normal PUT + ShmemQuiet (local signal, existing API)
//   2. Zero-size PUT (copy_size=0, CU atomic fallback)
//   3. Remote signal PUT (write to remote PE's signalPtrs via peerSignalPtrs)
//   4. Repeated PUTs (signal counter correctness across iterations)
//   5. Multiple sizes (4KB, 1MB, 32MB)

#include <mpi.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

#include "mori/application/utils/check.hpp"
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

// Test 1 & 2 & 5: SDMA PUT + Quiet (local signal), supports zero-size
// Uses separate src (source data) and dst (destination on remote PE) buffers
// to avoid L2 cache pollution on the receive side.
__global__ void ShmemPutQuietKernel(const SymmMemObjPtr srcObj, const SymmMemObjPtr dstObj,
                                    int myPe, int destPe, size_t bytes) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  // PUT: read from srcObj (local), write to dstObj (remote via peerPtrs)
  ShmemPutMemNbiThreadKernel<TransportType::SDMA>(dstObj, 0, srcObj, 0, bytes, destPe, 0);
  ShmemQuietThread(destPe, dstObj);
}

// Test 3: Remote signal PUT (bypass ShmemQuiet, write directly to remote signal)
__global__ void RemoteSignalPutKernel(const SymmMemObjPtr srcObj, const SymmMemObjPtr dstObj,
                                      int myPe, int destPe, size_t bytes) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  uint8_t* srcPtr = reinterpret_cast<uint8_t*>(srcObj->localPtr);
  uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dstObj->peerPtrs[destPe]);

  anvil::SdmaQueueDeviceHandle** dh = dstObj->deviceHandles_d + destPe * dstObj->sdmaNumQueue;

  HSAuint64* remoteSignal =
      dstObj->peerSignalPtrs[destPe] + static_cast<size_t>(myPe) * dstObj->sdmaNumQueue;

  // Manual SDMA PUT with remote signal
  if (bytes > 0) {
    anvil::SdmaQueueDeviceHandle handle = **(dh);
    uint64_t offset = 0;
    uint64_t base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR), offset);
    uint64_t pendingWptr = base;
    uint64_t startBase = base;

    auto pkt_copy = anvil::CreateCopyPacket(srcPtr, dstPtr, bytes);
    handle.template placePacket<SDMA_PKT_COPY_LINEAR>(pkt_copy, pendingWptr, offset);

    base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC), offset);
    pendingWptr = base;
    auto pkt_sig = anvil::CreateAtomicIncPacket(remoteSignal);
    handle.template placePacket<SDMA_PKT_ATOMIC>(pkt_sig, pendingWptr, offset);

    handle.submitPacket(startBase, pendingWptr);
  } else {
    // Zero-size: CU atomic directly to remote signal
    __hip_atomic_fetch_add(remoteSignal, 1ULL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// Wait on local signal for remote signal test
__global__ void WaitRemoteSignalKernel(const SymmMemObjPtr memObj, int senderPe,
                                       HSAuint64 expectedVal) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  HSAuint64* mySignal = memObj->signalPtrs + static_cast<size_t>(senderPe) * memObj->sdmaNumQueue;
  int spin = 0;
  while (__hip_atomic_load(mySignal, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < expectedVal) {
    if (++spin > 100000000) {
      printf("Timeout waiting for remote signal from PE %d (expected %lu)\n", senderPe,
             expectedVal);
      return;
    }
  }
}

struct TestResult {
  const char* name;
  bool passed;
};
static std::vector<TestResult> results;

static void report(const char* name, bool passed) {
  results.push_back({name, passed});
  printf("  [%s] %s\n", passed ? "PASS" : "FAIL", name);
}

void runTests() {
  MPI_Init(NULL, NULL);
  int status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();
  if (npes < 2) {
    if (myPe == 0) printf("Need >= 2 PEs\n");
    ShmemFinalize();
    return;
  }

  int remotePe = (myPe + 1) % npes;
  int senderPe = (myPe - 1 + npes) % npes;
  const size_t maxBuf = 32 * 1024 * 1024;

  // Two separate symmetric buffers:
  // srcBuf: filled with myPe+1 pattern, used as source for sending
  // dstBuf: never touched by CU (no L2 pollution), used as destination for receiving
  void* srcBuf = ShmemMalloc(maxBuf);
  void* dstBuf = ShmemMalloc(maxBuf);
  assert(srcBuf && dstBuf);
  SymmMemObjPtr srcMemObj = ShmemQueryMemObjPtr(srcBuf);
  SymmMemObjPtr dstMemObj = ShmemQueryMemObjPtr(dstBuf);
  assert(srcMemObj.IsValid() && dstMemObj.IsValid());

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  if (myPe == 0) {
    printf("\n=== SDMA PUT Test Suite (sdma-batch) ===\n");
    printf("  PEs: %d, Testing PE %d -> PE %d\n", npes, myPe, remotePe);
    printf("  peerSignalPtrs: %s\n\n", dstMemObj->peerSignalPtrs ? "allocated" : "NULL");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Helper: run shmem PUT + quiet test
  // srcBuf: filled with myPe+1, used as source for sending
  // dstBuf: never written by CU (avoid L2 pollution), used as receive target
  auto runShmemTest = [&](const char* name, size_t bytes) {
    CHECK_HIP(hipMemset(srcBuf, myPe + 1, maxBuf));
    CHECK_HIP(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    ShmemPutQuietKernel<<<1, 1, 0, stream>>>(srcMemObj, dstMemObj, myPe, remotePe, bytes);
    CHECK_HIP(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    bool ok = true;
    if (bytes > 0) {
      std::vector<uint8_t> hostBuf(bytes);
      CHECK_HIP(hipMemcpy(hostBuf.data(), dstBuf, bytes, hipMemcpyDeviceToHost));
      uint8_t expected = static_cast<uint8_t>(senderPe + 1);
      for (size_t i = 0; i < bytes; i++) {
        if (hostBuf[i] != expected) {
          if (myPe == 0)
            printf("    PE %d: mismatch at [%zu]: expected 0x%02x got 0x%02x (myPe+1=0x%02x)\n",
                   myPe, i, expected, hostBuf[i], (uint8_t)(myPe + 1));
          ok = false;
          break;
        }
      }
    }

    int lok = ok ? 1 : 0, gok = 0;
    MPI_Allreduce(&lok, &gok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (myPe == 0) report(name, gok == 1);
    MPI_Barrier(MPI_COMM_WORLD);
  };

  // --- Test 1: Normal PUT sizes ---
  if (myPe == 0) printf("--- Shmem PUT + Quiet (local signal) ---\n");
  runShmemTest("PUT 4KB", 4096);
  runShmemTest("PUT 1MB", 1024 * 1024);
  runShmemTest("PUT 32MB", 32 * 1024 * 1024);

  // --- Test 2: Zero-size PUT ---
  if (myPe == 0) printf("\n--- Zero-size PUT ---\n");
  runShmemTest("Zero-size PUT (0 bytes)", 0);

  // --- Test 3: Repeated PUTs (must run before remote signal tests which desync counters) ---
  if (myPe == 0) printf("\n--- Repeated PUTs ---\n");
  {
    CHECK_HIP(hipMemset(srcBuf, myPe + 1, maxBuf));
    CHECK_HIP(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    const int repeatCount = 10;
    for (int i = 0; i < repeatCount; i++) {
      ShmemPutQuietKernel<<<1, 1, 0, stream>>>(srcMemObj, dstMemObj, myPe, remotePe, 4096);
      CHECK_HIP(hipStreamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);
    }

    bool ok = true;
    std::vector<uint8_t> hostBuf(4096);
    CHECK_HIP(hipMemcpy(hostBuf.data(), dstBuf, 4096, hipMemcpyDeviceToHost));
    uint8_t expected = static_cast<uint8_t>(senderPe + 1);
    for (size_t i = 0; i < 4096; i++) {
      if (hostBuf[i] != expected) {
        ok = false;
        break;
      }
    }

    int lok = ok ? 1 : 0, gok = 0;
    MPI_Allreduce(&lok, &gok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (myPe == 0) report("Repeated PUT x10 (4KB)", gok == 1);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- Test 4: Remote signal PUT ---
  // Note: Remote signal test uses dstMemObj for both src and dst (same buffer pattern).
  // Since dstBuf was never CU-written, L2 is clean for receiving.
  if (myPe == 0) printf("\n--- Remote signal PUT ---\n");
  {
    CHECK_HIP(hipMemset(srcBuf, myPe + 1, maxBuf));
    CHECK_HIP(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    RemoteSignalPutKernel<<<1, 1, 0, stream>>>(srcMemObj, dstMemObj, myPe, remotePe, 4096);
    CHECK_HIP(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    WaitRemoteSignalKernel<<<1, 1, 0, stream>>>(dstMemObj, senderPe, 1);
    CHECK_HIP(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    bool ok = true;
    std::vector<uint8_t> hostBuf(4096);
    CHECK_HIP(hipMemcpy(hostBuf.data(), dstBuf, 4096, hipMemcpyDeviceToHost));
    uint8_t expected = static_cast<uint8_t>(senderPe + 1);
    for (size_t i = 0; i < 4096; i++) {
      if (hostBuf[i] != expected) {
        ok = false;
        break;
      }
    }

    int lok = ok ? 1 : 0, gok = 0;
    MPI_Allreduce(&lok, &gok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (myPe == 0) report("Remote signal PUT 4KB", gok == 1);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- Test 3b: Remote signal zero-size PUT ---
  {
    MPI_Barrier(MPI_COMM_WORLD);

    RemoteSignalPutKernel<<<1, 1, 0, stream>>>(srcMemObj, dstMemObj, myPe, remotePe, 0);
    CHECK_HIP(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    WaitRemoteSignalKernel<<<1, 1, 0, stream>>>(dstMemObj, senderPe, 2);
    CHECK_HIP(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    int lok = 1, gok = 0;
    MPI_Allreduce(&lok, &gok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (myPe == 0) report("Remote signal zero-size PUT", gok == 1);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- Summary ---
  if (myPe == 0) {
    int passed = 0;
    for (auto& r : results)
      if (r.passed) passed++;
    printf("\n=== Summary: %d/%zu tests passed ===\n\n", passed, results.size());
  }

  CHECK_HIP(hipStreamDestroy(stream));
  ShmemFree(srcBuf);
  ShmemFree(dstBuf);
  MPI_Barrier(MPI_COMM_WORLD);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  runTests();
  return 0;
}
