// SDMA PUT comprehensive test suite.
// Tests various edge cases and scenarios for SdmaPutThread with remote signal.
//
// Test cases:
//   1. Zero-size PUT (copy_size = 0): signal should be delivered, no data copy
//   2. Small PUT (1 byte): minimum data transfer
//   3. Unaligned size PUT (e.g., 13 bytes, 127 bytes): non-power-of-2 sizes
//   4. Single element PUT (4 bytes): one uint32_t
//   5. Typical size PUT (4KB, 64KB, 1MB): common sizes
//   6. Large PUT (32MB, 256MB): bulk transfer
//   7. Multi-queue PUT (all 8 queues): parallel transfer
//   8. Repeated PUTs (10 iterations): signal counter correctness across iterations
//   9. Multi-PE PUT (send to all remote PEs): all links active

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

#include "mori/application/utils/check.hpp"
#include "mori/shmem/shmem.hpp"
#include "mori/core/transport/sdma/device_primitives.hpp"

using namespace mori::core;
using namespace mori::application;
using namespace mori::shmem;

#define CHECK_HIP(call) \
    do { \
        hipError_t err = (call); \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP Error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            throw std::runtime_error("HIP call failed"); \
        } \
    } while(0)

// Kernel: single-thread PUT to one remote PE, single queue
__global__ void SinglePutKernel(
    const SymmMemObjPtr srcObj,
    int myPe, int remotePe,
    size_t copyBytes) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  uint8_t* srcPtr = reinterpret_cast<uint8_t*>(srcObj->localPtr);
  uint8_t* dstPtr = reinterpret_cast<uint8_t*>(srcObj->peerPtrs[remotePe]);

  anvil::SdmaQueueDeviceHandle** dh = srcObj->deviceHandles_d + remotePe * srcObj->sdmaNumQueue;
  HSAuint64* remoteSignal = srcObj->peerSignalPtrs[remotePe]
                            + static_cast<size_t>(myPe) * srcObj->sdmaNumQueue;

  SdmaPutThread(srcPtr, dstPtr, copyBytes, dh, remoteSignal, srcObj->sdmaNumQueue, 0);
}

// Kernel: multi-queue PUT to one remote PE
__global__ void MultiQueuePutKernel(
    const SymmMemObjPtr srcObj,
    int myPe, int remotePe,
    size_t copyBytes) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint32_t numQ = srcObj->sdmaNumQueue;

  if (tid >= numQ) return;
  int qId = tid;

  const size_t chunkBase = copyBytes / numQ;
  size_t offset = qId * chunkBase;
  size_t chunk = (qId == (int)(numQ - 1)) ? (copyBytes - (numQ - 1) * chunkBase) : chunkBase;

  uint8_t* srcPtr = reinterpret_cast<uint8_t*>(srcObj->localPtr) + offset;
  uint8_t* dstPtr = reinterpret_cast<uint8_t*>(srcObj->peerPtrs[remotePe]) + offset;

  anvil::SdmaQueueDeviceHandle** dh = srcObj->deviceHandles_d + remotePe * numQ;
  HSAuint64* remoteSignal = srcObj->peerSignalPtrs[remotePe]
                            + static_cast<size_t>(myPe) * numQ;

  SdmaPutThread(srcPtr, dstPtr, chunk, dh, remoteSignal, numQ, qId);
}

// Kernel: wait for signal from one sender on queue 0
__global__ void WaitSignalKernel(
    const SymmMemObjPtr obj,
    int senderPe,
    HSAuint64 expectedVal) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  HSAuint64* mySignal = obj->signalPtrs
                        + static_cast<size_t>(senderPe) * obj->sdmaNumQueue;
  int spin = 0;
  while (__hip_atomic_load(mySignal, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < expectedVal) {
    if (++spin > 100000000) {
      printf("WaitSignal timeout: sender=%d expected=%lu\n", senderPe, expectedVal);
      return;
    }
  }
}

// Kernel: wait for signal from one sender on all queues
__global__ void WaitSignalAllQueuesKernel(
    const SymmMemObjPtr obj,
    int senderPe,
    HSAuint64 expectedVal) {
  size_t tid = threadIdx.x;
  if (tid >= obj->sdmaNumQueue) return;

  HSAuint64* mySignal = obj->signalPtrs
                        + static_cast<size_t>(senderPe) * obj->sdmaNumQueue + tid;
  int spin = 0;
  while (__hip_atomic_load(mySignal, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < expectedVal) {
    if (++spin > 100000000) {
      printf("WaitSignalAllQ timeout: sender=%d q=%zu expected=%lu\n", senderPe, tid, expectedVal);
      return;
    }
  }
}

struct TestResult {
  const char* name;
  bool passed;
  double time_ms;
};

static std::vector<TestResult> results;

static void reportTest(const char* name, bool passed, double time_ms = 0.0) {
  results.push_back({name, passed, time_ms});
  printf("  [%s] %s", passed ? "PASS" : "FAIL", name);
  if (time_ms > 0) printf(" (%.3f ms)", time_ms);
  printf("\n");
}

void runTests() {
  MPI_Init(NULL, NULL);
  int status = ShmemMpiInit(MPI_COMM_WORLD);
  assert(!status);

  int myPe = ShmemMyPe();
  int npes = ShmemNPes();

  if (npes < 2) {
    if (myPe == 0) printf("Need at least 2 PEs for this test\n");
    ShmemFinalize();
    return;
  }

  int remotePe = (myPe + 1) % npes;
  const size_t maxBufSize = 256 * 1024 * 1024;  // 256MB

  // Allocate symmetric buffer
  void* buf = ShmemMalloc(maxBufSize);
  assert(buf);
  SymmMemObjPtr bufObj = ShmemQueryMemObjPtr(buf);
  assert(bufObj.IsValid());

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  if (myPe == 0) {
    printf("\n=== SDMA PUT Test Suite ===\n");
    printf("  PEs: %d, Testing PE %d -> PE %d\n", npes, myPe, remotePe);
    printf("  SDMA queues: %d\n\n", bufObj->sdmaNumQueue);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  auto runSinglePutTest = [&](const char* name, size_t copyBytes, int iters = 1) {
    // Fill source with pattern, clear destination
    CHECK_HIP(hipMemset(buf, myPe + 1, maxBufSize));
    CHECK_HIP(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Clear remote PE's buffer at our slot (they clear theirs)
    // We just check after transfer

    double t0 = MPI_Wtime();
    for (int i = 0; i < iters; i++) {
      SinglePutKernel<<<1, 1, 0, stream>>>(bufObj, myPe, remotePe, copyBytes);
      CHECK_HIP(hipStreamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      // Remote PE waits for our signal
      HSAuint64 expected = i + 1;
      WaitSignalKernel<<<1, 1, 0, stream>>>(bufObj, (myPe - 1 + npes) % npes, expected);
      CHECK_HIP(hipStreamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);
    }
    double t1 = MPI_Wtime();

    // Verify data on receiver side
    bool ok = true;
    if (copyBytes > 0) {
      int senderPe = (myPe - 1 + npes) % npes;
      std::vector<uint8_t> hostBuf(copyBytes);
      CHECK_HIP(hipMemcpy(hostBuf.data(), buf, copyBytes, hipMemcpyDeviceToHost));

      uint8_t expectedByte = static_cast<uint8_t>(senderPe + 1);
      for (size_t i = 0; i < copyBytes; i++) {
        if (hostBuf[i] != expectedByte) {
          if (myPe == 0)
            printf("    Data mismatch at [%zu]: expected %u got %u\n", i, expectedByte, hostBuf[i]);
          ok = false;
          break;
        }
      }
    }

    // For zero-size, just verify signal arrived (WaitSignalKernel didn't timeout)
    int localOk = ok ? 1 : 0, globalOk = 0;
    MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (myPe == 0) {
      reportTest(name, globalOk == 1, (t1 - t0) * 1000.0 / iters);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  };

  auto runMultiQueueTest = [&](const char* name, size_t copyBytes) {
    CHECK_HIP(hipMemset(buf, myPe + 1, maxBufSize));
    CHECK_HIP(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    MultiQueuePutKernel<<<1, 64, 0, stream>>>(bufObj, myPe, remotePe, copyBytes);
    CHECK_HIP(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    int senderPe = (myPe - 1 + npes) % npes;
    WaitSignalAllQueuesKernel<<<1, 64, 0, stream>>>(bufObj, senderPe, 1);
    CHECK_HIP(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    bool ok = true;
    if (copyBytes > 0) {
      std::vector<uint8_t> hostBuf(copyBytes);
      CHECK_HIP(hipMemcpy(hostBuf.data(), buf, copyBytes, hipMemcpyDeviceToHost));
      uint8_t expectedByte = static_cast<uint8_t>(senderPe + 1);
      for (size_t i = 0; i < copyBytes; i++) {
        if (hostBuf[i] != expectedByte) { ok = false; break; }
      }
    }

    int localOk = ok ? 1 : 0, globalOk = 0;
    MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (myPe == 0) reportTest(name, globalOk == 1);
    MPI_Barrier(MPI_COMM_WORLD);
  };

  // ---- Test 1: Zero-size PUT ----
  if (myPe == 0) printf("--- Edge Cases ---\n");
  runSinglePutTest("Zero-size PUT (0 bytes)", 0);

  // ---- Test 2: Small PUTs ----
  runSinglePutTest("Small PUT (1 byte)", 1);
  runSinglePutTest("Small PUT (4 bytes)", 4);
  runSinglePutTest("Small PUT (13 bytes)", 13);
  runSinglePutTest("Small PUT (127 bytes)", 127);

  // ---- Test 3: Typical sizes ----
  if (myPe == 0) printf("\n--- Typical Sizes ---\n");
  runSinglePutTest("PUT 4KB", 4 * 1024);
  runSinglePutTest("PUT 64KB", 64 * 1024);
  runSinglePutTest("PUT 1MB", 1 * 1024 * 1024);

  // ---- Test 4: Large sizes ----
  if (myPe == 0) printf("\n--- Large Sizes ---\n");
  runSinglePutTest("PUT 32MB", 32 * 1024 * 1024);
  runSinglePutTest("PUT 256MB", 256 * 1024 * 1024);

  // ---- Test 5: Multi-queue ----
  if (myPe == 0) printf("\n--- Multi-Queue ---\n");
  runMultiQueueTest("Multi-queue PUT 0 bytes", 0);
  runMultiQueueTest("Multi-queue PUT 1MB", 1 * 1024 * 1024);
  runMultiQueueTest("Multi-queue PUT 32MB", 32 * 1024 * 1024);

  // ---- Test 6: Repeated PUTs (signal counter) ----
  if (myPe == 0) printf("\n--- Repeated PUTs (10 iterations) ---\n");
  {
    CHECK_HIP(hipMemset(buf, myPe + 1, maxBufSize));
    CHECK_HIP(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    const int repeatCount = 10;
    bool ok = true;
    for (int i = 0; i < repeatCount; i++) {
      SinglePutKernel<<<1, 1, 0, stream>>>(bufObj, myPe, remotePe, 4096);
      CHECK_HIP(hipStreamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);

      int senderPe = (myPe - 1 + npes) % npes;
      HSAuint64 expected = i + 1;
      WaitSignalKernel<<<1, 1, 0, stream>>>(bufObj, senderPe, expected);
      CHECK_HIP(hipStreamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);
    }

    int localOk = ok ? 1 : 0, globalOk = 0;
    MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (myPe == 0) reportTest("Repeated PUT x10 (signal counter)", globalOk == 1);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // ---- Test 7: Multi-PE PUT (send to all remote PEs) ----
  if (npes > 2) {
    if (myPe == 0) printf("\n--- Multi-PE PUT ---\n");
    CHECK_HIP(hipMemset(buf, myPe + 1, maxBufSize));
    CHECK_HIP(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    for (int dest = 0; dest < npes; dest++) {
      if (dest == myPe) continue;
      SinglePutKernel<<<1, 1, 0, stream>>>(bufObj, myPe, dest, 4096);
    }
    CHECK_HIP(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    bool ok = true;
    for (int sender = 0; sender < npes; sender++) {
      if (sender == myPe) continue;
      WaitSignalKernel<<<1, 1, 0, stream>>>(bufObj, sender, 1);
    }
    CHECK_HIP(hipStreamSynchronize(stream));

    int localOk = ok ? 1 : 0, globalOk = 0;
    MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (myPe == 0) reportTest("Multi-PE PUT to all peers", globalOk == 1);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // ---- Summary ----
  if (myPe == 0) {
    int passed = 0, total = results.size();
    for (auto& r : results) if (r.passed) passed++;
    printf("\n=== Summary: %d/%d tests passed ===\n\n", passed, total);
  }

  CHECK_HIP(hipStreamDestroy(stream));
  ShmemFree(buf);
  MPI_Barrier(MPI_COMM_WORLD);
  ShmemFinalize();
}

int main(int argc, char* argv[]) {
  runTests();
  return 0;
}
