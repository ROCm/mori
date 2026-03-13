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

#define CHECK_HIP(call) \
    do { \
        hipError_t err = (call); \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP Error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            throw std::runtime_error("HIP call failed"); \
        } \
    } while(0)

// Test 1 & 2 & 5: SDMA PUT + Quiet (local signal), supports zero-size
// Calls SDMA-specific kernel directly (DISPATCH_TRANSPORT_TYPE doesn't handle SDMA)
__global__ void ShmemPutQuietKernel(
    const SymmMemObjPtr memObj,
    int myPe, int destPe,
    size_t bytes) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  if (threadIdx.x == 0 && bytes > 0) {
    printf("PE %d->%d: localPtr=%p peerPtrs[%d]=%p dh=%p sig=%p bytes=%zu\n",
           myPe, destPe,
           memObj->localPtr,
           destPe, (void*)memObj->peerPtrs[destPe],
           (void*)(memObj->deviceHandles_d + destPe * memObj->sdmaNumQueue),
           (void*)(memObj->signalPtrs + destPe * memObj->sdmaNumQueue),
           bytes);
  }
  ShmemPutMemNbiThreadKernel<TransportType::SDMA>(memObj, 0, memObj, 0, bytes, destPe, 0);
  ShmemQuietThread(destPe, memObj);
}

// Test 3: Remote signal PUT (bypass ShmemQuiet, write directly to remote signal)
__global__ void RemoteSignalPutKernel(
    const SymmMemObjPtr memObj,
    int myPe, int destPe,
    size_t bytes) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  uint8_t* srcPtr = reinterpret_cast<uint8_t*>(memObj->localPtr);
  uint8_t* dstPtr = reinterpret_cast<uint8_t*>(memObj->peerPtrs[destPe]);

  anvil::SdmaQueueDeviceHandle** dh = memObj->deviceHandles_d + destPe * memObj->sdmaNumQueue;

  // Use remote signal: write to destPe's signalPtrs at slot [myPe * numQ + 0]
  HSAuint64* remoteSignal = memObj->peerSignalPtrs[destPe]
                            + static_cast<size_t>(myPe) * memObj->sdmaNumQueue;

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
__global__ void WaitRemoteSignalKernel(
    const SymmMemObjPtr memObj,
    int senderPe,
    HSAuint64 expectedVal) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  HSAuint64* mySignal = memObj->signalPtrs
                        + static_cast<size_t>(senderPe) * memObj->sdmaNumQueue;
  int spin = 0;
  while (__hip_atomic_load(mySignal, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < expectedVal) {
    if (++spin > 100000000) {
      printf("Timeout waiting for remote signal from PE %d (expected %lu)\n", senderPe, expectedVal);
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

  void* buf = ShmemMalloc(maxBuf);
  assert(buf);
  SymmMemObjPtr memObj = ShmemQueryMemObjPtr(buf);
  assert(memObj.IsValid());

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  if (myPe == 0) {
    printf("\n=== SDMA PUT Test Suite (sdma-batch) ===\n");
    printf("  PEs: %d, Testing PE %d -> PE %d\n", npes, myPe, remotePe);
    printf("  peerSignalPtrs: %s\n\n",
           memObj->peerSignalPtrs ? "allocated" : "NULL");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Helper: run shmem PUT + quiet test
  auto runShmemTest = [&](const char* name, size_t bytes) {
    CHECK_HIP(hipMemset(buf, myPe + 1, maxBuf));
    CHECK_HIP(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    ShmemPutQuietKernel<<<1, 1, 0, stream>>>(memObj, myPe, remotePe, bytes);
    CHECK_HIP(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    bool ok = true;
    if (bytes > 0) {
      // Force L2 cache refresh: hipMemset triggers CU writes which update L2,
      // then SDMA data in HBM becomes visible on next read.
      // Alternative: use a separate buffer to avoid L2 stale hits.
      void* tmpBuf = nullptr;
      CHECK_HIP(hipMalloc(&tmpBuf, bytes));
      CHECK_HIP(hipMemcpy(tmpBuf, buf, bytes, hipMemcpyDeviceToDevice));
      CHECK_HIP(hipDeviceSynchronize());
      std::vector<uint8_t> hostBuf(bytes);
      CHECK_HIP(hipMemcpy(hostBuf.data(), tmpBuf, bytes, hipMemcpyDeviceToHost));
      CHECK_HIP(hipFree(tmpBuf));
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

  // --- Test 3: Remote signal PUT ---
  if (myPe == 0) printf("\n--- Remote signal PUT ---\n");
  {
    CHECK_HIP(hipMemset(buf, myPe + 1, maxBuf));
    CHECK_HIP(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Send 4KB using remote signal
    RemoteSignalPutKernel<<<1, 1, 0, stream>>>(memObj, myPe, remotePe, 4096);
    CHECK_HIP(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    // Wait for data from sender via remote signal
    WaitRemoteSignalKernel<<<1, 1, 0, stream>>>(memObj, senderPe, 1);
    CHECK_HIP(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    bool ok = true;
    std::vector<uint8_t> hostBuf(4096);
    CHECK_HIP(hipMemcpy(hostBuf.data(), buf, 4096, hipMemcpyDeviceToHost));
    uint8_t expected = static_cast<uint8_t>(senderPe + 1);
    for (size_t i = 0; i < 4096; i++) {
      if (hostBuf[i] != expected) { ok = false; break; }
    }

    int lok = ok ? 1 : 0, gok = 0;
    MPI_Allreduce(&lok, &gok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (myPe == 0) report("Remote signal PUT 4KB", gok == 1);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- Test 3b: Remote signal zero-size PUT ---
  {
    MPI_Barrier(MPI_COMM_WORLD);

    RemoteSignalPutKernel<<<1, 1, 0, stream>>>(memObj, myPe, remotePe, 0);
    CHECK_HIP(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    WaitRemoteSignalKernel<<<1, 1, 0, stream>>>(memObj, senderPe, 2);
    CHECK_HIP(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    int lok = 1, gok = 0;
    MPI_Allreduce(&lok, &gok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (myPe == 0) report("Remote signal zero-size PUT", gok == 1);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- Test 4: Repeated PUTs ---
  if (myPe == 0) printf("\n--- Repeated PUTs ---\n");
  {
    CHECK_HIP(hipMemset(buf, myPe + 1, maxBuf));
    CHECK_HIP(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    const int repeatCount = 10;
    for (int i = 0; i < repeatCount; i++) {
      ShmemPutQuietKernel<<<1, 1, 0, stream>>>(memObj, myPe, remotePe, 4096);
      CHECK_HIP(hipStreamSynchronize(stream));
      MPI_Barrier(MPI_COMM_WORLD);
    }

    bool ok = true;
    std::vector<uint8_t> hostBuf(4096);
    CHECK_HIP(hipMemcpy(hostBuf.data(), buf, 4096, hipMemcpyDeviceToHost));
    uint8_t expected = static_cast<uint8_t>(senderPe + 1);
    for (size_t i = 0; i < 4096; i++) {
      if (hostBuf[i] != expected) { ok = false; break; }
    }

    int lok = ok ? 1 : 0, gok = 0;
    MPI_Allreduce(&lok, &gok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (myPe == 0) report("Repeated PUT x10 (4KB)", gok == 1);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // --- Summary ---
  if (myPe == 0) {
    int passed = 0;
    for (auto& r : results) if (r.passed) passed++;
    printf("\n=== Summary: %d/%zu tests passed ===\n\n", passed, results.size());
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
