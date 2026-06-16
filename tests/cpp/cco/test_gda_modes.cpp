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
// Test: CCO host-side API lifecycle in SPMT (single-process multi-thread) mode.
// Covers, per rank thread:
//   1. ccoCommCreate
//   2. ccoMemAlloc: small (4 KiB), large (4 MiB), size==0 sentinel, alloc/free reuse
//   3. ccoWindowRegister: both overloads (pre-allocated ptr + convenience alloc)
//   4. ccoDevCommCreate × {NONE, FULL, RAIL}, validating gdaConnectionType honors QP count
//   5. teardown: WindowDeregister → MemFree → DevCommDestroy → CommDestroy
//
// The 4 MiB buffer + multiple subsequent sub-buffers (resource window per DevComm)
// also exercises the ROCm hipMemSetAccess sub-buffer regression path
// (SWDEV-568260, rocm-systems#2516); requires the patched libamdhip64.so from
// docker/Dockerfile.dev to pass.
//
// Single process, N threads.

#include <cstdio>
#include <thread>
#include <vector>

#include "hip/hip_runtime.h"
#include "mori/application/application_device_types.hpp"  // white-box: RdmaEndpointDevice (count QPs)
#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/cco/cco.hpp"
#include "mori/utils/mori_log.hpp"

#define HIP_CHECK(cmd)                                                            \
  do {                                                                            \
    hipError_t e = (cmd);                                                         \
    if (e != hipSuccess) {                                                        \
      fprintf(stderr, "[rank ?] HIP error %d at %s:%d\n", e, __FILE__, __LINE__); \
      exit(1);                                                                    \
    }                                                                             \
  } while (0)

static const size_t PER_RANK_VMM_SIZE = 64ULL * 1024 * 1024;
static const size_t WINDOW_SIZE_SMALL = 4096;            // 4 KiB
static const size_t WINDOW_SIZE_LARGE = 4096ULL * 1024;  // 4 MiB (triggers VMM regression path)

struct Result {
  int rank;
  bool passed;
  char detail[256];
};

// Helper: fail with formatted detail and return early.
#define FAIL(...)                                        \
  do {                                                   \
    snprintf(r->detail, sizeof(r->detail), __VA_ARGS__); \
    return false;                                        \
  } while (0)

// Read DevComm back to host, count non-zero QPs in the IBGDA endpoint array.
static int CountQpsFor(const mori::cco::ccoDevComm& dc, int worldSize) {
  if (dc.ibgda.endpoints == nullptr || dc.ibgda.numQpPerPe == 0) return 0;
  size_t total = static_cast<size_t>(worldSize) * dc.ibgda.numQpPerPe;
  std::vector<mori::application::RdmaEndpointDevice> eps(total);
  HIP_CHECK(
      hipMemcpy(eps.data(), dc.ibgda.endpoints, total * sizeof(eps[0]), hipMemcpyDeviceToHost));
  int count = 0;
  for (const auto& ep : eps) {
    if (ep.qpn != 0) count++;
  }
  return count;
}

// Exercise ccoMemAlloc / ccoMemFree edge cases on an established comm.
// Returns true on success; writes detail to r->detail and returns false on failure.
static bool exercise_mem_alloc(mori::cco::ccoComm* comm, Result* r) {
  // size==0 must return nullptr without error.
  void* z = reinterpret_cast<void*>(0x1);
  if (mori::cco::ccoMemAlloc(comm, 0, &z) != 0 || z != nullptr) {
    FAIL("MemAlloc(size=0) did not return nullptr (z=%p)", z);
  }

  // Round-trip: small alloc, free, alloc again — second should reuse the same slot.
  void* a = nullptr;
  void* b = nullptr;
  if (mori::cco::ccoMemAlloc(comm, WINDOW_SIZE_SMALL, &a) != 0) {
    FAIL("MemAlloc(small) #1 failed");
  }
  if (mori::cco::ccoMemFree(comm, a) != 0) {
    FAIL("MemFree(small) failed");
  }
  if (mori::cco::ccoMemAlloc(comm, WINDOW_SIZE_SMALL, &b) != 0) {
    FAIL("MemAlloc(small) #2 failed");
  }
  if (a != b) {
    FAIL("slot reuse: a=%p b=%p", a, b);
  }
  if (mori::cco::ccoMemFree(comm, b) != 0) {
    FAIL("MemFree(small) #2 failed");
  }
  return true;
}

// Exercise both ccoWindowRegister overloads on an established comm.
// On success, the user-allocated buffer is left registered as `*outWin` /
// `*outBuf` so the caller can verify window structure and clean up.
static bool exercise_window_register(mori::cco::ccoComm* comm, mori::cco::ccoWindow_t* outWin,
                                     void** outBuf, mori::cco::ccoWindow_t* outWinConv,
                                     void** outBufConv, Result* r) {
  // Overload B: pre-allocate via ccoMemAlloc, then register. Use 4 MiB to
  // exercise the VMM sub-buffer regression path (resource window allocations
  // from DevCommCreate later add more sub-buffers).
  void* buf = nullptr;
  if (mori::cco::ccoMemAlloc(comm, WINDOW_SIZE_LARGE, &buf) != 0) {
    FAIL("MemAlloc(large=%zu) failed", WINDOW_SIZE_LARGE);
  }
  HIP_CHECK(hipMemset(buf, 0, WINDOW_SIZE_LARGE));

  mori::cco::ccoWindow_t win = nullptr;
  if (mori::cco::ccoWindowRegister(comm, buf, WINDOW_SIZE_LARGE, &win) != 0) {
    mori::cco::ccoMemFree(comm, buf);
    FAIL("WindowRegister(ptr) failed");
  }
  if (win == nullptr) {
    mori::cco::ccoMemFree(comm, buf);
    FAIL("WindowRegister(ptr) returned null win");
  }

  // Overload A: convenience — library does the MemAlloc internally.
  mori::cco::ccoWindow_t winConv = nullptr;
  void* bufConv = nullptr;
  if (mori::cco::ccoWindowRegister(comm, WINDOW_SIZE_SMALL, &winConv, &bufConv) != 0) {
    mori::cco::ccoWindowDeregister(comm, win);
    mori::cco::ccoMemFree(comm, buf);
    FAIL("WindowRegister(size) convenience overload failed");
  }
  if (winConv == nullptr || bufConv == nullptr) {
    mori::cco::ccoWindowDeregister(comm, winConv);
    mori::cco::ccoWindowDeregister(comm, win);
    mori::cco::ccoMemFree(comm, buf);
    FAIL("WindowRegister(size) returned null (win=%p buf=%p)", winConv, bufConv);
  }

  // Verify the registered window's GPU-side struct reports our buffer back
  // through the flat-VA formula.
  mori::cco::ccoWindowDevice winHost;
  HIP_CHECK(hipMemcpy(&winHost, win, sizeof(winHost), hipMemcpyDeviceToHost));
  void* localVa =
      winHost.winBase + (static_cast<uint64_t>(winHost.lsaRank) * winHost.stride4G << 32);
  if (localVa != buf) {
    mori::cco::ccoWindowDeregister(comm, winConv);
    mori::cco::ccoWindowDeregister(comm, win);
    mori::cco::ccoMemFree(comm,
                          bufConv);  // convenience path: must MemFree even though we didn't alloc
    mori::cco::ccoMemFree(comm, buf);
    FAIL("flat-VA local lookup mismatch: localVa=%p buf=%p", localVa, buf);
  }

  *outWin = win;
  *outBuf = buf;
  *outWinConv = winConv;
  *outBufConv = bufConv;
  return true;
}

static void run_rank(int rank, int nranks, const mori::application::UniqueId& uid, Result* r) {
  r->rank = rank;
  r->passed = false;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int dev = rank % numDevices;
  HIP_CHECK(hipSetDevice(dev));

  // Enable inter-GPU P2P access so flat-VA peer maps work.
  for (int i = 0; i < numDevices; i++) {
    if (i == dev) continue;
    int canAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, dev, i));
    if (canAccess) (void)hipDeviceEnablePeerAccess(i, 0);
  }

  auto* bootNet = new mori::application::SocketBootstrapNetwork(uid, rank, nranks);

  // Phase 1: ccoCommCreate
  mori::cco::ccoComm* comm = nullptr;
  if (mori::cco::ccoCommCreate(bootNet, PER_RANK_VMM_SIZE, &comm) != 0) {
    snprintf(r->detail, sizeof(r->detail), "CommCreate failed");
    return;
  }

  // Phase 1.5: ccoMemAlloc / ccoMemFree edge cases
  if (!exercise_mem_alloc(comm, r)) {
    mori::cco::ccoCommDestroy(comm);
    return;
  }

  // Phase 2: ccoWindowRegister × 2 overloads (large buffer to exercise VMM
  // regression path)
  mori::cco::ccoWindow_t win = nullptr, winConv = nullptr;
  void* buf = nullptr;
  void* bufConv = nullptr;
  if (!exercise_window_register(comm, &win, &buf, &winConv, &bufConv, r)) {
    mori::cco::ccoCommDestroy(comm);
    return;
  }

  // Phase 3: ccoDevCommCreate × {NONE, FULL, RAIL}.
  auto makeReqs = [](mori::cco::ccoGdaConnectionType ct) {
    mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.gdaConnectionType = ct;
    reqs.lsaBarrierCount = 4;      // LSA barrier slab in resource window
    reqs.railGdaBarrierCount = 2;  // rail GDA barrier → IBGDA signal pool
    reqs.barrierCount = 3;         // hybrid LSA + rail GDA
    return reqs;
  };

  mori::cco::ccoDevComm dcNone{}, dcFull{}, dcRail{};
  bool haveNone = false, haveFull = false, haveRail = false;
  auto reqsNone = makeReqs(mori::cco::CCO_GDA_CONNECTION_NONE);
  auto reqsFull = makeReqs(mori::cco::CCO_GDA_CONNECTION_FULL);
  auto reqsRail = makeReqs(mori::cco::CCO_GDA_CONNECTION_RAIL);
  const int numQpPerPe = reqsFull.gdaContextCount;

  auto teardown_after_partial_devcomm = [&]() {
    if (haveRail) mori::cco::ccoDevCommDestroy(comm, &dcRail);
    if (haveFull) mori::cco::ccoDevCommDestroy(comm, &dcFull);
    if (haveNone) mori::cco::ccoDevCommDestroy(comm, &dcNone);
    mori::cco::ccoWindowDeregister(comm, winConv);
    mori::cco::ccoWindowDeregister(comm, win);
    mori::cco::ccoMemFree(comm, bufConv);
    mori::cco::ccoMemFree(comm, buf);
    mori::cco::ccoCommDestroy(comm);
  };

  if (mori::cco::ccoDevCommCreate(comm, &reqsNone, &dcNone) != 0) {
    snprintf(r->detail, sizeof(r->detail), "DevCommCreate NONE failed");
    teardown_after_partial_devcomm();
    return;
  }
  haveNone = true;
  if (mori::cco::ccoDevCommCreate(comm, &reqsFull, &dcFull) != 0) {
    snprintf(r->detail, sizeof(r->detail), "DevCommCreate FULL failed");
    teardown_after_partial_devcomm();
    return;
  }
  haveFull = true;
  if (mori::cco::ccoDevCommCreate(comm, &reqsRail, &dcRail) != 0) {
    snprintf(r->detail, sizeof(r->detail), "DevCommCreate RAIL failed");
    teardown_after_partial_devcomm();
    return;
  }
  haveRail = true;

  // Expectations on a uniform N-nodes × lsaSize layout:
  //   NONE : 0
  //   FULL : (worldSize - 1) * qpsPerPe
  //   RAIL : (nNodes - 1) * qpsPerPe  (one peer per other node at same lsaRank)
  // On single-node (nNodes == 1), RAIL collapses to NONE: expected 0.
  const int nNodes = dcFull.worldSize / dcFull.lsaSize;
  const int qpsNone = CountQpsFor(dcNone, dcFull.worldSize);
  const int qpsFull = CountQpsFor(dcFull, dcFull.worldSize);
  const int qpsRail = CountQpsFor(dcRail, dcFull.worldSize);
  const int expectedFull = (dcFull.worldSize - 1) * numQpPerPe;
  const int expectedRail = (nNodes - 1) * numQpPerPe;

  bool ok = true;
  if (qpsNone != 0) {
    snprintf(r->detail, sizeof(r->detail), "NONE: expected 0, got %d", qpsNone);
    ok = false;
  } else if (qpsFull != expectedFull) {
    snprintf(r->detail, sizeof(r->detail), "FULL: expected %d, got %d", expectedFull, qpsFull);
    ok = false;
  } else if (qpsRail != expectedRail) {
    snprintf(r->detail, sizeof(r->detail), "RAIL: expected %d ((nNodes-1)*qpsPerPe=%d*%d), got %d",
             expectedRail, nNodes - 1, numQpPerPe, qpsRail);
    ok = false;
  } else {
    snprintf(r->detail, sizeof(r->detail),
             "alloc/register/devcomm OK; NONE=0 FULL=%d RAIL=%d (worldSize=%d lsaSize=%d "
             "nNodes=%d qpsPerPe=%d)",
             qpsFull, qpsRail, dcFull.worldSize, dcFull.lsaSize, nNodes, numQpPerPe);
  }

  printf("[rank %d] NONE=%d FULL=%d RAIL=%d (expected: 0 / %d / %d)\n", rank, qpsNone, qpsFull,
         qpsRail, expectedFull, expectedRail);

  // Teardown in reverse order of creation:
  // DevComm × 3 → WindowDeregister × 2 → MemFree × 2 → CommDestroy.
  mori::cco::ccoDevCommDestroy(comm, &dcRail);
  mori::cco::ccoDevCommDestroy(comm, &dcFull);
  mori::cco::ccoDevCommDestroy(comm, &dcNone);
  mori::cco::ccoWindowDeregister(comm, winConv);
  mori::cco::ccoWindowDeregister(comm, win);
  mori::cco::ccoMemFree(comm, bufConv);
  mori::cco::ccoMemFree(comm, buf);
  mori::cco::ccoCommDestroy(comm);

  r->passed = ok;
  if (ok) printf("[rank %d] PASSED\n", rank);
}

int main(int argc, char** argv) {
  mori::ModuleLogger::GetInstance().GetLogger(mori::modules::APPLICATION);
  mori::ModuleLogger::GetInstance().GetLogger(mori::modules::SHMEM);

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int nranks = numDevices;
  if (argc > 1) nranks = std::min(atoi(argv[1]), numDevices);
  if (nranks < 2) {
    printf("Need at least 2 GPUs.\n");
    return 1;
  }

  printf("=== CCO GDA Connection Modes Test (%d ranks) ===\n\n", nranks);

  auto uid = mori::application::SocketBootstrapNetwork::GenerateUniqueIdWithInterface("lo", 18458);

  std::vector<Result> results(nranks);
  std::vector<std::thread> threads;
  for (int r = 0; r < nranks; r++) {
    threads.emplace_back(run_rank, r, nranks, std::cref(uid), &results[r]);
  }
  for (auto& t : threads) t.join();

  printf("\n=== Summary ===\n");
  int pass = 0, fail = 0;
  for (auto& r : results) {
    printf("  rank %d: [%s] %s\n", r.rank, r.passed ? "PASS" : "FAIL", r.detail);
    r.passed ? pass++ : fail++;
  }
  printf("\n%d passed, %d failed\n", pass, fail);
  return (fail == 0) ? 0 : 1;
}
