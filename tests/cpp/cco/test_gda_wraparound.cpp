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
// reserveWqeSlots deadlocks when the SQ counters cross 2^32 (ROCm/mori #626).
//
// wq.postIdx / dbTouchIdx / doneIdx are free-running uint32_t serial numbers. The
// flow-control gate at cco_scale_out.hpp:464-473 loads them into uint64_t locals:
//
//     uint64_t dbTouched = ...dbTouchIdx;          // widened, does NOT wrap
//     uint64_t entriesUntilMine = curPostIdx + numWqesNeeded - dbTouched;
//                                 \_______ still uint32, DOES wrap _______/
//
// Once a reservation window ends at or past 2^32 the sum folds to a small value
// while dbTouched stays near 4.29e9, the subtraction underflows to ~1.8e19, no
// free-entry count can exceed it, and the loop spins forever -- on an empty queue.
//
// Reaching 2^32 by actually posting WQEs is not something a test can do, so this
// seeds the counters to the boundary instead. Seeding to 0xFFFFFFFF makes the very
// first reservation wrap for any numWqesNeeded >= 1, and no smaller seed does when
// a put needs a single WQE -- the trigger is curPostIdx + numWqesNeeded >= 2^32.
//
// That seed leaves the QP unusable, and deliberately so: 0xFFFFFFFF % sqWqeNum is
// sqWqeNum-1, not 0, so the counters no longer agree with the NIC's own SQ indices,
// which are still at zero. While the bug is present that never matters -- the gate
// spins before any WQE is written. Once it is fixed the kernel runs on through and
// posts to a slot the NIC is not expecting, so the QP is scrap by the time the
// kernel returns and the test exits without unwinding it (see the end of run_test).
// Only the gate is under test here; delivery is covered elsewhere.
//
// WARNING: this wedges a GPU. reserveWqeSlots has no iteration bound, so the only
// way out is killing the process, which is what the test does after the deadline.
// Check `rocm-smi --showpids` before running on a shared box.
//
// Run:  ./test_gda_wraparound 2      (fork mode, 2 ranks / 2 GPUs)

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cco_test_harness.hpp"
#include "mori/cco/cco_scale_out.hpp"
#include "mori/core/transport/rdma/core_device_types.hpp"

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t COUNT = 64;  // floats per rank-pair -- one WQE's worth

// Any value with (postIdx + numWqesNeeded) >= 2^32 reproduces it; 0xFFFFFFFF does
// so for every numWqesNeeded >= 1.
static const uint32_t kSeed = 0xFFFFFFFFu;

// How long to let the kernel run before calling it a deadlock.
static const int kDeadlineMs = 3000;

// One put per peer. put() -> reserveWqeSlots() is where the spin happens; nothing
// after it is reached.
template <mori::core::ProviderType PrvdType, typename T>
__global__ void GdaPutKernel(mori::cco::ccoWindowDevice* sendWin,
                             mori::cco::ccoWindowDevice* recvWin, size_t count,
                             mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};

  int myRank = devComm.rank;
  size_t perPairBytes = count * sizeof(T);

  for (int r = 0; r < devComm.worldSize; r++) {
    if (r == myRank) continue;
    gda.put(r, reinterpret_cast<ccoWindow_t>(recvWin), myRank * perPairBytes,
            reinterpret_cast<ccoWindow_t>(sendWin), r * perPairBytes, perPairBytes,
            ccoGda_SignalInc{static_cast<ccoGdaSignal_t>(myRank)});
  }
}

// The gate from cco_scale_out.hpp:488-497 in both widths, so the printout shows
// what the seeded QP makes the device compute before and after the fix. `admit` is
// the gate's own `numFreeEntries > entriesUntilMine`.
struct GateTerms {
  uint64_t active, free, until;
  bool admit;
};

// Pre-fix: uint64_t locals. curPostIdx + numWqesNeeded still wraps in uint32 while
// dbTouched does not, so the subtraction underflows.
static GateTerms gateTermsWide(uint32_t dbTouchIdx, uint32_t doneIdx, uint32_t curPostIdx,
                               uint32_t numWqesNeeded, uint32_t sqWqeNum) {
  uint64_t dbTouched = dbTouchIdx;
  uint64_t dbDone = doneIdx;
  uint64_t active = dbTouched - dbDone;
  uint64_t free = sqWqeNum - active;
  uint64_t until = curPostIdx + numWqesNeeded - dbTouched;
  return {active, free, until, free > until};
}

// Post-fix: all uint32_t, so every term wraps in the same modulus.
static GateTerms gateTermsNarrow(uint32_t dbTouchIdx, uint32_t doneIdx, uint32_t curPostIdx,
                                 uint32_t numWqesNeeded, uint32_t sqWqeNum) {
  uint32_t active = dbTouchIdx - doneIdx;
  uint32_t free = sqWqeNum - active;
  uint32_t until = curPostIdx + numWqesNeeded - dbTouchIdx;
  return {active, free, until, free > until};
}

int run_test(int rank, int nranks, const mori::cco::ccoUniqueId& uid) {
  using namespace mori::cco;
  g_rank = rank;

  // The harness leaves each rank via _exit(), which does not flush stdio. Under a
  // pipe stdout is block-buffered, so everything below would be discarded.
  setvbuf(stdout, nullptr, _IONBF, 0);

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  HIP_CHECK(hipSetDevice(rank % numDevices));

  ccoComm* comm = nullptr;
  if (ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  size_t bufSize = COUNT * nranks * sizeof(float);
  void *sendBuf = nullptr, *recvBuf = nullptr;
  if (ccoMemAlloc(comm, bufSize, &sendBuf) != 0 || ccoMemAlloc(comm, bufSize, &recvBuf) != 0) {
    fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    return 1;
  }
  HIP_CHECK(hipMemset(sendBuf, 0, bufSize));
  HIP_CHECK(hipMemset(recvBuf, 0, bufSize));

  ccoWindow_t sendWin = nullptr, recvWin = nullptr;
  if (ccoWindowRegister(comm, sendBuf, bufSize, &sendWin) != 0 ||
      ccoWindowRegister(comm, recvBuf, bufSize, &recvWin) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = CCO_GDA_CONNECTION_FULL;
  reqs.gdaContextCount = 1;
  reqs.gdaSignalCount = nranks;
  reqs.gdaCounterCount = 0;
  ccoDevComm devComm{};
  if (ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }
  if (devComm.gdaConnType == CCO_GDA_CONNECTION_NONE) {
    fprintf(stderr, "[rank %d] gdaConnType is NONE -- no RDMA peers, cannot test\n", rank);
    return 1;
  }

  // ── seed every endpoint's SQ counters to the 2^32 boundary ──
  const size_t numEps = static_cast<size_t>(devComm.worldSize) * devComm.ibgda.numQpPerPe;
  std::vector<mori::core::RdmaEndpointDevice> eps(numEps);
  HIP_CHECK(hipMemcpy(eps.data(), devComm.ibgda.endpoints,
                      numEps * sizeof(mori::core::RdmaEndpointDevice), hipMemcpyDeviceToHost));

  uint32_t sqWqeNum = 0;
  for (size_t i = 0; i < numEps; i++) {
    if (eps[i].GetProviderType() == mori::core::ProviderType::Unknown) continue;  // empty slot
    if (sqWqeNum == 0) sqWqeNum = eps[i].wqHandle.sqWqeNum;
    eps[i].wqHandle.postIdx = kSeed;
    eps[i].wqHandle.dbTouchIdx = kSeed;
    eps[i].wqHandle.doneIdx = kSeed;
  }
  if (sqWqeNum == 0) {
    fprintf(stderr, "[rank %d] no connected endpoint\n", rank);
    return 1;
  }
  HIP_CHECK(hipMemcpy(devComm.ibgda.endpoints, eps.data(),
                      numEps * sizeof(mori::core::RdmaEndpointDevice), hipMemcpyHostToDevice));

  // What the seeded QP makes the gate compute. The queue is empty (active == 0) and
  // has sqWqeNum slots free; the uint64_t gate refuses to admit one WQE anyway.
  const GateTerms wide = gateTermsWide(kSeed, kSeed, kSeed, 1, sqWqeNum);
  const GateTerms narrow = gateTermsNarrow(kSeed, kSeed, kSeed, 1, sqWqeNum);
  printf("[rank %d] seeded postIdx=dbTouchIdx=doneIdx=0x%08X, sqWqeNum=%u\n", rank, kSeed,
         sqWqeNum);
  printf("[rank %d]   gate as uint64 (pre-fix):  active=%llu free=%llu until=%llu -> %s\n", rank,
         (unsigned long long)wide.active, (unsigned long long)wide.free,
         (unsigned long long)wide.until, wide.admit ? "admit" : "SPIN");
  printf("[rank %d]   gate as uint32 (fixed):    active=%llu free=%llu until=%llu -> %s\n", rank,
         (unsigned long long)narrow.active, (unsigned long long)narrow.free,
         (unsigned long long)narrow.until, narrow.admit ? "admit" : "SPIN");

  ccoBarrierAll(comm);

  // One block, one thread: a single wavefront on one CU, to keep the footprint on
  // a shared device as small as the reproduction allows.
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  CCO_GDA_DISPATCH(GdaPutKernel<P, float><<<1, 1, 0, stream>>>(sendWin, recvWin, COUNT, devComm));

  bool finished = false;
  for (int waited = 0; waited < kDeadlineMs; waited += 50) {
    if (hipStreamQuery(stream) == hipSuccess) {
      finished = true;
      break;
    }
    usleep(50 * 1000);
  }

  if (!finished) {
    printf("[rank %d] FAILED: kernel still running after %d ms -- reserveWqeSlots is spinning\n",
           rank, kDeadlineMs);
    printf("[rank %d]   on an empty SQ (%llu slots free) because entriesUntilMine underflowed to\n",
           rank, (unsigned long long)wide.free);
    printf("[rank %d]   %llu. See cco_scale_out.hpp:491. Killing the process to release the GPU.\n",
           rank, (unsigned long long)wide.until);
    fflush(stdout);
    _exit(1);  // the kernel cannot be cancelled; tear the context down
  }

  printf("[rank %d] PASSED: kernel returned -- the gate survives the 2^32 wraparound\n", rank);
  fflush(stdout);

  // The seed left the SQ counters out of step with the NIC's (see the header
  // comment), so the kernel just posted to a slot the NIC is not expecting. Leave
  // without unwinding the QP: deregistering and destroying a desynced endpoint can
  // block on completions that will never arrive, which would turn a passing run
  // into a hang. The process is exiting anyway and the driver reclaims everything.
  HIP_CHECK(hipStreamDestroy(stream));
  _exit(0);  // noreturn
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO GDA wraparound", "/tmp/cco_gda_wraparound_uid", 19893);
}
