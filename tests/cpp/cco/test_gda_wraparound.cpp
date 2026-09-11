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
// The SQ indices are free-running serial numbers (ROCm/mori #626).
//
// wq.postIdx / dbTouchIdx / doneIdx are uint32_t counters that wrap at 2^32, so
// only their order is meaningful. Treating them as magnitudes breaks two ways:
//
// (A) The serials cross 2^32. The collapsed-CQ drains published completions with an
//     atomic max over raw words, which across the wrap keeps the stale pre-wrap
//     value and freezes doneIdx; the flow-control gate mixed a wrapping uint32 sum
//     into uint64 terms, underflowed, and stopped admitting to an empty SQ; the
//     shmem drain compared doneIdx to dbTouchIdx unsigned and called a queue with
//     WQEs still in flight drained.
//
// (B) A completion arrives ahead of dbTouchIdx, no wrap involved. The doorbell is
//     rung before dbTouchIdx is published, so a bnxt poller can see a con_indx the
//     NIC has already retired while dbTouchIdx is a batch behind; rebuilding against
//     dbTouchIdx then lands a whole queue depth behind doneIdx and the atomic max
//     pins it there. Fires within the first sqWqeNum WQEs of a QP's life -- the
//     failure actually seen on bnxt_re (#653 review), not a 2^32 event.
//
// Part 1 drives both classes on a hand-built endpoint: no NIC, no peer, just a GPU.
// Part 2 is the end-to-end check, a real put on real QPs seeded to the boundary.
//
// Run:  ./test_gda_wraparound                     part 1, then part 2 on all GPUs
//       ./test_gda_wraparound 2                   same, 2 ranks
//       ./test_gda_wraparound --unit-only         part 1 only (no NIC needed)
//       ./test_gda_wraparound --case bnxt_dbrace  one case, in-process, for rocgdb

#include <signal.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cco_test_harness.hpp"  // HIP_CHECK, g_rank, ccoTestMain (part 2)
#include "mori/cco/cco_scale_out.hpp"
#include "mori/core/transport/rdma/core_device_types.hpp"
// shmem's drains are free functions over the same two handles, so part 1 can drive
// them too. Via shmem_device_api.hpp: shmem_ibgda_kernels.hpp specializes templates
// that header declares and does not compile on its own.
#include "mori/shmem/shmem_device_api.hpp"

using mori::core::Mlx5Cqe64;
using mori::core::ProviderType;
using mori::core::RdmaEndpointDevice;
using mori::core::WorkQueueHandle;

/* ══════════════════════════════════════════════════════════════════════════════ */
/*  Part 1 — unit cases over a hand-built endpoint                                */
/* ══════════════════════════════════════════════════════════════════════════════ */
// The endpoint is a core::RdmaEndpointDevice we fill in ourselves, its CQ a buffer
// the test writes, so nothing here touches a QP, a doorbell or an SQ. Neither seed
// is reachable by posting real WQEs, and because the handle lives in coherent host
// memory the host can play the other half of (B) -- publishing dbTouchIdx mid-kernel
// -- and hand a wedged kernel its exit condition rather than leave a runaway kernel
// behind. Each case still runs in its own child process, the parent holding a hard
// deadline as the backstop.

// A spin: the work under test is a handful of loads on an already-satisfiable
// condition, so it finishes in microseconds or never.
static const int kDeadlineMs = 3000;
// Per-case cap: HIP init + the deadline + the release attempt, with room to spare.
static const int kChildDeadlineMs = 30000;

enum Site { kQuiet, kGate, kDrain };

// One thread: both drains are single-poller by construction (mlx5's is lock-free,
// bnxt's runs under pollCqLock) and the gate is per-lane. DrainToLive=false is the
// dbTouchIdx-snapshot drain, the recycle path shmem's flow control leans on.
template <ProviderType P, Site S>
__global__ void SiteKernel(RdmaEndpointDevice* ep, uint32_t target, uint32_t* out) {
  if constexpr (S == kQuiet) {
    mori::cco::impl::quietUntil<P>(ep, target);
  } else if constexpr (S == kGate) {
    *out = mori::cco::impl::reserveWqeSlots<P>(ep, 1);
  } else if constexpr (P == ProviderType::MLX5) {
    mori::shmem::Mlx5CollapsedCqDrain<false>(ep->wqHandle, ep->cqHandle);
  } else {
    mori::shmem::BnxtCollapsedCqDrain<false>(ep->wqHandle, ep->cqHandle);
  }
}

// doneIdx is where the NIC was last seen. target is postIdx, is what the CQE reports
// complete, and is what every case must reach. dbTouchLag is how many WQEs
// dbTouchIdx trails target by: non-zero means the doorbell is rung but dbTouchIdx
// not yet published, and the host publishes it while the kernel runs.
struct Case {
  const char* name;
  Site site;
  ProviderType prvd;
  uint32_t doneIdx;
  uint32_t target;
  uint32_t dbTouchLag;
};

static const Case kCases[] = {
    // (A) the 2^32 wrap. Each is preceded by the same shape far from the boundary,
    // so a failure there means the fake endpoint is wrong rather than the code.
    {"mlx5_plain", kQuiet, ProviderType::MLX5, 0x00001000u, 0x00001010u, 0},
    {"mlx5_cqe16", kQuiet, ProviderType::MLX5, 0x0001FFF8u, 0x00020008u, 0},  // 16-bit CQE wraps
    {"mlx5_wrap32", kQuiet, ProviderType::MLX5, 0xFFFFFFF8u, 0x00000008u, 0},
    {"mlx5_gate32", kGate, ProviderType::MLX5, 0xFFFFFFFFu, 0xFFFFFFFFu, 0},  // empty SQ
    {"mlx5_shmem", kDrain, ProviderType::MLX5, 0x00001000u, 0x00001010u, 0},
    {"mlx5_shmem32", kDrain, ProviderType::MLX5, 0xFFFFFFF8u, 0x00000008u, 0},
    {"bnxt_plain", kQuiet, ProviderType::BNXT, 0x00001000u, 0x00001010u, 0},
    {"bnxt_wrap32", kQuiet, ProviderType::BNXT, 0xFFFFFFF0u, 0x00000000u, 0},

    // (B) the field case: 103 WQEs doorbelled and retired while dbTouchIdx still
    // reads 102. Rebuilding against it gives 103 > 102 and so 103 - 4096 =
    // 0xFFFFF067, the value rocgdb caught on the hung wave. Nothing near 2^32.
    {"bnxt_dbrace", kQuiet, ProviderType::BNXT, 0u, 103u, 1},
};

// bnxt's 4096 is the depth the field case was seen on, and the depth is what the bad
// rebuild subtracts.
static uint32_t SqWqeNum(const Case& c) { return c.prvd == ProviderType::MLX5 ? 1024 : 4096; }
static uint32_t DbTouch(const Case& c) { return c.target - c.dbTouchLag; }

static const char* SiteName(const Case& c) {
  const bool mlx5 = c.prvd == ProviderType::MLX5;
  switch (c.site) {
    case kQuiet:
      return mlx5 ? "quietUntil<MLX5>" : "quietUntil<BNXT>";
    case kGate:
      return mlx5 ? "reserveWqeSlots<MLX5>" : "reserveWqeSlots<BNXT>";
    default:
      return mlx5 ? "Mlx5CollapsedCqDrain" : "BnxtCollapsedCqDrain";
  }
}

// Coherent host memory: device-visible, and unlike device memory writable by the CPU
// while the kernel spins on it.
template <typename T>
static T* AllocShared(size_t bytes) {
  void* p = nullptr;
  HIP_CHECK(hipHostMalloc(&p, bytes, hipHostMallocCoherent | hipHostMallocMapped));
  memset(p, 0, bytes);
  return reinterpret_cast<T*>(p);
}

static RdmaEndpointDevice* BuildEndpoint(const Case& c) {
  auto* ep = AllocShared<RdmaEndpointDevice>(sizeof(RdmaEndpointDevice));
  void* cqe = AllocShared<void>(sizeof(Mlx5Cqe64));  // the larger of the two layouts

  // Both CQEs report a fully drained queue, in their provider's own units: mlx5's
  // wqe_counter is the index of the last WQE retired (the drain adds one), bnxt's
  // con_indx is the count consumed.
  if (c.prvd == ProviderType::MLX5) {
    ep->vendorId = mori::core::RdmaDeviceVendorId::Mellanox;
    auto* mcqe = reinterpret_cast<Mlx5Cqe64*>(cqe);
    mcqe->wqe_counter = HTOBE16(static_cast<uint16_t>(c.target - 1));
    mcqe->op_own = 0;  // opcode 0: a plain completion, so not decoded as an error
  } else {
    ep->vendorId = mori::core::RdmaDeviceVendorId::Broadcom;
    // con_indx is read raw (host order, 16 bits valid); the status is bits 8..15 of
    // the trailing bnxt_re_bcqe flags, where 0 is BNXT_RE_REQ_ST_OK.
    reinterpret_cast<bnxt_re_req_cqe*>(cqe)->con_indx = c.target & 0xFFFFu;
    reinterpret_cast<bnxt_re_bcqe*>(reinterpret_cast<char*>(cqe) + sizeof(bnxt_re_req_cqe))
        ->flg_st_typ_ph = 0;
  }

  WorkQueueHandle& wq = ep->wqHandle;
  wq.sqWqeNum = SqWqeNum(c);
  wq.postIdx = c.target;
  wq.dbTouchIdx = DbTouch(c);
  wq.doneIdx = c.doneIdx;

  // Collapsed CQ: one entry the NIC keeps overwriting with its latest completion.
  // sqAddr / dbrAddr stay null -- no path under test writes a WQE or rings a
  // doorbell, so a null there faults instead of stomping silently.
  ep->cqHandle.cqAddr = cqe;
  ep->cqHandle.cqeNum = 1;
  ep->cqHandle.cqeSize = sizeof(Mlx5Cqe64);
  return ep;
}

// Hand a wedged kernel its exit condition. Seq-cst so the store lands in the coherent
// mapping the wavefront is polling, not in a store buffer.
static void Release(RdmaEndpointDevice* ep, const Case& c) {
  if (c.site == kGate) {
    // curPostIdx + 1 has already wrapped to 0 in the kernel's registers, so zeroing
    // both counters makes entriesUntilMine 0 against a free queue either way.
    __atomic_store_n(&ep->wqHandle.dbTouchIdx, 0u, __ATOMIC_SEQ_CST);
    __atomic_store_n(&ep->wqHandle.doneIdx, 0u, __ATOMIC_SEQ_CST);
  } else {
    __atomic_store_n(&ep->wqHandle.doneIdx, c.target, __ATOMIC_SEQ_CST);
  }
}

// A non-null ep re-applies the release on every step: a drain that keeps the counter
// pinned is issuing back-to-back atomic max on it, so a single host store is likely
// to be read-modify-written away.
static bool Wait(hipStream_t stream, RdmaEndpointDevice* ep, const Case& c) {
  for (int waited = 0; waited < kDeadlineMs; waited += 20) {
    if (ep) Release(ep, c);
    if (hipStreamQuery(stream) == hipSuccess) return true;
    usleep(20 * 1000);
  }
  return false;
}

// Play the posting wave's second half: watch for doneIdx being driven backwards, then
// publish dbTouchIdx as the post path eventually does. Returns the bad value, or 0.
static uint32_t PublishLate(RdmaEndpointDevice* ep, const Case& c) {
  uint32_t bad = 0;
  for (int waited = 0; waited < 200; waited += 2) {
    uint32_t done = __atomic_load_n(&ep->wqHandle.doneIdx, __ATOMIC_SEQ_CST);
    if (static_cast<int32_t>(done - c.doneIdx) < 0) {
      bad = done;
      break;
    }
    usleep(2 * 1000);
  }
  __atomic_store_n(&ep->wqHandle.dbTouchIdx, c.target, __ATOMIC_SEQ_CST);
  return bad;
}

template <ProviderType P>
static void Launch(const Case& c, hipStream_t stream, RdmaEndpointDevice* ep, uint32_t* out) {
  switch (c.site) {
    case kQuiet:
      SiteKernel<P, kQuiet><<<1, 1, 0, stream>>>(ep, c.target, out);
      break;
    case kGate:
      SiteKernel<P, kGate><<<1, 1, 0, stream>>>(ep, c.target, out);
      break;
    case kDrain:
      SiteKernel<P, kDrain><<<1, 1, 0, stream>>>(ep, c.target, out);
      break;
  }
}

// Runs in the child. Returns 0 on pass, 1 on fail.
static int RunCase(const Case& c) {
  setvbuf(stdout, nullptr, _IONBF, 0);  // the child leaves via _exit, which skips flushing
  HIP_CHECK(hipSetDevice(0));

  RdmaEndpointDevice* ep = BuildEndpoint(c);
  uint32_t* out = AllocShared<uint32_t>(sizeof(uint32_t));
  printf("%-13s %-22s done=0x%08X db=0x%08X target=0x%08X  ", c.name, SiteName(c), c.doneIdx,
         DbTouch(c), c.target);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  if (c.prvd == ProviderType::MLX5) {
    Launch<ProviderType::MLX5>(c, stream, ep, out);
  } else {
    Launch<ProviderType::BNXT>(c, stream, ep, out);
  }
  HIP_CHECK(hipGetLastError());

  const uint32_t bad = c.dbTouchLag ? PublishLate(ep, c) : 0;

  if (!Wait(stream, nullptr, c)) {
    const uint32_t done = __atomic_load_n(&ep->wqHandle.doneIdx, __ATOMIC_SEQ_CST);
    printf("FAILED\n  %s still spinning after %d ms: ", SiteName(c), kDeadlineMs);
    if (c.site == kGate) {
      printf("empty SQ, %u slots free, entriesUntilMine underflowed to %llu\n", SqWqeNum(c),
             (unsigned long long)(uint64_t(uint32_t(c.target + 1)) - uint64_t(DbTouch(c))));
    } else {
      printf("doneIdx pinned at 0x%08X, %d WQEs short\n", done,
             -static_cast<int32_t>(done - c.target));
    }
    if (bad)
      printf("  doneIdx had been driven back to 0x%08X (= %u - %u)\n", bad, c.target, SqWqeNum(c));
    if (!Wait(stream, ep, c)) {
      printf("  release did not free it; leaving the kernel to the parent\n");
      _exit(1);
    }
    return 1;
  }

  // Ran to completion -- check it did the right thing, not just that it exited.
  const uint32_t done = __atomic_load_n(&ep->wqHandle.doneIdx, __ATOMIC_SEQ_CST);
  if (c.site == kGate) {
    if (*out != c.target) {
      printf("FAILED\n  reserved base 0x%08X, expected 0x%08X\n", *out, c.target);
      return 1;
    }
  } else if (done != c.target) {
    printf("FAILED\n  returned with doneIdx=0x%08X: %d WQEs outstanding, and the drain\n", done,
           static_cast<int32_t>(c.target - c.doneIdx));
    printf("  called the queue quiet. Callers reuse those buffers next.\n");
    return 1;
  }

  printf("PASSED\n");
  return 0;
}

// HIP does not survive a fork, so the parent must stay clear of it -- including for
// the "is there a GPU" question, which a child answers through its exit code.
static int ProbeGpuCount() {
  pid_t pid = fork();
  if (pid == 0) {
    int n = 0;
    _exit(hipGetDeviceCount(&n) == hipSuccess ? n : 0);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : 0;
}

// A spinning kernel cannot be cancelled, so each case gets a process of its own and
// the parent kills it if the in-child release did not.
static bool RunCaseInChild(const Case& c) {
  pid_t pid = fork();
  if (pid == 0) _exit(RunCase(c));

  for (int waited = 0; waited < kChildDeadlineMs; waited += 20) {
    int status = 0;
    if (waitpid(pid, &status, WNOHANG) == pid) {
      return WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }
    usleep(20 * 1000);
  }

  printf("  killing the child: %d ms with a kernel that will not end\n", kChildDeadlineMs);
  kill(pid, SIGKILL);
  waitpid(pid, nullptr, 0);
  return false;
}

static int RunUnitCases() {
  const size_t numCases = sizeof(kCases) / sizeof(kCases[0]);
  if (ProbeGpuCount() == 0) {
    printf("=== SQ indices: SKIPPED, no GPU ===\n");
    return 0;
  }

  printf("=== SQ indices (#626) -- %zu cases, no NIC ===\n", numCases);
  int failed = 0;
  for (const Case& c : kCases) {
    if (!RunCaseInChild(c)) failed++;
  }
  if (failed) {
    printf("=== %d/%zu FAILED ===\n\n", failed, numCases);
    return 1;
  }
  printf("=== all %zu passed ===\n\n", numCases);
  return 0;
}

/* ══════════════════════════════════════════════════════════════════════════════ */
/*  Part 2 — end-to-end: a real put with the counters seeded to the boundary       */
/* ══════════════════════════════════════════════════════════════════════════════ */
// 0xFFFFFFFF makes the very first reservation wrap for any numWqesNeeded >= 1, and no
// smaller seed does when a put needs a single WQE.
//
// The seed leaves the QP unusable, deliberately: 0xFFFFFFFF % sqWqeNum is sqWqeNum-1,
// not 0, so the counters no longer agree with the NIC's own SQ indices, still at
// zero. While the bug is present that never matters -- the gate spins before any WQE
// is written. Once fixed the kernel runs on and posts to a slot the NIC is not
// expecting, so the QP is scrap by the time the kernel returns and the test exits
// without unwinding it. Only the gate is under test here; delivery is covered
// elsewhere.
//
// WARNING: this wedges a GPU. reserveWqeSlots has no iteration bound, so the only way
// out is killing the process, which is what happens after the deadline. Check
// `rocm-smi --showpids` before running on a shared box.

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t COUNT = 64;  // floats per rank-pair -- one WQE's worth
static const uint32_t kSeed = 0xFFFFFFFFu;
static const int kE2eDeadlineMs = 3000;

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

  printf("[rank %d] seeded postIdx=dbTouchIdx=doneIdx=0x%08X on an empty SQ, sqWqeNum=%u\n", rank,
         kSeed, sqWqeNum);

  ccoBarrierAll(comm);

  // One block, one thread: a single wavefront on one CU, to keep the footprint on a
  // shared device as small as the reproduction allows.
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  CCO_GDA_DISPATCH(GdaPutKernel<P, float><<<1, 1, 0, stream>>>(sendWin, recvWin, COUNT, devComm));

  bool finished = false;
  for (int waited = 0; waited < kE2eDeadlineMs; waited += 50) {
    if (hipStreamQuery(stream) == hipSuccess) {
      finished = true;
      break;
    }
    usleep(50 * 1000);
  }

  if (!finished) {
    printf("[rank %d] FAILED: kernel still running after %d ms -- reserveWqeSlots is spinning\n",
           rank, kE2eDeadlineMs);
    printf("[rank %d]   on an empty SQ. Killing the process to release the GPU.\n", rank);
    fflush(stdout);
    _exit(1);  // the kernel cannot be cancelled; tear the context down
  }

  printf("[rank %d] PASSED: kernel returned -- the gate survives the 2^32 wraparound\n", rank);
  fflush(stdout);

  // The seed left the SQ counters out of step with the NIC's, so the kernel just
  // posted to a slot the NIC is not expecting. Leave without unwinding the QP:
  // deregistering and destroying a desynced endpoint can block on completions that
  // will never arrive, turning a passing run into a hang. The process is exiting
  // anyway and the driver reclaims everything.
  HIP_CHECK(hipStreamDestroy(stream));
  _exit(0);  // noreturn
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);

  // --case runs one unit case in-process (no fork, no deadline) so rocgdb can sit on
  // the spin. Both flags are consumed here; anything else goes to the harness, whose
  // own argv[1] is a rank count.
  if (argc > 2 && !strcmp(argv[1], "--case")) {
    for (const Case& c : kCases) {
      if (!strcmp(c.name, argv[2])) return RunCase(c);
    }
    fprintf(stderr, "unknown case '%s'\n", argv[2]);
    return 2;
  }
  // MORI_CCO_SKIP_GDA_FULL is set per-runner in CI for boxes without intranode
  // cross-rail RDMA, where part 2's FULL connection cannot form. Part 1 needs no NIC,
  // so honour the same signal here and run only that rather than being skipped whole.
  const bool skipGdaFull = getenv("MORI_CCO_SKIP_GDA_FULL") != nullptr;
  const bool unitOnly = argc > 1 && !strcmp(argv[1], "--unit-only");

  // Part 2 wedges QPs and needs peers, so it is pointless to run when the arithmetic
  // it depends on is already broken.
  if (RunUnitCases() != 0) return 1;
  if (unitOnly) return 0;
  if (skipGdaFull) {
    printf("=== end-to-end put: SKIPPED (MORI_CCO_SKIP_GDA_FULL) ===\n");
    return 0;
  }

  return ccoTestMain(argc, argv, "CCO GDA wraparound", "/tmp/cco_gda_wraparound_uid", 19893);
}
