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

// test: cco gda counter — LOCAL completion signalling (the local-side dual of
// the remote-completion signal tested in test_gda_signal_ut).
//
// Contrast with signal:
//   signal  — REMOTE peer's NIC atomic-adds into our signalBuf after a put's
//             data LANDS on the peer. "did my data arrive?"  (monotonic +
//             shadow; readSignal returns a delta).
//   counter — software-incremented after polling CQ confirms all pending WQEs
//             are locally complete. "is my sendBuf reusable?"  (absolute
//             value, no shadow; readCounter returns the raw count, resetCounter
//             zeroes it).
//
// counter() is a standalone API called after put/flush. It polls CQ for all
// GDA-team peers, then increments counterBuf[id].
//
//   inc    — put to every peer, then counter(CounterInc{p}); counter[p] == 1.
//   multi  — N rounds of put+counter per peer; counter[p] == N.
//   wait   — waitCounter blocks until the local completion count is reached.
//   reset  — resetCounter zeroes a slot; a fresh put+counter counts from 0.

#include "cco_test_harness.hpp"
#include "mori/cco/cco_scale_out.hpp"

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;
static const size_t COUNT = 256;  // float elements per rank-pair slice

static constexpr mori::core::ProviderType kPrvdType = CCO_GDA_BUILD_PROVIDER;  // per-NIC build

// SEND: each thread issues `putsPerPeer` puts to a distinct peer (no counter
// on put — counter is a separate API). After all puts + flush, one counter()
// call per peer increments the per-peer counter slot.
template <mori::core::ProviderType PrvdType, typename T>
__global__ void GdaCounterPutKernel(mori::cco::ccoWindowDevice* sendWin,
                                    mori::cco::ccoWindowDevice* recvWin, size_t count,
                                    int putsPerPeer, mori::cco::ccoDevComm devComm) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};

  int myRank = devComm.rank;
  int nRanks = devComm.worldSize;
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  size_t perPairBytes = count * sizeof(T);

  for (int r = tid; r < nRanks; r += nthreads) {
    if (r == myRank) continue;
    for (int k = 0; k < putsPerPeer; k++) {
      gda.put(r, reinterpret_cast<ccoWindow_t>(recvWin), myRank * perPairBytes,
              reinterpret_cast<ccoWindow_t>(sendWin), r * perPairBytes, perPairBytes);
    }
  }
  gda.flush(ccoCoopBlock{});
  for (int r = 0; r < nRanks; r++) {
    if (r == myRank) continue;
    for (int k = 0; k < putsPerPeer; k++) {
      gda.counter(ccoGda_CounterInc{static_cast<ccoGdaCounter_t>(r)}, ccoCoopBlock{});
    }
  }
}

// CHECK: single thread asserts counter[p] == expect for every peer p != myRank;
// own slot must stay 0. readCounter reads the absolute value (software-driven).
template <mori::core::ProviderType PrvdType>
__global__ void GdaCounterCheckKernel(mori::cco::ccoDevComm devComm, uint64_t expect, int round,
                                      int* errorFlag) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  if (threadIdx.x != 0) return;
  int myRank = devComm.rank;
  int nRanks = devComm.worldSize;
  for (int p = 0; p < nRanks; p++) {
    uint64_t want = (p == myRank) ? 0 : expect;
    uint64_t got = gda.readCounter(static_cast<ccoGdaCounter_t>(p));
    if (got != want) {
      printf("[rank %d] round %d: counter[%d] = %llu, expected %llu\n", myRank, round, p,
             (unsigned long long)got, (unsigned long long)want);
      atomicExch(errorFlag, 1);
    }
  }
}

// WAIT-then-check: issue puts + flush + counter(), then waitCounter blocks
// until counter[p] >= expect for each peer, and readCounter confirms the value.
template <mori::core::ProviderType PrvdType>
__global__ void GdaCounterWaitKernel(mori::cco::ccoDevComm devComm, uint64_t expect,
                                     int* errorFlag) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  if (threadIdx.x != 0) return;
  int myRank = devComm.rank;
  int nRanks = devComm.worldSize;
  for (int p = 0; p < nRanks; p++) {
    if (p == myRank) continue;
    gda.waitCounter(static_cast<ccoGdaCounter_t>(p), expect);
    uint64_t got = gda.readCounter(static_cast<ccoGdaCounter_t>(p));
    if (got != expect) {
      printf("[rank %d] wait: counter[%d] = %llu, expected %llu\n", myRank, p,
             (unsigned long long)got, (unsigned long long)expect);
      atomicExch(errorFlag, 1);
    }
  }
}

// RESET all counter slots [0, nSlots).
template <mori::core::ProviderType PrvdType>
__global__ void GdaCounterResetKernel(mori::cco::ccoDevComm devComm, int nSlots) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0)
    for (int i = 0; i < nSlots; i++) gda.resetCounter(static_cast<ccoGdaCounter_t>(i));
}

// ── UT context + helpers ─────────────────────────────────────────────────────
struct UtCtx {
  mori::cco::ccoComm* comm;
  mori::cco::ccoDevComm dc;
  mori::cco::ccoWindow_t sendWin;
  mori::cco::ccoWindow_t recvWin;
  hipStream_t stream;
  int* dErr;
  int nranks;
  int rank;
  uint64_t peers;

  template <typename LaunchFn>
  void step(LaunchFn&& launch) {
    launch();
    HIP_CHECK(hipStreamSynchronize(stream));
    mori::cco::ccoBarrierAll(comm);
  }
  void resetAll() {
    step([&] { GdaCounterResetKernel<kPrvdType><<<1, 1, 0, stream>>>(dc, nranks); });
  }
  void put(int putsPerPeer) {
    step([&] {
      GdaCounterPutKernel<kPrvdType, float>
          <<<1, 64, 0, stream>>>(sendWin, recvWin, COUNT, putsPerPeer, dc);
    });
  }
  void check(uint64_t expect, int round) {
    step([&] { GdaCounterCheckKernel<kPrvdType><<<1, 1, 0, stream>>>(dc, expect, round, dErr); });
  }
  int harvest(const char* name) {
    int hErr = 0;
    HIP_CHECK(hipMemcpy(&hErr, dErr, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemset(dErr, 0, sizeof(int)));
    if (rank == 0) printf("  [%-8s] %s\n", name, hErr == 0 ? "PASS" : "FAIL");
    mori::cco::ccoBarrierAll(comm);
    return hErr == 0 ? 0 : 1;
  }
};

// ── UT cases ─────────────────────────────────────────────────────────────────

// one put per peer → local counter[p] == 1
static int ut_inc(UtCtx& c) {
  c.resetAll();
  c.put(/*putsPerPeer=*/1);
  c.check(/*expect=*/1, /*round=*/0);
  return c.harvest("inc");
}

// N puts per peer → counter[p] == N (absolute count, no shadow consumption)
static int ut_multi(UtCtx& c) {
  const int N = 5;
  c.resetAll();
  c.put(/*putsPerPeer=*/N);
  c.check(/*expect=*/N, /*round=*/1);
  return c.harvest("multi");
}

// waitCounter blocks until the local completion count is reached
static int ut_wait(UtCtx& c) {
  const int N = 3;
  c.resetAll();
  c.put(/*putsPerPeer=*/N);
  c.step(
      [&] { GdaCounterWaitKernel<kPrvdType><<<1, 1, 0, c.stream>>>(c.dc, (uint64_t)N, c.dErr); });
  return c.harvest("wait");
}

// resetCounter zeroes the slot; a fresh put counts from 0 again
static int ut_reset(UtCtx& c) {
  c.resetAll();
  c.put(/*putsPerPeer=*/2);
  c.check(/*expect=*/2, /*round=*/2);
  c.resetAll();
  c.check(/*expect=*/0, /*round=*/3);  // after reset: absolute 0
  c.put(/*putsPerPeer=*/1);
  c.check(/*expect=*/1, /*round=*/4);  // reuse counts from 0
  return c.harvest("reset");
}

using UtFn = int (*)(UtCtx&);

static int run_all_tests(UtCtx& ctx) {
  static const struct {
    const char* name;
    UtFn fn;
  } kCases[] = {
      {"inc", ut_inc},
      {"multi", ut_multi},
      {"wait", ut_wait},
      {"reset", ut_reset},
  };
  int fails = 0;
  for (const auto& c : kCases) fails += c.fn(ctx);
  const int n = static_cast<int>(sizeof(kCases) / sizeof(kCases[0]));
  if (ctx.rank == 0) printf("=== %d/%d PASS ===\n", n - fails, n);
  return fails;
}

// ── host driver ──────────────────────────────────────────────────────────────

int run_test(int rank, int nranks, mori::application::BootstrapNetwork* bootNet) {
  g_rank = rank;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  int dev = rank % numDevices;
  HIP_CHECK(hipSetDevice(dev));
  for (int i = 0; i < numDevices; i++) {
    if (i == dev) continue;
    int canAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, dev, i));
    if (canAccess) (void)hipDeviceEnablePeerAccess(i, 0);
  }

  printf("[rank %d/%d] pid=%d GPU=%d\n", rank, nranks, getpid(), dev);

  mori::cco::ccoComm* comm = nullptr;
  if (mori::cco::ccoCommCreate(bootNet, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  size_t bufSize = COUNT * nranks * sizeof(float);
  void* sendBuf = nullptr;
  void* recvBuf = nullptr;
  if (mori::cco::ccoMemAlloc(comm, bufSize, &sendBuf) != 0 ||
      mori::cco::ccoMemAlloc(comm, bufSize, &recvBuf) != 0) {
    fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    return 1;
  }
  HIP_CHECK(hipMemset(sendBuf, 0, bufSize));
  HIP_CHECK(hipMemset(recvBuf, 0, bufSize));

  mori::cco::ccoWindow_t sendWin = nullptr;
  mori::cco::ccoWindow_t recvWin = nullptr;
  if (mori::cco::ccoWindowRegister(comm, sendBuf, bufSize, &sendWin) != 0 ||
      mori::cco::ccoWindowRegister(comm, recvBuf, bufSize, &recvWin) != 0) {
    fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  // devcomm: full gda connectivity + one counter slot per peer.
  mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_FULL;
  reqs.gdaContextCount = 1;
  reqs.gdaSignalCount = nranks;
  reqs.gdaCounterCount = nranks;
  mori::cco::ccoDevComm devComm{};
  if (mori::cco::ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }
  if (devComm.gdaConnType == mori::cco::CCO_GDA_CONNECTION_NONE) {
    fprintf(stderr, "[rank %d] gdaConnType collapsed to NONE\n", rank);
    return 1;
  }

  int* dErr = nullptr;
  HIP_CHECK(hipMalloc(&dErr, sizeof(int)));
  HIP_CHECK(hipMemset(dErr, 0, sizeof(int)));

  UtCtx ctx{comm,
            devComm,
            sendWin,
            recvWin,
            /*stream=*/nullptr,
            dErr,
            nranks,
            rank,
            static_cast<uint64_t>(nranks - 1)};
  HIP_CHECK(hipStreamCreate(&ctx.stream));

  mori::cco::ccoBarrierAll(comm);
  int fails = run_all_tests(ctx);

  HIP_CHECK(hipStreamDestroy(ctx.stream));
  HIP_CHECK(hipFree(dErr));
  mori::cco::ccoDevCommDestroy(comm, &devComm);
  mori::cco::ccoWindowDeregister(comm, recvWin);
  mori::cco::ccoWindowDeregister(comm, sendWin);
  mori::cco::ccoMemFree(comm, recvBuf);
  mori::cco::ccoMemFree(comm, sendBuf);
  mori::cco::ccoCommDestroy(comm);

  printf("[rank %d] %s\n", rank, fails == 0 ? "PASSED" : "FAILED");
  return fails == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO GDA counter", "/tmp/cco_gda_counter_uid", 19885);
}
