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

// test: cco gda signal atomicity — many writers contending on ONE slot.
//
// Unlike test_cco_gda_signal (which uses per-sender dedicated slots, so writes
// never collide), this test deliberately makes every peer hammer the SAME slot
// to exercise the RDMA atomic fetch-add path under contention. A plain RDMA
// WRITE would lose updates here; atomic add must produce the exact total.
//
// All rounds target slot 0 on the receiver. peers = nRanks - 1 (exclude self).
//
//   Round A — concurrent SignalInc on slot 0:
//       every peer does signal(me, SignalInc{0}).
//       expect: signal[0] == (nRanks - 1)
//
//   Round B — mixed SignalInc + SignalAdd on slot 0:
//       every peer p does signal(me, SignalInc{0})        (+1)
//                    and  signal(me, SignalAdd{0, p+1})   (+ (p+1))
//       expect: signal[0] == sum over p!=me of (1 + (p+1))
//
// Self-contained: does not depend on any shared test harness.

#include "cco_test_harness.hpp"
#include "mori/cco/cco_scale_out.hpp"

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;

static constexpr mori::core::ProviderType kPrvdType = CCO_GDA_BUILD_PROVIDER;  // per-NIC build

// ── shared op/check descriptors ──────────────────────────────────────────────
//
// All "send" rounds reduce to: each peer issues a list of signal ops; all
// "check" rounds reduce to: a single thread reads a list of slots and compares.
// These two POD descriptors + two generic kernels replace the per-round kernels.
enum SigOpKind { SIG_INC = 0, SIG_ADD = 1 };

struct SigOp {
  uint32_t slot;
  int kind;      // SigOpKind
  uint64_t val;  // ADD value; ignored for INC
};

struct SigCheck {
  uint32_t slot;
  uint64_t expected;
};

static constexpr int kMaxSigOps = 4;
static constexpr int kMaxSigChecks = 4;

// Generic SEND: every peer tid (tid != myRank) issues each op via signal(),
// then a collective flush drains all QPs. Covers contend / mixed / reuse /
// multi-slot / big-add rounds — the difference is just the op list.
template <mori::core::ProviderType PrvdType>
__global__ void GdaSignalSendKernel(mori::cco::ccoDevComm devComm, SigOp op0, SigOp op1, SigOp op2,
                                    int nOps) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  int myRank = devComm.rank;
  int nRanks = devComm.worldSize;
  int tid = threadIdx.x;

  if (tid < nRanks && tid != myRank) {
    SigOp ops[3] = {op0, op1, op2};
    for (int i = 0; i < nOps; i++) {
      if (ops[i].kind == SIG_INC)
        gda.signal(tid, ccoGda_SignalInc{static_cast<ccoGdaSignal_t>(ops[i].slot)});
      else
        gda.signal(tid, ccoGda_SignalAdd{static_cast<ccoGdaSignal_t>(ops[i].slot), ops[i].val});
    }
  }
  gda.flush(mori::cco::ccoCoopBlock{});
}

// Generic CHECK: single thread reads each slot (non-consuming readSignal) and
// asserts it equals `expected`. `bits` selects the 32- vs 64-bit read window.
// Covers the exact-total / multi-slot / 32-bit / post-reset(=0) checks.
template <mori::core::ProviderType PrvdType>
__global__ void GdaSignalCheckKernel(mori::cco::ccoDevComm devComm, SigCheck c0, SigCheck c1,
                                     SigCheck c2, int nChecks, int bits, int round,
                                     int* errorFlag) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  if (threadIdx.x != 0) return;
  SigCheck cs[3] = {c0, c1, c2};
  for (int i = 0; i < nChecks; i++) {
    uint64_t got = gda.readSignal(static_cast<ccoGdaSignal_t>(cs[i].slot), bits);
    if (got != cs[i].expected) {
      printf("[rank %d] round %d: signal[%u] = %llu, expected %llu\n", devComm.rank, round,
             cs[i].slot, (unsigned long long)got, (unsigned long long)cs[i].expected);
      atomicExch(errorFlag, 1);
    }
  }
}

// Generic RESET: zero slots [0, nSlots).
template <mori::core::ProviderType PrvdType>
__global__ void GdaSignalResetKernel(mori::cco::ccoDevComm devComm, int nSlots) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0)
    for (int i = 0; i < nSlots; i++) gda.resetSignal(static_cast<ccoGdaSignal_t>(i));
}

// Dedicated-slot SEND: each peer tid signals into ITS OWN slot (slot == myRank)
// on the target — i.e. one sender per slot, no contention. addVal<=0 selects
// SignalInc(+1); addVal>0 selects SignalAdd(+addVal). This is the per-sender
// pattern the standalone signal test exercised.
template <mori::core::ProviderType PrvdType>
__global__ void GdaSignalDedicatedSendKernel(mori::cco::ccoDevComm devComm, uint64_t addVal) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  int myRank = devComm.rank;
  int nRanks = devComm.worldSize;
  int tid = threadIdx.x;
  if (tid < nRanks && tid != myRank) {
    if (addVal == 0)
      gda.signal(tid, ccoGda_SignalInc{static_cast<ccoGdaSignal_t>(myRank)});
    else
      gda.signal(tid, ccoGda_SignalAdd{static_cast<ccoGdaSignal_t>(myRank), addVal});
  }
  gda.flush(mori::cco::ccoCoopBlock{});
}

// Dedicated-slot CHECK: slot s (for every peer s != myRank) must equal `expect`
// (exactly one sender wrote it); my own slot must stay 0.
template <mori::core::ProviderType PrvdType>
__global__ void GdaSignalDedicatedCheckKernel(mori::cco::ccoDevComm devComm, uint64_t expect,
                                              int round, int* errorFlag) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  if (threadIdx.x != 0) return;
  int myRank = devComm.rank;
  int nRanks = devComm.worldSize;
  for (int s = 0; s < nRanks; s++) {
    uint64_t want = (s == myRank) ? 0 : expect;
    uint64_t got = gda.readSignal(static_cast<ccoGdaSignal_t>(s));
    if (got != want) {
      printf("[rank %d] round %d: signal[%d] = %llu, expected %llu\n", myRank, round, s,
             (unsigned long long)got, (unsigned long long)want);
      atomicExch(errorFlag, 1);
    }
  }
}

// Round C only: waitSignal partial/full consume semantics (unique — mutates
// shadow). slot 0 holds `total`; consume k then the rest, asserting the
// remainder after each step.
template <mori::core::ProviderType PrvdType>
__global__ void GdaSignalConsumeKernel(mori::cco::ccoDevComm devComm, uint64_t total, uint64_t k,
                                       int* errorFlag) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  if (threadIdx.x != 0) return;

  gda.waitSignal(0, k);  // consume k
  uint64_t rem = gda.readSignal(0);
  if (rem != total - k) {
    printf("[rank %d] round C: after consume %llu, readSignal=%llu expected %llu\n", devComm.rank,
           (unsigned long long)k, (unsigned long long)rem, (unsigned long long)(total - k));
    atomicExch(errorFlag, 1);
  }

  gda.waitSignal(0, total - k);  // consume the rest
  uint64_t after = gda.readSignal(0);
  if (after != 0) {
    printf("[rank %d] round C: after full consume, readSignal=%llu expected 0\n", devComm.rank,
           (unsigned long long)after);
    atomicExch(errorFlag, 1);
  }
}

// Round D only: waitSignal-consume on slot 1 then assert residual == 0 (the
// per-sub-round drain that proves no-reset shadow reuse stays correct).
template <mori::core::ProviderType PrvdType>
__global__ void GdaSignalReuseCheckKernel(mori::cco::ccoDevComm devComm, uint64_t least, int round,
                                          int* errorFlag) {
  using namespace mori::cco;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  if (threadIdx.x != 0) return;
  gda.waitSignal(1, least);          // consume this sub-round's increments
  uint64_t rem = gda.readSignal(1);  // delta must be back to 0 for next sub-round
  if (rem != 0) {
    printf("[rank %d] round D[%d]: residual delta=%llu expected 0\n", devComm.rank, round,
           (unsigned long long)rem);
    atomicExch(errorFlag, 1);
  }
}

// ── host driver context ──────────────────────────────────────────────────────
//
// Every round is the same shape: launch a kernel, wait for it, then global
// barrier so all remote atomics land (and shadow advances) before the next
// step observes them. `Ctx::step()` captures that boilerplate so each round
// reads as a short, declarative sequence of named steps.
// ── UT context + per-case helpers ────────────────────────────────────────────
//
// Mirrors the lsa_barrier.cpp UT structure: a shared UtCtx holds the comm/stream/
// scratch state and a few declarative helpers (send / check / consume / fresh);
// each test case is a `static int ut_xxx(UtCtx&)` returning 0 on PASS, 1 on FAIL;
// run_all_tests dispatches a {name, fn} table.
struct UtCtx {
  mori::cco::ccoComm* comm;
  mori::cco::ccoDevComm dc;  // host-side snapshot, passed by value to kernels
  hipStream_t stream;
  int* dErr;  // one int, accumulates per-case failures device-side
  int nranks;
  int rank;
  uint64_t peers;  // nranks - 1

  // Run one kernel-launch lambda, sync, then global barrier so all remote
  // atomics land (and shadows advance) before the next step observes them.
  template <typename LaunchFn>
  void step(LaunchFn&& launch) {
    launch();
    HIP_CHECK(hipStreamSynchronize(stream));
    mori::cco::ccoBarrierAll(comm);
  }

  // Reset slots 0,1,2 (buf + shadow) to baseline. Every case opens with this so
  // cases are self-contained and may be reordered / removed / run alone.
  void fresh() {
    step([&] { GdaSignalResetKernel<kPrvdType><<<1, 1, 0, stream>>>(dc, 3); });
  }

  // Each peer issues the given signal op list to its peer slot.
  void send(SigOp a, SigOp b = {}, SigOp c = {}, int n = 1) {
    step([&] { GdaSignalSendKernel<kPrvdType><<<1, nranks, 0, stream>>>(dc, a, b, c, n); });
  }

  // Single-thread readSignal compare against expected (non-consuming).
  void check(int round, int bits, SigCheck a, SigCheck b = {}, SigCheck c = {}, int n = 1) {
    step([&] {
      GdaSignalCheckKernel<kPrvdType><<<1, 1, 0, stream>>>(dc, a, b, c, n, bits, round, dErr);
    });
  }

  // Read back and reset the device failure flag; returns 0 (PASS) / 1 (FAIL).
  int harvest(const char* name) {
    int hErr = 0;
    HIP_CHECK(hipMemcpy(&hErr, dErr, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemset(dErr, 0, sizeof(int)));
    if (rank == 0) printf("  [%-8s] %s\n", name, hErr == 0 ? "PASS" : "FAIL");
    mori::cco::ccoBarrierAll(comm);
    return hErr == 0 ? 0 : 1;
  }
};

static SigOp INC(uint32_t slot) { return SigOp{slot, SIG_INC, 0}; }
static SigOp ADD(uint32_t slot, uint64_t v) { return SigOp{slot, SIG_ADD, v}; }
static SigCheck CHK(uint32_t slot, uint64_t exp) { return SigCheck{slot, exp}; }

// ── UT cases ─────────────────────────────────────────────────────────────────

// concurrent SignalInc on slot 0 → total == peers (atomic loses nothing)
static int ut_inc_contend(UtCtx& c) {
  c.fresh();
  c.send(INC(0));
  c.check(/*round=*/0, /*bits=*/64, CHK(0, c.peers));
  return c.harvest("inc");
}

// mixed Inc + Add on slot 0 → peers * (1 + V)
static int ut_mixed(UtCtx& c) {
  const uint64_t V = 100;
  c.fresh();
  c.send(INC(0), ADD(0, V), {}, /*nOps=*/2);
  c.check(/*round=*/1, /*bits=*/64, CHK(0, c.peers * (1 + V)));
  return c.harvest("mixed");
}

// waitSignal partial then full consume advances the shadow correctly
static int ut_consume(UtCtx& c) {
  if (c.nranks < 3) {  // need total>=2 so a partial k in (0,total) exists
    if (c.rank == 0) printf("  [%-8s] SKIP (need >=3 ranks)\n", "consume");
    return 0;
  }
  c.fresh();
  c.send(INC(0));
  uint64_t k = c.peers / 2;
  c.step(
      [&] { GdaSignalConsumeKernel<kPrvdType><<<1, 1, 0, c.stream>>>(c.dc, c.peers, k, c.dErr); });
  return c.harvest("consume");
}

// multi-round shadow reuse on slot 1, NO reset between sub-rounds
static int ut_reuse(UtCtx& c) {
  c.fresh();  // once up front only
  for (int r = 0; r < 3; r++) {
    c.send(INC(1));
    c.step([&] {
      GdaSignalReuseCheckKernel<kPrvdType><<<1, 1, 0, c.stream>>>(c.dc, c.peers, r, c.dErr);
    });
  }
  return c.harvest("reuse");
}

// slots 0,1,2 accumulate independently (no cross-talk)
static int ut_multislot(UtCtx& c) {
  c.fresh();
  c.send(ADD(0, 1), ADD(1, 2), ADD(2, 3), /*nOps=*/3);
  c.check(/*round=*/4, /*bits=*/64, CHK(0, c.peers * 1), CHK(1, c.peers * 2), CHK(2, c.peers * 3),
          /*nChecks=*/3);
  return c.harvest("mslot");
}

// 32-bit readSignal window returns the correct (non-wrapping) delta
static int ut_bits32(UtCtx& c) {
  c.fresh();
  c.send(INC(0));
  c.check(/*round=*/5, /*bits=*/32, CHK(0, c.peers));
  return c.harvest("bits32");
}

// large SignalAdd values: peers * (1<<40), no 64-bit truncation
static int ut_big_add(UtCtx& c) {
  const uint64_t BIG = 1ULL << 40;
  c.fresh();
  c.send(ADD(0, BIG));
  c.check(/*round=*/6, /*bits=*/64, CHK(0, c.peers * BIG));
  return c.harvest("bigadd");
}

// resetSignal clears cleanly: post-reset delta==0, then reuse counts from 0
static int ut_reset(UtCtx& c) {
  c.fresh();
  c.check(/*round=*/7, /*bits=*/64, CHK(0, 0));  // post-reset: delta == 0
  c.send(INC(0));
  c.check(/*round=*/8, /*bits=*/64, CHK(0, c.peers));
  return c.harvest("reset");
}

// dedicated per-sender slots: each peer s writes only slot s (no contention).
// SignalInc → every slot s!=me == 1; SignalAdd{42} → == 42; own slot == 0.
static int ut_dedicated(UtCtx& c) {
  // reset all nranks slots (fresh() only covers 0,1,2)
  c.step([&] { GdaSignalResetKernel<kPrvdType><<<1, 1, 0, c.stream>>>(c.dc, c.nranks); });

  // round 1: SignalInc into own slot
  c.step([&] { GdaSignalDedicatedSendKernel<kPrvdType><<<1, c.nranks, 0, c.stream>>>(c.dc, 0); });
  c.step([&] {
    GdaSignalDedicatedCheckKernel<kPrvdType><<<1, 1, 0, c.stream>>>(c.dc, /*expect=*/1, 9, c.dErr);
  });

  // round 2: reset, then SignalAdd{42} into own slot
  c.step([&] { GdaSignalResetKernel<kPrvdType><<<1, 1, 0, c.stream>>>(c.dc, c.nranks); });
  c.step([&] { GdaSignalDedicatedSendKernel<kPrvdType><<<1, c.nranks, 0, c.stream>>>(c.dc, 42); });
  c.step([&] {
    GdaSignalDedicatedCheckKernel<kPrvdType>
        <<<1, 1, 0, c.stream>>>(c.dc, /*expect=*/42, 10, c.dErr);
  });
  return c.harvest("dedic");
}

using UtFn = int (*)(UtCtx&);

static int run_all_tests(UtCtx& ctx) {
  static const struct {
    const char* name;
    UtFn fn;
  } kCases[] = {
      {"inc", ut_inc_contend}, {"mixed", ut_mixed},     {"consume", ut_consume},
      {"reuse", ut_reuse},     {"mslot", ut_multislot}, {"bits32", ut_bits32},
      {"bigadd", ut_big_add},  {"reset", ut_reset},     {"dedic", ut_dedicated},
  };
  int fails = 0;
  for (const auto& c : kCases) fails += c.fn(ctx);
  const int n = static_cast<int>(sizeof(kCases) / sizeof(kCases[0]));
  if (ctx.rank == 0) printf("=== %d/%d PASS ===\n", n - fails, n);
  return fails;
}

// ── host driver: setup → run_all_tests → teardown ────────────────────────────

int run_test(int rank, int nranks, const mori::cco::ccoUniqueId& uid) {
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
  if (mori::cco::ccoCommCreate(uid, nranks, rank, PER_RANK_VMM_SIZE, &comm) != 0) {
    fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_FULL;
  reqs.gdaContextCount = 1;
  reqs.gdaSignalCount = std::max(nranks, 3);  // slot 0 contended; mslot uses 0,1,2
  reqs.gdaCounterCount = 0;
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

  UtCtx ctx{
      comm, devComm, /*stream=*/nullptr, dErr, nranks, rank, static_cast<uint64_t>(nranks - 1)};
  HIP_CHECK(hipStreamCreate(&ctx.stream));

  mori::cco::ccoBarrierAll(comm);
  int fails = run_all_tests(ctx);

  HIP_CHECK(hipStreamDestroy(ctx.stream));
  HIP_CHECK(hipFree(dErr));
  mori::cco::ccoDevCommDestroy(comm, &devComm);
  mori::cco::ccoCommDestroy(comm);

  printf("[rank %d] %s\n", rank, fails == 0 ? "PASSED" : "FAILED");
  return fails == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO GDA signal UT", "/tmp/cco_gda_signal_ut_uid", 19880);
}
