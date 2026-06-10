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

#ifdef MORI_WITH_MPI
#include <mpi.h>

#include "mori/application/bootstrap/mpi_bootstrap.hpp"
#endif

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "hip/hip_runtime.h"
#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/cco/cco.hpp"
#include "mori/cco/cco_device.hpp"
#include "mori/shmem/internal.hpp"

static int g_rank = 0;

#define HIP_CHECK(cmd)                                                                           \
  do {                                                                                           \
    hipError_t e = (cmd);                                                                        \
    if (e != hipSuccess) {                                                                       \
      fprintf(stderr, "[rank %d] HIP error %d (%s) at %s:%d\n", g_rank, e, hipGetErrorString(e), \
              __FILE__, __LINE__);                                                               \
      _exit(1);                                                                                  \
    }                                                                                            \
  } while (0)

static const size_t PER_RANK_VMM_SIZE = 256ULL * 1024 * 1024;

static constexpr mori::core::ProviderType kPrvdType = mori::core::ProviderType::PSD;

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
  using namespace mori::cco::gda;
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
  using namespace mori::cco::gda;
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
  using namespace mori::cco::gda;
  ccoGda<PrvdType> gda{devComm, /*ginContext=*/0};
  if (threadIdx.x == 0)
    for (int i = 0; i < nSlots; i++) gda.resetSignal(static_cast<ccoGdaSignal_t>(i));
}

// Round C only: waitSignal partial/full consume semantics (unique — mutates
// shadow). slot 0 holds `total`; consume k then the rest, asserting the
// remainder after each step.
template <mori::core::ProviderType PrvdType>
__global__ void GdaSignalConsumeKernel(mori::cco::ccoDevComm devComm, uint64_t total, uint64_t k,
                                       int* errorFlag) {
  using namespace mori::cco::gda;
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
  using namespace mori::cco::gda;
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

using UtFn = int (*)(UtCtx&);

static int run_all_tests(UtCtx& ctx) {
  static const struct {
    const char* name;
    UtFn fn;
  } kCases[] = {
      {"inc", ut_inc_contend}, {"mixed", ut_mixed},   {"consume", ut_consume}, {"reuse", ut_reuse},
      {"mslot", ut_multislot}, {"bits32", ut_bits32}, {"bigadd", ut_big_add},  {"reset", ut_reset},
  };
  int fails = 0;
  for (const auto& c : kCases) fails += c.fn(ctx);
  const int n = static_cast<int>(sizeof(kCases) / sizeof(kCases[0]));
  if (ctx.rank == 0) printf("=== %d/%d PASS ===\n", n - fails, n);
  return fails;
}

// ── host driver: setup → run_all_tests → teardown ────────────────────────────

static int run_test(int rank, int nranks, mori::application::BootstrapNetwork* bootNet) {
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

  mori::cco::ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = mori::cco::CCO_GDA_CONNECTION_FULL;
  reqs.gdaContextCount = 1;
  reqs.gdaSignalCount = std::max(nranks, 3);  // slot 0 contended; mslot uses 0,1,2
  reqs.gdaCounterCount = 0;
  mori::cco::ccoDevComm* devComm = nullptr;
  if (mori::cco::ccoDevCommCreate(comm, &reqs, &devComm) != 0) {
    fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }

  mori::cco::ccoDevComm devCommHost;
  HIP_CHECK(hipMemcpy(&devCommHost, devComm, sizeof(devCommHost), hipMemcpyDeviceToHost));
  if (devCommHost.gdaConnType == mori::cco::CCO_GDA_CONNECTION_NONE) {
    fprintf(stderr, "[rank %d] gdaConnType collapsed to NONE\n", rank);
    return 1;
  }

  int* dErr = nullptr;
  HIP_CHECK(hipMalloc(&dErr, sizeof(int)));
  HIP_CHECK(hipMemset(dErr, 0, sizeof(int)));

  UtCtx ctx{
      comm, devCommHost, /*stream=*/nullptr, dErr, nranks, rank, static_cast<uint64_t>(nranks - 1)};
  HIP_CHECK(hipStreamCreate(&ctx.stream));

  mori::cco::ccoBarrierAll(comm);
  int fails = run_all_tests(ctx);

  HIP_CHECK(hipStreamDestroy(ctx.stream));
  HIP_CHECK(hipFree(dErr));
  mori::cco::ccoDevCommDestroy(comm, devComm);
  mori::cco::ccoCommDestroy(comm);

  printf("[rank %d] %s\n", rank, fails == 0 ? "PASSED" : "FAILED");
  return fails == 0 ? 0 : 1;
}

// ── fork mode ─────────────────────────────────────────────────────────────────

static void write_file(const char* path, const void* data, size_t len) {
  FILE* f = fopen(path, "wb");
  fwrite(data, 1, len, f);
  fclose(f);
}

static bool read_file(const char* path, void* data, size_t len) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  bool ok = fread(data, 1, len, f) == len;
  fclose(f);
  return ok;
}

static int run_fork_mode(int nranks) {
  char uidPath[256];
  snprintf(uidPath, sizeof(uidPath), "/tmp/cco_gda_signal_ut_uid_%d", getpid());

  printf("=== CCO GDA signal UT (contention) Test (fork, %d ranks) ===\n", nranks);
  fflush(stdout);

  auto uid = mori::application::SocketBootstrapNetwork::GenerateUniqueIdWithInterface("lo", 19880);
  write_file(uidPath, &uid, sizeof(uid));

  std::vector<pid_t> children;
  for (int r = 0; r < nranks; r++) {
    pid_t pid = fork();
    if (pid == 0) {
      mori::application::UniqueId childUid;
      while (!read_file(uidPath, &childUid, sizeof(childUid))) usleep(10000);
      auto* boot = new mori::application::SocketBootstrapNetwork(childUid, r, nranks);
      _exit(run_test(r, nranks, boot));
    }
    children.push_back(pid);
  }

  int fail = 0;
  for (int r = 0; r < nranks; r++) {
    int status = 0;
    waitpid(children[r], &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      fprintf(stderr, "rank %d failed (status=%d)\n", r, status);
      fail++;
    }
  }

  unlink(uidPath);
  printf("\n=== %d/%d PASSED ===\n", nranks - fail, nranks);
  return fail > 0 ? 1 : 0;
}

// ── single-rank mode for cross-host ──────────────────────────────────────────

static int run_single_rank_mode(int argc, char** argv) {
  int rank = -1, worldSize = -1, gpuOffset = -1;
  const char* uidPath = nullptr;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--rank") && i + 1 < argc)
      rank = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--world") && i + 1 < argc)
      worldSize = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--uid-file") && i + 1 < argc)
      uidPath = argv[++i];
    else if (!strcmp(argv[i], "--gpu-offset") && i + 1 < argc)
      gpuOffset = atoi(argv[++i]);
  }
  if (rank < 0 || worldSize <= 0 || !uidPath) return -1;

  mori::application::UniqueId uid;
  for (int tries = 0; tries < 600; tries++) {
    FILE* f = fopen(uidPath, "rb");
    if (f) {
      size_t n = fread(&uid, 1, sizeof(uid), f);
      fclose(f);
      if (n == sizeof(uid)) break;
    }
    usleep(100000);
  }

  if (gpuOffset >= 0) HIP_CHECK(hipSetDevice(rank - gpuOffset));

  auto* boot = new mori::application::SocketBootstrapNetwork(uid, rank, worldSize);
  return run_test(rank, worldSize, boot);
}

static int run_gen_uid_mode(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: --gen-uid IFACE PORT OUTFILE\n");
    return 1;
  }
  const char* iface = argv[2];
  int port = atoi(argv[3]);
  const char* outPath = argv[4];
  auto uid = mori::application::SocketBootstrapNetwork::GenerateUniqueIdWithInterface(iface, port);
  FILE* f = fopen(outPath, "wb");
  if (!f) {
    fprintf(stderr, "fopen(%s) failed\n", outPath);
    return 1;
  }
  fwrite(&uid, 1, sizeof(uid), f);
  fclose(f);
  printf("Wrote UID (%zu bytes) for iface=%s port=%d to %s\n", sizeof(uid), iface, port, outPath);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && !strcmp(argv[1], "--gen-uid")) return run_gen_uid_mode(argc, argv);
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--rank")) return run_single_rank_mode(argc, argv);
  }

#ifdef MORI_WITH_MPI
  int mpiInitialized = 0;
  MPI_Initialized(&mpiInitialized);

  bool underMpi = mpiInitialized || getenv("OMPI_COMM_WORLD_SIZE") || getenv("PMI_SIZE") ||
                  getenv("PMI_RANK") || getenv("SLURM_PROCID");

  if (underMpi) {
    if (!mpiInitialized) MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (rank == 0) printf("=== CCO GDA signal UT (contention) Test (MPI, %d ranks) ===\n", nranks);
    auto* boot = new mori::application::MpiBootstrapNetwork(MPI_COMM_WORLD);
    return run_test(rank, nranks, boot);
  }
#endif

  // fork mode — detect local gpu count
  int nranks = 0;
  for (int i = 0; i < 64; i++) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/class/kfd/kfd/topology/nodes/%d/gpu_id", i);
    FILE* f = fopen(path, "r");
    if (!f) break;
    unsigned long gpuId = 0;
    if (fscanf(f, "%lu", &gpuId) == 1 && gpuId != 0) nranks++;
    fclose(f);
  }
  if (argc > 1) nranks = std::min(atoi(argv[1]), nranks);
  if (nranks < 2) {
    printf("Need at least 2 GPUs.\n");
    return 1;
  }

  return run_fork_mode(nranks);
}
