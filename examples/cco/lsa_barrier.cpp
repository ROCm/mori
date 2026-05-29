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

/*
 * CCo LSA Barrier Example (intra-node ccoLsaBarrierSession)
 *
 * Five UT cases, each on its own dedicated lsaBarrier slot:
 *
 *   slot 0/1/2 — visibility (BLOCK / WARP / THREAD groups)
 *                Per-iter producer/consumer protocol with two syncs per iter,
 *                all looped INSIDE the kernel. Verifies cross-GPU memory
 *                ordering + the inbox addressing math, AND exercises the
 *                back-to-back sync correctness path that depends on wait()
 *                using rolling less-equal comparison (cco_lsa_impl.hpp,
 *                NCCL trick: `(got - (epoch+1)) <= 0x7FFFFFFF`).
 *   slot 3 — stress / cross-session epoch persistence (BLOCK)
 *                K chained kernels × N in-kernel sync()s. ctor must restore
 *                epoch from state[]; dtor must persist it back.
 *   slot 4 — timeout (THREAD)
 *                lsaRank 0 deliberately skips arrive/wait. Survivors enter
 *                bar.sync(timeoutCycles) and must return rc=1 (timeout)
 *                instead of hanging. Skipped when lsaSize<2.
 *
 * Run:
 *   mpirun --allow-run-as-root -np 8 ./examples/lsa_barrier
 */

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>

#include "mori/cco/cco_api.hpp"
#include "mori/cco/cco_coop.hpp"
#include "mori/cco/cco_device_api.hpp"
#include "mori/cco/cco_lsa_impl.hpp"
#include "mori/cco/cco_lsa_types.hpp"
#include "mori/cco/cco_types.hpp"

using namespace mori::cco;

#define HIP_CHECK(cmd)                                                                            \
  do {                                                                                            \
    hipError_t e = (cmd);                                                                         \
    if (e != hipSuccess) {                                                                        \
      std::fprintf(stderr, "HIP error %d (%s) at %s:%d\n", e, hipGetErrorString(e), __FILE__,     \
                   __LINE__);                                                                     \
      std::exit(1);                                                                               \
    }                                                                                             \
  } while (0)

static const size_t PER_RANK_VMM_SIZE = 64ULL * 1024 * 1024;
// One uint32 cookie slot per rank in the symmetric send window.
static const size_t COOKIE_BYTES = 64;

// ============================================================================
// Device kernels
// ============================================================================

// Visibility kernel — generic over Coop type. Loops `iters` rounds inside the
// kernel; each round does (write cookie → bar.sync → read peer cookies →
// bar.sync). Exercises:
//   * cross-GPU visibility (cookies written before sync must be observed by
//     peers after their wait — relies on arrive() emitting a system-scope
//     fence before its relaxed inbox stores, paired with the ACQUIRE load
//     in waitInternal).
//   * back-to-back sync correctness — fast ranks bump inbox slots past
//     slow ranks' expected epoch values; wait() must use rolling less-equal
//     (NCCL trick) to avoid deadlocking on strict equality.
template <typename Coop>
__global__ void barrier_visibility_kernel(ccoDevComm* dc, ccoWindow_t win, size_t off,
                                          uint32_t iters, uint32_t slot, int* errors) {
  Coop g;
  ccoLsaBarrierSession<Coop> bar(g, dc, dc->lsaBarrier, slot);

  const int N = dc->lsaSize;
  const int myRank = dc->lsaRank;
  // Large odd multiplier; cookie = e*MUL + rank stays unique per (iter, rank).
  constexpr uint32_t MUL = 1000003u;

  uint32_t* myBuf = static_cast<uint32_t*>(getLocalPtr(win, off));

  int localErr = 0;
  for (uint32_t e = 1; e <= iters; ++e) {
    if (g.thread_rank() == 0) {
      myBuf[0] = e * MUL + static_cast<uint32_t>(myRank);
    }
    bar.sync(g);

    for (int p = g.thread_rank(); p < N; p += g.size()) {
      uint32_t v = static_cast<uint32_t*>(getLsaPeerPtr(win, p, off))[0];
      uint32_t expect = e * MUL + static_cast<uint32_t>(p);
      if (v != expect) localErr |= 1;
    }
    bar.sync(g);
  }

  // Every thread reports independently: peer p is read by whichever thread has
  // thread_rank()==p (mod size), so the remote peer is NOT always handled by
  // thread 0. Gating the atomicAdd on thread 0 alone would silently drop
  // failures detected by other threads (e.g. rank 0 reading its remote peer on
  // thread 1). Count any thread that saw a mismatch — nonzero == fail.
  if (localErr != 0) {
    atomicAdd(errors, 1);
  }
}

// Stress kernel — many syncs in a tight loop, no payload check. Verifies no
// hang under the rolling-less-equal wait path AND that ctor (epoch restore) /
// dtor (epoch persist) round-trip across separate kernel launches that reuse
// the same barrier slot.
__global__ void barrier_stress_kernel(ccoDevComm* dc, uint32_t iters, uint32_t slot,
                                      int* completedFlag) {
  ccoCoopBlock g;
  ccoLsaBarrierSession<ccoCoopBlock> bar(g, dc, dc->lsaBarrier, slot);
  for (uint32_t e = 0; e < iters; ++e) {
    bar.sync(g);
  }
  if (g.thread_rank() == 0) {
    *completedFlag = 1;
  }
}

// Timeout kernel — lsaRank `rankToSkip` exits without opening a session;
// every other rank enters sync(t) with a finite budget and stores the
// return code into outRc[lsaRank].
__global__ void barrier_timeout_kernel(ccoDevComm* dc, uint32_t slot, uint64_t timeoutCycles,
                                       int rankToSkip, int* outRc) {
  ccoCoopThread g;
  if (dc->lsaRank == rankToSkip) {
    return;
  }
  ccoLsaBarrierSession<ccoCoopThread> bar(g, dc, dc->lsaBarrier, slot);
  int rc = bar.sync(g, timeoutCycles);
  outRc[dc->lsaRank] = rc;
}

// ============================================================================
// UT framework
// ============================================================================

// Shared state passed to every UT function. UTs only need this — they don't
// touch MPI/DevComm setup directly, just consume what main() prepared.
struct UtCtx {
  int rank;          // world rank, used for "rank 0 prints" guards
  ccoComm* comm;     // for ccoBarrierAll between cases
  ccoDevComm* devComm;
  ccoDevComm dcHost;  // host-side snapshot (lsaSize, lsaRank, ...)
  ccoWindow_t sendWin;
  // Scratch device buffers, reused across cases.
  int* devErrors;                      // one int
  int* devRc;                          // [lsaSize]
};

using UtFn = int (*)(UtCtx&);

// Small helper: rank-0 prints a one-line PASS/FAIL summary.
static void log_result(const UtCtx& ctx, const char* name, uint32_t slot, int errors,
                       const char* tail = "") {
  if (ctx.rank != 0) return;
  std::printf("  [%-6s] slot=%u  errors=%d  %s%s\n", name, slot, errors,
              errors == 0 ? "PASS" : "FAIL", tail);
}

// ============================================================================
// UT cases
// ============================================================================

// Generic visibility runner — used by all three Coop-variant UTs.
template <typename Coop>
static int run_visibility(UtCtx& ctx, const char* name, uint32_t slot, uint32_t iters,
                          dim3 grid, dim3 block) {
  HIP_CHECK(hipMemset(ctx.devErrors, 0, sizeof(int)));
  hipLaunchKernelGGL(barrier_visibility_kernel<Coop>, grid, block, 0, 0, ctx.devComm, ctx.sendWin,
                     (size_t)0, iters, slot, ctx.devErrors);
  HIP_CHECK(hipDeviceSynchronize());

  int hostErr = 0;
  HIP_CHECK(hipMemcpy(&hostErr, ctx.devErrors, sizeof(int), hipMemcpyDeviceToHost));
  if (ctx.rank == 0) {
    std::printf("  [%-6s] slot=%u iters=%u  errors=%d  %s\n", name, slot, iters, hostErr,
                hostErr == 0 ? "PASS" : "FAIL");
  }
  ccoBarrierAll(ctx.comm);
  return hostErr == 0 ? 0 : 1;
}

// UT 1 — BLOCK visibility
static int ut_visibility_block(UtCtx& ctx) {
  return run_visibility<ccoCoopBlock>(ctx, "block", /*slot=*/0u,
                                      /*iters=*/10000u, dim3(1), dim3(256));
}

// UT 2 — WARP visibility
static int ut_visibility_warp(UtCtx& ctx) {
  return run_visibility<ccoCoopWarp>(ctx, "warp", /*slot=*/1u,
                                     /*iters=*/10000u, dim3(1), dim3(64));
}

// UT 3 — THREAD visibility
static int ut_visibility_thread(UtCtx& ctx) {
  return run_visibility<ccoCoopThread>(ctx, "thread", /*slot=*/2u,
                                       /*iters=*/3200u, dim3(1), dim3(1));
}

// UT 4 — stress / cross-session epoch persistence (slot 3)
//   K chained kernels × N in-kernel syncs. ctor must restore epoch from
//   state[] and dtor must persist it back. If persistence is broken, the
//   next kernel's wait either hangs (best case: caught here) or
//   false-positives (worst case: silently corrupts other tests).
static int ut_stress(UtCtx& ctx) {
  constexpr uint32_t kStressIters = 10;
  constexpr int kLaunches = 4;
  constexpr uint32_t kSlot = 3u;

  int* devCompleted = nullptr;
  HIP_CHECK(hipMalloc(&devCompleted, sizeof(int)));

  bool ok = true;
  for (int k = 0; k < kLaunches; ++k) {
    HIP_CHECK(hipMemset(devCompleted, 0, sizeof(int)));
    hipLaunchKernelGGL(barrier_stress_kernel, dim3(1), dim3(256), 0, 0, ctx.devComm, kStressIters,
                       kSlot, devCompleted);
    HIP_CHECK(hipDeviceSynchronize());
    int completed = 0;
    HIP_CHECK(hipMemcpy(&completed, devCompleted, sizeof(int), hipMemcpyDeviceToHost));
    if (completed != 1) {
      ok = false;
      break;
    }
    ccoBarrierAll(ctx.comm);
  }

  if (ctx.rank == 0) {
    std::printf("  [stress] slot=%u iters=%u launches=%d  %s\n", kSlot, kStressIters, kLaunches,
                ok ? "PASS" : "FAIL");
  }
  HIP_CHECK(hipFree(devCompleted));
  ccoBarrierAll(ctx.comm);
  return ok ? 0 : 1;
}

// UT 5 — timeout (slot 4)
//   lsaRank 0 skips arrive/wait. Survivors must return rc=1 (timeout)
//   instead of hanging. Skipped when lsaSize<2 (no peers to wait on).
//   50 Mcycles (~5–25 ms across HIP-supported clocks) is enough to bail.
static int ut_timeout(UtCtx& ctx) {
  constexpr uint32_t kSlot = 4u;
  constexpr uint64_t kTimeoutCycles = 500ULL * 1000 * 1000;
  constexpr int kRankToSkip = 0;
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  if (ctx.dcHost.lsaSize < 2) {
    if (ctx.rank == 0) {
      std::printf("  [timeo ] skipped (lsaSize=%d < 2)\n", ctx.dcHost.lsaSize);
    }
    return 0;
  }

  HIP_CHECK(hipMemset(ctx.devRc, 0xFF, sizeof(int) * ctx.dcHost.lsaSize));  // sentinel = -1

  HIP_CHECK(hipEventRecord(start, nullptr));
  hipLaunchKernelGGL(barrier_timeout_kernel, dim3(1), dim3(1), 0, 0, ctx.devComm, kSlot,
                     kTimeoutCycles, kRankToSkip, ctx.devRc);
  HIP_CHECK(hipEventRecord(stop, nullptr));
  
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<int> rcHost(ctx.dcHost.lsaSize);
  HIP_CHECK(hipMemcpy(rcHost.data(), ctx.devRc, sizeof(int) * ctx.dcHost.lsaSize,
                      hipMemcpyDeviceToHost));

  bool ok = true;
  if (ctx.dcHost.lsaRank != kRankToSkip && rcHost[ctx.dcHost.lsaRank] != 1) {
    ok = false;
  }
  if (ctx.rank == 0) {
    float elapsedTime;
    HIP_CHECK(hipEventElapsedTime(&elapsedTime, start, stop));
    std::printf("  [timeo ] slot=%u skipRank=%d  expected rc=1 on survivors  %s elapsedTime=%f ms\n", kSlot,
                kRankToSkip, ok ? "PASS" : "FAIL", elapsedTime);
  }
  ccoBarrierAll(ctx.comm);
  return ok ? 0 : 1;
}

// Aggregate driver — runs all UTs in order, returns total #failures.
static int run_all_tests(UtCtx& ctx) {
  static const struct {
    const char* name;
    UtFn fn;
  } kCases[] = {
      // {"block", ut_visibility_block},
      // {"warp", ut_visibility_warp},
      // {"thread", ut_visibility_thread},
      {"stress", ut_stress},
      // {"timeo", ut_timeout},
  };

  int fails = 0;
  for (const auto& c : kCases) {
    fails += c.fn(ctx);
  }
  if (ctx.rank == 0) {
    std::printf("=== %d/%zu PASS ===\n", static_cast<int>(sizeof(kCases) / sizeof(kCases[0])) -
                                              fails,
                sizeof(kCases) / sizeof(kCases[0]));
  }
  return fails;
}

// ============================================================================
// Host driver — setup, single call to run_all_tests, teardown.
// ============================================================================

int main(int argc, char* argv[]) {
#ifndef MORI_WITH_MPI
  std::fprintf(stderr, "lsa_barrier requires MORI_WITH_MPI (enable WITH_MPI).\n");
  return 1;
#endif

  int rank, nranks;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  // ── Phase 1: communicator ──
  auto* boot = new mori::application::MpiBootstrapNetwork(MPI_COMM_WORLD);
  ccoComm* comm = nullptr;
  assert(ccoCommCreate(boot, PER_RANK_VMM_SIZE, &comm) == 0);

  // ── Phase 2: send window (one uint32 cookie slot) ──
  void* sendBuf = nullptr;
  ccoWindow_t sendWin = nullptr;
  assert(ccoWindowRegister(comm, COOKIE_BYTES, &sendWin, &sendBuf) == 0);
  HIP_CHECK(hipMemset(sendBuf, 0, COOKIE_BYTES));

  // ── Phase 3: DevComm with 5 LSA barrier slots ──
  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
  reqs.lsaBarrierCount = 5;
  ccoDevComm* devComm = nullptr;
  assert(ccoDevCommCreate(comm, &reqs, &devComm) == 0);

  ccoDevComm dcHost{};
  HIP_CHECK(hipMemcpy(&dcHost, devComm, sizeof(dcHost), hipMemcpyDeviceToHost));
  if (rank == 0) {
    std::printf("=== LSA barrier example: world=%d lsa=%d ===\n", dcHost.worldSize, dcHost.lsaSize);
  }

  // ── Scratch buffers reused across UTs ──
  int* devErrors = nullptr;
  int* devRc = nullptr;
  HIP_CHECK(hipMalloc(&devErrors, sizeof(int)));
  HIP_CHECK(hipMalloc(&devRc, sizeof(int) * dcHost.lsaSize));

  ccoBarrierAll(comm);

  // ── Run ALL UTs in one shot ──
  UtCtx ctx{rank, comm, devComm, dcHost, sendWin, devErrors, devRc};
  int fails = run_all_tests(ctx);

  // ── Teardown ──
  HIP_CHECK(hipFree(devRc));
  HIP_CHECK(hipFree(devErrors));
  ccoDevCommDestroy(comm, devComm);
  ccoWindowDeregister(comm, sendWin);
  ccoMemFree(comm, sendBuf);

  // Bootstrap ownership transfers to ccoComm at ccoCommCreate; ccoCommDestroy
  // does `bootNet->Finalize()` + `delete bootNet`, which calls MPI_Finalize().
  ccoCommDestroy(comm);

  return fails != 0 ? 1 : 0;
}
