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
 * UT cases, each on its own dedicated lsaBarrier slot:
 *   slot 0/1/2 — visibility, BLOCK/WARP/THREAD: cross-GPU ordering, inbox
 *                addressing, and back-to-back sync (rolling less-equal wait).
 *   slot 3     — stress: K chained kernels exercise ctor/dtor epoch round-trip.
 *   slot 4     — timeout: a rank skips arrive/wait; survivors must return rc=1.
 *   slot 5     — arrive()/wait() split: decoupled producer/consumer path.
 *   slot 6     — epoch persistence: cookie sequence continues across launches
 *                (buffer not reset), so a broken round-trip surfaces as a
 *                payload mismatch, not just a hang.
 *   slot 7/8   — multi-slot isolation: two interleaved sessions must not collide.
 *   slot 9     — epoch wraparound: preset near 2^32, sync across the boundary.
 *   slot 0     — single-rank no-op (lsaSize==1 only).
 *
 * Run:
 *   mpirun --allow-run-as-root -np 8 ./examples/lsa_barrier
 */

#include <algorithm>
#include <cstdlib>

// Always-on check: these tests build under -DNDEBUG (Release), where CCO_MUST()
// would drop the wrapped expression together with its side effects. CCO_MUST
// always evaluates expr and aborts the rank on failure (mpirun then tears down
// the whole job).
#define CCO_MUST(expr)                                                                    \
  do {                                                                                    \
    if (!(expr)) {                                                                        \
      std::fprintf(stderr, "[cco lsa test] CHECK FAILED: %s at %s:%d\n", #expr, __FILE__, \
                   __LINE__);                                                             \
      std::abort();                                                                       \
    }                                                                                     \
  } while (0)
#include <cstdio>
#include <vector>

#include "cco_test_harness.hpp"
#include "mori/cco/cco.hpp"  // CCO single header (host + device)

using namespace mori::cco;

// HIP_CHECK comes from cco_test_harness.hpp.

static const size_t PER_RANK_VMM_SIZE = 64ULL * 1024 * 1024;
// Symmetric send window. Only a few cookie bytes are actually used, but the
// window must be at least the VMM allocation granularity (the convenience
// ccoWindowRegister overload fails for sub-granularity sizes), so size it to
// one page.
static const size_t COOKIE_BYTES = 4096;

// ============================================================================
// Device kernels
// ============================================================================

// Visibility kernel — generic over Coop type. Each round writes a cookie,
// syncs, reads peer cookies, syncs. Exercises cross-GPU visibility (system-
// scope fence in arrive() paired with ACQUIRE load in wait) and back-to-back
// sync correctness (rolling less-equal wait, so fast ranks can't deadlock slow).
template <typename Coop>
__global__ void barrier_visibility_kernel(ccoDevComm dc, ccoWindow_t win, size_t off,
                                          uint32_t epochBase, uint32_t iters, uint32_t slot,
                                          int* errors) {
  Coop g;
  ccoLsaBarrierSession<Coop> bar(g, &dc, ccoTeamLsa(dc), dc.lsaBarrier, slot);

  const int N = dc.lsaSize;
  const int myRank = dc.lsaRank;
  // cookie = tag + rank, unique per (iter, rank). epochBase continues the
  // sequence across launches (persistence UT); pass 0 for a fresh sequence.
  constexpr uint32_t MUL = 1000003u;

  uint32_t* myBuf = static_cast<uint32_t*>(ccoGetLocalPtr(win, off));

  int localErr = 0;
  for (uint32_t e = 1; e <= iters; ++e) {
    uint32_t tag = (epochBase + e) * MUL;
    if (g.thread_rank() == 0) {
      myBuf[0] = tag + static_cast<uint32_t>(myRank);
    }
    bar.sync(g);

    for (int p = g.thread_rank(); p < N; p += g.size()) {
      uint32_t v = static_cast<uint32_t*>(ccoGetLsaPeerPtr(win, p, off))[0];
      if (v != tag + static_cast<uint32_t>(p)) localErr |= 1;
    }
    bar.sync(g);
  }

  // Peers are striped across threads, so any thread may detect a mismatch —
  // every thread reports independently rather than gating on thread 0.
  if (localErr != 0) {
    atomicAdd(errors, 1);
  }
}

// Stress kernel — many syncs in a tight loop, no payload check. Verifies no
// hang under the rolling-less-equal wait path AND that ctor (epoch restore) /
// dtor (epoch persist) round-trip across separate kernel launches that reuse
// the same barrier slot.
__global__ void barrier_stress_kernel(ccoDevComm dc, uint32_t iters, uint32_t slot,
                                      int* completedFlag) {
  ccoCoopBlock g;
  ccoLsaBarrierSession<ccoCoopBlock> bar(g, &dc, ccoTeamLsa(dc), dc.lsaBarrier, slot);
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
__global__ void barrier_timeout_kernel(ccoDevComm dc, uint32_t slot, uint64_t timeoutCycles,
                                       int rankToSkip, int* outRc) {
  ccoCoopThread g;
  if (dc.lsaRank == rankToSkip) {
    return;
  }
  ccoLsaBarrierSession<ccoCoopThread> bar(g, &dc, ccoTeamLsa(dc), dc.lsaBarrier, slot);
  int rc = bar.sync(g, timeoutCycles);
  outRc[dc.lsaRank] = rc;
}

// Timeout kernel (decoupled) — same setup as barrier_timeout_kernel, but the
// survivors use the direct arrive() + wait(coop, timeoutCycles) pair instead of
// the fused sync(coop, timeoutCycles). Exercises the 2-arg wait() overload
// directly; rankToSkip never arrives, so survivors' wait() must return 1.
__global__ void barrier_timeout_wait_kernel(ccoDevComm dc, uint32_t slot, uint64_t timeoutCycles,
                                            int rankToSkip, int* outRc) {
  ccoCoopThread g;
  if (dc.lsaRank == rankToSkip) {
    return;
  }
  ccoLsaBarrierSession<ccoCoopThread> bar(g, &dc, ccoTeamLsa(dc), dc.lsaBarrier, slot);
  bar.arrive(g);
  int rc = bar.wait(g, timeoutCycles);
  outRc[dc.lsaRank] = rc;
}

// Split kernel — same payload check as visibility, but the first barrier of
// each round uses the decoupled arrive()/wait() pair (with local work between)
// instead of fused sync(). Verifies arrive publishes the cookie and wait
// observes all peers before the read-back.
template <typename Coop>
__global__ void barrier_split_kernel(ccoDevComm dc, ccoWindow_t win, size_t off, uint32_t iters,
                                     uint32_t slot, int* errors) {
  Coop g;
  ccoLsaBarrierSession<Coop> bar(g, &dc, ccoTeamLsa(dc), dc.lsaBarrier, slot);

  const int N = dc.lsaSize;
  const int myRank = dc.lsaRank;
  constexpr uint32_t MUL = 1000003u;

  uint32_t* myBuf = static_cast<uint32_t*>(ccoGetLocalPtr(win, off));

  int localErr = 0;
  for (uint32_t e = 1; e <= iters; ++e) {
    if (g.thread_rank() == 0) {
      myBuf[0] = e * MUL + static_cast<uint32_t>(myRank);
    }
    // Decoupled: signal arrival, do unrelated local work, then block on peers.
    bar.arrive(g);
    uint32_t acc = 0;
    for (int s = 0; s < 128; ++s) acc += e + s;
    if (acc == 0xFFFFFFFFu) myBuf[0] ^= acc;  // keep the loop from being elided
    bar.wait(g);

    for (int p = g.thread_rank(); p < N; p += g.size()) {
      uint32_t v = static_cast<uint32_t*>(ccoGetLsaPeerPtr(win, p, off))[0];
      if (v != e * MUL + static_cast<uint32_t>(p)) localErr |= 1;
    }
    bar.sync(g);  // separator before the next overwrite
  }

  if (localErr != 0) {
    atomicAdd(errors, 1);
  }
}

// Multi-slot kernel — launched with TWO blocks; each block drives its OWN
// barrier slot concurrently (block 0 → slotA, block 1 → slotB), with its own
// payload offset. The two barriers run in parallel rather than interleaved by
// one block, so if per-index inbox addressing (index*lsaSize in ucInbox) is
// wrong the concurrent slots stomp each other and the payload check fails.
__global__ void barrier_multislot_kernel(ccoDevComm dc, ccoWindow_t win, uint32_t slotA,
                                         uint32_t slotB, uint32_t iters, int* errors) {
  ccoCoopBlock g;

  // Each block picks its own slot / payload offset / cookie multiplier.
  const uint32_t slot = (blockIdx.x == 0) ? slotA : slotB;
  const size_t off = blockIdx.x * sizeof(uint32_t);
  const uint32_t MUL = (blockIdx.x == 0) ? 1000003u : 2000029u;

  ccoLsaBarrierSession<ccoCoopBlock> bar(g, &dc, ccoTeamLsa(dc), dc.lsaBarrier, slot);

  const int N = dc.lsaSize;
  const int myRank = dc.lsaRank;
  uint32_t* myBuf = static_cast<uint32_t*>(ccoGetLocalPtr(win, off));

  int localErr = 0;
  for (uint32_t e = 1; e <= iters; ++e) {
    if (g.thread_rank() == 0) {
      myBuf[0] = e * MUL + static_cast<uint32_t>(myRank);
    }
    bar.sync(g);
    for (int p = g.thread_rank(); p < N; p += g.size()) {
      uint32_t v = static_cast<uint32_t*>(ccoGetLsaPeerPtr(win, p, off))[0];
      if (v != e * MUL + static_cast<uint32_t>(p)) localErr |= 1;
    }
    bar.sync(g);  // separator before the next overwrite
  }

  if (localErr != 0) {
    atomicAdd(errors, 1);
  }
}

// Preset kernel — seeds a barrier slot's persisted epoch AND its inbox slots to
// `val` on the LOCAL rank, mirroring ccoLsaBarrierSession's ctor / ucInbox()
// addressing. Parks the epoch just below 2^32 so the next run crosses the
// wraparound boundary; presetting the inbox to the same value keeps the first
// compare consistent (no false-match against a stale 0).
//
// Unicast-only state layout (cco_lsa_types.hpp):
//   [0, nB)        unicast epoch   <- ctor restores state[slot]
//   [nB, ...)      ucInbox[slot][peer] = state[nB + slot*lsaSize + peer]
__global__ void barrier_preset_kernel(ccoDevComm dc, uint32_t slot, uint32_t val) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  const auto& rw = dc.resourceWindow_inlined;
  char* base = rw.winBase + ((uint64_t)dc.lsaRank * rw.stride4G << 32);
  uint32_t* state = reinterpret_cast<uint32_t*>(base + dc.lsaBarrier.bufOffset);
  const int nB = dc.lsaBarrier.nBarriers;
  const int N = dc.lsaSize;
  state[slot] = val;  // unicast epoch slot restored by ctor
  for (int peer = 0; peer < N; ++peer) {
    state[nB + slot * N + peer] = val;  // ucInbox[slot][peer]
  }
}

// ============================================================================
// UT framework
// ============================================================================

// Shared state passed to every UT function. UTs only need this — they don't
// touch MPI/DevComm setup directly, just consume what main() prepared.
struct UtCtx {
  int rank;       // world rank, used for "rank 0 prints" guards
  ccoComm* comm;  // for ccoBarrierAll between cases
  ccoDevComm
      dcHost;  // host DevComm struct (filled by ccoDevCommCreate); passed by value to kernels
  ccoWindow_t sendWin;
  // Scratch device buffers, reused across cases.
  int* devErrors;  // one int
  int* devRc;      // [lsaSize]
};

using UtFn = int (*)(UtCtx&);

// ============================================================================
// UT cases
// ============================================================================

// Generic visibility runner — used by all three Coop-variant UTs.
template <typename Coop>
static int run_visibility(UtCtx& ctx, const char* name, uint32_t slot, uint32_t iters, dim3 grid,
                          dim3 block) {
  HIP_CHECK(hipMemset(ctx.devErrors, 0, sizeof(int)));
  hipLaunchKernelGGL(barrier_visibility_kernel<Coop>, grid, block, 0, 0, ctx.dcHost, ctx.sendWin,
                     (size_t)0, /*epochBase=*/0u, iters, slot, ctx.devErrors);
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
  constexpr uint32_t kStressIters = 4096;
  constexpr int kLaunches = 4;
  constexpr uint32_t kSlot = 3u;

  int* devCompleted = nullptr;
  HIP_CHECK(hipMalloc(&devCompleted, sizeof(int)));

  bool ok = true;
  for (int k = 0; k < kLaunches; ++k) {
    HIP_CHECK(hipMemset(devCompleted, 0, sizeof(int)));
    hipLaunchKernelGGL(barrier_stress_kernel, dim3(1), dim3(256), 0, 0, ctx.dcHost, kStressIters,
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
  hipLaunchKernelGGL(barrier_timeout_kernel, dim3(1), dim3(1), 0, 0, ctx.dcHost, kSlot,
                     kTimeoutCycles, kRankToSkip, ctx.devRc);
  HIP_CHECK(hipEventRecord(stop, nullptr));

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<int> rcHost(ctx.dcHost.lsaSize);
  HIP_CHECK(
      hipMemcpy(rcHost.data(), ctx.devRc, sizeof(int) * ctx.dcHost.lsaSize, hipMemcpyDeviceToHost));

  bool ok = true;
  if (ctx.dcHost.lsaRank != kRankToSkip && rcHost[ctx.dcHost.lsaRank] != 1) {
    ok = false;
  }
  if (ctx.rank == 0) {
    float elapsedTime;
    HIP_CHECK(hipEventElapsedTime(&elapsedTime, start, stop));
    std::printf(
        "  [timeo ] slot=%u skipRank=%d  expected rc=1 on survivors  %s elapsedTime=%f ms\n", kSlot,
        kRankToSkip, ok ? "PASS" : "FAIL", elapsedTime);
  }
  ccoBarrierAll(ctx.comm);
  return ok ? 0 : 1;
}

// UT 5b — timeout via direct arrive() + wait(coop, timeoutCycles) (slot 10).
// Same contract as ut_timeout but exercises the 2-arg wait() overload directly
// (ut_timeout goes through sync(coop, timeoutCycles)).
static int ut_timeout_wait(UtCtx& ctx) {
  constexpr uint32_t kSlot = 10u;
  constexpr uint64_t kTimeoutCycles = 500ULL * 1000 * 1000;
  constexpr int kRankToSkip = 0;

  if (ctx.dcHost.lsaSize < 2) {
    if (ctx.rank == 0) {
      std::printf("  [twait ] skipped (lsaSize=%d < 2)\n", ctx.dcHost.lsaSize);
    }
    return 0;
  }

  HIP_CHECK(hipMemset(ctx.devRc, 0xFF, sizeof(int) * ctx.dcHost.lsaSize));  // sentinel = -1

  hipLaunchKernelGGL(barrier_timeout_wait_kernel, dim3(1), dim3(1), 0, 0, ctx.dcHost, kSlot,
                     kTimeoutCycles, kRankToSkip, ctx.devRc);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<int> rcHost(ctx.dcHost.lsaSize);
  HIP_CHECK(
      hipMemcpy(rcHost.data(), ctx.devRc, sizeof(int) * ctx.dcHost.lsaSize, hipMemcpyDeviceToHost));

  bool ok = true;
  if (ctx.dcHost.lsaRank != kRankToSkip && rcHost[ctx.dcHost.lsaRank] != 1) {
    ok = false;
  }
  if (ctx.rank == 0) {
    std::printf("  [twait ] slot=%u skipRank=%d  expected rc=1 on survivors (direct wait)  %s\n",
                kSlot, kRankToSkip, ok ? "PASS" : "FAIL");
  }
  ccoBarrierAll(ctx.comm);
  return ok ? 0 : 1;
}

// UT 6 — arrive()/wait() split (slot 5)
static int ut_arrive_wait_split(UtCtx& ctx) {
  constexpr uint32_t kSlot = 5u;
  constexpr uint32_t kIters = 5000u;

  HIP_CHECK(hipMemset(ctx.devErrors, 0, sizeof(int)));
  hipLaunchKernelGGL(barrier_split_kernel<ccoCoopBlock>, dim3(1), dim3(256), 0, 0, ctx.dcHost,
                     ctx.sendWin, (size_t)0, kIters, kSlot, ctx.devErrors);
  HIP_CHECK(hipDeviceSynchronize());

  int hostErr = 0;
  HIP_CHECK(hipMemcpy(&hostErr, ctx.devErrors, sizeof(int), hipMemcpyDeviceToHost));
  if (ctx.rank == 0) {
    std::printf("  [split ] slot=%u iters=%u  errors=%d  %s\n", kSlot, kIters, hostErr,
                hostErr == 0 ? "PASS" : "FAIL");
  }
  ccoBarrierAll(ctx.comm);
  return hostErr == 0 ? 0 : 1;
}

// UT 7 — epoch persistence via cookie continuity (slot 6)
//   Unlike ut_stress (hang-only), this CONTINUES the cookie sequence across
//   launches and never resets the buffer, so a broken ctor/dtor epoch round-
//   trip surfaces as a payload mismatch on the launch boundary, not just a hang.
static int ut_persist_cookie(UtCtx& ctx) {
  constexpr uint32_t kSlot = 6u;
  constexpr uint32_t kItersPerLaunch = 200u;
  constexpr int kLaunches = 4;

  HIP_CHECK(hipMemset(ctx.devErrors, 0, sizeof(int)));
  // NOTE: do NOT reset ctx.sendWin between launches — stale payload is exactly
  // what exposes a false-matched first sync when persistence is broken.
  for (int k = 0; k < kLaunches; ++k) {
    uint32_t epochBase = static_cast<uint32_t>(k) * kItersPerLaunch;
    hipLaunchKernelGGL(barrier_visibility_kernel<ccoCoopBlock>, dim3(1), dim3(256), 0, 0,
                       ctx.dcHost, ctx.sendWin, (size_t)0, epochBase, kItersPerLaunch, kSlot,
                       ctx.devErrors);
    HIP_CHECK(hipDeviceSynchronize());
    ccoBarrierAll(ctx.comm);
  }

  int hostErr = 0;
  HIP_CHECK(hipMemcpy(&hostErr, ctx.devErrors, sizeof(int), hipMemcpyDeviceToHost));
  if (ctx.rank == 0) {
    std::printf("  [persis] slot=%u iters=%u launches=%d  errors=%d  %s\n", kSlot, kItersPerLaunch,
                kLaunches, hostErr, hostErr == 0 ? "PASS" : "FAIL");
  }
  ccoBarrierAll(ctx.comm);
  return hostErr == 0 ? 0 : 1;
}

// UT 8 — multi-slot isolation (slots 7 & 8)
static int ut_multislot(UtCtx& ctx) {
  constexpr uint32_t kSlotA = 7u;
  constexpr uint32_t kSlotB = 8u;
  constexpr uint32_t kIters = 5000u;

  HIP_CHECK(hipMemset(ctx.devErrors, 0, sizeof(int)));
  // Two blocks: block 0 drives slotA, block 1 drives slotB, concurrently.
  hipLaunchKernelGGL(barrier_multislot_kernel, dim3(2), dim3(256), 0, 0, ctx.dcHost, ctx.sendWin,
                     kSlotA, kSlotB, kIters, ctx.devErrors);
  HIP_CHECK(hipDeviceSynchronize());

  int hostErr = 0;
  HIP_CHECK(hipMemcpy(&hostErr, ctx.devErrors, sizeof(int), hipMemcpyDeviceToHost));
  if (ctx.rank == 0) {
    std::printf("  [mslot ] slots=%u,%u iters=%u  errors=%d  %s\n", kSlotA, kSlotB, kIters, hostErr,
                hostErr == 0 ? "PASS" : "FAIL");
  }
  ccoBarrierAll(ctx.comm);
  return hostErr == 0 ? 0 : 1;
}

// UT 9 — epoch wraparound (slot 9)
//   Park epoch + inbox just below 2^32, then sync across the boundary. The
//   rolling less-equal compare must stay correct as epoch+1 wraps to 0.
static int ut_wraparound(UtCtx& ctx) {
  constexpr uint32_t kSlot = 9u;
  constexpr uint32_t kPreset = 0xFFFFFFFEu;  // 2 syncs later epoch wraps past 0
  constexpr uint32_t kIters = 64u;

  // Seed epoch + inbox on every rank, then make sure all presets land before
  // anyone opens a session on this slot.
  hipLaunchKernelGGL(barrier_preset_kernel, dim3(1), dim3(1), 0, 0, ctx.dcHost, kSlot, kPreset);
  HIP_CHECK(hipDeviceSynchronize());
  ccoBarrierAll(ctx.comm);

  HIP_CHECK(hipMemset(ctx.devErrors, 0, sizeof(int)));
  hipLaunchKernelGGL(barrier_visibility_kernel<ccoCoopBlock>, dim3(1), dim3(256), 0, 0, ctx.dcHost,
                     ctx.sendWin, (size_t)0, /*epochBase=*/0u, kIters, kSlot, ctx.devErrors);
  HIP_CHECK(hipDeviceSynchronize());

  int hostErr = 0;
  HIP_CHECK(hipMemcpy(&hostErr, ctx.devErrors, sizeof(int), hipMemcpyDeviceToHost));
  if (ctx.rank == 0) {
    std::printf("  [wrap  ] slot=%u preset=0x%08X iters=%u  errors=%d  %s\n", kSlot, kPreset,
                kIters, hostErr, hostErr == 0 ? "PASS" : "FAIL");
  }
  ccoBarrierAll(ctx.comm);
  return hostErr == 0 ? 0 : 1;
}

// UT 10 — single-rank no-op (slot 0, reused)
//   Only meaningful with lsaSize==1: arrive writes nothing (nranks-1==0) and
//   wait spins zero times, so sync() must be an instant no-op that completes.
static int ut_single_rank(UtCtx& ctx) {
  if (ctx.dcHost.lsaSize != 1) {
    if (ctx.rank == 0) {
      std::printf("  [single] skipped (lsaSize=%d != 1)\n", ctx.dcHost.lsaSize);
    }
    return 0;
  }

  constexpr uint32_t kIters = 1000u;
  int* devCompleted = nullptr;
  HIP_CHECK(hipMalloc(&devCompleted, sizeof(int)));
  HIP_CHECK(hipMemset(devCompleted, 0, sizeof(int)));
  hipLaunchKernelGGL(barrier_stress_kernel, dim3(1), dim3(256), 0, 0, ctx.dcHost, kIters,
                     /*slot=*/0u, devCompleted);
  HIP_CHECK(hipDeviceSynchronize());
  int completed = 0;
  HIP_CHECK(hipMemcpy(&completed, devCompleted, sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(devCompleted));

  bool ok = (completed == 1);
  if (ctx.rank == 0) {
    std::printf("  [single] iters=%u  %s\n", kIters, ok ? "PASS" : "FAIL");
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
      {"block", ut_visibility_block},
      {"warp", ut_visibility_warp},
      {"thread", ut_visibility_thread},
      {"stress", ut_stress},
      {"timeo", ut_timeout},
      {"twait", ut_timeout_wait},
      {"split", ut_arrive_wait_split},
      {"persis", ut_persist_cookie},
      {"mslot", ut_multislot},
      {"wrap", ut_wraparound},
      {"single", ut_single_rank},
  };

  int fails = 0;
  for (const auto& c : kCases) {
    fails += c.fn(ctx);
  }
  if (ctx.rank == 0) {
    std::printf("=== %d/%zu PASS ===\n",
                static_cast<int>(sizeof(kCases) / sizeof(kCases[0])) - fails,
                sizeof(kCases) / sizeof(kCases[0]));
  }
  return fails;
}

// ============================================================================
// Host driver — setup, single call to run_all_tests, teardown.
// ============================================================================

int run_test(int rank, int nranks, mori::application::BootstrapNetwork* bootNet) {
  g_rank = rank;

  // Bind each rank to its own GPU BEFORE ccoCommCreate (which pins
  // allocations to the current device).
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  HIP_CHECK(hipSetDevice(rank % numDevices));

  // ── Phase 1: communicator ──
  ccoComm* comm = nullptr;
  if (ccoCommCreate(bootNet, PER_RANK_VMM_SIZE, &comm) != 0) {
    std::fprintf(stderr, "[rank %d] CommCreate failed\n", rank);
    return 1;
  }

  // ── Phase 2: send window (cookie slots) ──
  // Allocate then register (the same path the GDA tests use) rather than the
  // convenience register-and-alloc overload.
  void* sendBuf = nullptr;
  ccoWindow_t sendWin = nullptr;
  // NOTE: do NOT use assert() for these — tests are built with -DNDEBUG
  // (CMAKE_BUILD_TYPE=Release), which strips assert, so a failed call would
  // silently leave sendBuf=nullptr and surface as a bogus hipMemset error.
  if (ccoMemAlloc(comm, COOKIE_BYTES, &sendBuf) != 0) {
    std::fprintf(stderr, "[rank %d] MemAlloc failed\n", rank);
    return 1;
  }
  HIP_CHECK(hipMemset(sendBuf, 0, COOKIE_BYTES));
  if (ccoWindowRegister(comm, sendBuf, COOKIE_BYTES, &sendWin) != 0) {
    std::fprintf(stderr, "[rank %d] WindowRegister failed\n", rank);
    return 1;
  }

  // ── Phase 3: DevComm with LSA barrier slots (0..10 used by the UTs) ──
  ccoDevCommRequirements reqs = CCO_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.gdaConnectionType = CCO_GDA_CONNECTION_NONE;
  reqs.lsaBarrierCount = 11;
  ccoDevComm dcHost{};
  if (ccoDevCommCreate(comm, &reqs, &dcHost) != 0) {
    std::fprintf(stderr, "[rank %d] DevCommCreate failed\n", rank);
    return 1;
  }
  if (rank == 0) {
    std::printf("=== LSA barrier: world=%d lsa=%d ===\n", dcHost.worldSize, dcHost.lsaSize);
  }

  // ── Scratch buffers reused across UTs ──
  int* devErrors = nullptr;
  int* devRc = nullptr;
  HIP_CHECK(hipMalloc(&devErrors, sizeof(int)));
  HIP_CHECK(hipMalloc(&devRc, sizeof(int) * dcHost.lsaSize));

  ccoBarrierAll(comm);

  // ── Run ALL UTs in one shot ──
  UtCtx ctx{rank, comm, dcHost, sendWin, devErrors, devRc};
  int fails = run_all_tests(ctx);

  // ── Teardown ──
  HIP_CHECK(hipFree(devRc));
  HIP_CHECK(hipFree(devErrors));
  ccoDevCommDestroy(comm, &dcHost);
  ccoWindowDeregister(comm, sendWin);
  ccoMemFree(comm, sendBuf);
  ccoCommDestroy(comm);

  printf("[rank %d] %s\n", rank, fails == 0 ? "PASSED" : "FAILED");
  return fails != 0 ? 1 : 0;
}

int main(int argc, char** argv) {
  return ccoTestMain(argc, argv, "CCO LSA barrier", "/tmp/cco_lsa_barrier_uid", 19882);
}
