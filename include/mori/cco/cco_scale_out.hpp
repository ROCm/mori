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
// cco_scale_out.hpp — CCO scale-out (GDA / cross-node IBGDA RDMA) device layer.
//
// Split out of cco.hpp. This header is the *sole* consumer of the RDMA core, so
// keeping it separate lets host-only and LSA-only (scale-up) translation units
// include just cco.hpp without pulling in the heavy provider RDMA headers.
//
// It is self-contained: it includes cco.hpp for the shared GPU-side types,
// cooperative groups, teams, the LSA barrier session, and the host control-plane
// API, then adds the GDA device layer (ccoGda<PrvdType> + the provider-
// specialized primitive layer in mori::cco::impl). Include THIS header (not
// cco.hpp) when you need GDA.
#pragma once

#include "mori/cco/cco.hpp"

// GDA (RDMA) device layer pulls in the core RDMA device aggregator (provider
// primitives + core::RdmaEndpointDevice/WorkQueueHandle/...). Device-only; no
// application headers — cco's GDA path depends only on core.
#if defined(__HIPCC__) || defined(__CUDACC__)
#include "mori/core/transport/rdma/rdma_device.hpp"
#endif

namespace mori {
namespace cco {

#if defined(__HIPCC__) || defined(__CUDACC__)

/* ════════════════════════════════════════════════════════════════════════════
 *  5. GDA (RDMA) device layer.
 *
 *  Cross-node one-sided RDMA (put/get/signal/counter) over IBGDA QPs, plus
 *  the provider-specialized primitive layer it builds on. Lives directly in
 *  mori::cco (like the LSA layer above) — the ccoGda* prefix is the namespace;
 *  device-only (uses RDMA core + device builtins).
 * ════════════════════════════════════════════════════════════════════════════ */

// ── Compile-time GDA provider dispatch (per-NIC build, like shmem) ───────────
// Provider is fixed at build time from the NIC auto-detected by
// cmake/MoriDetectDevice.cmake (MORI_DEVICE_NIC_*; the python/mori/jit path
// mirrors it), so only the one ccoGda<P> specialization is built.
#if defined(MORI_DEVICE_NIC_BNXT)
#define CCO_GDA_BUILD_PROVIDER ::mori::core::ProviderType::BNXT
#elif defined(MORI_DEVICE_NIC_IONIC)
#define CCO_GDA_BUILD_PROVIDER ::mori::core::ProviderType::PSD
#else
#define CCO_GDA_BUILD_PROVIDER ::mori::core::ProviderType::MLX5  // default
#endif

// Launch a GDA kernel against the build's provider; `P` is a constexpr provider:
//   CCO_GDA_DISPATCH(MyKernel<P, float><<<g, b, 0, s>>>(win, win, n, devComm));
#define CCO_GDA_DISPATCH(...)                  \
  do {                                         \
    constexpr auto P = CCO_GDA_BUILD_PROVIDER; \
    __VA_ARGS__;                               \
  } while (0)

// ── low-level type aliases / enums + ccoGda<PrvdType> class declaration ──
// Window handles use the shared ccoWindow_t (= ccoWindowDevice*) declared above.
typedef struct {
  int qpIdx;
  uint64_t postIdx;
} ccoGdaRequest_t;

typedef uint32_t ccoGdaSignal_t;
typedef uint32_t ccoGdaCounter_t;

enum ccoGdaThreadMode : uint32_t {
  ccoGdaThreadIndependent = 0,
  ccoGdaThreadAggregate = 1,
};

enum ccoGdaOptFlags {
  ccoGdaOptFlagsDefault = 0,
  ccoGdaOptFlagsMaySkipCreditCheck = (1 << 0),
  ccoGdaOptFlagsAggregateRequests = (1 << 1),
};

typedef enum ccoGdaSignalOp_t {
  ccoGdaSignalInc = 0,
  ccoGdaSignalAdd,
} ccoGdaSignalOp_t;

struct ccoGda_NoSignal {};

struct ccoGda_SignalInc {
  ccoGdaSignal_t signalId;
  __device__ inline ccoGda_SignalInc(ccoGdaSignal_t id) : signalId(id) {}
};

struct ccoGda_SignalAdd {
  ccoGdaSignal_t signalId;
  uint64_t value;
  __device__ inline ccoGda_SignalAdd(ccoGdaSignal_t id, uint64_t val) : signalId(id), value(val) {}
};

struct ccoGda_CounterInc {
  ccoGdaCounter_t counterId;
  __device__ inline ccoGda_CounterInc(ccoGdaCounter_t id) : counterId(id) {}
};

struct ccoGdaCtx {
  int rank;
  int worldSize;
  void* handle;
  int contextId;
};

template <core::ProviderType PrvdType>
struct ccoGda {
  ccoDevComm const& comm;
  int rank;    // my index in the GDA team [0, nRanks)
  int nRanks;  // GDA team size, derived from gdaConnType at construction
  uint32_t contextId;
  void* _gdaHandle;

  // constructor
  __device__ inline ccoGda(ccoDevComm const&, int contextIndex);

  template <ccoTeamMode TeamMode>
  __device__ inline int resolveWorldPeer(int peer) const;

  // put: rdma write with optional remote signal.
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD, ccoGdaThreadMode ThreadMode = ccoGdaThreadIndependent,
            typename RemoteAction = ccoGda_NoSignal, typename Coop = ccoCoopThread>
  __device__ inline void put(int peer, ccoWindow_t dstWin, size_t dstOffset, ccoWindow_t srcWin,
                             size_t srcOffset, size_t bytes,
                             RemoteAction remoteAction = ccoGda_NoSignal{}, Coop coop = Coop{},
                             uint32_t optFlags = ccoGdaOptFlagsDefault);

  // putValue: write an immediate value (≤8 bytes) with optional remote signal.
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD, ccoGdaThreadMode ThreadMode = ccoGdaThreadIndependent,
            typename T, typename RemoteAction = ccoGda_NoSignal,
            typename Coop = ccoCoopThread>
  __device__ inline void putValue(int peer, ccoWindow_t dstWin, size_t dstOffset, T value,
                                  RemoteAction remoteAction = ccoGda_NoSignal{}, Coop coop = Coop{},
                                  uint32_t optFlags = ccoGdaOptFlagsDefault);

  // get: rdma read — pull peer's window content into our local window.
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD, ccoGdaThreadMode ThreadMode = ccoGdaThreadIndependent,
            typename Coop = ccoCoopThread>
  __device__ inline void get(int peer, ccoWindow_t remoteWin, size_t remoteOffset,
                             ccoWindow_t localWin, size_t localOffset, size_t bytes,
                             Coop coop = Coop{}, uint32_t optFlags = ccoGdaOptFlagsDefault);

  // ── signal ──────────────────────────────────────────────────────────────

  // signal: send a signal-only message to peer (no data payload).
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD, typename RemoteAction = ccoGda_NoSignal,
            typename Coop = ccoCoopThread>
  __device__ inline void signal(int peer, RemoteAction remoteAction, Coop coop = Coop{});

  // readSignal: read the local value of one signal slot.
  __device__ inline uint64_t readSignal(ccoGdaSignal_t signalId, int bits = 64);

  // waitSignal: block until the local signal slot reaches `least`.
  template <typename Coop = ccoCoopThread>
  __device__ inline void waitSignal(ccoGdaSignal_t signalId, uint64_t least, Coop coop = Coop{},
                                    int bits = 64);

  // resetSignal: zero one local signal slot.
  __device__ inline void resetSignal(ccoGdaSignal_t signalId);

  // ── counter ─────────────────────────────────────────────────────────────
  // counter: poll CQ for all GDA-team peers (quietUntil), then increment
  // counterBuf[localAction.counterId]. Requires ≥warp coop.
  template <typename LocalAction, typename Coop = ccoCoopWarp>
  __device__ inline void counter(LocalAction localAction, Coop coop = Coop{});

  // readCounter: read the local value of one counter slot.
  __device__ inline uint64_t readCounter(ccoGdaCounter_t counterId, int bits = 56);

  // waitCounter: block until the local counter slot reaches `least`.
  template <typename Coop = ccoCoopThread>
  __device__ inline void waitCounter(ccoGdaCounter_t counterId, uint64_t least, Coop coop = Coop{},
                                     int bits = 56);

  // resetCounter: zero one local counter slot.
  __device__ inline void resetCounter(ccoGdaCounter_t counterId);

  // ── completion ──────────────────────────────────────────────────────────

  // flush = flushAsync + wait per peer.
  // flushAsync rings the doorbell if any WQEs are pending (skips if already
  // rung), then wait polls CQ until all submitted WQEs complete.

  // flush: ring doorbell + poll CQ for every peer.
  // peers are distributed across the Coop group (default: warp).
  // all threads in the group must call flush together.
  template <typename Coop = ccoCoopWarp>
  __device__ inline void flush(Coop coop = Coop{});

  // flush(peer): poll CQ for a single peer until its submitted WQEs complete.
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD, typename Coop = ccoCoopWarp>
  __device__ inline void flush(int peer, Coop coop = Coop{});

  // flushAsync: ring doorbell for peer and return a request handle that
  // wait() can later be used to wait on individually.
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD, typename Coop = ccoCoopThread>
  __device__ inline void flushAsync(int peer, ccoGdaRequest_t* outRequest, Coop coop = Coop{});

  // wait: block on a request handle previously returned by flushAsync.
  template <typename Coop = ccoCoopWarp>
  __device__ inline void wait(ccoGdaRequest_t& request, Coop coop = Coop{});
};

// ── GDA barrier session ──────────────────────────────────────────────────
//
// Signal-based cross-node barrier. Each rank sends a signal (NIC atomic-add)
// to every peer, then polls for the reciprocal signals. Uses the signal
// slots reserved by ccoGdaBarrierHandle (allocated at DevComm creation via
// railGdaBarrierCount / barrierCount in ccoDevCommRequirements).
//
// Usage:
//   ccoGdaBarrierSession<PrvdType, ccoCoopBlock> session(ccoCoopBlock{}, gda,
//       devComm.railGdaBarrier, /*index=*/0);
//   session.sync(ccoCoopBlock{});
//
// Or one-shot:
//   ccoGdaBarrier(ccoCoopBlock{}, gda, devComm.railGdaBarrier, 0);

template <core::ProviderType PrvdType, typename Coop>
struct ccoGdaBarrierSession {
  Coop coop;
  ccoGda<PrvdType>& gda;
  ccoGdaBarrierHandle handle;
  uint32_t index;

  __device__ inline ccoGdaBarrierSession(Coop coop, ccoGda<PrvdType>& gda,
                                         ccoGdaBarrierHandle handle, uint32_t index);
  __device__ inline ~ccoGdaBarrierSession() {}

  ccoGdaBarrierSession(ccoGdaBarrierSession const&) = delete;

  __device__ inline void sync(Coop);
};

template <core::ProviderType PrvdType, typename Coop>
__device__ inline void ccoGdaBarrier(Coop coop, ccoGda<PrvdType>& gda, ccoGdaBarrierHandle handle,
                                     uint32_t index);

// ── provider-specialized primitive layer (putImpl/getImpl/...) ──
//
// Internal implementation. Device kernels use the public ccoGda<> facade
// (declared above) and the cco:: types — never these directly. Kept in a
// dedicated `impl` namespace so the public surface stays clean and these
// helpers don't leak into ADL or autocomplete.
namespace impl {

// Poll the CQ until wq.doneIdx reaches targetIdx. Post-#395 collapsed-CQ model:
// reconstruct the completed WQE count directly from the CQE counter (no
// outstandingWqe[] table). Mirrors shmem's *CollapsedCqDrain / PSD quiet, but
// exits at the caller's targetIdx instead of a dbTouchIdx/postIdx snapshot.
template <core::ProviderType PrvdType>
__device__ inline static void quietUntil(core::RdmaEndpointDevice* ep, uint32_t targetIdx) {
  core::WorkQueueHandle* wq = &ep->wqHandle;
  core::CompletionQueueHandle* cq = &ep->cqHandle;

  if constexpr (PrvdType == core::ProviderType::PSD) {
    constexpr uint32_t PENDING_WORK_MASK = 0x800000;
#ifdef IONIC_CCQE
    // CCQE: cqeNum==1, NIC overwrites CQE[0] with latest MSN.
    volatile ionic_v1_cqe* cqe = reinterpret_cast<volatile ionic_v1_cqe*>(cq->cqAddr);
    while ((wq->doneIdx - targetIdx) & PENDING_WORK_MASK) {
      uint32_t msn = BE32TOH(*(volatile uint32_t*)(&cqe->send.msg_msn));
      asm volatile("" ::: "memory");
      if (!((msn - targetIdx) & PENDING_WORK_MASK)) {
        wq->doneIdx = msn;
      }
    }
#else
    // Non-CCQE: warp-parallel poll with color bit alternation.
    // Mirrors shmem ShmemQuietThreadKernelPsdImpl.
    const uint64_t activeMask = core::GetActiveLaneMask();
    const uint32_t myLogicalLaneId = core::GetActiveLaneNum(activeMask);
    const int myLaneId = core::WarpLaneId();
    constexpr uint32_t MAX_GREED = 10;
    constexpr uint32_t CQ_DOORBELL_GRACE = 100;
    uint32_t wqeCounter;

    while ((wq->doneIdx - targetIdx) & PENDING_WORK_MASK) {
      if (!core::spin_lock_try_acquire_shared(&cq->pollCqLock, activeMask)) continue;
      uint32_t greedRemaining = MAX_GREED;
      while ((wq->doneIdx - targetIdx) & PENDING_WORK_MASK) {
        const uint64_t oldDoneIdx = wq->doneIdx;
        const uint32_t curConsIdx = cq->cq_consumer;
        uint32_t myCqPos = curConsIdx + myLogicalLaneId;
        const int opcode =
            core::PollCq<core::ProviderType::PSD>(cq->cqAddr, cq->cqeNum, &myCqPos, &wqeCounter);
        if (opcode > 0) {
          MORI_PRINTF("quietUntil[PSD]: poll err %d\n", opcode);
          assert(false);
        }
        asm volatile("" ::: "memory");
        const uint64_t successMask = __ballot(opcode == 0);
        const int highestLane = core::GetLastActiveLaneID(successMask);
        if (highestLane == -1) continue;
        if (myLaneId == highestLane) {
          cq->cq_consumer = myCqPos + 1;
          if (((cq->cq_consumer - cq->cq_dbpos) & (cq->cqeNum - 1)) >= CQ_DOORBELL_GRACE) {
            cq->cq_dbpos = cq->cq_consumer;
            core::UpdateCqDbrRecord<core::ProviderType::PSD>(*cq, myCqPos + 1);
          }
          wq->doneIdx = wqeCounter;
        }
        if (!((wq->doneIdx - targetIdx) & PENDING_WORK_MASK)) {
          if (wq->doneIdx == oldDoneIdx) break;
          if (greedRemaining == 0) break;
          --greedRemaining;
        }
      }
      core::spin_lock_release_shared(&cq->pollCqLock, activeMask);
      break;
    }
#endif
  } else if constexpr (PrvdType == core::ProviderType::MLX5) {
    // MLX5: collapsed CQ — read CQE[0] (volatile), reconstruct the 16-bit
    // wqe_counter against doneIdx, advance via max. Mirrors shmem Mlx5CollapsedCqDrain.
    auto done = [&]() {
      return (int32_t)(__hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) -
                       targetIdx) >= 0;
    };
    volatile core::Mlx5Cqe64* cqe = reinterpret_cast<volatile core::Mlx5Cqe64*>(cq->cqAddr);
    __threadfence();
    while (!done()) {
      uint32_t cons = __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      uint16_t wqeCounter = BE16TOH(cqe->wqe_counter);
      uint8_t opcode =
          (reinterpret_cast<volatile uint8_t*>(cq->cqAddr)[sizeof(core::Mlx5Cqe64) - 1]) >> 4;
      if (opcode == core::MORI_MLX5_CQE_REQ_ERR || opcode == core::MORI_MLX5_CQE_RESP_ERR) {
        auto error = core::Mlx5HandleErrorCqe(reinterpret_cast<core::Mlx5ErrCqe*>(cq->cqAddr));
        MORI_PRINTF("quietUntil[MLX5]: CQE error %s\n", core::WcStatusString(error));
        assert(false);
        break;
      }
      uint16_t comp16 = static_cast<uint16_t>(wqeCounter + 1);
      uint32_t completed = (cons & ~0xffffu) | comp16;
      if (completed < cons) completed += 0x10000u;
      __hip_atomic_fetch_max(&wq->doneIdx, completed, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      asm volatile("" ::: "memory");
    }
    __threadfence();
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    // BNXT: collapsed CQ (cqeNum==1) — single poller (pollCqLock); others spin
    // re-reading doneIdx (the holder advances it). Reconstruct the completed count
    // from CQE con_indx against dbTouchIdx, advance doneIdx via max. Non-blocking
    // PollCqOnce (cco flow-control may wait on slots not yet doorbelled, so a
    // blocking poll would deadlock). Mirrors shmem BnxtCollapsedCqDrain.
    const uint32_t mask = wq->sqWqeNum - 1;  // sqWqeNum is a power of two
    auto done = [&]() {
      return (int32_t)(__hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) -
                       targetIdx) >= 0;
    };
    while (!done()) {
      if (!core::AcquireLockOnce(&cq->pollCqLock)) continue;
      __threadfence();
      while (!done()) {
        uint32_t consIdxIgnored = 0;  // cqeNum==1 always reads CQE[0]
        uint32_t wqeCounter = 0;
        int opcode = core::PollCqOnce<core::ProviderType::BNXT>(cq->cqAddr, cq->cqeNum,
                                                                consIdxIgnored, &wqeCounter);
        if (opcode < 0) continue;  // no new completion yet
        if (opcode != BNXT_RE_REQ_ST_OK) {
          MORI_PRINTF("quietUntil[BNXT]: CQE error opcode=%d\n", opcode);
          assert(false);
          break;
        }
        // Largest V <= dbTouchIdx with V % sqWqeNum == con_indx % sqWqeNum.
        uint32_t dbTouch =
            __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        uint32_t completed = (dbTouch & ~mask) | (wqeCounter & mask);
        if (completed > dbTouch) completed -= wq->sqWqeNum;
        __hip_atomic_fetch_max(&wq->doneIdx, completed, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      __threadfence();
      core::ReleaseLock(&cq->pollCqLock);
    }
  }
}

// BNXT: exclusive prefix-sum of psnCnt over active lanes. Returns this lane's
// prefix; *outTotal = warp total. All active lanes must call together.
__device__ inline static uint32_t warpActivePsnPrefix(uint32_t psnCnt, uint64_t activemask,
                                                      uint32_t* outTotal) {
  const uint32_t myPhys = static_cast<uint32_t>(__lane_id());
  uint32_t excl = 0, total = 0;
  uint64_t m = activemask;
  while (m) {
    int l = __ffsll(static_cast<unsigned long long>(m)) - 1;
    uint32_t v = __shfl(psnCnt, l);
    total += v;
    if (static_cast<uint32_t>(l) < myPhys) excl += v;
    m &= m - 1;
  }
  *outTotal = total;
  return excl;
}

// BNXT warp-aggregate PSN: each active lane contributes dataPsnCnt data packets,
// the leader optionally one signal packet. The leader advances wq->msnPack once by
// the warp totals; returns this lane's data-PSN base and the signal PSN (outSignalPsn).
// BNXT PSN advances by PACKET count, not WQE count.
__device__ inline static uint32_t warpAggregateBnxtPsn(core::WorkQueueHandle* wq,
                                                       uint32_t dataPsnCnt, bool hasSignalPacket,
                                                       uint32_t totalWqes, uint64_t activemask,
                                                       int leaderLane, bool isLeader,
                                                       uint32_t* outSignalPsn) {
  uint32_t warpDataPsnTotal = 0;
  uint32_t myExcl = warpActivePsnPrefix(dataPsnCnt, activemask, &warpDataPsnTotal);
  uint32_t warpTotalPsn = warpDataPsnTotal + (hasSignalPacket ? 1u : 0u);
  uint32_t warpPsnBase = 0;
  if (isLeader) {
    uint32_t slotIgnored = 0;
    core::atomic_add_packed_msn_and_psn(&wq->msnPack, totalWqes, warpTotalPsn, &slotIgnored,
                                        &warpPsnBase);
  }
  warpPsnBase = __shfl(warpPsnBase, leaderLane);
  if (outSignalPsn) *outSignalPsn = warpPsnBase + warpDataPsnTotal;
  return warpPsnBase + myExcl;
}

// Reserve numWqesNeeded SQ slots (per-lane) and wait for SQ space. BNXT also
// reserves lanePsnCnt packet PSNs on wq->msnPack and returns the base via
// outPsnBase (advances by PACKET count); lanePsnCnt/outPsnBase ignored elsewhere.
template <core::ProviderType PrvdType>
__device__ inline static uint32_t reserveWqeSlots(core::RdmaEndpointDevice* ep,
                                                  uint32_t numWqesNeeded, uint32_t lanePsnCnt = 0,
                                                  uint32_t* outPsnBase = nullptr) {
  core::WorkQueueHandle* wq = &ep->wqHandle;

  uint32_t curPostIdx = atomicAdd(&wq->postIdx, numWqesNeeded);
  if constexpr (PrvdType == core::ProviderType::BNXT) {
    uint32_t slotIgnored = 0;
    uint32_t psnBase = 0;
    core::atomic_add_packed_msn_and_psn(&wq->msnPack, numWqesNeeded, lanePsnCnt, &slotIgnored,
                                        &psnBase);
    if (outPsnBase) *outPsnBase = psnBase;
  }
  while (true) {
    uint64_t dbTouched =
        __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t dbDone = __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t numActiveSqEntries = dbTouched - dbDone;
    uint64_t numFreeEntries = wq->sqWqeNum - numActiveSqEntries;
    uint64_t entriesUntilMine = curPostIdx + numWqesNeeded - dbTouched;
    if (numFreeEntries > entriesUntilMine) {
      break;
    }
    if constexpr (PrvdType == core::ProviderType::BNXT) {
      // BNXT: drain to the doorbelled snapshot, not our un-doorbelled
      // reservation — else we'd wait on WQEs that may never ring (self-deadlock).
      quietUntil<PrvdType>(ep, static_cast<uint32_t>(dbTouched));
    } else {
      quietUntil<PrvdType>(ep, curPostIdx);
    }
  }
  return curPostIdx;
}

// Warp-aggregate flow control: wait until the SQ has room for [base, base+totalWqes).
// BNXT drains to the doorbelled snapshot (avoids self-deadlock); others drain to
// the reservation.
template <core::ProviderType PrvdType>
__device__ inline static void waitSqSpace(core::RdmaEndpointDevice* ep, uint32_t base,
                                          uint32_t totalWqes) {
  core::WorkQueueHandle* wq = &ep->wqHandle;
  while (true) {
    uint64_t dbTouched =
        __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t dbDone = __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t numActiveSqEntries = dbTouched - dbDone;
    uint64_t numFreeEntries = wq->sqWqeNum - numActiveSqEntries;
    uint64_t entriesUntilBatchLast = base + totalWqes - dbTouched;
    if (numFreeEntries > entriesUntilBatchLast) break;
    if constexpr (PrvdType == core::ProviderType::BNXT) {
      quietUntil<PrvdType>(ep, static_cast<uint32_t>(dbTouched));
    } else {
      quietUntil<PrvdType>(ep, base);
    }
  }
}

// PSD/BNXT: walk the active lane mask, ringing one lane at a time. These providers
// share one dbrAddr across QPs (Ionic per ibv_context; BNXT per UAR page), so lanes
// ringing distinct QPs in one SIMT store would coalesce — only one doorbell survives
// and the rest hang. Serializing per lane avoids that; MLX5 has per-QP dbrAddr.
// No __syncwarp: wavefronts are lock-step (predication already orders the stores)
// and a __syncwarp would deadlock on divergent entry.
template <core::ProviderType PrvdType>
__device__ inline static void ringDoorbellWalk(core::WorkQueueHandle* wq, uint32_t dbrRecVal,
                                               uint64_t dbrVal) {
  if constexpr (PrvdType == core::ProviderType::BNXT) {
    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, dbrRecVal);
    __threadfence_system();
  }
  uint64_t mask = core::GetActiveLaneMask();
  while (mask) {
    int lane = __ffsll(static_cast<unsigned long long>(mask)) - 1;
    if (__lane_id() == lane) {
      core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
    }
    mask &= ~(1ull << lane);
  }
}

// Wait for this QP's doorbell turn (preserve per-QP ordering), then ring.
// LeaderOnly=true: caller guarantees one active lane rings (per-peer group leader)
// → single store. LeaderOnly=false: multiple lanes may ring shared-dbrAddr QPs →
// serialize via ringDoorbellWalk to avoid coalescing.
template <core::ProviderType PrvdType, bool LeaderOnly = false>
__device__ inline static void ringDoorbellOrdered(core::RdmaEndpointDevice* ep, uint32_t myPostIdx,
                                                  uint32_t numWqes, uint64_t dbrVal) {
  core::WorkQueueHandle* wq = &ep->wqHandle;
  core::CompletionQueueHandle* cq = &ep->cqHandle;

  // Wait for my turn to ring doorbell (preserve ordering)
  while (true) {
    uint64_t dbTouched =
        __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (dbTouched == myPostIdx) {
      break;
    }
  }

  // Ring doorbell - provider-specific sequence
  __threadfence_system();

  if constexpr (LeaderOnly) {
    // Single active lane: MLX5/BNXT update the DBR record first, then one store.
    if constexpr (PrvdType != core::ProviderType::PSD) {
      core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, myPostIdx + numWqes);
      __threadfence_system();
    }
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
  } else if constexpr (PrvdType == core::ProviderType::MLX5) {
    // MLX5: per-QP dbrAddr (no coalescing across lanes); update DBR record + ring.
    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, myPostIdx + numWqes);
    __threadfence_system();
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
  } else {
    // PSD/BNXT: lanes may share a dbrAddr → serialize per lane.
    ringDoorbellWalk<PrvdType>(wq, myPostIdx + numWqes, dbrVal);
  }

  __threadfence_system();

  // Update bookkeeping
  __hip_atomic_fetch_add(&cq->needConsIdx, numWqes, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_store(&wq->dbTouchIdx, myPostIdx + numWqes, __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_AGENT);
}

// Helper: calculate number of WQEs needed for atomic operation
template <core::ProviderType PrvdType>
__device__ inline static uint32_t getAtomicWqeCount(core::atomicType amo_op, uint32_t bytes) {
  if constexpr (PrvdType == core::ProviderType::MLX5) {
    // MLX5: some extended atomic ops need 2 WQEs (64-bit masked CAS)
    return core::get_num_wqes_in_atomic(amo_op, bytes);
  } else {
    // PSD/BNXT: always 1 WQE per atomic
    return 1;
  }
}

// Construct provider-correct dbrVal from already-posted WQE state
template <core::ProviderType PrvdType>
__device__ inline static uint64_t buildFlushDbrVal(core::WorkQueueHandle* wq, uint32_t postIdx,
                                                   uint32_t qpn) {
  // postIdx is the next-free slot; the last posted WQE is at postIdx-1
  uint32_t lastWqeIdx = (postIdx - 1) & (wq->sqWqeNum - 1);

  if constexpr (PrvdType == core::ProviderType::PSD) {
    return wq->sq_dbval | (postIdx & (wq->sqWqeNum - 1));
  } else if constexpr (PrvdType == core::ProviderType::MLX5) {
    // Read back ctrl seg first qword from SQ buffer
    uintptr_t wqeAddr =
        reinterpret_cast<uintptr_t>(wq->sqAddr) + (lastWqeIdx << core::MORI_MLX5_SEND_WQE_SHIFT);
    return *reinterpret_cast<volatile uint64_t*>(wqeAddr);
  } else {
    // BNXT: reconstruct db header
    uint8_t flags = (postIdx >> (__ffs(wq->sqWqeNum) - 1)) & 0x1;
    uint32_t epoch = (flags & BNXT_RE_FLAG_EPOCH_TAIL_MASK) << BNXT_RE_DB_EPOCH_TAIL_SHIFT;
    return core::bnxt_re_init_db_hdr(
        ((postIdx & (wq->sqWqeNum - 1)) * BNXT_RE_NUM_SLOT_PER_WQE) | epoch, 0, qpn,
        BNXT_RE_QUE_TYPE_SQ);
  }
}

// putImpl - post one warp-aggregated put for the active lanes (all targeting this
// ep/qpn; the facade groups by peer). Each lane posts its data WQE into a contiguous
// reservation; the leader posts the shared signal WQE and rings one doorbell.
template <core::ProviderType PrvdType>
__device__ inline static void putImpl(
    // Hardware resources (already selected endpoint)
    core::RdmaEndpointDevice* ep, uint32_t qpn,

    // Data transfer parameters (already parsed addresses and keys)
    uintptr_t localAddr, uint32_t localKey,    // local buffer
    uintptr_t remoteAddr, uint32_t remoteKey,  // remote buffer
    size_t bytes,

    // Signal parameters (already parsed)
    bool hasSignal, uintptr_t signalRemoteAddr, uint32_t signalRemoteKey, ccoGdaSignalOp_t signalOp,
    uint64_t signalOpArg,

    // Optimization flags
    uint32_t optFlags = ccoGdaOptFlagsDefault) {
  core::WorkQueueHandle* wq = &ep->wqHandle;
  uint32_t signalWqes =
      hasSignal ? getAtomicWqeCount<PrvdType>(core::AMO_FETCH_ADD, sizeof(uint64_t)) : 0;

  uint64_t activemask = core::GetActiveLaneMask();
  int leaderLane = core::GetLastActiveLaneID(activemask);
  uint32_t numActiveLanes = core::GetActiveLaneCount(activemask);
  uint32_t myLogicalLaneId = core::GetActiveLaneNum(activemask);
  bool isLeader = (myLogicalLaneId == numActiveLanes - 1);
  uint32_t totalWqes = numActiveLanes + signalWqes;

  uint32_t base = 0;
  if (isLeader) base = atomicAdd(&wq->postIdx, totalWqes);
  base = __shfl(base, leaderLane);
  uint32_t mySlot = base + myLogicalLaneId;
  uint32_t signalSlot = base + numActiveLanes;
  uintptr_t atomicLaddr = reinterpret_cast<uintptr_t>(ep->atomicIbuf.addr);
  uint32_t atomicLkey = ep->atomicIbuf.lkey;

  if constexpr (PrvdType == core::ProviderType::BNXT) {
    // Reserve per-packet PSNs before the SQ-space wait so PSN order matches slot
    // order across concurrent warps, then post with packet PSN.
    uint32_t dataPsnCnt = (bytes == 0) ? 1 : (bytes + wq->mtuSize - 1) / wq->mtuSize;
    uint32_t sigPsn = 0;
    uint32_t dataPsn = warpAggregateBnxtPsn(wq, dataPsnCnt, hasSignal, totalWqes, activemask,
                                            leaderLane, isLeader, &sigPsn);
    waitSqSpace<PrvdType>(ep, base, totalWqes);
    uint64_t dbrVal = core::PostWrite<PrvdType>(*wq, mySlot, mySlot, dataPsn, true /*cqeSignal*/,
                                                qpn, localAddr, localKey, remoteAddr, remoteKey,
                                                bytes);
    __threadfence();
    if (isLeader) {
      if (hasSignal) {
        dbrVal = core::PostAtomic<PrvdType, uint64_t>(
            *wq, signalSlot, signalSlot, sigPsn, true /*cqeSignal*/, qpn, atomicLaddr, atomicLkey,
            signalRemoteAddr, signalRemoteKey, signalOpArg, 0 /*compare*/, core::AMO_FETCH_ADD);
      }
      if (!(optFlags & ccoGdaOptFlagsAggregateRequests))
        ringDoorbellOrdered<PrvdType, /*LeaderOnly=*/true>(ep, base, totalWqes, dbrVal);
    }
  } else {
    // MLX5/PSD: the WQE slot index doubles as the PSN.
    waitSqSpace<PrvdType>(ep, base, totalWqes);
    uint64_t dbrVal = core::PostWrite<PrvdType>(*wq, mySlot, mySlot, mySlot, true /*cqeSignal*/, qpn,
                                                localAddr, localKey, remoteAddr, remoteKey, bytes);
    __threadfence();
    if (isLeader) {
      if (hasSignal) {
        dbrVal = core::PostAtomic<PrvdType, uint64_t>(
            *wq, signalSlot, signalSlot, signalSlot, true /*cqeSignal*/, qpn, atomicLaddr,
            atomicLkey, signalRemoteAddr, signalRemoteKey, signalOpArg, 0 /*compare*/,
            core::AMO_FETCH_ADD);
      }
      if (!(optFlags & ccoGdaOptFlagsAggregateRequests))
        ringDoorbellOrdered<PrvdType, /*LeaderOnly=*/true>(ep, base, totalWqes, dbrVal);
    }
  }
}

// putValueImpl - Inline write for small values. One group per call (the facade
// groups lanes by peer); same warp-aggregate posting as putImpl.
template <core::ProviderType PrvdType, typename T>
__device__ inline static void putValueImpl(core::RdmaEndpointDevice* ep, uint32_t qpn,
                                           uintptr_t remoteAddr, uint32_t remoteKey, T value,
                                           bool hasSignal, uintptr_t signalRemoteAddr,
                                           uint32_t signalRemoteKey, ccoGdaSignalOp_t signalOp,
                                           uint64_t signalOpArg,
                                           uint32_t optFlags = ccoGdaOptFlagsDefault) {
  static_assert(sizeof(T) <= 8, "putValue only supports types <= 8 bytes");

  core::WorkQueueHandle* wq = &ep->wqHandle;
  uint32_t signalWqes =
      hasSignal ? getAtomicWqeCount<PrvdType>(core::AMO_FETCH_ADD, sizeof(uint64_t)) : 0;

  uint64_t activemask = core::GetActiveLaneMask();
  int leaderLane = core::GetLastActiveLaneID(activemask);
  uint32_t numActiveLanes = core::GetActiveLaneCount(activemask);
  uint32_t myLogicalLaneId = core::GetActiveLaneNum(activemask);
  bool isLeader = (myLogicalLaneId == numActiveLanes - 1);
  uint32_t totalWqes = numActiveLanes + signalWqes;

  uint32_t base = 0;
  if (isLeader) base = atomicAdd(&wq->postIdx, totalWqes);
  base = __shfl(base, leaderLane);
  uint32_t mySlot = base + myLogicalLaneId;
  uint32_t signalSlot = base + numActiveLanes;
  uintptr_t atomicLaddr = reinterpret_cast<uintptr_t>(ep->atomicIbuf.addr);
  uint32_t atomicLkey = ep->atomicIbuf.lkey;

  if constexpr (PrvdType == core::ProviderType::BNXT) {
    // Reserve per-packet PSNs first (inline write is 1 packet), then post.
    uint32_t sigPsn = 0;
    uint32_t dataPsn = warpAggregateBnxtPsn(wq, /*dataPsnCnt=*/1, hasSignal, totalWqes, activemask,
                                            leaderLane, isLeader, &sigPsn);
    waitSqSpace<PrvdType>(ep, base, totalWqes);
    uint64_t dbrVal = core::PostWriteInline<PrvdType>(*wq, mySlot, mySlot, dataPsn,
                                                      true /*cqeSignal*/, qpn, &value, remoteAddr,
                                                      remoteKey, sizeof(T));
    __threadfence();
    if (isLeader) {
      if (hasSignal) {
        dbrVal = core::PostAtomic<PrvdType, uint64_t>(
            *wq, signalSlot, signalSlot, sigPsn, true /*cqeSignal*/, qpn, atomicLaddr, atomicLkey,
            signalRemoteAddr, signalRemoteKey, signalOpArg, 0, core::AMO_FETCH_ADD);
      }
      if (!(optFlags & ccoGdaOptFlagsAggregateRequests))
        ringDoorbellOrdered<PrvdType, /*LeaderOnly=*/true>(ep, base, totalWqes, dbrVal);
    }
  } else {
    // MLX5/PSD: the WQE slot index doubles as the PSN.
    waitSqSpace<PrvdType>(ep, base, totalWqes);
    uint64_t dbrVal = core::PostWriteInline<PrvdType>(*wq, mySlot, mySlot, mySlot, true /*cqeSignal*/,
                                                      qpn, &value, remoteAddr, remoteKey, sizeof(T));
    __threadfence();
    if (isLeader) {
      if (hasSignal) {
        dbrVal = core::PostAtomic<PrvdType, uint64_t>(
            *wq, signalSlot, signalSlot, signalSlot, true /*cqeSignal*/, qpn, atomicLaddr,
            atomicLkey, signalRemoteAddr, signalRemoteKey, signalOpArg, 0, core::AMO_FETCH_ADD);
      }
      if (!(optFlags & ccoGdaOptFlagsAggregateRequests))
        ringDoorbellOrdered<PrvdType, /*LeaderOnly=*/true>(ep, base, totalWqes, dbrVal);
    }
  }
}

// getImpl - RDMA read. One group per call (the facade groups lanes by peer):
// each active lane posts its read WQE into a contiguous reservation, the leader
// rings one doorbell.
template <core::ProviderType PrvdType>
__device__ inline static void getImpl(core::RdmaEndpointDevice* ep, uint32_t qpn,
                                      uintptr_t localAddr, uint32_t localKey, uintptr_t remoteAddr,
                                      uint32_t remoteKey, size_t bytes,
                                      uint32_t optFlags = ccoGdaOptFlagsDefault) {
  core::WorkQueueHandle* wq = &ep->wqHandle;
  uint64_t activemask = core::GetActiveLaneMask();
  int leaderLane = core::GetLastActiveLaneID(activemask);
  uint32_t numActiveLanes = core::GetActiveLaneCount(activemask);
  uint32_t myLogicalLaneId = core::GetActiveLaneNum(activemask);
  bool isLeader = (myLogicalLaneId == numActiveLanes - 1);
  uint32_t totalWqes = numActiveLanes;

  uint32_t base = 0;
  if (isLeader) base = atomicAdd(&wq->postIdx, totalWqes);
  base = __shfl(base, leaderLane);
  uint32_t mySlot = base + myLogicalLaneId;

  if constexpr (PrvdType == core::ProviderType::BNXT) {
    // Reserve per-packet PSNs first (read consumes response-packet PSNs), then post.
    uint32_t dataPsnCnt = (bytes == 0) ? 1 : (bytes + wq->mtuSize - 1) / wq->mtuSize;
    uint32_t dataPsn = warpAggregateBnxtPsn(wq, dataPsnCnt, /*hasSignalPacket=*/false, totalWqes,
                                            activemask, leaderLane, isLeader,
                                            /*outSignalPsn=*/nullptr);
    waitSqSpace<PrvdType>(ep, base, totalWqes);
    uint64_t dbrVal = core::PostRead<PrvdType>(*wq, mySlot, mySlot, dataPsn, true /*cqeSignal*/, qpn,
                                               localAddr, localKey, remoteAddr, remoteKey, bytes);
    __threadfence();
    if (isLeader && !(optFlags & ccoGdaOptFlagsAggregateRequests))
      ringDoorbellOrdered<PrvdType, /*LeaderOnly=*/true>(ep, base, totalWqes, dbrVal);
  } else {
    // MLX5/PSD: the WQE slot index doubles as the PSN.
    waitSqSpace<PrvdType>(ep, base, totalWqes);
    uint64_t dbrVal = core::PostRead<PrvdType>(*wq, mySlot, mySlot, mySlot, true /*cqeSignal*/, qpn,
                                               localAddr, localKey, remoteAddr, remoteKey, bytes);
    __threadfence();
    if (isLeader && !(optFlags & ccoGdaOptFlagsAggregateRequests))
      ringDoorbellOrdered<PrvdType, /*LeaderOnly=*/true>(ep, base, totalWqes, dbrVal);
  }
}

// FlushAsync: ring doorbell for pending WQEs (skip if already rung),
// return the postIdx for later wait.
template <core::ProviderType PrvdType>
__device__ inline static void flushAsyncImpl(core::RdmaEndpointDevice* ep, uint32_t qpn,
                                             uint32_t* outPostIdx) {
  core::WorkQueueHandle* wq = &ep->wqHandle;
  core::CompletionQueueHandle* cq = &ep->cqHandle;

  uint32_t curPostIdx = wq->postIdx;
  *outPostIdx = curPostIdx;

  uint64_t dbTouched =
      __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  if (dbTouched == curPostIdx) return;

  uint32_t numPendingWqes = curPostIdx - static_cast<uint32_t>(dbTouched);
  uint64_t dbrVal = buildFlushDbrVal<PrvdType>(wq, curPostIdx, qpn);

  __threadfence_system();

  // flush() is multi-lane (each lane flushes a different peer/QP), so PSD/BNXT
  // must serialize doorbells per lane (shared dbrAddr/UAR); MLX5 has a per-QP
  // dbrAddr and rings directly.
  if constexpr (PrvdType == core::ProviderType::MLX5) {
    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, curPostIdx);
    __threadfence_system();
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
  } else {
    ringDoorbellWalk<PrvdType>(wq, curPostIdx, dbrVal);
  }

  __threadfence_system();

  __hip_atomic_fetch_add(&cq->needConsIdx, numPendingWqes, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_store(&wq->dbTouchIdx, static_cast<uint64_t>(curPostIdx), __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_AGENT);
}

// Wait: wait for async request to complete
template <core::ProviderType PrvdType>
__device__ inline static void waitImpl(core::RdmaEndpointDevice* ep, uint32_t postIdx) {
  quietUntil<PrvdType>(ep, postIdx);
}

// Signal: send signal to remote peer (RDMA atomic increment/add)
template <core::ProviderType PrvdType>
__device__ inline static void signalImpl(core::RdmaEndpointDevice* ep, uint32_t qpn,
                                         uintptr_t signalRemoteAddr, uint32_t signalRemoteKey,
                                         ccoGdaSignalOp_t signalOp, uint64_t signalOpArg,
                                         uint32_t optFlags = ccoGdaOptFlagsDefault) {
  core::WorkQueueHandle* wq = &ep->wqHandle;
  // RDMA atomic requires a local buffer for the FetchAdd result (even if unused).
  uintptr_t atomicLaddr = reinterpret_cast<uintptr_t>(ep->atomicIbuf.addr);
  uint32_t atomicLkey = ep->atomicIbuf.lkey;
  uint64_t addValue = (signalOp == ccoGdaSignalInc) ? 1 : signalOpArg;

  if constexpr (PrvdType == core::ProviderType::BNXT) {
    // Reserve a WQE slot + 1 packet PSN; post the atomic with the packet PSN.
    uint32_t psnBase = 0;
    uint32_t curPostIdx = reserveWqeSlots<PrvdType>(ep, 1, /*lanePsnCnt=*/1, &psnBase);
    uint64_t dbrVal = core::PostAtomic<PrvdType, uint64_t>(
        *wq, curPostIdx, curPostIdx, psnBase, true /*cqeSignal*/, qpn, atomicLaddr, atomicLkey,
        signalRemoteAddr, signalRemoteKey, addValue, 0 /*compare*/, core::AMO_FETCH_ADD);
    if (!(optFlags & ccoGdaOptFlagsAggregateRequests))
      ringDoorbellOrdered<PrvdType>(ep, curPostIdx, 1, dbrVal);
  } else {
    // MLX5/PSD: the WQE slot index doubles as the PSN.
    uint32_t curPostIdx = reserveWqeSlots<PrvdType>(ep, 1);
    uint64_t dbrVal = core::PostAtomic<PrvdType, uint64_t>(
        *wq, curPostIdx, curPostIdx, curPostIdx, true /*cqeSignal*/, qpn, atomicLaddr, atomicLkey,
        signalRemoteAddr, signalRemoteKey, addValue, 0 /*compare*/, core::AMO_FETCH_ADD);
    if (!(optFlags & ccoGdaOptFlagsAggregateRequests))
      ringDoorbellOrdered<PrvdType>(ep, curPostIdx, 1, dbrVal);
  }
}

// ReadSignal: read local signal value
template <core::ProviderType PrvdType>
__device__ inline static uint64_t readSignalImpl(volatile uint64_t* signalBuf,
                                                 volatile uint64_t* signalShadows,
                                                 ccoGdaSignal_t signalId, int bits) {
  uint64_t val =
      __hip_atomic_load(&signalBuf[signalId], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  uint64_t shadow = signalShadows[signalId];
  uint64_t mask = (bits >= 64) ? UINT64_MAX : ((1ULL << bits) - 1);
  return (val - shadow) & mask;
}

// WaitSignal: wait until local signal reaches specified value
template <core::ProviderType PrvdType>
__device__ inline static void waitSignalImpl(volatile uint64_t* signalBuf,
                                             volatile uint64_t* signalShadows,
                                             ccoGdaSignal_t signalId, uint64_t least, int bits) {
  uint64_t mask = (bits >= 64) ? UINT64_MAX : ((1ULL << bits) - 1);
  uint64_t shadow = signalShadows[signalId];

  while (true) {
    uint64_t val =
        __hip_atomic_load(&signalBuf[signalId], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    uint64_t delta = (val - shadow) & mask;
    if (delta >= least) {
      // Update shadow to consume
      signalShadows[signalId] = (shadow + least) & mask;
      break;
    }
    // Spin wait
    asm volatile("" ::: "memory");
  }
}

// ResetSignal: reset local signal to zero
template <core::ProviderType PrvdType>
__device__ inline static void resetSignalImpl(volatile uint64_t* signalBuf,
                                              volatile uint64_t* signalShadows,
                                              ccoGdaSignal_t signalId) {
  signalBuf[signalId] = 0;
  signalShadows[signalId] = 0;
}

// ReadCounter: read local counter value
template <core::ProviderType PrvdType>
__device__ inline static uint64_t readCounterImpl(volatile uint64_t* counterBuf,
                                                  ccoGdaCounter_t counterId, int bits) {
  uint64_t val =
      __hip_atomic_load(&counterBuf[counterId], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  uint64_t mask = (bits >= 64) ? UINT64_MAX : ((1ULL << bits) - 1);
  return val & mask;
}

// WaitCounter: wait until local counter reaches specified value
template <core::ProviderType PrvdType>
__device__ inline static void waitCounterImpl(volatile uint64_t* counterBuf,
                                              ccoGdaCounter_t counterId, uint64_t least, int bits) {
  uint64_t mask = (bits >= 64) ? UINT64_MAX : ((1ULL << bits) - 1);

  while (true) {
    uint64_t val =
        __hip_atomic_load(&counterBuf[counterId], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    if ((val & mask) >= least) {
      break;
    }
    // Spin wait
    asm volatile("" ::: "memory");
  }
}

// ResetCounter: reset local counter to zero
template <core::ProviderType PrvdType>
__device__ inline static void resetCounterImpl(volatile uint64_t* counterBuf,
                                               ccoGdaCounter_t counterId) {
  counterBuf[counterId] = 0;
}

// peer<->world translation (internal helpers, used by the ccoGda methods below).
// translate a GDA team-local peer index to a global rank.
// FULL:      identity
// RAIL:      teamPeer is node_id; global = teamPeer * lsaSize + lsaRank
// CROSSNODE: team = [0,nodeStart) ∪ {self at nodeStart} ∪ [nodeStart+lsaSize,worldSize)
// NONE:      returns -1
__device__ inline int GdaPeerToWorld(ccoDevComm const& comm, int teamPeer) {
  switch (comm.gdaConnType) {
    case CCO_GDA_CONNECTION_FULL:
      return teamPeer;
    case CCO_GDA_CONNECTION_RAIL:
      return teamPeer * comm.lsaSize + comm.lsaRank;
    case CCO_GDA_CONNECTION_CROSSNODE: {
      int nodeStart = (comm.rank / comm.lsaSize) * comm.lsaSize;
      if (teamPeer < nodeStart) return teamPeer;
      if (teamPeer == nodeStart) return comm.rank;
      return teamPeer + comm.lsaSize - 1;
    }
    default:
      return -1;
  }
}

// translate a global rank to a GDA team-local peer index (inverse of GdaPeerToWorld).
// FULL:      identity
// RAIL:      teamPeer = globalPeer / lsaSize (node_id of globalPeer)
// CROSSNODE: reverse the team layout described above
// NONE:      returns -1
__device__ inline int WorldPeerToGda(ccoDevComm const& comm, int globalPeer) {
  switch (comm.gdaConnType) {
    case CCO_GDA_CONNECTION_FULL:
      return globalPeer;
    case CCO_GDA_CONNECTION_RAIL:
      return globalPeer / comm.lsaSize;
    case CCO_GDA_CONNECTION_CROSSNODE: {
      int nodeStart = (comm.rank / comm.lsaSize) * comm.lsaSize;
      if (globalPeer < nodeStart) return globalPeer;
      if (globalPeer == comm.rank) return nodeStart;
      return globalPeer - comm.lsaSize + 1;
    }
    default:
      return -1;
  }
}

}  // namespace impl

// ── ccoGda<PrvdType> method definitions ──
// Public facade: thin per-method wrappers that select the endpoint and dispatch
// to the impl:: primitive layer above.
template <core::ProviderType PrvdType>
template <ccoTeamMode TeamMode>
__device__ inline int ccoGda<PrvdType>::resolveWorldPeer(int peer) const {
  if constexpr (TeamMode == CCO_TEAM_WORLD) {
    return peer;
  } else {
    return impl::GdaPeerToWorld(comm, peer);
  }
}

template <core::ProviderType PrvdType>
__device__ inline ccoGda<PrvdType>::ccoGda(ccoDevComm const& comm_, int contextIndex)
    : comm(comm_), contextId(contextIndex) {
  this->_gdaHandle = (void*)&comm.ibgda;
  switch (comm.gdaConnType) {
    case CCO_GDA_CONNECTION_FULL:
      this->rank = comm.rank;
      this->nRanks = comm.worldSize;
      break;
    case CCO_GDA_CONNECTION_RAIL:
      this->rank = comm.rank / comm.lsaSize;
      this->nRanks = comm.worldSize / comm.lsaSize;
      break;
    case CCO_GDA_CONNECTION_CROSSNODE: {
      int nodeStart = (comm.rank / comm.lsaSize) * comm.lsaSize;
      this->rank = nodeStart;
      this->nRanks = comm.worldSize - comm.lsaSize + 1;
      break;
    }
    default:  // CCO_GDA_CONNECTION_NONE
      this->rank = 0;
      this->nRanks = 0;
      break;
  }
}

// put: RDMA write with optional signal
template <core::ProviderType PrvdType>
template <ccoTeamMode TeamMode, ccoGdaThreadMode ThreadMode, typename RemoteAction, typename Coop>
__device__ inline void ccoGda<PrvdType>::put(int peer, ccoWindow_t dstWin, size_t dstOffset,
                                             ccoWindow_t srcWin, size_t srcOffset, size_t bytes,
                                             RemoteAction remoteAction, Coop coop,
                                             uint32_t optFlags) {
  if constexpr (ThreadMode == ccoGdaThreadAggregate) {
    static_assert(std::is_same_v<Coop, ccoCoopThread>,
                  "ccoGdaThreadAggregate requires ccoCoopThread — all warp lanes must enter putImpl.");
  }
  coop.sync();
  if (coop.thread_rank() == 0) {
    // Each active lane resolves its own peer endpoint/keys.
    int worldPeer = resolveWorldPeer<TeamMode>(peer);

    ccoWindowDevice* dstWinDev = reinterpret_cast<ccoWindowDevice*>(dstWin);
    ccoWindowDevice* srcWinDev = reinterpret_cast<ccoWindowDevice*>(srcWin);

    uint32_t srcLkey = srcWinDev->ibgdaWin.lkey;
    uint32_t dstRkey = dstWinDev->ibgdaWin.peerRkeys[worldPeer];

    uintptr_t localAddr = srcOffset;
    uintptr_t remoteAddr = dstOffset;

    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    core::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
    uint32_t qpn = ep->qpn;

    constexpr bool hasSignal = !std::is_same_v<RemoteAction, ccoGda_NoSignal>;
    uintptr_t signalRaddr = 0;
    uint32_t signalRkey = 0;
    ccoGdaSignalOp_t signalOp = ccoGdaSignalInc;
    uint64_t signalOpArg = 0;

    if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalInc>) {
      signalRaddr = remoteAction.signalId * sizeof(uint64_t);
      signalRkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[worldPeer];
      signalOp = ccoGdaSignalInc;
      signalOpArg = 1;
    } else if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalAdd>) {
      signalRaddr = remoteAction.signalId * sizeof(uint64_t);
      signalRkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[worldPeer];
      signalOp = ccoGdaSignalAdd;
      signalOpArg = remoteAction.value;
    }

    // Only mixed-peer thread scope needs per-peer grouping; ThreadAggregate and
    // CoopWarp/CoopBlock are a single group.
    if constexpr (ThreadMode == ccoGdaThreadIndependent && std::is_same_v<Coop, ccoCoopThread>) {
      // Group active lanes by peer: each peer reserves contiguously + one doorbell.
      bool needTurn = true;
      for (uint64_t turns = __ballot(needTurn); turns != 0; turns = __ballot(needTurn)) {
        int lead = __ffsll(static_cast<unsigned long long>(turns)) - 1;
        if (peer != __shfl(peer, lead)) continue;
        needTurn = false;
        impl::putImpl<PrvdType>(ep, qpn, localAddr, srcLkey, remoteAddr, dstRkey, bytes, hasSignal,
                                signalRaddr, signalRkey, signalOp, signalOpArg, optFlags);
      }
    } else {
      impl::putImpl<PrvdType>(ep, qpn, localAddr, srcLkey, remoteAddr, dstRkey, bytes, hasSignal,
                              signalRaddr, signalRkey, signalOp, signalOpArg, optFlags);
    }
  }
  coop.sync();
}

// putValue: write immediate value (≤8 bytes)
template <core::ProviderType PrvdType>
template <ccoTeamMode TeamMode, ccoGdaThreadMode ThreadMode, typename T, typename RemoteAction,
          typename Coop>
__device__ inline void ccoGda<PrvdType>::putValue(int peer, ccoWindow_t dstWin, size_t dstOffset,
                                                  T value, RemoteAction remoteAction, Coop coop,
                                                  uint32_t optFlags) {
  static_assert(sizeof(T) <= 8, "putValue only supports types <= 8 bytes");
  if constexpr (ThreadMode == ccoGdaThreadAggregate) {
    static_assert(std::is_same_v<Coop, ccoCoopThread>,
                  "ccoGdaThreadAggregate requires ccoCoopThread — all warp lanes must enter putValueImpl.");
  }

  coop.sync();
  if (coop.thread_rank() == 0) {
    int worldPeer = resolveWorldPeer<TeamMode>(peer);

    ccoWindowDevice* dstWinDev = reinterpret_cast<ccoWindowDevice*>(dstWin);
    uint32_t dstRkey = dstWinDev->ibgdaWin.peerRkeys[worldPeer];
    uintptr_t remoteAddr = dstOffset;

    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    core::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
    uint32_t qpn = ep->qpn;

    constexpr bool hasSignal = !std::is_same_v<RemoteAction, ccoGda_NoSignal>;
    uintptr_t signalRaddr = 0;
    uint32_t signalRkey = 0;
    ccoGdaSignalOp_t signalOp = ccoGdaSignalInc;
    uint64_t signalOpArg = 0;

    if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalInc>) {
      signalRaddr = remoteAction.signalId * sizeof(uint64_t);
      signalRkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[worldPeer];
      signalOp = ccoGdaSignalInc;
      signalOpArg = 1;
    } else if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalAdd>) {
      signalRaddr = remoteAction.signalId * sizeof(uint64_t);
      signalRkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[worldPeer];
      signalOp = ccoGdaSignalAdd;
      signalOpArg = remoteAction.value;
    }

    // Only mixed-peer thread scope needs per-peer grouping; else a single group.
    if constexpr (ThreadMode == ccoGdaThreadIndependent && std::is_same_v<Coop, ccoCoopThread>) {
      bool needTurn = true;
      for (uint64_t turns = __ballot(needTurn); turns != 0; turns = __ballot(needTurn)) {
        int lead = __ffsll(static_cast<unsigned long long>(turns)) - 1;
        if (peer != __shfl(peer, lead)) continue;
        needTurn = false;
        impl::putValueImpl<PrvdType, T>(ep, qpn, remoteAddr, dstRkey, value, hasSignal, signalRaddr,
                                        signalRkey, signalOp, signalOpArg, optFlags);
      }
    } else {
      impl::putValueImpl<PrvdType, T>(ep, qpn, remoteAddr, dstRkey, value, hasSignal, signalRaddr,
                                      signalRkey, signalOp, signalOpArg, optFlags);
    }
  }
  coop.sync();
}

// get: RDMA read
template <core::ProviderType PrvdType>
template <ccoTeamMode TeamMode, ccoGdaThreadMode ThreadMode, typename Coop>
__device__ inline void ccoGda<PrvdType>::get(int peer, ccoWindow_t remoteWin, size_t remoteOffset,
                                             ccoWindow_t localWin, size_t localOffset, size_t bytes,
                                             Coop coop, uint32_t optFlags) {
  if constexpr (ThreadMode == ccoGdaThreadAggregate) {
    static_assert(std::is_same_v<Coop, ccoCoopThread>,
                  "ccoGdaThreadAggregate requires ccoCoopThread — all warp lanes must enter getImpl.");
  }
  coop.sync();
  if (coop.thread_rank() == 0) {
    int worldPeer = resolveWorldPeer<TeamMode>(peer);

    ccoWindowDevice* remoteWinDev = reinterpret_cast<ccoWindowDevice*>(remoteWin);
    ccoWindowDevice* localWinDev = reinterpret_cast<ccoWindowDevice*>(localWin);

    uint32_t remoteRkey = remoteWinDev->ibgdaWin.peerRkeys[worldPeer];
    uint32_t localLkey = localWinDev->ibgdaWin.lkey;

    uintptr_t remoteAddr = remoteOffset;
    uintptr_t localAddr = localOffset;

    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    core::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
    uint32_t qpn = ep->qpn;

    // Only mixed-peer thread scope needs per-peer grouping; else a single group.
    if constexpr (ThreadMode == ccoGdaThreadIndependent && std::is_same_v<Coop, ccoCoopThread>) {
      bool needTurn = true;
      for (uint64_t turns = __ballot(needTurn); turns != 0; turns = __ballot(needTurn)) {
        int lead = __ffsll(static_cast<unsigned long long>(turns)) - 1;
        if (peer != __shfl(peer, lead)) continue;
        needTurn = false;
        impl::getImpl<PrvdType>(ep, qpn, localAddr, localLkey, remoteAddr, remoteRkey, bytes,
                                optFlags);
      }
    } else {
      impl::getImpl<PrvdType>(ep, qpn, localAddr, localLkey, remoteAddr, remoteRkey, bytes,
                              optFlags);
    }
  }
  coop.sync();
}

// signal: send to remote peer
template <core::ProviderType PrvdType>
template <ccoTeamMode TeamMode, typename RemoteAction, typename Coop>
__device__ inline void ccoGda<PrvdType>::signal(int peer, RemoteAction remoteAction, Coop coop) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    int worldPeer = resolveWorldPeer<TeamMode>(peer);

    // select endpoint first to get ibgda context
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    core::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
    uint32_t qpn = ep->qpn;

    // parse RemoteAction
    ccoGdaSignalOp_t signalOp = ccoGdaSignalInc;
    uint64_t signalOpArg = 0;
    uintptr_t signalRaddr = 0;
    uint32_t signalRkey = 0;

    if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalInc>) {
      signalRaddr = remoteAction.signalId * sizeof(uint64_t);
      signalRkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[worldPeer];
      signalOp = ccoGdaSignalInc;
      signalOpArg = 1;
    } else if constexpr (std::is_same_v<RemoteAction, ccoGda_SignalAdd>) {
      signalRaddr = remoteAction.signalId * sizeof(uint64_t);
      signalRkey = comm.resourceWindow_inlined.ibgdaWin.peerRkeys[worldPeer];
      signalOp = ccoGdaSignalAdd;
      signalOpArg = remoteAction.value;
    }

    // call primitive signal
    impl::signalImpl<PrvdType>(ep, qpn, signalRaddr, signalRkey, signalOp, signalOpArg);
  }
  coop.sync();
}

// flush = flushAsync + wait per peer.
// flushAsync rings the doorbell if any WQEs are pending (skips if already rung),
// then wait polls CQ until all submitted WQEs complete.

// flush all peers: distribute peers across the Coop group (default: warp).
// all threads in the group must call flush together.
template <core::ProviderType PrvdType>
template <typename Coop>
__device__ inline void ccoGda<PrvdType>::flush(Coop coop) {
  static_assert(!std::is_same_v<Coop, ccoCoopThread>,
                "flush() requires at least ccoCoopWarp. "
                "ccoCoopThread causes each thread to independently enter quietUntil "
                "on different QPs, breaking the warp-level pollCqLock.");
  coop.sync();
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
  for (int teamPeer = coop.thread_rank(); teamPeer < this->nRanks; teamPeer += coop.size()) {
    if (teamPeer == this->rank) continue;
    // endpoints are world-indexed; the loop walks the GDA team.
    int worldPeer = impl::GdaPeerToWorld(comm, teamPeer);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    core::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
    uint32_t postIdx = 0;
    impl::flushAsyncImpl<PrvdType>(ep, ep->qpn, &postIdx);
    impl::waitImpl<PrvdType>(ep, postIdx);
  }
  coop.sync();
}

// flush single peer: ring doorbell if needed, then poll CQ until complete.
template <core::ProviderType PrvdType>
template <ccoTeamMode TeamMode, typename Coop>
__device__ inline void ccoGda<PrvdType>::flush(int peer, Coop coop) {
  static_assert(!std::is_same_v<Coop, ccoCoopThread>,
                "flush(peer) requires at least ccoCoopWarp. "
                "ccoCoopThread allows concurrent per-thread calls on different QPs, "
                "which breaks the warp-level pollCqLock inside quietUntil.");
  coop.sync();
  if (coop.thread_rank() == 0) {
    int worldPeer = resolveWorldPeer<TeamMode>(peer);
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    core::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
    uint32_t postIdx = 0;
    impl::flushAsyncImpl<PrvdType>(ep, ep->qpn, &postIdx);
    impl::waitImpl<PrvdType>(ep, postIdx);
  }
  coop.sync();
}

// flushAsync: ring doorbell for peer, return a request handle for wait().
template <core::ProviderType PrvdType>
template <ccoTeamMode TeamMode, typename Coop>
__device__ inline void ccoGda<PrvdType>::flushAsync(int peer, ccoGdaRequest_t* outRequest,
                                                    Coop coop) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    int worldPeer = resolveWorldPeer<TeamMode>(peer);
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    core::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];

    uint32_t postIdx = 0;
    impl::flushAsyncImpl<PrvdType>(ep, ep->qpn, &postIdx);

    outRequest->qpIdx = qpIdx;
    outRequest->postIdx = static_cast<uint64_t>(postIdx);
  }
  coop.sync();
}

// wait: poll CQ until the request returned by flushAsync completes.
template <core::ProviderType PrvdType>
template <typename Coop>
__device__ inline void ccoGda<PrvdType>::wait(ccoGdaRequest_t& request, Coop coop) {
  static_assert(!std::is_same_v<Coop, ccoCoopThread>,
                "wait() requires at least ccoCoopWarp. "
                "ccoCoopThread allows concurrent per-thread calls on different QPs, "
                "which breaks the warp-level pollCqLock inside quietUntil.");
  coop.sync();
  if (coop.thread_rank() == 0) {
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    impl::waitImpl<PrvdType>(&ibgda->endpoints[request.qpIdx],
                             static_cast<uint32_t>(request.postIdx));
  }
  coop.sync();
}

// counter: poll CQ for all GDA-team peers, then software-increment counterBuf.
template <core::ProviderType PrvdType>
template <typename LocalAction, typename Coop>
__device__ inline void ccoGda<PrvdType>::counter(LocalAction localAction, Coop coop) {
  static_assert(!std::is_same_v<Coop, ccoCoopThread>,
                "counter() requires at least ccoCoopWarp. "
                "ccoCoopThread causes each thread to independently enter quietUntil "
                "on different QPs, breaking the warp-level pollCqLock.");
  coop.sync();

  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);

  for (int teamPeer = coop.thread_rank(); teamPeer < this->nRanks; teamPeer += coop.size()) {
    if (teamPeer == this->rank) continue;
    // endpoints are world-indexed; the loop walks the GDA team.
    int worldPeer = impl::GdaPeerToWorld(comm, teamPeer);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    core::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
    impl::quietUntil<PrvdType>(ep, ep->wqHandle.postIdx);
  }

  coop.sync();

  if (coop.thread_rank() == 0) {
    if constexpr (std::is_same_v<LocalAction, ccoGda_CounterInc>) {
      atomicAdd(&ibgda->counterBuf[localAction.counterId], (uint64_t)1);
    }
  }

  coop.sync();
}

// readSignal: read local signal value
template <core::ProviderType PrvdType>
__device__ inline uint64_t ccoGda<PrvdType>::readSignal(ccoGdaSignal_t signalId, int bits) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
  return impl::readSignalImpl<PrvdType>(ibgda->signalBuf, ibgda->signalShadows, signalId, bits);
}

// waitSignal: wait until local signal reaches specified value
template <core::ProviderType PrvdType>
template <typename Coop>
__device__ inline void ccoGda<PrvdType>::waitSignal(ccoGdaSignal_t signalId, uint64_t least,
                                                    Coop coop, int bits) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    impl::waitSignalImpl<PrvdType>(ibgda->signalBuf, ibgda->signalShadows, signalId, least, bits);
  }
  coop.sync();
}

// resetSignal: reset local signal to zero
template <core::ProviderType PrvdType>
__device__ inline void ccoGda<PrvdType>::resetSignal(ccoGdaSignal_t signalId) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
  impl::resetSignalImpl<PrvdType>(ibgda->signalBuf, ibgda->signalShadows, signalId);
}

// readCounter: read local counter value
template <core::ProviderType PrvdType>
__device__ inline uint64_t ccoGda<PrvdType>::readCounter(ccoGdaCounter_t counterId, int bits) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
  return impl::readCounterImpl<PrvdType>(ibgda->counterBuf, counterId, bits);
}

// waitCounter: wait until local counter reaches specified value
template <core::ProviderType PrvdType>
template <typename Coop>
__device__ inline void ccoGda<PrvdType>::waitCounter(ccoGdaCounter_t counterId, uint64_t least,
                                                     Coop coop, int bits) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    impl::waitCounterImpl<PrvdType>(ibgda->counterBuf, counterId, least, bits);
  }
  coop.sync();
}

// resetCounter: reset local counter to zero
template <core::ProviderType PrvdType>
__device__ inline void ccoGda<PrvdType>::resetCounter(ccoGdaCounter_t counterId) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
  impl::resetCounterImpl<PrvdType>(ibgda->counterBuf, counterId);
}

// ── ccoGdaBarrierSession method definitions ──

template <core::ProviderType PrvdType, typename Coop>
__device__ inline ccoGdaBarrierSession<PrvdType, Coop>::ccoGdaBarrierSession(
    Coop coop_, ccoGda<PrvdType>& gda_, ccoGdaBarrierHandle handle_, uint32_t index_)
    : coop(coop_), gda(gda_), handle(handle_), index(index_) {}

template <core::ProviderType PrvdType, typename Coop>
__device__ inline void ccoGdaBarrierSession<PrvdType, Coop>::sync(Coop) {
  static_assert(!std::is_same_v<Coop, ccoCoopThread>,
                "GDA barrier requires at least ccoCoopWarp. "
                "ccoCoopThread causes each thread to independently enter signalImpl / "
                "waitSignalImpl on different QPs, breaking the warp-level pollCqLock.");
  this->coop.sync();

  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(gda._gdaHandle);
  int myRank = gda.rank;
  int nRanks = gda.nRanks;

  // Each barrier instance uses nRanks signal slots starting at signalBase.
  // slot[peer] at our rank: peer writes here to notify us.
  // slot[myRank] at peer's rank: we write here to notify peer.
  ccoGdaSignal_t signalBase = handle.signal0 + index * nRanks;

  // Phase 1: signal every peer (distribute across coop lanes).
  // Peer rotation: (myRank+1+i) % nRanks spreads load evenly.
  for (int i = this->coop.thread_rank(); i < nRanks - 1; i += this->coop.size()) {
    int peer = 1 + myRank + i;
    if (peer >= nRanks) peer -= nRanks;

    // endpoints/peerRkeys are world-indexed; peer is GDA team-local.
    int worldPeer = impl::GdaPeerToWorld(gda.comm, peer);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (gda.contextId % ibgda->numQpPerPe);
    core::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];

    uintptr_t signalRaddr = (signalBase + myRank) * sizeof(uint64_t);
    uint32_t signalRkey = gda.comm.resourceWindow_inlined.ibgdaWin.peerRkeys[worldPeer];

    impl::signalImpl<PrvdType>(ep, ep->qpn, signalRaddr, signalRkey, ccoGdaSignalInc, 1);
  }

  this->coop.sync();

  // Phase 2: wait for every peer's reciprocal signal.
  for (int i = this->coop.thread_rank(); i < nRanks - 1; i += this->coop.size()) {
    int peer = 1 + myRank + i;
    if (peer >= nRanks) peer -= nRanks;

    ccoGdaSignal_t slotId = signalBase + peer;
    impl::waitSignalImpl<PrvdType>(ibgda->signalBuf, ibgda->signalShadows, slotId, 1, 64);
  }

  this->coop.sync();
}

template <core::ProviderType PrvdType, typename Coop>
__device__ inline void ccoGdaBarrier(Coop coop, ccoGda<PrvdType>& gda, ccoGdaBarrierHandle handle,
                                     uint32_t index) {
  ccoGdaBarrierSession<PrvdType, Coop> session(coop, gda, handle, index);
  session.sync(coop);
}

#endif  // defined(__HIPCC__) || defined(__CUDACC__)

}  // namespace cco
}  // namespace mori
