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

// GDA (RDMA) device layer pulls in the provider RDMA core. Device-only.
#if defined(__HIPCC__) || defined(__CUDACC__)
#include "mori/core/transport/rdma/rdma.hpp"
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
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD, typename RemoteAction = ccoGda_NoSignal,
            typename Coop = ccoCoopThread>
  __device__ inline void put(int peer, ccoWindow_t dstWin, size_t dstOffset, ccoWindow_t srcWin,
                             size_t srcOffset, size_t bytes,
                             RemoteAction remoteAction = ccoGda_NoSignal{}, Coop coop = Coop{},
                             uint32_t optFlags = ccoGdaOptFlagsDefault);

  // putValue: write an immediate value (≤8 bytes) with optional remote signal.
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD, typename T,
            typename RemoteAction = ccoGda_NoSignal, typename Coop = ccoCoopThread>
  __device__ inline void putValue(int peer, ccoWindow_t dstWin, size_t dstOffset, T value,
                                  RemoteAction remoteAction = ccoGda_NoSignal{}, Coop coop = Coop{},
                                  uint32_t optFlags = ccoGdaOptFlagsDefault);

  // get: rdma read — pull peer's window content into our local window.
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD, typename Coop = ccoCoopThread>
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

// Record the logical WQE id at its SQ slot so quietUntil can map a CQE's slot
// back to the monotonic WQE id. BNXT indexes by SQ slot (% sqWqeNum); PSD uses
// the large software table. MLX5 does not use the table here.
template <core::ProviderType PrvdType>
__device__ inline static void recordOutstandingWqe(core::WorkQueueHandle* wq, uint32_t idx) {
  if constexpr (PrvdType == core::ProviderType::BNXT) {
    wq->outstandingWqe[idx % wq->sqWqeNum] = idx;
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    wq->outstandingWqe[idx % OUTSTANDING_TABLE_SIZE] = idx;
  }
}

// Poll CQ and update doneIdx until it catches up to targetIdx
template <core::ProviderType PrvdType>
__device__ inline static void quietUntil(application::RdmaEndpointDevice* ep, uint32_t targetIdx) {
  core::WorkQueueHandle* wq = &ep->wqHandle;
  core::CompletionQueueHandle* cq = &ep->cqHandle;

  if constexpr (PrvdType == core::ProviderType::PSD) {
    // PSD/Ionic: 24-bit MSN field, use sign bit (0x800000) for wraparound comparison
    while ((wq->doneIdx - targetIdx) & 0x800000) {
      uint64_t activemask = core::GetActiveLaneMask();
      if (!core::spin_lock_try_acquire_shared(&cq->pollCqLock, activemask)) {
        continue;
      }

      uint32_t greed = 10;
      bool pollErr = false;
      while ((wq->doneIdx - targetIdx) & 0x800000) {
        uint32_t oldDoneIdx = wq->doneIdx;
        int err = core::PollCqOnce2(*wq, *cq, activemask, cq->cqAddr, cq->cqeNum, 0);
        if (err != 0) {
          MORI_PRINTF("quietUntil[PSD]: PollCqOnce2 failed, err=%d\n", err);
          pollErr = true;
          break;
        }
        asm volatile("" ::: "memory");

        if (!((wq->doneIdx - targetIdx) & 0x800000)) break;
        if (wq->doneIdx == oldDoneIdx) break;
        if (!greed--) break;
      }

      core::spin_lock_release_shared(&cq->pollCqLock, activemask);

      if (pollErr) break;
    }
  } else if constexpr (PrvdType == core::ProviderType::MLX5) {
    // MLX5: 16-bit wqe_counter, poll CQ and update DBR record
    // Use 16-bit wraparound comparison
    while ((int16_t)(wq->doneIdx - targetIdx) < 0) {
      uint32_t wqeCounter = 0;
      int err = core::PollCq<PrvdType>(cq->cqAddr, cq->cqeNum, &cq->consIdx, &wqeCounter);
      if (err >= 0) {
        wq->doneIdx = wqeCounter;
        core::UpdateCqDbrRecord<PrvdType>(*cq, cq->consIdx);
      }
      asm volatile("" ::: "memory");
    }
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    // BNXT: poll until the monotonic doneIdx reaches targetIdx. Mirrors shmem's
    // BNXT quiet — a single poller per CQ (pollCqLock); each thread claims a CQE
    // slot via atomic cq_consumer, maps the CQE's SQ slot back to the logical WQE
    // id through outstandingWqe[], and advances doneIdx with fetch_max. Threads
    // that can't take the lock spin re-reading doneIdx (the holder advances it).
    auto doneLt = [&]() {
      return (int32_t)(targetIdx - __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED,
                                                     __HIP_MEMORY_SCOPE_AGENT)) > 0;
    };
    while (doneLt()) {
      // Only one thread per CQ polls; others spin re-reading doneIdx, which the
      // holder advances. BNXT runs with cqeNum==1, so the single CQE is a
      // rolling slot the NIC overwrites; PollCqOnce returns its current
      // con_indx (the completed SQ slot) without a phase check.
      if (!core::AcquireLockOnce(&cq->pollCqLock)) continue;
      while (doneLt()) {
        uint32_t idx = cq->cq_consumer;
        uint32_t wqeCounter = 0;
        int opcode = core::PollCqOnce<PrvdType>(cq->cqAddr, cq->cqeNum, idx, &wqeCounter);
        if (opcode < 0) continue;  // no new completion yet
        cq->cq_consumer = idx + 1;
        // con_indx points at the slot past the completed WQE; step back one and
        // map it through outstandingWqe[] to the monotonic logical WQE id.
        uint32_t slot = (wqeCounter + wq->sqWqeNum - 1) % wq->sqWqeNum;
        uint64_t wqeId = wq->outstandingWqe[slot] + 1;
        __hip_atomic_fetch_max(&wq->doneIdx, (uint32_t)wqeId, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
      }
      core::ReleaseLock(&cq->pollCqLock);
    }
  }
}

// Reserve WQE slots and wait for SQ space.
//
// In thread mode (ccoCoopThread) every lane of the warp reaches here — the
// put/get facade's thread_rank()==0 gate passes for all threads — so when those
// lanes target the SAME work queue, one leader reserves all their slots with a
// single atomicAdd and runs the SQ-space spin once for the whole warp, instead
// of every lane contending on wq->postIdx and spinning independently (mirrors
// shmem's warp-aggregated reserve). Each lane takes a distinct slot range at
// base + logicalLaneId * numWqesNeeded; this assumes same-QP lanes share
// numWqesNeeded, true here since it depends only on the constexpr hasSignal and
// the atomic type. Lanes targeting different QPs fall back to the per-lane
// reserve. In warp/block mode only lane 0 reaches here, so the aggregation would
// be a no-op and is compiled out entirely.
template <core::ProviderType PrvdType, typename Coop = ccoCoopThread>
__device__ inline static uint32_t reserveWqeSlots(application::RdmaEndpointDevice* ep,
                                                  uint32_t numWqesNeeded) {
  core::WorkQueueHandle* wq = &ep->wqHandle;

  if constexpr (std::is_same_v<Coop, ccoCoopThread>) {
    uint64_t activemask = core::GetActiveLaneMask();
    int leaderLane = core::GetFirstActiveLaneID(activemask);
    uint32_t leaderQpn = __shfl(ep->qpn, leaderLane);
    bool sameQp = (ep->qpn == leaderQpn);

    if (__ballot(sameQp) == activemask) {
      // All active lanes target the same QP: one leader reserves the whole warp.
      uint32_t numActiveLanes = core::GetActiveLaneCount(activemask);
      uint32_t myLogicalLaneId = core::GetActiveLaneNum(activemask);
      uint32_t warpWqes = numActiveLanes * numWqesNeeded;

      uint32_t base = 0;
      if (myLogicalLaneId == 0) {
        base = atomicAdd(&wq->postIdx, warpWqes);
      }
      base = __shfl(base, leaderLane);
      uint32_t curPostIdx = base + myLogicalLaneId * numWqesNeeded;

      // Flow control once per warp: wait until the SQ has room for the whole warp.
      while (true) {
        uint64_t dbTouched =
            __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        uint64_t dbDone =
            __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        uint64_t numActiveSqEntries = dbTouched - dbDone;
        uint64_t numFreeEntries = wq->sqWqeNum - numActiveSqEntries;
        uint64_t entriesUntilWarpLast = base + warpWqes - dbTouched;
        if (numFreeEntries > entriesUntilWarpLast) {
          break;
        }
        quietUntil<PrvdType>(ep, base);
      }
      return curPostIdx;
    }
    // Mixed QPs: fall through to the per-lane reserve below.
  }

  // Per-lane reserve: each thread allocates and flow-controls on its own.
  uint32_t curPostIdx = atomicAdd(&wq->postIdx, numWqesNeeded);
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
    quietUntil<PrvdType>(ep, curPostIdx);
  }
  return curPostIdx;
}

// PSD/Ionic only: walk the warp's active lane mask and let one lane at a
// time issue the doorbell MMIO store. Ionic's dbrAddr is shared across every
// QP of the same ibv_context; multiple lanes of one warp storing to that
// shared address in one SIMT instruction get coalesced into a single
// transaction and only one lane's dbrVal survives. Atomic-store ordering
// does not protect against this. MLX5/BNXT each have a per-QP dbrAddr so
// multi-lane stores hit distinct addresses and stay on the fast path.
__device__ inline static void ringDoorbellWarpPsd(void* dbrAddr, uint64_t dbrVal) {
  uint64_t mask = core::GetActiveLaneMask();
  while (mask) {
    int lane = __ffsll(static_cast<unsigned long long>(mask)) - 1;
    if (__lane_id() == lane) {
      core::RingDoorbell<core::ProviderType::PSD>(dbrAddr, dbrVal);
    }
    __syncwarp();
    mask &= ~(1ull << lane);
  }
}

// Wait for doorbell ordering and ring doorbell
template <core::ProviderType PrvdType>
__device__ inline static void ringDoorbellOrdered(application::RdmaEndpointDevice* ep,
                                                  uint32_t myPostIdx, uint32_t numWqes,
                                                  uint64_t dbrVal) {
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

  if constexpr (PrvdType == core::ProviderType::PSD) {
    // PSD/Ionic: shared dbrAddr, lane-serialize to avoid SIMT same-address
    // store coalescing dropping doorbells.
    ringDoorbellWarpPsd(wq->dbrAddr, dbrVal);
  } else if constexpr (PrvdType == core::ProviderType::MLX5) {
    // MLX5: must update DBR record before ringing doorbell
    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, myPostIdx + numWqes);
    __threadfence_system();
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    // BNXT: update DBR record, then ring doorbell. BNXT dedups the UAR page
    // across endpoints (see BnxtCqContainer / TryRegisterUar), so distinct QPs
    // in one warp can share the same dbrAddr. A per-thread GDA put has each
    // active lane targeting a *different* peer's QP in the same SIMT step; if
    // those QPs share a UAR, same-address store coalescing drops all but one
    // lane's doorbell — the dropped WQE is never fetched and quiet hangs. Walk
    // the warp's active lanes one at a time so every doorbell store survives.
    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, myPostIdx + numWqes);
    __threadfence_system();
    uint64_t mask = core::GetActiveLaneMask();
    while (mask) {
      int lane = __ffsll(static_cast<unsigned long long>(mask)) - 1;
      if (__lane_id() == lane) {
        core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
      }
      __syncwarp();
      mask &= ~(1ull << lane);
    }
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
        reinterpret_cast<uintptr_t>(wq->sqAddr) + (lastWqeIdx << MLX5_SEND_WQE_SHIFT);
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

// putImpl - Pure hardware operation layer
template <core::ProviderType PrvdType, typename Coop = ccoCoopThread>
__device__ inline static void putImpl(
    // Hardware resources (already selected endpoint)
    application::RdmaEndpointDevice* ep, uint32_t qpn,

    // Data transfer parameters (already parsed addresses and keys)
    uintptr_t localAddr, uint32_t localKey,    // local buffer
    uintptr_t remoteAddr, uint32_t remoteKey,  // remote buffer
    size_t bytes,

    // Signal parameters (already parsed)
    bool hasSignal, uintptr_t signalRemoteAddr, uint32_t signalRemoteKey, ccoGdaSignalOp_t signalOp,
    uint64_t signalOpArg,

    // Optimization flags
    uint32_t optFlags = ccoGdaOptFlagsDefault) {
  // Get work queue handle
  core::WorkQueueHandle* wq = &ep->wqHandle;

  // Calculate total WQEs needed (always 1 for the data write)
  uint32_t numWqesNeeded = 1;
  if (hasSignal) {
    numWqesNeeded += getAtomicWqeCount<PrvdType>(core::AMO_FETCH_ADD, sizeof(uint64_t));
  }

  // Reserve WQE slots (with flow control)
  uint32_t curPostIdx = reserveWqeSlots<PrvdType, Coop>(ep, numWqesNeeded);

  // Post RDMA Write for data transfer
  uint32_t wqeIdx = curPostIdx;
  recordOutstandingWqe<PrvdType>(wq, wqeIdx);
  uint64_t dbrVal = core::PostWrite<PrvdType>(*wq, wqeIdx, wqeIdx, wqeIdx, true /*cqeSignal*/, qpn,
                                              localAddr, localKey, remoteAddr, remoteKey, bytes);
  wqeIdx++;

  // Post atomic for signal (remote peer notification)
  if (hasSignal) {
    recordOutstandingWqe<PrvdType>(wq, wqeIdx);

    uintptr_t atomicLaddr = reinterpret_cast<uintptr_t>(ep->atomicIbuf.addr);
    uint32_t atomicLkey = ep->atomicIbuf.lkey;

    dbrVal = core::PostAtomic<PrvdType, uint64_t>(
        *wq, wqeIdx, wqeIdx, wqeIdx, true /*cqeSignal*/, qpn, atomicLaddr, atomicLkey,
        signalRemoteAddr, signalRemoteKey, signalOpArg, 0 /*compare*/, core::AMO_FETCH_ADD);
  }

  // Ring doorbell (ordered) unless AggregateRequests is set
  if (!(optFlags & ccoGdaOptFlagsAggregateRequests)) {
    ringDoorbellOrdered<PrvdType>(ep, curPostIdx, numWqesNeeded, dbrVal);
  }
}

// New putValueImpl - Inline write for small values
template <core::ProviderType PrvdType, typename T>
__device__ inline static void putValueImpl(application::RdmaEndpointDevice* ep, uint32_t qpn,
                                           uintptr_t remoteAddr, uint32_t remoteKey, T value,
                                           bool hasSignal, uintptr_t signalRemoteAddr,
                                           uint32_t signalRemoteKey, ccoGdaSignalOp_t signalOp,
                                           uint64_t signalOpArg,
                                           uint32_t optFlags = ccoGdaOptFlagsDefault) {
  static_assert(sizeof(T) <= 8, "putValue only supports types <= 8 bytes");

  core::WorkQueueHandle* wq = &ep->wqHandle;

  // Calculate WQEs needed
  uint32_t numWqesNeeded = 1;
  if (hasSignal) {
    numWqesNeeded += getAtomicWqeCount<PrvdType>(core::AMO_FETCH_ADD, sizeof(uint64_t));
  }

  // Reserve WQE slots
  uint32_t curPostIdx = reserveWqeSlots<PrvdType>(ep, numWqesNeeded);

  // Post inline write
  uint32_t wqeIdx = curPostIdx;
  recordOutstandingWqe<PrvdType>(wq, wqeIdx);
  uint64_t dbrVal = core::PostWriteInline<PrvdType>(*wq, wqeIdx, wqeIdx, wqeIdx, true /*cqeSignal*/,
                                                    qpn, &value, remoteAddr, remoteKey, sizeof(T));
  wqeIdx++;

  // Post atomic for signal if requested
  if (hasSignal) {
    recordOutstandingWqe<PrvdType>(wq, wqeIdx);

    uintptr_t atomicLaddr = reinterpret_cast<uintptr_t>(ep->atomicIbuf.addr);
    uint32_t atomicLkey = ep->atomicIbuf.lkey;

    dbrVal = core::PostAtomic<PrvdType, uint64_t>(
        *wq, wqeIdx, wqeIdx, wqeIdx, true /*cqeSignal*/, qpn, atomicLaddr, atomicLkey,
        signalRemoteAddr, signalRemoteKey, signalOpArg, 0, core::AMO_FETCH_ADD);
  }

  // Ring doorbell unless AggregateRequests is set
  if (!(optFlags & ccoGdaOptFlagsAggregateRequests)) {
    ringDoorbellOrdered<PrvdType>(ep, curPostIdx, numWqesNeeded, dbrVal);
  }
}

// New getImpl - RDMA read
template <core::ProviderType PrvdType, typename Coop = ccoCoopThread>
__device__ inline static void getImpl(application::RdmaEndpointDevice* ep, uint32_t qpn,
                                      uintptr_t localAddr, uint32_t localKey, uintptr_t remoteAddr,
                                      uint32_t remoteKey, size_t bytes,
                                      uint32_t optFlags = ccoGdaOptFlagsDefault) {
  core::WorkQueueHandle* wq = &ep->wqHandle;

  // Reserve WQE slot
  uint32_t curPostIdx = reserveWqeSlots<PrvdType, Coop>(ep, 1);

  // Post RDMA Read
  recordOutstandingWqe<PrvdType>(wq, curPostIdx);
  uint64_t dbrVal =
      core::PostRead<PrvdType>(*wq, curPostIdx, curPostIdx, curPostIdx, true /*cqeSignal*/, qpn,
                               localAddr, localKey, remoteAddr, remoteKey, bytes);

  // Ring doorbell unless AggregateRequests is set
  if (!(optFlags & ccoGdaOptFlagsAggregateRequests)) {
    ringDoorbellOrdered<PrvdType>(ep, curPostIdx, 1, dbrVal);
  }
}

// FlushAsync: ring doorbell for pending WQEs (skip if already rung),
// return the postIdx for later wait.
template <core::ProviderType PrvdType>
__device__ inline static void flushAsyncImpl(application::RdmaEndpointDevice* ep, uint32_t qpn,
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

  if constexpr (PrvdType == core::ProviderType::PSD) {
    ringDoorbellWarpPsd(wq->dbrAddr, dbrVal);
  } else {
    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, curPostIdx);
    __threadfence_system();
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
  }

  __threadfence_system();

  __hip_atomic_fetch_add(&cq->needConsIdx, numPendingWqes, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_store(&wq->dbTouchIdx, static_cast<uint64_t>(curPostIdx), __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_AGENT);
}

// Wait: wait for async request to complete
template <core::ProviderType PrvdType>
__device__ inline static void waitImpl(application::RdmaEndpointDevice* ep, uint32_t postIdx) {
  quietUntil<PrvdType>(ep, postIdx);
}

// Signal: send signal to remote peer (RDMA atomic increment/add)
template <core::ProviderType PrvdType>
__device__ inline static void signalImpl(application::RdmaEndpointDevice* ep, uint32_t qpn,
                                         uintptr_t signalRemoteAddr, uint32_t signalRemoteKey,
                                         ccoGdaSignalOp_t signalOp, uint64_t signalOpArg,
                                         uint32_t optFlags = ccoGdaOptFlagsDefault) {
  core::WorkQueueHandle* wq = &ep->wqHandle;

  // Reserve WQE slot
  uint32_t curPostIdx = reserveWqeSlots<PrvdType>(ep, 1);

  // Post RDMA atomic operation
  recordOutstandingWqe<PrvdType>(wq, curPostIdx);

  // RDMA atomic requires local buffer for FetchAdd result (even if unused)
  uintptr_t atomicLaddr = reinterpret_cast<uintptr_t>(ep->atomicIbuf.addr);
  uint32_t atomicLkey = ep->atomicIbuf.lkey;

  uint64_t addValue = (signalOp == ccoGdaSignalInc) ? 1 : signalOpArg;
  uint64_t dbrVal = core::PostAtomic<PrvdType, uint64_t>(
      *wq, curPostIdx, curPostIdx, curPostIdx, true /*cqeSignal*/, qpn, atomicLaddr, atomicLkey,
      signalRemoteAddr, signalRemoteKey, addValue, 0 /*compare*/, core::AMO_FETCH_ADD);

  // Ring doorbell unless AggregateRequests is set
  if (!(optFlags & ccoGdaOptFlagsAggregateRequests)) {
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
template <ccoTeamMode TeamMode, typename RemoteAction, typename Coop>
__device__ inline void ccoGda<PrvdType>::put(int peer, ccoWindow_t dstWin, size_t dstOffset,
                                             ccoWindow_t srcWin, size_t srcOffset, size_t bytes,
                                             RemoteAction remoteAction, Coop coop,
                                             uint32_t optFlags) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    int worldPeer = resolveWorldPeer<TeamMode>(peer);

    // step 1: parse windows to extract lkey/rkey
    ccoWindowDevice* dstWinDev = reinterpret_cast<ccoWindowDevice*>(dstWin);
    ccoWindowDevice* srcWinDev = reinterpret_cast<ccoWindowDevice*>(srcWin);

    uint32_t srcLkey = srcWinDev->ibgdaWin.lkey;
    uint32_t dstRkey = dstWinDev->ibgdaWin.peerRkeys[worldPeer];

    uintptr_t localAddr = srcOffset;
    uintptr_t remoteAddr = dstOffset;

    // step 2: select endpoint (world-indexed endpoints + contextId)
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    application::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
    uint32_t qpn = ep->qpn;

    // step 3: parse RemoteAction -> signal parameters
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

    // step 4: call primitive API (PrvdType is compile-time determined)
    impl::putImpl<PrvdType, Coop>(ep, qpn,
                                  localAddr, srcLkey,   // local
                                  remoteAddr, dstRkey,  // remote
                                  bytes, hasSignal, signalRaddr, signalRkey, signalOp, signalOpArg,
                                  optFlags);
  }
  coop.sync();
}

// putValue: write immediate value (≤8 bytes)
template <core::ProviderType PrvdType>
template <ccoTeamMode TeamMode, typename T, typename RemoteAction, typename Coop>
__device__ inline void ccoGda<PrvdType>::putValue(int peer, ccoWindow_t dstWin, size_t dstOffset,
                                                  T value, RemoteAction remoteAction, Coop coop,
                                                  uint32_t optFlags) {
  static_assert(sizeof(T) <= 8, "putValue only supports types <= 8 bytes");

  coop.sync();
  if (coop.thread_rank() == 0) {
    int worldPeer = resolveWorldPeer<TeamMode>(peer);

    // step 1: parse window to extract rkey
    ccoWindowDevice* dstWinDev = reinterpret_cast<ccoWindowDevice*>(dstWin);
    uint32_t dstRkey = dstWinDev->ibgdaWin.peerRkeys[worldPeer];
    uintptr_t remoteAddr = dstOffset;

    // step 2: select endpoint
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    application::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
    uint32_t qpn = ep->qpn;

    // step 3: parse RemoteAction
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

    // step 4: call primitive API
    impl::putValueImpl<PrvdType, T>(ep, qpn, remoteAddr, dstRkey, value, hasSignal, signalRaddr,
                                    signalRkey, signalOp, signalOpArg, optFlags);
  }
  coop.sync();
}

// get: RDMA read
template <core::ProviderType PrvdType>
template <ccoTeamMode TeamMode, typename Coop>
__device__ inline void ccoGda<PrvdType>::get(int peer, ccoWindow_t remoteWin, size_t remoteOffset,
                                             ccoWindow_t localWin, size_t localOffset, size_t bytes,
                                             Coop coop, uint32_t optFlags) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    int worldPeer = resolveWorldPeer<TeamMode>(peer);

    // step 1: parse windows
    ccoWindowDevice* remoteWinDev = reinterpret_cast<ccoWindowDevice*>(remoteWin);
    ccoWindowDevice* localWinDev = reinterpret_cast<ccoWindowDevice*>(localWin);

    uint32_t remoteRkey = remoteWinDev->ibgdaWin.peerRkeys[worldPeer];
    uint32_t localLkey = localWinDev->ibgdaWin.lkey;

    uintptr_t remoteAddr = remoteOffset;
    uintptr_t localAddr = localOffset;

    // step 2: select endpoint
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    application::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
    uint32_t qpn = ep->qpn;

    // step 3: call primitive API
    impl::getImpl<PrvdType, Coop>(ep, qpn, localAddr, localLkey, remoteAddr, remoteRkey, bytes,
                                  optFlags);
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
    application::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
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
    application::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
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
    application::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
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
    application::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];

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
    application::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
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
    application::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];

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
