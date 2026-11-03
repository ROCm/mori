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
// The sole consumer of the RDMA core, split from cco.hpp so host-only and
// LSA-only TUs can include cco.hpp without the heavy provider RDMA headers.
// Includes cco.hpp and adds the GDA device layer (ccoGda<PrvdType> + the
// provider-specialized primitives in mori::cco::impl). Include this (not
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

// Template argument of put / putValue / get (the last one), so an unused flag
// costs nothing. A flag the implementation does not support is a compile error.
enum ccoGdaOptFlags : uint32_t {
  ccoGdaOptFlagsDefault = 0,
  // Not implemented: the SQ space check always runs. Rejected at compile time.
  ccoGdaOptFlagsMaySkipCreditCheck = (1 << 0),
  // Write the WQEs and mark them ready, but do not ring the doorbell. A later
  // non-deferred post, signal, flush, flushAsync or counter on the same QP rings
  // for them. Until one of those runs, the NIC has not been told about them.
  // Pays when one warp posts many WQEs to a QP and releases them once; for one
  // or two per release, or many warps on one QP, plain posts cost the same and
  // start the transfer sooner.
  ccoGdaOptFlagsDeferDoorbell = (1 << 1),
  // Caller-ordered posting. On a deferred put / putValue / get: write the WQEs
  // and publish nothing (no ready mark, no system fence). On flush(peer) /
  // flushAsync: ring once straight up to the postIdx snapshot instead of
  // sweeping ready marks. The caller guarantees that (a) every WQE below a flush's
  // snapshot has been completely written by the time the flush starts (e.g. each
  // poster bumps a counter after its post returns and the flusher waits for it),
  // (b) every post to the QP uses this flag, and (c) the WQEs posted between two
  // flushes fit in the SQ (nothing rings for them, so a full SQ never drains).
  // Concurrent posters are fine; concurrent flushers on one QP serialize on the
  // doorbell lock. This is the pre-#625 deferred doorbell with the race moved to
  // the caller's contract.
  ccoGdaOptFlagsPostedBeforeFlush = (1 << 2),
};
inline constexpr uint32_t kCcoGdaSupportedOptFlags =
    ccoGdaOptFlagsDeferDoorbell | ccoGdaOptFlagsPostedBeforeFlush;
// flush(peer) / flushAsync take only this one.
inline constexpr uint32_t kCcoGdaSupportedFlushOptFlags = ccoGdaOptFlagsPostedBeforeFlush;

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

// Atomic add at an arbitrary offset in a caller-owned window, rather than at a
// signal slot in the DevComm's resource window.
//
// ccoGda_SignalInc / ccoGda_SignalAdd cannot express that: they resolve to
// signalId * sizeof(uint64_t) in the DevComm's own resource window.
//
// The remote op is identical (a NIC atomic add); only the target resolution
// differs, so this rides the same single-reservation path as ccoGda_SignalAdd --
// fused into `put`'s own doorbell when passed to put, no extra WQE. `offset` is
// window-relative with iova=0, the same convention as put's dstOffset.
//
// waitSignal/readSignal/resetSignal do NOT see these: the target is the caller's
// window, so the caller polls it directly.
struct ccoGda_WindowSignalAdd {
  ccoWindow_t win;
  size_t offset;
  uint64_t value;
  __device__ inline ccoGda_WindowSignalAdd(ccoWindow_t w, size_t off, uint64_t val)
      : win(w), offset(off), value(val) {}
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
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD,
            ccoGdaThreadMode ThreadMode = ccoGdaThreadIndependent,
            typename RemoteAction = ccoGda_NoSignal, typename Coop = ccoCoopThread,
            uint32_t OptFlags = ccoGdaOptFlagsDefault>
  __device__ inline void put(int peer, ccoWindow_t dstWin, size_t dstOffset, ccoWindow_t srcWin,
                             size_t srcOffset, size_t bytes,
                             RemoteAction remoteAction = ccoGda_NoSignal{}, Coop coop = Coop{});

  // putValue: write an immediate value (≤8 bytes) with optional remote signal.
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD,
            ccoGdaThreadMode ThreadMode = ccoGdaThreadIndependent, typename T,
            typename RemoteAction = ccoGda_NoSignal, typename Coop = ccoCoopThread,
            uint32_t OptFlags = ccoGdaOptFlagsDefault>
  __device__ inline void putValue(int peer, ccoWindow_t dstWin, size_t dstOffset, T value,
                                  RemoteAction remoteAction = ccoGda_NoSignal{},
                                  Coop coop = Coop{});

  // get: rdma read — pull peer's window content into our local window.
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD,
            ccoGdaThreadMode ThreadMode = ccoGdaThreadIndependent, typename Coop = ccoCoopThread,
            uint32_t OptFlags = ccoGdaOptFlagsDefault>
  __device__ inline void get(int peer, ccoWindow_t remoteWin, size_t remoteOffset,
                             ccoWindow_t localWin, size_t localOffset, size_t bytes,
                             Coop coop = Coop{});

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
  // flushAsync snapshots the reservation counter and rings the doorbell up to it,
  // which covers deferred posts; wait polls the CQ until the WQEs up to that
  // snapshot complete.

  // flush: poll CQ for every peer.
  // peers are distributed across the Coop group (default: warp).
  // all threads in the group must call flush together.
  template <typename Coop = ccoCoopWarp>
  __device__ inline void flush(Coop coop = Coop{});

  // flush(peer): poll CQ for a single peer until its submitted WQEs complete.
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD, typename Coop = ccoCoopWarp,
            uint32_t OptFlags = ccoGdaOptFlagsDefault>
  __device__ inline void flush(int peer, Coop coop = Coop{});

  // flushAsync: snapshot this peer's QP progress and return a request handle
  // that wait() can later be used to wait on individually. It rings up to the
  // snapshot, so it may wait for posts on the QP still writing their WQEs; only
  // the CQ drain is left to wait(). The request is valid in the coop leader.
  template <ccoTeamMode TeamMode = CCO_TEAM_WORLD, typename Coop = ccoCoopThread,
            uint32_t OptFlags = ccoGdaOptFlagsDefault>
  __device__ inline void flushAsync(int peer, ccoGdaRequest_t* outRequest, Coop coop = Coop{});

  // wait: block on a request handle previously returned by flushAsync, called
  // with the same coop (the leader's request is the one waited on).
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

// Poll the CQ until wq.doneIdx reaches targetIdx. Collapsed-CQ model:
// reconstruct the completed WQE count directly from the CQE counter (no
// outstandingWqe[] table); exits at the caller's targetIdx.
template <core::ProviderType PrvdType>
__device__ inline static void quietUntil(core::RdmaEndpointDevice* ep, uint32_t targetIdx) {
  core::WorkQueueHandle* wq = &ep->wqHandle;
  core::CompletionQueueHandle* cq = &ep->cqHandle;

  if constexpr (PrvdType == core::ProviderType::PSD) {
    constexpr uint32_t PENDING_WORK_MASK = 0x800000;
    constexpr uint32_t MSN_MASK = 0xFFFFFF;  // ionic reports a 24-bit MSN
#ifdef IONIC_CCQE
    // CCQE: cqeNum==1, NIC overwrites CQE[0] with latest MSN.
    volatile ionic_v1_cqe* cqe = reinterpret_cast<volatile ionic_v1_cqe*>(cq->cqAddr);
    while ((wq->doneIdx - targetIdx) & PENDING_WORK_MASK) {
      uint32_t msn = BE32TOH(*(volatile uint32_t*)(&cqe->send.msg_msn));
      asm volatile("" ::: "memory");
      if (!((msn - targetIdx) & PENDING_WORK_MASK)) {
        // Widen the 24-bit MSN into doneIdx's 32-bit serial space: advance by the
        // masked forward delta instead of storing msn raw, else doneIdx ends up in
        // a different modulus than dbTouchIdx and the in-flight count goes bad past
        // 2^24. Never advance beyond what was doorbelled -- a WQE the NIC was never
        // told about cannot have completed, and skipping over it recycles live SQ
        // slots.
        // doneIdx is read once into `cur` and written once, so concurrent pollers
        // computing off the same CQE all land on the same value, as they did when
        // this was a bare `doneIdx = msn`. A read-modify-write here could reload
        // between the two and apply delta twice, running doneIdx past the NIC.
        uint32_t dbTouched =
            __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        uint32_t cur = wq->doneIdx;
        uint32_t delta = (msn - cur) & MSN_MASK;
        if (delta != 0 && static_cast<int32_t>(delta) <= static_cast<int32_t>(dbTouched - cur)) {
          wq->doneIdx = cur + delta;
        }
      }
    }
#else
    // Non-CCQE: warp-parallel poll with color bit alternation.
    //
    // The poll below is only correct when every participating lane shares one
    // CQ: the pollCqLock is acquired by the first active lane and broadcast to
    // the rest, each lane derives its CQ slot from its index within the active
    // mask, and a ballot elects a single lane to commit cq_consumer/doneIdx.
    // Callers can legitimately arrive with lanes on different endpoints (e.g.
    // flush() fans peers across the cooperative group), so group the active
    // lanes by CQ and give each distinct CQ its own turn.
    const int myLaneId = core::WarpLaneId();
    constexpr uint32_t MAX_GREED = 10;
    constexpr uint32_t CQ_DOORBELL_GRACE = 100;
    const unsigned long long myCqKey = reinterpret_cast<unsigned long long>(cq);

    bool needTurn = true;
    for (uint64_t turns = __ballot(needTurn); turns != 0; turns = __ballot(needTurn)) {
      const int lead = __ffsll(static_cast<unsigned long long>(turns)) - 1;
      if (myCqKey != __shfl(myCqKey, lead)) continue;
      needTurn = false;

      const uint64_t activeMask = core::GetActiveLaneMask();
      const uint32_t myLogicalLaneId = core::GetActiveLaneNum(activeMask);
      uint32_t wqeCounter;

      // The poll needs every group lane until the end, so wait for the group's
      // furthest target (thread-scope wait passes one per lane).
      uint32_t target = __shfl(targetIdx, lead);
      for (uint64_t rest = activeMask; rest != 0; rest &= rest - 1) {
        uint32_t t = __shfl(targetIdx, __ffsll(static_cast<unsigned long long>(rest)) - 1);
        if (static_cast<int32_t>(t - target) > 0) target = t;
      }

      while ((wq->doneIdx - target) & PENDING_WORK_MASK) {
        if (!core::spin_lock_try_acquire_shared(&cq->pollCqLock, activeMask)) continue;
        uint32_t greedRemaining = MAX_GREED;
        while ((wq->doneIdx - target) & PENDING_WORK_MASK) {
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
            // Widen the 24-bit MSN into doneIdx's 32-bit serial space; see the CCQE
            // branch above for why the raw store is wrong and why dbTouchIdx caps it.
            uint32_t dbTouched =
                __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            uint32_t cur = wq->doneIdx;
            uint32_t delta = (wqeCounter - cur) & MSN_MASK;

            // Decide before consuming: the doorbell is rung before dbTouchIdx is
            // published (ringAndPublish), so a live MSN can be ahead of the cap, and the
            // color bit is keyed on cq_consumer -- moving it first would drop that
            // completion for good. Left in place, a later pass takes it.
            bool aheadOfDoorbell =
                delta != 0 && static_cast<int32_t>(delta) > static_cast<int32_t>(dbTouched - cur);
            if (!aheadOfDoorbell) {
              if (delta != 0) wq->doneIdx = cur + delta;

              cq->cq_consumer = myCqPos + 1;
              if (((cq->cq_consumer - cq->cq_dbpos) & (cq->cqeNum - 1)) >= CQ_DOORBELL_GRACE) {
                cq->cq_dbpos = cq->cq_consumer;
                core::UpdateCqDbrRecord<core::ProviderType::PSD>(*cq, myCqPos + 1);
              }
            }
          }
          if (!((wq->doneIdx - target) & PENDING_WORK_MASK)) {
            if (wq->doneIdx == oldDoneIdx) break;
            if (greedRemaining == 0) break;
            --greedRemaining;
          }
        }
        core::spin_lock_release_shared(&cq->pollCqLock, activeMask);
        break;
      }
    }
#endif
  } else if constexpr (PrvdType == core::ProviderType::MLX5) {
    // MLX5: collapsed CQ — read CQE[0] (volatile), reconstruct the 16-bit
    // wqe_counter against doneIdx, advance via max.
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
      // Rebuild the 32-bit completion from the 16-bit wqe_counter via the forward
      // delta, accepted only within (cons, dbTouchIdx] to drop stale CQEs. window
      // is signed: once doneIdx reaches dbTouchIdx it is <= 0 and rejects all.
      uint16_t comp16 = static_cast<uint16_t>(wqeCounter + 1);
      uint16_t delta = static_cast<uint16_t>(comp16 - static_cast<uint16_t>(cons));
      uint32_t dbTouched =
          __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      int32_t window = static_cast<int32_t>(dbTouched - cons);
      uint32_t completed = cons;
      if (delta != 0 && static_cast<int32_t>(delta) <= window) {
        completed = cons + delta;
      }
      core::AtomicMaxSerial(&wq->doneIdx, completed);
      asm volatile("" ::: "memory");
    }
    __threadfence();
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    // BNXT: collapsed CQ (cqeNum==1) — single poller (pollCqLock); others spin
    // re-reading doneIdx (the holder advances it). Reconstruct the completed count
    // from CQE con_indx against dbTouchIdx, advance doneIdx via max. Non-blocking
    // PollCqOnce (cco flow-control may wait on slots not yet doorbelled, so a
    // blocking poll would deadlock).
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
        // con_indx is the consumed count mod sqWqeNum: rebuild the serial as a
        // forward delta from doneIdx, as the mlx5 drain does with wqe_counter.
        // Anchoring on dbTouchIdx instead (largest congruent V <= dbTouchIdx, minus
        // sqWqeNum when it overshot) went a whole queue depth *backwards* whenever
        // the CQE ran ahead of dbTouchIdx -- routine, since the doorbell is rung
        // before dbTouchIdx is published -- and as a raw word that value won the
        // atomic max this used to be and pinned doneIdx for good. Issue #626.
        // Flow control never leaves sqWqeNum WQEs outstanding, so delta == 0 is
        // unambiguously "nothing new" rather than a full queue.
        uint32_t cur = __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        uint32_t dbTouch =
            __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        uint32_t delta = (wqeCounter - cur) & mask;
        // Still capped at what was doorbelled: con_indx alone cannot tell a live
        // completion from a stale one. Refusing one costs nothing -- PollSingleCqe is
        // level-triggered for cqeNum == 1, so the next pass re-reads the same value.
        if (delta != 0 && static_cast<int32_t>(delta) <= static_cast<int32_t>(dbTouch - cur)) {
          // Sole writer under pollCqLock, and delta > 0 is already an advance.
          __hip_atomic_store(&wq->doneIdx, cur + delta, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
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

// ── Doorbell coalescing ──────────────────────────────────────────────────────
//
// A post reserves SQ slots (postIdx) and writes its WQEs. If it is first in line
// (dbTouchIdx == its first slot) it rings for itself, as every post used to.
// Otherwise it publishes a ready mark in wq->readyMarks at its first slot:
// tag = (first index << 32) | count, plus the doorbell value its last WQE
// produced. Whoever holds the QP's doorbell lock (wq->postSendLock) walks the
// marks forward from dbTouchIdx, rings ONCE for the whole contiguous ready run,
// and publishes the new dbTouchIdx. A post is done when dbTouchIdx has passed
// its last WQE, whoever rang.
//
// So a post behind others does not wait for each of them to ring in turn, a
// burst of posts on one QP shares doorbells, and a post may skip ringing
// entirely (ccoGdaOptFlagsDeferDoorbell): the next post or flush to sweep picks
// it up.
//
// dbTouchIdx keeps its meaning, "the NIC has been told about every WQE below
// this", which quietUntil's MSN cap and the SQ space check rely on. The sweep
// only crosses a slot whose mark carries that slot's own index, so it never rings
// for a WQE that is reserved but not yet written -- the bug that removed the old
// deferred doorbell (#625). Every post rewrites the tag at its first slot -- its
// mark, or count 0 on the fast path -- so a stale tag carries the index of the
// last post that started there, and a never-written one has count 0. A stale tag
// matches only if no post started at that slot for exactly a multiple of 2^32
// WQEs. (Without the fast-path write, any run of 2^32 fast-path posts would do.)

__device__ inline static uint64_t readyTag(uint32_t base, uint32_t count) {
  return (static_cast<uint64_t>(base) << 32) | count;
}

// Memory ordering here follows the rest of this file: relaxed agent-scope atomics
// (coherent across XCDs without flushing L2) ordered by explicit system fences.
// Acquire/release at agent scope would add an L2 invalidate or write-back to
// every post, which measured at ~2 us on MI308X.

// Publish [base, base + count) as written. Called by one lane per post, after
// every lane's WQE stores: the system fence makes them, and dbrVal, visible to
// the NIC and to other waves before the tag is.
__device__ inline static void markWqesReady(core::WorkQueueHandle* wq, uint32_t base,
                                            uint32_t count, uint64_t dbrVal) {
  core::WqeReadyMark* mark = &wq->readyMarks[base & (wq->sqWqeNum - 1)];
  __hip_atomic_store(&mark->dbrVal, dbrVal, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __threadfence_system();
  __hip_atomic_store(&mark->tag, readyTag(base, count), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// The WQE count of the post at `idx` if it has published its mark, else 0.
__device__ inline static uint32_t readyCount(core::WorkQueueHandle* wq, uint32_t idx) {
  uint64_t tag = __hip_atomic_load(&wq->readyMarks[idx & (wq->sqWqeNum - 1)].tag, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
  return static_cast<uint32_t>(tag >> 32) == idx ? static_cast<uint32_t>(tag) : 0;
}

// Walk the active lane mask, ringing one lane's doorbell at a time. Needed by
// ALL providers when several lanes ring different QPs together: PSD/BNXT share
// one dbrAddr across QPs (Ionic per ibv_context; BNXT per UAR page), and MLX5 —
// despite a per-QP dbrAddr — still loses doorbell stores when multiple lanes
// issue them in a single SIMT step (the stores coalesce and only one survives,
// leaving the other QPs' WQEs unfetched → hang). Serializing per lane avoids it.
// No __syncwarp: wavefronts are lock-step (predication already orders the stores)
// and a __syncwarp would deadlock on divergent entry.
template <core::ProviderType PrvdType>
__device__ inline static void ringDoorbellWalk(core::WorkQueueHandle* wq, uint32_t dbrRecVal,
                                               uint64_t dbrVal) {
  if constexpr (PrvdType == core::ProviderType::BNXT) {
    // BNXT: single shared DBR record for the UAR page → update once up front.
    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, dbrRecVal);
    __threadfence_system();
  }
  uint64_t mask = core::GetActiveLaneMask();
  while (mask) {
    int lane = __ffsll(static_cast<unsigned long long>(mask)) - 1;
    if (__lane_id() == lane) {
      // MLX5 keeps a *per-QP* DBR record, so each lane must publish its own
      // producer index right before its doorbell store (BNXT did this once
      // above; PSD has no separate DBR-record write on this path).
      if constexpr (PrvdType == core::ProviderType::MLX5) {
        core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, dbrRecVal);
        __threadfence_system();
      }
      core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
    }
    mask &= ~(1ull << lane);
  }
}

__device__ inline static bool tryLockDoorbell(core::WorkQueueHandle* wq) {
  uint32_t unlocked = 0;
  return __hip_atomic_compare_exchange_strong(&wq->postSendLock, &unlocked, 1u, __ATOMIC_RELAXED,
                                              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// The next holder must see the dbTouchIdx this one published. That store is a
// coherent (agent-scope atomic) store, so waiting for it to complete is enough;
// a release here would also write back L2.
__device__ inline static void unlockDoorbell(core::WorkQueueHandle* wq) {
#if defined(__GFX12__)
  __asm__ volatile("s_wait_storecnt 0x0" ::: "memory");  // gfx12 has no s_waitcnt
#else
  __builtin_amdgcn_s_waitcnt(0);
#endif
  __hip_atomic_store(&wq->postSendLock, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// The right to ring from dbTouchIdx: held by the doorbell lock, or by the post
// whose first WQE is at dbTouchIdx and which has not marked it (nobody else can
// move dbTouchIdx past an unmarked slot). With it, and the WQEs in [from, to)
// written and visible to the NIC: ring once for them and publish `to`.
template <core::ProviderType PrvdType>
__device__ inline static void ringAndPublish(core::RdmaEndpointDevice* ep, uint32_t from,
                                             uint32_t to, uint64_t dbrVal) {
  core::WorkQueueHandle* wq = &ep->wqHandle;
  ringDoorbellWalk<PrvdType>(wq, to, dbrVal);
  // The doorbell must land before whoever sees the new dbTouchIdx rings a later one.
  __threadfence_system();
  __hip_atomic_fetch_add(&ep->cqHandle.needConsIdx, to - from, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
  __hip_atomic_store(&wq->dbTouchIdx, to, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// Walk the ready marks from `to` and return where the contiguous ready run ends;
// *lastBase is the first index of its last post. Each mark read is a coherent
// load (~1 us on MI308X), so read kScanBatch slots at once and walk them in
// registers: a run of single-WQE posts costs one round trip per batch instead
// of one per post. A slot inside a multi-WQE post holds a stale mark, which is
// harmless; the walk only looks at slots where a post starts.
__device__ inline static uint32_t scanReady(core::WorkQueueHandle* wq, uint32_t to,
                                            uint32_t* lastBase) {
  constexpr uint32_t kScanBatch = 8;
  const uint32_t mask = wq->sqWqeNum - 1;
  while (true) {
    uint64_t tags[kScanBatch];
#pragma unroll
    for (uint32_t i = 0; i < kScanBatch; i++) {
      tags[i] = __hip_atomic_load(&wq->readyMarks[(to + i) & mask].tag, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
    }
    const uint32_t start = to;
#pragma unroll
    for (uint32_t i = 0; i < kScanBatch; i++) {
      if (to - start != i) continue;  // `to` is not at this slot
      if (static_cast<uint32_t>(tags[i] >> 32) != to || static_cast<uint32_t>(tags[i]) == 0) {
        return to;
      }
      *lastBase = to;
      to += static_cast<uint32_t>(tags[i]);
    }
  }
}

// If the post at dbTouchIdx is ready and the doorbell lock is free, ring for the
// contiguous ready run and publish dbTouchIdx. Returns without waiting otherwise:
// either the lock holder will cover the run, or the post at dbTouchIdx is still
// being written and its owner will. Lanes on different QPs may call this
// together; lanes on the same QP contend for its lock and one wins.
//
// No fence before ringing: every marked post fenced its WQEs to system scope
// before publishing its mark, and the doorbell store depends on having read it.
template <core::ProviderType PrvdType>
__device__ inline static void ringReady(core::RdmaEndpointDevice* ep) {
  core::WorkQueueHandle* wq = &ep->wqHandle;
  if (wq->readyMarks == nullptr) return;  // a hand-built endpoint with no SQ (unit tests)
  uint32_t from = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  if (readyCount(wq, from) == 0 || !tryLockDoorbell(wq)) return;
  // Re-read under the lock: the previous holder may have moved it.
  from = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  uint32_t lastBase = from;
  uint32_t to = scanReady(wq, from, &lastBase);
  if (to != from) {
    // The address depends on the marks just read, so this load comes after them.
    uint64_t dbrVal = __hip_atomic_load(&wq->readyMarks[lastBase & (wq->sqWqeNum - 1)].dbrVal,
                                        __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    ringAndPublish<PrvdType>(ep, from, to, dbrVal);
  }
  unlockDoorbell(wq);
}

// Ring until the NIC has been told about every WQE below `target`. Every slot
// below a postIdx snapshot is eventually written and marked by its owner, so
// this terminates whether or not those owners ring themselves.
template <core::ProviderType PrvdType>
__device__ inline static void ringUntil(core::RdmaEndpointDevice* ep, uint32_t target) {
  core::WorkQueueHandle* wq = &ep->wqHandle;
  while (static_cast<int32_t>(
             __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) -
             target) < 0) {
    ringReady<PrvdType>(ep);
  }
}

// Finish a post of `count` WQEs at `base` whose last WQE yielded `dbrVal`: mark
// it ready and, unless deferred, return once a doorbell covers it.
//
// Fast path: a post that is first in line (dbTouchIdx == base) rings for itself
// without a mark or the lock: one system fence before the doorbell, as when every
// post rang its own. It does not look for ready posts behind it -- that read
// costs as much as the doorbell -- they ring for themselves once it publishes.
template <core::ProviderType PrvdType, bool DeferDoorbell, bool NoMark = false>
__device__ inline static void submitWqes(core::RdmaEndpointDevice* ep, uint32_t base,
                                         uint32_t count, uint64_t dbrVal) {
  static_assert(!NoMark || DeferDoorbell, "PostedBeforeFlush requires DeferDoorbell");
  // PostedBeforeFlush: the caller orders this post before the flush that rings it.
  if constexpr (NoMark) return;
  core::WorkQueueHandle* wq = &ep->wqHandle;
  bool rang = false;
  if constexpr (!DeferDoorbell) {
    if (__hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == base) {
      // Retire this slot's old mark (see "Doorbell coalescing"); count 0 never matches.
      __hip_atomic_store(&wq->readyMarks[base & (wq->sqWqeNum - 1)].tag, readyTag(base, 0),
                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __threadfence_system();  // our WQEs, before the doorbell
      ringAndPublish<PrvdType>(ep, base, base + count, dbrVal);
      rang = true;
    }
  }
  if (!rang) markWqesReady(wq, base, count, dbrVal);
  // Lanes of one wave can post to one QP (thread-scope signal): all of them ring
  // or mark before any spins. Convergent, so -O2 cannot fold this back into an
  // if/else that runs the spin while the ringing lane is masked off.
  __builtin_amdgcn_wave_barrier();
  if constexpr (!DeferDoorbell) {
    if (!rang) ringUntil<PrvdType>(ep, base + count);
  }
}

// Wait until the SQ has room for the WQEs up to `end`. Each pass first rings for
// whatever is ready, then drains to dbTouchIdx: waiting on a reservation instead
// could wait on WQEs nobody will ring -- a deferred post's, or our own.
template <core::ProviderType PrvdType>
__device__ inline static void waitSqSpaceUntil(core::RdmaEndpointDevice* ep, uint32_t end) {
  core::WorkQueueHandle* wq = &ep->wqHandle;
  while (true) {
    // All uint32_t: widening to uint64_t leaves `end` wrapping in 32 bits while
    // dbTouched does not, so the difference underflows and this loop spins forever
    // on an empty SQ. Issue #626.
    uint32_t dbTouched =
        __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint32_t dbDone = __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint32_t numActiveSqEntries = dbTouched - dbDone;
    uint32_t numFreeEntries = wq->sqWqeNum - numActiveSqEntries;
    uint32_t entriesUntilEnd = end - dbTouched;
    if (numFreeEntries > entriesUntilEnd) break;
    ringReady<PrvdType>(ep);
    quietUntil<PrvdType>(
        ep, __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
  }
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
  waitSqSpaceUntil<PrvdType>(ep, curPostIdx + numWqesNeeded);
  return curPostIdx;
}

// Warp-aggregate flow control: wait until the SQ has room for [base, base+totalWqes).
template <core::ProviderType PrvdType>
__device__ inline static void waitSqSpace(core::RdmaEndpointDevice* ep, uint32_t base,
                                          uint32_t totalWqes) {
  waitSqSpaceUntil<PrvdType>(ep, base + totalWqes);
}

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

// putImpl - post one warp-aggregated put for the active lanes (all targeting this
// ep/qpn; the facade groups by QP). Each lane posts its data WQE into a contiguous
// reservation; the leader posts the shared signal WQE and rings one doorbell.
template <core::ProviderType PrvdType, bool DeferDoorbell = false, bool NoMark = false>
__device__ inline static void putImpl(
    // Hardware resources (already selected endpoint)
    core::RdmaEndpointDevice* ep, uint32_t qpn,

    // Data transfer parameters (already parsed addresses and keys)
    uintptr_t localAddr, uint32_t localKey,    // local buffer
    uintptr_t remoteAddr, uint32_t remoteKey,  // remote buffer
    size_t bytes,

    // Signal parameters (already parsed)
    bool hasSignal, uintptr_t signalRemoteAddr, uint32_t signalRemoteKey, ccoGdaSignalOp_t signalOp,
    uint64_t signalOpArg) {
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
    uint64_t dbrVal =
        core::PostWrite<PrvdType>(*wq, mySlot, mySlot, dataPsn, true /*cqeSignal*/, qpn, localAddr,
                                  localKey, remoteAddr, remoteKey, bytes);
    __threadfence();
    if (isLeader) {
      if (hasSignal) {
        dbrVal = core::PostAtomic<PrvdType, uint64_t>(
            *wq, signalSlot, signalSlot, sigPsn, true /*cqeSignal*/, qpn, atomicLaddr, atomicLkey,
            signalRemoteAddr, signalRemoteKey, signalOpArg, 0 /*compare*/, core::AMO_FETCH_ADD);
      }
      submitWqes<PrvdType, DeferDoorbell, NoMark>(ep, base, totalWqes, dbrVal);
    }
  } else {
    // MLX5/PSD: the WQE slot index doubles as the PSN.
    waitSqSpace<PrvdType>(ep, base, totalWqes);
    uint64_t dbrVal =
        core::PostWrite<PrvdType>(*wq, mySlot, mySlot, mySlot, true /*cqeSignal*/, qpn, localAddr,
                                  localKey, remoteAddr, remoteKey, bytes);
    __threadfence();
    if (isLeader) {
      if (hasSignal) {
        dbrVal = core::PostAtomic<PrvdType, uint64_t>(
            *wq, signalSlot, signalSlot, signalSlot, true /*cqeSignal*/, qpn, atomicLaddr,
            atomicLkey, signalRemoteAddr, signalRemoteKey, signalOpArg, 0 /*compare*/,
            core::AMO_FETCH_ADD);
      }
      submitWqes<PrvdType, DeferDoorbell, NoMark>(ep, base, totalWqes, dbrVal);
    }
  }
}

// putValueImpl - Inline write for small values. One group per call (the facade
// groups lanes by QP); same warp-aggregate posting as putImpl.
template <core::ProviderType PrvdType, typename T, bool DeferDoorbell = false, bool NoMark = false>
__device__ inline static void putValueImpl(core::RdmaEndpointDevice* ep, uint32_t qpn,
                                           uintptr_t remoteAddr, uint32_t remoteKey, T value,
                                           bool hasSignal, uintptr_t signalRemoteAddr,
                                           uint32_t signalRemoteKey, ccoGdaSignalOp_t signalOp,
                                           uint64_t signalOpArg) {
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
    uint64_t dbrVal =
        core::PostWriteInline<PrvdType>(*wq, mySlot, mySlot, dataPsn, true /*cqeSignal*/, qpn,
                                        &value, remoteAddr, remoteKey, sizeof(T));
    __threadfence();
    if (isLeader) {
      if (hasSignal) {
        dbrVal = core::PostAtomic<PrvdType, uint64_t>(
            *wq, signalSlot, signalSlot, sigPsn, true /*cqeSignal*/, qpn, atomicLaddr, atomicLkey,
            signalRemoteAddr, signalRemoteKey, signalOpArg, 0, core::AMO_FETCH_ADD);
      }
      submitWqes<PrvdType, DeferDoorbell, NoMark>(ep, base, totalWqes, dbrVal);
    }
  } else {
    // MLX5/PSD: the WQE slot index doubles as the PSN.
    waitSqSpace<PrvdType>(ep, base, totalWqes);
    uint64_t dbrVal =
        core::PostWriteInline<PrvdType>(*wq, mySlot, mySlot, mySlot, true /*cqeSignal*/, qpn,
                                        &value, remoteAddr, remoteKey, sizeof(T));
    __threadfence();
    if (isLeader) {
      if (hasSignal) {
        dbrVal = core::PostAtomic<PrvdType, uint64_t>(
            *wq, signalSlot, signalSlot, signalSlot, true /*cqeSignal*/, qpn, atomicLaddr,
            atomicLkey, signalRemoteAddr, signalRemoteKey, signalOpArg, 0, core::AMO_FETCH_ADD);
      }
      submitWqes<PrvdType, DeferDoorbell, NoMark>(ep, base, totalWqes, dbrVal);
    }
  }
}

// getImpl - RDMA read. One group per call (the facade groups lanes by QP):
// each active lane posts its read WQE into a contiguous reservation, the leader
// rings one doorbell.
template <core::ProviderType PrvdType, bool DeferDoorbell = false, bool NoMark = false>
__device__ inline static void getImpl(core::RdmaEndpointDevice* ep, uint32_t qpn,
                                      uintptr_t localAddr, uint32_t localKey, uintptr_t remoteAddr,
                                      uint32_t remoteKey, size_t bytes) {
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
    uint64_t dbrVal =
        core::PostRead<PrvdType>(*wq, mySlot, mySlot, dataPsn, true /*cqeSignal*/, qpn, localAddr,
                                 localKey, remoteAddr, remoteKey, bytes);
    __threadfence();
    if (isLeader) submitWqes<PrvdType, DeferDoorbell, NoMark>(ep, base, totalWqes, dbrVal);
  } else {
    // MLX5/PSD: the WQE slot index doubles as the PSN.
    waitSqSpace<PrvdType>(ep, base, totalWqes);
    uint64_t dbrVal = core::PostRead<PrvdType>(*wq, mySlot, mySlot, mySlot, true /*cqeSignal*/, qpn,
                                               localAddr, localKey, remoteAddr, remoteKey, bytes);
    __threadfence();
    if (isLeader) submitWqes<PrvdType, DeferDoorbell, NoMark>(ep, base, totalWqes, dbrVal);
  }
}

// FlushAsync: snapshot the reservation counter and make sure the NIC has been
// told about every WQE below it, deferred ones included, so a later wait() can
// drain to it.
//
// postIdx is a *reservation* counter: a snapshot routinely covers slots whose
// owner is still writing them. ringUntil never rings past a slot that is not
// marked ready, so it just waits for those owners; every reserved slot does get
// written, so it terminates.
template <core::ProviderType PrvdType>
__device__ inline static void flushAsyncImpl(core::RdmaEndpointDevice* ep, uint32_t* outPostIdx) {
  core::WorkQueueHandle* wq = &ep->wqHandle;

  uint32_t postIdx = __hip_atomic_load(&wq->postIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  *outPostIdx = postIdx;

  // Keep flush's release semantics: prior local writes must be visible before the
  // caller treats the transfer as handed off.
  __threadfence_system();
  ringUntil<PrvdType>(ep, postIdx);
}

// The doorbell value that tells the NIC about every WQE below postIdx (the
// pre-#625 flush computed it the same way).
template <core::ProviderType PrvdType>
__device__ inline static uint64_t buildFlushDbrVal(core::WorkQueueHandle* wq, uint32_t postIdx,
                                                   uint32_t qpn) {
  uint32_t lastWqeIdx = (postIdx - 1) & (wq->sqWqeNum - 1);
  if constexpr (PrvdType == core::ProviderType::PSD) {
    return wq->sq_dbval | (postIdx & (wq->sqWqeNum - 1));
  } else if constexpr (PrvdType == core::ProviderType::MLX5) {
    uintptr_t wqeAddr =
        reinterpret_cast<uintptr_t>(wq->sqAddr) + (lastWqeIdx << core::MORI_MLX5_SEND_WQE_SHIFT);
    return *reinterpret_cast<volatile uint64_t*>(wqeAddr);
  } else {
    uint8_t flags = (postIdx >> (__ffs(wq->sqWqeNum) - 1)) & 0x1;
    uint32_t epoch = (flags & BNXT_RE_FLAG_EPOCH_TAIL_MASK) << BNXT_RE_DB_EPOCH_TAIL_SHIFT;
    return core::bnxt_re_init_db_hdr(
        ((postIdx & (wq->sqWqeNum - 1)) * BNXT_RE_NUM_SLOT_PER_WQE) | epoch, 0, qpn,
        BNXT_RE_QUE_TYPE_SQ);
  }
}

// flushAsync under ccoGdaOptFlagsPostedBeforeFlush: every WQE below the snapshot
// is written (caller's contract) and none is marked, so ring once straight up to
// it under the doorbell lock, which orders us against another flusher.
template <core::ProviderType PrvdType>
__device__ inline static void flushAsyncPostedImpl(core::RdmaEndpointDevice* ep,
                                                   uint32_t* outPostIdx) {
  core::WorkQueueHandle* wq = &ep->wqHandle;
  uint32_t postIdx = __hip_atomic_load(&wq->postIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  *outPostIdx = postIdx;
  __threadfence_system();  // the posters' WQEs (and our prior writes), before the doorbell
  if (__hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == postIdx)
    return;
  while (!tryLockDoorbell(wq)) {
  }
  uint32_t from = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  if (static_cast<int32_t>(postIdx - from) > 0) {
    ringAndPublish<PrvdType>(ep, from, postIdx, buildFlushDbrVal<PrvdType>(wq, postIdx, ep->qpn));
  }
  unlockDoorbell(wq);
}

template <core::ProviderType PrvdType>
__device__ inline static void waitImpl(core::RdmaEndpointDevice* ep, uint32_t postIdx) {
  quietUntil<PrvdType>(ep, postIdx);
}

template <core::ProviderType PrvdType>
__device__ inline static void signalImpl(core::RdmaEndpointDevice* ep, uint32_t qpn,
                                         uintptr_t signalRemoteAddr, uint32_t signalRemoteKey,
                                         ccoGdaSignalOp_t signalOp, uint64_t signalOpArg) {
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
    submitWqes<PrvdType, /*DeferDoorbell=*/false>(ep, curPostIdx, 1, dbrVal);
  } else {
    // MLX5/PSD: the WQE slot index doubles as the PSN.
    uint32_t curPostIdx = reserveWqeSlots<PrvdType>(ep, 1);
    uint64_t dbrVal = core::PostAtomic<PrvdType, uint64_t>(
        *wq, curPostIdx, curPostIdx, curPostIdx, true /*cqeSignal*/, qpn, atomicLaddr, atomicLkey,
        signalRemoteAddr, signalRemoteKey, addValue, 0 /*compare*/, core::AMO_FETCH_ADD);
    submitWqes<PrvdType, /*DeferDoorbell=*/false>(ep, curPostIdx, 1, dbrVal);
  }
}

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
    asm volatile("" ::: "memory");
  }
}

template <core::ProviderType PrvdType>
__device__ inline static void resetSignalImpl(volatile uint64_t* signalBuf,
                                              volatile uint64_t* signalShadows,
                                              ccoGdaSignal_t signalId) {
  signalBuf[signalId] = 0;
  signalShadows[signalId] = 0;
}

template <core::ProviderType PrvdType>
__device__ inline static uint64_t readCounterImpl(volatile uint64_t* counterBuf,
                                                  ccoGdaCounter_t counterId, int bits) {
  uint64_t val =
      __hip_atomic_load(&counterBuf[counterId], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  uint64_t mask = (bits >= 64) ? UINT64_MAX : ((1ULL << bits) - 1);
  return val & mask;
}

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
    asm volatile("" ::: "memory");
  }
}

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
template <ccoTeamMode TeamMode, ccoGdaThreadMode ThreadMode, typename RemoteAction, typename Coop,
          uint32_t OptFlags>
__device__ inline void ccoGda<PrvdType>::put(int peer, ccoWindow_t dstWin, size_t dstOffset,
                                             ccoWindow_t srcWin, size_t srcOffset, size_t bytes,
                                             RemoteAction remoteAction, Coop coop) {
  static_assert((OptFlags & ~kCcoGdaSupportedOptFlags) == 0, "unsupported ccoGdaOptFlags");
  constexpr bool kDefer = (OptFlags & ccoGdaOptFlagsDeferDoorbell) != 0;
  constexpr bool kNoMark = (OptFlags & ccoGdaOptFlagsPostedBeforeFlush) != 0;
  static_assert(!kNoMark || kDefer, "ccoGdaOptFlagsPostedBeforeFlush requires DeferDoorbell");
  if constexpr (ThreadMode == ccoGdaThreadAggregate) {
    static_assert(
        std::is_same_v<Coop, ccoCoopThread>,
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
    } else if constexpr (std::is_same_v<RemoteAction, ccoGda_WindowSignalAdd>) {
      // Caller's window, not the resource window: the offset is already
      // window-relative (iova=0), so it is the raddr as-is.
      signalRaddr = remoteAction.offset;
      signalRkey =
          reinterpret_cast<ccoWindowDevice*>(remoteAction.win)->ibgdaWin.peerRkeys[worldPeer];
      signalOp = ccoGdaSignalAdd;
      signalOpArg = remoteAction.value;
    }

    // Only mixed-QP thread scope needs per-QP grouping; ThreadAggregate and
    // CoopWarp/CoopBlock are a single group.
    if constexpr (ThreadMode == ccoGdaThreadIndependent && std::is_same_v<Coop, ccoCoopThread>) {
      // Group active lanes by QP (peer and context): each reserves contiguously.
      bool needTurn = true;
      for (uint64_t turns = __ballot(needTurn); turns != 0; turns = __ballot(needTurn)) {
        int lead = __ffsll(static_cast<unsigned long long>(turns)) - 1;
        if (qpIdx != __shfl(qpIdx, lead)) continue;
        needTurn = false;
        impl::putImpl<PrvdType, kDefer, kNoMark>(ep, qpn, localAddr, srcLkey, remoteAddr, dstRkey,
                                                 bytes, hasSignal, signalRaddr, signalRkey,
                                                 signalOp, signalOpArg);
      }
    } else {
      impl::putImpl<PrvdType, kDefer, kNoMark>(ep, qpn, localAddr, srcLkey, remoteAddr, dstRkey,
                                               bytes, hasSignal, signalRaddr, signalRkey, signalOp,
                                               signalOpArg);
    }
  }
  coop.sync();
}

// putValue: write immediate value (≤8 bytes)
template <core::ProviderType PrvdType>
template <ccoTeamMode TeamMode, ccoGdaThreadMode ThreadMode, typename T, typename RemoteAction,
          typename Coop, uint32_t OptFlags>
__device__ inline void ccoGda<PrvdType>::putValue(int peer, ccoWindow_t dstWin, size_t dstOffset,
                                                  T value, RemoteAction remoteAction, Coop coop) {
  static_assert(sizeof(T) <= 8, "putValue only supports types <= 8 bytes");
  static_assert((OptFlags & ~kCcoGdaSupportedOptFlags) == 0, "unsupported ccoGdaOptFlags");
  constexpr bool kDefer = (OptFlags & ccoGdaOptFlagsDeferDoorbell) != 0;
  constexpr bool kNoMark = (OptFlags & ccoGdaOptFlagsPostedBeforeFlush) != 0;
  static_assert(!kNoMark || kDefer, "ccoGdaOptFlagsPostedBeforeFlush requires DeferDoorbell");
  if constexpr (ThreadMode == ccoGdaThreadAggregate) {
    static_assert(
        std::is_same_v<Coop, ccoCoopThread>,
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
    } else if constexpr (std::is_same_v<RemoteAction, ccoGda_WindowSignalAdd>) {
      // Caller's window, not the resource window: the offset is already
      // window-relative (iova=0), so it is the raddr as-is.
      signalRaddr = remoteAction.offset;
      signalRkey =
          reinterpret_cast<ccoWindowDevice*>(remoteAction.win)->ibgdaWin.peerRkeys[worldPeer];
      signalOp = ccoGdaSignalAdd;
      signalOpArg = remoteAction.value;
    }

    // Only mixed-QP thread scope needs per-QP grouping; else a single group.
    if constexpr (ThreadMode == ccoGdaThreadIndependent && std::is_same_v<Coop, ccoCoopThread>) {
      bool needTurn = true;
      for (uint64_t turns = __ballot(needTurn); turns != 0; turns = __ballot(needTurn)) {
        int lead = __ffsll(static_cast<unsigned long long>(turns)) - 1;
        if (qpIdx != __shfl(qpIdx, lead)) continue;
        needTurn = false;
        impl::putValueImpl<PrvdType, T, kDefer, kNoMark>(ep, qpn, remoteAddr, dstRkey, value,
                                                         hasSignal, signalRaddr, signalRkey,
                                                         signalOp, signalOpArg);
      }
    } else {
      impl::putValueImpl<PrvdType, T, kDefer, kNoMark>(ep, qpn, remoteAddr, dstRkey, value,
                                                       hasSignal, signalRaddr, signalRkey, signalOp,
                                                       signalOpArg);
    }
  }
  coop.sync();
}

// get: RDMA read
template <core::ProviderType PrvdType>
template <ccoTeamMode TeamMode, ccoGdaThreadMode ThreadMode, typename Coop, uint32_t OptFlags>
__device__ inline void ccoGda<PrvdType>::get(int peer, ccoWindow_t remoteWin, size_t remoteOffset,
                                             ccoWindow_t localWin, size_t localOffset, size_t bytes,
                                             Coop coop) {
  static_assert((OptFlags & ~kCcoGdaSupportedOptFlags) == 0, "unsupported ccoGdaOptFlags");
  constexpr bool kDefer = (OptFlags & ccoGdaOptFlagsDeferDoorbell) != 0;
  constexpr bool kNoMark = (OptFlags & ccoGdaOptFlagsPostedBeforeFlush) != 0;
  static_assert(!kNoMark || kDefer, "ccoGdaOptFlagsPostedBeforeFlush requires DeferDoorbell");
  if constexpr (ThreadMode == ccoGdaThreadAggregate) {
    static_assert(
        std::is_same_v<Coop, ccoCoopThread>,
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

    // Only mixed-QP thread scope needs per-QP grouping; else a single group.
    if constexpr (ThreadMode == ccoGdaThreadIndependent && std::is_same_v<Coop, ccoCoopThread>) {
      bool needTurn = true;
      for (uint64_t turns = __ballot(needTurn); turns != 0; turns = __ballot(needTurn)) {
        int lead = __ffsll(static_cast<unsigned long long>(turns)) - 1;
        if (qpIdx != __shfl(qpIdx, lead)) continue;
        needTurn = false;
        impl::getImpl<PrvdType, kDefer, kNoMark>(ep, qpn, localAddr, localLkey, remoteAddr,
                                                 remoteRkey, bytes);
      }
    } else {
      impl::getImpl<PrvdType, kDefer, kNoMark>(ep, qpn, localAddr, localLkey, remoteAddr,
                                               remoteRkey, bytes);
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

    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    core::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
    uint32_t qpn = ep->qpn;

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
    } else if constexpr (std::is_same_v<RemoteAction, ccoGda_WindowSignalAdd>) {
      // Caller's window, not the resource window: the offset is already
      // window-relative (iova=0), so it is the raddr as-is.
      signalRaddr = remoteAction.offset;
      signalRkey =
          reinterpret_cast<ccoWindowDevice*>(remoteAction.win)->ibgdaWin.peerRkeys[worldPeer];
      signalOp = ccoGdaSignalAdd;
      signalOpArg = remoteAction.value;
    }

    impl::signalImpl<PrvdType>(ep, qpn, signalRaddr, signalRkey, signalOp, signalOpArg);
  }
  coop.sync();
}

// flush all peers: distribute peers across the Coop group (default: warp).
// all threads in the group must call flush together.
template <core::ProviderType PrvdType>
template <typename Coop>
__device__ inline void ccoGda<PrvdType>::flush(Coop coop) {
  static_assert(!std::is_same_v<Coop, ccoCoopThread>,
                "flush() requires at least ccoCoopWarp. "
                "ccoCoopThread reports size()==1, so every thread walks every peer and "
                "flushes each QP once per thread, duplicating doorbell rings and "
                "advancing cq->needConsIdx once per thread.");
  coop.sync();
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
  for (int teamPeer = coop.thread_rank(); teamPeer < this->nRanks; teamPeer += coop.size()) {
    if (teamPeer == this->rank) continue;
    // endpoints are world-indexed; the loop walks the GDA team.
    int worldPeer = impl::GdaPeerToWorld(comm, teamPeer);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    core::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
    uint32_t postIdx = 0;
    impl::flushAsyncImpl<PrvdType>(ep, &postIdx);
    impl::waitImpl<PrvdType>(ep, postIdx);
  }
  coop.sync();
}

// flush single peer: snapshot the QP's postIdx, then poll CQ until it completes.
template <core::ProviderType PrvdType>
template <ccoTeamMode TeamMode, typename Coop, uint32_t OptFlags>
__device__ inline void ccoGda<PrvdType>::flush(int peer, Coop coop) {
  static_assert((OptFlags & ~kCcoGdaSupportedFlushOptFlags) == 0, "unsupported flush opt flag");
  static_assert(!std::is_same_v<Coop, ccoCoopThread>,
                "flush(peer) requires at least ccoCoopWarp. "
                "With ccoCoopThread every thread runs the body, so two threads passing the "
                "same peer each flush that QP and advance cq->needConsIdx twice. "
                "(Lanes on *different* QPs are fine: quietUntil groups active lanes by CQ.)");
  coop.sync();
  if (coop.thread_rank() == 0) {
    int worldPeer = resolveWorldPeer<TeamMode>(peer);
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    core::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
    uint32_t postIdx = 0;
    if constexpr ((OptFlags & ccoGdaOptFlagsPostedBeforeFlush) != 0) {
      impl::flushAsyncPostedImpl<PrvdType>(ep, &postIdx);
    } else {
      impl::flushAsyncImpl<PrvdType>(ep, &postIdx);
    }
    impl::waitImpl<PrvdType>(ep, postIdx);
  }
  coop.sync();
}

// flushAsync: snapshot this peer's QP, return a request handle for wait().
template <core::ProviderType PrvdType>
template <ccoTeamMode TeamMode, typename Coop, uint32_t OptFlags>
__device__ inline void ccoGda<PrvdType>::flushAsync(int peer, ccoGdaRequest_t* outRequest,
                                                    Coop coop) {
  static_assert((OptFlags & ~kCcoGdaSupportedFlushOptFlags) == 0, "unsupported flush opt flag");
  coop.sync();
  if (coop.thread_rank() == 0) {
    int worldPeer = resolveWorldPeer<TeamMode>(peer);
    ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    core::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];

    uint32_t postIdx = 0;
    if constexpr ((OptFlags & ccoGdaOptFlagsPostedBeforeFlush) != 0) {
      impl::flushAsyncPostedImpl<PrvdType>(ep, &postIdx);
    } else {
      impl::flushAsyncImpl<PrvdType>(ep, &postIdx);
    }

    outRequest->qpIdx = qpIdx;
    outRequest->postIdx = static_cast<uint64_t>(postIdx);
  }
  coop.sync();
}

// wait: poll CQ until the request returned by flushAsync completes.
template <core::ProviderType PrvdType>
template <typename Coop>
__device__ inline void ccoGda<PrvdType>::wait(ccoGdaRequest_t& request, Coop coop) {
  // ccoCoopThread: each lane passes its own request; lanes on one CQ wait for the furthest.
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
                "ccoCoopThread reports size()==1 and makes thread_rank()==0 true for every "
                "thread, so every thread walks every peer and counterBuf is incremented "
                "once per thread instead of once.");
  coop.sync();

  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(_gdaHandle);

  for (int teamPeer = coop.thread_rank(); teamPeer < this->nRanks; teamPeer += coop.size()) {
    if (teamPeer == this->rank) continue;
    // endpoints are world-indexed; the loop walks the GDA team.
    int worldPeer = impl::GdaPeerToWorld(comm, teamPeer);
    int qpIdx = worldPeer * ibgda->numQpPerPe + (contextId % ibgda->numQpPerPe);
    core::RdmaEndpointDevice* ep = &ibgda->endpoints[qpIdx];
    uint32_t postIdx = 0;
    impl::flushAsyncImpl<PrvdType>(ep, &postIdx);  // rings for deferred posts first
    impl::waitImpl<PrvdType>(ep, postIdx);
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
                "ccoCoopThread reports size()==1, so every thread signals every peer and the "
                "remote signal is incremented once per thread; coop.sync() also degenerates "
                "to a no-op, removing the phase-1/phase-2 separation.");
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
