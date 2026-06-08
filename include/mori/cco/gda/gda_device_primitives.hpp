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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
#pragma once

// clang-format off
#include "mori/shmem/internal.hpp"
#include "mori/core/transport/rdma/rdma.hpp"
// clang-format off

namespace mori {
namespace cco {
namespace gda {

// Poll CQ and update doneIdx until it catches up to targetIdx
template <core::ProviderType PrvdType>
__device__ inline static void quietUntil(shmem::ShmemRdmaEndpoint* ep, uint32_t targetIdx) {
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
      while ((wq->doneIdx - targetIdx) & 0x800000) {
        uint32_t oldDoneIdx = wq->doneIdx;
        int err = core::PollCqOnce2(*wq, *cq, activemask, cq->cqAddr, cq->cqeNum, 0);
        if (err != 0) {
          MORI_PRINTF("quietUntil[PSD]: PollCqOnce2 failed, err=%d\n", err);
          break;
        }
        asm volatile("" ::: "memory");

        if (!((wq->doneIdx - targetIdx) & 0x800000)) break;
        if (wq->doneIdx == oldDoneIdx) break;
        if (!greed--) break;
      }

      core::spin_lock_release_shared(&cq->pollCqLock, activemask);
      break;
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
    // BNXT: similar to MLX5, 16-bit wqe_counter
    while ((int16_t)(wq->doneIdx - targetIdx) < 0) {
      uint32_t wqeCounter = 0;
      int err = core::PollCq<PrvdType>(cq->cqAddr, cq->cqeNum, &cq->consIdx, &wqeCounter);
      if (err >= 0) {
        wq->doneIdx = wqeCounter;
        core::UpdateCqDbrRecord<PrvdType>(*cq, cq->consIdx);
      }
      asm volatile("" ::: "memory");
    }
  }
}

// Reserve WQE slots and wait for SQ space
template <core::ProviderType PrvdType>
__device__ inline static uint32_t reserveWqeSlots(shmem::ShmemRdmaEndpoint* ep,
                                                  uint32_t numWqesNeeded) {
  core::WorkQueueHandle* wq = &ep->wqHandle;

  // Atomically allocate WQE slots
  uint32_t curPostIdx = atomicAdd(&wq->postIdx, numWqesNeeded);

  // Flow control: wait until SQ has enough space
  while (true) {
    uint64_t dbTouched =
        __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t dbDone = __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    uint64_t numActiveSqEntries = dbTouched - dbDone;
    uint64_t numFreeEntries = wq->sqWqeNum - numActiveSqEntries;
    uint64_t entriesUntilMine = curPostIdx + numWqesNeeded - dbTouched;

    if (numFreeEntries > entriesUntilMine) {
      break;  // Enough space available
    }

    // Not enough space, poll CQ to free up slots
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
__device__ inline static void ringDoorbellOrdered(shmem::ShmemRdmaEndpoint* ep, uint32_t myPostIdx,
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
    // BNXT: similar to MLX5
    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, myPostIdx + numWqes);
    __threadfence_system();
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
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

// New putImpl - Pure hardware operation layer
template <core::ProviderType PrvdType>
__device__ inline static void putImpl(
    // Hardware resources (already selected endpoint)
    shmem::ShmemRdmaEndpoint* ep, uint32_t qpn,

    // Data transfer parameters (already parsed addresses and keys)
    bool hasData, uintptr_t localAddr, uint32_t localKey,  // local buffer
    uintptr_t remoteAddr, uint32_t remoteKey,              // remote buffer
    size_t bytes,

    // Signal parameters (already parsed)
    bool hasSignal, uintptr_t signalRemoteAddr, uint32_t signalRemoteKey, ccoGdaSignalOp_t signalOp,
    uint64_t signalOpArg,

    // Counter parameters (already parsed)
    bool hasCounter, uintptr_t counterRemoteAddr, uint32_t counterRemoteKey,

    // Optimization flags
    uint32_t optFlags = ccoGdaOptFlagsDefault) {
  if (!hasData && !hasSignal && !hasCounter) return;

  // Get work queue handle
  core::WorkQueueHandle* wq = &ep->wqHandle;

  // Calculate total WQEs needed
  uint32_t numWqesNeeded = hasData ? 1 : 0;
  if (hasSignal) {
    numWqesNeeded += getAtomicWqeCount<PrvdType>(core::AMO_FETCH_ADD, sizeof(uint64_t));
  }
  if (hasCounter) {
    numWqesNeeded += getAtomicWqeCount<PrvdType>(core::AMO_FETCH_ADD, sizeof(uint64_t));
  }

  // Reserve WQE slots (with flow control)
  uint32_t curPostIdx = reserveWqeSlots<PrvdType>(ep, numWqesNeeded);

  // Post RDMA Write for data transfer
  uint64_t dbrVal = 0;
  uint32_t wqeIdx = curPostIdx;

  if (hasData) {
    if constexpr (PrvdType == core::ProviderType::PSD) {
      wq->outstandingWqe[wqeIdx % OUTSTANDING_TABLE_SIZE] = wqeIdx;
    }
    dbrVal = core::PostWrite<PrvdType>(*wq, wqeIdx, wqeIdx, wqeIdx, true /*cqeSignal*/, qpn,
                                       localAddr, localKey, remoteAddr, remoteKey, bytes);
    wqeIdx++;
  }

  // Post atomic for signal (remote peer notification)
  if (hasSignal) {
    if constexpr (PrvdType == core::ProviderType::PSD) {
      wq->outstandingWqe[wqeIdx % OUTSTANDING_TABLE_SIZE] = wqeIdx;
    }

    uintptr_t atomicLaddr = reinterpret_cast<uintptr_t>(ep->atomicIbuf.addr);
    uint32_t atomicLkey = ep->atomicIbuf.lkey;

    dbrVal = core::PostAtomic<PrvdType, uint64_t>(
        *wq, wqeIdx, wqeIdx, wqeIdx, true /*cqeSignal*/, qpn, atomicLaddr, atomicLkey,
        signalRemoteAddr, signalRemoteKey, signalOpArg, 0 /*compare*/, core::AMO_FETCH_ADD);
    wqeIdx++;
  }

  // Post atomic for counter (NIC loopback write to local memory)
  if (hasCounter) {
    if constexpr (PrvdType == core::ProviderType::PSD) {
      wq->outstandingWqe[wqeIdx % OUTSTANDING_TABLE_SIZE] = wqeIdx;
    }

    uintptr_t atomicLaddr = reinterpret_cast<uintptr_t>(ep->atomicIbuf.addr);
    uint32_t atomicLkey = ep->atomicIbuf.lkey;

    dbrVal = core::PostAtomic<PrvdType, uint64_t>(
        *wq, wqeIdx, wqeIdx, wqeIdx, true /*cqeSignal*/, qpn, atomicLaddr, atomicLkey,
        counterRemoteAddr, counterRemoteKey, 1 /*add 1*/, 0 /*compare*/, core::AMO_FETCH_ADD);
  }

  // Ring doorbell (ordered) unless AggregateRequests is set
  if (!(optFlags & ccoGdaOptFlagsAggregateRequests)) {
    ringDoorbellOrdered<PrvdType>(ep, curPostIdx, numWqesNeeded, dbrVal);
  }
}

// New putValueImpl - Inline write for small values
template <core::ProviderType PrvdType, typename T>
__device__ inline static void putValueImpl(shmem::ShmemRdmaEndpoint* ep, uint32_t qpn,
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
  if constexpr (PrvdType == core::ProviderType::PSD) {
    wq->outstandingWqe[wqeIdx % OUTSTANDING_TABLE_SIZE] = wqeIdx;
  }
  uint64_t dbrVal = core::PostWriteInline<PrvdType>(*wq, wqeIdx, wqeIdx, wqeIdx, true /*cqeSignal*/,
                                                    qpn, &value, remoteAddr, remoteKey, sizeof(T));
  wqeIdx++;

  // Post atomic for signal if requested
  if (hasSignal) {
    if constexpr (PrvdType == core::ProviderType::PSD) {
      wq->outstandingWqe[wqeIdx % OUTSTANDING_TABLE_SIZE] = wqeIdx;
    }

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
template <core::ProviderType PrvdType>
__device__ inline static void getImpl(shmem::ShmemRdmaEndpoint* ep, uint32_t qpn,
                                      uintptr_t localAddr, uint32_t localKey, uintptr_t remoteAddr,
                                      uint32_t remoteKey, size_t bytes,
                                      uint32_t optFlags = ccoGdaOptFlagsDefault) {
  if (bytes == 0) return;

  core::WorkQueueHandle* wq = &ep->wqHandle;

  // Reserve WQE slot
  uint32_t curPostIdx = reserveWqeSlots<PrvdType>(ep, 1);

  // Post RDMA Read
  if constexpr (PrvdType == core::ProviderType::PSD) {
    wq->outstandingWqe[curPostIdx % OUTSTANDING_TABLE_SIZE] = curPostIdx;
  }
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
__device__ inline static void flushAsyncImpl(shmem::ShmemRdmaEndpoint* ep, uint32_t qpn,
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
__device__ inline static void waitImpl(shmem::ShmemRdmaEndpoint* ep, uint32_t postIdx) {
  quietUntil<PrvdType>(ep, postIdx);
}

// Signal: send signal to remote peer (RDMA atomic increment/add)
template <core::ProviderType PrvdType>
__device__ inline static void signalImpl(shmem::ShmemRdmaEndpoint* ep, uint32_t qpn,
                                         uintptr_t signalRemoteAddr, uint32_t signalRemoteKey,
                                         ccoGdaSignalOp_t signalOp, uint64_t signalOpArg,
                                         uint32_t optFlags = ccoGdaOptFlagsDefault) {
  core::WorkQueueHandle* wq = &ep->wqHandle;

  // Reserve WQE slot
  uint32_t curPostIdx = reserveWqeSlots<PrvdType>(ep, 1);

  // Post RDMA atomic operation
  if constexpr (PrvdType == core::ProviderType::PSD) {
    wq->outstandingWqe[curPostIdx % OUTSTANDING_TABLE_SIZE] = curPostIdx;
  }

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

}  // namespace gda
}  // namespace cco
}  // namespace mori
