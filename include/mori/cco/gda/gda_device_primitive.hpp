// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License — see LICENSE for details.
#pragma once

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/cco/cco_types.hpp"
#include "mori/core/transport/rdma/device_primitives.hpp"
#include "mori/core/transport/rdma/providers/bnxt/bnxt_device_primitives.hpp"
#include "mori/core/transport/rdma/providers/ionic/ionic_device_primitives.hpp"
#include "mori/core/transport/rdma/providers/mlx5/mlx5_device_primitives.hpp"
#include "mori/shmem/internal.hpp"

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
    // PSD/Ionic: no DBR record update needed, just ring doorbell
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
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

// Core put implementation
template <core::ProviderType PrvdType>
__device__ inline static void putImpl(ccoGdaCtx ctx, int peer, ccoWindow_t dstWin, size_t dstOffset,
                                      ccoWindow_t srcWin, size_t srcOffset, size_t bytes,
                                      bool isSignal, ccoGdaSignal_t signalId,
                                      ccoGdaSignalOp_t signalOp, uint64_t signalOpArg,
                                      bool isCounter, ccoGdaCounter_t counterId,
                                      uint32_t optFlags = ccoGdaOptFlagsDefault) {
  if (bytes == 0 && !isSignal) return;

  // 1. Get IBGDA context and select endpoint
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(ctx.handle);
  int qpIdx = peer * ibgda->numQpPerPe + (ctx.contextId % ibgda->numQpPerPe);
  shmem::ShmemRdmaEndpoint* ep = &ibgda->endpoints[qpIdx];
  core::WorkQueueHandle* wq = &ep->wqHandle;
  uint32_t qpn = ep->qpn;

  // 2. Get window info and build addresses (iova=0 mode)
  ccoWindowDevice* src = reinterpret_cast<ccoWindowDevice*>(srcWin);
  ccoWindowDevice* dst = reinterpret_cast<ccoWindowDevice*>(dstWin);

  uintptr_t laddr = srcOffset;
  uintptr_t raddr = dstOffset;
  uint32_t lkey = src->ibgdaWin.lkey;
  uint32_t rkey = dst->ibgdaWin.peerRkeys[peer];

  // 3. Calculate total WQEs needed (provider-specific for atomics)
  uint32_t numWqesNeeded = (bytes > 0) ? 1 : 0;
  if (isSignal) {
    numWqesNeeded += getAtomicWqeCount<PrvdType>(core::AMO_FETCH_ADD, sizeof(uint64_t));
  }

  // 4. Reserve WQE slots (with flow control)
  uint32_t curPostIdx = reserveWqeSlots<PrvdType>(ep, numWqesNeeded);

  // 5. Post RDMA Write for data transfer
  uint64_t dbrVal = 0;
  uint32_t wqeIdx = curPostIdx;

  if (bytes > 0) {
    if constexpr (PrvdType == core::ProviderType::PSD) {
      wq->outstandingWqe[wqeIdx % OUTSTANDING_TABLE_SIZE] = wqeIdx;
    }
    dbrVal = core::PostWrite<PrvdType>(*wq, wqeIdx, wqeIdx, wqeIdx, true /*cqeSignal*/, qpn, laddr,
                                       lkey, raddr, rkey, bytes);
    wqeIdx++;
  }

  // 6. Post atomic for signal if requested
  if (isSignal) {
    // Signal buffer is at offset 0 in resourceWindow, indexed by signalId
    uintptr_t signalRaddr = signalId * sizeof(uint64_t);
    uint32_t signalRkey = dst->ibgdaWin.peerRkeys[peer];  // TODO: use resource window rkey

    uint64_t atomicVal = (signalOp == ccoGdaSignalInc) ? 1 : signalOpArg;

    if constexpr (PrvdType == core::ProviderType::PSD) {
      wq->outstandingWqe[wqeIdx % OUTSTANDING_TABLE_SIZE] = wqeIdx;
    }

    // Use atomic ibuf for result storage
    uintptr_t atomicLaddr = reinterpret_cast<uintptr_t>(ep->atomicIbuf.addr);
    uint32_t atomicLkey = ep->atomicIbuf.lkey;

    dbrVal = core::PostAtomic<PrvdType, uint64_t>(
        *wq, wqeIdx, wqeIdx, wqeIdx, true /*cqeSignal*/, qpn, atomicLaddr, atomicLkey, signalRaddr,
        signalRkey, atomicVal, 0 /*compare*/, core::AMO_FETCH_ADD);
  }

  // 7. Ring doorbell (ordered) unless AggregateRequests is set
  if (!(optFlags & ccoGdaOptFlagsAggregateRequests)) {
    ringDoorbellOrdered<PrvdType>(ep, curPostIdx, numWqesNeeded, dbrVal);
  }
}

// putValue implementation (inline write for small values)
template <core::ProviderType PrvdType, typename T>
__device__ inline static void putValueImpl(ccoGdaCtx ctx, int peer, ccoWindow_t dstWin,
                                           size_t dstOffset, T value, bool isSignal,
                                           ccoGdaSignal_t signalId, ccoGdaSignalOp_t signalOp,
                                           uint64_t signalOpArg,
                                           uint32_t optFlags = ccoGdaOptFlagsDefault) {
  static_assert(sizeof(T) <= 8, "putValue only supports types <= 8 bytes");

  // 1. Get IBGDA context and select endpoint
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(ctx.handle);
  int qpIdx = peer * ibgda->numQpPerPe + (ctx.contextId % ibgda->numQpPerPe);
  shmem::ShmemRdmaEndpoint* ep = &ibgda->endpoints[qpIdx];
  core::WorkQueueHandle* wq = &ep->wqHandle;
  uint32_t qpn = ep->qpn;

  // 2. Get window info
  ccoWindowDevice* dst = reinterpret_cast<ccoWindowDevice*>(dstWin);
  uintptr_t raddr = dstOffset;
  uint32_t rkey = dst->ibgdaWin.peerRkeys[peer];

  // 3. Calculate WQEs needed (provider-specific for atomics)
  uint32_t numWqesNeeded = 1;
  if (isSignal) {
    numWqesNeeded += getAtomicWqeCount<PrvdType>(core::AMO_FETCH_ADD, sizeof(uint64_t));
  }

  // 4. Reserve WQE slots
  uint32_t curPostIdx = reserveWqeSlots<PrvdType>(ep, numWqesNeeded);

  // 5. Post inline write
  uint32_t wqeIdx = curPostIdx;
  if constexpr (PrvdType == core::ProviderType::PSD) {
    wq->outstandingWqe[wqeIdx % OUTSTANDING_TABLE_SIZE] = wqeIdx;
  }
  uint64_t dbrVal = core::PostWriteInline<PrvdType>(*wq, wqeIdx, wqeIdx, wqeIdx, true /*cqeSignal*/,
                                                    qpn, &value, raddr, rkey, sizeof(T));
  wqeIdx++;

  // 6. Post atomic for signal if requested
  if (isSignal) {
    uintptr_t signalRaddr = signalId * sizeof(uint64_t);
    uint32_t signalRkey = dst->ibgdaWin.peerRkeys[peer];
    uint64_t atomicVal = (signalOp == ccoGdaSignalInc) ? 1 : signalOpArg;

    if constexpr (PrvdType == core::ProviderType::PSD) {
      wq->outstandingWqe[wqeIdx % OUTSTANDING_TABLE_SIZE] = wqeIdx;
    }

    // Use atomic ibuf for result storage
    uintptr_t atomicLaddr = reinterpret_cast<uintptr_t>(ep->atomicIbuf.addr);
    uint32_t atomicLkey = ep->atomicIbuf.lkey;

    dbrVal = core::PostAtomic<PrvdType, uint64_t>(*wq, wqeIdx, wqeIdx, wqeIdx, true /*cqeSignal*/,
                                                  qpn, atomicLaddr, atomicLkey, signalRaddr,
                                                  signalRkey, atomicVal, 0, core::AMO_FETCH_ADD);
  }

  // 7. Ring doorbell unless AggregateRequests is set
  if (!(optFlags & ccoGdaOptFlagsAggregateRequests)) {
    ringDoorbellOrdered<PrvdType>(ep, curPostIdx, numWqesNeeded, dbrVal);
  }
}

// Get implementation (RDMA read)
template <core::ProviderType PrvdType>
__device__ inline static void getImpl(ccoGdaCtx ctx, int peer, ccoWindow_t remoteWin,
                                      size_t remoteOffset, ccoWindow_t localWin, size_t localOffset,
                                      size_t bytes, uint32_t optFlags = ccoGdaOptFlagsDefault) {
  if (bytes == 0) return;

  // 1. Get IBGDA context and select endpoint
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(ctx.handle);
  int qpIdx = peer * ibgda->numQpPerPe + (ctx.contextId % ibgda->numQpPerPe);
  shmem::ShmemRdmaEndpoint* ep = &ibgda->endpoints[qpIdx];
  core::WorkQueueHandle* wq = &ep->wqHandle;
  uint32_t qpn = ep->qpn;

  // 2. Get addresses and keys (iova=0 mode)
  ccoWindowDevice* local = reinterpret_cast<ccoWindowDevice*>(localWin);
  ccoWindowDevice* remote = reinterpret_cast<ccoWindowDevice*>(remoteWin);

  uintptr_t laddr = localOffset;
  uintptr_t raddr = remoteOffset;
  uint32_t lkey = local->ibgdaWin.lkey;
  uint32_t rkey = remote->ibgdaWin.peerRkeys[peer];

  // 3. Reserve WQE slot
  uint32_t curPostIdx = reserveWqeSlots<PrvdType>(ep, 1);

  // 4. Post RDMA Read
  if constexpr (PrvdType == core::ProviderType::PSD) {
    wq->outstandingWqe[curPostIdx % OUTSTANDING_TABLE_SIZE] = curPostIdx;
  }
  uint64_t dbrVal =
      core::PostRead<PrvdType>(*wq, curPostIdx, curPostIdx, curPostIdx, true /*cqeSignal*/, qpn,
                               laddr, lkey, raddr, rkey, bytes);

  // 5. Ring doorbell unless AggregateRequests is set
  if (!(optFlags & ccoGdaOptFlagsAggregateRequests)) {
    ringDoorbellOrdered<PrvdType>(ep, curPostIdx, 1, dbrVal);
  }
}

// Flush: ring doorbell for pending WQEs and wait for completion
// 1. Lock → atomic_max(dbTouchIdx) → ring doorbell if advanced → unlock
// 2. Wait for completion via quietUntil
template <core::ProviderType PrvdType>
__device__ inline static void flushImpl(ccoGdaCtx ctx, int peer) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(ctx.handle);
  int qpIdx = peer * ibgda->numQpPerPe + (ctx.contextId % ibgda->numQpPerPe);
  shmem::ShmemRdmaEndpoint* ep = &ibgda->endpoints[qpIdx];
  core::WorkQueueHandle* wq = &ep->wqHandle;
  core::CompletionQueueHandle* cq = &ep->cqHandle;
  uint32_t qpn = ep->qpn;

  // Get current postIdx as target
  uint32_t postIdx = __hip_atomic_load(&wq->postIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  if (postIdx == 0) return;

  // Ring doorbell for any pending WQEs
  uint64_t activemask = core::GetActiveLaneMask();
  while (!core::spin_lock_try_acquire_shared(&wq->postSendLock, activemask)) {
    // Spin
  }

  // Atomic max on dbTouchIdx
  uint32_t oldDbTouch =
      __hip_atomic_fetch_max(&wq->dbTouchIdx, postIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

  // Only ring doorbell if we advanced dbTouchIdx
  if (oldDbTouch < postIdx) {
    __threadfence_system();

    uint32_t lastWqeIdx = (postIdx - 1) & (wq->sqWqeNum - 1);
    uint64_t dbrVal;

    if constexpr (PrvdType == core::ProviderType::PSD) {
      // PSD: doorbell value = sq_dbval | wrapped index
      dbrVal = wq->sq_dbval | (postIdx & (wq->sqWqeNum - 1));
      core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
    } else if constexpr (PrvdType == core::ProviderType::MLX5) {
      // MLX5: read control segment from last WQE for doorbell value
      uintptr_t wqeAddr =
          reinterpret_cast<uintptr_t>(wq->sqAddr) + (lastWqeIdx << MLX5_SEND_WQE_SHIFT);
      dbrVal = *reinterpret_cast<uint64_t*>(wqeAddr);

      core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, postIdx);
      __threadfence_system();
      core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
    } else if constexpr (PrvdType == core::ProviderType::BNXT) {
      // BNXT: compute doorbell value via bnxt_re_init_db_hdr
      uint8_t flags = (postIdx >> (__ffs(wq->sqWqeNum) - 1)) & 0x1;
      uint32_t epoch = (flags & BNXT_RE_FLAG_EPOCH_TAIL_MASK) << BNXT_RE_DB_EPOCH_TAIL_SHIFT;
      uint32_t slotIdx = (postIdx & (wq->sqWqeNum - 1)) * BNXT_RE_NUM_SLOT_PER_WQE;
      dbrVal = bnxt_re_init_db_hdr(slotIdx | epoch, 0, qpn, BNXT_RE_QUE_TYPE_SQ);

      core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, postIdx);
      __threadfence_system();
      core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
    }

    __threadfence_system();

    // Update needConsIdx for the WQEs we just rang doorbell for
    __hip_atomic_fetch_add(&cq->needConsIdx, postIdx - oldDbTouch, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
  }

  core::spin_lock_release_shared(&wq->postSendLock, activemask);

  // Wait for all operations up to postIdx to complete
  quietUntil<PrvdType>(ep, postIdx);
}

// FlushAsync: ring doorbell for pending WQEs and return ticket for later wait
template <core::ProviderType PrvdType>
__device__ inline static void flushAsyncImpl(ccoGdaCtx ctx, int peer, ccoGdaRequest_t* outRequest) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(ctx.handle);
  int qpIdx = peer * ibgda->numQpPerPe + (ctx.contextId % ibgda->numQpPerPe);
  shmem::ShmemRdmaEndpoint* ep = &ibgda->endpoints[qpIdx];
  core::WorkQueueHandle* wq = &ep->wqHandle;
  core::CompletionQueueHandle* cq = &ep->cqHandle;
  uint32_t qpn = ep->qpn;

  // Get current postIdx as target
  uint32_t postIdx = __hip_atomic_load(&wq->postIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  *outRequest = reinterpret_cast<ccoGdaRequest_t>(static_cast<uintptr_t>(postIdx));

  if (postIdx == 0) return;

  // Ring doorbell for any pending WQEs
  uint64_t activemask = core::GetActiveLaneMask();
  while (!core::spin_lock_try_acquire_shared(&wq->postSendLock, activemask)) {
    // Spin
  }

  // Atomic max on dbTouchIdx
  uint32_t oldDbTouch =
      __hip_atomic_fetch_max(&wq->dbTouchIdx, postIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

  // Only ring doorbell if we advanced dbTouchIdx
  if (oldDbTouch < postIdx) {
    __threadfence_system();

    uint32_t lastWqeIdx = (postIdx - 1) & (wq->sqWqeNum - 1);
    uint64_t dbrVal;

    if constexpr (PrvdType == core::ProviderType::PSD) {
      dbrVal = wq->sq_dbval | (postIdx & (wq->sqWqeNum - 1));
      core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
    } else if constexpr (PrvdType == core::ProviderType::MLX5) {
      uintptr_t wqeAddr =
          reinterpret_cast<uintptr_t>(wq->sqAddr) + (lastWqeIdx << MLX5_SEND_WQE_SHIFT);
      dbrVal = *reinterpret_cast<uint64_t*>(wqeAddr);

      core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, postIdx);
      __threadfence_system();
      core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
    } else if constexpr (PrvdType == core::ProviderType::BNXT) {
      uint8_t flags = (postIdx >> (__ffs(wq->sqWqeNum) - 1)) & 0x1;
      uint32_t epoch = (flags & BNXT_RE_FLAG_EPOCH_TAIL_MASK) << BNXT_RE_DB_EPOCH_TAIL_SHIFT;
      uint32_t slotIdx = (postIdx & (wq->sqWqeNum - 1)) * BNXT_RE_NUM_SLOT_PER_WQE;
      dbrVal = bnxt_re_init_db_hdr(slotIdx | epoch, 0, qpn, BNXT_RE_QUE_TYPE_SQ);

      core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, postIdx);
      __threadfence_system();
      core::RingDoorbell<PrvdType>(wq->dbrAddr, dbrVal);
    }

    __threadfence_system();

    __hip_atomic_fetch_add(&cq->needConsIdx, postIdx - oldDbTouch, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
  }

  core::spin_lock_release_shared(&wq->postSendLock, activemask);
}

// Wait: wait for async request to complete
template <core::ProviderType PrvdType>
__device__ inline static void waitImpl(ccoGdaCtx ctx, int peer, ccoGdaRequest_t request) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(ctx.handle);
  int qpIdx = peer * ibgda->numQpPerPe + (ctx.contextId % ibgda->numQpPerPe);
  shmem::ShmemRdmaEndpoint* ep = &ibgda->endpoints[qpIdx];

  uint32_t targetIdx = reinterpret_cast<uintptr_t>(request);
  quietUntil<PrvdType>(ep, targetIdx);
}

// Signal operations
template <core::ProviderType PrvdType>
__device__ inline static void signalImpl(ccoGdaCtx ctx, int peer, ccoGdaSignal_t signalId,
                                         ccoGdaSignalOp_t signalOp, uint64_t signalOpArg,
                                         uint32_t optFlags = ccoGdaOptFlagsDefault) {
  // Post atomic to remote signal buffer
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(ctx.handle);
  int qpIdx = peer * ibgda->numQpPerPe + (ctx.contextId % ibgda->numQpPerPe);
  shmem::ShmemRdmaEndpoint* ep = &ibgda->endpoints[qpIdx];
  core::WorkQueueHandle* wq = &ep->wqHandle;
  uint32_t qpn = ep->qpn;

  // TODO: get signal buffer rkey from resource window
  uintptr_t signalRaddr = signalId * sizeof(uint64_t);
  uint32_t signalRkey = 0;  // TODO: proper rkey

  uint64_t atomicVal = (signalOp == ccoGdaSignalInc) ? 1 : signalOpArg;

  // Calculate WQEs needed (provider-specific)
  uint32_t numWqes = getAtomicWqeCount<PrvdType>(core::AMO_FETCH_ADD, sizeof(uint64_t));
  uint32_t curPostIdx = reserveWqeSlots<PrvdType>(ep, numWqes);

  if constexpr (PrvdType == core::ProviderType::PSD) {
    wq->outstandingWqe[curPostIdx % OUTSTANDING_TABLE_SIZE] = curPostIdx;
  }

  // Use atomic ibuf for result storage
  uintptr_t atomicLaddr = reinterpret_cast<uintptr_t>(ep->atomicIbuf.addr);
  uint32_t atomicLkey = ep->atomicIbuf.lkey;

  uint64_t dbrVal = core::PostAtomic<PrvdType, uint64_t>(
      *wq, curPostIdx, curPostIdx, curPostIdx, true, qpn, atomicLaddr, atomicLkey, signalRaddr,
      signalRkey, atomicVal, 0, core::AMO_FETCH_ADD);

  // Ring doorbell unless AggregateRequests is set
  if (!(optFlags & ccoGdaOptFlagsAggregateRequests)) {
    ringDoorbellOrdered<PrvdType>(ep, curPostIdx, numWqes, dbrVal);
  }
}

__device__ inline static void resetSignalImpl(ccoGdaCtx ctx, ccoGdaSignal_t signalId) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(ctx.handle);
  ibgda->signalBuf[signalId] = 0;
  ibgda->signalShadows[signalId] = 0;
}

__device__ inline static uint64_t readSignalImpl(ccoGdaCtx ctx, ccoGdaSignal_t signalId, int bits) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(ctx.handle);
  uint64_t val =
      __hip_atomic_load(&ibgda->signalBuf[signalId], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  uint64_t shadow = ibgda->signalShadows[signalId];
  uint64_t mask = (bits >= 64) ? UINT64_MAX : ((1ULL << bits) - 1);
  return (val - shadow) & mask;
}

__device__ inline static void waitSignalImpl(ccoGdaCtx ctx, ccoGdaSignal_t signalId, uint64_t least,
                                             int bits) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(ctx.handle);
  uint64_t mask = (bits >= 64) ? UINT64_MAX : ((1ULL << bits) - 1);
  uint64_t shadow = ibgda->signalShadows[signalId];

  while (true) {
    uint64_t val =
        __hip_atomic_load(&ibgda->signalBuf[signalId], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    uint64_t delta = (val - shadow) & mask;
    if (delta >= least) {
      // Update shadow to consume
      ibgda->signalShadows[signalId] = (shadow + least) & mask;
      break;
    }
    // Spin
    asm volatile("" ::: "memory");
  }
}

// Counter operations
__device__ inline static uint64_t readCounterImpl(ccoGdaCtx ctx, ccoGdaCounter_t counterId,
                                                  int bits) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(ctx.handle);
  uint64_t val =
      __hip_atomic_load(&ibgda->counterBuf[counterId], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  uint64_t mask = (bits >= 64) ? UINT64_MAX : ((1ULL << bits) - 1);
  return val & mask;
}

__device__ inline static void resetCounterImpl(ccoGdaCtx ctx, ccoGdaCounter_t counterId) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(ctx.handle);
  ibgda->counterBuf[counterId] = 0;
}

__device__ inline static void waitCounterImpl(ccoGdaCtx ctx, ccoGdaCounter_t counterId,
                                              uint64_t least, int bits) {
  ccoIbgdaContext* ibgda = reinterpret_cast<ccoIbgdaContext*>(ctx.handle);
  uint64_t mask = (bits >= 64) ? UINT64_MAX : ((1ULL << bits) - 1);

  while (true) {
    uint64_t val = __hip_atomic_load(&ibgda->counterBuf[counterId], __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_SYSTEM);
    if ((val & mask) >= least) {
      break;
    }
    asm volatile("" ::: "memory");
  }
}

}  // namespace gda
}  // namespace cco
}  // namespace mori