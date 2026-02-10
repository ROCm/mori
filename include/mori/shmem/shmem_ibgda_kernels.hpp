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
#pragma once

#include <assert.h>
#include <mpi.h>

#include "mori/application/application.hpp"
#include "mori/core/core.hpp"
#include "mori/shmem/internal.hpp"
#include "mori/shmem/shmem_api.hpp"

namespace mori {
namespace shmem {

#ifdef ENABLE_BNXT
#define DISPATCH_MLX5 0
#define DISPATCH_BNXT 1
#define DISPATCH_PSD 0
#elif ENABLE_IONIC
#define DISPATCH_MLX5 0
#define DISPATCH_BNXT 0
#define DISPATCH_PSD 1
#else
#define DISPATCH_MLX5 1
#define DISPATCH_BNXT 0
#define DISPATCH_PSD 0
#endif

#define DISPATCH_PROVIDER_TYPE(func, ...)                             \
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();               \
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;     \
  core::ProviderType prvdType = ep[pe].GetProviderType();             \
  if (DISPATCH_MLX5 && prvdType == core::ProviderType::MLX5) {        \
    func<core::ProviderType::MLX5>(__VA_ARGS__);                      \
  } else if (DISPATCH_BNXT && prvdType == core::ProviderType::BNXT) { \
    func<core::ProviderType::BNXT>(__VA_ARGS__);                      \
  } else if (DISPATCH_PSD && prvdType == core::ProviderType::PSD) {   \
    func<core::ProviderType::PSD>(__VA_ARGS__);                       \
  } else {                                                            \
    assert(false && "Unsupported or disabled provider type");         \
  }

#define DISPATCH_PROVIDER_TYPE_EP(ep, func, ...)                      \
  core::ProviderType prvdType = ep[pe].GetProviderType();             \
  if (DISPATCH_MLX5 && prvdType == core::ProviderType::MLX5) {        \
    func<core::ProviderType::MLX5>(__VA_ARGS__);                      \
  } else if (DISPATCH_BNXT && prvdType == core::ProviderType::BNXT) { \
    func<core::ProviderType::BNXT>(__VA_ARGS__);                      \
  } else if (DISPATCH_PSD && prvdType == core::ProviderType::PSD) {   \
    func<core::ProviderType::PSD>(__VA_ARGS__);                       \
  } else {                                                            \
    assert(false && "Unsupported or disabled provider type");         \
  }

#define DISPATCH_PROVIDER_TYPE_COMPILE_TIME(func, ...) \
  do {                                                 \
    if constexpr (DISPATCH_BNXT == 1) {                \
      func<core::ProviderType::BNXT>(__VA_ARGS__);     \
    } else if constexpr (DISPATCH_PSD == 1) {          \
      func<core::ProviderType::PSD>(__VA_ARGS__);      \
    } else {                                           \
      func<core::ProviderType::MLX5>(__VA_ARGS__);     \
    }                                                  \
  } while (0)

#define DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_RETURN(func, type, ...) \
  [&]() {                                                                \
    if constexpr (DISPATCH_BNXT == 1) {                                  \
      return func<core::ProviderType::BNXT, type>(__VA_ARGS__);          \
    } else if constexpr (DISPATCH_PSD == 1) {                            \
      return func<core::ProviderType::PSD, type>(__VA_ARGS__);           \
    } else {                                                             \
      return func<core::ProviderType::MLX5, type>(__VA_ARGS__);          \
    }                                                                    \
  }()

#define DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_BOOL(func, boolParam, ...) \
  do {                                                                      \
    if constexpr (DISPATCH_BNXT == 1) {                                     \
      func<core::ProviderType::BNXT, boolParam>(__VA_ARGS__);               \
    } else if constexpr (DISPATCH_PSD == 1) {                               \
      func<core::ProviderType::PSD, boolParam>(__VA_ARGS__);                \
    } else {                                                                \
      func<core::ProviderType::MLX5, boolParam>(__VA_ARGS__);               \
    }                                                                       \
  } while (0)

/* ---------------------------------------------------------------------------------------------- */
/*                                    VMM Heap Helper Functions                                   */
/* ---------------------------------------------------------------------------------------------- */
// Query VMM local key with chunk boundary calculation
inline __device__ void VmmQueryLocalKey(uintptr_t addr, size_t max_size, uint32_t& out_lkey,
                                        size_t& out_chunk_size) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::SymmMemObj* heapObj = globalGpuStates->heapObj;
  uintptr_t heapBase = globalGpuStates->heapBaseAddr;

  size_t offsetFromHeapBase = addr - heapBase;
  size_t chunkIdx = offsetFromHeapBase >> globalGpuStates->vmmChunkSizeShift;

  application::VMMChunkKey chunkKey = heapObj->vmmLkeyInfo[chunkIdx];

  out_lkey = chunkKey.key;
  size_t chunk_remaining = chunkKey.next_addr - addr;
  out_chunk_size = chunk_remaining < max_size ? chunk_remaining : max_size;
  MORI_PRINTF(
      "blockId %d, threadId %d VMM Heap: single transfer,chunkIdx: %zu, srcAddr: %p, lkey: %x, "
      "chunk_remaining: %zu, out_chunk_size: %zu\n",
      blockIdx.x, threadIdx.x, chunkIdx, addr, out_lkey, chunk_remaining, out_chunk_size);
}

// Query VMM remote address and key with chunk boundary calculation
inline __device__ void VmmQueryRemoteAddr(uintptr_t addr, int pe, size_t max_size,
                                          uintptr_t& out_raddr, uint32_t& out_rkey,
                                          size_t& out_chunk_size) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::SymmMemObj* heapObj = globalGpuStates->heapObj;
  uintptr_t heapBase = globalGpuStates->heapBaseAddr;

  size_t offsetFromHeapBase = addr - heapBase;
  size_t chunkIdx = offsetFromHeapBase >> globalGpuStates->vmmChunkSizeShift;

  out_raddr = heapObj->peerPtrs[pe] + offsetFromHeapBase;

  application::VMMChunkKey chunkKey = heapObj->vmmRkeyInfo[chunkIdx * heapObj->worldSize + pe];

  out_rkey = chunkKey.key;
  size_t chunk_remaining = chunkKey.next_addr - addr;
  out_chunk_size = chunk_remaining < max_size ? chunk_remaining : max_size;
  MORI_PRINTF(
      "blockId %d, threadId %d VMM Heap: single transfer,chunkIdx: %zu, dstAddr: %p, raddr: %lx, "
      "rkey: %x, chunk_remaining: %zu, out_chunk_size: %zu\n",
      blockIdx.x, threadIdx.x, chunkIdx, addr, out_raddr, out_rkey, chunk_remaining,
      out_chunk_size);
}

// Lookup VMM remote address and key (for small fixed-size transfers)
inline __device__ void VmmLookupRemote(uintptr_t addr, int pe, uintptr_t& out_raddr,
                                       uint32_t& out_rkey) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::SymmMemObj* heapObj = globalGpuStates->heapObj;
  uintptr_t heapBase = globalGpuStates->heapBaseAddr;

  size_t offsetFromHeapBase = addr - heapBase;
  size_t chunkIdx = offsetFromHeapBase >> globalGpuStates->vmmChunkSizeShift;

  out_raddr = heapObj->peerPtrs[pe] + offsetFromHeapBase;

  out_rkey = heapObj->vmmRkeyInfo[chunkIdx * heapObj->worldSize + pe].key;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ void ShmemQuietThreadKernelSerialImpl(int pe, int qpId) {
  if (core::GetActiveLaneNum() != 0) return;
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int epIndex = pe * globalGpuStates->numQpPerPe + (qpId % globalGpuStates->numQpPerPe);
  core::WorkQueueHandle& wq = ep[epIndex].wqHandle;
  core::CompletionQueueHandle& cq = ep[epIndex].cqHandle;
  if (!core::AcquireLockOnce(&cq.pollCqLock)) return;
  while (true) {
    uint32_t dbTouchIdx =
        __hip_atomic_load(&wq.dbTouchIdx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
    uint32_t doneIdx = __hip_atomic_load(&wq.doneIdx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
    if (dbTouchIdx == doneIdx) {
      break;
    }

    uint32_t my_cq_consumer =
        __hip_atomic_fetch_add(&cq.cq_consumer, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    uint32_t wqe_counter;
    int opcode =
        core::PollCq<core::ProviderType::BNXT>(cq.cqAddr, cq.cqeNum, &my_cq_consumer, &wqe_counter);
    if (opcode != BNXT_RE_REQ_ST_OK) {
      int rank = globalGpuStates->rank;
      uint32_t my_cq_index = my_cq_consumer % cq.cqeNum;
      MORI_PRINTF("rank %d dest pe %d consIdx %d opcode %d\n", rank, pe, my_cq_index, opcode);
      assert(false);
    }
    wqe_counter = (wqe_counter + wq.sqWqeNum - 1) % wq.sqWqeNum;
    uint64_t wqe_id = wq.outstandingWqe[wqe_counter] + 1;

    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __hip_atomic_fetch_max(&wq.doneIdx, wqe_id, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  core::ReleaseLock(&cq.pollCqLock);
}

inline __device__ void ShmemQuietThreadKernelPsdImpl(int pe, int qpId) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  const int epIndex = pe * globalGpuStates->numQpPerPe + (qpId % globalGpuStates->numQpPerPe);
  core::WorkQueueHandle& wqHandle = globalGpuStates->rdmaEndpoints[epIndex].wqHandle;
  core::CompletionQueueHandle& cqHandle = globalGpuStates->rdmaEndpoints[epIndex].cqHandle;

  const uint64_t activeMask = core::GetActiveLaneMask();
  const uint32_t myLogicalLaneId = core::GetActiveLaneNum(activeMask);
  const int myLaneId = core::WarpLaneId();

  constexpr uint32_t PENDING_WORK_MASK = 0x800000;  // Bit 23: sign bit for 24-bit counter
  const uint32_t dbTouchedIdx = wqHandle.dbTouchIdx;
  constexpr uint32_t MAX_GREED = 10;
  constexpr uint32_t CQ_DOORBELL_GRACE = 100;  // IONIC_CQ_GRACE
  uint32_t wqeCounter;

  // Outer loop: retry lock acquisition until work is done
  while ((wqHandle.doneIdx - dbTouchedIdx) & PENDING_WORK_MASK) {
    if (!core::spin_lock_try_acquire_shared(&cqHandle.pollCqLock, activeMask)) {
      continue;  // Lock acquisition failed, retry
    }

    // Inner loop: process CQEs while holding the lock
    uint32_t greedRemaining = MAX_GREED;
    while ((wqHandle.doneIdx - dbTouchedIdx) & PENDING_WORK_MASK) {
      const uint64_t oldDoneIdx = wqHandle.doneIdx;

      const uint32_t curConsIdx = cqHandle.cq_consumer;
      uint32_t myCqPos = curConsIdx + myLogicalLaneId;

      // Poll CQE
      const int opcode = core::PollCq<core::ProviderType::PSD>(cqHandle.cqAddr, cqHandle.cqeNum,
                                                               &myCqPos, &wqeCounter);
      if (opcode > 0) {
        MORI_PRINTF("rank %d dest pe %d consIdx %d opcode %d\n", globalGpuStates->rank, pe, myCqPos,
                    opcode);
        assert(false);
      }
      asm volatile("" ::: "memory");

      const uint64_t successMask = __ballot(opcode == 0);
      const int highestLane = core::GetLastActiveLaneID(successMask);

      if (highestLane == -1) {
        continue;
      }

      if (myLaneId == highestLane) {
        cqHandle.cq_consumer = myCqPos + 1;

        if (((cqHandle.cq_consumer - cqHandle.cq_dbpos) & (cqHandle.cqeNum - 1)) >=
            CQ_DOORBELL_GRACE) {
          cqHandle.cq_dbpos = cqHandle.cq_consumer;
          core::UpdateCqDbrRecord<core::ProviderType::PSD>(cqHandle, myCqPos + 1);
        }

        wqHandle.doneIdx = wqeCounter;
        // __hip_atomic_fetch_max(&wqHandle.doneIdx, wqeCounter, __ATOMIC_RELAXED,
        // __HIP_MEMORY_SCOPE_AGENT);
      }

      if (!((wqHandle.doneIdx - dbTouchedIdx) & PENDING_WORK_MASK)) {
        if (wqHandle.doneIdx == oldDoneIdx) {
          break;
        }
        if (greedRemaining == 0) {
          break;
        }
        --greedRemaining;
      }
    }

    core::spin_lock_release_shared(&cqHandle.pollCqLock, activeMask);
    break;  // Work done, exit outer loop
  }
}

inline __device__ void ShmemQuietThreadKernelMlnxImpl(int pe, int qpId) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int epIndex = pe * globalGpuStates->numQpPerPe + (qpId % globalGpuStates->numQpPerPe);
  core::WorkQueueHandle& wq = ep[epIndex].wqHandle;
  core::CompletionQueueHandle& cq = ep[epIndex].cqHandle;

  constexpr size_t BROADCAST_SIZE = 1024 / warpSize;
  __shared__ uint64_t wqe_broadcast[BROADCAST_SIZE];
  uint8_t warp_id = core::FlatBlockThreadId() / warpSize;
  wqe_broadcast[warp_id] = 0;

  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader{my_logical_lane_id == 0};
  const uint64_t leader_phys_lane_id = core::GetFirstActiveLaneID(activemask);

  while (true) {
    bool done{false};
    uint32_t quiet_amount{0};
    uint32_t warp_cq_consumer{0};
    while (!done) {
      uint32_t active =
          __hip_atomic_load(&cq.activeIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      uint32_t posted =
          __hip_atomic_load(&cq.needConsIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      uint32_t completed =
          __hip_atomic_load(&cq.consIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (!(posted - completed)) {
        return;
      }
      int32_t quiet_val = posted - active;
      if (quiet_val <= 0) {
        continue;
      }
      quiet_amount = min(num_active_lanes, quiet_val);
      if (is_leader) {
        done = __hip_atomic_compare_exchange_strong(&cq.activeIdx, &active, active + quiet_amount,
                                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                                    __HIP_MEMORY_SCOPE_AGENT);
        if (done) {
          warp_cq_consumer = __hip_atomic_fetch_add(&cq.cq_consumer, quiet_amount, __ATOMIC_RELAXED,
                                                    __HIP_MEMORY_SCOPE_AGENT);
        }
      }
      done = __shfl(done, leader_phys_lane_id);
    }
    warp_cq_consumer = __shfl(warp_cq_consumer, leader_phys_lane_id);
    uint32_t my_cq_consumer = warp_cq_consumer + my_logical_lane_id;
    uint32_t my_cq_index = my_cq_consumer % cq.cqeNum;

    if (my_logical_lane_id < quiet_amount) {
      uint32_t wqe_counter;
      int opcode = core::PollCq<core::ProviderType::MLX5>(cq.cqAddr, cq.cqeNum, &my_cq_consumer,
                                                          &wqe_counter);
      if (opcode == MLX5_CQE_RESP_ERR || opcode == MLX5_CQE_REQ_ERR) {
        int rank = globalGpuStates->rank;
        MORI_PRINTF("rank %d dest pe %d consIdx %d opcode %d\n", rank, pe, my_cq_index, opcode);
        core::DumpMlx5Wqe(wq.sqAddr, my_cq_index);
        assert(false);
      }
      uint64_t wqe_id = wq.outstandingWqe[wqe_counter];
      __hip_atomic_fetch_max(&wqe_broadcast[warp_id], wqe_id, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_WORKGROUP);
      __atomic_signal_fence(__ATOMIC_SEQ_CST);
    }
    if (is_leader) {
      uint64_t completed{0};
      do {
        completed = __hip_atomic_load(&cq.consIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      } while (completed != warp_cq_consumer);

      core::UpdateCqDbrRecord<core::ProviderType::MLX5>(
          cq, (uint32_t)(warp_cq_consumer + quiet_amount));

      __atomic_signal_fence(__ATOMIC_SEQ_CST);
      uint64_t doneIdx = wqe_broadcast[warp_id];
      __hip_atomic_fetch_max(&wq.doneIdx, doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_fetch_add(&cq.consIdx, quiet_amount, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemQuietThreadKernelImpl(int pe, int qpId) {
  if constexpr (PrvdType == core::ProviderType::BNXT) {
    ShmemQuietThreadKernelSerialImpl(pe, qpId);
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    ShmemQuietThreadKernelPsdImpl(pe, qpId);
  } else if constexpr (PrvdType == core::ProviderType::MLX5) {
    ShmemQuietThreadKernelMlnxImpl(pe, qpId);
  } else {
    static_assert(false);
  }
}

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::RDMA>() {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  int rank = globalGpuStates->rank;
  int worldSize = globalGpuStates->worldSize;
  for (int peId = 0; peId < worldSize; peId++) {
    if (peId != rank && globalGpuStates->transportTypes[peId] == application::TransportType::RDMA) {
      for (int qpId = 0; qpId < globalGpuStates->numQpPerPe; qpId++) {
        DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemQuietThreadKernelImpl, peId, qpId);
      }
    }
  }
}

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::RDMA>(int pe) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  int rank = globalGpuStates->rank;
  if (pe == rank) return;
  if (globalGpuStates->transportTypes[pe] != application::TransportType::RDMA) return;
  for (int qpId = 0; qpId < globalGpuStates->numQpPerPe; qpId++) {
    DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemQuietThreadKernelImpl, pe, qpId);
  }
}

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::RDMA>(int pe, int qpId) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int rank = globalGpuStates->rank;
  if (pe == rank) return;
  if (globalGpuStates->transportTypes[pe] != application::TransportType::RDMA) return;
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemQuietThreadKernelImpl, pe, qpId);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Point-to-Point                                         */
/* ---------------------------------------------------------------------------------------------- */
template <core::ProviderType PrvdType>
inline __device__ void ShmemPutMemNbiThreadKernelImpl(const application::SymmMemObjPtr dest,
                                                      size_t destOffset,
                                                      const application::SymmMemObjPtr source,
                                                      size_t sourceOffset, size_t bytes, int pe,
                                                      int qpId) {
  if (bytes == 0) return;

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int epIndex = pe * globalGpuStates->numQpPerPe + (qpId % globalGpuStates->numQpPerPe);
  core::WorkQueueHandle* wq = &ep[epIndex].wqHandle;
  core::CompletionQueueHandle* cq = &ep[epIndex].cqHandle;
  uint32_t qpn = ep[epIndex].handle.qpn;

  bool needsChunking = globalGpuStates->useVMMHeap;
  size_t currentOffset = 0;
  size_t remaining = bytes;

  while (true) {
    // Check if current thread still has data to transfer
    bool has_remaining = (remaining > 0);

    // Synchronize within warp: get mask of threads that still have work
    uint64_t activemask = __ballot(has_remaining);
    if (activemask == 0) {
      break;  // All threads in warp are done
    }

    // Recalculate active lane info for threads with remaining data
    uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
    uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
    bool is_leader{my_logical_lane_id == num_active_lanes - 1};
    const uint64_t leader_phys_lane_id = core::GetLastActiveLaneID(activemask);

    // Inactive threads skip actual work but stay synchronized
    if (!has_remaining) {
      continue;
    }

    // Get RDMA keys and addresses based on mode
    uint32_t lkey, rkey;
    uintptr_t srcAddr, raddr;
    size_t transfer_size;

    if (!needsChunking) {
      // Isolation or Static Heap - direct access
      // Keys are uniform, transfer entire remaining bytes in one shot
      lkey = source->lkey;
      srcAddr = reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset + currentOffset;
      raddr = dest->peerPtrs[pe] + destOffset + currentOffset;
      rkey = dest->peerRkeys[pe];
      transfer_size = remaining;
    } else {
      // Slow path: VMM Heap - query keys for current chunk
      srcAddr = reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset + currentOffset;
      size_t src_chunk_size;
      VmmQueryLocalKey(srcAddr, remaining, lkey, src_chunk_size);

      uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset + currentOffset;
      size_t dst_chunk_size;
      VmmQueryRemoteAddr(dstAddr, pe, remaining, raddr, rkey, dst_chunk_size);

      transfer_size = src_chunk_size < dst_chunk_size ? src_chunk_size : dst_chunk_size;
      // MORI_PRINTF("blockId %d, threadId %d VMM Heap: single transfer,srcAddr: %p, dstAddr: %p,
      // lkey: %x, raddr: %lx, rkey: %x, transfer_size: %zu\n", blockIdx.x, threadIdx.x, srcAddr,
      // dstAddr, lkey, raddr, rkey, transfer_size);
    }
    MORI_PRINTF("blockIdx.x=%d, threadIdx.x=%d, remaining=%zu, transfer_size=%zu\n", blockIdx.x,
                threadIdx.x, remaining, transfer_size);
    // Post RDMA write (unified code for both fast and slow paths)
    uint32_t warp_sq_counter{0};
    uint32_t warp_msntbl_counter{0}, warp_psn_counter{0};
    uint32_t my_sq_counter{0}, my_msntbl_counter{0}, my_psn_counter{0};
    uint32_t psnCnt = 0;

    if constexpr (PrvdType == core::ProviderType::BNXT) {
      psnCnt = (transfer_size + wq->mtuSize - 1) / wq->mtuSize;
    }
    if (is_leader) {
      if constexpr (PrvdType == core::ProviderType::MLX5) {
        warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT);
      } else if constexpr (PrvdType == core::ProviderType::BNXT) {
        core::atomic_add_packed_msn_and_psn(&wq->msnPack, num_active_lanes,
                                            psnCnt * num_active_lanes, &warp_msntbl_counter,
                                            &warp_psn_counter);
        warp_sq_counter = warp_msntbl_counter;
        __hip_atomic_fetch_max(&wq->postIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
      } else if constexpr (PrvdType == core::ProviderType::PSD) {
        warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT);
      } else {
        static_assert(false);
      }
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    if constexpr (PrvdType == core::ProviderType::MLX5) {
      my_sq_counter = warp_sq_counter + my_logical_lane_id;
    } else if constexpr (PrvdType == core::ProviderType::BNXT) {
      warp_msntbl_counter = __shfl(warp_msntbl_counter, leader_phys_lane_id);
      warp_psn_counter = __shfl(warp_psn_counter, leader_phys_lane_id);
      my_sq_counter = warp_sq_counter + my_logical_lane_id;
      my_msntbl_counter = warp_msntbl_counter + my_logical_lane_id;
      my_psn_counter = warp_psn_counter + psnCnt * my_logical_lane_id;
    } else if constexpr (PrvdType == core::ProviderType::PSD) {
      my_sq_counter = warp_sq_counter + my_logical_lane_id;
    } else {
      static_assert(false);
    }

    while (true) {
      uint64_t db_touched =
          __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      uint64_t db_done =
          __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      uint64_t num_active_sq_entries = db_touched - db_done;
      uint64_t num_free_entries = wq->sqWqeNum - num_active_sq_entries;
      uint64_t num_entries_until_warp_last_entry = warp_sq_counter + num_active_lanes - db_touched;
      if (num_free_entries > num_entries_until_warp_last_entry) {
        break;
      }
      ShmemQuietThreadKernelImpl<PrvdType>(pe, qpId);
    }

    uint64_t dbr_val;
    if constexpr (PrvdType == core::ProviderType::MLX5) {
      wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
      dbr_val =
          core::PostWrite<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, is_leader,
                                    qpn, srcAddr, lkey, raddr, rkey, transfer_size);
    } else if constexpr (PrvdType == core::ProviderType::BNXT) {
      wq->outstandingWqe[my_sq_counter % wq->sqWqeNum] = my_sq_counter;
      dbr_val =
          core::PostWrite<PrvdType>(*wq, my_sq_counter, my_msntbl_counter, my_psn_counter,
                                    is_leader, qpn, srcAddr, lkey, raddr, rkey, transfer_size);
    } else if constexpr (PrvdType == core::ProviderType::PSD) {
      wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
      dbr_val =
          core::PostWrite<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, is_leader,
                                    qpn, srcAddr, lkey, raddr, rkey, transfer_size);
    } else {
      static_assert(false);
    }
    __threadfence_system();
    if (is_leader) {
      uint64_t db_touched{0};
      do {
        db_touched = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      } while (db_touched != warp_sq_counter);

      core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, warp_sq_counter + num_active_lanes);
      __threadfence_system();
      core::RingDoorbell<PrvdType>(wq->dbrAddr, dbr_val);
      __threadfence_system();

      __hip_atomic_fetch_add(&cq->needConsIdx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(&wq->dbTouchIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
    __threadfence_system();

    // Move to next chunk (for VMM heap) or exit loop (for Isolation/Static heap)
    currentOffset += transfer_size;
    remaining -= transfer_size;
  }
}

template <>
inline __device__ void ShmemPutMemNbiThreadKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe, int qpId) {
  bool need_turn{true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemPutMemNbiThreadKernelImpl, dest, destOffset, source,
                                          sourceOffset, bytes, pe, qpId);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemPutMemNbiWarpKernelImpl(const application::SymmMemObjPtr dest,
                                                    size_t destOffset,
                                                    const application::SymmMemObjPtr source,
                                                    size_t sourceOffset, size_t bytes, int pe,
                                                    int qpId) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemPutMemNbiThreadKernelImpl<PrvdType>(dest, destOffset, source, sourceOffset, bytes, pe,
                                             qpId);
  }
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemPutMemNbiBlockKernelImpl(const application::SymmMemObjPtr dest,
                                                     size_t destOffset,
                                                     const application::SymmMemObjPtr source,
                                                     size_t sourceOffset, size_t bytes, int pe,
                                                     int qpId) {
  int threadId = core::FlatBlockThreadId();
  if (threadId == 0) {
    ShmemPutMemNbiThreadKernelImpl<PrvdType>(dest, destOffset, source, sourceOffset, bytes, pe,
                                             qpId);
  }
}

template <>
inline __device__ void ShmemPutMemNbiWarpKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe, int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemPutMemNbiWarpKernelImpl, dest, destOffset, source,
                                      sourceOffset, bytes, pe, qpId);
}

template <>
inline __device__ void ShmemPutMemNbiBlockKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe, int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemPutMemNbiBlockKernelImpl, dest, destOffset, source,
                                      sourceOffset, bytes, pe, qpId);
}

// TODO: deal with bytes count limit
// TODO: put size api only support 1,2,4,8,16 in nvshmem, should we do that?
template <core::ProviderType PrvdType>
inline __device__ void ShmemPutSizeImmNbiThreadKernelImpl(const application::SymmMemObjPtr dest,
                                                          size_t destOffset, void* val,
                                                          size_t bytes, int pe, int qpId) {
  if (bytes == 0) return;
  // assert(destOffset + bytes <= dest->size);

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();

  // Get correct rkey for VMM heap or use direct rkey for Isolation/Static Heap
  uintptr_t raddr;
  uint32_t rkey;
  if (globalGpuStates->useVMMHeap) {
    // VMM Heap: data is small (≤16 bytes), won't cross chunk boundary
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset;
    VmmLookupRemote(dstAddr, pe, raddr, rkey);
  } else {
    // Isolation or Static Heap: direct access
    raddr = dest->peerPtrs[pe] + destOffset;
    rkey = dest->peerRkeys[pe];
  }
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int epIndex = pe * globalGpuStates->numQpPerPe + (qpId % globalGpuStates->numQpPerPe);
  core::WorkQueueHandle* wq = &ep[epIndex].wqHandle;
  core::CompletionQueueHandle* cq = &ep[epIndex].cqHandle;
  uint32_t qpn = ep[epIndex].handle.qpn;

  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader{my_logical_lane_id == num_active_lanes - 1};
  const uint64_t leader_phys_lane_id = core::GetLastActiveLaneID(activemask);
  uint32_t warp_sq_counter{0};
  uint32_t warp_msntbl_counter{0}, warp_psn_counter{0};
  uint32_t my_sq_counter{0}, my_msntbl_counter{0}, my_psn_counter{0};

  if constexpr (PrvdType == core::ProviderType::MLX5) {
    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    if (is_leader) {
      core::atomic_add_packed_msn_and_psn(&wq->msnPack, num_active_lanes, num_active_lanes,
                                          &warp_msntbl_counter, &warp_psn_counter);
      warp_sq_counter = warp_msntbl_counter;
      __hip_atomic_fetch_max(&wq->postIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    warp_msntbl_counter = __shfl(warp_msntbl_counter, leader_phys_lane_id);
    warp_psn_counter = __shfl(warp_psn_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
    my_msntbl_counter = warp_msntbl_counter + my_logical_lane_id;
    my_psn_counter = warp_psn_counter + my_logical_lane_id;
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
  } else {
    static_assert(false);
  }

  while (true) {
    uint64_t db_touched =
        __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t db_done = __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t num_active_sq_entries = db_touched - db_done;
    uint64_t num_free_entries = wq->sqWqeNum - num_active_sq_entries;
    uint64_t num_entries_until_warp_last_entry = warp_sq_counter + num_active_lanes - db_touched;
    if (num_free_entries > num_entries_until_warp_last_entry) {
      break;
    }
    ShmemQuietThreadKernelImpl<PrvdType>(pe, qpId);
  }

  uint64_t dbr_val;
  if constexpr (PrvdType == core::ProviderType::MLX5) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
    dbr_val = core::PostWriteInline<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter,
                                              is_leader, qpn, val, raddr, rkey, bytes);
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    wq->outstandingWqe[my_sq_counter % wq->sqWqeNum] = my_sq_counter;
    dbr_val = core::PostWriteInline<PrvdType>(*wq, my_sq_counter, my_msntbl_counter, my_psn_counter,
                                              is_leader, qpn, val, raddr, rkey, bytes);
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
    dbr_val = core::PostWriteInline<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter,
                                              is_leader, qpn, val, raddr, rkey, bytes);
  } else {
    static_assert(false);
  }
  __threadfence_system();
  if (is_leader) {
    uint64_t db_touched = 0;
    do {
      db_touched = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != warp_sq_counter);

    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, warp_sq_counter + num_active_lanes);
    __threadfence_system();
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbr_val);
    __threadfence_system();

    __hip_atomic_fetch_add(&cq->needConsIdx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&wq->dbTouchIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }
  __threadfence_system();
}

template <>
inline __device__ void ShmemPutSizeImmNbiThreadKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe,
    int qpId) {
  bool need_turn{true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemPutSizeImmNbiThreadKernelImpl, dest, destOffset, val,
                                          bytes, pe, qpId);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemPutSizeImmNbiWarpKernelImpl(const application::SymmMemObjPtr dest,
                                                        size_t destOffset, void* val, size_t bytes,
                                                        int pe, int qpId) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemPutSizeImmNbiThreadKernelImpl<PrvdType>(dest, destOffset, val, bytes, pe, qpId);
  }
}

template <>
inline __device__ void ShmemPutSizeImmNbiWarpKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe,
    int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemPutSizeImmNbiWarpKernelImpl, dest, destOffset, val,
                                      bytes, pe, qpId);
}

template <core::ProviderType PrvdType, bool onlyOneSignal = true>
inline __device__ void ShmemPutMemNbiSignalThreadKernelImpl(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  if (bytes == 0) return;
  // assert(sourceOffset + bytes <= source->size && destOffset + bytes <= dest->size);

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int epIndex = pe * globalGpuStates->numQpPerPe + (qpId % globalGpuStates->numQpPerPe);
  core::WorkQueueHandle* wq = &ep[epIndex].wqHandle;
  core::CompletionQueueHandle* cq = &ep[epIndex].cqHandle;
  uint32_t qpn = ep[epIndex].handle.qpn;

  bool needsChunking = globalGpuStates->useVMMHeap;
  size_t currentOffset = 0;
  size_t remaining = bytes;

  while (true) {
    // Check if current thread still has data to transfer
    bool has_remaining = (remaining > 0);

    // Synchronize within warp: get mask of threads that still have work
    uint64_t activemask = __ballot(has_remaining);
    if (activemask == 0) {
      break;  // All threads in warp are done
    }

    // Recalculate active lane info for threads with remaining data
    uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
    uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
    bool is_leader{my_logical_lane_id == num_active_lanes - 1};
    const uint64_t leader_phys_lane_id = core::GetLastActiveLaneID(activemask);

    // Inactive threads skip actual work but stay synchronized
    if (!has_remaining) {
      continue;
    }

    // Get RDMA keys and addresses for current chunk
    uint32_t lkey, rkey;
    uintptr_t laddr, raddr;
    size_t transfer_size;

    if (!needsChunking) {
      // Fast path: Isolation or Static Heap
      lkey = source->lkey;
      laddr = reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset + currentOffset;
      raddr = dest->peerPtrs[pe] + destOffset + currentOffset;
      rkey = dest->peerRkeys[pe];
      transfer_size = remaining;
    } else {
      // Slow path: VMM Heap - query keys for current chunk
      uintptr_t srcAddr =
          reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset + currentOffset;
      size_t src_chunk_size;
      VmmQueryLocalKey(srcAddr, remaining, lkey, src_chunk_size);
      laddr = srcAddr;

      uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset + currentOffset;
      size_t dst_chunk_size;
      VmmQueryRemoteAddr(dstAddr, pe, remaining, raddr, rkey, dst_chunk_size);

      transfer_size = src_chunk_size < dst_chunk_size ? src_chunk_size : dst_chunk_size;
    }

    // Each thread checks if this is its last chunk
    bool my_is_last_chunk = (transfer_size == remaining);

    // Synchronize: only send signal if ALL active threads are on their last chunk
    // This ensures warp-uniform decision on num_wqes
    uint64_t all_last_mask = __ballot(my_is_last_chunk);
    bool isLastChunk = (all_last_mask == activemask);

    uint32_t warp_sq_counter{0};
    uint32_t warp_msntbl_counter{0}, warp_psn_counter{0};
    uint32_t my_sq_counter{0}, my_msntbl_counter{0}, my_psn_counter{0};
    uint32_t psnCnt = 0;
    // For last chunk: add 1 WQE for signal; for other chunks: just put
    uint32_t num_wqes = isLastChunk ? (onlyOneSignal ? num_active_lanes + 1 : num_active_lanes * 2)
                                    : num_active_lanes;

    if constexpr (PrvdType == core::ProviderType::BNXT) {
      psnCnt = (transfer_size + wq->mtuSize - 1) / wq->mtuSize;
    }
    if (is_leader) {
      if constexpr (PrvdType == core::ProviderType::MLX5) {
        warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_wqes, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT);
      } else if constexpr (PrvdType == core::ProviderType::BNXT) {
        uint32_t total_psn = psnCnt * num_active_lanes;
        if (isLastChunk) {
          total_psn += (onlyOneSignal ? 1 : num_active_lanes);
        }
        core::atomic_add_packed_msn_and_psn(&wq->msnPack, num_wqes, total_psn, &warp_msntbl_counter,
                                            &warp_psn_counter);
        warp_sq_counter = warp_msntbl_counter;
        __hip_atomic_fetch_max(&wq->postIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
      } else if constexpr (PrvdType == core::ProviderType::PSD) {
        warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_wqes, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT);
      } else {
        static_assert(false);
      }
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    if constexpr (PrvdType == core::ProviderType::MLX5) {
      my_sq_counter = warp_sq_counter +
                      (isLastChunk && !onlyOneSignal ? my_logical_lane_id * 2 : my_logical_lane_id);
    } else if constexpr (PrvdType == core::ProviderType::BNXT) {
      warp_msntbl_counter = __shfl(warp_msntbl_counter, leader_phys_lane_id);
      warp_psn_counter = __shfl(warp_psn_counter, leader_phys_lane_id);
      my_sq_counter = warp_sq_counter +
                      (isLastChunk && !onlyOneSignal ? my_logical_lane_id * 2 : my_logical_lane_id);
      my_msntbl_counter =
          warp_msntbl_counter +
          (isLastChunk && !onlyOneSignal ? my_logical_lane_id * 2 : my_logical_lane_id);
      my_psn_counter =
          warp_psn_counter + (isLastChunk && !onlyOneSignal ? (psnCnt + 1) * my_logical_lane_id
                                                            : psnCnt * my_logical_lane_id);
    } else if constexpr (PrvdType == core::ProviderType::PSD) {
      my_sq_counter = warp_sq_counter +
                      (isLastChunk && !onlyOneSignal ? my_logical_lane_id * 2 : my_logical_lane_id);
    } else {
      static_assert(false);
    }

    while (true) {
      uint64_t db_touched =
          __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      uint64_t db_done =
          __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      uint64_t num_active_sq_entries = db_touched - db_done;
      uint64_t num_free_entries = wq->sqWqeNum - num_active_sq_entries;
      uint64_t num_entries_until_warp_last_entry = warp_sq_counter + num_wqes - db_touched;
      if (num_free_entries > num_entries_until_warp_last_entry) {
        break;
      }
      ShmemQuietThreadKernelImpl<PrvdType>(pe, qpId);
    }

    // Post RDMA write for this chunk
    // Note: PostWrite always returns a valid doorbell value, regardless of cqeSignal
    uint64_t dbr_val;
    if constexpr (PrvdType == core::ProviderType::MLX5) {
      wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
      dbr_val = core::PostWrite<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter,
                                          is_leader, qpn, laddr, lkey, raddr, rkey, transfer_size);
    } else if constexpr (PrvdType == core::ProviderType::BNXT) {
      wq->outstandingWqe[my_sq_counter % wq->sqWqeNum] = my_sq_counter;
      dbr_val = core::PostWrite<PrvdType>(*wq, my_sq_counter, my_msntbl_counter, my_psn_counter,
                                          is_leader, qpn, laddr, lkey, raddr, rkey, transfer_size);
    } else if constexpr (PrvdType == core::ProviderType::PSD) {
      wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
      dbr_val = core::PostWrite<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter,
                                          is_leader, qpn, laddr, lkey, raddr, rkey, transfer_size);
    } else {
      static_assert(false);
    }

    // Post signal only for the last chunk (and update dbr_val)
    if (isLastChunk) {
      // assert(signalDestOffset + sizeof(signalValue) <= signalDest->size);
      uintptr_t signalRaddr;
      uint32_t signalRkey;
      if (!needsChunking) {
        signalRaddr = signalDest->peerPtrs[pe] + signalDestOffset;
        signalRkey = signalDest->peerRkeys[pe];
      } else {
        uintptr_t signalAddr = reinterpret_cast<uintptr_t>(signalDest->localPtr) + signalDestOffset;
        VmmLookupRemote(signalAddr, pe, signalRaddr, signalRkey);
      }
      if (signalOp == core::atomicType::AMO_SET || signalOp == core::atomicType::AMO_SIGNAL_SET) {
        // TODO: not support masked atomic yet, use write inline for now
        bool should_signal = onlyOneSignal ? is_leader : true;
        if (should_signal) {
          if constexpr (PrvdType == core::ProviderType::MLX5) {
            wq->outstandingWqe[(my_sq_counter + 1) % OUTSTANDING_TABLE_SIZE] = my_sq_counter + 1;
            dbr_val = core::PostWriteInline<PrvdType>(
                *wq, my_sq_counter + 1, my_sq_counter + 1, my_sq_counter + 1, is_leader, qpn,
                &signalValue, signalRaddr, signalRkey, sizeof(signalValue));
          } else if constexpr (PrvdType == core::ProviderType::BNXT) {
            wq->outstandingWqe[(my_sq_counter + 1) % wq->sqWqeNum] = my_sq_counter + 1;
            dbr_val = core::PostWriteInline<PrvdType>(
                *wq, my_sq_counter + 1, my_msntbl_counter + 1, my_psn_counter + psnCnt, is_leader,
                qpn, &signalValue, signalRaddr, signalRkey, sizeof(signalValue));
          } else if constexpr (PrvdType == core::ProviderType::PSD) {
            wq->outstandingWqe[(my_sq_counter + 1) % OUTSTANDING_TABLE_SIZE] = my_sq_counter + 1;
            dbr_val = core::PostWriteInline<PrvdType>(
                *wq, my_sq_counter + 1, my_sq_counter + 1, my_sq_counter + 1, is_leader, qpn,
                &signalValue, signalRaddr, signalRkey, sizeof(signalValue));
          }
        }
      } else if (signalOp == core::atomicType::AMO_ADD ||
                 signalOp == core::atomicType::AMO_SIGNAL_ADD) {
        core::IbufHandle* ibuf = &ep[epIndex].atomicIbuf;
        bool should_signal = onlyOneSignal ? is_leader : true;
        if (should_signal) {
          if constexpr (PrvdType == core::ProviderType::MLX5) {
            wq->outstandingWqe[(my_sq_counter + 1) % OUTSTANDING_TABLE_SIZE] = my_sq_counter + 1;
            dbr_val = core::PostAtomic<PrvdType>(
                *wq, my_sq_counter + 1, my_sq_counter + 1, my_sq_counter + 1, is_leader, qpn,
                ibuf->addr, ibuf->lkey, signalRaddr, signalRkey, &signalValue, &signalValue,
                sizeof(signalValue), core::atomicType::AMO_ADD);
          } else if constexpr (PrvdType == core::ProviderType::BNXT) {
            wq->outstandingWqe[(my_sq_counter + 1) % wq->sqWqeNum] = my_sq_counter + 1;
            dbr_val = core::PostAtomic<PrvdType>(
                *wq, my_sq_counter + 1, my_msntbl_counter + 1, my_psn_counter + psnCnt, is_leader,
                qpn, ibuf->addr, ibuf->lkey, signalRaddr, signalRkey, &signalValue, &signalValue,
                sizeof(signalValue), core::atomicType::AMO_ADD);
          } else if constexpr (PrvdType == core::ProviderType::PSD) {
            wq->outstandingWqe[(my_sq_counter + 1) % OUTSTANDING_TABLE_SIZE] = my_sq_counter + 1;
            dbr_val = core::PostAtomic<PrvdType>(
                *wq, my_sq_counter + 1, my_sq_counter + 1, my_sq_counter + 1, is_leader, qpn,
                ibuf->addr, ibuf->lkey, signalRaddr, signalRkey, &signalValue, &signalValue,
                sizeof(signalValue), core::atomicType::AMO_ADD);
          }
        }
      } else {
        assert(false);
      }
    }

    __threadfence_system();
    if (is_leader) {
      uint64_t db_touched{0};
      do {
        db_touched = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      } while (db_touched != warp_sq_counter);

      core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, warp_sq_counter + num_wqes);
      __threadfence_system();
      // Ring doorbell every iteration (dbr_val comes from PostWrite or signal WQE)
      core::RingDoorbell<PrvdType>(wq->dbrAddr, dbr_val);
      __threadfence_system();

      __hip_atomic_fetch_add(&cq->needConsIdx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(&wq->dbTouchIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
    __threadfence_system();

    // Move to next chunk
    currentOffset += transfer_size;
    remaining -= transfer_size;
  }
}

template <>
inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::RDMA, true>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  bool need_turn{true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_BOOL(
          ShmemPutMemNbiSignalThreadKernelImpl, true, dest, destOffset, source, sourceOffset, bytes,
          signalDest, signalDestOffset, signalValue, signalOp, pe, qpId);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <>
inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::RDMA, false>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  bool need_turn{true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_BOOL(
          ShmemPutMemNbiSignalThreadKernelImpl, false, dest, destOffset, source, sourceOffset,
          bytes, signalDest, signalDestOffset, signalValue, signalOp, pe, qpId);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <core::ProviderType PrvdType, bool onlyOneSignal = true>
inline __device__ void ShmemPutMemNbiSignalWarpKernelImpl(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemPutMemNbiSignalThreadKernelImpl<PrvdType, onlyOneSignal>(
        dest, destOffset, source, sourceOffset, bytes, signalDest, signalDestOffset, signalValue,
        signalOp, pe, qpId);
  }
}

template <core::ProviderType PrvdType, bool onlyOneSignal = true>
inline __device__ void ShmemPutMemNbiSignalBlockKernelImpl(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  if (core::FlatBlockThreadId() == 0) {
    ShmemPutMemNbiSignalThreadKernelImpl<PrvdType, onlyOneSignal>(
        dest, destOffset, source, sourceOffset, bytes, signalDest, signalDestOffset, signalValue,
        signalOp, pe, qpId);
  }
}

template <>
inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::RDMA, true>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_BOOL(ShmemPutMemNbiSignalWarpKernelImpl, true, dest,
                                                destOffset, source, sourceOffset, bytes, signalDest,
                                                signalDestOffset, signalValue, signalOp, pe, qpId);
}

template <>
inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::RDMA, false>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_BOOL(ShmemPutMemNbiSignalWarpKernelImpl, false, dest,
                                                destOffset, source, sourceOffset, bytes, signalDest,
                                                signalDestOffset, signalValue, signalOp, pe, qpId);
}

template <>
inline __device__ void ShmemPutMemNbiSignalBlockKernel<application::TransportType::RDMA, true>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_BOOL(ShmemPutMemNbiSignalBlockKernelImpl, true, dest,
                                                destOffset, source, sourceOffset, bytes, signalDest,
                                                signalDestOffset, signalValue, signalOp, pe, qpId);
}

template <>
inline __device__ void ShmemPutMemNbiSignalBlockKernel<application::TransportType::RDMA, false>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_BOOL(ShmemPutMemNbiSignalBlockKernelImpl, false, dest,
                                                destOffset, source, sourceOffset, bytes, signalDest,
                                                signalDestOffset, signalValue, signalOp, pe, qpId);
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernelImpl(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes,
    core::atomicType amoType, int pe, int qpId) {
  if (bytes == 0) return;
  // assert(destOffset + bytes <= dest->size);

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int epIndex = pe * globalGpuStates->numQpPerPe + (qpId % globalGpuStates->numQpPerPe);
  core::WorkQueueHandle* wq = &ep[epIndex].wqHandle;
  core::CompletionQueueHandle* cq = &ep[epIndex].cqHandle;
  uint32_t qpn = ep[epIndex].handle.qpn;
  core::IbufHandle* ibuf = &ep[epIndex].atomicIbuf;

  // Get correct rkey for VMM heap or use direct rkey for Isolation/Static Heap
  uintptr_t raddr;
  uint32_t rkey;
  if (globalGpuStates->useVMMHeap) {
    // VMM Heap: atomic data is small (≤8 bytes), won't cross chunk boundary
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset;
    VmmLookupRemote(dstAddr, pe, raddr, rkey);
  } else {
    // Isolation or Static Heap: direct access
    raddr = dest->peerPtrs[pe] + destOffset;
    rkey = dest->peerRkeys[pe];
  }

  uintptr_t laddr = ibuf->addr;
  uintptr_t lkey = ibuf->lkey;

  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader{my_logical_lane_id == num_active_lanes - 1};
  const uint64_t leader_phys_lane_id = core::GetLastActiveLaneID(activemask);

  uint32_t warp_sq_counter = 0;
  uint32_t warp_msntbl_counter = 0, warp_psn_counter = 0;
  uint32_t my_sq_counter = 0, my_msntbl_counter = 0, my_psn_counter = 0;

  if constexpr (PrvdType == core::ProviderType::MLX5) {
    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    if (is_leader) {
      core::atomic_add_packed_msn_and_psn(&wq->msnPack, num_active_lanes, num_active_lanes,
                                          &warp_msntbl_counter, &warp_psn_counter);
      warp_sq_counter = warp_msntbl_counter;
      __hip_atomic_fetch_max(&wq->postIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    warp_msntbl_counter = __shfl(warp_msntbl_counter, leader_phys_lane_id);
    warp_psn_counter = __shfl(warp_psn_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
    my_msntbl_counter = warp_msntbl_counter + my_logical_lane_id;
    my_psn_counter = warp_psn_counter + my_logical_lane_id;
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
  } else {
    static_assert(false);
  }

  while (true) {
    uint64_t db_touched =
        __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t db_done = __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t num_active_sq_entries = db_touched - db_done;
    uint64_t num_free_entries = wq->sqWqeNum - num_active_sq_entries;
    uint64_t num_entries_until_warp_last_entry = warp_sq_counter + num_active_lanes - db_touched;
    if (num_free_entries > num_entries_until_warp_last_entry) break;
    ShmemQuietThreadKernelImpl<PrvdType>(pe, qpId);
  }

  if constexpr (PrvdType == core::ProviderType::MLX5) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    wq->outstandingWqe[my_sq_counter % wq->sqWqeNum] = my_sq_counter;
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
  }

  uint64_t dbr_val;
  if constexpr (PrvdType == core::ProviderType::MLX5) {
    dbr_val =
        core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, is_leader, qpn,
                                   laddr, lkey, raddr, rkey, val, val, bytes, amoType);
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    dbr_val =
        core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_msntbl_counter, my_psn_counter, is_leader,
                                   qpn, laddr, lkey, raddr, rkey, val, val, bytes, amoType);
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    dbr_val =
        core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, is_leader, qpn,
                                   laddr, lkey, raddr, rkey, val, val, bytes, amoType);
  }

  __threadfence_system();
  if (is_leader) {
    uint64_t db_touched = 0;
    do {
      db_touched = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != warp_sq_counter);

    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, warp_sq_counter + num_active_lanes);
    __threadfence_system();
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbr_val);

    __hip_atomic_fetch_add(&cq->needConsIdx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&wq->dbTouchIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }

  __threadfence_system();
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes,
    core::atomicType amoType, int pe, int qpId) {
  bool need_turn{true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemAtomicSizeNonFetchThreadKernelImpl, dest, destOffset,
                                          val, bytes, amoType, pe, qpId);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernelImpl(const application::SymmMemObjPtr dest,
                                                             size_t destOffset, void* val,
                                                             size_t bytes, core::atomicType amoType,
                                                             int pe, int qpId) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemAtomicSizeNonFetchThreadKernelImpl<PrvdType>(dest, destOffset, val, bytes, amoType, pe,
                                                      qpId);
  }
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes,
    core::atomicType amoType, int pe, int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemAtomicSizeNonFetchWarpKernelImpl, dest, destOffset, val,
                                      bytes, amoType, pe, qpId);
}

inline __device__ uint32_t ShmemGetAtomicIbufSlot(core::IbufHandle* ibuf, uint32_t num_slots = 1) {
  uint32_t base_slot = atomicAdd(&ibuf->head, num_slots);
  uint32_t nslots = ibuf->nslots;
  uint32_t last_slot = base_slot + num_slots;
  while (last_slot - __hip_atomic_load(&ibuf->tail, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >
         nslots) {
    ;
  }
  __threadfence_block();
  return base_slot;
}

inline __device__ void ShmemReleaseAtomicIbufSlot(core::IbufHandle* ibuf, uint32_t base_slots,
                                                  uint32_t num_slots) {
  uint32_t last_slot = base_slots + num_slots;
  while (atomicCAS(&ibuf->tail, base_slots, last_slot) != base_slots) {
    ;
  }
  __threadfence_block();
}

template <core::ProviderType PrvdType, typename T>
inline __device__ T ShmemAtomicTypeFetchThreadKernelImpl(const application::SymmMemObjPtr dest,
                                                         size_t destOffset, void* val,
                                                         void* compare, size_t bytes,
                                                         core::atomicType amoType, int pe,
                                                         int qpId) {
  // assert(destOffset + bytes <= dest->size);
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int epIndex = pe * globalGpuStates->numQpPerPe + (qpId % globalGpuStates->numQpPerPe);
  core::WorkQueueHandle* wq = &ep[epIndex].wqHandle;
  core::CompletionQueueHandle* cq = &ep[epIndex].cqHandle;
  uint32_t qpn = ep[epIndex].handle.qpn;
  core::IbufHandle* ibuf = &ep[epIndex].atomicIbuf;

  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader = (my_logical_lane_id == num_active_lanes - 1);
  uint64_t leader_phys_lane_id = core::GetLastActiveLaneID(activemask);

  uint32_t base_slot = 0;
  if (is_leader) {
    base_slot = ShmemGetAtomicIbufSlot(ibuf, num_active_lanes);
  }
  uint32_t my_slot = __shfl(base_slot, leader_phys_lane_id) + my_logical_lane_id;
  uint32_t my_slot_index = my_slot & (ibuf->nslots - 1);
  uintptr_t laddr = ibuf->addr + (my_slot_index + 1) * application::ATOMIC_IBUF_SLOT_SIZE;
  uintptr_t lkey = ibuf->lkey;

  // Get correct rkey for VMM heap or use direct rkey for Isolation/Static Heap
  uintptr_t raddr;
  uint32_t rkey;
  if (globalGpuStates->useVMMHeap) {
    // VMM Heap: atomic data is small (≤8 bytes), won't cross chunk boundary
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dest->localPtr) + destOffset;
    VmmLookupRemote(dstAddr, pe, raddr, rkey);
  } else {
    // Isolation or Static Heap: direct access
    raddr = dest->peerPtrs[pe] + destOffset;
    rkey = dest->peerRkeys[pe];
  }

  uint32_t warp_sq_counter = 0;
  uint32_t warp_msntbl_counter = 0, warp_psn_counter = 0;
  uint32_t my_sq_counter = 0, my_msntbl_counter = 0, my_psn_counter = 0;

  if constexpr (PrvdType == core::ProviderType::MLX5) {
    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    if (is_leader) {
      core::atomic_add_packed_msn_and_psn(&wq->msnPack, num_active_lanes, num_active_lanes,
                                          &warp_msntbl_counter, &warp_psn_counter);
      warp_sq_counter = warp_msntbl_counter;
      __hip_atomic_fetch_max(&wq->postIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    warp_msntbl_counter = __shfl(warp_msntbl_counter, leader_phys_lane_id);
    warp_psn_counter = __shfl(warp_psn_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
    my_msntbl_counter = warp_msntbl_counter + my_logical_lane_id;
    my_psn_counter = warp_psn_counter + my_logical_lane_id;
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
  } else {
    static_assert(false);
  }

  while (true) {
    uint64_t db_touched =
        __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t db_done = __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t num_active_sq_entries = db_touched - db_done;
    uint64_t num_free_entries = wq->sqWqeNum - num_active_sq_entries;
    uint64_t num_entries_until_warp_last_entry = warp_sq_counter + num_active_lanes - db_touched;
    if (num_free_entries > num_entries_until_warp_last_entry) break;
    ShmemQuietThreadKernelImpl<PrvdType>(pe, qpId);
  }

  if constexpr (PrvdType == core::ProviderType::MLX5) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    wq->outstandingWqe[my_sq_counter % wq->sqWqeNum] = my_sq_counter;
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
  }

  uint64_t dbr_val;
  if constexpr (PrvdType == core::ProviderType::MLX5) {
    dbr_val =
        core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, is_leader, qpn,
                                   laddr, lkey, raddr, rkey, val, compare, bytes, amoType);
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    dbr_val =
        core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_msntbl_counter, my_psn_counter, is_leader,
                                   qpn, laddr, lkey, raddr, rkey, val, compare, bytes, amoType);
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    dbr_val =
        core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, is_leader, qpn,
                                   laddr, lkey, raddr, rkey, val, compare, bytes, amoType);
  }

  __threadfence_system();
  if (is_leader) {
    uint64_t db_touched = 0;
    do {
      db_touched = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != warp_sq_counter);

    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, warp_sq_counter + num_active_lanes);
    __threadfence_system();
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbr_val);

    __hip_atomic_fetch_add(&cq->needConsIdx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&wq->dbTouchIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }

  ShmemQuietThreadKernelImpl<PrvdType>(pe, qpId);
  T ret = *reinterpret_cast<volatile T*>(laddr);
  if (sizeof(T) == 4) ret = BSWAP32((uint32_t)ret);

  if (is_leader) {
    ShmemReleaseAtomicIbufSlot(ibuf, base_slot, num_active_lanes);
  }

  return ret;
}

#define DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL(TypeName, T)                            \
  template <>                                                                                \
  inline __device__ T ShmemAtomicTypeFetchThreadKernel<application::TransportType::RDMA, T>( \
      const application::SymmMemObjPtr dest, size_t destOffset, void* val, void* compare,    \
      size_t bytes, core::atomicType amoType, int pe, int qpId) {                            \
    bool need_turn{true};                                                                    \
    uint64_t turns = __ballot(need_turn);                                                    \
    T result{};                                                                              \
    while (turns) {                                                                          \
      uint8_t lane = __ffsll((unsigned long long)turns) - 1;                                 \
      int pe_turn = __shfl(pe, lane);                                                        \
      if (pe_turn == pe) {                                                                   \
        result = DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_RETURN(                            \
            ShmemAtomicTypeFetchThreadKernelImpl, T, dest, destOffset, val, compare, bytes,  \
            amoType, pe, qpId);                                                              \
        need_turn = false;                                                                   \
      }                                                                                      \
      turns = __ballot(need_turn);                                                           \
    }                                                                                        \
    return result;                                                                           \
  }

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL(Uint32, uint32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL(Uint64, uint64_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL(Int32, int32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL(Int64, int64_t)

template <core::ProviderType PrvdType, typename T>
inline __device__ T ShmemAtomicTypeFetchWarpKernelImpl(const application::SymmMemObjPtr dest,
                                                       size_t destOffset, void* val, void* compare,
                                                       size_t bytes, core::atomicType amoType,
                                                       int pe, int qpId) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    return ShmemAtomicTypeFetchThreadKernelImpl<PrvdType, T>(dest, destOffset, val, compare, bytes,
                                                             amoType, pe, qpId);
  }
  return T{};
}

#define DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL(TypeName, T)                                   \
  template <>                                                                                     \
  inline __device__ T ShmemAtomicTypeFetchWarpKernel<application::TransportType::RDMA, T>(        \
      const application::SymmMemObjPtr dest, size_t destOffset, void* val, void* compare,         \
      size_t bytes, core::atomicType amoType, int pe, int qpId) {                                 \
    return DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_RETURN(ShmemAtomicTypeFetchWarpKernelImpl, T, \
                                                           dest, destOffset, val, compare, bytes, \
                                                           amoType, pe, qpId);                    \
  }

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL(Uint32, uint32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL(Uint64, uint64_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL(Int32, int32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL(Int64, int64_t)

/* ---------------------------------------------------------------------------------------------- */
/*                      Pure Address-Based RDMA Kernels (New API)                                 */
/* ---------------------------------------------------------------------------------------------- */

// Convert SHMEM heap address to remote address and key
// Supports both Static Heap and VMM Heap modes
inline __device__ void QueryRemoteAddr(const void* localAddr, int pe, uintptr_t& out_raddr,
                                       uint32_t& out_rkey) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  uintptr_t localAddrInt = reinterpret_cast<uintptr_t>(localAddr);

  if (globalGpuStates->heapObj == nullptr) {
    assert(false);
    out_raddr = 0;
    out_rkey = 0;
    return;
  }

  // Check if address is in symmetric heap range
  if (localAddrInt < globalGpuStates->heapBaseAddr ||
      localAddrInt >= globalGpuStates->heapEndAddr) {
    assert(false);
    out_raddr = 0;
    out_rkey = 0;
    return;
  }

  // Calculate offset within the symmetric heap
  size_t offset = localAddrInt - globalGpuStates->heapBaseAddr;
  application::SymmMemObj* heapObj = globalGpuStates->heapObj;

  if (globalGpuStates->useVMMHeap) {
    // VMM Heap: need to get chunk-specific rkey
    VmmLookupRemote(localAddrInt, pe, out_raddr, out_rkey);
  } else {
    // Static Heap: direct rkey access
    out_raddr = heapObj->peerPtrs[pe] + offset;
    out_rkey = heapObj->peerRkeys[pe];
  }
}

// New pure address-based PutMemNbi kernel for RDMA
template <core::ProviderType PrvdType>
inline __device__ void ShmemPutMemNbiThreadKernelAddrImpl(const void* dest, const void* source,
                                                          size_t bytes, int pe, int qpId) {
  if (bytes == 0) return;

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int epIndex = pe * globalGpuStates->numQpPerPe + (qpId % globalGpuStates->numQpPerPe);
  core::WorkQueueHandle* wq = &ep[epIndex].wqHandle;
  core::CompletionQueueHandle* cq = &ep[epIndex].cqHandle;
  uint32_t qpn = ep[epIndex].handle.qpn;

  // Determine if chunking is needed (VMM heap mode)
  bool needsChunking = globalGpuStates->useVMMHeap;

  uintptr_t srcStartAddr = reinterpret_cast<uintptr_t>(source);
  uintptr_t dstStartAddr = reinterpret_cast<uintptr_t>(dest);
  size_t remaining = bytes;
  size_t currentOffset = 0;

  // Main loop: process one chunk per iteration
  while (true) {
    // Check if current thread still has data to transfer
    bool has_remaining = (remaining > 0);

    // Synchronize within warp: get mask of threads that still have work
    uint64_t activemask = __ballot(has_remaining);
    if (activemask == 0) {
      break;  // All threads in warp are done
    }

    // Recalculate active lane info for threads with remaining data
    uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
    uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
    bool is_leader{my_logical_lane_id == num_active_lanes - 1};
    const uint64_t leader_phys_lane_id = core::GetLastActiveLaneID(activemask);

    // Inactive threads skip actual work but stay synchronized
    if (!has_remaining) {
      continue;
    }

    // Determine transfer size for this chunk
    uintptr_t srcAddr = srcStartAddr + currentOffset;
    uintptr_t dstAddr = dstStartAddr + currentOffset;
    uint32_t lkey = globalGpuStates->heapObj->lkey;
    uintptr_t raddr;
    uint32_t rkey;
    size_t transfer_size;

    if (!needsChunking) {
      // Static Heap: single transfer, direct rkey access
      transfer_size = remaining;
      size_t offset = dstAddr - globalGpuStates->heapBaseAddr;
      raddr = globalGpuStates->heapObj->peerPtrs[pe] + offset;
      rkey = globalGpuStates->heapObj->peerRkeys[pe];
    } else {
      // VMM Heap: get chunk-specific rkey and size
      size_t src_chunk_size, dst_chunk_size;
      VmmQueryLocalKey(srcAddr, remaining, lkey, src_chunk_size);
      VmmQueryRemoteAddr(dstAddr, pe, remaining, raddr, rkey, dst_chunk_size);
      transfer_size = src_chunk_size < dst_chunk_size ? src_chunk_size : dst_chunk_size;
    }
    MORI_PRINTF("blockIdx.x=%d, threadIdx.x=%d, remaining=%zu, transfer_size=%zu\n", blockIdx.x,
                threadIdx.x, remaining, transfer_size);
    uint32_t warp_sq_counter{0};
    uint32_t warp_msntbl_counter{0}, warp_psn_counter{0};
    uint32_t my_sq_counter{0}, my_msntbl_counter{0}, my_psn_counter{0};
    uint32_t psnCnt = 0;

    if constexpr (PrvdType == core::ProviderType::BNXT) {
      psnCnt = (transfer_size + wq->mtuSize - 1) / wq->mtuSize;
    }
    if (is_leader) {
      if constexpr (PrvdType == core::ProviderType::MLX5) {
        warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT);
      } else if constexpr (PrvdType == core::ProviderType::BNXT) {
        core::atomic_add_packed_msn_and_psn(&wq->msnPack, num_active_lanes,
                                            psnCnt * num_active_lanes, &warp_msntbl_counter,
                                            &warp_psn_counter);
        warp_sq_counter = warp_msntbl_counter;
        __hip_atomic_fetch_max(&wq->postIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
      } else if constexpr (PrvdType == core::ProviderType::PSD) {
        warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT);
      } else {
        static_assert(false);
      }
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    if constexpr (PrvdType == core::ProviderType::MLX5) {
      my_sq_counter = warp_sq_counter + my_logical_lane_id;
    } else if constexpr (PrvdType == core::ProviderType::BNXT) {
      warp_msntbl_counter = __shfl(warp_msntbl_counter, leader_phys_lane_id);
      warp_psn_counter = __shfl(warp_psn_counter, leader_phys_lane_id);
      my_sq_counter = warp_sq_counter + my_logical_lane_id;
      my_msntbl_counter = warp_msntbl_counter + my_logical_lane_id;
      my_psn_counter = warp_psn_counter + my_logical_lane_id * psnCnt;
    } else if constexpr (PrvdType == core::ProviderType::PSD) {
      my_sq_counter = warp_sq_counter + my_logical_lane_id;
    } else {
      static_assert(false);
    }

    while (true) {
      uint64_t db_touched =
          __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      uint64_t db_done =
          __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      uint64_t num_active_sq_entries = db_touched - db_done;
      uint64_t num_free_entries = wq->sqWqeNum - num_active_sq_entries;
      uint64_t num_entries_until_warp_last_entry = warp_sq_counter + num_active_lanes - db_touched;
      if (num_free_entries > num_entries_until_warp_last_entry) {
        break;
      }
      ShmemQuietThreadKernelImpl<PrvdType>(pe, qpId);
    }

    // Post RDMA write for this chunk
    uint64_t dbr_val;
    if constexpr (PrvdType == core::ProviderType::MLX5) {
      wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
      dbr_val =
          core::PostWrite<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, is_leader,
                                    qpn, srcAddr, lkey, raddr, rkey, transfer_size);
    } else if constexpr (PrvdType == core::ProviderType::BNXT) {
      wq->outstandingWqe[my_sq_counter % wq->sqWqeNum] = my_sq_counter;
      dbr_val =
          core::PostWrite<PrvdType>(*wq, my_sq_counter, my_msntbl_counter, my_psn_counter,
                                    is_leader, qpn, srcAddr, lkey, raddr, rkey, transfer_size);
    } else if constexpr (PrvdType == core::ProviderType::PSD) {
      wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
      dbr_val =
          core::PostWrite<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, is_leader,
                                    qpn, srcAddr, lkey, raddr, rkey, transfer_size);
    } else {
      static_assert(false);
    }

    __threadfence_system();
    if (is_leader) {
      uint64_t db_touched{0};
      do {
        db_touched = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      } while (db_touched != warp_sq_counter);

      core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, warp_sq_counter + num_active_lanes);
      __threadfence_system();
      core::RingDoorbell<PrvdType>(wq->dbrAddr, dbr_val);
      __threadfence_system();

      __hip_atomic_fetch_add(&cq->needConsIdx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(&wq->dbTouchIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
    __threadfence_system();

    // Move to next chunk
    currentOffset += transfer_size;
    remaining -= transfer_size;
  }
}

template <>
inline __device__ void ShmemPutMemNbiThreadKernel<application::TransportType::RDMA>(
    const void* dest, const void* source, size_t bytes, int pe, int qpId) {
  bool need_turn{true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemPutMemNbiThreadKernelAddrImpl, dest, source, bytes,
                                          pe, qpId);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemPutMemNbiWarpKernelAddrImpl(const void* dest, const void* source,
                                                        size_t bytes, int pe, int qpId) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemPutMemNbiThreadKernelAddrImpl<PrvdType>(dest, source, bytes, pe, qpId);
  }
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemPutMemNbiBlockKernelAddrImpl(const void* dest, const void* source,
                                                         size_t bytes, int pe, int qpId) {
  if (core::FlatBlockThreadId() == 0) {
    ShmemPutMemNbiThreadKernelAddrImpl<PrvdType>(dest, source, bytes, pe, qpId);
  }
}

template <>
inline __device__ void ShmemPutMemNbiWarpKernel<application::TransportType::RDMA>(
    const void* dest, const void* source, size_t bytes, int pe, int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemPutMemNbiWarpKernelAddrImpl, dest, source, bytes, pe,
                                      qpId);
}

template <>
inline __device__ void ShmemPutMemNbiBlockKernel<application::TransportType::RDMA>(
    const void* dest, const void* source, size_t bytes, int pe, int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemPutMemNbiBlockKernelAddrImpl, dest, source, bytes, pe,
                                      qpId);
}

// New pure address-based PutSizeImmNbi kernel for RDMA
template <core::ProviderType PrvdType>
inline __device__ void ShmemPutSizeImmNbiThreadKernelAddrImpl(const void* dest, void* val,
                                                              size_t bytes, int pe, int qpId) {
  if (bytes == 0) return;

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int epIndex = pe * globalGpuStates->numQpPerPe + (qpId % globalGpuStates->numQpPerPe);
  core::WorkQueueHandle* wq = &ep[epIndex].wqHandle;
  core::CompletionQueueHandle* cq = &ep[epIndex].cqHandle;
  uint32_t qpn = ep[epIndex].handle.qpn;

  // Convert addresses to remote addresses (supports both Static Heap and VMM Heap)
  uintptr_t raddr;
  uint32_t rkey;
  QueryRemoteAddr(dest, pe, raddr, rkey);

  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader{my_logical_lane_id == num_active_lanes - 1};
  const uint64_t leader_phys_lane_id = core::GetLastActiveLaneID(activemask);
  uint32_t warp_sq_counter{0};
  uint32_t warp_msntbl_counter{0}, warp_psn_counter{0};
  uint32_t my_sq_counter{0}, my_msntbl_counter{0}, my_psn_counter{0};

  if constexpr (PrvdType == core::ProviderType::MLX5) {
    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    if (is_leader) {
      core::atomic_add_packed_msn_and_psn(&wq->msnPack, num_active_lanes, num_active_lanes,
                                          &warp_msntbl_counter, &warp_psn_counter);
      warp_sq_counter = warp_msntbl_counter;
      __hip_atomic_fetch_max(&wq->postIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    warp_msntbl_counter = __shfl(warp_msntbl_counter, leader_phys_lane_id);
    warp_psn_counter = __shfl(warp_psn_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
    my_msntbl_counter = warp_msntbl_counter + my_logical_lane_id;
    my_psn_counter = warp_psn_counter + my_logical_lane_id;
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
  } else {
    static_assert(false);
  }

  while (true) {
    uint64_t db_touched =
        __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t db_done = __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t num_active_sq_entries = db_touched - db_done;
    uint64_t num_free_entries = wq->sqWqeNum - num_active_sq_entries;
    uint64_t num_entries_until_warp_last_entry = warp_sq_counter + num_active_lanes - db_touched;
    if (num_free_entries > num_entries_until_warp_last_entry) {
      break;
    }
    ShmemQuietThreadKernelImpl<PrvdType>(pe, qpId);
  }

  uint64_t dbr_val;
  if constexpr (PrvdType == core::ProviderType::MLX5) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
    dbr_val = core::PostWriteInline<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter,
                                              is_leader, qpn, val, raddr, rkey, bytes);
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    wq->outstandingWqe[my_sq_counter % wq->sqWqeNum] = my_sq_counter;
    dbr_val = core::PostWriteInline<PrvdType>(*wq, my_sq_counter, my_msntbl_counter, my_psn_counter,
                                              is_leader, qpn, val, raddr, rkey, bytes);
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
    dbr_val = core::PostWriteInline<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter,
                                              is_leader, qpn, val, raddr, rkey, bytes);
  } else {
    static_assert(false);
  }

  if (is_leader) {
    uint64_t db_touched = 0;
    do {
      db_touched = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != warp_sq_counter);

    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, warp_sq_counter + num_active_lanes);
    // __threadfence_system();
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbr_val);
    // __threadfence_system();

    __hip_atomic_fetch_add(&cq->needConsIdx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&wq->dbTouchIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }
}

template <>
inline __device__ void ShmemPutSizeImmNbiThreadKernel<application::TransportType::RDMA>(
    const void* dest, void* val, size_t bytes, int pe, int qpId) {
  bool need_turn{true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemPutSizeImmNbiThreadKernelAddrImpl, dest, val, bytes,
                                          pe, qpId);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemPutSizeImmNbiWarpKernelAddrImpl(const void* dest, void* val,
                                                            size_t bytes, int pe, int qpId) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemPutSizeImmNbiThreadKernelAddrImpl<PrvdType>(dest, val, bytes, pe, qpId);
  }
}

template <>
inline __device__ void ShmemPutSizeImmNbiWarpKernel<application::TransportType::RDMA>(
    const void* dest, void* val, size_t bytes, int pe, int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemPutSizeImmNbiWarpKernelAddrImpl, dest, val, bytes, pe,
                                      qpId);
}

template <core::ProviderType PrvdType, bool onlyOneSignal = true>
inline __device__ void ShmemPutMemNbiSignalThreadKernelAddrImpl(
    const void* dest, const void* source, size_t bytes, const void* signalDest,
    uint64_t signalValue, core::atomicType signalOp, int pe, int qpId) {
  if (bytes == 0) return;

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int epIndex = pe * globalGpuStates->numQpPerPe + (qpId % globalGpuStates->numQpPerPe);
  core::WorkQueueHandle* wq = &ep[epIndex].wqHandle;
  core::CompletionQueueHandle* cq = &ep[epIndex].cqHandle;
  uint32_t qpn = ep[epIndex].handle.qpn;

  // Determine if chunking is needed (VMM heap mode)
  bool needsChunking = globalGpuStates->useVMMHeap;

  // Query signal destination address once (signal always goes to same address)
  uintptr_t signalRaddr;
  uint32_t signalRkey;
  QueryRemoteAddr(signalDest, pe, signalRaddr, signalRkey);

  uintptr_t srcStartAddr = reinterpret_cast<uintptr_t>(source);
  uintptr_t dstStartAddr = reinterpret_cast<uintptr_t>(dest);
  size_t remaining = bytes;
  size_t currentOffset = 0;

  // Main loop: process one chunk per iteration
  while (true) {
    // Check if current thread still has data to transfer
    bool has_remaining = (remaining > 0);

    // Synchronize within warp: get mask of threads that still have work
    uint64_t activemask = __ballot(has_remaining);
    if (activemask == 0) {
      break;  // All threads in warp are done
    }

    // Recalculate active lane info for threads with remaining data
    uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
    uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
    bool is_leader{my_logical_lane_id == num_active_lanes - 1};
    const uint64_t leader_phys_lane_id = core::GetLastActiveLaneID(activemask);

    // Inactive threads skip actual work but stay synchronized
    if (!has_remaining) {
      continue;
    }

    // Determine transfer size for this chunk
    uintptr_t srcAddr = srcStartAddr + currentOffset;
    uintptr_t dstAddr = dstStartAddr + currentOffset;
    uint32_t lkey = globalGpuStates->heapObj->lkey;
    uintptr_t raddr;
    uint32_t rkey;
    size_t transfer_size;

    if (!needsChunking) {
      // Static Heap: single transfer, direct rkey access
      transfer_size = remaining;
      size_t offset = dstAddr - globalGpuStates->heapBaseAddr;
      raddr = globalGpuStates->heapObj->peerPtrs[pe] + offset;
      rkey = globalGpuStates->heapObj->peerRkeys[pe];
    } else {
      // VMM Heap: get chunk-specific rkey and size
      size_t src_chunk_size, dst_chunk_size;
      VmmQueryLocalKey(srcAddr, remaining, lkey, src_chunk_size);
      VmmQueryRemoteAddr(dstAddr, pe, remaining, raddr, rkey, dst_chunk_size);
      transfer_size = src_chunk_size < dst_chunk_size ? src_chunk_size : dst_chunk_size;
    }

    // Each thread checks if this is its last chunk
    bool my_is_last_chunk = (transfer_size == remaining);

    // Synchronize: only send signal if ALL active threads are on their last chunk
    // This ensures warp-uniform decision on num_wqes
    uint64_t all_last_mask = __ballot(my_is_last_chunk);
    bool isLastChunk = (all_last_mask == activemask);

    uint32_t warp_sq_counter{0};
    uint32_t warp_msntbl_counter{0}, warp_psn_counter{0};
    uint32_t my_sq_counter{0}, my_msntbl_counter{0}, my_psn_counter{0};
    uint32_t psnCnt = 0;
    // For last chunk: add 1 WQE for signal; for other chunks: just put
    uint32_t num_wqes = isLastChunk ? (onlyOneSignal ? num_active_lanes + 1 : num_active_lanes * 2)
                                    : num_active_lanes;

    if constexpr (PrvdType == core::ProviderType::BNXT) {
      psnCnt = (transfer_size + wq->mtuSize - 1) / wq->mtuSize;
    }
    if (is_leader) {
      if constexpr (PrvdType == core::ProviderType::MLX5) {
        warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_wqes, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT);
      } else if constexpr (PrvdType == core::ProviderType::BNXT) {
        uint32_t total_psn = psnCnt * num_active_lanes;
        if (isLastChunk) {
          total_psn += (onlyOneSignal ? 1 : num_active_lanes);
        }
        core::atomic_add_packed_msn_and_psn(&wq->msnPack, num_wqes, total_psn, &warp_msntbl_counter,
                                            &warp_psn_counter);
        warp_sq_counter = warp_msntbl_counter;
        __hip_atomic_fetch_max(&wq->postIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
      } else if constexpr (PrvdType == core::ProviderType::PSD) {
        warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_wqes, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT);
      } else {
        static_assert(false);
      }
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    if constexpr (PrvdType == core::ProviderType::MLX5) {
      my_sq_counter = warp_sq_counter +
                      (isLastChunk && !onlyOneSignal ? my_logical_lane_id * 2 : my_logical_lane_id);
    } else if constexpr (PrvdType == core::ProviderType::BNXT) {
      warp_msntbl_counter = __shfl(warp_msntbl_counter, leader_phys_lane_id);
      warp_psn_counter = __shfl(warp_psn_counter, leader_phys_lane_id);
      my_sq_counter = warp_sq_counter +
                      (isLastChunk && !onlyOneSignal ? my_logical_lane_id * 2 : my_logical_lane_id);
      my_msntbl_counter =
          warp_msntbl_counter +
          (isLastChunk && !onlyOneSignal ? my_logical_lane_id * 2 : my_logical_lane_id);
      my_psn_counter =
          warp_psn_counter + (isLastChunk && !onlyOneSignal ? (psnCnt + 1) * my_logical_lane_id
                                                            : psnCnt * my_logical_lane_id);
    } else if constexpr (PrvdType == core::ProviderType::PSD) {
      my_sq_counter = warp_sq_counter +
                      (isLastChunk && !onlyOneSignal ? my_logical_lane_id * 2 : my_logical_lane_id);
    } else {
      static_assert(false);
    }

    while (true) {
      uint64_t db_touched =
          __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      uint64_t db_done =
          __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      uint64_t num_active_sq_entries = db_touched - db_done;
      uint64_t num_free_entries = wq->sqWqeNum - num_active_sq_entries;
      uint64_t num_entries_until_warp_last_entry = warp_sq_counter + num_wqes - db_touched;
      if (num_free_entries > num_entries_until_warp_last_entry) {
        break;
      }
      ShmemQuietThreadKernelImpl<PrvdType>(pe, qpId);
    }

    // Post RDMA write for this chunk
    uint64_t dbr_val;
    if constexpr (PrvdType == core::ProviderType::MLX5) {
      wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
      dbr_val = core::PostWrite<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, false,
                                          qpn, srcAddr, lkey, raddr, rkey, transfer_size);
    } else if constexpr (PrvdType == core::ProviderType::BNXT) {
      wq->outstandingWqe[my_sq_counter % wq->sqWqeNum] = my_sq_counter;
      dbr_val = core::PostWrite<PrvdType>(*wq, my_sq_counter, my_msntbl_counter, my_psn_counter,
                                          false, qpn, srcAddr, lkey, raddr, rkey, transfer_size);
    } else if constexpr (PrvdType == core::ProviderType::PSD) {
      wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
      dbr_val = core::PostWrite<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, false,
                                          qpn, srcAddr, lkey, raddr, rkey, transfer_size);
    } else {
      static_assert(false);
    }

    // Post signal only for the last chunk
    if (isLastChunk) {
      if (signalOp == core::atomicType::AMO_SET || signalOp == core::atomicType::AMO_SIGNAL_SET) {
        // TODO: not support masked atomic yet, use write inline for now
        bool should_signal = onlyOneSignal ? is_leader : true;
        if (should_signal) {
          if constexpr (PrvdType == core::ProviderType::MLX5) {
            wq->outstandingWqe[(my_sq_counter + 1) % OUTSTANDING_TABLE_SIZE] = my_sq_counter + 1;
            dbr_val = core::PostWriteInline<PrvdType>(
                *wq, my_sq_counter + 1, my_sq_counter + 1, my_sq_counter + 1, is_leader, qpn,
                &signalValue, signalRaddr, signalRkey, sizeof(signalValue));
          } else if constexpr (PrvdType == core::ProviderType::BNXT) {
            wq->outstandingWqe[(my_sq_counter + 1) % wq->sqWqeNum] = my_sq_counter + 1;
            dbr_val = core::PostWriteInline<PrvdType>(
                *wq, my_sq_counter + 1, my_msntbl_counter + 1, my_psn_counter + psnCnt, is_leader,
                qpn, &signalValue, signalRaddr, signalRkey, sizeof(signalValue));
          } else if constexpr (PrvdType == core::ProviderType::PSD) {
            wq->outstandingWqe[(my_sq_counter + 1) % OUTSTANDING_TABLE_SIZE] = my_sq_counter + 1;
            dbr_val = core::PostWriteInline<PrvdType>(
                *wq, my_sq_counter + 1, my_sq_counter + 1, my_sq_counter + 1, is_leader, qpn,
                &signalValue, signalRaddr, signalRkey, sizeof(signalValue));
          }
        }
      } else if (signalOp == core::atomicType::AMO_ADD ||
                 signalOp == core::atomicType::AMO_SIGNAL_ADD) {
        core::IbufHandle* ibuf = &ep[epIndex].atomicIbuf;
        bool should_signal = onlyOneSignal ? is_leader : true;
        if (should_signal) {
          if constexpr (PrvdType == core::ProviderType::MLX5) {
            wq->outstandingWqe[(my_sq_counter + 1) % OUTSTANDING_TABLE_SIZE] = my_sq_counter + 1;
            dbr_val = core::PostAtomic<PrvdType>(
                *wq, my_sq_counter + 1, my_sq_counter + 1, my_sq_counter + 1, is_leader, qpn,
                ibuf->addr, ibuf->lkey, signalRaddr, signalRkey, &signalValue, &signalValue,
                sizeof(signalValue), core::atomicType::AMO_ADD);
          } else if constexpr (PrvdType == core::ProviderType::BNXT) {
            wq->outstandingWqe[(my_sq_counter + 1) % wq->sqWqeNum] = my_sq_counter + 1;
            dbr_val = core::PostAtomic<PrvdType>(
                *wq, my_sq_counter + 1, my_msntbl_counter + 1, my_psn_counter + psnCnt, is_leader,
                qpn, ibuf->addr, ibuf->lkey, signalRaddr, signalRkey, &signalValue, &signalValue,
                sizeof(signalValue), core::atomicType::AMO_ADD);
          } else if constexpr (PrvdType == core::ProviderType::PSD) {
            wq->outstandingWqe[(my_sq_counter + 1) % OUTSTANDING_TABLE_SIZE] = my_sq_counter + 1;
            dbr_val = core::PostAtomic<PrvdType>(
                *wq, my_sq_counter + 1, my_sq_counter + 1, my_sq_counter + 1, is_leader, qpn,
                ibuf->addr, ibuf->lkey, signalRaddr, signalRkey, &signalValue, &signalValue,
                sizeof(signalValue), core::atomicType::AMO_ADD);
          }
        }
      } else {
        assert(false && "signal unsupported atomic type");
      }
    }

    __threadfence_system();
    if (is_leader) {
      uint64_t db_touched{0};
      do {
        db_touched = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      } while (db_touched != warp_sq_counter);

      core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, warp_sq_counter + num_wqes);
      __threadfence_system();
      core::RingDoorbell<PrvdType>(wq->dbrAddr, dbr_val);
      __threadfence_system();

      __hip_atomic_fetch_add(&cq->needConsIdx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(&wq->dbTouchIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_AGENT);
    }
    __threadfence_system();

    // Move to next chunk
    currentOffset += transfer_size;
    remaining -= transfer_size;
  }
}

template <>
inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::RDMA, true>(
    const void* dest, const void* source, size_t bytes, const void* signalDest,
    uint64_t signalValue, core::atomicType signalOp, int pe, int qpId) {
  bool need_turn{true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_BOOL(ShmemPutMemNbiSignalThreadKernelAddrImpl, true,
                                                    dest, source, bytes, signalDest, signalValue,
                                                    signalOp, pe, qpId);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <>
inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::RDMA, false>(
    const void* dest, const void* source, size_t bytes, const void* signalDest,
    uint64_t signalValue, core::atomicType signalOp, int pe, int qpId) {
  bool need_turn{true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_BOOL(ShmemPutMemNbiSignalThreadKernelAddrImpl, false,
                                                    dest, source, bytes, signalDest, signalValue,
                                                    signalOp, pe, qpId);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <core::ProviderType PrvdType, bool onlyOneSignal = true>
inline __device__ void ShmemPutMemNbiSignalWarpKernelAddrImpl(const void* dest, const void* source,
                                                              size_t bytes, const void* signalDest,
                                                              uint64_t signalValue,
                                                              core::atomicType signalOp, int pe,
                                                              int qpId) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemPutMemNbiSignalThreadKernelAddrImpl<PrvdType, onlyOneSignal>(
        dest, source, bytes, signalDest, signalValue, signalOp, pe, qpId);
  }
}

template <core::ProviderType PrvdType, bool onlyOneSignal = true>
inline __device__ void ShmemPutMemNbiSignalBlockKernelAddrImpl(const void* dest, const void* source,
                                                               size_t bytes, const void* signalDest,
                                                               uint64_t signalValue,
                                                               core::atomicType signalOp, int pe,
                                                               int qpId) {
  if (core::FlatBlockThreadId() == 0) {
    ShmemPutMemNbiSignalThreadKernelAddrImpl<PrvdType, onlyOneSignal>(
        dest, source, bytes, signalDest, signalValue, signalOp, pe, qpId);
  }
}

template <>
inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::RDMA, true>(
    const void* dest, const void* source, size_t bytes, const void* signalDest,
    uint64_t signalValue, core::atomicType signalOp, int pe, int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_BOOL(ShmemPutMemNbiSignalWarpKernelAddrImpl, true, dest,
                                                source, bytes, signalDest, signalValue, signalOp,
                                                pe, qpId);
}

template <>
inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::RDMA, false>(
    const void* dest, const void* source, size_t bytes, const void* signalDest,
    uint64_t signalValue, core::atomicType signalOp, int pe, int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_BOOL(ShmemPutMemNbiSignalWarpKernelAddrImpl, false, dest,
                                                source, bytes, signalDest, signalValue, signalOp,
                                                pe, qpId);
}

template <>
inline __device__ void ShmemPutMemNbiSignalBlockKernel<application::TransportType::RDMA, true>(
    const void* dest, const void* source, size_t bytes, const void* signalDest,
    uint64_t signalValue, core::atomicType signalOp, int pe, int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_BOOL(ShmemPutMemNbiSignalBlockKernelAddrImpl, true, dest,
                                                source, bytes, signalDest, signalValue, signalOp,
                                                pe, qpId);
}

template <>
inline __device__ void ShmemPutMemNbiSignalBlockKernel<application::TransportType::RDMA, false>(
    const void* dest, const void* source, size_t bytes, const void* signalDest,
    uint64_t signalValue, core::atomicType signalOp, int pe, int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_BOOL(ShmemPutMemNbiSignalBlockKernelAddrImpl, false,
                                                dest, source, bytes, signalDest, signalValue,
                                                signalOp, pe, qpId);
}

// New pure address-based Atomic operations for RDMA
template <core::ProviderType PrvdType>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernelAddrImpl(const void* dest, void* val,
                                                                   size_t bytes,
                                                                   core::atomicType amoType, int pe,
                                                                   int qpId) {
  if (bytes == 0) return;

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int epIndex = pe * globalGpuStates->numQpPerPe + (qpId % globalGpuStates->numQpPerPe);
  core::WorkQueueHandle* wq = &ep[epIndex].wqHandle;
  core::CompletionQueueHandle* cq = &ep[epIndex].cqHandle;
  uint32_t qpn = ep[epIndex].handle.qpn;
  core::IbufHandle* ibuf = &ep[epIndex].atomicIbuf;

  // Convert addresses to remote addresses (supports both Static Heap and VMM Heap)
  uintptr_t raddr;
  uint32_t rkey;
  QueryRemoteAddr(dest, pe, raddr, rkey);
  uintptr_t laddr = ibuf->addr;
  uint32_t lkey = ibuf->lkey;

  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader{my_logical_lane_id == num_active_lanes - 1};
  const uint64_t leader_phys_lane_id = core::GetLastActiveLaneID(activemask);

  uint32_t warp_sq_counter = 0;
  uint32_t warp_msntbl_counter = 0, warp_psn_counter = 0;
  uint32_t my_sq_counter = 0, my_msntbl_counter = 0, my_psn_counter = 0;

  if constexpr (PrvdType == core::ProviderType::MLX5) {
    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    if (is_leader) {
      core::atomic_add_packed_msn_and_psn(&wq->msnPack, num_active_lanes, num_active_lanes,
                                          &warp_msntbl_counter, &warp_psn_counter);
      warp_sq_counter = warp_msntbl_counter;
      __hip_atomic_fetch_max(&wq->postIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    warp_msntbl_counter = __shfl(warp_msntbl_counter, leader_phys_lane_id);
    warp_psn_counter = __shfl(warp_psn_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
    my_msntbl_counter = warp_msntbl_counter + my_logical_lane_id;
    my_psn_counter = warp_psn_counter + my_logical_lane_id;
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
  } else {
    static_assert(false);
  }

  while (true) {
    uint64_t db_touched =
        __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t db_done = __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t num_active_sq_entries = db_touched - db_done;
    uint64_t num_free_entries = wq->sqWqeNum - num_active_sq_entries;
    uint64_t num_entries_until_warp_last_entry = warp_sq_counter + num_active_lanes - db_touched;
    if (num_free_entries > num_entries_until_warp_last_entry) break;
    ShmemQuietThreadKernelImpl<PrvdType>(pe, qpId);
  }

  if constexpr (PrvdType == core::ProviderType::MLX5) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    wq->outstandingWqe[my_sq_counter % wq->sqWqeNum] = my_sq_counter;
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
  }

  uint64_t dbr_val;
  if constexpr (PrvdType == core::ProviderType::MLX5) {
    dbr_val =
        core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, is_leader, qpn,
                                   laddr, lkey, raddr, rkey, val, val, bytes, amoType);
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    dbr_val =
        core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_msntbl_counter, my_psn_counter, is_leader,
                                   qpn, laddr, lkey, raddr, rkey, val, val, bytes, amoType);
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    dbr_val =
        core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, is_leader, qpn,
                                   laddr, lkey, raddr, rkey, val, val, bytes, amoType);
  }

  // __threadfence_system();
  if (is_leader) {
    uint64_t db_touched = 0;
    do {
      db_touched = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != warp_sq_counter);

    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, warp_sq_counter + num_active_lanes);
    // __threadfence_system();
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbr_val);

    __hip_atomic_fetch_add(&cq->needConsIdx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&wq->dbTouchIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::RDMA>(
    const void* dest, void* val, size_t bytes, core::atomicType amoType, int pe, int qpId) {
  bool need_turn{true};
  uint64_t turns = __ballot(need_turn);
  while (turns) {
    uint8_t lane = __ffsll((unsigned long long)turns) - 1;
    int pe_turn = __shfl(pe, lane);
    if (pe_turn == pe) {
      DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemAtomicSizeNonFetchThreadKernelAddrImpl, dest, val,
                                          bytes, amoType, pe, qpId);
      need_turn = false;
    }
    turns = __ballot(need_turn);
  }
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernelAddrImpl(const void* dest, void* val,
                                                                 size_t bytes,
                                                                 core::atomicType amoType, int pe,
                                                                 int qpId) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemAtomicSizeNonFetchThreadKernelAddrImpl<PrvdType>(dest, val, bytes, amoType, pe, qpId);
  }
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernel<application::TransportType::RDMA>(
    const void* dest, void* val, size_t bytes, core::atomicType amoType, int pe, int qpId) {
  DISPATCH_PROVIDER_TYPE_COMPILE_TIME(ShmemAtomicSizeNonFetchWarpKernelAddrImpl, dest, val, bytes,
                                      amoType, pe, qpId);
}

// New pure address-based Atomic Fetch operations for RDMA
template <core::ProviderType PrvdType, typename T>
inline __device__ T ShmemAtomicTypeFetchThreadKernelAddrImpl(const void* dest, void* val,
                                                             void* compare, size_t bytes,
                                                             core::atomicType amoType, int pe,
                                                             int qpId) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int epIndex = pe * globalGpuStates->numQpPerPe + (qpId % globalGpuStates->numQpPerPe);
  core::WorkQueueHandle* wq = &ep[epIndex].wqHandle;
  core::CompletionQueueHandle* cq = &ep[epIndex].cqHandle;
  uint32_t qpn = ep[epIndex].handle.qpn;
  core::IbufHandle* ibuf = &ep[epIndex].atomicIbuf;

  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader = (my_logical_lane_id == num_active_lanes - 1);
  uint64_t leader_phys_lane_id = core::GetLastActiveLaneID(activemask);

  uint32_t base_slot = 0;
  if (is_leader) {
    base_slot = ShmemGetAtomicIbufSlot(ibuf, num_active_lanes);
  }
  uint32_t my_slot = __shfl(base_slot, leader_phys_lane_id) + my_logical_lane_id;
  uint32_t my_slot_index = my_slot & (ibuf->nslots - 1);
  uintptr_t laddr = ibuf->addr + (my_slot_index + 1) * application::ATOMIC_IBUF_SLOT_SIZE;
  uint32_t lkey = ibuf->lkey;

  // Convert addresses to remote addresses (supports both Static Heap and VMM Heap)
  uintptr_t raddr;
  uint32_t rkey;
  QueryRemoteAddr(dest, pe, raddr, rkey);

  uint32_t warp_sq_counter = 0;
  uint32_t warp_msntbl_counter = 0, warp_psn_counter = 0;
  uint32_t my_sq_counter = 0, my_msntbl_counter = 0, my_psn_counter = 0;

  if constexpr (PrvdType == core::ProviderType::MLX5) {
    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    if (is_leader) {
      core::atomic_add_packed_msn_and_psn(&wq->msnPack, num_active_lanes, num_active_lanes,
                                          &warp_msntbl_counter, &warp_psn_counter);
      warp_sq_counter = warp_msntbl_counter;
      __hip_atomic_fetch_max(&wq->postIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    warp_msntbl_counter = __shfl(warp_msntbl_counter, leader_phys_lane_id);
    warp_psn_counter = __shfl(warp_psn_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
    my_msntbl_counter = warp_msntbl_counter + my_logical_lane_id;
    my_psn_counter = warp_psn_counter + my_logical_lane_id;
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_active_lanes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
  } else {
    static_assert(false);
  }

  while (true) {
    uint64_t db_touched =
        __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t db_done = __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t num_active_sq_entries = db_touched - db_done;
    uint64_t num_free_entries = wq->sqWqeNum - num_active_sq_entries;
    uint64_t num_entries_until_warp_last_entry = warp_sq_counter + num_active_lanes - db_touched;
    if (num_free_entries > num_entries_until_warp_last_entry) break;
    ShmemQuietThreadKernelImpl<PrvdType>(pe, qpId);
  }

  if constexpr (PrvdType == core::ProviderType::MLX5) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    wq->outstandingWqe[my_sq_counter % wq->sqWqeNum] = my_sq_counter;
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
  }

  uint64_t dbr_val;
  if constexpr (PrvdType == core::ProviderType::MLX5) {
    dbr_val =
        core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, is_leader, qpn,
                                   laddr, lkey, raddr, rkey, val, compare, bytes, amoType);
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    dbr_val =
        core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_msntbl_counter, my_psn_counter, is_leader,
                                   qpn, laddr, lkey, raddr, rkey, val, compare, bytes, amoType);
  } else if constexpr (PrvdType == core::ProviderType::PSD) {
    dbr_val =
        core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, is_leader, qpn,
                                   laddr, lkey, raddr, rkey, val, compare, bytes, amoType);
  }

  // __threadfence_system();
  if (is_leader) {
    uint64_t db_touched = 0;
    do {
      db_touched = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != warp_sq_counter);

    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, warp_sq_counter + num_active_lanes);
    // __threadfence_system();
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbr_val);

    __hip_atomic_fetch_add(&cq->needConsIdx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&wq->dbTouchIdx, warp_sq_counter + num_active_lanes, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }

  ShmemQuietThreadKernelImpl<PrvdType>(pe, qpId);
  T ret = *reinterpret_cast<volatile T*>(laddr);
  if (sizeof(T) == 4) ret = BSWAP32((uint32_t)ret);

  if (is_leader) {
    ShmemReleaseAtomicIbufSlot(ibuf, base_slot, num_active_lanes);
  }

  return ret;
}

#define DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL_RDMA_ADDR(TypeName, T)                       \
  template <>                                                                                     \
  inline __device__ T ShmemAtomicTypeFetchThreadKernel<application::TransportType::RDMA, T>(      \
      const void* dest, void* val, void* compare, size_t bytes, core::atomicType amoType, int pe, \
      int qpId) {                                                                                 \
    bool need_turn{true};                                                                         \
    uint64_t turns = __ballot(need_turn);                                                         \
    T result{};                                                                                   \
    while (turns) {                                                                               \
      uint8_t lane = __ffsll((unsigned long long)turns) - 1;                                      \
      int pe_turn = __shfl(pe, lane);                                                             \
      if (pe_turn == pe) {                                                                        \
        result = DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_RETURN(                                 \
            ShmemAtomicTypeFetchThreadKernelAddrImpl, T, dest, val, compare, bytes, amoType, pe,  \
            qpId);                                                                                \
        need_turn = false;                                                                        \
      }                                                                                           \
      turns = __ballot(need_turn);                                                                \
    }                                                                                             \
    return result;                                                                                \
  }

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL_RDMA_ADDR(Uint32, uint32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL_RDMA_ADDR(Uint64, uint64_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL_RDMA_ADDR(Int32, int32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL_RDMA_ADDR(Int64, int64_t)

template <core::ProviderType PrvdType, typename T>
inline __device__ T ShmemAtomicTypeFetchWarpKernelAddrImpl(const void* dest, void* val,
                                                           void* compare, size_t bytes,
                                                           core::atomicType amoType, int pe,
                                                           int qpId) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    return ShmemAtomicTypeFetchThreadKernelAddrImpl<PrvdType, T>(dest, val, compare, bytes, amoType,
                                                                 pe, qpId);
  }
  return T{};
}

#define DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL_RDMA_ADDR(TypeName, T)                         \
  template <>                                                                                     \
  inline __device__ T ShmemAtomicTypeFetchWarpKernel<application::TransportType::RDMA, T>(        \
      const void* dest, void* val, void* compare, size_t bytes, core::atomicType amoType, int pe, \
      int qpId) {                                                                                 \
    return DISPATCH_PROVIDER_TYPE_COMPILE_TIME_WITH_RETURN(                                       \
        ShmemAtomicTypeFetchWarpKernelAddrImpl, T, dest, val, compare, bytes, amoType, pe, qpId); \
  }

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL_RDMA_ADDR(Uint32, uint32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL_RDMA_ADDR(Uint64, uint64_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL_RDMA_ADDR(Int32, int32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL_RDMA_ADDR(Int64, int64_t)

}  // namespace shmem
}  // namespace mori
