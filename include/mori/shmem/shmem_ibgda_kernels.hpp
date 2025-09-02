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
#include "mori/shmem/shmem_api.hpp"
#include "src/shmem/internal.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ void ShmemQuietThreadKernelImpl(int pe) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int rank = globalGpuStates->rank;
  application::CompletionQueueHandle& cq = ep[pe].cqHandle;
  application::WorkQueueHandle& wq = ep[pe].wqHandle;
  core::ProviderType prvdType = ep[pe].GetProviderType();

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
      uint16_t wqe_counter;
      if (prvdType == core::ProviderType::MLX5) {
        int opcode = core::PollCq<core::ProviderType::MLX5>(cq.cqAddr, cq.cqeNum, &my_cq_consumer,
                                                            &wqe_counter);
        __threadfence_system();
        if (opcode == MLX5_CQE_RESP_ERR || opcode == MLX5_CQE_REQ_ERR) {
          printf("rank %d dest pe %d consIdx %d opcode %d\n", rank, pe, my_cq_index, opcode);
          core::DumpWqe(wq.sqAddr, my_cq_index);
          assert(false);
        }
      } else {
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
      if (prvdType == core::ProviderType::MLX5) {
        core::UpdateCqDbrRecord<core::ProviderType::MLX5>(
            cq.dbrRecAddr, (uint32_t)(warp_cq_consumer + quiet_amount));
      } else {
        assert(false);
      }
      __atomic_signal_fence(__ATOMIC_SEQ_CST);
      uint64_t doneIdx = wqe_broadcast[warp_id];
      __hip_atomic_fetch_max(&wq.doneIdx, doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_fetch_add(&cq.consIdx, quiet_amount, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::RDMA>() {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int rank = globalGpuStates->rank;
  int worldSize = globalGpuStates->worldSize;
  for (int pe = 0; pe < worldSize; pe++) {
    if (pe == rank) continue;
    if (globalGpuStates->transportTypes[pe] != application::TransportType::RDMA) continue;
    ShmemQuietThreadKernelImpl(pe);
  }
}

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::RDMA>(int pe) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int rank = globalGpuStates->rank;
  if (pe == rank) return;
  if (globalGpuStates->transportTypes[pe] != application::TransportType::RDMA) return;
  ShmemQuietThreadKernelImpl(pe);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Point-to-Point                                         */
/* ---------------------------------------------------------------------------------------------- */
template <core::ProviderType PrvdType>
inline __device__ void ShmemPutMemNbiThreadKernelImpl(const application::SymmMemObjPtr dest,
                                                      size_t destOffset,
                                                      const application::RdmaMemoryRegion& source,
                                                      size_t sourceOffset, size_t bytes, int pe) {
  if (bytes == 0) return;
  uintptr_t laddr = source.addr + sourceOffset;
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uintptr_t rkey = dest->peerRkeys[pe];

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  int rank = globalGpuStates->rank;
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  application::WorkQueueHandle& wq = ep[pe].wqHandle;
  application::CompletionQueueHandle& cq = ep[pe].cqHandle;

  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader{my_logical_lane_id == 0};
  const uint64_t leader_phys_lane_id = core::GetFirstActiveLaneID(activemask);
  uint8_t num_wqes{num_active_lanes};
  uint64_t warp_sq_counter{0};

  if (is_leader) {
    warp_sq_counter =
        __hip_atomic_fetch_add(&wq.postIdx, num_wqes, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
  uint64_t my_sq_counter = warp_sq_counter + my_logical_lane_id;
  uint64_t my_sq_index = my_sq_counter % wq.sqWqeNum;
  while (true) {
    uint64_t db_touched =
        __hip_atomic_load(&wq.dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t db_done = __hip_atomic_load(&wq.doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t num_active_sq_entries = db_touched - db_done;
    uint64_t num_free_entries = min(wq.sqWqeNum, cq.cqeNum) - num_active_sq_entries;
    uint64_t num_entries_until_warp_last_entry = warp_sq_counter + num_active_lanes - db_touched;
    if (num_free_entries > num_entries_until_warp_last_entry) {
      break;
    }
    ShmemQuietThreadKernelImpl(pe);
  }
  wq.outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
  uint64_t dbr_val =
      core::PostWrite<PrvdType>(wq.sqAddr, wq.sqWqeNum, nullptr, my_sq_counter, ep[pe].handle.qpn,
                                laddr, source.lkey, raddr, rkey, bytes);

  if (is_leader) {
    uint64_t db_touched{0};
    do {
      db_touched = __hip_atomic_load(&wq.dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != warp_sq_counter);

    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(wq.sqAddr);
    uint64_t* ctrl_wqe_8B_for_db = reinterpret_cast<uint64_t*>(
        &base_ptr[64 * ((warp_sq_counter + num_wqes - 1) % wq.sqWqeNum)]);
    core::UpdateSendDbrRecord<PrvdType>(wq.dbrRecAddr, warp_sq_counter + num_wqes);
    __threadfence_system();
    core::RingDoorbell<PrvdType>(wq.dbrAddr, *ctrl_wqe_8B_for_db);

    __hip_atomic_fetch_add(&cq.needConsIdx, num_wqes, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&wq.dbTouchIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }
  __threadfence_system();
#if DEBUG == 1
  if (rank == 0)
    printf("rank=%d, blockIdx.x=%d, threadIdx.x=%d, needConsIdx=%u, bytes=%lu, pe=%d\n", rank,
           blockIdx.x, threadIdx.x,
           __hip_atomic_load(&cq.needConsIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT), bytes,
           pe);
#endif
}

#define DISPATCH_PROVIDER_TYPE(func, ...)                         \
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();           \
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints; \
  core::ProviderType prvdType = ep[pe].GetProviderType();         \
  if (prvdType == core::ProviderType::MLX5) {                     \
    func<core::ProviderType::MLX5>(__VA_ARGS__);                  \
  } else {                                                        \
    assert(false);                                                \
  }

template <>
inline __device__ void ShmemPutMemNbiThreadKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, size_t bytes, int pe) {
  DISPATCH_PROVIDER_TYPE(ShmemPutMemNbiThreadKernelImpl, dest, destOffset, source, sourceOffset,
                         bytes, pe);
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemPutMemNbiWarpKernelImpl(const application::SymmMemObjPtr dest,
                                                    size_t destOffset,
                                                    const application::RdmaMemoryRegion& source,
                                                    size_t sourceOffset, size_t bytes, int pe) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemPutMemNbiThreadKernelImpl<PrvdType>(dest, destOffset, source, sourceOffset, bytes, pe);
  }
}

template <>
inline __device__ void ShmemPutMemNbiWarpKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, size_t bytes, int pe) {
  DISPATCH_PROVIDER_TYPE(ShmemPutMemNbiWarpKernelImpl, dest, destOffset, source, sourceOffset,
                         bytes, pe);
}

// TODO: deal with bytes count limit
// TODO: put size api only support 1,2,4,8,16 in nvshmem, should we do that?
template <core::ProviderType PrvdType>
inline __device__ void ShmemPutSizeImmNbiThreadKernelImpl(const application::SymmMemObjPtr dest,
                                                          size_t destOffset, void* val,
                                                          size_t bytes, int pe) {
  if (bytes == 0) return;

  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uintptr_t rkey = dest->peerRkeys[pe];

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  int rank = globalGpuStates->rank;
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  application::WorkQueueHandle& wq = ep[pe].wqHandle;
  application::CompletionQueueHandle& cq = ep[pe].cqHandle;

  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader = (my_logical_lane_id == 0);
  uint64_t leader_phys_lane_id = core::GetFirstActiveLaneID(activemask);
  uint8_t num_wqes = num_active_lanes;

  uint64_t warp_sq_counter = 0;
  if (is_leader) {
    warp_sq_counter =
        __hip_atomic_fetch_add(&wq.postIdx, num_wqes, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);

  uint64_t my_sq_counter = warp_sq_counter + my_logical_lane_id;
  uint64_t my_sq_index = my_sq_counter % wq.sqWqeNum;

  while (true) {
    uint64_t db_touched =
        __hip_atomic_load(&wq.dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t db_done = __hip_atomic_load(&wq.doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t num_active_sq_entries = db_touched - db_done;
    uint64_t num_free_entries = min(wq.sqWqeNum, cq.cqeNum) - num_active_sq_entries;
    uint64_t num_entries_until_warp_finish = (warp_sq_counter + num_active_lanes) - db_touched;
    if (num_free_entries > num_entries_until_warp_finish) {
      break;
    }
    ShmemQuietThreadKernelImpl(pe);
  }

  wq.outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;

  uint64_t dbr_val = core::PostWriteInline<PrvdType>(wq.sqAddr, wq.sqWqeNum, nullptr, my_sq_counter,
                                                     ep[pe].handle.qpn, val, raddr, rkey, bytes);

  if (is_leader) {
    uint64_t db_touched{0};
    do {
      db_touched = __hip_atomic_load(&wq.dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != warp_sq_counter);

    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(wq.sqAddr);
    uint64_t* ctrl_wqe_8B_for_db = reinterpret_cast<uint64_t*>(
        &base_ptr[64 * ((warp_sq_counter + num_wqes - 1) % wq.sqWqeNum)]);

    core::UpdateSendDbrRecord<PrvdType>(wq.dbrRecAddr, warp_sq_counter + num_wqes);
    __threadfence_system();
    core::RingDoorbell<PrvdType>(wq.dbrAddr, *ctrl_wqe_8B_for_db);

    __hip_atomic_fetch_add(&cq.needConsIdx, num_wqes, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&wq.dbTouchIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }
  __threadfence_system();
#if DEBUG==1
  if (rank == 0)
    printf("IMM rank=%d, blockIdx.x=%d, threadIdx.x=%d, needConsIdx=%u, bytes=%lu, pe=%d\n", rank,
           blockIdx.x, threadIdx.x,
           __hip_atomic_load(&cq.needConsIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT), bytes,
           pe);
#endif
}

template <>
inline __device__ void ShmemPutSizeImmNbiThreadKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe) {
  DISPATCH_PROVIDER_TYPE(ShmemPutSizeImmNbiThreadKernelImpl, dest, destOffset, val, bytes, pe);
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemPutSizeImmNbiWarpKernelImpl(const application::SymmMemObjPtr dest,
                                                        size_t destOffset, void* val, size_t bytes,
                                                        int pe) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemPutSizeImmNbiThreadKernelImpl<PrvdType>(dest, destOffset, val, bytes, pe);
  }
}

template <>
inline __device__ void ShmemPutSizeImmNbiWarpKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe) {
  DISPATCH_PROVIDER_TYPE(ShmemPutSizeImmNbiWarpKernelImpl, dest, destOffset, val, bytes, pe);
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernelImpl(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, void* val, size_t bytes,
    int pe, core::atomicType amoType) {
  if (bytes == 0) return;
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uintptr_t rkey = dest->peerRkeys[pe];
  uintptr_t laddr = source.addr + sourceOffset;
  uintptr_t lkey = source.lkey;

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  application::WorkQueueHandle& wq = ep[pe].wqHandle;
  application::CompletionQueueHandle& cq = ep[pe].cqHandle;
  uint32_t* lock = globalGpuStates->endpointLock;

  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader = (my_logical_lane_id == 0);
  uint64_t leader_phys_lane_id = core::GetFirstActiveLaneID(activemask);
  uint32_t numWqesPerCmd = core::get_num_wqes_in_atomic(amoType, bytes);
  uint8_t num_wqes = num_active_lanes * numWqesPerCmd;

  uint64_t warp_sq_counter = 0;
  if (is_leader) {
    warp_sq_counter =
        __hip_atomic_fetch_add(&wq.postIdx, num_wqes, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);

  uint64_t my_sq_counter = warp_sq_counter + my_logical_lane_id * numWqesPerCmd;
  uint64_t my_sq_index = my_sq_counter % wq.sqWqeNum;

  while (true) {
    uint64_t db_touched =
        __hip_atomic_load(&wq.dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t db_done = __hip_atomic_load(&wq.doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    int64_t num_active_sq_entries = db_touched - db_done;
    if (num_active_sq_entries < 0) {
      continue;
    }

    uint64_t num_free_entries = min(wq.sqWqeNum, cq.cqeNum * numWqesPerCmd) - num_active_sq_entries;
    uint64_t num_entries_until_warp_finish = (warp_sq_counter + num_wqes) - db_touched;
    if (num_free_entries > num_entries_until_warp_finish) break;
    ShmemQuietThreadKernelImpl(pe);
  }

  wq.outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter + numWqesPerCmd - 1;
  ;

  uint64_t dbr_val =
      core::PostAtomic<PrvdType>(wq.sqAddr, wq.sqWqeNum, nullptr, my_sq_counter, ep[pe].handle.qpn,
                                 laddr, lkey, raddr, rkey, val, 0, bytes, amoType);
  __threadfence_system();
  if (is_leader) {
    uint64_t db_touched = 0;
    do {
      db_touched = __hip_atomic_load(&wq.dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != warp_sq_counter);

    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(wq.sqAddr);
    uint64_t* ctrl_wqe_8B_for_db = reinterpret_cast<uint64_t*>(
        &base_ptr[64 * ((warp_sq_counter + num_wqes - numWqesPerCmd) % wq.sqWqeNum)]);

    core::UpdateSendDbrRecord<PrvdType>(wq.dbrRecAddr, warp_sq_counter + num_wqes);
    __threadfence_system();
    core::RingDoorbell<PrvdType>(wq.dbrAddr, *ctrl_wqe_8B_for_db);

    __hip_atomic_fetch_add(&cq.needConsIdx, num_active_lanes, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&wq.dbTouchIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }

  __threadfence_system();
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, void* val, size_t bytes,
    int pe, core::atomicType amoType) {
  DISPATCH_PROVIDER_TYPE(ShmemAtomicSizeNonFetchThreadKernelImpl, dest, destOffset, source,
                         sourceOffset, val, bytes, pe, amoType);
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernelImpl(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, void* val, size_t bytes,
    int pe, core::atomicType amoType) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemAtomicSizeNonFetchThreadKernelImpl<PrvdType>(dest, destOffset, source, sourceOffset, val,
                                                      bytes, pe, amoType);
  }
  // ShmemQuietThreadKernelImpl(pe);
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, void* val, size_t bytes,
    int pe, core::atomicType amoType) {
  DISPATCH_PROVIDER_TYPE(ShmemAtomicSizeNonFetchWarpKernelImpl, dest, destOffset, source,
                         sourceOffset, val, bytes, pe, amoType);
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemAtomicSizeFetchThreadKernelImpl(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, void* val, void* compare,
    size_t bytes, int pe, core::atomicType amoType) {
  if (bytes == 0) return;
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uintptr_t rkey = dest->peerRkeys[pe];
  uintptr_t laddr = source.addr + sourceOffset;
  uintptr_t lkey = source.lkey;

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  application::WorkQueueHandle& wq = ep[pe].wqHandle;
  application::CompletionQueueHandle& cq = ep[pe].cqHandle;

  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader = (my_logical_lane_id == 0);
  uint64_t leader_phys_lane_id = core::GetFirstActiveLaneID(activemask);
  uint32_t numWqesPerCmd = core::get_num_wqes_in_atomic(amoType, bytes);
  uint8_t num_wqes = num_active_lanes * numWqesPerCmd;

  uint64_t warp_sq_counter = 0;
  if (is_leader) {
    warp_sq_counter =
        __hip_atomic_fetch_add(&wq.postIdx, num_wqes, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);

  uint64_t my_sq_counter = warp_sq_counter + my_logical_lane_id * numWqesPerCmd;
  uint64_t my_sq_index = my_sq_counter % wq.sqWqeNum;

  while (true) {
    uint64_t db_touched =
        __hip_atomic_load(&wq.dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t db_done = __hip_atomic_load(&wq.doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    uint64_t num_active_sq_entries = db_touched - db_done;
    uint64_t num_free_entries = min(wq.sqWqeNum, cq.cqeNum) - num_active_sq_entries;
    uint64_t num_entries_until_warp_finish = (warp_sq_counter + num_active_lanes) - db_touched;
    if (num_free_entries > num_entries_until_warp_finish) {
      break;
    }
    ShmemQuietThreadKernelImpl(pe);
  }

  wq.outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter + numWqesPerCmd - 1;

  uint64_t dbr_val =
      core::PostAtomic<PrvdType>(wq.sqAddr, wq.sqWqeNum, nullptr, my_sq_counter, ep[pe].handle.qpn,
                                 laddr, lkey, raddr, rkey, val, compare, bytes, amoType);
  __threadfence_system();
  if (is_leader) {
    uint64_t db_touched = 0;
    do {
      db_touched = __hip_atomic_load(&wq.dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != warp_sq_counter);

    uint8_t* base_ptr = reinterpret_cast<uint8_t*>(wq.sqAddr);
    uint64_t* ctrl_wqe_8B_for_db = reinterpret_cast<uint64_t*>(
        &base_ptr[64 * ((warp_sq_counter + num_wqes - numWqesPerCmd) % wq.sqWqeNum)]);

    core::UpdateSendDbrRecord<PrvdType>(wq.dbrRecAddr, warp_sq_counter + num_wqes);
    __threadfence_system();
    core::RingDoorbell<PrvdType>(wq.dbrAddr, *ctrl_wqe_8B_for_db);

    __hip_atomic_fetch_add(&cq.needConsIdx, num_active_lanes, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&wq.dbTouchIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }

  __threadfence_system();
  // ShmemQuietThreadKernelImpl(pe);
}

template <>
inline __device__ void ShmemAtomicSizeFetchThreadKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, void* val, void* compare,
    size_t bytes, int pe, core::atomicType amoType) {
  DISPATCH_PROVIDER_TYPE(ShmemAtomicSizeFetchThreadKernelImpl, dest, destOffset, source,
                         sourceOffset, val, compare, bytes, pe, amoType);
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemAtomicSizeFetchWarpKernelImpl(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, void* val, void* compare,
    size_t bytes, int pe, core::atomicType amoType) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemAtomicSizeFetchThreadKernelImpl<PrvdType>(dest, destOffset, source, sourceOffset, val,
                                                   compare, bytes, pe, amoType);
  }
}

template <>
inline __device__ void ShmemAtomicSizeFetchWarpKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, void* val, void* compare,
    size_t bytes, int pe, core::atomicType amoType) {
  DISPATCH_PROVIDER_TYPE(ShmemAtomicSizeFetchWarpKernelImpl, dest, destOffset, source, sourceOffset,
                         val, compare, bytes, pe, amoType);
}

}  // namespace shmem
}  // namespace mori
