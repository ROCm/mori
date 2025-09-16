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

#ifdef ENABLE_BNXT
#define DISPATCH_MLX5 0
#define DISPATCH_BNXT 1
#else
#define DISPATCH_MLX5 1
#define DISPATCH_BNXT 0
#endif

#define DISPATCH_PROVIDER_TYPE(func, ...)                             \
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();               \
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;     \
  core::ProviderType prvdType = ep[pe].GetProviderType();             \
  if (DISPATCH_MLX5 && prvdType == core::ProviderType::MLX5) {        \
    func<core::ProviderType::MLX5>(__VA_ARGS__);                      \
  } else if (DISPATCH_BNXT && prvdType == core::ProviderType::BNXT) { \
    func<core::ProviderType::BNXT>(__VA_ARGS__);                      \
  } else {                                                            \
    assert(false && "Unsupported or disabled provider type");         \
  }

#define DISPATCH_PROVIDER_TYPE_EP(ep, func, ...)                      \
  core::ProviderType prvdType = ep[pe].GetProviderType();             \
  if (DISPATCH_MLX5 && prvdType == core::ProviderType::MLX5) {        \
    func<core::ProviderType::MLX5>(__VA_ARGS__);                      \
  } else if (DISPATCH_BNXT && prvdType == core::ProviderType::BNXT) { \
    func<core::ProviderType::BNXT>(__VA_ARGS__);                      \
  } else {                                                            \
    assert(false && "Unsupported or disabled provider type");         \
  }

/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */

template <core::ProviderType PrvdType>
inline __device__ void ShmemQuietThreadKernelSerialImpl(int pe) {
  if (threadIdx.x != 0) return;
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  core::CompletionQueueHandle& cq = ep[pe].cqHandle;
  core::WorkQueueHandle& wq = ep[pe].wqHandle;
  while (true) {
    bool done{false};
    uint32_t quiet_amount{0};
    uint32_t my_cq_consumer{0};

    uint32_t dbTouchIdx =
        __hip_atomic_load(&wq.dbTouchIdx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
    uint32_t doneIdx = __hip_atomic_load(&wq.doneIdx, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_AGENT);
    // printf("dbTouchIdx: %u, doneIdx: %u\n", dbTouchIdx, doneIdx);
    if (dbTouchIdx == doneIdx) {
      return;
    }

    my_cq_consumer =
        __hip_atomic_fetch_add(&cq.cq_consumer, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    uint16_t wqe_counter;
    uint64_t wqe_id;
    int opcode = core::PollCq<PrvdType>(cq.cqAddr, cq.cqeNum, &my_cq_consumer, &wqe_counter);
    if constexpr (PrvdType == core::ProviderType::MLX5) {
      if (opcode == MLX5_CQE_RESP_ERR || opcode == MLX5_CQE_REQ_ERR) {
        int rank = globalGpuStates->rank;
        uint32_t my_cq_index = my_cq_consumer % cq.cqeNum;
        printf("rank %d dest pe %d consIdx %d opcode %d\n", rank, pe, my_cq_index, opcode);
        core::DumpMlx5Wqe(wq.sqAddr, my_cq_index);
        assert(false);
      }
      wqe_id = wq.outstandingWqe[wqe_counter];
    } else if constexpr (PrvdType == core::ProviderType::BNXT) {
      if (opcode != BNXT_RE_REQ_ST_OK) {
        int rank = globalGpuStates->rank;
        uint32_t my_cq_index = my_cq_consumer % cq.cqeNum;
        printf("rank %d dest pe %d consIdx %d opcode %d\n", rank, pe, my_cq_index, opcode);
        assert(false);
      }
      wqe_counter = (wqe_counter + wq.sqWqeNum - 1) % wq.sqWqeNum;
      wqe_id = wq.outstandingWqe[wqe_counter] + 1;
    }

    // core::UpdateCqDbrRecord<PrvdType>(cq.dbrRecAddr, (uint32_t)(my_cq_consumer + 1), cq.cqeNum);

    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __hip_atomic_fetch_max(&wq.doneIdx, wqe_id, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemQuietThreadKernelImpl(int pe) {
  if constexpr (PrvdType == core::ProviderType::BNXT) {
    ShmemQuietThreadKernelSerialImpl<PrvdType>(pe);
    return;
  }
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  core::CompletionQueueHandle& cq = ep[pe].cqHandle;
  core::WorkQueueHandle& wq = ep[pe].wqHandle;

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
      int opcode = core::PollCq<PrvdType>(cq.cqAddr, cq.cqeNum, &my_cq_consumer, &wqe_counter);
      if constexpr (PrvdType == core::ProviderType::MLX5) {
        if (opcode == MLX5_CQE_RESP_ERR || opcode == MLX5_CQE_REQ_ERR) {
          int rank = globalGpuStates->rank;
          printf("rank %d dest pe %d consIdx %d opcode %d\n", rank, pe, my_cq_index, opcode);
          core::DumpMlx5Wqe(wq.sqAddr, my_cq_index);
          assert(false);
        }
      } else if constexpr (PrvdType == core::ProviderType::BNXT) {
        if (opcode != BNXT_RE_REQ_ST_OK) {
          int rank = globalGpuStates->rank;
          printf("rank %d dest pe %d consIdx %d opcode %d\n", rank, pe, my_cq_index, opcode);
          assert(false);
        }
        wqe_counter = (BNXT_RE_NUM_SLOT_PER_WQE * (wqe_counter + wq.sqWqeNum - 1) % wq.sqWqeNum);
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

      core::UpdateCqDbrRecord<PrvdType>(cq.dbrRecAddr, (uint32_t)(warp_cq_consumer + quiet_amount),
                                        cq.cqeNum);

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
  for (int pe = blockIdx.x; pe < worldSize; pe += gridDim.x) {
    if (pe != rank && globalGpuStates->transportTypes[pe] == application::TransportType::RDMA) {
      DISPATCH_PROVIDER_TYPE_EP(ep, ShmemQuietThreadKernelImpl, pe);
    }
  }
}

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::RDMA>(int pe) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  int rank = globalGpuStates->rank;
  if (pe == rank) return;
  if (globalGpuStates->transportTypes[pe] != application::TransportType::RDMA) return;
  DISPATCH_PROVIDER_TYPE_EP(ep, ShmemQuietThreadKernelImpl, pe);
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
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  core::WorkQueueHandle* wq = &ep[pe].wqHandle;
  core::CompletionQueueHandle* cq = &ep[pe].cqHandle;

  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader{my_logical_lane_id == num_active_lanes - 1};
  const uint64_t leader_phys_lane_id = core::GetLastActiveLaneID(activemask);
  uint8_t num_wqes{num_active_lanes};
  uint32_t warp_sq_counter{0};
  uint32_t warp_msntbl_counter{0}, warp_psn_counter{0};
  uint32_t my_sq_counter{0}, my_msntbl_counter{0}, my_psn_counter{0};
  uint32_t psnCnt = 0;

  if constexpr (PrvdType == core::ProviderType::BNXT) {
    psnCnt = (bytes + wq->mtuSize - 1) / wq->mtuSize;
  }
  if (is_leader) {
    if constexpr (PrvdType == core::ProviderType::MLX5) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_wqes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    } else if constexpr (PrvdType == core::ProviderType::BNXT) {
      core::atomic_add_packed_msn_and_psn(&wq->msnPack, num_wqes, psnCnt * num_wqes,
                                          &warp_msntbl_counter, &warp_psn_counter);
      warp_sq_counter = warp_msntbl_counter;
      __hip_atomic_fetch_max(&wq->postIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
    } else {
      assert(false);
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
  } else {
    assert(false);
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
    ShmemQuietThreadKernelImpl<PrvdType>(pe);
  }
  uint64_t dbr_val;
  if constexpr (PrvdType == core::ProviderType::MLX5) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
    dbr_val = core::PostWrite<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, is_leader,
                                        ep[pe].handle.qpn, laddr, source.lkey, raddr, rkey, bytes);
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    wq->outstandingWqe[my_sq_counter % wq->sqWqeNum] = my_sq_counter;
    dbr_val =
        core::PostWrite<PrvdType>(*wq, my_sq_counter, my_msntbl_counter, my_psn_counter, is_leader,
                                  ep[pe].handle.qpn, laddr, source.lkey, raddr, rkey, bytes);
  } else {
    assert(false);
  }
  // __threadfence_system();
  if (is_leader) {
    uint64_t db_touched{0};
    do {
      db_touched = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != warp_sq_counter);

    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, warp_sq_counter + num_wqes);
    // __threadfence_system();
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbr_val);
    __threadfence_system();

    __hip_atomic_fetch_add(&cq->needConsIdx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&wq->dbTouchIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }
  // __threadfence_system();
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
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  core::WorkQueueHandle* wq = &ep[pe].wqHandle;
  core::CompletionQueueHandle* cq = &ep[pe].cqHandle;

  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader{my_logical_lane_id == num_active_lanes - 1};
  const uint64_t leader_phys_lane_id = core::GetLastActiveLaneID(activemask);
  uint8_t num_wqes{num_active_lanes};
  uint32_t warp_sq_counter{0};
  uint32_t warp_msntbl_counter{0}, warp_psn_counter{0};
  uint32_t my_sq_counter{0}, my_msntbl_counter{0}, my_psn_counter{0};

  if constexpr (PrvdType == core::ProviderType::MLX5) {
    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_wqes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    if (is_leader) {
      core::atomic_add_packed_msn_and_psn(&wq->msnPack, num_wqes, num_wqes, &warp_msntbl_counter,
                                          &warp_psn_counter);
      warp_sq_counter = warp_msntbl_counter;
      __hip_atomic_fetch_max(&wq->postIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    warp_msntbl_counter = __shfl(warp_msntbl_counter, leader_phys_lane_id);
    warp_psn_counter = __shfl(warp_psn_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
    my_msntbl_counter = warp_msntbl_counter + my_logical_lane_id;
    my_psn_counter = warp_psn_counter + my_logical_lane_id;
  } else {
    assert(false);
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
    ShmemQuietThreadKernelImpl<PrvdType>(pe);
  }

  uint64_t dbr_val;
  if constexpr (PrvdType == core::ProviderType::MLX5) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] = my_sq_counter;
    dbr_val =
        core::PostWriteInline<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter, is_leader,
                                        ep[pe].handle.qpn, val, raddr, rkey, bytes);
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    wq->outstandingWqe[my_sq_counter % wq->sqWqeNum] = my_sq_counter;
    dbr_val =
        core::PostWriteInline<PrvdType>(*wq, my_sq_counter, my_msntbl_counter, my_psn_counter,
                                        is_leader, ep[pe].handle.qpn, val, raddr, rkey, bytes);
  } else {
    assert(false);
  }
  // __threadfence_system();
  if (is_leader) {
    uint64_t db_touched = 0;
    do {
      db_touched = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != warp_sq_counter);

    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, warp_sq_counter + num_wqes);
    // __threadfence_system();
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbr_val);
    __threadfence_system();

    __hip_atomic_fetch_add(&cq->needConsIdx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&wq->dbTouchIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }
  // __threadfence_system();
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
  core::WorkQueueHandle* wq = &ep[pe].wqHandle;
  core::CompletionQueueHandle* cq = &ep[pe].cqHandle;

  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader{my_logical_lane_id == num_active_lanes - 1};
  const uint64_t leader_phys_lane_id = core::GetLastActiveLaneID(activemask);

  uint32_t warp_sq_counter = 0;
  uint32_t warp_msntbl_counter = 0, warp_psn_counter = 0;
  uint32_t my_sq_counter = 0, my_msntbl_counter = 0, my_psn_counter = 0;
  uint8_t num_wqes;

  if constexpr (PrvdType == core::ProviderType::MLX5) {
    uint32_t numWqesPerCmd = core::get_num_wqes_in_atomic(amoType, bytes);
    num_wqes = num_active_lanes * numWqesPerCmd;
    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_wqes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id * numWqesPerCmd;
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    num_wqes = num_active_lanes;
    if (is_leader) {
      core::atomic_add_packed_msn_and_psn(&wq->msnPack, num_wqes, num_wqes, &warp_msntbl_counter,
                                          &warp_psn_counter);
      warp_sq_counter = warp_msntbl_counter;
      __hip_atomic_fetch_max(&wq->postIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    warp_msntbl_counter = __shfl(warp_msntbl_counter, leader_phys_lane_id);
    warp_psn_counter = __shfl(warp_psn_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
    my_msntbl_counter = warp_msntbl_counter + my_logical_lane_id;
    my_psn_counter = warp_psn_counter + my_logical_lane_id;
  } else {
    assert(false);
  }

  while (true) {
    uint64_t db_touched =
        __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t db_done = __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t num_active_sq_entries = db_touched - db_done;
    uint64_t num_free_entries = wq->sqWqeNum - num_active_sq_entries;
    uint64_t num_entries_until_warp_last_entry = warp_sq_counter + num_wqes - db_touched;
    if (num_free_entries > num_entries_until_warp_last_entry) break;
    ShmemQuietThreadKernelImpl<PrvdType>(pe);
  }

  if constexpr (PrvdType == core::ProviderType::MLX5) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] =
        my_sq_counter + core::get_num_wqes_in_atomic(amoType, bytes) - 1;
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    wq->outstandingWqe[my_sq_counter % wq->sqWqeNum] = my_sq_counter;
  }

  uint64_t dbr_val;
  if constexpr (PrvdType == core::ProviderType::MLX5) {
    dbr_val = core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter,
                                         is_leader, ep[pe].handle.qpn, laddr, lkey, raddr, rkey,
                                         val, val, bytes, amoType);
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    dbr_val = core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_msntbl_counter, my_psn_counter,
                                         is_leader, ep[pe].handle.qpn, laddr, lkey, raddr, rkey,
                                         val, val, bytes, amoType);
  }

  // __threadfence_system();
  if (is_leader) {
    uint64_t db_touched = 0;
    do {
      db_touched = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != warp_sq_counter);

    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, warp_sq_counter + num_wqes);
    __threadfence_system();
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbr_val);

    __hip_atomic_fetch_add(&cq->needConsIdx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&wq->dbTouchIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }

  // __threadfence_system();
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
  // ShmemQuietThreadKernelImpl<PrvdType>(pe);
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
  core::WorkQueueHandle* wq = &ep[pe].wqHandle;
  core::CompletionQueueHandle* cq = &ep[pe].cqHandle;

  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader = (my_logical_lane_id == num_active_lanes - 1);
  uint64_t leader_phys_lane_id = core::GetLastActiveLaneID(activemask);

  uint32_t warp_sq_counter = 0;
  uint32_t warp_msntbl_counter = 0, warp_psn_counter = 0;
  uint32_t my_sq_counter = 0, my_msntbl_counter = 0, my_psn_counter = 0;
  uint8_t num_wqes;

  if constexpr (PrvdType == core::ProviderType::MLX5) {
    uint32_t numWqesPerCmd = core::get_num_wqes_in_atomic(amoType, bytes);
    num_wqes = num_active_lanes * numWqesPerCmd;
    if (is_leader) {
      warp_sq_counter = __hip_atomic_fetch_add(&wq->postIdx, num_wqes, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id * numWqesPerCmd;
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    num_wqes = num_active_lanes;
    if (is_leader) {
      core::atomic_add_packed_msn_and_psn(&wq->msnPack, num_wqes, num_wqes, &warp_msntbl_counter,
                                          &warp_psn_counter);
      warp_sq_counter = warp_msntbl_counter;
      __hip_atomic_fetch_max(&wq->postIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
    }
    warp_sq_counter = __shfl(warp_sq_counter, leader_phys_lane_id);
    warp_msntbl_counter = __shfl(warp_msntbl_counter, leader_phys_lane_id);
    warp_psn_counter = __shfl(warp_psn_counter, leader_phys_lane_id);
    my_sq_counter = warp_sq_counter + my_logical_lane_id;
    my_msntbl_counter = warp_msntbl_counter + my_logical_lane_id;
    my_psn_counter = warp_psn_counter + my_logical_lane_id;
  } else {
    assert(false);
  }

  while (true) {
    uint64_t db_touched =
        __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t db_done = __hip_atomic_load(&wq->doneIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    uint64_t num_active_sq_entries = db_touched - db_done;
    uint64_t num_free_entries = wq->sqWqeNum - num_active_sq_entries;
    uint64_t num_entries_until_warp_last_entry = warp_sq_counter + num_wqes - db_touched;
    if (num_free_entries > num_entries_until_warp_last_entry) break;
    ShmemQuietThreadKernelImpl<PrvdType>(pe);
  }

  if constexpr (PrvdType == core::ProviderType::MLX5) {
    wq->outstandingWqe[my_sq_counter % OUTSTANDING_TABLE_SIZE] =
        my_sq_counter + core::get_num_wqes_in_atomic(amoType, bytes) - 1;
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    wq->outstandingWqe[my_sq_counter % wq->sqWqeNum] = my_sq_counter;
  }

  uint64_t dbr_val;
  if constexpr (PrvdType == core::ProviderType::MLX5) {
    dbr_val = core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_sq_counter, my_sq_counter,
                                         is_leader, ep[pe].handle.qpn, laddr, lkey, raddr, rkey,
                                         val, compare, bytes, amoType);
  } else if constexpr (PrvdType == core::ProviderType::BNXT) {
    dbr_val = core::PostAtomic<PrvdType>(*wq, my_sq_counter, my_msntbl_counter, my_psn_counter,
                                         is_leader, ep[pe].handle.qpn, laddr, lkey, raddr, rkey,
                                         val, compare, bytes, amoType);
  }

  // __threadfence_system();
  if (is_leader) {
    uint64_t db_touched = 0;
    do {
      db_touched = __hip_atomic_load(&wq->dbTouchIdx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } while (db_touched != warp_sq_counter);

    core::UpdateSendDbrRecord<PrvdType>(wq->dbrRecAddr, warp_sq_counter + num_wqes);
    __threadfence_system();
    core::RingDoorbell<PrvdType>(wq->dbrAddr, dbr_val);

    __hip_atomic_fetch_add(&cq->needConsIdx, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&wq->dbTouchIdx, warp_sq_counter + num_wqes, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }

  // __threadfence_system();
  // ShmemQuietThreadKernelImpl<PrvdType>(pe);
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
