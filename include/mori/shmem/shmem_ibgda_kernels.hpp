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
/*                                         Point-to-Point                                         */
/* ---------------------------------------------------------------------------------------------- */
template <core::ProviderType PrvdType>
inline __device__ void ShmemPutMemNbiThreadKernelImpl(const application::SymmMemObjPtr dest,
                                                      size_t destOffset,
                                                      const application::MemoryRegion& source,
                                                      size_t sourceOffset, size_t bytes, int pe) {
  if (bytes == 0) return;
  uintptr_t laddr = source.addr + sourceOffset;
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uintptr_t rkey = dest->peerRkeys[pe];

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  application::WorkQueueHandle& wq = ep[pe].wqHandle;
  uint32_t* lock = globalGpuStates->endpointLock;

  int rank = globalGpuStates->rank;
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  uint64_t dbrVal = core::PostWrite<PrvdType>(wq.sqAddr, wq.sqWqeNum, curPostIdx, ep[pe].handle.qpn,
                                              laddr, source.lkey, raddr, rkey, bytes);
  while (core::AtomicLoadRelaxed(&wq.readyIdx) != curPostIdx) {
  }
  core::UpdateSendDbrRecord<PrvdType>(wq.dbrRecAddr, curPostIdx);
  // __threadfence_system();
  core::RingDoorbell<PrvdType>(wq.dbrAddr, dbrVal);
  atomicAdd(&wq.readyIdx, 1);
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
    const application::MemoryRegion& source, size_t sourceOffset, size_t bytes, int pe) {
  DISPATCH_PROVIDER_TYPE(ShmemPutMemNbiThreadKernelImpl, dest, destOffset, source, sourceOffset,
                         bytes, pe);
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemPutMemNbiWarpKernelImpl(const application::SymmMemObjPtr dest,
                                                    size_t destOffset,
                                                    const application::MemoryRegion& source,
                                                    size_t sourceOffset, size_t bytes, int pe) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemPutMemNbiThreadKernelImpl<PrvdType>(dest, destOffset, source, sourceOffset, bytes, pe);
  }
}

template <>
inline __device__ void ShmemPutMemNbiWarpKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::MemoryRegion& source, size_t sourceOffset, size_t bytes, int pe) {
  DISPATCH_PROVIDER_TYPE(ShmemPutMemNbiWarpKernelImpl, dest, destOffset, source, sourceOffset,
                         bytes, pe);
}

// TODO: deal with bytes count limit
// TODO: put size api only support 1,2,4,8,16 in nvshmem, should we do that?
template <core::ProviderType PrvdType>
inline __device__ void ShmemPutSizeImmNbiThreadKernelImpl(const application::SymmMemObjPtr dest,
                                                          size_t destOffset, void* val,
                                                          size_t bytes, int pe) {
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uintptr_t rkey = dest->peerRkeys[pe];

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  application::WorkQueueHandle& wq = ep[pe].wqHandle;
  uint32_t* lock = globalGpuStates->endpointLock;

  int rank = globalGpuStates->rank;
  uint32_t curPostIdx = atomicAdd(&wq.postIdx, 1);
  uint64_t dbrVal = core::PostWriteInline<PrvdType>(wq.sqAddr, wq.sqWqeNum, curPostIdx,
                                                    ep[pe].handle.qpn, val, raddr, rkey, bytes);
  while (core::AtomicLoadRelaxed(&wq.readyIdx) != curPostIdx) {
  }
  core::UpdateSendDbrRecord<PrvdType>(wq.dbrRecAddr, curPostIdx);
  // __threadfence_system();
  core::RingDoorbell<PrvdType>(wq.dbrAddr, dbrVal);
  atomicAdd(&wq.readyIdx, 1);
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


inline __device__ void ShmemQuietThreadKernelImpl();

template <core::ProviderType PrvdType>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernelImpl(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe,
    core::atomicType amoType) {
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uintptr_t rkey = dest->peerRkeys[pe];

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  application::WorkQueueHandle& wq = ep[pe].wqHandle;
  uint32_t* lock = globalGpuStates->endpointLock;

  int rank = globalGpuStates->rank;
  uint64_t dbrVal = core::PostAtomic<PrvdType>(wq.sqAddr, wq.sqWqeNum, &wq.postIdx, ep[pe].handle.qpn,
                                               0, 0, raddr, rkey, val, 0, bytes, amoType);
  core::UpdateDbrAndRingDbSend<PrvdType>(wq.dbrRecAddr, wq.postIdx, wq.dbrAddr, dbrVal, lock + pe);
  // core::UpdateSendDbrRecord<PrvdType>(wq.dbrRecAddr, wq.postIdx);
  // __threadfence_system();
  // core::RingDoorbell<PrvdType>(wq.dbrAddr, dbrVal);
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe,
    core::atomicType amoType) {
  DISPATCH_PROVIDER_TYPE(ShmemAtomicSizeNonFetchThreadKernelImpl, dest, destOffset, val, bytes, pe,
                         amoType);
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernelImpl(const application::SymmMemObjPtr dest,
                                                             size_t destOffset, void* val,
                                                             size_t bytes, int pe,
                                                             core::atomicType amoType) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemAtomicSizeNonFetchThreadKernelImpl<PrvdType>(dest, destOffset, val, bytes, pe, amoType);
  }
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe,
    core::atomicType amoType) {
  DISPATCH_PROVIDER_TYPE(ShmemAtomicSizeNonFetchWarpKernelImpl, dest, destOffset, val, bytes, pe,
                         amoType);
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemAtomicSizeFetchThreadKernelImpl(const application::SymmMemObjPtr dest,
                                                            size_t destOffset,
                                                            const application::MemoryRegion& source,
                                                            size_t sourceOffset, void* val,
                                                            void* compare, size_t bytes, int pe,
                                                            core::atomicType amoType) {
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uintptr_t rkey = dest->peerRkeys[pe];
  uintptr_t laddr = source.addr + sourceOffset;
  uintptr_t lkey = source.lkey;

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  application::WorkQueueHandle& wq = ep[pe].wqHandle;
  uint32_t* lock = globalGpuStates->endpointLock;

  int rank = globalGpuStates->rank;
  uint64_t dbrVal =
      core::PostAtomic<PrvdType>(wq.sqAddr, wq.sqWqeNum, &wq.postIdx, ep[pe].handle.qpn, laddr,
                                 lkey, raddr, rkey, val, compare, bytes, amoType);
  core::UpdateDbrAndRingDbSend<PrvdType>(wq.dbrRecAddr, wq.postIdx, wq.dbrAddr, dbrVal, lock + pe);

  // core::UpdateSendDbrRecord<PrvdType>(wq.dbrRecAddr, wq.postIdx);
  // __threadfence_system();
  // core::RingDoorbell<PrvdType>(wq.dbrAddr, dbrVal);

  // queit to poll CQ
  ShmemQuietThreadKernelImpl();
}

template <>
inline __device__ void ShmemAtomicSizeFetchThreadKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::MemoryRegion& source, size_t sourceOffset, void* val, void* compare,
    size_t bytes, int pe, core::atomicType amoType) {
  DISPATCH_PROVIDER_TYPE(ShmemAtomicSizeFetchThreadKernelImpl, dest, destOffset, source,
                         sourceOffset, val, compare, bytes, pe, amoType);
}

template <core::ProviderType PrvdType>
inline __device__ void ShmemAtomicSizeFetchWarpKernelImpl(const application::SymmMemObjPtr dest,
                                                          size_t destOffset,
                                                          const application::MemoryRegion& source,
                                                          size_t sourceOffset, void* val,
                                                          void* compare, size_t bytes, int pe,
                                                          core::atomicType amoType) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemAtomicSizeFetchThreadKernelImpl<PrvdType>(dest, destOffset, source, sourceOffset, val,
                                                   compare, bytes, pe, amoType);
  }
}

template <>
inline __device__ void ShmemAtomicSizeFetchWarpKernel<application::TransportType::RDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::MemoryRegion& source, size_t sourceOffset, void* val, void* compare,
    size_t bytes, int pe, core::atomicType amoType) {
  DISPATCH_PROVIDER_TYPE(ShmemAtomicSizeFetchWarpKernelImpl, dest, destOffset, source, sourceOffset,
                         val, compare, bytes, pe, amoType);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ void ShmemQuietThreadKernelImpl() {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;

  int rank = globalGpuStates->rank;
  int worldSize = globalGpuStates->worldSize;

  for (int i = 0; i < worldSize; i++) {
    if (i == rank) continue;
    if (globalGpuStates->transportTypes[i] != application::TransportType::RDMA) continue;

    application::CompletionQueueHandle& cq = ep[i].cqHandle;
    application::WorkQueueHandle& wq = ep[i].wqHandle;
    core::ProviderType prvdType = ep[i].GetProviderType();

    // Assume every wqe generates a cqe, so we can use work queue postIdx
    // TODO: 1 Should not use postIdx since 1 wqe can inc postIdx by > 1
    // TODO: 2 How to prevent cqe overflow?
    while (true) {
      uint32_t consIdx = core::AtomicLoadSeqCst(&cq.consIdx);
      uint32_t postIdx = core::AtomicLoadSeqCst(&wq.postIdx);
      if ((consIdx + 1) >= postIdx) break;
      if (prvdType == core::ProviderType::MLX5) {
        int opcode =
            core::PollCq<core::ProviderType::MLX5>(cq.cqAddr, cq.cqeSize, cq.cqeNum, &cq.consIdx);
        if (opcode == MLX5_CQE_RESP_ERR || opcode == MLX5_CQE_REQ_ERR) {
          printf("rank %d dest pe %d consIdx %d opcode %d\n", rank, i, consIdx, opcode);
          core::DumpWqe(wq.sqAddr, consIdx);
          assert(false);
        }
        core::UpdateCqDbrRecord<core::ProviderType::MLX5>(cq.dbrRecAddr, consIdx);
      } else {
        assert(false);
      }
    }
  }
}

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::RDMA>() {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->rdmaEndpoints;
  ShmemQuietThreadKernelImpl();
}

}  // namespace shmem
}  // namespace mori
