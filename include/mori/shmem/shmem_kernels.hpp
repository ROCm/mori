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
__device__ GpuStates* GetGlobalGpuStatesPtr() { return &globalGpuStates; }

template <core::ProviderType PrvdType>
__device__ void ShmemPutMemNbiThread(const application::SymmMemObj* dest, size_t destOffset,
                                     const application::MemoryRegion& source, size_t sourceOffset,
                                     size_t nelems, int pe) {
  uintptr_t laddr = source.addr + sourceOffset;
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uintptr_t rkey = dest->peerRkeys[pe];

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->epsStartAddr;
  application::WorkQueueHandle& wq = ep[pe].wqHandle;

  int rank = globalGpuStates->rank;
  uint64_t dbrVal =
      core::PostWrite<PrvdType>(wq.sqAddr, wq.sqWqeNum, &wq.postIdx, ep[pe].handle.qpn, laddr,
                                source.lkey, raddr, rkey, nelems);
  core::UpdateSendDbrRecord<PrvdType>(wq.dbrRecAddr, wq.postIdx);
  __threadfence_system();
  core::RingDoorbell<PrvdType>(wq.dbrAddr, dbrVal);
  __threadfence_system();
}

// TODO: deal with bytes count limit
template <core::ProviderType PrvdType, uint32_t SIZE>
__device__ void ShmemPutSizeNbiThread(const application::SymmMemObj* dest, size_t destOffset,
                                      void* val, int pe) {
  uintptr_t raddr = dest->peerPtrs[pe] + destOffset;
  uintptr_t rkey = dest->peerRkeys[pe];

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->epsStartAddr;
  application::WorkQueueHandle& wq = ep[pe].wqHandle;

  int rank = globalGpuStates->rank;
  uint64_t dbrVal = core::PostWriteInline<PrvdType>(wq.sqAddr, wq.sqWqeNum, &wq.postIdx,
                                                    ep[pe].handle.qpn, val, raddr, rkey, SIZE);
  core::UpdateSendDbrRecord<PrvdType>(wq.dbrRecAddr, wq.postIdx);
  __threadfence_system();
  core::RingDoorbell<PrvdType>(wq.dbrAddr, dbrVal);
  __threadfence_system();
}

template <core::ProviderType PrvdType, typename T>
__device__ void ShmemPutTypeNbiThread(const application::SymmMemObj* dest, size_t destOffset, T val,
                                      int pe) {
  static_assert(sizeof(T) <= core::MaxInlineDataSizePerWqe);
  ShmemPutSizeNbiThread<PrvdType, sizeof(T)>(dest, destOffset, &val, pe);
}

#define DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(TypeName, T)                                  \
  template <core::ProviderType PrvdType>                                                   \
  __device__ void ShmemPut##TypeName##NbiThread(const application::SymmMemObj* dest,       \
                                                size_t destOffset, uint32_t val, int pe) { \
    ShmemPutTypeNbiThread<PrvdType, uint32_t>(dest, destOffset, val, pe);                  \
  }
DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Uint8, uint8_t)
DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Uint16, uint16_t)
DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Uint32, uint32_t)
DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Uint64, uint64_t)

__device__ void ShmemPutMemNbiWarp(const application::SymmMemObj* dest,
                                   const application::MemoryRegion& source, size_t nelems, int pe);

/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */
template <core::ProviderType PrvdType>
__device__ void ShmemQuietThread() {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::RdmaEndpoint* ep = globalGpuStates->epsStartAddr;

  int rank = globalGpuStates->rank;
  int worldSize = globalGpuStates->worldSize;

  for (int i = 0; i < worldSize; i++) {
    // This assume we do not have endpoint to self
    if (i == rank) continue;

    application::CompletionQueueHandle& cq = ep[i].cqHandle;
    application::WorkQueueHandle& wq = ep[i].wqHandle;

    // Assume every wqe generates a cqe, so we can use work queue postIdx
    // TODO: 1 Should not use postIdx since 1 wqe can inc postIdx by > 1
    // TODO: 2 How to prevent cqe overflow?
    int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    while ((core::atomicLoadSeqCst(&cq.consIdx) + 1) < core::atomicLoadSeqCst(&wq.postIdx)) {
      int opcode = core::PollCq<PrvdType>(cq.cqAddr, cq.cqeSize, cq.cqeNum, &cq.consIdx);
      if (opcode == MLX5_CQE_RESP_ERR || opcode == MLX5_CQE_REQ_ERR) {
        printf("rank %d opcode %d\n", rank, opcode);
        assert(false);
      }
      core::UpdateCqDbrRecord<PrvdType>(cq.dbrRecAddr, core::atomicLoadSeqCst(&cq.consIdx));
    }
  }
}

template <core::ProviderType PrvdType, typename T>
__device__ void ShmemTypeWaitUntilGreaterThan(T* addr, T val) {
  while (core::atomicLoadRelaxed(addr) == val) {
  }
}

#define DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(TypeName, T)            \
  template <core::ProviderType PrvdType>                                  \
  __device__ void Shmem##TypeName##WaitUntilGreaterThan(T* addr, T val) { \
    ShmemTypeWaitUntilGreaterThan<PrvdType, T>(addr, val);                \
  }

DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint8, uint8_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint16, uint16_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint32, uint32_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint64, uint64_t)

}  // namespace shmem
}  // namespace mori
