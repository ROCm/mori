#pragma once

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
  __threadfence_system();
  core::UpdateSendDbrRecord<PrvdType>(wq.dbrRecAddr, wq.postIdx);
  __threadfence_system();
  core::RingDoorbell<PrvdType>(wq.dbrAddr, dbrVal);
  __threadfence_system();
}

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
    if (i == rank) continue;
    application::CompletionQueueHandle& cq = ep[i].cqHandle;
    application::WorkQueueHandle& wq = ep[i].wqHandle;
    if (cq.consIdx < wq.postIdx) {
      int opcode = core::PollCq<PrvdType>(cq.cqAddr, cq.cqeSize, cq.cqeNum, &cq.consIdx);
      if (opcode == MLX5_CQE_RESP_ERR || opcode == MLX5_CQE_REQ_ERR) {
        printf("rank %d opcode %d\n", rank, opcode);
      }
    }
  }
}

}  // namespace shmem
}  // namespace mori
