#pragma once

#include "mori/core/core.hpp"
#include "mori/shmem/shmem_api.hpp"

namespace mori {
namespace shmem {

__device__ void ShmemPutMemNbiThread(const application::SymmMemObj* dest,
                                     const application::MemoryRegion& source, size_t nelems,
                                     int pe) {
  // TODO queue state

  uintptr_t raddr = dest.peerPtrs[pe];
  uint64_t rkey = dest.peerRkeys[pe];
  uint64_t dbrVal = core::PostWrite(queue_buff_addr, wqe_num, post_idx, qpn, source.addr,
                                    source.lkey, raddr, rkey, nelems);
  core::UpdateSendDbrRecord(dbr_rec_addr, wqe_idx);
  core::RingDoorbell(dbr_addr, dbrVal)
}

__device__ void ShmemPutMemNbiWarp(SymmMemObj* dest, const application::MemoryRegion& source,
                                   size_t nelems, int pe) {
  int laneId = threadIdx.x % warpSize;
  if (laneId == 0) {
    ShmemPutMemNbiThread(dest, source, elems, pe);
  }
  __syncwarp();
}

}  // namespace shmem
}  // namespace mori
