#pragma once

#include <mpi.h>

#include "mori/application/application.hpp"
#include "mori/shmem/shmem_api.hpp"
#include "src/shmem/internal.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                         Point-to-Point                                         */
/* ---------------------------------------------------------------------------------------------- */
__device__ application::RdmaEndpoint* GetEpsStartAddr() { return epsStartAddr; }

__device__ void ShmemPutMemNbiThread(const application::SymmMemObj* dest,
                                     const application::MemoryRegion& source, size_t nelems,
                                     int pe);

__device__ void ShmemPutMemNbiWarp(const application::SymmMemObj* dest,
                                   const application::MemoryRegion& source, size_t nelems, int pe);

}  // namespace shmem
}  // namespace mori
