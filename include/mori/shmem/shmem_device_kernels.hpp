#pragma once

#include "mori/application/application.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                         Point-to-Point                                         */
/* ---------------------------------------------------------------------------------------------- */
template <application::TransportType TsptType>
__device__ void ShmemPutMemNbiThreadKernel(const application::SymmMemObjPtr dest, size_t destOffset,
                                           const application::MemoryRegion& source,
                                           size_t sourceOffset, size_t bytes, int pe);

template <application::TransportType TsptType>
__device__ void ShmemPutSizeImmNbiThreadKernel(const application::SymmMemObjPtr dest,
                                               size_t destOffset, void* val, size_t bytes, int pe);

/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */
template <application::TransportType>
__device__ void ShmemQuietThreadKernel();

}  // namespace shmem
}  // namespace mori