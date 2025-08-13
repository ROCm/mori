#pragma once

#include "mori/application/application.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                         Point-to-Point                                         */
/* ---------------------------------------------------------------------------------------------- */
template <application::TransportType TsptType>
inline __device__ void ShmemPutMemNbiThreadKernel(const application::SymmMemObjPtr dest,
                                                  size_t destOffset,
                                                  const application::RdmaMemoryRegion& source,
                                                  size_t sourceOffset, size_t bytes, int pe);

template <application::TransportType TsptType>
inline __device__ void ShmemPutMemNbiWarpKernel(const application::SymmMemObjPtr dest,
                                                size_t destOffset,
                                                const application::RdmaMemoryRegion& source,
                                                size_t sourceOffset, size_t bytes, int pe);

template <application::TransportType TsptType>
inline __device__ void ShmemPutSizeImmNbiThreadKernel(const application::SymmMemObjPtr dest,
                                                      size_t destOffset, void* val, size_t bytes,
                                                      int pe);

template <application::TransportType TsptType>
inline __device__ void ShmemPutSizeImmNbiWarpKernel(const application::SymmMemObjPtr dest,
                                                    size_t destOffset, void* val, size_t bytes,
                                                    int pe);

template <application::TransportType TsptType>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernel(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, void* val, size_t bytes,
    int pe, core::atomicType amoType);

template <application::TransportType TsptType>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernel(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, void* val, size_t bytes,
    int pe, core::atomicType amoType);

template <application::TransportType TsptType>
inline __device__ void ShmemAtomicSizeFetchThreadKernel(const application::SymmMemObjPtr dest,
                                                        size_t destOffset,
                                                        const application::RdmaMemoryRegion& source,
                                                        size_t sourceOffset, void* val,
                                                        void* compare, size_t bytes, int pe,
                                                        core::atomicType amoType);

template <application::TransportType TsptType>
inline __device__ void ShmemAtomicSizeFetchWarpKernel(const application::SymmMemObjPtr dest,
                                                      size_t destOffset,
                                                      const application::RdmaMemoryRegion& source,
                                                      size_t sourceOffset, void* val, void* compare,
                                                      size_t bytes, int pe,
                                                      core::atomicType amoType);

/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */
template <application::TransportType>
inline __device__ void ShmemQuietThreadKernel();

template <application::TransportType>
inline __device__ void ShmemQuietThreadKernel(int pe);

}  // namespace shmem
}  // namespace mori