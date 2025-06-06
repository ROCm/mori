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
template <>
inline __device__ void ShmemPutMemNbiThreadKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::MemoryRegion& source, size_t sourceOffset, size_t bytes, int pe) {
  uint8_t* srcPtr = reinterpret_cast<uint8_t*>(source.addr + sourceOffset);
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);
  core::ThreadCopy<uint8_t>(destPtr, srcPtr, bytes);
}

template <>
inline __device__ void ShmemPutMemNbiWarpKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::MemoryRegion& source, size_t sourceOffset, size_t bytes, int pe) {
  uint8_t* srcPtr = reinterpret_cast<uint8_t*>(source.addr + sourceOffset);
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);
  core::WarpCopy<uint8_t>(destPtr, srcPtr, bytes);
}

template <>
inline __device__ void ShmemPutSizeImmNbiThreadKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe) {
  uint8_t* srcPtr = reinterpret_cast<uint8_t*>(val);
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);
  core::ThreadCopyAtomic<uint8_t>(destPtr, srcPtr, bytes);
}

template <>
inline __device__ void ShmemPutSizeImmNbiWarpKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe) {
  uint8_t* srcPtr = reinterpret_cast<uint8_t*>(val);
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);
  core::WarpCopyAtomic<uint8_t>(destPtr, srcPtr, bytes);
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe,
    core::atomicType amoType){
      //TODO
      ;
    }

template <>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe,
    core::atomicType amoType){
      //TODO
      ;
    }

template <>
inline __device__ void ShmemAtomicSizeFetchThreadKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::MemoryRegion& source, size_t sourceOffset, void* val, void* compare,
    size_t bytes, int pe, core::atomicType amoType){
      //TODO
      ;
    }

template <>
inline __device__ void ShmemAtomicSizeFetchWarpKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::MemoryRegion& source, size_t sourceOffset, void* val, void* compare,
    size_t bytes, int pe, core::atomicType amoType){
      //TODO
      ;
    }

/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */
template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::P2P>() {}

}  // namespace shmem
}  // namespace mori
