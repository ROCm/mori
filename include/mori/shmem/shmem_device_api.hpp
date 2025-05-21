#pragma once

#include <assert.h>
#include <mpi.h>

#include "mori/application/application.hpp"
#include "mori/core/core.hpp"
#include "mori/shmem/shmem_device_kernels.hpp"
#include "mori/shmem/shmem_ibgda_kernels.hpp"
#include "mori/shmem/shmem_p2p_kernels.hpp"
#include "src/shmem/internal.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                         Point-to-Point                                         */
/* ---------------------------------------------------------------------------------------------- */
__device__ void ShmemPutMemNbiThread(const application::SymmMemObjPtr dest, size_t destOffset,
                                     const application::MemoryRegion& source, size_t sourceOffset,
                                     size_t bytes, int pe) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::TransportType transportType = globalGpuStates->transportTypes[pe];
  if (transportType == application::TransportType::RDMA) {
    ShmemPutMemNbiThreadKernel<application::TransportType::RDMA>(dest, destOffset, source,
                                                                 sourceOffset, bytes, pe);
  } else {
    ShmemPutMemNbiThreadKernel<application::TransportType::P2P>(dest, destOffset, source,
                                                                sourceOffset, bytes, pe);
  }
}

__device__ void ShmemPutMemNbiThread(const application::SymmMemObjPtr dest, size_t destOffset,
                                     const application::SymmMemObjPtr source, size_t sourceOffset,
                                     size_t bytes, int pe) {
  int rank = GetGlobalGpuStatesPtr()->rank;
  ShmemPutMemNbiThread(dest, destOffset, source->GetMemoryRegion(rank), sourceOffset, bytes, pe);
}

template <typename T>
__device__ void ShmemPutTypeNbiThread(const application::SymmMemObjPtr dest, size_t destElmOffset,
                                      const application::MemoryRegion& source, size_t srcElmOffset,
                                      size_t nelems, int pe) {
  constexpr size_t typeSize = sizeof(T);
  ShmemPutMemNbiThread(dest, destElmOffset * typeSize, source, srcElmOffset * typeSize,
                       nelems * typeSize, pe);
}

template <typename T>
__device__ void ShmemPutTypeNbiThread(const application::SymmMemObjPtr dest, size_t destElmOffset,
                                      const application::SymmMemObjPtr source, size_t srcElmOffset,
                                      size_t nelems, int pe) {
  int rank = GetGlobalGpuStatesPtr()->rank;
  ShmemPutTypeNbiThread<T>(dest, destElmOffset, source->GetMemoryRegion(rank), srcElmOffset, nelems,
                           pe);
}

#define DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(TypeName, T)                                    \
  __device__ void ShmemPut##TypeName##NbiThread(                                             \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                           \
      const application::MemoryRegion& source, size_t srcElmOffset, size_t nelems, int pe) { \
    ShmemPutTypeNbiThread<T>(dest, destElmOffset, source, srcElmOffset, nelems, pe);         \
  }                                                                                          \
  __device__ void ShmemPut##TypeName##NbiThread(                                             \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                           \
      const application::SymmMemObjPtr source, size_t srcElmOffset, size_t nelems, int pe) { \
    ShmemPutTypeNbiThread<T>(dest, destElmOffset, source, srcElmOffset, nelems, pe);         \
  }

DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Uint8, uint8_t)
DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Uint16, uint16_t)
DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Uint32, uint32_t)
DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Uint64, uint64_t)
DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Float, float)
DEFINE_SHMEM_PUT_TYPE_NBI_THREAD_API(Double, double)

// TODO: deal with bytes count limit
__device__ void ShmemPutSizeImmNbiThread(const application::SymmMemObjPtr dest, size_t destOffset,
                                         void* val, size_t bytes, int pe) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  application::TransportType transportType = globalGpuStates->transportTypes[pe];
  if (transportType == application::TransportType::RDMA) {
    ShmemPutSizeImmNbiThreadKernel<application::TransportType::RDMA>(dest, destOffset, val, bytes,
                                                                     pe);
  } else {
    ShmemPutSizeImmNbiThreadKernel<application::TransportType::P2P>(dest, destOffset, val, bytes,
                                                                    pe);
  }
}

template <typename T>
__device__ void ShmemPutTypeImmNbiThread(const application::SymmMemObjPtr dest, size_t destOffset,
                                         T val, int pe) {
  static_assert(sizeof(T) <= core::MaxInlineDataSizePerWqe);
  ShmemPutSizeImmNbiThread(dest, destOffset, &val, sizeof(T), pe);
}

#define DEFINE_SHMEM_PUT_TYPE_IMM_NBI_THREAD_API(TypeName, T)                                 \
  __device__ void ShmemPut##TypeName##ImmNbiThread(const application::SymmMemObjPtr dest,     \
                                                   size_t destOffset, uint32_t val, int pe) { \
    ShmemPutTypeImmNbiThread<uint32_t>(dest, destOffset, val, pe);                            \
  }

DEFINE_SHMEM_PUT_TYPE_IMM_NBI_THREAD_API(Uint8, uint8_t)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_THREAD_API(Uint16, uint16_t)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_THREAD_API(Uint32, uint32_t)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_THREAD_API(Uint64, uint64_t)

__device__ void ShmemPutMemNbiWarp(const application::SymmMemObj* dest,
                                   const application::MemoryRegion& source, size_t nelems, int pe);

/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */
__device__ void ShmemQuietThread() { ShmemQuietThreadKernel<application::TransportType::RDMA>(); }

template <typename T>
__device__ void ShmemTypeWaitUntilGreaterThan(T* addr, T val) {
  while (core::AtomicLoadRelaxed(addr) <= val) {
  }
}

#define DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(TypeName, T)            \
  __device__ void Shmem##TypeName##WaitUntilGreaterThan(T* addr, T val) { \
    ShmemTypeWaitUntilGreaterThan<T>(addr, val);                          \
  }

DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint8, uint8_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint16, uint16_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint32, uint32_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint64, uint64_t)

template <typename T>
__device__ void ShmemTypeWaitUntilEquals(T* addr, T val) {
  while (core::AtomicLoadRelaxed(addr) != val) {
  }
}

#define DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(TypeName, T)              \
  __device__ void Shmem##TypeName##WaitUntilEquals(T* addr, T val) { \
    ShmemTypeWaitUntilEquals<T>(addr, val);                          \
  }

DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Uint8, uint8_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Uint16, uint16_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Uint32, uint32_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Uint64, uint64_t)

}  // namespace shmem
}  // namespace mori
