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

#define DISPATCH_TRANSPORT_TYPE(func, pe, ...)                                    \
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();                           \
  application::TransportType transportType = globalGpuStates->transportTypes[pe]; \
  if (transportType == application::TransportType::RDMA) {                        \
    func<application::TransportType::RDMA>(__VA_ARGS__);                          \
  } else if (transportType == application::TransportType::P2P) {                  \
    func<application::TransportType::P2P>(__VA_ARGS__);                           \
  } else {                                                                        \
    assert(false);                                                                \
  }

/* ---------------------------------------------------------------------------------------------- */
/*                                         Point-to-Point                                         */
/* ---------------------------------------------------------------------------------------------- */
#define DEFINE_SHMEM_PUT_MEM_NBI_API_TEMPLATE(Scope)                                            \
  inline __device__ void ShmemPutMemNbi##Scope(                                                 \
      const application::SymmMemObjPtr dest, size_t destOffset,                                 \
      const application::MemoryRegion& source, size_t sourceOffset, size_t bytes, int pe) {     \
    DISPATCH_TRANSPORT_TYPE(ShmemPutMemNbi##Scope##Kernel, pe, dest, destOffset, source,        \
                            sourceOffset, bytes, pe);                                           \
  }                                                                                             \
  inline __device__ void ShmemPutMemNbi##Scope(                                                 \
      const application::SymmMemObjPtr dest, size_t destOffset,                                 \
      const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe) {     \
    int rank = GetGlobalGpuStatesPtr()->rank;                                                   \
    ShmemPutMemNbi##Scope(dest, destOffset, source->GetMemoryRegion(rank), sourceOffset, bytes, \
                          pe);                                                                  \
  }

DEFINE_SHMEM_PUT_MEM_NBI_API_TEMPLATE(Thread)
DEFINE_SHMEM_PUT_MEM_NBI_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_PUT_TYPE_NBI_API_TEMPLATE(Scope)                                           \
  template <typename T>                                                                         \
  inline __device__ void ShmemPutTypeNbi##Scope(                                                \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                              \
      const application::MemoryRegion& source, size_t srcElmOffset, size_t nelems, int pe) {    \
    constexpr size_t typeSize = sizeof(T);                                                      \
    ShmemPutMemNbi##Scope(dest, destElmOffset* typeSize, source, srcElmOffset* typeSize,        \
                          nelems* typeSize, pe);                                                \
  }                                                                                             \
  template <typename T>                                                                         \
  inline __device__ void ShmemPutTypeNbi##Scope(                                                \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                              \
      const application::SymmMemObjPtr source, size_t srcElmOffset, size_t nelems, int pe) {    \
    int rank = GetGlobalGpuStatesPtr()->rank;                                                   \
    ShmemPutTypeNbi##Scope<T>(dest, destElmOffset, source->GetMemoryRegion(rank), srcElmOffset, \
                              nelems, pe);                                                      \
  }

DEFINE_SHMEM_PUT_TYPE_NBI_API_TEMPLATE(Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_PUT_TYPE_NBI_API(TypeName, T, Scope)                                    \
  inline __device__ void ShmemPut##TypeName##Nbi##Scope(                                     \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                           \
      const application::MemoryRegion& source, size_t srcElmOffset, size_t nelems, int pe) { \
    ShmemPutTypeNbi##Scope<T>(dest, destElmOffset, source, srcElmOffset, nelems, pe);        \
  }                                                                                          \
  inline __device__ void ShmemPut##TypeName##Nbi##Scope(                                     \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                           \
      const application::SymmMemObjPtr source, size_t srcElmOffset, size_t nelems, int pe) { \
    ShmemPutTypeNbi##Scope<T>(dest, destElmOffset, source, srcElmOffset, nelems, pe);        \
  }

DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint8, uint8_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint16, uint16_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Float, float, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Double, double, Thread)

DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint8, uint8_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint16, uint16_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Float, float, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Double, double, Warp)

// TODO: deal with bytes count limit
#define SHMEM_PUT_SIZE_IMM_NBI_API(Scope)                                                          \
  inline __device__ void ShmemPutSizeImmNbi##Scope(                                                \
      const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe) { \
    DISPATCH_TRANSPORT_TYPE(ShmemPutSizeImmNbi##Scope##Kernel, pe, dest, destOffset, val, bytes,   \
                            pe);                                                                   \
  }

SHMEM_PUT_SIZE_IMM_NBI_API(Thread)
SHMEM_PUT_SIZE_IMM_NBI_API(Warp)

#define SHMEM_PUT_TYPE_IMM_NBI_API_TEMPLATE(Scope)                                        \
  template <typename T>                                                                   \
  inline __device__ void ShmemPutTypeImmNbi##Scope(const application::SymmMemObjPtr dest, \
                                                   size_t destOffset, T val, int pe) {    \
    static_assert(sizeof(T) <= core::MaxInlineDataSizePerWqe);                            \
    ShmemPutSizeImmNbi##Scope(dest, destOffset, &val, sizeof(T), pe);                     \
  }

SHMEM_PUT_TYPE_IMM_NBI_API_TEMPLATE(Thread)
SHMEM_PUT_TYPE_IMM_NBI_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(TypeName, T, Scope)                           \
  inline __device__ void ShmemPut##TypeName##ImmNbi##Scope(                             \
      const application::SymmMemObjPtr dest, size_t destOffset, uint32_t val, int pe) { \
    ShmemPutTypeImmNbi##Scope<T>(dest, destOffset, val, pe);                     \
  }

DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint8, uint8_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint16, uint16_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint64, uint64_t, Thread)

DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint8, uint8_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint16, uint16_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint64, uint64_t, Warp)

#define SHMEM_ATOMIC_SIZE_NONFETCH_API_TEMPLATE(Scope)                                           \
  inline __device__ void ShmemAtomicSizeNonFetch##Scope(                                         \
      const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe, \
      core::atomicType amoType) {                                                                \
    DISPATCH_TRANSPORT_TYPE(ShmemAtomicSizeNonFetch##Scope##Kernel, pe, dest, destOffset, val,   \
                            bytes, pe, amoType);                                                 \
  }

SHMEM_ATOMIC_SIZE_NONFETCH_API_TEMPLATE(Thread)
SHMEM_ATOMIC_SIZE_NONFETCH_API_TEMPLATE(Warp)

#define SHMEM_ATOMIC_TYPE_NONFETCH_API_TEMPLATE(Scope)                                        \
  template <typename T>                                                                        \
  inline __device__ void ShmemAtomicTypeNonFetch##Scope(const application::SymmMemObjPtr dest, \
                                                        size_t destOffset, T val, int pe,      \
                                                        core::atomicType amoType) {            \
    ShmemAtomicSizeNonFetch##Scope(dest, destOffset, &val, sizeof(T), pe, amoType);            \
  }

SHMEM_ATOMIC_TYPE_NONFETCH_API_TEMPLATE(Thread)
SHMEM_ATOMIC_TYPE_NONFETCH_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(TypeName, T, Scope)              \
  inline __device__ void ShmemAtomic##TypeName##NonFetch##Scope(               \
      const application::SymmMemObjPtr dest, size_t destOffset, T val, int pe, \
      core::atomicType amoType) {                                              \
    ShmemAtomicTypeNonFetch##Scope<T>(dest, destOffset, val, pe, amoType);     \
  }

DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Int32, int32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Int64, int64_t, Thread)

DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Int32, int32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Int64, int64_t, Warp)

#define SHMEM_ATOMIC_SIZE_FETCH_API_TEMPLATE(Scope)                                            \
  inline __device__ void ShmemAtomicSizeFetch##Scope(                                          \
      const application::SymmMemObjPtr dest, size_t destOffset,                                \
      const application::MemoryRegion& source, size_t sourceOffset, void* val, void* compare,  \
      size_t bytes, int pe, core::atomicType amoType) {                                        \
    DISPATCH_TRANSPORT_TYPE(ShmemAtomicSizeFetch##Scope##Kernel, pe, dest, destOffset, source, \
                            sourceOffset, val, compare, bytes, pe, amoType);                   \
  }

SHMEM_ATOMIC_SIZE_FETCH_API_TEMPLATE(Thread)
SHMEM_ATOMIC_SIZE_FETCH_API_TEMPLATE(Warp)

#define SHMEM_ATOMIC_TYPE_FETCH_API_TEMPLATE(Scope)                                                \
  template <typename T>                                                                            \
  inline __device__ T ShmemAtomicTypeFetch##Scope(                                                 \
      const application::SymmMemObjPtr dest, size_t destOffset,                                    \
      const application::MemoryRegion& source, size_t sourceOffset, T val, T compare, int pe,      \
      core::atomicType amoType) {                                                                  \
    ShmemAtomicSizeFetch##Scope(dest, destOffset, source, sourceOffset, &val, &compare, sizeof(T), \
                                pe, amoType);                                                      \
    uintptr_t fetchResultPtr = source.addr + sourceOffset;                                         \
    return core::AtomicLoadRelaxedSystem<T>(reinterpret_cast<T*>(fetchResultPtr));                 \
  }

SHMEM_ATOMIC_TYPE_FETCH_API_TEMPLATE(Thread)
SHMEM_ATOMIC_TYPE_FETCH_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(TypeName, T, Scope)                                  \
  inline __device__ T ShmemAtomic##TypeName##Fetch##Scope(                                      \
      const application::SymmMemObjPtr dest, size_t destOffset,                                 \
      const application::MemoryRegion& source, size_t sourceOffset, T val, T compare, int pe,   \
      core::atomicType amoType) {                                                               \
    return ShmemAtomicTypeFetch##Scope<T>(dest, destOffset, source, sourceOffset, val, compare, \
                                          pe, amoType);                                         \
  }

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Int32, int32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Int64, int64_t, Thread)

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Int32, int32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Int64, int64_t, Warp)

/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ void ShmemQuietThread() {
  ShmemQuietThreadKernel<application::TransportType::RDMA>();
}

template <typename T>
inline __device__ T ShmemTypeWaitUntilGreaterThan(T* addr, T val) {
  T got;
  do {
    got = core::AtomicLoadRelaxedSystem(addr);
  } while (got <= val);
  return got;
  // while (core::AtomicLoadRelaxedSystem(addr) <= val) {
  // }
}

#define DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(TypeName, T)                \
  inline __device__ T Shmem##TypeName##WaitUntilGreaterThan(T* addr, T val) { \
    return ShmemTypeWaitUntilGreaterThan<T>(addr, val);                       \
  }

DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint8, uint8_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint16, uint16_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint32, uint32_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint64, uint64_t)

template <typename T>
inline __device__ void ShmemTypeWaitUntilEquals(T* addr, T val) {
  while (core::AtomicLoadRelaxedSystem(addr) != val) {
  }
}

#define DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(TypeName, T)                     \
  inline __device__ void Shmem##TypeName##WaitUntilEquals(T* addr, T val) { \
    ShmemTypeWaitUntilEquals<T>(addr, val);                                 \
  }

DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Uint8, uint8_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Uint16, uint16_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Uint32, uint32_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Uint64, uint64_t)

}  // namespace shmem
}  // namespace mori
