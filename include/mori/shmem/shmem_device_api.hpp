// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include <assert.h>
#include <mpi.h>

#include "mori/application/application.hpp"
#include "mori/core/core.hpp"
#include "mori/shmem/internal.hpp"
#include "mori/shmem/shmem_device_kernels.hpp"
#include "mori/shmem/shmem_ibgda_kernels.hpp"
#include "mori/shmem/shmem_p2p_kernels.hpp"
#include "mori/shmem/shmem_sdma_kernels.hpp"

namespace mori {
namespace shmem {

#define DISPATCH_TRANSPORT_TYPE(func, pe, ...)                                    \
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();                           \
  application::TransportType transportType = globalGpuStates->transportTypes[pe]; \
  if (transportType == application::TransportType::RDMA) {                        \
    func<application::TransportType::RDMA>(__VA_ARGS__);                          \
  } else if (transportType == application::TransportType::P2P) {                  \
    func<application::TransportType::P2P>(__VA_ARGS__);                           \
  } else if (transportType == application::TransportType::SDMA) {                 \
    func<application::TransportType::SDMA>(__VA_ARGS__);                          \
  } else {                                                                        \
    assert(false);                                                                \
  }

#define DISPATCH_TRANSPORT_TYPE_WITH_BOOL(func, boolParam, pe, ...)               \
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();                           \
  application::TransportType transportType = globalGpuStates->transportTypes[pe]; \
  if (transportType == application::TransportType::RDMA) {                        \
    func<application::TransportType::RDMA, boolParam>(__VA_ARGS__);               \
  } else if (transportType == application::TransportType::P2P) {                  \
    func<application::TransportType::P2P, boolParam>(__VA_ARGS__);                \
  } else {                                                                        \
    assert(false);                                                                \
  }

#define DISPATCH_TRANSPORT_DATA_TYPE_WITH_RETURN(func, pe, type, ...)               \
  [&]() {                                                                           \
    GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();                           \
    application::TransportType transportType = globalGpuStates->transportTypes[pe]; \
    if (transportType == application::TransportType::RDMA) {                        \
      return func<application::TransportType::RDMA, type>(__VA_ARGS__);             \
    } else if (transportType == application::TransportType::P2P) {                  \
      return func<application::TransportType::P2P, type>(__VA_ARGS__);              \
    } else {                                                                        \
      assert(false);                                                                \
      return type{};                                                                \
    }                                                                               \
  }()

/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ void ShmemQuietThread() {
  ShmemQuietThreadKernel<application::TransportType::RDMA>();
}

inline __device__ void ShmemQuietThread(int pe) {
  DISPATCH_TRANSPORT_TYPE(ShmemQuietThreadKernel, pe, pe);
}

inline __device__ void ShmemQuietThread(int pe, int qpId) {
  DISPATCH_TRANSPORT_TYPE(ShmemQuietThreadKernel, pe, pe, qpId);
}

inline __device__ void ShmemFenceThread() {
  ShmemQuietThread();
  __threadfence_system();
}

inline __device__ void ShmemFenceThread(int pe) {
  ShmemQuietThread(pe);
  __threadfence_system();
}

inline __device__ void ShmemFenceThread(int pe, int qpId) {
  ShmemQuietThread(pe, qpId);
  __threadfence_system();
}
inline __device__ void ShmemQuietThread(int pe, const application::SymmMemObjPtr dest) {
  ShmemQuietThreadKernel<application::TransportType::SDMA>(pe, dest);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Point-to-Point                                         */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ uint64_t ShmemPtrP2p(const uint64_t destPtr, const int myPe, int destPe) {
  // If same PE, return the pointer directly
  if (myPe == destPe) {
    return destPtr;
  }

  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();

  application::TransportType transportType = globalGpuStates->transportTypes[destPe];
  if (transportType == application::TransportType::RDMA) {
    return 0;
  }

  uintptr_t localAddrInt = static_cast<uintptr_t>(destPtr);

  if (localAddrInt < globalGpuStates->heapBaseAddr ||
      localAddrInt >= globalGpuStates->heapEndAddr) {
    assert(false && "dest addr not in symmetric heap");
    return 0;
  }

  size_t offset = localAddrInt - globalGpuStates->heapBaseAddr;

  application::SymmMemObj* heapObj = globalGpuStates->heapObj;
  uint64_t raddr = heapObj->peerPtrs[destPe] + offset;

  return raddr;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        PutNbi APIs                                             */
/* ---------------------------------------------------------------------------------------------- */
#define DEFINE_SHMEM_PUT_MEM_NBI_API_TEMPLATE(Scope)                                      \
  inline __device__ void ShmemPutMemNbi##Scope(                                           \
      const application::SymmMemObjPtr dest, size_t destOffset,                           \
      const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe, \
      int qpId = 0) {                                                                     \
    DISPATCH_TRANSPORT_TYPE(ShmemPutMemNbi##Scope##Kernel, pe, dest, destOffset, source,  \
                            sourceOffset, bytes, pe, qpId);                               \
  }

DEFINE_SHMEM_PUT_MEM_NBI_API_TEMPLATE(Thread)
DEFINE_SHMEM_PUT_MEM_NBI_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_PUT_TYPE_NBI_API_TEMPLATE(Scope)                                      \
  template <typename T>                                                                    \
  inline __device__ void ShmemPutTypeNbi##Scope(                                           \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                         \
      const application::SymmMemObjPtr source, size_t srcElmOffset, size_t nelems, int pe, \
      int qpId = 0) {                                                                      \
    constexpr size_t typeSize = sizeof(T);                                                 \
    ShmemPutMemNbi##Scope(dest, destElmOffset * typeSize, source, srcElmOffset * typeSize, \
                          nelems * typeSize, pe, qpId);                                    \
  }

DEFINE_SHMEM_PUT_TYPE_NBI_API_TEMPLATE(Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_PUT_TYPE_NBI_API(TypeName, T, Scope)                                   \
  inline __device__ void ShmemPut##TypeName##Nbi##Scope(                                    \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                          \
      const application::SymmMemObjPtr source, size_t srcElmOffset, size_t nelems, int pe,  \
      int qpId = 0) {                                                                       \
    ShmemPutTypeNbi##Scope<T>(dest, destElmOffset, source, srcElmOffset, nelems, pe, qpId); \
  }

DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint8, uint8_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Int8, int8_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Schar, signed char, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint16, uint16_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Int16, int16_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Int32, int32_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Int64, int64_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Float, float, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Double, double, Thread)

DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint8, uint8_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Int8, int8_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Schar, signed char, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint16, uint16_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Int16, int16_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Int32, int32_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Int64, int64_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Float, float, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Double, double, Warp)

/* ---------------------------------------------------------------------------------------------- */
/*                                       PutNbi Inline APIs                                       */
/* ---------------------------------------------------------------------------------------------- */
// TODO: deal with bytes count limit
#define SHMEM_PUT_SIZE_IMM_NBI_API(Scope)                                                        \
  inline __device__ void ShmemPutSizeImmNbi##Scope(const application::SymmMemObjPtr dest,        \
                                                   size_t destOffset, void* val, size_t bytes,   \
                                                   int pe, int qpId = 0) {                       \
    DISPATCH_TRANSPORT_TYPE(ShmemPutSizeImmNbi##Scope##Kernel, pe, dest, destOffset, val, bytes, \
                            pe, qpId);                                                           \
  }

SHMEM_PUT_SIZE_IMM_NBI_API(Thread)
SHMEM_PUT_SIZE_IMM_NBI_API(Warp)

#define SHMEM_PUT_TYPE_IMM_NBI_API_TEMPLATE(Scope)                                             \
  template <typename T>                                                                        \
  inline __device__ void ShmemPutTypeImmNbi##Scope(                                            \
      const application::SymmMemObjPtr dest, size_t destOffset, T val, int pe, int qpId = 0) { \
    static_assert(sizeof(T) <= core::MaxInlineDataSizePerWqe);                                 \
    ShmemPutSizeImmNbi##Scope(dest, destOffset, &val, sizeof(T), pe, qpId);                    \
  }

SHMEM_PUT_TYPE_IMM_NBI_API_TEMPLATE(Thread)
SHMEM_PUT_TYPE_IMM_NBI_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(TypeName, T, Scope)                                     \
  inline __device__ void ShmemPut##TypeName##ImmNbi##Scope(const application::SymmMemObjPtr dest, \
                                                           size_t destOffset, uint32_t val,       \
                                                           int pe, int qpId = 0) {                \
    ShmemPutTypeImmNbi##Scope<T>(dest, destOffset, val, pe, qpId);                                \
  }

DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint8, uint8_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Int8, int8_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Schar, signed char, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint16, uint16_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Int16, int16_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Int32, int32_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Int64, int64_t, Thread)

DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint8, uint8_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Int8, int8_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Schar, signed char, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint16, uint16_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Int16, int16_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Int32, int32_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_API(Int64, int64_t, Warp)

/* ---------------------------------------------------------------------------------------------- */
/*                                      PutNbi with Signal APIs                                   */
/* ---------------------------------------------------------------------------------------------- */
// PutNbi with Signal - Memory version
#define DEFINE_SHMEM_PUT_MEM_NBI_SIGNAL_API_TEMPLATE(Scope)                                       \
  template <bool onlyOneSignal = true>                                                            \
  inline __device__ void ShmemPutMemNbiSignal##Scope(                                             \
      const application::SymmMemObjPtr dest, size_t destOffset,                                   \
      const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,                 \
      const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue, \
      core::atomicType signalOp, int pe, int qpId = 0) {                                          \
    DISPATCH_TRANSPORT_TYPE_WITH_BOOL(ShmemPutMemNbiSignal##Scope##Kernel, onlyOneSignal, pe,     \
                                      dest, destOffset, source, sourceOffset, bytes, signalDest,  \
                                      signalDestOffset, signalValue, signalOp, pe, qpId);         \
  }

DEFINE_SHMEM_PUT_MEM_NBI_SIGNAL_API_TEMPLATE(Thread)
DEFINE_SHMEM_PUT_MEM_NBI_SIGNAL_API_TEMPLATE(Warp)

// PutNbi with Signal - Typed version
#define DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API_TEMPLATE(Scope)                                      \
  template <typename T, bool onlyOneSignal = true>                                                \
  inline __device__ void ShmemPutTypeNbiSignal##Scope(                                            \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                                \
      const application::SymmMemObjPtr source, size_t srcElmOffset, size_t nelems,                \
      const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue, \
      core::atomicType signalOp, int pe, int qpId = 0) {                                          \
    constexpr size_t typeSize = sizeof(T);                                                        \
    ShmemPutMemNbiSignal##Scope<onlyOneSignal>(                                                   \
        dest, destElmOffset * typeSize, source, srcElmOffset * typeSize, nelems * typeSize,       \
        signalDest, signalDestOffset, signalValue, signalOp, pe, qpId);                           \
  }

DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API_TEMPLATE(Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API_TEMPLATE(Warp)

// PutNbi with Signal - Concrete typed versions
#define DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(TypeName, T, Scope)                                  \
  template <bool onlyOneSignal = true>                                                            \
  inline __device__ void ShmemPut##TypeName##NbiSignal##Scope(                                    \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                                \
      const application::SymmMemObjPtr source, size_t srcElmOffset, size_t nelems,                \
      const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue, \
      core::atomicType signalOp, int pe, int qpId = 0) {                                          \
    ShmemPutTypeNbiSignal##Scope<T, onlyOneSignal>(dest, destElmOffset, source, srcElmOffset,     \
                                                   nelems, signalDest, signalDestOffset,          \
                                                   signalValue, signalOp, pe, qpId);              \
  }

DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Uint8, uint8_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Int8, int8_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Schar, signed char, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Uint16, uint16_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Int16, int16_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Int32, int32_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Int64, int64_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Float, float, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Double, double, Thread)

DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Uint8, uint8_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Int8, int8_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Schar, signed char, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Uint16, uint16_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Int16, int16_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Int32, int32_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Int64, int64_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Float, float, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Double, double, Warp)

#define SHMEM_ATOMIC_SIZE_NONFETCH_API_TEMPLATE(Scope)                                         \
  inline __device__ void ShmemAtomicSizeNonFetch##Scope(                                       \
      const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes,       \
      core::atomicType amoType, int pe, int qpId = 0) {                                        \
    DISPATCH_TRANSPORT_TYPE(ShmemAtomicSizeNonFetch##Scope##Kernel, pe, dest, destOffset, val, \
                            bytes, amoType, pe, qpId);                                         \
  }

SHMEM_ATOMIC_SIZE_NONFETCH_API_TEMPLATE(Thread)
SHMEM_ATOMIC_SIZE_NONFETCH_API_TEMPLATE(Warp)

#define SHMEM_ATOMIC_TYPE_NONFETCH_API_TEMPLATE(Scope)                                           \
  template <typename T>                                                                          \
  inline __device__ void ShmemAtomicTypeNonFetch##Scope(                                         \
      const application::SymmMemObjPtr dest, size_t destOffset, T val, core::atomicType amoType, \
      int pe, int qpId = 0) {                                                                    \
    ShmemAtomicSizeNonFetch##Scope(dest, destOffset, &val, sizeof(T), amoType, pe, qpId);        \
  }

SHMEM_ATOMIC_TYPE_NONFETCH_API_TEMPLATE(Thread)
SHMEM_ATOMIC_TYPE_NONFETCH_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(TypeName, T, Scope)                                \
  inline __device__ void ShmemAtomic##TypeName##NonFetch##Scope(                                 \
      const application::SymmMemObjPtr dest, size_t destOffset, T val, core::atomicType amoType, \
      int pe, int qpId = 0) {                                                                    \
    ShmemAtomicTypeNonFetch##Scope<T>(dest, destOffset, val, amoType, pe, qpId);                 \
  }

DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Int32, int32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Int64, int64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Long, long, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Ulong, unsigned long, Thread)

DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Int32, int32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Int64, int64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Long, long, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_API(Ulong, unsigned long, Warp)

/* ---------------------------------------------------------------------------------------------- */
/*                                       Atomic Fetch APIs                                        */
/* ---------------------------------------------------------------------------------------------- */
#define SHMEM_ATOMIC_TYPE_FETCH_API_TEMPLATE(Scope)                                              \
  template <typename T>                                                                          \
  inline __device__ T ShmemAtomicTypeFetch##Scope(                                               \
      const application::SymmMemObjPtr dest, size_t destOffset, T val, T compare,                \
      core::atomicType amoType, int pe, int qpId = 0) {                                          \
    T result = DISPATCH_TRANSPORT_DATA_TYPE_WITH_RETURN(ShmemAtomicTypeFetch##Scope##Kernel, pe, \
                                                        T, dest, destOffset, &val, &compare,     \
                                                        sizeof(T), amoType, pe, qpId);           \
    return result;                                                                               \
  }

SHMEM_ATOMIC_TYPE_FETCH_API_TEMPLATE(Thread)
SHMEM_ATOMIC_TYPE_FETCH_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(TypeName, T, Scope)                                \
  inline __device__ T ShmemAtomic##TypeName##Fetch##Scope(                                    \
      const application::SymmMemObjPtr dest, size_t destOffset, T val, T compare,             \
      core::atomicType amoType, int pe, int qpId = 0) {                                       \
    return ShmemAtomicTypeFetch##Scope<T>(dest, destOffset, val, compare, amoType, pe, qpId); \
  }

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Int32, int32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Int64, int64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Long, long, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Ulong, unsigned long, Thread)

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Int32, int32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Int64, int64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Long, long, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_API(Ulong, unsigned long, Warp)

/* ---------------------------------------------------------------------------------------------- */
/*                              Atomic Add Convenience APIs (NonFetch)                            */
/* ---------------------------------------------------------------------------------------------- */
#define DEFINE_SHMEM_ATOMIC_TYPE_ADD_API(TypeName, T, Scope)                                   \
  inline __device__ void Shmem##TypeName##AtomicAdd##Scope(                                    \
      const application::SymmMemObjPtr dest, size_t destOffset, T val, int pe, int qpId = 0) { \
    ShmemAtomicTypeNonFetch##Scope<T>(dest, destOffset, val, core::AMO_ADD, pe, qpId);         \
  }

DEFINE_SHMEM_ATOMIC_TYPE_ADD_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_API(Int32, int32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_API(Int64, int64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_API(Long, long, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_API(Ulong, unsigned long, Thread)

DEFINE_SHMEM_ATOMIC_TYPE_ADD_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_API(Int32, int32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_API(Int64, int64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_API(Long, long, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_API(Ulong, unsigned long, Warp)

/* ---------------------------------------------------------------------------------------------- */
/*                              Atomic Add Convenience APIs (Fetch)                               */
/* ---------------------------------------------------------------------------------------------- */
#define DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_API(TypeName, T, Scope)                                 \
  inline __device__ T Shmem##TypeName##AtomicFetchAdd##Scope(                                      \
      const application::SymmMemObjPtr dest, size_t destOffset, T val, int pe, int qpId = 0) {     \
    T compare = 0;                                                                                 \
    return ShmemAtomicTypeFetch##Scope<T>(dest, destOffset, val, compare, core::AMO_FETCH_ADD, pe, \
                                          qpId);                                                   \
  }

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_API(Int32, int32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_API(Int64, int64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_API(Long, long, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_API(Ulong, unsigned long, Thread)

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_API(Int32, int32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_API(Int64, int64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_API(Long, long, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_API(Ulong, unsigned long, Warp)

/* ---------------------------------------------------------------------------------------------- */
/*                          Pure Address-Based APIs (OpenSHMEM Style)                             */
/* ---------------------------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------------------------------- */
/*                                        PutNbi APIs                                             */
/* ---------------------------------------------------------------------------------------------- */
#define DEFINE_SHMEM_PUT_MEM_NBI_ADDR_API_TEMPLATE(Scope)                                      \
  inline __device__ void ShmemPutMemNbi##Scope(void* dest, const void* source, size_t bytes,   \
                                               int pe, int qpId = 0) {                         \
    DISPATCH_TRANSPORT_TYPE(ShmemPutMemNbi##Scope##Kernel, pe, dest, source, bytes, pe, qpId); \
  }

DEFINE_SHMEM_PUT_MEM_NBI_ADDR_API_TEMPLATE(Thread)
DEFINE_SHMEM_PUT_MEM_NBI_ADDR_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API_TEMPLATE(Scope)                                       \
  template <typename T>                                                                          \
  inline __device__ void ShmemPutTypeNbi##Scope(T* dest, const T* source, size_t nelems, int pe, \
                                                int qpId = 0) {                                  \
    ShmemPutMemNbi##Scope(dest, source, nelems * sizeof(T), pe, qpId);                           \
  }

DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API_TEMPLATE(Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(TypeName, T, Scope)                                   \
  inline __device__ void ShmemPut##TypeName##Nbi##Scope(T* dest, const T* source, size_t nelems, \
                                                        int pe, int qpId = 0) {                  \
    ShmemPutTypeNbi##Scope<T>(dest, source, nelems, pe, qpId);                                   \
  }

DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Uint8, uint8_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Int8, int8_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Schar, signed char, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Uint16, uint16_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Int16, int16_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Int32, int32_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Int64, int64_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Float, float, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Double, double, Thread)

DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Uint8, uint8_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Int8, int8_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Schar, signed char, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Uint16, uint16_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Int16, int16_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Int32, int32_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Int64, int64_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Float, float, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Double, double, Warp)

/* ---------------------------------------------------------------------------------------------- */
/*                                       PutNbi Inline APIs                                       */
/* ---------------------------------------------------------------------------------------------- */
#define SHMEM_PUT_SIZE_IMM_NBI_ADDR_API(Scope)                                                  \
  inline __device__ void ShmemPutSizeImmNbi##Scope(void* dest, void* val, size_t bytes, int pe, \
                                                   int qpId = 0) {                              \
    DISPATCH_TRANSPORT_TYPE(ShmemPutSizeImmNbi##Scope##Kernel, pe, dest, val, bytes, pe, qpId); \
  }

SHMEM_PUT_SIZE_IMM_NBI_ADDR_API(Thread)
SHMEM_PUT_SIZE_IMM_NBI_ADDR_API(Warp)

#define SHMEM_PUT_TYPE_IMM_NBI_ADDR_API_TEMPLATE(Scope)                                    \
  template <typename T>                                                                    \
  inline __device__ void ShmemPutTypeImmNbi##Scope(T* dest, T val, int pe, int qpId = 0) { \
    static_assert(sizeof(T) <= core::MaxInlineDataSizePerWqe);                             \
    ShmemPutSizeImmNbi##Scope(dest, &val, sizeof(T), pe, qpId);                            \
  }

SHMEM_PUT_TYPE_IMM_NBI_ADDR_API_TEMPLATE(Thread)
SHMEM_PUT_TYPE_IMM_NBI_ADDR_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(TypeName, T, Scope)                                 \
  inline __device__ void ShmemPut##TypeName##ImmNbi##Scope(T* dest, T val, int pe, int qpId = 0) { \
    ShmemPutTypeImmNbi##Scope<T>(dest, val, pe, qpId);                                             \
  }

DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Uint8, uint8_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Int8, int8_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Schar, signed char, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Uint16, uint16_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Int16, int16_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Int32, int32_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Int64, int64_t, Thread)

DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Uint8, uint8_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Int8, int8_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Schar, signed char, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Uint16, uint16_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Int16, int16_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Int32, int32_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_PUT_TYPE_IMM_NBI_ADDR_API(Int64, int64_t, Warp)

/* ---------------------------------------------------------------------------------------------- */
/*                                      PutNbi with Signal APIs                                   */
/* ---------------------------------------------------------------------------------------------- */
#define DEFINE_SHMEM_PUT_MEM_NBI_SIGNAL_ADDR_API_TEMPLATE(Scope)                                  \
  template <bool onlyOneSignal = true>                                                            \
  inline __device__ void ShmemPutMemNbiSignal##Scope(                                             \
      void* dest, const void* source, size_t bytes, void* signalDest, uint64_t signalValue,       \
      core::atomicType signalOp, int pe, int qpId = 0) {                                          \
    DISPATCH_TRANSPORT_TYPE_WITH_BOOL(ShmemPutMemNbiSignal##Scope##Kernel, onlyOneSignal, pe,     \
                                      dest, source, bytes, signalDest, signalValue, signalOp, pe, \
                                      qpId);                                                      \
  }

DEFINE_SHMEM_PUT_MEM_NBI_SIGNAL_ADDR_API_TEMPLATE(Thread)
DEFINE_SHMEM_PUT_MEM_NBI_SIGNAL_ADDR_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API_TEMPLATE(Scope)                            \
  template <typename T, bool onlyOneSignal = true>                                           \
  inline __device__ void ShmemPutTypeNbiSignal##Scope(                                       \
      T* dest, const T* source, size_t nelems, uint64_t* signalDest, uint64_t signalValue,   \
      core::atomicType signalOp, int pe, int qpId = 0) {                                     \
    ShmemPutMemNbiSignal##Scope<onlyOneSignal>(dest, source, nelems * sizeof(T), signalDest, \
                                               signalValue, signalOp, pe, qpId);             \
  }

DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API_TEMPLATE(Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(TypeName, T, Scope)                             \
  template <bool onlyOneSignal = true>                                                            \
  inline __device__ void ShmemPut##TypeName##NbiSignal##Scope(                                    \
      T* dest, const T* source, size_t nelems, uint64_t* signalDest, uint64_t signalValue,        \
      core::atomicType signalOp, int pe, int qpId = 0) {                                          \
    ShmemPutTypeNbiSignal##Scope<T, onlyOneSignal>(dest, source, nelems, signalDest, signalValue, \
                                                   signalOp, pe, qpId);                           \
  }

DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Uint8, uint8_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Int8, int8_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Schar, signed char, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Uint16, uint16_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Int16, int16_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Int32, int32_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Int64, int64_t, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Float, float, Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Double, double, Thread)

DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Uint8, uint8_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Int8, int8_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Schar, signed char, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Uint16, uint16_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Int16, int16_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Int32, int32_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Int64, int64_t, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Float, float, Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Double, double, Warp)

#define SHMEM_ATOMIC_SIZE_NONFETCH_ADDR_API_TEMPLATE(Scope)                                        \
  inline __device__ void ShmemAtomicSizeNonFetch##Scope(                                           \
      void* dest, void* val, size_t bytes, core::atomicType amoType, int pe, int qpId = 0) {       \
    DISPATCH_TRANSPORT_TYPE(ShmemAtomicSizeNonFetch##Scope##Kernel, pe, dest, val, bytes, amoType, \
                            pe, qpId);                                                             \
  }

SHMEM_ATOMIC_SIZE_NONFETCH_ADDR_API_TEMPLATE(Thread)
SHMEM_ATOMIC_SIZE_NONFETCH_ADDR_API_TEMPLATE(Warp)

#define SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API_TEMPLATE(Scope)                                       \
  template <typename T>                                                                           \
  inline __device__ void ShmemAtomicTypeNonFetch##Scope(T* dest, T val, core::atomicType amoType, \
                                                        int pe, int qpId = 0) {                   \
    ShmemAtomicSizeNonFetch##Scope(dest, &val, sizeof(T), amoType, pe, qpId);                     \
  }

SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API_TEMPLATE(Thread)
SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API(TypeName, T, Scope)  \
  inline __device__ void ShmemAtomic##TypeName##NonFetch##Scope(        \
      T* dest, T val, core::atomicType amoType, int pe, int qpId = 0) { \
    ShmemAtomicTypeNonFetch##Scope<T>(dest, val, amoType, pe, qpId);    \
  }

DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API(Int32, int32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API(Int64, int64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API(Long, long, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API(Ulong, unsigned long, Thread)

DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API(Int32, int32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API(Int64, int64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API(Long, long, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_NONFETCH_ADDR_API(Ulong, unsigned long, Warp)

/* ---------------------------------------------------------------------------------------------- */
/*                                       Atomic Fetch APIs                                        */
/* ---------------------------------------------------------------------------------------------- */
#define SHMEM_ATOMIC_TYPE_FETCH_ADDR_API_TEMPLATE(Scope)                                           \
  template <typename T>                                                                            \
  inline __device__ T ShmemAtomicTypeFetch##Scope(                                                 \
      T* dest, T val, T compare, core::atomicType amoType, int pe, int qpId = 0) {                 \
    T result =                                                                                     \
        DISPATCH_TRANSPORT_DATA_TYPE_WITH_RETURN(ShmemAtomicTypeFetch##Scope##Kernel, pe, T, dest, \
                                                 &val, &compare, sizeof(T), amoType, pe, qpId);    \
    return result;                                                                                 \
  }

SHMEM_ATOMIC_TYPE_FETCH_ADDR_API_TEMPLATE(Thread)
SHMEM_ATOMIC_TYPE_FETCH_ADDR_API_TEMPLATE(Warp)

#define DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADDR_API(TypeName, T, Scope)                \
  inline __device__ T ShmemAtomic##TypeName##Fetch##Scope(                         \
      T* dest, T val, T compare, core::atomicType amoType, int pe, int qpId = 0) { \
    return ShmemAtomicTypeFetch##Scope<T>(dest, val, compare, amoType, pe, qpId);  \
  }

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADDR_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADDR_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADDR_API(Int32, int32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADDR_API(Int64, int64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADDR_API(Long, long, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADDR_API(Ulong, unsigned long, Thread)

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADDR_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADDR_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADDR_API(Int32, int32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADDR_API(Int64, int64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADDR_API(Long, long, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADDR_API(Ulong, unsigned long, Warp)

/* ---------------------------------------------------------------------------------------------- */
/*                   Atomic Add Convenience APIs (NonFetch, Pure Address)                         */
/* ---------------------------------------------------------------------------------------------- */
#define DEFINE_SHMEM_ATOMIC_TYPE_ADD_ADDR_API(TypeName, T, Scope)                                  \
  inline __device__ void Shmem##TypeName##AtomicAdd##Scope(T* dest, T val, int pe, int qpId = 0) { \
    ShmemAtomicTypeNonFetch##Scope<T>(dest, val, core::AMO_ADD, pe, qpId);                         \
  }

DEFINE_SHMEM_ATOMIC_TYPE_ADD_ADDR_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_ADDR_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_ADDR_API(Int32, int32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_ADDR_API(Int64, int64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_ADDR_API(Long, long, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_ADDR_API(Ulong, unsigned long, Thread)

DEFINE_SHMEM_ATOMIC_TYPE_ADD_ADDR_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_ADDR_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_ADDR_API(Int32, int32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_ADDR_API(Int64, int64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_ADDR_API(Long, long, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_ADD_ADDR_API(Ulong, unsigned long, Warp)

/* ---------------------------------------------------------------------------------------------- */
/*                     Atomic Add Convenience APIs (Fetch, Pure Address)                          */
/* ---------------------------------------------------------------------------------------------- */
#define DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_ADDR_API(TypeName, T, Scope)                       \
  inline __device__ T Shmem##TypeName##AtomicFetchAdd##Scope(T* dest, T val, int pe,          \
                                                             int qpId = 0) {                  \
    T compare = 0;                                                                            \
    return ShmemAtomicTypeFetch##Scope<T>(dest, val, compare, core::AMO_FETCH_ADD, pe, qpId); \
  }

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_ADDR_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_ADDR_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_ADDR_API(Int32, int32_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_ADDR_API(Int64, int64_t, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_ADDR_API(Long, long, Thread)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_ADDR_API(Ulong, unsigned long, Thread)

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_ADDR_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_ADDR_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_ADDR_API(Int32, int32_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_ADDR_API(Int64, int64_t, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_ADDR_API(Long, long, Warp)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_ADD_ADDR_API(Ulong, unsigned long, Warp)

/* ---------------------------------------------------------------------------------------------- */
/*                                    Wait Until Greater Than APIs                                */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ T ShmemTypeWaitUntilGreaterThan(T* addr, T val) {
  T got;
  do {
    got = core::AtomicLoadRelaxedSystem(addr);
  } while (got <= val);
  return got;
}

#define DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(TypeName, T)                \
  inline __device__ T Shmem##TypeName##WaitUntilGreaterThan(T* addr, T val) { \
    return ShmemTypeWaitUntilGreaterThan<T>(addr, val);                       \
  }

DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint8, uint8_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Int8, int8_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Schar, signed char)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint16, uint16_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Int16, int16_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint32, uint32_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Int32, int32_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Uint64, uint64_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_GREATER_THAN(Int64, int64_t)

/* ---------------------------------------------------------------------------------------------- */
/*                                       Wait Until Equal APIs                                    */
/* ---------------------------------------------------------------------------------------------- */
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
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Int8, int8_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Schar, signed char)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Uint16, uint16_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Int16, int16_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Uint32, uint32_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Int32, int32_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Uint64, uint64_t)
DEFINE_SHMEM_TYPE_WAIT_UNTIL_EQUAL(Int64, int64_t)

/* ---------------------------------------------------------------------------------------------- */
/*                                       Query APIs                                               */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ int ShmemMyPe() {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  return globalGpuStates->rank;
}

inline __device__ int ShmemNPes() {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  return globalGpuStates->worldSize;
}

}  // namespace shmem
}  // namespace mori
