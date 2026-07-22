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

#include "mori/application/application_device_types.hpp"
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
  uintptr_t localAddrInt = static_cast<uintptr_t>(destPtr);

  if (localAddrInt < globalGpuStates->heapBaseAddr ||
      localAddrInt >= globalGpuStates->heapEndAddr) {
    return 0;
  }

  size_t offset = localAddrInt - globalGpuStates->heapBaseAddr;

  application::SymmMemObj* heapObj = globalGpuStates->heapObj;
  // Use p2pPeerPtrs directly - returns 0 for non-P2P peers
  uint64_t raddr = heapObj->p2pPeerPtrs[destPe] + offset;

  return raddr;
}

// Overload that gets myPe from globalGpuStates
inline __device__ uint64_t ShmemPtrP2p(const uint64_t destPtr, int destPe) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  return ShmemPtrP2p(destPtr, globalGpuStates->rank, destPe);
}

// Overload that takes SymmMemObjPtr with offset (consistent with other shmem APIs)
inline __device__ uint64_t ShmemPtrP2p(const application::SymmMemObjPtr& memObjPtr, size_t offset,
                                       const int myPe, int destPe) {
  // If same PE, return local pointer + offset
  if (myPe == destPe) {
    return reinterpret_cast<uintptr_t>(memObjPtr->localPtr) + offset;
  }

  // Use p2pPeerPtrs directly - returns 0 for non-P2P peers
  // operator-> automatically returns gpu pointer in device code
  uint64_t raddr = memObjPtr->p2pPeerPtrs[destPe] + offset;

  return raddr;
}

// Overload that takes SymmMemObjPtr with offset and gets myPe from globalGpuStates
inline __device__ uint64_t ShmemPtrP2p(const application::SymmMemObjPtr& memObjPtr, size_t offset,
                                       int destPe) {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  return ShmemPtrP2p(memObjPtr, offset, globalGpuStates->rank, destPe);
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
DEFINE_SHMEM_PUT_MEM_NBI_API_TEMPLATE(Block)

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
DEFINE_SHMEM_PUT_TYPE_NBI_API_TEMPLATE(Block)

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

DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint8, uint8_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Int8, int8_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Schar, signed char, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint16, uint16_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Int16, int16_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint32, uint32_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Int32, int32_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Uint64, uint64_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Int64, int64_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Float, float, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_API(Double, double, Block)

/* ---------------------------------------------------------------------------------------------- */
/*                                         GetNbi APIs                                            */
/* ---------------------------------------------------------------------------------------------- */
#define DEFINE_SHMEM_GET_MEM_NBI_API_TEMPLATE(Scope)                                      \
  inline __device__ void ShmemGetMemNbi##Scope(                                           \
      const application::SymmMemObjPtr dest, size_t destOffset,                           \
      const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe, \
      int qpId = 0) {                                                                     \
    DISPATCH_TRANSPORT_TYPE(ShmemGetMemNbi##Scope##Kernel, pe, dest, destOffset, source,  \
                            sourceOffset, bytes, pe, qpId);                               \
  }

DEFINE_SHMEM_GET_MEM_NBI_API_TEMPLATE(Thread)
DEFINE_SHMEM_GET_MEM_NBI_API_TEMPLATE(Warp)
DEFINE_SHMEM_GET_MEM_NBI_API_TEMPLATE(Block)

#define DEFINE_SHMEM_GET_TYPE_NBI_API_TEMPLATE(Scope)                                      \
  template <typename T>                                                                    \
  inline __device__ void ShmemGetTypeNbi##Scope(                                           \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                         \
      const application::SymmMemObjPtr source, size_t srcElmOffset, size_t nelems, int pe, \
      int qpId = 0) {                                                                      \
    constexpr size_t typeSize = sizeof(T);                                                 \
    ShmemGetMemNbi##Scope(dest, destElmOffset * typeSize, source, srcElmOffset * typeSize, \
                          nelems * typeSize, pe, qpId);                                    \
  }

DEFINE_SHMEM_GET_TYPE_NBI_API_TEMPLATE(Thread)
DEFINE_SHMEM_GET_TYPE_NBI_API_TEMPLATE(Warp)
DEFINE_SHMEM_GET_TYPE_NBI_API_TEMPLATE(Block)

#define DEFINE_SHMEM_GET_TYPE_NBI_API(TypeName, T, Scope)                                   \
  inline __device__ void ShmemGet##TypeName##Nbi##Scope(                                    \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                          \
      const application::SymmMemObjPtr source, size_t srcElmOffset, size_t nelems, int pe,  \
      int qpId = 0) {                                                                       \
    ShmemGetTypeNbi##Scope<T>(dest, destElmOffset, source, srcElmOffset, nelems, pe, qpId); \
  }

DEFINE_SHMEM_GET_TYPE_NBI_API(Uint8, uint8_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_API(Int8, int8_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_API(Schar, signed char, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_API(Uint16, uint16_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_API(Int16, int16_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_API(Int32, int32_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_API(Int64, int64_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_API(Float, float, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_API(Double, double, Thread)

DEFINE_SHMEM_GET_TYPE_NBI_API(Uint8, uint8_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_API(Int8, int8_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_API(Schar, signed char, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_API(Uint16, uint16_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_API(Int16, int16_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_API(Int32, int32_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_API(Int64, int64_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_API(Float, float, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_API(Double, double, Warp)

DEFINE_SHMEM_GET_TYPE_NBI_API(Uint8, uint8_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_API(Int8, int8_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_API(Schar, signed char, Block)
DEFINE_SHMEM_GET_TYPE_NBI_API(Uint16, uint16_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_API(Int16, int16_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_API(Uint32, uint32_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_API(Int32, int32_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_API(Uint64, uint64_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_API(Int64, int64_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_API(Float, float, Block)
DEFINE_SHMEM_GET_TYPE_NBI_API(Double, double, Block)

/* ---------------------------------------------------------------------------------------------- */
/*                                     Blocking GET APIs                                          */
/* ---------------------------------------------------------------------------------------------- */
#define DEFINE_SHMEM_GET_MEM_API_TEMPLATE(Scope)                                          \
  inline __device__ void ShmemGetMem##Scope(                                              \
      const application::SymmMemObjPtr dest, size_t destOffset,                           \
      const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe, \
      int qpId = 0) {                                                                     \
    ShmemGetMemNbi##Scope(dest, destOffset, source, sourceOffset, bytes, pe, qpId);       \
    ShmemQuietThread(pe, qpId);                                                           \
  }

DEFINE_SHMEM_GET_MEM_API_TEMPLATE(Thread)
DEFINE_SHMEM_GET_MEM_API_TEMPLATE(Warp)
DEFINE_SHMEM_GET_MEM_API_TEMPLATE(Block)

#define DEFINE_SHMEM_GET_TYPE_API_TEMPLATE(Scope)                                          \
  template <typename T>                                                                    \
  inline __device__ void ShmemGetType##Scope(                                              \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                         \
      const application::SymmMemObjPtr source, size_t srcElmOffset, size_t nelems, int pe, \
      int qpId = 0) {                                                                      \
    constexpr size_t typeSize = sizeof(T);                                                 \
    ShmemGetMem##Scope(dest, destElmOffset * typeSize, source, srcElmOffset * typeSize,    \
                       nelems * typeSize, pe, qpId);                                       \
  }

DEFINE_SHMEM_GET_TYPE_API_TEMPLATE(Thread)
DEFINE_SHMEM_GET_TYPE_API_TEMPLATE(Warp)
DEFINE_SHMEM_GET_TYPE_API_TEMPLATE(Block)

#define DEFINE_SHMEM_GET_TYPE_API(TypeName, T, Scope)                                      \
  inline __device__ void ShmemGet##TypeName##Scope(                                        \
      const application::SymmMemObjPtr dest, size_t destElmOffset,                         \
      const application::SymmMemObjPtr source, size_t srcElmOffset, size_t nelems, int pe, \
      int qpId = 0) {                                                                      \
    ShmemGetType##Scope<T>(dest, destElmOffset, source, srcElmOffset, nelems, pe, qpId);   \
  }

DEFINE_SHMEM_GET_TYPE_API(Uint8, uint8_t, Thread)
DEFINE_SHMEM_GET_TYPE_API(Int8, int8_t, Thread)
DEFINE_SHMEM_GET_TYPE_API(Schar, signed char, Thread)
DEFINE_SHMEM_GET_TYPE_API(Uint16, uint16_t, Thread)
DEFINE_SHMEM_GET_TYPE_API(Int16, int16_t, Thread)
DEFINE_SHMEM_GET_TYPE_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_GET_TYPE_API(Int32, int32_t, Thread)
DEFINE_SHMEM_GET_TYPE_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_GET_TYPE_API(Int64, int64_t, Thread)
DEFINE_SHMEM_GET_TYPE_API(Float, float, Thread)
DEFINE_SHMEM_GET_TYPE_API(Double, double, Thread)

DEFINE_SHMEM_GET_TYPE_API(Uint8, uint8_t, Warp)
DEFINE_SHMEM_GET_TYPE_API(Int8, int8_t, Warp)
DEFINE_SHMEM_GET_TYPE_API(Schar, signed char, Warp)
DEFINE_SHMEM_GET_TYPE_API(Uint16, uint16_t, Warp)
DEFINE_SHMEM_GET_TYPE_API(Int16, int16_t, Warp)
DEFINE_SHMEM_GET_TYPE_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_GET_TYPE_API(Int32, int32_t, Warp)
DEFINE_SHMEM_GET_TYPE_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_GET_TYPE_API(Int64, int64_t, Warp)
DEFINE_SHMEM_GET_TYPE_API(Float, float, Warp)
DEFINE_SHMEM_GET_TYPE_API(Double, double, Warp)

DEFINE_SHMEM_GET_TYPE_API(Uint8, uint8_t, Block)
DEFINE_SHMEM_GET_TYPE_API(Int8, int8_t, Block)
DEFINE_SHMEM_GET_TYPE_API(Schar, signed char, Block)
DEFINE_SHMEM_GET_TYPE_API(Uint16, uint16_t, Block)
DEFINE_SHMEM_GET_TYPE_API(Int16, int16_t, Block)
DEFINE_SHMEM_GET_TYPE_API(Uint32, uint32_t, Block)
DEFINE_SHMEM_GET_TYPE_API(Int32, int32_t, Block)
DEFINE_SHMEM_GET_TYPE_API(Uint64, uint64_t, Block)
DEFINE_SHMEM_GET_TYPE_API(Int64, int64_t, Block)
DEFINE_SHMEM_GET_TYPE_API(Float, float, Block)
DEFINE_SHMEM_GET_TYPE_API(Double, double, Block)

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
DEFINE_SHMEM_PUT_MEM_NBI_SIGNAL_API_TEMPLATE(Block)

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
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API_TEMPLATE(Block)

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

DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Uint8, uint8_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Int8, int8_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Schar, signed char, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Uint16, uint16_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Int16, int16_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Uint32, uint32_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Int32, int32_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Uint64, uint64_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Int64, int64_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Float, float, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_API(Double, double, Block)

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
DEFINE_SHMEM_PUT_MEM_NBI_ADDR_API_TEMPLATE(Block)

#define DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API_TEMPLATE(Scope)                                       \
  template <typename T>                                                                          \
  inline __device__ void ShmemPutTypeNbi##Scope(T* dest, const T* source, size_t nelems, int pe, \
                                                int qpId = 0) {                                  \
    ShmemPutMemNbi##Scope(dest, source, nelems * sizeof(T), pe, qpId);                           \
  }

DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API_TEMPLATE(Thread)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API_TEMPLATE(Warp)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API_TEMPLATE(Block)

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

DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Uint8, uint8_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Int8, int8_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Schar, signed char, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Uint16, uint16_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Int16, int16_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Uint32, uint32_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Int32, int32_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Uint64, uint64_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Int64, int64_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Float, float, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_ADDR_API(Double, double, Block)

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
DEFINE_SHMEM_PUT_MEM_NBI_SIGNAL_ADDR_API_TEMPLATE(Block)

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
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API_TEMPLATE(Block)

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

DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Uint8, uint8_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Int8, int8_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Schar, signed char, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Uint16, uint16_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Int16, int16_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Uint32, uint32_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Int32, int32_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Uint64, uint64_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Int64, int64_t, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Float, float, Block)
DEFINE_SHMEM_PUT_TYPE_NBI_SIGNAL_ADDR_API(Double, double, Block)

/* ---------------------------------------------------------------------------------------------- */
/*                                        GetNbi APIs (Address-Based)                             */
/* ---------------------------------------------------------------------------------------------- */
#define DEFINE_SHMEM_GET_MEM_NBI_ADDR_API_TEMPLATE(Scope)                                      \
  inline __device__ void ShmemGetMemNbi##Scope(void* dest, const void* source, size_t bytes,   \
                                               int pe, int qpId = 0) {                         \
    DISPATCH_TRANSPORT_TYPE(ShmemGetMemNbi##Scope##Kernel, pe, dest, source, bytes, pe, qpId); \
  }

DEFINE_SHMEM_GET_MEM_NBI_ADDR_API_TEMPLATE(Thread)
DEFINE_SHMEM_GET_MEM_NBI_ADDR_API_TEMPLATE(Warp)
DEFINE_SHMEM_GET_MEM_NBI_ADDR_API_TEMPLATE(Block)

#define DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API_TEMPLATE(Scope)                                       \
  template <typename T>                                                                          \
  inline __device__ void ShmemGetTypeNbi##Scope(T* dest, const T* source, size_t nelems, int pe, \
                                                int qpId = 0) {                                  \
    ShmemGetMemNbi##Scope(dest, source, nelems * sizeof(T), pe, qpId);                           \
  }

DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API_TEMPLATE(Thread)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API_TEMPLATE(Warp)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API_TEMPLATE(Block)

#define DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(TypeName, T, Scope)                                   \
  inline __device__ void ShmemGet##TypeName##Nbi##Scope(T* dest, const T* source, size_t nelems, \
                                                        int pe, int qpId = 0) {                  \
    ShmemGetTypeNbi##Scope<T>(dest, source, nelems, pe, qpId);                                   \
  }

DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Uint8, uint8_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Int8, int8_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Schar, signed char, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Uint16, uint16_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Int16, int16_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Int32, int32_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Int64, int64_t, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Float, float, Thread)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Double, double, Thread)

DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Uint8, uint8_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Int8, int8_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Schar, signed char, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Uint16, uint16_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Int16, int16_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Int32, int32_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Int64, int64_t, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Float, float, Warp)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Double, double, Warp)

DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Uint8, uint8_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Int8, int8_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Schar, signed char, Block)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Uint16, uint16_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Int16, int16_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Uint32, uint32_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Int32, int32_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Uint64, uint64_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Int64, int64_t, Block)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Float, float, Block)
DEFINE_SHMEM_GET_TYPE_NBI_ADDR_API(Double, double, Block)

/* ---------------------------------------------------------------------------------------------- */
/*                                Blocking GET APIs (Address-Based)                               */
/* ---------------------------------------------------------------------------------------------- */
#define DEFINE_SHMEM_GET_MEM_ADDR_API_TEMPLATE(Scope)                                             \
  inline __device__ void ShmemGetMem##Scope(void* dest, const void* source, size_t bytes, int pe, \
                                            int qpId = 0) {                                       \
    ShmemGetMemNbi##Scope(dest, source, bytes, pe, qpId);                                         \
    ShmemQuietThread(pe, qpId);                                                                   \
  }

DEFINE_SHMEM_GET_MEM_ADDR_API_TEMPLATE(Thread)
DEFINE_SHMEM_GET_MEM_ADDR_API_TEMPLATE(Warp)
DEFINE_SHMEM_GET_MEM_ADDR_API_TEMPLATE(Block)

#define DEFINE_SHMEM_GET_TYPE_ADDR_API_TEMPLATE(Scope)                                        \
  template <typename T>                                                                       \
  inline __device__ void ShmemGetType##Scope(T* dest, const T* source, size_t nelems, int pe, \
                                             int qpId = 0) {                                  \
    ShmemGetMem##Scope(dest, source, nelems * sizeof(T), pe, qpId);                           \
  }

DEFINE_SHMEM_GET_TYPE_ADDR_API_TEMPLATE(Thread)
DEFINE_SHMEM_GET_TYPE_ADDR_API_TEMPLATE(Warp)
DEFINE_SHMEM_GET_TYPE_ADDR_API_TEMPLATE(Block)

#define DEFINE_SHMEM_GET_TYPE_ADDR_API(TypeName, T, Scope)                                  \
  inline __device__ void ShmemGet##TypeName##Scope(T* dest, const T* source, size_t nelems, \
                                                   int pe, int qpId = 0) {                  \
    ShmemGetType##Scope<T>(dest, source, nelems, pe, qpId);                                 \
  }

DEFINE_SHMEM_GET_TYPE_ADDR_API(Uint8, uint8_t, Thread)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Int8, int8_t, Thread)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Schar, signed char, Thread)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Uint16, uint16_t, Thread)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Int16, int16_t, Thread)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Uint32, uint32_t, Thread)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Int32, int32_t, Thread)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Uint64, uint64_t, Thread)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Int64, int64_t, Thread)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Float, float, Thread)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Double, double, Thread)

DEFINE_SHMEM_GET_TYPE_ADDR_API(Uint8, uint8_t, Warp)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Int8, int8_t, Warp)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Schar, signed char, Warp)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Uint16, uint16_t, Warp)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Int16, int16_t, Warp)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Uint32, uint32_t, Warp)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Int32, int32_t, Warp)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Uint64, uint64_t, Warp)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Int64, int64_t, Warp)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Float, float, Warp)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Double, double, Warp)

DEFINE_SHMEM_GET_TYPE_ADDR_API(Uint8, uint8_t, Block)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Int8, int8_t, Block)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Schar, signed char, Block)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Uint16, uint16_t, Block)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Int16, int16_t, Block)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Uint32, uint32_t, Block)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Int32, int32_t, Block)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Uint64, uint64_t, Block)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Int64, int64_t, Block)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Float, float, Block)
DEFINE_SHMEM_GET_TYPE_ADDR_API(Double, double, Block)

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

/* ---------------------------------------------------------------------------------------------- */
/*                                       Device Barrier APIs                                      */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ void ShmemInternalBarrierThread() {
  GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
  uint64_t* pSync = globalGpuStates->internalSyncPtr;
  int pe = globalGpuStates->rank;
  constexpr int coordinatorPe = 0;
  int n_pes = globalGpuStates->worldSize;

  // Use uint64_t for flag value and sync operations
  constexpr uint64_t syncValue = 1;

  if (pe == coordinatorPe) {
    // Wait for all non-leader PEs to signal arrival
    for (int i = 1; i < n_pes; i++) {
      ShmemUint64WaitUntilEquals(&pSync[i], syncValue);
      pSync[i] = 0;
    }
    __threadfence_system();

    // Notify all other PEs that barrier is complete
    for (int i = 1, j = coordinatorPe + 1; i < n_pes; ++i, ++j) {
      pSync[0] = syncValue;
      ShmemPutUint64ImmNbiThread(&pSync[0], syncValue, j, 0);
      __threadfence_system();
    }
    pSync[0] = 0;
  } else {
    // Non-leader PEs signal arrival to leader
    size_t pe_offset = (pe - coordinatorPe);
    pSync[pe_offset] = syncValue;
    ShmemPutUint64ImmNbiThread(&pSync[pe_offset], syncValue, coordinatorPe, 0);
    __threadfence_system();

    // Wait for leader to signal completion
    ShmemUint64WaitUntilEquals(&pSync[0], syncValue);
    pSync[0] = 0;
    pSync[pe_offset] = 0;
    __threadfence_system();
  }
}

inline __device__ void ShmemInternalBarrierBlock() {
  if (threadIdx.x == 0) {
    // Thread 0 performs the inter-PE barrier
    GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
    uint64_t* pSync = globalGpuStates->internalSyncPtr;
    int pe = globalGpuStates->rank;
    constexpr int coordinatorPe = 0;
    int n_pes = globalGpuStates->worldSize;

    constexpr uint64_t syncValue = 1;

    if (pe == coordinatorPe) {
      for (int i = 1; i < n_pes; i++) {
        ShmemUint64WaitUntilEquals(&pSync[i], syncValue);
        pSync[i] = 0;
      }
      __threadfence_system();

      // Notify all other PEs that barrier is complete
      for (int i = 1, j = coordinatorPe + 1; i < n_pes; ++i, ++j) {
        pSync[0] = syncValue;
        ShmemPutUint64ImmNbiThread(&pSync[0], syncValue, j, 0);
        __threadfence_system();
      }
      pSync[0] = 0;
    } else {
      // Non-leader PEs: signal arrival to leader
      size_t pe_offset = (pe - coordinatorPe);
      pSync[pe_offset] = syncValue;
      ShmemPutUint64ImmNbiThread(&pSync[pe_offset], syncValue, coordinatorPe, 0);
      __threadfence_system();

      // Wait for leader to signal completion
      ShmemUint64WaitUntilEquals(&pSync[0], syncValue);
      pSync[0] = 0;
      pSync[pe_offset] = 0;
      __threadfence_system();
    }
  }

  // All threads in the block wait here until thread 0 completes the inter-PE barrier
  __syncthreads();
}

// Dissemination barrier — a topology-cheaper drop-in for the PE0-coordinator
// funnel ShmemInternalBarrierBlock. The funnel serializes all n PEs through PE0
// (PE0 does n-1 sequential inbound waits + n-1 sequential outbound puts, several
// crossing the node boundary over RDMA). The dissemination algorithm replaces
// that O(n) serial critical path with ceil(log2 n) parallel rounds (every PE
// does 1 put + 1 wait per round): for n=8 that is 3 rounds. Same global scope
// (all PEs), same put/wait primitives, so it preserves the quiet-drain +
// all-PEs-arrived semantics the prepare relies on.
//
// Correctness: monotonic per-PE generation counter (never reset) disambiguates
// successive barrier calls, so no per-op flag clear (and no clear-vs-signal
// race) is needed. Each call: gen = ++localGen. Round r (dist = 1<<r): signal
// partner (pe+dist)%n with gen, then wait until my slot[r] >= gen (written by
// (pe-dist+n)%n). Waiting each round before advancing gives the standard
// dissemination guarantee that no PE exits until all entered; the ≤1-barrier
// cross-PE skew that property permits is handled by the monotonic gen (a peer
// racing one call ahead only writes a LARGER value, which still satisfies the
// >= gen wait). Uses a DISJOINT pSync slot region [DISSEM_BASE, DISSEM_BASE+R)
// + a PE-local gen slot, so it never aliases the funnel's slots [0,n) even if
// the two barrier flavors interleave within one op.
inline __device__ void ShmemInternalBarrierDissemBlock() {
  if (threadIdx.x == 0) {
    GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
    uint64_t* pSync = globalGpuStates->internalSyncPtr;
    int pe = globalGpuStates->rank;
    int n_pes = globalGpuStates->worldSize;

    // Disjoint from the funnel's slots [0, n_pes). MORI_INTERNAL_SYNC_SIZE is
    // 128 uint64 slots; DISSEM_BASE=64 leaves room for up to ~63 rounds (far
    // beyond any realistic PE count) and a PE-local generation slot at 127.
    constexpr int DISSEM_BASE = 64;
    constexpr int DISSEM_GEN_SLOT = 127;

    uint64_t gen = pSync[DISSEM_GEN_SLOT] + 1;
    pSync[DISSEM_GEN_SLOT] = gen;
    __threadfence();

    int r = 0;
    for (int dist = 1; dist < n_pes; dist <<= 1, ++r) {
      int partner = pe + dist;
      if (partner >= n_pes) partner -= n_pes;
      // Signal partner's round-r slot with this call's generation.
      ShmemPutUint64ImmNbiThread(&pSync[DISSEM_BASE + r], gen, partner, 0);
      __threadfence_system();
      // Wait for the round-r signal from (pe - dist + n_pes) % n_pes.
      ShmemUint64WaitUntilGreaterThan(&pSync[DISSEM_BASE + r], gen - 1);
    }
    __threadfence_system();
  }
  __syncthreads();
}

inline __device__ void ShmemBarrierAllDissemBlock() {
  ShmemQuietThread();
  ShmemInternalBarrierDissemBlock();
}

// HIERARCHICAL 2-LEVEL barrier — a topology-AWARE drop-in for the flat
// funnel/dissem barriers. Both of those cross the RDMA node boundary many times:
// the funnel serializes ALL n-1 arrivals + n-1 releases through PE0 (~n/rpn of
// EACH crossing the boundary SEQUENTIALLY = the small-size rendezvous-latency
// pole); dissem parallelizes into log2(n) rounds but STILL issues a cross-node
// RDMA put for every PE in the boundary-crossing round. This barrier exploits the
// XGMI(intra-node)/RDMA(inter-node) asymmetry and crosses the boundary EXACTLY
// ONCE each way (only the per-node coordinators exchange over RDMA):
// 1. intra-node gather : each local PE signals its node coordinator over XGMI.
// 2. inter-node exchange: coordinators all-to-all over RDMA (numNodes-1 puts).
// 3. intra-node release: coordinator releases its local PEs over XGMI.
// For w16 (n=16, ranksPerNode=8, 2 nodes) = 2 cross-node RDMA ops total vs the
// funnel's ~16 serial. Same GLOBAL all-PE arrived-before-any-exits semantics.
// Monotonic per-PE generation counter (never reset) => graph-replay-safe, no
// clear-vs-signal race (same discipline as dissem). Uses a DISJOINT pSync slot
// region so it never aliases the funnel [0,n) or dissem [64..~70]+127 flavors.
inline __device__ void ShmemInternalBarrierHierBlock(int ranksPerNode) {
  if (threadIdx.x == 0) {
    GpuStates* globalGpuStates = GetGlobalGpuStatesPtr();
    uint64_t* pSync = globalGpuStates->internalSyncPtr;
    int pe = globalGpuStates->rank;
    int n_pes = globalGpuStates->worldSize;
    // Degenerate ranksPerNode => behave as a single-node (all-local) barrier.
    if (ranksPerNode <= 0 || ranksPerNode > n_pes) ranksPerNode = n_pes;

    // Disjoint slot layout in the 128-uint64 internalSync region:
    // HIER_PEER_BASE[96 .. 96+numNodes) coordinator inter-node inbox (by src node)
    // HIER_LOCAL_BASE[112 .. 112+rpn) local PE -> coordinator inbox (by localIdx)
    // HIER_REL_SLOT [120] coordinator -> local release
    // HIER_GEN_SLOT [126] PE-local monotonic generation
    constexpr int HIER_PEER_BASE = 96;
    constexpr int HIER_LOCAL_BASE = 112;
    constexpr int HIER_REL_SLOT = 120;
    constexpr int HIER_GEN_SLOT = 126;
    // Capacity of each disjoint region carved out of the 128-uint64 internalSync
    // area. The PEER inbox spans [HIER_PEER_BASE, HIER_LOCAL_BASE) and the LOCAL
    // inbox spans [HIER_LOCAL_BASE, HIER_REL_SLOT); exceeding either would alias
    // an adjacent region and corrupt the barrier.
    constexpr int HIER_MAX_NODES = HIER_LOCAL_BASE - HIER_PEER_BASE;          // 16
    constexpr int HIER_MAX_RANKS_PER_NODE = HIER_REL_SLOT - HIER_LOCAL_BASE;  // 8

    int numNodes = n_pes / ranksPerNode;
    // Fail closed on topologies this fixed slot layout cannot represent:
    //  - non-divisible world => integer numNodes drops the remainder node and its
    //    PEs are never released (hang);
    //  - ranksPerNode slots overflow [HIER_LOCAL_BASE, HIER_REL_SLOT);
    //  - numNodes slots overflow [HIER_PEER_BASE, HIER_LOCAL_BASE).
    // Shipped w16 (rpn=8, numNodes=2, 16%8==0) is within all bounds.
    assert(n_pes % ranksPerNode == 0 &&
           "ShmemInternalBarrierHierBlock: worldSize not divisible by ranksPerNode");
    assert(ranksPerNode <= HIER_MAX_RANKS_PER_NODE &&
           "ShmemInternalBarrierHierBlock: ranksPerNode exceeds internalSync slot capacity");
    assert(numNodes <= HIER_MAX_NODES &&
           "ShmemInternalBarrierHierBlock: numNodes exceeds internalSync slot capacity");

    uint64_t gen = pSync[HIER_GEN_SLOT] + 1;
    pSync[HIER_GEN_SLOT] = gen;
    __threadfence();

    int node = pe / ranksPerNode;
    int localIdx = pe % ranksPerNode;
    int coord = node * ranksPerNode;

    if (localIdx != 0) {
      // Non-coordinator: signal my node coordinator (XGMI), wait for release.
      ShmemPutUint64ImmNbiThread(&pSync[HIER_LOCAL_BASE + localIdx], gen, coord, 0);
      __threadfence_system();
      ShmemUint64WaitUntilGreaterThan(&pSync[HIER_REL_SLOT], gen - 1);
    } else {
      // Coordinator.
      // 1. wait for all local PEs to arrive (XGMI reads of my own inbox).
      for (int i = 1; i < ranksPerNode; ++i)
        ShmemUint64WaitUntilGreaterThan(&pSync[HIER_LOCAL_BASE + i], gen - 1);
      __threadfence_system();
      // 2. inter-node all-to-all among coordinators (the ONLY cross-node RDMA hop).
      for (int nn = 0; nn < numNodes; ++nn) {
        if (nn == node) continue;
        ShmemPutUint64ImmNbiThread(&pSync[HIER_PEER_BASE + node], gen, nn * ranksPerNode, 0);
      }
      __threadfence_system();
      for (int nn = 0; nn < numNodes; ++nn) {
        if (nn == node) continue;
        ShmemUint64WaitUntilGreaterThan(&pSync[HIER_PEER_BASE + nn], gen - 1);
      }
      __threadfence_system();
      // 3. release my local PEs (XGMI). All n PEs have now ARRIVED.
      for (int i = 1; i < ranksPerNode; ++i)
        ShmemPutUint64ImmNbiThread(&pSync[HIER_REL_SLOT], gen, coord + i, 0);
      __threadfence_system();
    }
  }
  // All threads in the block wait until thread 0 completes the inter-PE barrier.
  __syncthreads();
}

inline __device__ void ShmemBarrierAllHierBlock(int ranksPerNode) {
  ShmemQuietThread();
  ShmemInternalBarrierHierBlock(ranksPerNode);
}

inline __device__ void ShmemBarrierAllThread() {
  ShmemQuietThread();
  ShmemInternalBarrierThread();
}

inline __device__ void ShmemBarrierAllBlock() {
  ShmemQuietThread();
  ShmemInternalBarrierBlock();
}

}  // namespace shmem
}  // namespace mori
