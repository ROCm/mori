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

#include <type_traits>

#include "mori/application/application.hpp"
#include "mori/core/core.hpp"
#include "mori/shmem/internal.hpp"
#include "mori/shmem/shmem_api.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                                         Point-to-Point                                         */
/* ---------------------------------------------------------------------------------------------- */
template <>
inline __device__ void ShmemPutMemNbiThreadKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe, int qpId) {
  uint8_t* srcPtr =
      reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset);
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);
  core::ThreadCopy<uint8_t>(destPtr, srcPtr, bytes);
}

template <>
inline __device__ void ShmemPutMemNbiWarpKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes, int pe, int qpId) {
  uint8_t* srcPtr =
      reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset);
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);
  core::WarpCopy<uint8_t>(destPtr, srcPtr, bytes);
}

template <>
inline __device__ void ShmemPutSizeImmNbiThreadKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe,
    int qpId) {
  uint8_t* srcPtr = reinterpret_cast<uint8_t*>(val);
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);
  switch (bytes) {
    case 1:
      core::AtomicStoreRelaxedSystem(destPtr, reinterpret_cast<uint8_t*>(val)[0]);
      break;
    case 2:
      core::AtomicStoreRelaxedSystem(reinterpret_cast<uint16_t*>(destPtr),
                                     reinterpret_cast<uint16_t*>(val)[0]);
      break;
    case 4:
      core::AtomicStoreRelaxedSystem(reinterpret_cast<uint32_t*>(destPtr),
                                     reinterpret_cast<uint32_t*>(val)[0]);
      break;
    case 8:
      core::AtomicStoreRelaxedSystem(reinterpret_cast<uint64_t*>(destPtr),
                                     reinterpret_cast<uint64_t*>(val)[0]);
      break;
    case 16:
      reinterpret_cast<uint4*>(destPtr)[0] = reinterpret_cast<uint4*>(val)[0];
      break;
    default:
      MORI_PRINTF(
          "Size must be one of [1,2,4,8,16] bytes, got %lu, for arbitrary size, use ShmemPutMemNbi "
          "APIs",
          bytes);
      assert(false);
  }
}

template <>
inline __device__ void ShmemPutSizeImmNbiWarpKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe,
    int qpId) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0)
    ShmemPutSizeImmNbiThreadKernel<application::TransportType::P2P>(dest, destOffset, val, bytes,
                                                                    pe);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    PutMemNbi with Signal                                       */
/* ---------------------------------------------------------------------------------------------- */
template <>
inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::P2P, true>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  if (bytes == 0) return;

  // Execute put operation
  uint8_t* srcPtr =
      reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset);
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);
  core::ThreadCopy<uint8_t>(destPtr, srcPtr, bytes);

  // Execute signal operation (only once for onlyOneSignal=true)
  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader = (my_logical_lane_id == num_active_lanes - 1);

  if (is_leader) {
    uint64_t* signalPtr = reinterpret_cast<uint64_t*>(signalDest->peerPtrs[pe] + signalDestOffset);
    if (signalOp == core::atomicType::AMO_SET || signalOp == core::atomicType::AMO_SIGNAL_SET) {
      core::AtomicStoreSeqCstSystem(signalPtr, signalValue);
    } else if (signalOp == core::atomicType::AMO_ADD ||
               signalOp == core::atomicType::AMO_SIGNAL_ADD) {
      core::AtomicAddSeqCstSystem(signalPtr, signalValue);
    } else {
      assert(false && "Unsupported signal operation");
    }
  }
}

template <>
inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::P2P, false>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  if (bytes == 0) return;

  // Execute put operation
  uint8_t* srcPtr =
      reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset);
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);
  core::ThreadCopy<uint8_t>(destPtr, srcPtr, bytes);

  // Execute signal operation (every thread signals for onlyOneSignal=false)
  uint64_t* signalPtr = reinterpret_cast<uint64_t*>(signalDest->peerPtrs[pe] + signalDestOffset);
  if (signalOp == core::atomicType::AMO_SET || signalOp == core::atomicType::AMO_SIGNAL_SET) {
    core::AtomicStoreSeqCstSystem(signalPtr, signalValue);
  } else if (signalOp == core::atomicType::AMO_ADD ||
             signalOp == core::atomicType::AMO_SIGNAL_ADD) {
    core::AtomicAddSeqCstSystem(signalPtr, signalValue);
  } else {
    assert(false && "Unsupported signal operation");
  }
}

template <>
inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::P2P, true>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  if (bytes == 0) return;

  // Execute put operation (all lanes participate)
  uint8_t* srcPtr =
      reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset);
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);
  core::WarpCopy<uint8_t>(destPtr, srcPtr, bytes);

  // Execute signal operation (only lane 0 for onlyOneSignal=true)
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    uint64_t* signalPtr = reinterpret_cast<uint64_t*>(signalDest->peerPtrs[pe] + signalDestOffset);
    if (signalOp == core::atomicType::AMO_SET || signalOp == core::atomicType::AMO_SIGNAL_SET) {
      core::AtomicStoreSeqCstSystem(signalPtr, signalValue);
    } else if (signalOp == core::atomicType::AMO_ADD ||
               signalOp == core::atomicType::AMO_SIGNAL_ADD) {
      core::AtomicAddSeqCstSystem(signalPtr, signalValue);
    } else {
      assert(false && "Unsupported signal operation");
    }
  }
}

template <>
inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::P2P, false>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId) {
  if (bytes == 0) return;

  // Execute put operation (all lanes participate)
  uint8_t* srcPtr =
      reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(source->localPtr) + sourceOffset);
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);
  core::WarpCopy<uint8_t>(destPtr, srcPtr, bytes);

  // Execute signal operation (lane 0 only, but signals once per active lane)
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    uint64_t activemask = core::GetActiveLaneMask();
    uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);

    uint64_t* signalPtr = reinterpret_cast<uint64_t*>(signalDest->peerPtrs[pe] + signalDestOffset);
    if (signalOp == core::atomicType::AMO_SET || signalOp == core::atomicType::AMO_SIGNAL_SET) {
      core::AtomicStoreSeqCstSystem(signalPtr, signalValue);
    } else if (signalOp == core::atomicType::AMO_ADD ||
               signalOp == core::atomicType::AMO_SIGNAL_ADD) {
      core::AtomicAddSeqCstSystem(signalPtr, signalValue * num_active_lanes);
    } else {
      assert(false && "Unsupported signal operation");
    }
  }
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes,
    core::atomicType amoType, int pe, int qpId) {
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);
  switch (bytes) {
    case 4: {
      int argVal = *reinterpret_cast<int*>(val);
      int* ptr4 = reinterpret_cast<int*>(destPtr);
      auto casLoop = [=] __device__(core::atomicType atype, int operand) {
        while (true) {
          int oldVal = core::AtomicLoadSeqCst(ptr4);
          int newVal = oldVal;
          switch (atype) {
            case core::AMO_INC:
              newVal = oldVal + 1;
              break;
            case core::AMO_ADD:
            case core::AMO_SIGNAL_ADD:
              newVal = oldVal + operand;
              break;
            case core::AMO_AND:
              newVal = oldVal & operand;
              break;
            case core::AMO_OR:
              newVal = oldVal | operand;
              break;
            case core::AMO_XOR:
              newVal = oldVal ^ operand;
              break;
            default:
              break;
          }
          int expected = oldVal;
          int prev = core::AtomicCompareExchangeSystem(ptr4, &expected, newVal);
          if (prev == oldVal) break;
        }
      };
      switch (amoType) {
        case core::AMO_SET:
        case core::AMO_SIGNAL_SET:
          core::AtomicStoreSeqCstSystem(ptr4, argVal);
          break;

        case core::AMO_INC:
        case core::AMO_ADD:
        case core::AMO_AND:
        case core::AMO_OR:
        case core::AMO_XOR:
        case core::AMO_SIGNAL_ADD:
          casLoop(amoType, argVal);
          break;

        default:
          MORI_PRINTF("Error: Unsupported 4-byte atomicType (%d) in NonFetchThreadKernel.\n",
                      amoType);
          break;
      }
      break;
    }
    case 8: {
      long long argVal = *reinterpret_cast<long long*>(val);
      long long* ptr8 = reinterpret_cast<long long*>(destPtr);

      auto casLoop64 = [=] __device__(core::atomicType atype, long long operand) {
        while (true) {
          long long oldVal = core::AtomicLoadSeqCst(ptr8);
          long long newVal = oldVal;
          switch (atype) {
            case core::AMO_INC:
              newVal = oldVal + 1;
              break;
            case core::AMO_ADD:
            case core::AMO_SIGNAL_ADD:
              newVal = oldVal + operand;
              break;
            case core::AMO_AND:
              newVal = oldVal & operand;
              break;
            case core::AMO_OR:
              newVal = oldVal | operand;
              break;
            case core::AMO_XOR:
              newVal = oldVal ^ operand;
              break;
            default:
              break;
          }
          long long expected = oldVal;
          long long prev = core::AtomicCompareExchangeSystem(ptr8, &expected, newVal);
          if (prev == oldVal) break;
        }
      };

      switch (amoType) {
        case core::AMO_SET:
        case core::AMO_SIGNAL_SET:
          core::AtomicStoreSeqCstSystem(ptr8, argVal);
          break;

        case core::AMO_INC:
        case core::AMO_ADD:
        case core::AMO_AND:
        case core::AMO_OR:
        case core::AMO_XOR:
        case core::AMO_SIGNAL_ADD:
          casLoop64(amoType, argVal);
          break;
        default:
          MORI_PRINTF("Error: Unsupported 8-byte atomicType (%d) in NonFetchThreadKernel.\n",
                      amoType);
          break;
      }
      break;
    }
    default:
      MORI_PRINTF("Error: Unsupported data size (%zu bytes) in NonFetchThreadKernel.\n", bytes);
      break;
  }
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes,
    core::atomicType amoType, int pe, int qpId) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::P2P>(dest, destOffset, val,
                                                                         bytes, amoType, pe);
  }
}

template <typename T>
inline __device__ T ShmemAtomicTypeFetchThreadKernelImplP2P(const application::SymmMemObjPtr dest,
                                                            size_t destOffset, void* val,
                                                            void* compare, size_t bytes,
                                                            core::atomicType amoType, int pe,
                                                            int qpId) {
  T* destPtr = reinterpret_cast<T*>(dest->peerPtrs[pe] + destOffset);
  T* fetchResPtr = reinterpret_cast<T*>(val);
  T cmpVal = (compare != nullptr) ? *reinterpret_cast<T*>(compare) : T{};

  auto casLoop = [=] __device__(T * addr, core::atomicType op, T operand, T cmpVal, T * oldResult) {
    T oldVal = core::AtomicLoadSeqCstSystem(addr);
    while (true) {
      T newVal = oldVal;
      switch (op) {
        case core::AMO_FETCH_INC:
          newVal = oldVal + T{1};
          break;
        case core::AMO_FETCH_ADD:
          newVal = oldVal + operand;
          break;
        case core::AMO_FETCH_AND:
          if constexpr (std::is_integral_v<T>) {
            newVal = oldVal & operand;
          }
          break;
        case core::AMO_FETCH_OR:
          if constexpr (std::is_integral_v<T>) {
            newVal = oldVal | operand;
          }
          break;
        case core::AMO_FETCH_XOR:
          if constexpr (std::is_integral_v<T>) {
            newVal = oldVal ^ operand;
          }
          break;
        case core::AMO_SWAP:
          newVal = operand;
          break;
        case core::AMO_COMPARE_SWAP:
          if (oldVal == cmpVal) {
            newVal = operand;
          } else {
            newVal = oldVal;
          }
          break;
        default:
          break;
      }

      T expected = oldVal;
      T prev = core::AtomicCompareExchangeSystem(addr, &expected, newVal);
      if (prev == oldVal) {
        *oldResult = oldVal;
        break;
      }
      oldVal = prev;
    }
    return oldVal;
  };

  T* valPtr = reinterpret_cast<T*>(val);
  T operand = *valPtr;

  switch (amoType) {
    case core::AMO_FETCH_INC:
    case core::AMO_FETCH_ADD:
    case core::AMO_FETCH_AND:
    case core::AMO_FETCH_OR:
    case core::AMO_FETCH_XOR:
    case core::AMO_SWAP:
    case core::AMO_COMPARE_SWAP: {
      T result = casLoop(destPtr, amoType, operand, cmpVal, fetchResPtr);
      return result;
    } break;
    default:
      if constexpr (sizeof(T) == 4) {
        MORI_PRINTF("Error: Unsupported 4-byte atomicType (%d) in TypeFetchThreadKernel.\n",
                    amoType);
      } else if constexpr (sizeof(T) == 8) {
        MORI_PRINTF("Error: Unsupported 8-byte atomicType (%d) in TypeFetchThreadKernel.\n",
                    amoType);
      }
      break;
  }
  return T{};
}

#define DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL_P2P(TypeName, T)                        \
  template <>                                                                                \
  inline __device__ T ShmemAtomicTypeFetchThreadKernel<application::TransportType::P2P, T>(  \
      const application::SymmMemObjPtr dest, size_t destOffset, void* val, void* compare,    \
      size_t bytes, core::atomicType amoType, int pe, int qpId) {                            \
    return ShmemAtomicTypeFetchThreadKernelImplP2P<T>(dest, destOffset, val, compare, bytes, \
                                                      amoType, pe, qpId);                    \
  }

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL_P2P(Uint32, uint32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL_P2P(Uint64, uint64_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL_P2P(Int32, int32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL_P2P(Int64, int64_t)

#define DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL_P2P(TypeName, T)                          \
  template <>                                                                                \
  inline __device__ T ShmemAtomicTypeFetchWarpKernel<application::TransportType::P2P, T>(    \
      const application::SymmMemObjPtr dest, size_t destOffset, void* val, void* compare,    \
      size_t bytes, core::atomicType amoType, int pe, int qpId) {                            \
    return ShmemAtomicTypeFetchThreadKernelImplP2P<T>(dest, destOffset, val, compare, bytes, \
                                                      amoType, pe, qpId);                    \
  }

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL_P2P(Uint32, uint32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL_P2P(Uint64, uint64_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL_P2P(Int32, int32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL_P2P(Int64, int64_t)

/* ---------------------------------------------------------------------------------------------- */
/*                            Pure Address-Based Point-to-Point (New)                             */
/* ---------------------------------------------------------------------------------------------- */

// New pure address-based PutMemNbi for P2P transport
template <>
inline __device__ void ShmemPutMemNbiThreadKernel<application::TransportType::P2P>(
    const void* dest, const void* source, size_t bytes, int pe, int qpId) {
  RemoteAddrInfo remoteInfo = ShmemAddrToRemoteAddr(dest, pe);
  uint8_t* srcPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(source));
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(remoteInfo.raddr);
  core::ThreadCopy<uint8_t>(destPtr, srcPtr, bytes);
}

template <>
inline __device__ void ShmemPutMemNbiWarpKernel<application::TransportType::P2P>(const void* dest,
                                                                                 const void* source,
                                                                                 size_t bytes,
                                                                                 int pe, int qpId) {
  RemoteAddrInfo remoteInfo = ShmemAddrToRemoteAddr(dest, pe);
  uint8_t* srcPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(source));
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(remoteInfo.raddr);
  core::WarpCopy<uint8_t>(destPtr, srcPtr, bytes);
}

// New pure address-based PutSizeImmNbi for P2P transport
template <>
inline __device__ void ShmemPutSizeImmNbiThreadKernel<application::TransportType::P2P>(
    const void* dest, void* val, size_t bytes, int pe, int qpId) {
  RemoteAddrInfo remoteInfo = ShmemAddrToRemoteAddr(dest, pe);
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(remoteInfo.raddr);
  switch (bytes) {
    case 1:
      core::AtomicStoreRelaxedSystem(destPtr, reinterpret_cast<uint8_t*>(val)[0]);
      break;
    case 2:
      core::AtomicStoreRelaxedSystem(reinterpret_cast<uint16_t*>(destPtr),
                                     reinterpret_cast<uint16_t*>(val)[0]);
      break;
    case 4:
      core::AtomicStoreRelaxedSystem(reinterpret_cast<uint32_t*>(destPtr),
                                     reinterpret_cast<uint32_t*>(val)[0]);
      break;
    case 8:
      core::AtomicStoreRelaxedSystem(reinterpret_cast<uint64_t*>(destPtr),
                                     reinterpret_cast<uint64_t*>(val)[0]);
      break;
    case 16:
      reinterpret_cast<uint4*>(destPtr)[0] = reinterpret_cast<uint4*>(val)[0];
      break;
    default:
      printf("Size must be one of [1,2,4,8,16] bytes, got %lu\n", bytes);
      assert(false);
  }
}

template <>
inline __device__ void ShmemPutSizeImmNbiWarpKernel<application::TransportType::P2P>(
    const void* dest, void* val, size_t bytes, int pe, int qpId) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0)
    ShmemPutSizeImmNbiThreadKernel<application::TransportType::P2P>(dest, val, bytes, pe, qpId);
}

template <>
inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::P2P, true>(
    const void* dest, const void* source, size_t bytes, const void* signalDest,
    uint64_t signalValue, core::atomicType signalOp, int pe, int qpId) {
  if (bytes == 0) return;

  // Translate dest address to remote address
  RemoteAddrInfo destInfo = ShmemAddrToRemoteAddr(dest, pe);
  // Translate signal dest address to remote address
  RemoteAddrInfo signalDestInfo = ShmemAddrToRemoteAddr(signalDest, pe);

  // Execute put operation
  uint8_t* srcPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(source));
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(destInfo.raddr);
  core::ThreadCopy<uint8_t>(destPtr, srcPtr, bytes);

  // Execute signal operation (only once for onlyOneSignal=true)
  uint64_t activemask = core::GetActiveLaneMask();
  uint8_t num_active_lanes = core::GetActiveLaneCount(activemask);
  uint8_t my_logical_lane_id = core::GetActiveLaneNum(activemask);
  bool is_leader = (my_logical_lane_id == num_active_lanes - 1);

  if (is_leader) {
    uint64_t* signalPtr = reinterpret_cast<uint64_t*>(signalDestInfo.raddr);
    if (signalOp == core::atomicType::AMO_SET || signalOp == core::atomicType::AMO_SIGNAL_SET) {
      core::AtomicStoreSeqCstSystem(signalPtr, signalValue);
    } else if (signalOp == core::atomicType::AMO_ADD ||
               signalOp == core::atomicType::AMO_SIGNAL_ADD) {
      core::AtomicAddSeqCstSystem(signalPtr, signalValue);
    } else {
      assert(false && "Unsupported signal operation");
    }
  }
}

template <>
inline __device__ void ShmemPutMemNbiSignalThreadKernel<application::TransportType::P2P, false>(
    const void* dest, const void* source, size_t bytes, const void* signalDest,
    uint64_t signalValue, core::atomicType signalOp, int pe, int qpId) {
  if (bytes == 0) return;

  // Translate dest address to remote address
  RemoteAddrInfo destInfo = ShmemAddrToRemoteAddr(dest, pe);
  // Translate signal dest address to remote address
  RemoteAddrInfo signalDestInfo = ShmemAddrToRemoteAddr(signalDest, pe);

  // Execute put operation
  uint8_t* srcPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(source));
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(destInfo.raddr);
  core::ThreadCopy<uint8_t>(destPtr, srcPtr, bytes);

  // Execute signal operation (every thread signals for onlyOneSignal=false)
  uint64_t* signalPtr = reinterpret_cast<uint64_t*>(signalDestInfo.raddr);
  if (signalOp == core::atomicType::AMO_SET || signalOp == core::atomicType::AMO_SIGNAL_SET) {
    core::AtomicStoreSeqCstSystem(signalPtr, signalValue);
  } else if (signalOp == core::atomicType::AMO_ADD ||
             signalOp == core::atomicType::AMO_SIGNAL_ADD) {
    core::AtomicAddSeqCstSystem(signalPtr, signalValue);
  } else {
    assert(false && "Unsupported signal operation");
  }
}

template <>
inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::P2P, true>(
    const void* dest, const void* source, size_t bytes, const void* signalDest,
    uint64_t signalValue, core::atomicType signalOp, int pe, int qpId) {
  if (bytes == 0) return;

  // Translate dest address to remote address
  RemoteAddrInfo destInfo = ShmemAddrToRemoteAddr(dest, pe);

  // Translate signal dest address to remote address
  RemoteAddrInfo signalDestInfo = ShmemAddrToRemoteAddr(signalDest, pe);

  // Execute put operation (all lanes participate)
  uint8_t* srcPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(source));
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(destInfo.raddr);
  core::WarpCopy<uint8_t>(destPtr, srcPtr, bytes);

  // Execute signal operation (only lane 0 for onlyOneSignal=true)
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    uint64_t* signalPtr = reinterpret_cast<uint64_t*>(signalDestInfo.raddr);
    if (signalOp == core::atomicType::AMO_SET || signalOp == core::atomicType::AMO_SIGNAL_SET) {
      core::AtomicStoreSeqCstSystem(signalPtr, signalValue);
    } else if (signalOp == core::atomicType::AMO_ADD ||
               signalOp == core::atomicType::AMO_SIGNAL_ADD) {
      core::AtomicAddSeqCstSystem(signalPtr, signalValue);
    } else {
      assert(false && "Unsupported signal operation");
    }
  }
}

template <>
inline __device__ void ShmemPutMemNbiSignalWarpKernel<application::TransportType::P2P, false>(
    const void* dest, const void* source, size_t bytes, const void* signalDest,
    uint64_t signalValue, core::atomicType signalOp, int pe, int qpId) {
  if (bytes == 0) return;

  // Translate dest address to remote address
  RemoteAddrInfo destInfo = ShmemAddrToRemoteAddr(dest, pe);

  // Translate signal dest address to remote address
  RemoteAddrInfo signalDestInfo = ShmemAddrToRemoteAddr(signalDest, pe);

  // Execute put operation (all lanes participate)
  uint8_t* srcPtr = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(source));
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(destInfo.raddr);
  core::WarpCopy<uint8_t>(destPtr, srcPtr, bytes);

  // Execute signal operation (all lanes signal for onlyOneSignal=false)
  uint64_t* signalPtr = reinterpret_cast<uint64_t*>(signalDestInfo.raddr);
  if (signalOp == core::atomicType::AMO_SET || signalOp == core::atomicType::AMO_SIGNAL_SET) {
    core::AtomicStoreSeqCstSystem(signalPtr, signalValue);
  } else if (signalOp == core::atomicType::AMO_ADD ||
             signalOp == core::atomicType::AMO_SIGNAL_ADD) {
    core::AtomicAddSeqCstSystem(signalPtr, signalValue);
  } else {
    assert(false && "Unsupported signal operation");
  }
}

// New pure address-based Atomic operations for P2P transport
template <>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::P2P>(
    const void* dest, void* val, size_t bytes, core::atomicType amoType, int pe, int qpId) {
  RemoteAddrInfo remoteInfo = ShmemAddrToRemoteAddr(dest, pe);

  uint8_t* destPtr = reinterpret_cast<uint8_t*>(remoteInfo.raddr);
  switch (bytes) {
    case 4: {
      int argVal = *reinterpret_cast<int*>(val);
      int* ptr4 = reinterpret_cast<int*>(destPtr);
      auto casLoop = [=] __device__(core::atomicType atype, int operand) {
        while (true) {
          int oldVal = core::AtomicLoadSeqCst(ptr4);
          int newVal = oldVal;
          switch (atype) {
            case core::AMO_INC:
              newVal = oldVal + 1;
              break;
            case core::AMO_ADD:
            case core::AMO_SIGNAL_ADD:
              newVal = oldVal + operand;
              break;
            case core::AMO_AND:
              newVal = oldVal & operand;
              break;
            case core::AMO_OR:
              newVal = oldVal | operand;
              break;
            case core::AMO_XOR:
              newVal = oldVal ^ operand;
              break;
            default:
              break;
          }
          int expected = oldVal;
          int prev = core::AtomicCompareExchangeSystem(ptr4, &expected, newVal);
          if (prev == oldVal) break;
        }
      };
      switch (amoType) {
        case core::AMO_SET:
        case core::AMO_SIGNAL_SET:
          core::AtomicStoreSeqCstSystem(ptr4, argVal);
          break;
        case core::AMO_INC:
        case core::AMO_ADD:
        case core::AMO_AND:
        case core::AMO_OR:
        case core::AMO_XOR:
        case core::AMO_SIGNAL_ADD:
          casLoop(amoType, argVal);
          break;
        default:
          printf("Error: Unsupported 4-byte atomicType (%d)\n", amoType);
          break;
      }
      break;
    }
    case 8: {
      long long argVal = *reinterpret_cast<long long*>(val);
      long long* ptr8 = reinterpret_cast<long long*>(destPtr);
      auto casLoop64 = [=] __device__(core::atomicType atype, long long operand) {
        while (true) {
          long long oldVal = core::AtomicLoadSeqCst(ptr8);
          long long newVal = oldVal;
          switch (atype) {
            case core::AMO_INC:
              newVal = oldVal + 1;
              break;
            case core::AMO_ADD:
            case core::AMO_SIGNAL_ADD:
              newVal = oldVal + operand;
              break;
            case core::AMO_AND:
              newVal = oldVal & operand;
              break;
            case core::AMO_OR:
              newVal = oldVal | operand;
              break;
            case core::AMO_XOR:
              newVal = oldVal ^ operand;
              break;
            default:
              break;
          }
          long long expected = oldVal;
          long long prev = core::AtomicCompareExchangeSystem(ptr8, &expected, newVal);
          if (prev == oldVal) break;
        }
      };
      switch (amoType) {
        case core::AMO_SET:
        case core::AMO_SIGNAL_SET:
          core::AtomicStoreSeqCstSystem(ptr8, argVal);
          break;
        case core::AMO_INC:
        case core::AMO_ADD:
        case core::AMO_AND:
        case core::AMO_OR:
        case core::AMO_XOR:
        case core::AMO_SIGNAL_ADD:
          casLoop64(amoType, argVal);
          break;
        default:
          printf("Error: Unsupported 8-byte atomicType (%d)\n", amoType);
          break;
      }
      break;
    }
    default:
      printf("Error: Unsupported data size (%zu bytes)\n", bytes);
      break;
  }
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernel<application::TransportType::P2P>(
    const void* dest, void* val, size_t bytes, core::atomicType amoType, int pe, int qpId) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::P2P>(dest, val, bytes, amoType,
                                                                         pe, qpId);
  }
}

// New pure address-based Atomic Fetch operations for P2P transport
template <typename T>
inline __device__ T ShmemAtomicTypeFetchThreadKernelImplP2P_Addr(const void* dest, void* val,
                                                                 void* compare, size_t bytes,
                                                                 core::atomicType amoType, int pe,
                                                                 int qpId) {
  RemoteAddrInfo remoteInfo = ShmemAddrToRemoteAddr(dest, pe);
  T* destPtr = reinterpret_cast<T*>(remoteInfo.raddr);
  T* fetchResPtr = reinterpret_cast<T*>(val);
  T cmpVal = (compare != nullptr) ? *reinterpret_cast<T*>(compare) : T{};

  auto casLoop = [=] __device__(T * addr, core::atomicType op, T operand, T cmpVal, T * oldResult) {
    T oldVal = core::AtomicLoadSeqCstSystem(addr);
    while (true) {
      T newVal = oldVal;
      switch (op) {
        case core::AMO_FETCH_INC:
          newVal = oldVal + T{1};
          break;
        case core::AMO_FETCH_ADD:
          newVal = oldVal + operand;
          break;
        case core::AMO_FETCH_AND:
          if constexpr (std::is_integral_v<T>) {
            newVal = oldVal & operand;
          }
          break;
        case core::AMO_FETCH_OR:
          if constexpr (std::is_integral_v<T>) {
            newVal = oldVal | operand;
          }
          break;
        case core::AMO_FETCH_XOR:
          if constexpr (std::is_integral_v<T>) {
            newVal = oldVal ^ operand;
          }
          break;
        case core::AMO_SWAP:
          newVal = operand;
          break;
        case core::AMO_COMPARE_SWAP:
          if (oldVal == cmpVal) {
            newVal = operand;
          } else {
            newVal = oldVal;
          }
          break;
        default:
          break;
      }

      T expected = oldVal;
      T prev = core::AtomicCompareExchangeSystem(addr, &expected, newVal);
      if (prev == oldVal) {
        *oldResult = oldVal;
        break;
      }
      oldVal = prev;
    }
    return oldVal;
  };

  T* valPtr = reinterpret_cast<T*>(val);
  T operand = *valPtr;

  switch (amoType) {
    case core::AMO_FETCH_INC:
    case core::AMO_FETCH_ADD:
    case core::AMO_FETCH_AND:
    case core::AMO_FETCH_OR:
    case core::AMO_FETCH_XOR:
    case core::AMO_SWAP:
    case core::AMO_COMPARE_SWAP: {
      T result = casLoop(destPtr, amoType, operand, cmpVal, fetchResPtr);
      return result;
    } break;
    default:
      printf("Error: Unsupported atomicType (%d)\n", amoType);
      break;
  }
  return T{};
}

#define DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL_P2P_ADDR(TypeName, T)                         \
  template <>                                                                                      \
  inline __device__ T ShmemAtomicTypeFetchThreadKernel<application::TransportType::P2P, T>(        \
      const void* dest, void* val, void* compare, size_t bytes, core::atomicType amoType, int pe,  \
      int qpId) {                                                                                  \
    return ShmemAtomicTypeFetchThreadKernelImplP2P_Addr<T>(dest, val, compare, bytes, amoType, pe, \
                                                           qpId);                                  \
  }

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL_P2P_ADDR(Uint32, uint32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL_P2P_ADDR(Uint64, uint64_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL_P2P_ADDR(Int32, int32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_THREAD_KERNEL_P2P_ADDR(Int64, int64_t)

#define DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL_P2P_ADDR(TypeName, T)                          \
  template <>                                                                                     \
  inline __device__ T ShmemAtomicTypeFetchWarpKernel<application::TransportType::P2P, T>(         \
      const void* dest, void* val, void* compare, size_t bytes, core::atomicType amoType, int pe, \
      int qpId) {                                                                                 \
    int laneId = threadIdx.x & (warpSize - 1);                                                    \
    if (laneId == 0) {                                                                            \
      return ShmemAtomicTypeFetchThreadKernelImplP2P_Addr<T>(dest, val, compare, bytes, amoType,  \
                                                             pe, qpId);                           \
    }                                                                                             \
    return T{};                                                                                   \
  }

DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL_P2P_ADDR(Uint32, uint32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL_P2P_ADDR(Uint64, uint64_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL_P2P_ADDR(Int32, int32_t)
DEFINE_SHMEM_ATOMIC_TYPE_FETCH_WARP_KERNEL_P2P_ADDR(Int64, int64_t)

/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */
template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::P2P>() {}

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::P2P>(int pe) {}

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::P2P>(int pe, int qpId) {}

}  // namespace shmem
}  // namespace mori
