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
    const application::RdmaMemoryRegion& source, size_t sourceOffset, size_t bytes, int pe) {
  uint8_t* srcPtr = reinterpret_cast<uint8_t*>(source.addr + sourceOffset);
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);
  core::ThreadCopy<uint8_t>(destPtr, srcPtr, bytes);
}

template <>
inline __device__ void ShmemPutMemNbiWarpKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, size_t bytes, int pe) {
  uint8_t* srcPtr = reinterpret_cast<uint8_t*>(source.addr + sourceOffset);
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);
  core::WarpCopy<uint8_t>(destPtr, srcPtr, bytes);
}

template <>
inline __device__ void ShmemPutSizeImmNbiThreadKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe) {
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
      printf(
          "Size must be one of [1,2,4,8,16] bytes, got %lu, for arbitrary size, use ShmemPutMemNbi "
          "APIs",
          bytes);
      assert(false);
  }
}

template <>
inline __device__ void ShmemPutSizeImmNbiWarpKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0)
    ShmemPutSizeImmNbiThreadKernel<application::TransportType::P2P>(dest, destOffset, val, bytes,
                                                                    pe);
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, void* val, size_t bytes,
    int pe, core::atomicType amoType) {
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
          printf("Error: Unsupported 4-byte atomicType (%d) in NonFetchThreadKernel.\n", amoType);
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
          printf("Error: Unsupported 8-byte atomicType (%d) in NonFetchThreadKernel.\n", amoType);
          break;
      }
      break;
    }
    default:
      printf("Error: Unsupported data size (%zu bytes) in NonFetchThreadKernel.\n", bytes);
      break;
  }
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, void* val, size_t bytes,
    int pe, core::atomicType amoType) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::P2P>(
        dest, destOffset, source, sourceOffset, val, bytes, pe, amoType);
  }
}

template <>
inline __device__ void ShmemAtomicSizeFetchThreadKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, void* val, void* compare,
    size_t bytes, int pe, core::atomicType amoType) {
  uint8_t* destPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);
  switch (bytes) {
    case 4: {
      int* fetchResPtr = reinterpret_cast<int*>(val);
      int cmpVal = (compare != nullptr) ? *reinterpret_cast<int*>(compare) : 0;
      int* remoteIntPtr = reinterpret_cast<int*>(destPtr);
      auto casLoop = [=] __device__(int* addr, core::atomicType op, int operand, int cmpVal,
                                    int* oldResult) {
        int oldVal = core::AtomicLoadSeqCstSystem(addr);
        while (true) {
          int newVal = oldVal;
          switch (op) {
            case core::AMO_FETCH_INC:
              newVal = oldVal + 1;
              break;
            case core::AMO_FETCH_ADD:
              newVal = oldVal + operand;
              break;
            case core::AMO_FETCH_AND:
              newVal = oldVal & operand;
              break;
            case core::AMO_FETCH_OR:
              newVal = oldVal | operand;
              break;
            case core::AMO_FETCH_XOR:
              newVal = oldVal ^ operand;
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

          int expected = oldVal;
          int prev = core::AtomicCompareExchangeSystem(addr, &expected, newVal);
          if (prev == oldVal) {
            *oldResult = oldVal;
            break;
          }
        }
        return oldVal;
      };
      int* operandIntPtr = reinterpret_cast<int*>(source.addr + sourceOffset);
      int operandInt = *operandIntPtr;
      switch (amoType) {
        case core::AMO_FETCH_INC:
        case core::AMO_FETCH_ADD:
        case core::AMO_FETCH_AND:
        case core::AMO_FETCH_OR:
        case core::AMO_FETCH_XOR:
        case core::AMO_SWAP:
        case core::AMO_COMPARE_SWAP: {
          *operandIntPtr = casLoop(remoteIntPtr, amoType, operandInt, cmpVal, fetchResPtr);
        } break;

        default:
          printf("Error: Unsupported 4-byte atomicType (%d) in FetchThreadKernel.\n", amoType);
          break;
      }

      break;
    }
    case 8: {
      long long* fetchResPtr = reinterpret_cast<long long*>(val);
      long long cmpValLL = (compare != nullptr) ? *reinterpret_cast<long long*>(compare) : 0LL;
      long long* remoteLLPtr = reinterpret_cast<long long*>(destPtr);
      auto casLoop64 = [=] __device__(long long* addr, core::atomicType op, long long operand,
                                      long long cmpValLL, long long* oldResult) {
        long long oldVal = core::AtomicLoadSeqCstSystem(addr);
        while (true) {
          long long newVal = oldVal;
          switch (op) {
            case core::AMO_FETCH_INC:
              newVal = oldVal + 1;
              break;
            case core::AMO_FETCH_ADD:
              newVal = oldVal + operand;
              break;
            case core::AMO_FETCH_AND:
              newVal = oldVal & operand;
              break;
            case core::AMO_FETCH_OR:
              newVal = oldVal | operand;
              break;
            case core::AMO_FETCH_XOR:
              newVal = oldVal ^ operand;
              break;
            case core::AMO_SWAP:
              newVal = operand;
              break;
            case core::AMO_COMPARE_SWAP:
              if (oldVal == cmpValLL) {
                newVal = operand;
              } else {
                newVal = oldVal;
              }
              break;
            default:
              break;
          }

          long long expected = oldVal;
          long long prev = core::AtomicCompareExchangeSystem(addr, &expected, newVal);
          if (prev == oldVal) {
            *oldResult = oldVal;
            break;
          }
        }
        return oldVal;
      };
      long long* operandLLPtr = reinterpret_cast<long long*>(source.addr + sourceOffset);
      long long operandLL = *operandLLPtr;
      switch (amoType) {
        case core::AMO_FETCH_INC:
        case core::AMO_FETCH_ADD:
        case core::AMO_FETCH_AND:
        case core::AMO_FETCH_OR:
        case core::AMO_FETCH_XOR:
        case core::AMO_SWAP:
        case core::AMO_COMPARE_SWAP: {
          *operandLLPtr = casLoop64(remoteLLPtr, amoType, operandLL, cmpValLL, fetchResPtr);
        } break;

        default:
          printf("Error: Unsupported 8-byte atomicType (%d) in FetchThreadKernel.\n", amoType);
          break;
      }
      break;
    }
    default:
      printf("Error: Unsupported data size (%zu bytes) in FetchThreadKernel.\n", bytes);
      break;
  }
}

template <>
inline __device__ void ShmemAtomicSizeFetchWarpKernel<application::TransportType::P2P>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, void* val, void* compare,
    size_t bytes, int pe, core::atomicType amoType) {
  int laneId = threadIdx.x & (warpSize - 1);
  if (laneId == 0) {
    ShmemAtomicSizeFetchThreadKernel<application::TransportType::P2P>(
        dest, destOffset, source, sourceOffset, val, compare, bytes, pe, amoType);
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */
template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::P2P>() {}

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::P2P>(int pe) {}

}  // namespace shmem
}  // namespace mori
