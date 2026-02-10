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

#include "mori/application/application.hpp"

namespace mori {
namespace shmem {

/* ---------------------------------------------------------------------------------------------- */
/*                         Pure Address-Based Point-to-Point (New API)                            */
/* ---------------------------------------------------------------------------------------------- */

// New pure address-based kernels - dest and source are direct pointers
template <application::TransportType TsptType>
inline __device__ void ShmemPutMemNbiThreadKernel(const void* dest, const void* source,
                                                  size_t bytes, int pe, int qpId = 0);

template <application::TransportType TsptType>
inline __device__ void ShmemPutMemNbiWarpKernel(const void* dest, const void* source, size_t bytes,
                                                int pe, int qpId = 0);

template <application::TransportType TsptType>
inline __device__ void ShmemPutMemNbiBlockKernel(const void* dest, const void* source, size_t bytes,
                                                 int pe, int qpId = 0);

template <application::TransportType TsptType>
inline __device__ void ShmemPutSizeImmNbiThreadKernel(const void* dest, void* val, size_t bytes,
                                                      int pe, int qpId = 0);

template <application::TransportType TsptType>
inline __device__ void ShmemPutSizeImmNbiWarpKernel(const void* dest, void* val, size_t bytes,
                                                    int pe, int qpId = 0);

template <application::TransportType TsptType, bool onlyOneSignal = true>
inline __device__ void ShmemPutMemNbiSignalThreadKernel(const void* dest, const void* source,
                                                        size_t bytes, const void* signalDest,
                                                        uint64_t signalValue,
                                                        core::atomicType signalOp, int pe,
                                                        int qpId = 0);

template <application::TransportType TsptType, bool onlyOneSignal = true>
inline __device__ void ShmemPutMemNbiSignalWarpKernel(const void* dest, const void* source,
                                                      size_t bytes, const void* signalDest,
                                                      uint64_t signalValue,
                                                      core::atomicType signalOp, int pe,
                                                      int qpId = 0);

template <application::TransportType TsptType>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernel(const void* dest, void* val,
                                                           size_t bytes, core::atomicType amoType,
                                                           int pe, int qpId = 0);

template <application::TransportType TsptType>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernel(const void* dest, void* val, size_t bytes,
                                                         core::atomicType amoType, int pe,
                                                         int qpId = 0);

template <application::TransportType TsptType, typename T>
inline __device__ T ShmemAtomicTypeFetchThreadKernel(const void* dest, void* val, void* compare,
                                                     size_t bytes, core::atomicType amoType, int pe,
                                                     int qpId = 0);

template <application::TransportType TsptType, typename T>
inline __device__ T ShmemAtomicTypeFetchWarpKernel(const void* dest, void* val, void* compare,
                                                   size_t bytes, core::atomicType amoType, int pe,
                                                   int qpId = 0);

/* ---------------------------------------------------------------------------------------------- */
/*                         SymmMemObjPtr-Based Point-to-Point (Old API)                           */
/* ---------------------------------------------------------------------------------------------- */

// Legacy kernels - kept for backward compatibility
template <application::TransportType TsptType>
inline __device__ void ShmemPutMemNbiThreadKernel(const application::SymmMemObjPtr dest,
                                                  size_t destOffset,
                                                  const application::SymmMemObjPtr source,
                                                  size_t sourceOffset, size_t bytes, int pe,
                                                  int qpId = 0);

template <application::TransportType TsptType>
inline __device__ void ShmemPutMemNbiWarpKernel(const application::SymmMemObjPtr dest,
                                                size_t destOffset,
                                                const application::SymmMemObjPtr source,
                                                size_t sourceOffset, size_t bytes, int pe,
                                                int qpId = 0);

template <application::TransportType TsptType>
inline __device__ void ShmemPutMemNbiBlockKernel(const application::SymmMemObjPtr dest,
                                                 size_t destOffset,
                                                 const application::SymmMemObjPtr source,
                                                 size_t sourceOffset, size_t bytes, int pe,
                                                 int qpId = 0);

template <application::TransportType TsptType>
inline __device__ void ShmemPutSizeImmNbiThreadKernel(const application::SymmMemObjPtr dest,
                                                      size_t destOffset, void* val, size_t bytes,
                                                      int pe, int qpId = 0);

template <application::TransportType TsptType>
inline __device__ void ShmemPutSizeImmNbiWarpKernel(const application::SymmMemObjPtr dest,
                                                    size_t destOffset, void* val, size_t bytes,
                                                    int pe, int qpId = 0);

template <application::TransportType TsptType, bool onlyOneSignal = true>
inline __device__ void ShmemPutMemNbiSignalThreadKernel(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId = 0);

template <application::TransportType TsptType, bool onlyOneSignal = true>
inline __device__ void ShmemPutMemNbiSignalWarpKernel(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::SymmMemObjPtr source, size_t sourceOffset, size_t bytes,
    const application::SymmMemObjPtr signalDest, size_t signalDestOffset, uint64_t signalValue,
    core::atomicType signalOp, int pe, int qpId = 0);

template <application::TransportType TsptType>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernel(const application::SymmMemObjPtr dest,
                                                           size_t destOffset, void* val,
                                                           size_t bytes, core::atomicType amoType,
                                                           int pe, int qpId = 0);

template <application::TransportType TsptType>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernel(const application::SymmMemObjPtr dest,
                                                         size_t destOffset, void* val, size_t bytes,
                                                         core::atomicType amoType, int pe,
                                                         int qpId = 0);

template <application::TransportType TsptType, typename T>
inline __device__ T ShmemAtomicTypeFetchThreadKernel(const application::SymmMemObjPtr dest,
                                                     size_t destOffset, void* val, void* compare,
                                                     size_t bytes, core::atomicType amoType, int pe,
                                                     int qpId = 0);

template <application::TransportType TsptType, typename T>
inline __device__ T ShmemAtomicTypeFetchWarpKernel(const application::SymmMemObjPtr dest,
                                                   size_t destOffset, void* val, void* compare,
                                                   size_t bytes, core::atomicType amoType, int pe,
                                                   int qpId = 0);

/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */
template <application::TransportType>
inline __device__ void ShmemQuietThreadKernel();

template <application::TransportType>
inline __device__ void ShmemQuietThreadKernel(int pe);

template <application::TransportType>
inline __device__ void ShmemQuietThreadKernel(int pe, int qpId);

template <application::TransportType>
inline __device__ void ShmemQuietThreadKernel(int pe, const application::SymmMemObjPtr dest);

}  // namespace shmem
}  // namespace mori
