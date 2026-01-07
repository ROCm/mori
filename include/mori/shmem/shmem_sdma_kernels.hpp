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
#include "mori/shmem/shmem_api.hpp"
#include "src/shmem/internal.hpp"

#include "mori/shmem/shmem_p2p_kernels.hpp"

namespace mori {
namespace shmem {
/* ---------------------------------------------------------------------------------------------- */
/*                                         Point-to-Point                                         */
/* ---------------------------------------------------------------------------------------------- */

template <>
inline __device__ void ShmemPutMemNbiThreadKernel<application::TransportType::SDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, size_t bytes, int pe,
    int qpId) {
    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(source.addr + sourceOffset);
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);

    anvil::SdmaQueueDeviceHandle** devicehandles = dest->deviceHandles_d + pe*dest->sdmaNumQueue;

    HSAuint64* signals = dest->signalPtrs + pe*dest->sdmaNumQueue;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + pe*dest->sdmaNumQueue;

    core::SdmaPutThread(srcPtr, dstPtr, bytes, devicehandles, signals, expectedSignals, dest->sdmaNumQueue);
}

template <>
inline __device__ void ShmemPutMemNbiWarpKernel<application::TransportType::SDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset,
    const application::RdmaMemoryRegion& source, size_t sourceOffset, size_t bytes, int pe,
    int qpId) {
    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(source.addr + sourceOffset);
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[pe] + destOffset);

    anvil::SdmaQueueDeviceHandle** devicehandles = dest->deviceHandles_d + pe*dest->sdmaNumQueue;
    HSAuint64* signals = dest->signalPtrs + pe*dest->sdmaNumQueue;
    HSAuint64* expectedSignal = dest->expectSignalsPtr + pe*dest->sdmaNumQueue;

    core::SdmaPutWarp(srcPtr, dstPtr, bytes, devicehandles, signals, expectedSignal, dest->sdmaNumQueue);
}

template <>
inline __device__ void ShmemPutSizeImmNbiThreadKernel<application::TransportType::SDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe,
    int qpId) {
}
template <>
inline __device__ void ShmemPutSizeImmNbiWarpKernel<application::TransportType::SDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes, int pe,
    int qpId) {
}
/* ---------------------------------------------------------------------------------------------- */
/*                                    PutMemNbi with Signal                                       */
/* ---------------------------------------------------------------------------------------------- */

template <>
inline __device__ void ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes,
    core::atomicType amoType, int pe, int qpId) {
        ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::P2P>(dest, destOffset, val, bytes, amoType, pe);
}

template <>
inline __device__ void ShmemAtomicSizeNonFetchWarpKernel<application::TransportType::SDMA>(
    const application::SymmMemObjPtr dest, size_t destOffset, void* val, size_t bytes,
    core::atomicType amoType, int pe, int qpId) {
        ShmemAtomicSizeNonFetchWarpKernel<application::TransportType::P2P>(dest, destOffset, val, bytes, amoType, pe);
}


/* ---------------------------------------------------------------------------------------------- */
/*                                         Synchronization                                        */
/* ---------------------------------------------------------------------------------------------- */

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::SDMA>() {}

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::SDMA>(int pe) {}

template <>
inline __device__ void ShmemQuietThreadKernel<application::TransportType::SDMA>(int pe, int qpId) {}

template <application::TransportType>
inline __device__ void ShmemQuietThreadKernel(int pe, const application::SymmMemObjPtr dest){
    anvil::SdmaQueueDeviceHandle** devicehandles = dest->deviceHandles_d + pe*dest->sdmaNumQueue;
    HSAuint64* signals = dest->signalPtrs + pe*dest->sdmaNumQueue;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + pe*dest->sdmaNumQueue;

    core::SdmaQueitThread(signals, expectedSignals, dest->sdmaNumQueue);
}

template <application::TransportType>
inline __device__ void ShmemQuietWarpKernel(int pe, const application::SymmMemObjPtr dest){
    anvil::SdmaQueueDeviceHandle** devicehandles = dest->deviceHandles_d + pe*dest->sdmaNumQueue;
    HSAuint64* signals = dest->signalPtrs + pe*dest->sdmaNumQueue;
     HSAuint64* expectedSignals = dest->expectSignalsPtr + pe*dest->sdmaNumQueue;

    core::SdmaQueitWarp(signals, expectedSignals, dest->sdmaNumQueue);
}


}  // namespace shmem
}  // namespace mori
