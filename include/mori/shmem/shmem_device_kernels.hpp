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
