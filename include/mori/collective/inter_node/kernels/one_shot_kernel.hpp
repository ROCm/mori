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

#include <hip/hip_runtime.h>

#include <cstddef>

#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

// One-shot all-reduce: single phase.
// Every GPU reads the full buffer from all peers, accumulates locally, and writes the result.
template <typename T>
__global__ void OneShotAllReduceKernel(int myPe, int npes,
                                       const application::SymmMemObjPtr srcMemObj,
                                       const application::SymmMemObjPtr dstMemObj,
                                       const application::SymmMemObjPtr scratchMemObj,
                                       const application::SymmMemObjPtr flagsMemObj,
                                       size_t elementCount) {
  if (elementCount == 0 || npes <= 0) {
    return;
  }

  T* __restrict__ src = reinterpret_cast<T*>(srcMemObj->localPtr);
  T* __restrict__ dst = reinterpret_cast<T*>(dstMemObj->localPtr);
  T* __restrict__ scratch = reinterpret_cast<T*>(scratchMemObj->localPtr);
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t threadsPerGrid = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
  const size_t stride = threadsPerGrid > 0 ? threadsPerGrid : 1;

  const size_t bytesPerElement = sizeof(T);
  const size_t bytesPerPeer = elementCount * bytesPerElement;
  const size_t elemsPerPeer = elementCount;

  for (size_t idx = threadLinearId; idx < elementCount; idx += stride) {
    dst[idx] = src[idx];
  }
  __syncthreads();

  const size_t bytesPerThread =
      (bytesPerPeer + stride - 1) / stride;  // ceil division to cover all bytes

  for (int remotePe = 0; remotePe < npes; ++remotePe) {
    if (remotePe == myPe) {
      continue;
    }

    size_t threadByteOffset = bytesPerThread * threadLinearId;
    if (threadByteOffset < bytesPerPeer) {
      size_t sendBytes = bytesPerThread;
      size_t remaining = bytesPerPeer - threadByteOffset;
      if (sendBytes > remaining) {
        sendBytes = remaining;
      }
      size_t destByteOffset = static_cast<size_t>(myPe) * bytesPerPeer + threadByteOffset;
      shmem::ShmemPutMemNbiSignalThread<true>(
          scratchMemObj, destByteOffset, srcMemObj, threadByteOffset, sendBytes, flagsMemObj,
          static_cast<size_t>(myPe) * sizeof(uint64_t), 1, core::atomicType::AMO_ADD, remotePe);
      shmem::ShmemQuietThread();
    }
    __syncthreads();
  }

  for (int sender = 0; sender < npes; ++sender) {
    if (sender == myPe) {
      continue;
    }

    if (threadLinearId == 0) {
      int spinCount = 0;
      while (core::AtomicLoadRelaxed(flags + sender) == 0) {
        ++spinCount;
        if (spinCount > 10000000) {
          printf("PE %d: Timeout waiting for data from peer %d\n", myPe, sender);
          break;
        }
      }
    }
    __syncthreads();

    size_t senderBase = static_cast<size_t>(sender) * elemsPerPeer;
    for (size_t idx = threadLinearId; idx < elementCount; idx += stride) {
      dst[idx] += scratch[senderBase + idx];
    }
    __syncthreads();

    if (threadLinearId == 0) {
      flags[sender] = 0;
    }
    __syncthreads();
  }
}

}  // namespace collective
}  // namespace mori
