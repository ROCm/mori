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

template <typename T>
__global__ void OneShotAllReduceKernelSingleBlock(int myPe, int npes,
                                                  const application::SymmMemObjPtr srcMemObj,
                                                  const application::SymmMemObjPtr dstMemObj,
                                                  const application::SymmMemObjPtr scratchMemObj,
                                                  const application::SymmMemObjPtr flagsMemObj,
                                                  size_t elementCount, uint64_t epoch) {
  if (elementCount == 0 || npes <= 0) {
    return;
  }

  T* __restrict__ src = reinterpret_cast<T*>(srcMemObj->localPtr);
  T* __restrict__ dst = reinterpret_cast<T*>(dstMemObj->localPtr);
  T* __restrict__ scratch = reinterpret_cast<T*>(scratchMemObj->localPtr);
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);

  const int tid = threadIdx.x;
  const int numThreads = blockDim.x;
  const size_t totalBytes = elementCount * sizeof(T);

  if (npes == 1) {
    for (size_t idx = tid; idx < elementCount; idx += numThreads) {
      dst[idx] = src[idx];
    }
    return;
  }

  const size_t bytesPerThread = (totalBytes + numThreads - 1) / numThreads;
  const size_t myByteStart = static_cast<size_t>(tid) * bytesPerThread;

  if (myByteStart < totalBytes) {
    size_t myByteEnd = myByteStart + bytesPerThread;
    if (myByteEnd > totalBytes) {
      myByteEnd = totalBytes;
    }
    const size_t myBytes = myByteEnd - myByteStart;

    for (int remotePe = 0; remotePe < npes; ++remotePe) {
      if (remotePe == myPe) {
        continue;
      }
      size_t destByteOffset = static_cast<size_t>(myPe) * totalBytes + myByteStart;
      shmem::ShmemPutMemNbiThread(scratchMemObj, destByteOffset, srcMemObj, myByteStart, myBytes,
                                  remotePe);
    }
  }

  __threadfence_system();
  shmem::ShmemQuietThread();
  __syncthreads();

  if (tid == 0) {
    for (int remotePe = 0; remotePe < npes; ++remotePe) {
      if (remotePe == myPe) {
        continue;
      }
      shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(flagsMemObj,
                                                     static_cast<size_t>(myPe) * sizeof(uint64_t),
                                                     epoch, core::atomicType::AMO_SET, remotePe);
    }
  }
  __syncthreads();

  for (int sender = tid; sender < npes; sender += numThreads) {
    if (sender == myPe) {
      continue;
    }

    const int maxSpinCount = 100000000;
    int spinCount = 0;
    uint64_t flagValue;

    do {
      flagValue = core::AtomicLoadSeqCstSystem(flags + sender);
      if (flagValue >= epoch) {
        break;
      }
      spinCount++;
      if (spinCount > maxSpinCount) {
        printf("PE %d: Timeout waiting for data from peer %d (epoch %lu, got %lu)\n", myPe, sender,
               epoch, flagValue);
        break;
      }
    } while (true);
  }

  __syncthreads();
  __threadfence_system();

  for (size_t idx = tid; idx < elementCount; idx += numThreads) {
    T acc = src[idx];

    for (int sender = 0; sender < npes; ++sender) {
      if (sender == myPe) {
        continue;
      }
      acc += scratch[static_cast<size_t>(sender) * elementCount + idx];
    }

    dst[idx] = acc;
  }
}

}  // namespace collective
}  // namespace mori
