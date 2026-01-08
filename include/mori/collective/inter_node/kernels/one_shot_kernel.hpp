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

/**
 * One-shot AllReduce - Single block version
 *
 * This version uses a single block to avoid cross-block synchronization issues.
 * Suitable for small to medium messages where single block provides enough parallelism.
 *
 * Algorithm:
 * 1. Send: Each PE sends its entire input buffer to all other PEs' scratch buffers
 * 2. Signal: After quiet, signal to all peers that data is ready (using epoch)
 * 3. Wait: Wait for data from all peers (by checking their flags)
 * 4. Reduce: Reduce all received data into output buffer
 */
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

  // Device-side pointers
  T* __restrict__ src = reinterpret_cast<T*>(srcMemObj->localPtr);
  T* __restrict__ dst = reinterpret_cast<T*>(dstMemObj->localPtr);
  T* __restrict__ scratch = reinterpret_cast<T*>(scratchMemObj->localPtr);
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);

  const int tid = threadIdx.x;
  const int numThreads = blockDim.x;
  const size_t totalBytes = elementCount * sizeof(T);

  // Special case: single PE, just copy
  if (npes == 1) {
    for (size_t idx = tid; idx < elementCount; idx += numThreads) {
      dst[idx] = src[idx];
    }
    return;
  }

  // ========================================================================
  // Phase 1: Send local data to all peers' scratch buffers
  // ========================================================================
  // Each thread handles a portion of the data
  const size_t bytesPerThread = (totalBytes + numThreads - 1) / numThreads;
  const size_t myByteStart = static_cast<size_t>(tid) * bytesPerThread;

  if (myByteStart < totalBytes) {
    size_t myByteEnd = myByteStart + bytesPerThread;
    if (myByteEnd > totalBytes) {
      myByteEnd = totalBytes;
    }
    const size_t myBytes = myByteEnd - myByteStart;

    // Send to all peers
    for (int remotePe = 0; remotePe < npes; ++remotePe) {
      if (remotePe == myPe) {
        continue;
      }
      size_t destByteOffset = static_cast<size_t>(myPe) * totalBytes + myByteStart;
      shmem::ShmemPutMemNbiThread(scratchMemObj, destByteOffset, srcMemObj, myByteStart, myBytes,
                                  remotePe);
    }
  }

  // Ensure all RDMA writes are complete
  __threadfence_system();
  shmem::ShmemQuietThread();
  __syncthreads();

  // ========================================================================
  // Phase 2: Signal to all peers that data is ready
  // ========================================================================
  if (tid == 0) {
    for (int remotePe = 0; remotePe < npes; ++remotePe) {
      if (remotePe == myPe) {
        continue;
      }
      // Set flag[myPe] = epoch on remote PE
      shmem::ShmemAtomicTypeNonFetchThread<uint64_t>(flagsMemObj,
                                                     static_cast<size_t>(myPe) * sizeof(uint64_t),
                                                     epoch, core::atomicType::AMO_SET, remotePe);
    }
  }
  __syncthreads();

  // ========================================================================
  // Phase 3: Wait for data from all peers
  // ========================================================================
  // Distribute waiting across threads to reduce contention
  for (int sender = tid; sender < npes; sender += numThreads) {
    if (sender == myPe) {
      continue;
    }

    const int maxSpinCount = 100000000;
    int spinCount = 0;
    uint64_t flagValue;

    do {
      // Use SYSTEM scope for cross-GPU visibility
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

  // Ensure all waits complete before reduction
  __syncthreads();
  __threadfence_system();

  // ========================================================================
  // Phase 4: Reduce all received data into output
  // ========================================================================
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
