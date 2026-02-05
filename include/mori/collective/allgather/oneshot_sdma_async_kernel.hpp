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
#include "mori/core/transport/rdma/device_primitives.hpp"

namespace mori {
namespace collective {

template <typename T>
__global__ void OneShotAllGatherSdmaAsyncPutKernel(int myPe, int npes,
		                               T* input,
                                       const application::SymmMemObjPtr srcMemObj,
                                       const application::SymmMemObjPtr dstMemObj,
                                       const application::SymmMemObjPtr flagsMemObj,
                                       size_t elementCount) {
  if (elementCount == 0 || npes <= 0) {
    return;
  }

  T* __restrict__ inputData = input;
  T* __restrict__ dst = reinterpret_cast<T*>(dstMemObj->localPtr);
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t threadsPerGrid = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
  const size_t stride = threadsPerGrid > 0 ? threadsPerGrid : 1;

  const size_t bytesPerElement = sizeof(T);
  const size_t bytesPerPeer = elementCount * bytesPerElement;
  const size_t elemsPerPeer = elementCount;

  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;

  if(threadLinearId < npes * dstMemObj->sdmaNumQueue){
    int qId = threadLinearId % dstMemObj->sdmaNumQueue;
    int remotePe = threadLinearId / dstMemObj->sdmaNumQueue;
    const size_t sendBytes_rand = bytesPerPeer/8;
    size_t destByteOffset = myPe*bytesPerPeer + qId*sendBytes_rand;
    size_t srcByteOffset = qId*sendBytes_rand;
    size_t sendBytes = 0;

    if( qId == 7) sendBytes = bytesPerPeer - 7*sendBytes_rand;
    else sendBytes = sendBytes_rand;

    if (laneId == 0) printf(" new no copy !!!!!!!!!!!!!!!!!!!\n");
    application::SymmMemObjPtr dest = dstMemObj;
    uint8_t* srcPtr = reinterpret_cast<uint8_t *>(inputData) + srcByteOffset;
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + destByteOffset;
    anvil::SdmaQueueDeviceHandle** devicehandles = dest->deviceHandles_d + remotePe*dest->sdmaNumQueue;
    HSAuint64* signals = dest->signalPtrs + remotePe*dest->sdmaNumQueue;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe*dest->sdmaNumQueue;
    core::SdmaPutThread(srcPtr, dstPtr, sendBytes, devicehandles, signals, expectedSignals, dest->sdmaNumQueue, qId);
  }
}

__global__ void OneShotAllGatherSdmaAsyncWaitKernel(int myPe, int npes,
                                        const application::SymmMemObjPtr dstMemObj,
                                        const application::SymmMemObjPtr flagsMemObj) {
  int flag_val = 1;
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);

  const size_t threadLinearId = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;

  if(threadLinearId < npes){
    int remotePe = threadLinearId;
    shmem::ShmemQuietThread(remotePe,dstMemObj);
    shmem::ShmemAtomicSizeNonFetchThread(flagsMemObj, static_cast<size_t>(myPe) * sizeof(uint64_t), &flag_val, 8, core::atomicType::AMO_ADD,remotePe);
  }
  __syncthreads();

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
  }
}
}  // namespace collective
}  // namespace mori