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
#if 0
// One-shot all-reduce: single phase.
// Every GPU reads the full buffer from all peers, accumulates locally, and writes the result.
template <typename T>
__global__ void OneShotAll2allSdmaKernel(int myPe, int npes,
                                       const application::SymmMemObjPtr srcMemObj,
                                       const application::SymmMemObjPtr dstMemObj,
                                       const application::SymmMemObjPtr flagsMemObj,
                                       size_t elementCount) {
  if (elementCount == 0 || npes <= 0) {
    return;
  }

  T* __restrict__ src = reinterpret_cast<T*>(srcMemObj->localPtr);
  T* __restrict__ dst = reinterpret_cast<T*>(dstMemObj->localPtr);
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
  int flag_val = 1;

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

    shmem::ShmemPutMemNbiThread(dstMemObj, destByteOffset, srcMemObj, srcByteOffset, sendBytes, remotePe, qId);
  }

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
#endif
#if 1
template <typename T>
__global__ void OneShotAll2allSdmaKernel(int myPe, int npes,
                                         const application::SymmMemObjPtr inputTransitMemObj,  // Changed to input transit buffer
                                         const application::SymmMemObjPtr outputTransitMemObj, // Output transit buffer
                                         const application::SymmMemObjPtr flagsMemObj,
                                         size_t elementCount) {
    if (elementCount == 0 || npes <= 0) {
        return;
    }

    // Get input transit buffer pointer (contains only current PE's data)
    T* __restrict__ myInputData = reinterpret_cast<T*>(inputTransitMemObj->localPtr);
    
    // Get output transit buffer pointer (will receive data from all PEs)
    T* __restrict__ allOutputData = reinterpret_cast<T*>(outputTransitMemObj->localPtr);
    
    uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
    int flag_val = 1;

    const size_t threadLinearId =
        static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;

    const size_t bytesPerElement = sizeof(T);
    const size_t bytesPerPeer = elementCount * bytesPerElement;

    // Key modification: each thread is responsible for sending its own data to other PEs
    if (threadLinearId < npes * outputTransitMemObj->sdmaNumQueue) {
        int qId = threadLinearId % outputTransitMemObj->sdmaNumQueue;
        int targetPe = threadLinearId / outputTransitMemObj->sdmaNumQueue;
        
        // Calculate bytes to send
        const size_t sendBytes_rand = bytesPerPeer / 8;
        size_t srcByteOffset = targetPe * bytesPerPeer + qId * sendBytes_rand;  // Read from input transit buffer
        size_t destByteOffset = myPe * bytesPerPeer + qId * sendBytes_rand;  // Write to targetPe's position in output transit buffer
        //printf("myPe:%u, threadLinearId:%u, srcByteOffset:0x%x, destByteOffset:0x%x\n", myPe, threadLinearId, srcByteOffset, destByteOffset);        
        size_t sendBytes = 0;

        if (qId == 7) {
            sendBytes = bytesPerPeer - 7 * sendBytes_rand;
        } else {
            sendBytes = sendBytes_rand;
        }
        
        // Send my data to target PE
        shmem::ShmemPutMemNbiThread(outputTransitMemObj, destByteOffset, 
                                   inputTransitMemObj, srcByteOffset, 
                                   sendBytes, targetPe, qId);
    }

    // Synchronization and flag setting
    if (threadLinearId < npes) {
        int targetPe = threadLinearId;
        shmem::ShmemQuietThread(targetPe, outputTransitMemObj);
        shmem::ShmemAtomicSizeNonFetchThread(flagsMemObj, 
                                             static_cast<size_t>(myPe) * sizeof(uint64_t),
                                             &flag_val, 8, core::atomicType::AMO_ADD, targetPe);        
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
                    printf("Kernel[PE %d]: Timeout waiting for data from peer %d\n", myPe, sender);
                    break;
                }
            }
        }
        __syncthreads();
    }
    #if 0
    // Debug information: check data in output transit buffer
    if (threadLinearId == 0) {
        printf("Kernel[PE %d]: Checking output data...\n", myPe);
        for (int pe = 0; pe < npes; pe++) {
            T* peData = allOutputData + pe * elementCount;
            printf("  Data from PE %d (first 2 values): %u %u\n", 
                   pe, static_cast<uint32_t>(peData[0]), static_cast<uint32_t>(peData[1]));
        }
    }
    #endif
}
#endif
}  // namespace collective
}  // namespace mori