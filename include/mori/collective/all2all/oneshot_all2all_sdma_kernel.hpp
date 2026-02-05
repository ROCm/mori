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
__global__ void OneShotAll2allSdmaKernel(int myPe, int npes,
		                                 T* input,
                                         const application::SymmMemObjPtr inputTransitMemObj,  // Changed to input transit buffer
                                         const application::SymmMemObjPtr outputTransitMemObj, // Output transit buffer
                                         const application::SymmMemObjPtr flagsMemObj,
                                         size_t elementCount) {
    if (elementCount == 0 || npes <= 0) {
        return;
    }

    // Get input transit buffer pointer (contains only current PE's data)
    T* __restrict__ inputData = input;
    
    // Get output transit buffer pointer (will receive data from all PEs)
    T* __restrict__ stageData = reinterpret_cast<T*>(outputTransitMemObj->localPtr);
    
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
        application::SymmMemObjPtr dest = outputTransitMemObj;
        uint8_t* srcPtr = reinterpret_cast<uint8_t *>(inputData) + srcByteOffset;
        uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[targetPe] + destByteOffset);
        anvil::SdmaQueueDeviceHandle** devicehandles = dest->deviceHandles_d + targetPe*dest->sdmaNumQueue;
        HSAuint64* signals = dest->signalPtrs + targetPe*dest->sdmaNumQueue;
        HSAuint64* expectedSignals = dest->expectSignalsPtr + targetPe*dest->sdmaNumQueue;
        core::SdmaPutThread(srcPtr, dstPtr, sendBytes, devicehandles, signals, expectedSignals, dest->sdmaNumQueue, qId);
    }

    // Synchronization and flag setting
    if (threadLinearId < npes) {
        int targetPe = threadLinearId;
        shmem::ShmemQuietThread(targetPe, outputTransitMemObj);
        if (threadLinearId < npes) {
          flags[threadLinearId] = 0;
        }
        __syncthreads();
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
            T* peData = stageData + pe * elementCount;
            printf("  Data from PE %d (first 2 values): %u %u\n", 
                   pe, static_cast<uint32_t>(peData[0]), static_cast<uint32_t>(peData[1]));
        }
    }
    #endif
}

}  // namespace collective
}  // namespace mori
