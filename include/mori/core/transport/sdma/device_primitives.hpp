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

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "mori/application/transport/sdma/anvil_device.hpp"

namespace mori {
namespace core {

/* ---------------------------------------------------------------------------------------------- */
/*                                           Post Tasks                                           */
/* ---------------------------------------------------------------------------------------------- */

// SdmaPutThread: COPY data + ATOMIC increment signal.
// signalAddr is the remote PE's signal memory (mapped locally via IPC).
// The remote PE reads its own signalPtrs to detect completion.
inline __device__ void SdmaPutThread(void* srcBuf, void* dstBuf, size_t copy_size,
                                anvil::SdmaQueueDeviceHandle** deviceHandles,
                                HSAuint64* signalAddr, uint32_t queNum, uint32_t qId)
{
   if (copy_size == 0) {
      __hip_atomic_fetch_add(signalAddr + qId, 1ULL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      return;
   }

   uint64_t base = 0;
   uint64_t pendingWptr = 0;
   uint64_t startBase = 0;
   uint64_t offset = 0;
   char* srcPtr = reinterpret_cast<char*>(srcBuf);
   char* dstPtr = reinterpret_cast<char*>(dstBuf);

   // printf("[SDMA DBG] qId=%u srcBuf=%p dstBuf=%p copy_size=%zu signalAddr=%p\n",
   //        qId, srcBuf, dstBuf, copy_size, (void*)signalAddr);

   anvil::SdmaQueueDeviceHandle* handlePtr = *(deviceHandles+qId);
   // printf("[SDMA DBG] qId=%u deviceHandles=%p handlePtr=%p\n",
   //        qId, (void*)deviceHandles, (void*)handlePtr);
   // if (handlePtr == nullptr) {
   //    printf("[SDMA ERR] qId=%u handlePtr is NULL!\n", qId);
   //    assert(false);
   // }
   anvil::SdmaQueueDeviceHandle handle = *handlePtr;
   // printf("[SDMA DBG] qId=%u queueBuf=%p wptr=%p rptr=%p doorbell=%p\n",
   //        qId, (void*)handle.queueBuf, (void*)handle.wptr, (void*)handle.rptr, (void*)handle.doorbell);

   base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR), offset);
   pendingWptr = base;
   startBase = base;
   // printf("[SDMA DBG] qId=%u after 1st reserve: base=%lu offset=%lu\n", qId, base, offset);

   auto packet_d = anvil::CreateCopyPacket(srcPtr, dstPtr, copy_size);
   handle.template placePacket<SDMA_PKT_COPY_LINEAR>(packet_d, pendingWptr, offset);

   base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC), offset);
   // printf("[SDMA DBG] qId=%u after 2nd reserve: base=%lu startBase=%lu gap=%ld\n",
   //        qId, base, startBase, (long)(base - pendingWptr));
   pendingWptr = base;
   auto packet_s = anvil::CreateAtomicIncPacket(signalAddr + qId);
   handle.template placePacket<SDMA_PKT_ATOMIC>(packet_s, pendingWptr, offset);

   handle.submitPacket(startBase, pendingWptr);
}


inline __device__ void SdmaPutWarp(void* srcBuf, void* dstBuf, size_t copy_size,
                                anvil::SdmaQueueDeviceHandle** deviceHandles,
                                HSAuint64* signalAddr, uint32_t queNum)
{
   const int laneId = threadIdx.x % warpSize;

   if(laneId >= queNum) return;
   int queueId = laneId;

   if (copy_size == 0) {
      __hip_atomic_fetch_add(signalAddr + queueId, 1ULL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      return;
   }

   uint64_t base = 0;
   uint64_t pendingWptr = 0;
   uint64_t startBase = 0;
   uint64_t offset = 0;
   size_t perq_send_size =0;

   const size_t rand_size = copy_size / queNum; // per queue rand data

   char* srcPtr = reinterpret_cast<char*>(srcBuf);
   char* dstPtr = reinterpret_cast<char*>(dstBuf);

   anvil::SdmaQueueDeviceHandle handle = **(deviceHandles+queueId);
   base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR), offset);
   pendingWptr = base;
   startBase = base;

   if(queueId < (queNum -1)) perq_send_size = rand_size;
   else perq_send_size = copy_size - (queNum -1)*rand_size;

   auto packet_d = anvil::CreateCopyPacket(srcPtr, dstPtr, perq_send_size);
   handle.template placePacket<SDMA_PKT_COPY_LINEAR>(packet_d, pendingWptr, offset);
   srcPtr += perq_send_size;
   dstPtr += perq_send_size;

   base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC), offset);
   pendingWptr = base;
   auto packet_s = anvil::CreateAtomicIncPacket(signalAddr + queueId);
   handle.template placePacket<SDMA_PKT_ATOMIC>(packet_s, pendingWptr, offset);

   handle.submitPacket(startBase, pendingWptr);
}



/* ---------------------------------------------------------------------------------------------- */
/*                                         Completion Queue                                       */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ void SdmaQueitThread(HSAuint64* signals, HSAuint64* expectedSignals, uint32_t queNum)
{
   for(int q =0; q < queNum; q++){
      anvil::waitForSignal(signals+q, *(expectedSignals+q));
   }
}

inline __device__ void SdmaQueitWarp(HSAuint64* signals, HSAuint64* expectedSignals, uint32_t queNum)
{
   const int laneId = threadIdx.x % warpSize;

   if(laneId >=queNum) return;
   int queueId = laneId;
   anvil::waitForSignal(signals+queueId, *(expectedSignals + queueId));
}
}
}
