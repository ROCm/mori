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

__global__ void SdmaPutThread(void* srcBuf, void* dstBuf, size_t copy_size, 
                                anvil::SdmaQueueDeviceHandle** deviceHandles, 
                                HSAuint64* signals, HSAuint64 expectedSignal)
{
   uint64_t base = 0;
   uint64_t pendingWptr = 0;
   uint64_t startBase = 0;

//   const int warpId = threadIdx.x / warpSize;
//   const int laneId = threadIdx.x % warpSize;
//   const int nWarps = blockDim.x / warpSize;

   const size_t rand_size = copy_size / 8; // per queue rand data

   char* srcPtr = reinterpret_cast<char*>(srcBuf);
   char* dstPtr = reinterpret_cast<char*>(dstBuf);

   for(int q =0; q<8; q++){
      anvil::SdmaQueueDeviceHandle handle = **(deviceHandles+q);
      base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR));
      pendingWptr = base;
      startBase = base;

      if(q < 7) perq_send_size = rand_size;
      else perq_send_size = copy_size - 7*rand_size;
      
      auto packet = anvil::CreateCopyPacket(srcPtr, dstPtr, perq_send_size);
      handle.template placePacket(packet, pendingWptr);
      srcPtr += perq_send_size;
      dstPtr += perq_send_size;

      base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC));
      pendingWptr = base;
      HSAuint64* signal = signals + q;
      auto packet = anvil::CreateAtomicIncPacket(signal);
      handle.template placePacket<SDMA_PKT_ATOMIC>(packet, pendingWptr);

      handle.submitPacket(startBase, pendingWptr);
   }

}


__global__ void SdmaPutWarp(void* srcBuf, void* dstBuf, size_t copy_size, 
                                anvil::SdmaQueueDeviceHandle** deviceHandles, 
                                HSAuint64* signals, HSAuint64 expectedSignal)
{
   uint64_t base = 0;
   uint64_t pendingWptr = 0;
   uint64_t startBase = 0;

//   const int warpId = threadIdx.x / warpSize;
   const int laneId = threadIdx.x % warpSize;
//   const int nWarps = blockDim.x / warpSize;

   if(lanId >= 8) return;
   int queueId = lanId;
   const size_t rand_size = copy_size / 8; // per queue rand data

   char* srcPtr = reinterpret_cast<char*>(srcBuf);
   char* dstPtr = reinterpret_cast<char*>(dstBuf);

   anvil::SdmaQueueDeviceHandle handle = **(deviceHandles+queueId);
   base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR));
   pendingWptr = base;
   startBase = base;

   if(queueId < 7) perq_send_size = rand_size;
   else perq_send_size = copy_size - 7*rand_size;
   
   auto packet = anvil::CreateCopyPacket(srcPtr, dstPtr, perq_send_size);
   handle.template placePacket(packet, pendingWptr);
   srcPtr += perq_send_size;
   dstPtr += perq_send_size;

   base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC));
   pendingWptr = base;
   HSAuint64* signal = signals + queueId;
   auto packet = anvil::CreateAtomicIncPacket(signal);
   handle.template placePacket<SDMA_PKT_ATOMIC>(packet, pendingWptr);

   handle.submitPacket(startBase, pendingWptr);
}



/* ---------------------------------------------------------------------------------------------- */
/*                                         Completion Queue                                       */
/* ---------------------------------------------------------------------------------------------- */
__global__ void SdmaQueitThread(HSAuint64* signals, HSAuint64 expectedSignal)
{
   for(int q =0; q <8; q++){
      anvil::waitForSignal(signal+q, expectedSignal);
   }
}

__global__ void SdmaQueitWarp(HSAuint64* signals, HSAuint64 expectedSignal)
{
   const int laneId = threadIdx.x % warpSize;

   if(laneId >=8 ) return;
   int queueId = lanId;
   anvil::waitForSignal(signal+queueId, expectedSignal);
}
}
}