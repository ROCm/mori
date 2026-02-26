/**
 * @acknowledgements:
 * - Original implementation by: Sidler, David
 * - Source: https://github.com/AARInternal/shader_sdma
 * 
 * @note: This code is adapted/modified from the implementation by Sidler, David
 */

#include "sdma_rate_kernel.h"

__global__ void packet_rate_kernel(void* srcBuf, void* dstBuf, size_t copy_size, size_t numCopyCommands,
                                   anvil::SdmaQueueDeviceHandle** deviceHandles, HSAuint64* signals,
                                   HSAuint64 expectedSignal, long long int* start_clock_count,
                                   long long int* end_clock_count)
{
   uint64_t base = 0;
   uint64_t pendingWptr = 0;

   const int warpId = threadIdx.x / warpSize;
   const int laneId = threadIdx.x % warpSize;
   const int nWarps = blockDim.x / warpSize;

   const size_t total_size = copy_size * numCopyCommands;

   anvil::SdmaQueueDeviceHandle handle = *deviceHandles[warpId];

   const size_t offset = warpId * total_size;
   char* srcPtr = reinterpret_cast<char*>(srcBuf) + offset;
   char* dstPtr = reinterpret_cast<char*>(dstBuf) + offset;

   // Ensure all warps consumes in a WG
   const int signalIdx = blockIdx.x * nWarps + warpId;
   HSAuint64* signal = signals + signalIdx;

   if (laneId != 0)
      return;

   uint64_t startBase;

   for (int c = 0; c < numCopyCommands; ++c)
   {
      if (laneId == 0)
      {
         base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR));
         pendingWptr = base;
         if (c == 0)
         {
            startBase = base;
         }

         auto packet = anvil::CreateCopyPacket(srcPtr, dstPtr, copy_size);

         handle.template placePacket<SDMA_PKT_COPY_LINEAR>(packet, pendingWptr);
         srcPtr += copy_size;
         dstPtr += copy_size;
      }
   }

   if (laneId == 0)
   {
      base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC));

      pendingWptr = base;
      auto packet = anvil::CreateAtomicIncPacket(signal);
      handle.template placePacket<SDMA_PKT_ATOMIC>(packet, pendingWptr);
   }

   if (laneId == 0)
   {
      __threadfence_system();
      start_clock_count[signalIdx] = wall_clock64();
   }

   if (laneId == 0)
   {
      handle.submitPacket(startBase, pendingWptr);
   }

   if (laneId == 0)
   {
      if (anvil::waitForSignal(signal, expectedSignal)) // all warps consumed
      {
         end_clock_count[signalIdx] = wall_clock64();
      }
      else
      {
         end_clock_count[signalIdx] = -1;
      }
   }

   return;
}
