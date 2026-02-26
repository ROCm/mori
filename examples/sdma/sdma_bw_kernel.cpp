/**
 * @acknowledgements:
 * - Original implementation by: Sidler, David
 * - Source: https://github.com/AARInternal/shader_sdma
 * 
 * @note: This code is adapted/modified from the implementation by Sidler, David
 */

#include "sdma_bw_kernel.h"

// Multi WG map to Same Queue
__global__ void multiQueueSDMATransferQueueMapWG(size_t iteration_id, void* srcBuf, void** dstBufs, size_t copy_size,
                                                 size_t numCopyCommands, int numOfDestinations,
                                                 int numOfQueues,
                                                 int numOfWGPerQueue,
                                                 anvil::SdmaQueueDeviceHandle** deviceHandle, HSAuint64* signals,
                                                 HSAuint64 expectedSignal, long long int* start_clock_count,
                                                 long long int* end_clock_count)
{
   const int warpId = threadIdx.x / warpSize;
   const int laneId = threadIdx.x % warpSize;
   const int nWarps = blockDim.x / warpSize;

   // WG position within it's destination group
   const int numOfWGPerDst = numOfWGPerQueue * numOfQueues;
   const int blockIdWithInDst = blockIdx.x % numOfWGPerDst;

   // k WGs share single queue
   const int queueSelector = blockIdWithInDst / numOfWGPerQueue; // different groups get different queue Id
   const int dstIdx = blockIdx.x % numOfDestinations;
   const int queueIdx = (dstIdx * numOfQueues) + queueSelector;

   // Each warp handles it's poriton of the src and dst buffer
   const size_t total_size = copy_size * numCopyCommands;
   const size_t offset = blockIdWithInDst * total_size;

   anvil::SdmaQueueDeviceHandle handle = *deviceHandle[queueIdx];

   char* srcPtr = reinterpret_cast<char*>(srcBuf) + offset;
   char* dstPtr = reinterpret_cast<char*>(dstBufs[dstIdx]) + offset;

   // Ensure all warps consumes in a WG
   const int signalIdx = blockIdx.x * nWarps + warpId;
   HSAuint64* signal = signals + signalIdx;

   if (laneId == 0)
   {
      start_clock_count[signalIdx] = wall_clock64();
   }

   for (size_t c = 0; c < numCopyCommands; ++c)
   {
      if (laneId == 0)
      {
         anvil::put(handle, dstPtr, srcPtr, copy_size);
      }
      srcPtr += copy_size;
      dstPtr += copy_size;
   }

   if (laneId == 0)
   {
      anvil::signal(handle, signal);
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
