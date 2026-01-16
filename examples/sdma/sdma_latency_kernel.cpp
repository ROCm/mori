#include "sdma_latency_kernel.h"

// Each WG wants to access all destinations.
// Each warps send their data to different destinations
// The warp selects the SDMA engine via sdmaEngineMap[srcGPU][dstGPU].
// Then each workgroup will target different queues [0-7], first WG handle all ) numbered queue of SDMA engines
// queueIdx = deviceHandle[engineId * 8 + blockId.x]
// So, multiple warps handling different destinations from the same block. However, if nWarps in a WG > num_dst, then
// may write to the same queue After all copy packets, each warp places an ATOMIC_DEC packet to its assigned queue and
// waits for completion.
template <typename T, bool TIMESTAMPING_EN>
__global__ void multiQueueSDMATransfer(size_t iteration_id, void* srcBuf, void* dstBuf, size_t copy_size,
                                       size_t numCopyCommands, anvil::SdmaQueueDeviceHandle* deviceHandle,
                                       HSAuint64* signals, HSAuint64 expectedSignal, long long int* start_clock_count,
                                       long long int* end_clock_count, TimeStampBreakdown* timestamp_breakdown)
{
   uint64_t base = 0;
   uint64_t pendingWptr = 0;

   const int warpId = threadIdx.x / warpSize;
   const int laneId = threadIdx.x % warpSize;
   const int nWarps = blockDim.x / warpSize;

   // Each warp handles it's poriton of the src and dst buffer
   const size_t total_size = copy_size * numCopyCommands;

   anvil::SdmaQueueDeviceHandle handle = *deviceHandle;

   const size_t offset = warpId * total_size;
   char* srcPtr = reinterpret_cast<char*>(srcBuf) + offset;
   char* dstPtr = reinterpret_cast<char*>(dstBuf) + offset;

   // Ensure all warps consumes in a WG
   const int signalIdx = 0; // blockIdx.x * nWarps + warpId;
   HSAuint64* signal = signals + signalIdx;

   if (laneId == 0)
   {
      start_clock_count[signalIdx] = wall_clock64();
   }

   const int warpGlobalIndex = iteration_id * (gridDim.x * nWarps) + blockIdx.x * nWarps + warpId;

   for (int c = 0; c < numCopyCommands; ++c)
   {
      const int packetGlobalIndex = iteration_id * (gridDim.x * nWarps * numCopyCommands) +
                                    blockIdx.x * (nWarps * numCopyCommands) + warpId * numCopyCommands + c;

      if (laneId == 0)
      {
         if constexpr (TIMESTAMPING_EN)
         {
            timestamp_breakdown->reserveQueueSpace_st[packetGlobalIndex] = wall_clock64();
         }

         base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR));

         if constexpr (TIMESTAMPING_EN)
         {
            timestamp_breakdown->reserveQueueSpace_e[packetGlobalIndex] = wall_clock64();
         }
      }

      pendingWptr = base;

      auto packet = anvil::CreateCopyPacket(srcPtr, dstPtr, copy_size);

      if (laneId == 0)
      {
         if constexpr (TIMESTAMPING_EN)
         {
            timestamp_breakdown->entailPacket_st[packetGlobalIndex] = wall_clock64();
         }

         handle.template placePacket<SDMA_PKT_COPY_LINEAR>(packet, pendingWptr);

         if constexpr (TIMESTAMPING_EN)
         {
            timestamp_breakdown->entailPacket_e[packetGlobalIndex] = timestamp_breakdown->submit_st[packetGlobalIndex] =
                wall_clock64();
         }

         handle.submitPacket(base, pendingWptr);

         if constexpr (TIMESTAMPING_EN)
         {
            timestamp_breakdown->submit_e[packetGlobalIndex] = wall_clock64();
         }
      }
      srcPtr += copy_size;
      dstPtr += copy_size;
   }

   if (laneId == 0)
   {
      if constexpr (TIMESTAMPING_EN)
      {
         timestamp_breakdown->iterTimeStamps[warpGlobalIndex][PerIterationTimeStamps::RESERVE_SPACE_FENCE_START] =
             wall_clock64();
      }
      base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC));
      if constexpr (TIMESTAMPING_EN)
      {
         timestamp_breakdown->iterTimeStamps[warpGlobalIndex][PerIterationTimeStamps::RESERVE_SPACE_FENCE_END] =
             wall_clock64();
      }
   }

   pendingWptr = base;
   auto packet = anvil::CreateAtomicIncPacket(signal);
   if (laneId == 0)
   {
      if constexpr (TIMESTAMPING_EN)
      {
         timestamp_breakdown->iterTimeStamps[warpGlobalIndex][PerIterationTimeStamps::ENTAIL_FENCE_PACKET_START] =
             wall_clock64();
      }
      handle.template placePacket<SDMA_PKT_ATOMIC>(packet, pendingWptr);
      if constexpr (TIMESTAMPING_EN)
      {
         long long int ts = wall_clock64();
         timestamp_breakdown->iterTimeStamps[warpGlobalIndex][PerIterationTimeStamps::ENTAIL_FENCE_PACKET_END] = ts;
         timestamp_breakdown->iterTimeStamps[warpGlobalIndex][PerIterationTimeStamps::SUBMIT_FENCE_PACKET_START] = ts;
      }
   }

   if (laneId == 0)
   {
      handle.submitPacket(base, pendingWptr);
      if constexpr (TIMESTAMPING_EN)
      {
         long long int ts = wall_clock64();
         timestamp_breakdown->iterTimeStamps[warpGlobalIndex][PerIterationTimeStamps::SUBMIT_FENCE_PACKET_END] = ts;
         timestamp_breakdown->iterTimeStamps[warpGlobalIndex][PerIterationTimeStamps::TRANSFER_START] = ts;
      }
   }

   if (laneId == 0)
   {
      if (anvil::waitForSignal(signal, expectedSignal)) // all warps consumed
      {
         if constexpr (TIMESTAMPING_EN)
         {
            timestamp_breakdown->iterTimeStamps[warpGlobalIndex][PerIterationTimeStamps::TRANSFER_END] = wall_clock64();
         }
         end_clock_count[signalIdx] = wall_clock64();
      }
      else
      {
         if constexpr (TIMESTAMPING_EN)
         {
            timestamp_breakdown->iterTimeStamps[warpGlobalIndex][PerIterationTimeStamps::TRANSFER_END] = -1;
         }
         end_clock_count[signalIdx] = -1;
      }
   }

   return;
}

#define INSTANTIATE_KERNEL(T, L)                                                                                       \
   template __global__ void multiQueueSDMATransfer<T, L>(                                                              \
       size_t iteration_id, void* srcBuf, void* dstBufs, size_t copySize, size_t numCopyCommands,                      \
       anvil::SdmaQueueDeviceHandle* deviceHandle, HSAuint64* signals, HSAuint64 expectedSignal,                       \
       long long int* start_clock_count, long long int* end_clock_count, TimeStampBreakdown* timestamp_breakdown);

INSTANTIATE_KERNEL(anvil::SdmaQueueDeviceHandle, false)
INSTANTIATE_KERNEL(anvil::SdmaQueueDeviceHandle, true)
INSTANTIATE_KERNEL(anvil::SdmaQueueSingleProducerDeviceHandle, false)
INSTANTIATE_KERNEL(anvil::SdmaQueueSingleProducerDeviceHandle, true)
