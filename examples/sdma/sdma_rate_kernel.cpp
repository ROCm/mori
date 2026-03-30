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
/**
 * @acknowledgements:
 * - Original implementation by: Sidler, David
 * - Source: https://github.com/AARInternal/shader_sdma
 *
 * @note: This code is adapted/modified from the implementation by Sidler, David
 */
#include "sdma_rate_kernel.h"

__global__ void packet_rate_kernel(void* srcBuf, void* dstBuf, size_t copy_size,
                                   size_t numCopyCommands,
                                   anvil::SdmaQueueDeviceHandle** deviceHandles, HSAuint64* signals,
                                   HSAuint64 expectedSignal, long long int* start_clock_count,
                                   long long int* end_clock_count) {
  uint64_t base = 0;
  uint64_t pendingWptr = 0;
  uint64_t slot_offset = 0;
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

  if (laneId != 0) return;

  uint64_t startBase;

  for (int c = 0; c < numCopyCommands; ++c) {
    if (laneId == 0) {
      base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR), slot_offset);
      pendingWptr = base;
      if (c == 0) {
        startBase = base;
      }

      auto packet = anvil::CreateCopyPacket(srcPtr, dstPtr, copy_size);

      handle.template placePacket<SDMA_PKT_COPY_LINEAR>(packet, pendingWptr, slot_offset);
      srcPtr += copy_size;
      dstPtr += copy_size;
    }
  }

  if (laneId == 0) {
    base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC), slot_offset);

    pendingWptr = base;
    auto packet = anvil::CreateAtomicIncPacket(signal);
    handle.template placePacket<SDMA_PKT_ATOMIC>(packet, pendingWptr, slot_offset);
  }

  if (laneId == 0) {
    __threadfence_system();
    start_clock_count[signalIdx] = wall_clock64();
  }

  if (laneId == 0) {
    handle.submitPacket(startBase, pendingWptr);
  }

  if (laneId == 0) {
    if (anvil::waitForSignal(signal, expectedSignal))  // all warps consumed
    {
      end_clock_count[signalIdx] = wall_clock64();
    } else {
      end_clock_count[signalIdx] = -1;
    }
  }

  return;
}
