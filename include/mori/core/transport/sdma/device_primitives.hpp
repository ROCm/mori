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

#include "mori/core/transport/sdma/anvil_device.hpp"
#include "mori/core/utils/utils.hpp"  // warpSize

namespace mori {
namespace core {

/* ---------------------------------------------------------------------------------------------- */
/*                                           Post Tasks                                           */
/* ---------------------------------------------------------------------------------------------- */
template <bool Signal = true>
inline __device__ void SdmaPostCopy(anvil::SdmaQueueDeviceHandle** deviceHandles,
                                    HSAuint64* signals, HSAuint64* expectedSignals, void* srcPtr,
                                    void* dstPtr, size_t size, int qId, bool ring = true) {
  uint64_t offset = 0;
  anvil::SdmaQueueDeviceHandle handle = **(deviceHandles + qId);

  uint64_t startBase = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR), offset);
  uint64_t pendingWptr = startBase;

  auto packet_d = anvil::CreateCopyPacket(srcPtr, dstPtr, size);
  handle.template placePacket<SDMA_PKT_COPY_LINEAR>(packet_d, pendingWptr, offset);

  if constexpr (Signal) {
    pendingWptr = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC), offset);
    HSAuint64* signal = signals + qId;
    auto packet_s = anvil::CreateAtomicIncPacket(signal);
    handle.template placePacket<SDMA_PKT_ATOMIC>(packet_s, pendingWptr, offset);
    expectedSignals[qId]++;
  }

  if (ring) handle.submitPacket(startBase, pendingWptr);
}

// Ring the doorbell for everything placed-but-not-rung on this queue.
inline __device__ void SdmaRingQueueDbr(anvil::SdmaQueueDeviceHandle& handle) {
  uint64_t base =
      __hip_atomic_load(handle.committedWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  uint64_t pending =
      __hip_atomic_load(handle.cachedWptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  if (pending != base) handle.submitPacket(base, pending);
}

// Queue this lane/thread drives for warp/block scope, or -1 when beyond queNum.
inline __device__ int SdmaWarpQueueId(uint32_t queNum) {
  const int laneId = threadIdx.x % warpSize;
  return laneId < static_cast<int>(queNum) ? laneId : -1;
}
inline __device__ int SdmaBlockQueueId(uint32_t queNum) {
  const int tid = static_cast<int>(threadIdx.x);
  return tid < static_cast<int>(queNum) ? tid : -1;
}

// Multi-queue split: the caller's rank in the coop group selects the queue; the
// last active queue absorbs the remainder so uneven sizes are fully covered.
template <bool Signal = true>
inline __device__ void SdmaPutMultiQueue(void* srcBuf, void* dstBuf, size_t copy_size,
                                         anvil::SdmaQueueDeviceHandle** deviceHandles,
                                         HSAuint64* signals, HSAuint64* expectedSignals,
                                         uint32_t queNum, int rank, bool ring = true) {
  if (rank >= static_cast<int>(queNum)) return;
  const int queueId = rank;
  const size_t rand_size = copy_size / queNum;  // per queue slice size
  // Too small to split (copy_size < queNum): queue 0 sends the whole thing on a
  // single queue, the rest stay idle — avoids posting 0-byte copies.
  if (rand_size == 0) {
    if (rank == 0 && copy_size > 0) {
      SdmaPostCopy<Signal>(deviceHandles, signals, expectedSignals, srcBuf, dstBuf, copy_size, 0,
                           ring);
    }
    return;
  }
  const size_t perq_send_size =
      (queueId < static_cast<int>(queNum - 1)) ? rand_size : (copy_size - (queNum - 1) * rand_size);
  const size_t byteOffset = static_cast<size_t>(queueId) * rand_size;

  char* srcPtr = reinterpret_cast<char*>(srcBuf) + byteOffset;
  char* dstPtr = reinterpret_cast<char*>(dstBuf) + byteOffset;

  SdmaPostCopy<Signal>(deviceHandles, signals, expectedSignals, srcPtr, dstPtr, perq_send_size,
                       queueId, ring);
}

// Thread scope: one thread drives a single queue `qId` with the full copy.
template <bool Signal = true>
inline __device__ void SdmaPutThread(void* srcBuf, void* dstBuf, size_t copy_size,
                                     anvil::SdmaQueueDeviceHandle** deviceHandles,
                                     HSAuint64* signals, HSAuint64* expectedSignals,
                                     uint32_t /*queNum*/, uint32_t qId, bool ring = true) {
  SdmaPostCopy<Signal>(deviceHandles, signals, expectedSignals, srcBuf, dstBuf, copy_size,
                       static_cast<int>(qId), ring);
}

// Warp scope: one lane per queue (queueId == laneId), split across all queues.
template <bool Signal = true>
inline __device__ void SdmaPutWarp(void* srcBuf, void* dstBuf, size_t copy_size,
                                   anvil::SdmaQueueDeviceHandle** deviceHandles, HSAuint64* signals,
                                   HSAuint64* expectedSignals, uint32_t queNum, bool ring = true) {
  const int laneId = threadIdx.x % warpSize;
  SdmaPutMultiQueue<Signal>(srcBuf, dstBuf, copy_size, deviceHandles, signals, expectedSignals,
                            queNum, laneId, ring);
}

// Block scope: one thread per queue (queueId == threadIdx.x), split across all
// queues. Lets a transfer use up to blockDim.x queues (i.e. > warpSize).
template <bool Signal = true>
inline __device__ void SdmaPutBlock(void* srcBuf, void* dstBuf, size_t copy_size,
                                    anvil::SdmaQueueDeviceHandle** deviceHandles,
                                    HSAuint64* signals, HSAuint64* expectedSignals, uint32_t queNum,
                                    bool ring = true) {
  SdmaPutMultiQueue<Signal>(srcBuf, dstBuf, copy_size, deviceHandles, signals, expectedSignals,
                            queNum, static_cast<int>(threadIdx.x), ring);
}

// Commit (ring pending packets) per coop scope.
inline __device__ void SdmaCommitThread(anvil::SdmaQueueDeviceHandle** deviceHandles,
                                        uint32_t /*queNum*/, uint32_t qId) {
  anvil::SdmaQueueDeviceHandle handle = **(deviceHandles + qId);
  SdmaRingQueueDbr(handle);
}

inline __device__ void SdmaCommitWarp(anvil::SdmaQueueDeviceHandle** deviceHandles,
                                      uint32_t queNum) {
  const int q = SdmaWarpQueueId(queNum);
  if (q < 0) return;
  anvil::SdmaQueueDeviceHandle handle = **(deviceHandles + q);
  SdmaRingQueueDbr(handle);
}

inline __device__ void SdmaCommitBlock(anvil::SdmaQueueDeviceHandle** deviceHandles,
                                       uint32_t queNum) {
  const int q = SdmaBlockQueueId(queNum);
  if (q < 0) return;
  anvil::SdmaQueueDeviceHandle handle = **(deviceHandles + q);
  SdmaRingQueueDbr(handle);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Completion Queue                                       */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ void SdmaQuietThread(HSAuint64* signals, HSAuint64* expectedSignals,
                                       uint32_t queNum) {
  for (uint32_t q = 0; q < queNum; q++) {
    anvil::waitForSignal(signals + q, *(expectedSignals + q));
  }
}

inline __device__ void SdmaQuietWarp(HSAuint64* signals, HSAuint64* expectedSignals,
                                     uint32_t queNum) {
  const int q = SdmaWarpQueueId(queNum);
  if (q < 0) return;
  anvil::waitForSignal(signals + q, *(expectedSignals + q));
}

inline __device__ void SdmaQuietBlock(HSAuint64* signals, HSAuint64* expectedSignals,
                                      uint32_t queNum) {
  const int q = SdmaBlockQueueId(queNum);
  if (q < 0) return;
  anvil::waitForSignal(signals + q, *(expectedSignals + q));
}
}  // namespace core
}  // namespace mori
