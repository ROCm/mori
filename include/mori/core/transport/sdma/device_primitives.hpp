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

inline __device__ void SdmaPutThread(void* srcBuf, void* dstBuf, size_t copy_size,
                                     anvil::SdmaQueueDeviceHandle** deviceHandles,
                                     HSAuint64* signals, HSAuint64* expectedSignals,
                                     uint32_t queNum, uint32_t qId) {
  uint64_t base = 0;
  uint64_t pendingWptr = 0;
  uint64_t startBase = 0;
  uint64_t offset = 0;
  char* srcPtr = reinterpret_cast<char*>(srcBuf);
  char* dstPtr = reinterpret_cast<char*>(dstBuf);

  anvil::SdmaQueueDeviceHandle handle = **(deviceHandles + qId);
  base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR), offset);
  pendingWptr = base;
  startBase = base;

  auto packet_d = anvil::CreateCopyPacket(srcPtr, dstPtr, copy_size);
  handle.template placePacket<SDMA_PKT_COPY_LINEAR>(packet_d, pendingWptr, offset);

  base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC), offset);
  pendingWptr = base;
  HSAuint64* signal = signals + qId;
  auto packet_s = anvil::CreateAtomicIncPacket(signal);
  handle.template placePacket<SDMA_PKT_ATOMIC>(packet_s, pendingWptr, offset);

  handle.submitPacket(startBase, pendingWptr);
  expectedSignals[qId]++;
}

// COPY_LINEAR + REMOTE ADD64(+1) on the same queue, one doorbell. FIFO order
// => the peer sees the counter increment only after the copied bytes land. The
// ADD64 must ride the SDMA engine (peerCounterAddr is XGMI P2P engine-mapped,
// not CU-atomic-safe). No local signal; expectedSignals not bumped.
inline __device__ void SdmaPutCopyRemoteAddThread(void* srcBuf, void* dstBuf, size_t copy_size,
                                                  anvil::SdmaQueueDeviceHandle** deviceHandles,
                                                  uint32_t qId, void* peerCounterAddr) {
  // Single COPY_LINEAR + one trailing ADD64 in one doorbell.
  uint64_t offset = 0;
  anvil::SdmaQueueDeviceHandle handle = **(deviceHandles + qId);
  uint64_t base = handle.ReserveQueueSpace(
      sizeof(SDMA_PKT_COPY_LINEAR) + sizeof(SDMA_PKT_ATOMIC), offset);
  uint64_t startBase = base;
  uint64_t pendingWptr = base;
  char* srcC = reinterpret_cast<char*>(srcBuf);
  char* dstC = reinterpret_cast<char*>(dstBuf);
  auto copy_packet = anvil::CreateCopyPacket(srcC, dstC, copy_size);
  handle.template placePacket<SDMA_PKT_COPY_LINEAR>(copy_packet, pendingWptr, offset);
  // ADD64(+1) targeting the REMOTE peer tail counter (+1 per drained step).
  auto add_packet = anvil::CreateAtomicIncPacket(reinterpret_cast<HSAuint64*>(peerCounterAddr));
  handle.template placePacket<SDMA_PKT_ATOMIC>(add_packet, pendingWptr, 0);
  handle.submitPacket(startBase, pendingWptr);
}

inline __device__ void SdmaPutWarp(void* srcBuf, void* dstBuf, size_t copy_size,
                                   anvil::SdmaQueueDeviceHandle** deviceHandles, HSAuint64* signals,
                                   HSAuint64* expectedSignals, uint32_t queNum) {
  uint64_t base = 0;
  uint64_t pendingWptr = 0;
  uint64_t startBase = 0;
  size_t perq_send_size = 0;
  uint64_t offset = 0;
  const int laneId = threadIdx.x % warpSize;

  if (laneId >= queNum) return;
  int queueId = laneId;
  const size_t rand_size = copy_size / queNum;  // per queue rand data

  char* srcPtr = reinterpret_cast<char*>(srcBuf);
  char* dstPtr = reinterpret_cast<char*>(dstBuf);

  // Each queue copies a disjoint chunk [q*rand_size, (q+1)*rand_size), last
  // queue takes the remainder; advance src/dst to this queue's chunk start.
  srcPtr += static_cast<size_t>(queueId) * rand_size;
  dstPtr += static_cast<size_t>(queueId) * rand_size;

  anvil::SdmaQueueDeviceHandle handle = **(deviceHandles + queueId);
  base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR), offset);
  pendingWptr = base;
  startBase = base;

  if (queueId < (queNum - 1))
    perq_send_size = rand_size;
  else
    perq_send_size = copy_size - (queNum - 1) * rand_size;

  auto packet_d = anvil::CreateCopyPacket(srcPtr, dstPtr, perq_send_size);
  handle.template placePacket<SDMA_PKT_COPY_LINEAR>(packet_d, pendingWptr, offset);

  base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC), offset);
  pendingWptr = base;
  HSAuint64* signal = signals + queueId;
  auto packet_s = anvil::CreateAtomicIncPacket(signal);
  handle.template placePacket<SDMA_PKT_ATOMIC>(packet_s, pendingWptr, offset);

  handle.submitPacket(startBase, pendingWptr);
  expectedSignals[laneId]++;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                         Completion Queue                                       */
/* ---------------------------------------------------------------------------------------------- */
inline __device__ void SdmaQueitThread(HSAuint64* signals, HSAuint64* expectedSignals,
                                       uint32_t queNum) {
  for (int q = 0; q < queNum; q++) {
    anvil::waitForSignal(signals + q, *(expectedSignals + q));
  }
}

inline __device__ void SdmaQueitWarp(HSAuint64* signals, HSAuint64* expectedSignals,
                                     uint32_t queNum) {
  const int laneId = threadIdx.x % warpSize;

  if (laneId >= queNum) return;
  int queueId = laneId;
  anvil::waitForSignal(signals + queueId, *(expectedSignals + queueId));
}
}  // namespace core
}  // namespace mori
