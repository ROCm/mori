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

inline __device__ void SdmaPutThread(void* srcBuf, void* dstBuf, size_t copy_size,
                                     anvil::SdmaQueueDeviceHandle** deviceHandles,
                                     HSAuint64* signals, HSAuint64* expectedSignals,
                                     uint32_t queNum, uint32_t qId, int ndesc = 1) {
  uint64_t base = 0;
  uint64_t pendingWptr = 0;
  uint64_t startBase = 0;
  uint64_t offset = 0;
  char* srcPtr = reinterpret_cast<char*>(srcBuf);
  char* dstPtr = reinterpret_cast<char*>(dstBuf);

  anvil::SdmaQueueDeviceHandle handle = **(deviceHandles + qId);
  HSAuint64* signal = signals + qId;

  // MORI_HIER_PUT_NDESC: ndesc>1 tiles the copy into contiguous sub-descriptors +
  // one trailing atomic-inc, single doorbell. Atomic fires after all sub-copies in
  // FIFO order and expectedSignals bumps once, so one quiet guards the column;
  // byte-identical to the single-descriptor path. ndesc<=1 => single descriptor.
  if (ndesc <= 1) {
    base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR), offset);
    pendingWptr = base;
    startBase = base;

    auto packet_d = anvil::CreateCopyPacket(srcPtr, dstPtr, copy_size);
    handle.template placePacket<SDMA_PKT_COPY_LINEAR>(packet_d, pendingWptr, offset);

    base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC), offset);
    pendingWptr = base;
    auto packet_s = anvil::CreateAtomicIncPacket(signal);
    handle.template placePacket<SDMA_PKT_ATOMIC>(packet_s, pendingWptr, offset);

    handle.submitPacket(startBase, pendingWptr);
    expectedSignals[qId]++;
    return;
  }

  // Split into equal sub-ranges tiled on the 16B SDMA_PKT_COPY_LINEAR granularity
  // (CDNA3 quad-word alignment); keeps every sub-descriptor legal and byte-identical.
  const size_t unit = 16;
  const size_t nU = (copy_size + unit - 1) / unit;
  size_t n = static_cast<size_t>(ndesc);
  if (n > nU) n = nU < 1 ? 1 : nU;
  const size_t uPerD = (nU + n - 1) / n;
  int nReal = 0;
  for (size_t d = 0; d < n; ++d) {
    size_t s = d * uPerD * unit;
    if (s >= copy_size) break;
    ++nReal;
  }
  if (nReal < 1) nReal = 1;

  const size_t reserveBytes =
      static_cast<size_t>(nReal) * sizeof(SDMA_PKT_COPY_LINEAR) + sizeof(SDMA_PKT_ATOMIC);
  base = handle.ReserveQueueSpace(reserveBytes, offset);
  pendingWptr = base;
  startBase = base;

  // Wrap padding (offset) applies only to the first placement; rest contiguous.
  uint64_t placeOffset = offset;
  for (int d = 0; d < nReal; ++d) {
    size_t s = static_cast<size_t>(d) * uPerD * unit;
    size_t e = s + uPerD * unit;
    if (e > copy_size) e = copy_size;
    auto packet_d = anvil::CreateCopyPacket(srcPtr + s, dstPtr + s, e - s);
    handle.template placePacket<SDMA_PKT_COPY_LINEAR>(packet_d, pendingWptr, placeOffset);
    placeOffset = 0;
  }
  auto packet_s = anvil::CreateAtomicIncPacket(signal);
  handle.template placePacket<SDMA_PKT_ATOMIC>(packet_s, pendingWptr, placeOffset);

  handle.submitPacket(startBase, pendingWptr);
  expectedSignals[qId]++;
}

// MORI_HIER_QFLAG: COPY_LINEAR + peer-flag WRITE as an SDMA FENCE packet on the
// same queue, one doorbell. FIFO order => peer sees the flag strictly after the
// copied bytes (landing fence); the reader's seq-cst system acquire +
// __threadfence_system publish the landed bytes to consumers. Peer flag is the
// completion token; no local signal, expectedSignals not bumped. flagVal must fit
// 32 bits (written to the low word of the u64 flag slot).
inline __device__ void SdmaPutFencedFlagThread(void* srcBuf, void* dstBuf, size_t copy_size,
                                               anvil::SdmaQueueDeviceHandle** deviceHandles,
                                               HSAuint64* signals, HSAuint64* expectedSignals,
                                               uint32_t queNum, uint32_t qId, void* peerFlagAddr,
                                               uint32_t flagVal) {
  (void)signals;
  (void)expectedSignals;
  (void)queNum;
  uint64_t offset = 0;
  anvil::SdmaQueueDeviceHandle handle = **(deviceHandles + qId);
  uint64_t base =
      handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR) + sizeof(SDMA_PKT_FENCE), offset);
  uint64_t startBase = base;
  uint64_t pendingWptr = base;
  auto copy_packet = anvil::CreateCopyPacket(reinterpret_cast<char*>(srcBuf),
                                             reinterpret_cast<char*>(dstBuf), copy_size);
  handle.template placePacket<SDMA_PKT_COPY_LINEAR>(copy_packet, pendingWptr, offset);
  auto fence_packet = anvil::CreateFencePacket(reinterpret_cast<HSAuint64*>(peerFlagAddr), flagVal);
  handle.template placePacket<SDMA_PKT_FENCE>(fence_packet, pendingWptr, 0);
  handle.submitPacket(startBase, pendingWptr);
}

// COPY_LINEAR + a LOCAL atomic-inc of signals[qId] so the caller drains this
// submit (SdmaQueitThread) then sets the peer flag via P2P AMO_SET; no remote flag
// rides the copy engine. Distinct from SdmaPutFencedFlagThread (COPY+FENCE to
// peerFlagAddr). peerFlagAddr/flagVal unused here.
inline __device__ void SdmaPutCopySignalThread(void* srcBuf, void* dstBuf, size_t copy_size,
                                               anvil::SdmaQueueDeviceHandle** deviceHandles,
                                               HSAuint64* signals, HSAuint64* expectedSignals,
                                               uint32_t queNum, uint32_t qId, void* peerFlagAddr,
                                               uint32_t flagVal) {
  (void)queNum;
  (void)flagVal;
  (void)peerFlagAddr;
  uint64_t offset = 0;
  anvil::SdmaQueueDeviceHandle handle = **(deviceHandles + qId);
  HSAuint64* signal = signals + qId;
  uint64_t base =
      handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR) + sizeof(SDMA_PKT_ATOMIC), offset);
  uint64_t startBase = base;
  uint64_t pendingWptr = base;
  auto copy_packet = anvil::CreateCopyPacket(reinterpret_cast<char*>(srcBuf),
                                             reinterpret_cast<char*>(dstBuf), copy_size);
  handle.template placePacket<SDMA_PKT_COPY_LINEAR>(copy_packet, pendingWptr, offset);
  auto sig_packet = anvil::CreateAtomicIncPacket(signal);
  handle.template placePacket<SDMA_PKT_ATOMIC>(sig_packet, pendingWptr, 0);
  handle.submitPacket(startBase, pendingWptr);
  expectedSignals[qId]++;
}

// Tail-counter completion: COPY_LINEAR + a REMOTE ADD64(+1) on the same queue, one
// doorbell. SQ FIFO => peer sees the +1 only after the bytes land (landing fence).
// The ADD64 must ride the SDMA engine, not a CU atomic: peerPtrs are engine-mapped
// (XGMI P2P), not CU-P2P-atomic-safe. Peer counter is the sole completion token; no
// local signal, expectedSignals not bumped. peerCounterAddr = peer uint64 counter.
inline __device__ void SdmaPutCopyRemoteAddThread(void* srcBuf, void* dstBuf, size_t copy_size,
                                                  anvil::SdmaQueueDeviceHandle** deviceHandles,
                                                  uint32_t qId, void* peerCounterAddr) {
  // A single COPY_LINEAR >8 MiB faults on the CDNA3 SDMA microcode: split into
  // <=8 MiB contiguous sub-descriptors, then the trailing ADD64. Hard copy-size
  // cap, not latency pipelining.
  const size_t kMaxCopy = 8u * 1024u * 1024u;
  size_t nCopy = (copy_size + kMaxCopy - 1) / kMaxCopy;
  if (nCopy < 1) nCopy = 1;
  uint64_t offset = 0;
  anvil::SdmaQueueDeviceHandle handle = **(deviceHandles + qId);
  uint64_t base = handle.ReserveQueueSpace(
      nCopy * sizeof(SDMA_PKT_COPY_LINEAR) + sizeof(SDMA_PKT_ATOMIC), offset);
  uint64_t startBase = base;
  uint64_t pendingWptr = base;
  char* srcC = reinterpret_cast<char*>(srcBuf);
  char* dstC = reinterpret_cast<char*>(dstBuf);
  uint64_t placeOffset = offset;
  for (size_t i = 0; i < nCopy; ++i) {
    size_t o = i * kMaxCopy;
    size_t len = copy_size - o;
    if (len > kMaxCopy) len = kMaxCopy;
    auto copy_packet = anvil::CreateCopyPacket(srcC + o, dstC + o, len);
    handle.template placePacket<SDMA_PKT_COPY_LINEAR>(copy_packet, pendingWptr, placeOffset);
    placeOffset = 0;  // wrap padding applies only to the first placement
  }
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
  const size_t rand_size = copy_size / queNum;

  char* srcPtr = reinterpret_cast<char*>(srcBuf);
  char* dstPtr = reinterpret_cast<char*>(dstBuf);

  // Each queue owns a disjoint chunk (last absorbs the remainder). src/dst MUST be
  // advanced to this queue's chunk start, else all queues copy from offset 0
  // (overlapping front, tail never copied).
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
