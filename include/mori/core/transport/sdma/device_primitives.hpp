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
                                     uint32_t queNum, uint32_t qId, int ndesc = 1,
                                     int cacheHint = 0) {
  uint64_t base = 0;
  uint64_t pendingWptr = 0;
  uint64_t startBase = 0;
  uint64_t offset = 0;
  char* srcPtr = reinterpret_cast<char*>(srcBuf);
  char* dstPtr = reinterpret_cast<char*>(dstBuf);

  anvil::SdmaQueueDeviceHandle handle = **(deviceHandles + qId);
  HSAuint64* signal = signals + qId;

  // DESCRIPTOR PIPELINING (see HierPutNdesc, MORI_HIER_PUT_NDESC). ndesc>1 splits
  // the single per-peer linear copy into `ndesc` contiguous sub-descriptors placed
  // BACK-TO-BACK on the SAME queue followed by ONE trailing atomic-inc, in ONE
  // doorbell ring. Distinct from the QSplit lever (multiple QUEUES/engines) and
  // DEEP_PIPE (temporal sub-chunks with per-sub landing flags): here a single
  // engine gets several descriptors queued ahead, so its descriptor-fetch latency
  // overlaps with in-flight DMA of the previous descriptor instead of the engine
  // idling between a giant copy and the next op. Bit-exact BY CONSTRUCTION: the
  // sub-copies are disjoint contiguous sub-ranges of the SAME bytes, the atomic
  // fires AFTER all of them in FIFO order, and expectedSignals bumps ONCE => the
  // single quiet still guards the whole column. ndesc<=1 => byte-identical path.
  if (ndesc <= 1) {
    base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_COPY_LINEAR), offset);
    pendingWptr = base;
    startBase = base;

    auto packet_d = anvil::CreateCopyPacket(srcPtr, dstPtr, copy_size, cacheHint);
    handle.template placePacket<SDMA_PKT_COPY_LINEAR>(packet_d, pendingWptr, offset);

    base = handle.ReserveQueueSpace(sizeof(SDMA_PKT_ATOMIC), offset);
    pendingWptr = base;
    auto packet_s = anvil::CreateAtomicIncPacket(signal);
    handle.template placePacket<SDMA_PKT_ATOMIC>(packet_s, pendingWptr, offset);

    handle.submitPacket(startBase, pendingWptr);
    expectedSignals[qId]++;
    return;
  }

  // Multi-descriptor path: split into equal 16B-aligned sub-ranges.
  const size_t unit = 16;
  const size_t nU = (copy_size + unit - 1) / unit;
  size_t n = static_cast<size_t>(ndesc);
  if (n > nU) n = nU < 1 ? 1 : nU;
  const size_t uPerD = (nU + n - 1) / n;
  // Count descriptors that carry nonzero bytes.
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

  // Place each copy sub-descriptor. Wrap padding (offset) applies only to the
  // first placement; subsequent packets are contiguous (offset 0).
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

// COPY-ENGINE FLAG DELIVERY (queued completion, MORI_HIER_QFLAG). Place the peer
// completion-flag WRITE as an SDMA FENCE packet on the SAME queue, immediately
// AFTER the linear copy, in ONE doorbell ring. The engine processes queue packets
// strictly in FIFO order and completes each packet's writes before the next, so
// the peer observes the flag write STRICTLY AFTER the copied bytes -- exactly the
// ordering the drain (SdmaQueitThread) + __threadfence_system + separate direct
// P2P AMO were emulating, but delivered natively by the copy engine. This removes
// the per-peer send-CQ drain, the system-scope fence, and the separate flag store
// from the all-to-all completion critical path (the RCCL copy-engine completion
// model). No local completion signal is enqueued and expectedSignals is NOT bumped
// (nothing drains this queue for this push); the peer flag IS the completion token.
// BIT-EXACT by construction: the flag write is FIFO-ordered after its data on one
// engine, so no reader can see the flag ahead of the bytes; the reader's existing
// seq-cst SYSTEM acquire + its own __threadfence_system still publish the landed
// bytes to consumers. ``flagVal`` must fit in 32 bits (op-generation token); it is
// written to the low word of the u64 flag slot (peer slot zero-init, high word 0),
// matching the AMO_SET the direct path performed.
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

// Pipelined-ring relay copy: copy + LOCAL
// completion-signal ONLY, NO remote flag op on the copy engine. On this MI300X/mlx5
// SDMA microcode folding a REMOTE flag op onto the copy engine is unreliable
// (SDMA_OP_FENCE no-ops its store, WRITE_UNTILED to a peer XGMI addr drops at late
// FIFO hops, a remote ATOMIC stalls the engine drain), so the pipelined-ring relay
// enqueues COPY_LINEAR + a LOCAL SDMA_PKT_ATOMIC inc of signals[qId] (so the caller
// can drain THIS submit via SdmaQueitThread once per step) and sets the peer flag
// the proven way (P2P AMO_SET) AFTER the drain. Distinct from
// SdmaPutFencedFlagThread (COPY+FENCE-to-peerFlagAddr) so the qFlag path is
// untouched. peerFlagAddr/flagVal kept for signature parity, unused here.
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
  uint64_t base = handle.ReserveQueueSpace(
      sizeof(SDMA_PKT_COPY_LINEAR) + sizeof(SDMA_PKT_ATOMIC), offset);
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

// Tail-counter completion model: COPY_LINEAR + a
// REMOTE SDMA_PKT_ATOMIC ADD64(+1) on the SAME queue, ONE doorbell. The atomic rides the
// SQ FIFO strictly AFTER the linear copy, so the destination peer observes the counter
// increment only after the copied bytes have landed (flag-after-bytes delivered NATIVELY
// by the copy engine, with NO CPU send-drain and NO __threadfence_system on the critical
// path). Distinct from SdmaPutCopySignalThread (which increments a LOCAL signal so the
// caller can drain): here the ADD64 targets a REMOTE peer counter address (the peer's
// flag-buffer slot, XGMI P2P engine-mapped) so the receiver's monotonic tail counter is
// bumped by the ENGINE. This is the exact op mori uses for every cross-agent SDMA signal,
// posted with system-scope visibility to the remote agent.
//
// Why the engine, not a CU atomic: a raw CU
// atomicAdd/AtomicAddReleaseSystem on flagsMemObj->peerPtrs[nextPe] HSA-FAULTS (0x1016)
// at G=8 (the _intra flag obj's peerPtrs are ENGINE-addressed, not CU-P2P-atomic-safe),
// and the mori P2P AMO helper is a CAS loop that never observes success and SPINS. The
// SDMA engine, by contrast, IS the addressing domain those peerPtrs belong to => an
// SDMA_PKT_ATOMIC to peerPtrs[nextPe] is the correct, non-faulting publish path.
// No local signal is enqueued and expectedSignals is NOT bumped (nothing drains this
// queue for this push; the peer counter is the sole completion token). ``peerCounterAddr``
// must be the ADD64 target on the destination peer (uint64_t counter, XGMI P2P mapped).
inline __device__ void SdmaPutCopyRemoteAddThread(void* srcBuf, void* dstBuf, size_t copy_size,
                                                  anvil::SdmaQueueDeviceHandle** deviceHandles,
                                                  uint32_t qId, void* peerCounterAddr) {
  // A single SDMA_PKT_COPY_LINEAR whose byte count exceeds ~8 MiB faults on this
  // CDNA3 SDMA microcode ("Write access to a read-only page"): the large single copy
  // runs the count field off the destination allocation onto the next (read-only)
  // page. Split the copy into <=8 MiB contiguous sub-descriptors placed back-to-back on
  // the same queue, then one trailing ADD64 to the peer counter. The engine drains
  // queue packets in strict FIFO order, so the +1 still lands strictly after every byte
  // of every sub-copy (flag-after-bytes native); the accumulating ADD is unchanged (one
  // +1 per drained ring step). At copy_size <= 8 MiB this is byte-identical to the prior
  // single-descriptor path. Distinct from ndesc latency pipelining: this is a hard
  // copy-size cap dictated by the microcode count limit.
  const size_t kMaxCopy = 8u * 1024u * 1024u;  // proven-safe single COPY_LINEAR byte count
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
  // CreateAtomicIncPacket = SDMA_OP_ATOMIC / ADD64 with SRC_DATA=+1, targeting the given
  // address. Point it at the REMOTE peer's tail counter to accumulate +1 per drained step.
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

  // Each queue copies a disjoint contiguous chunk. Queue q owns bytes
  // [q*rand_size, (q+1)*rand_size); the last queue absorbs the remainder.
  // The src/dst pointers MUST be advanced to this queue's chunk start, else
  // every queue copies from offset 0 (overlapping front, tail never copied) —
  // which is why multi-queue runs were previously broken (NUM_CHANNELS forced 1).
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
