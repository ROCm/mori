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

#include <cstddef>

#include "mori/core/transport/rdma/device_primitives.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

// SDMA completion drain for the sub-group gathers. MUST index by GLOBAL pe
// (`pe*nq`), matching the put side: `pe % 8` under-indexes 2nd+ nodes (peBase>=8)
// and drains unarmed slots, firing the flag before the bytes land => torn reads.
__device__ __forceinline__ void SubGroupSdmaDrainPe(int pe, const application::SymmMemObjPtr dest) {
  const uint32_t nq = dest->sdmaNumQueue;
  core::SdmaQueitThread(dest->signalPtrs + static_cast<size_t>(pe) * nq,
                        dest->expectSignalsPtr + static_cast<size_t>(pe) * nq, nq);
}

template <typename T>
__device__ void OneShotAllGatherSdmaKernel_body(int myPe, int npes, T* input,
                                                const application::SymmMemObjPtr srcMemObj,
                                                const application::SymmMemObjPtr dstMemObj,
                                                const application::SymmMemObjPtr flagsMemObj,
                                                size_t elementCount, size_t dstBaseOffset = 0,
                                                uint64_t flagVal = 1) {
  if (elementCount == 0 || npes <= 0) {
    return;
  }

  T* __restrict__ inputData = input;
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;

  const size_t bytesPerElement = sizeof(T);
  const size_t bytesPerPeer = elementCount * bytesPerElement;
  const size_t elemsPerPeer = elementCount;

  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;

  // Each peer's whole shard is one COPY_LINEAR on a single SDMA queue: one copy
  // engine already saturates one XGMI link, so splitting across engines only adds
  // descriptor overhead without raising bandwidth.
  if (warpId < npes && laneId == 0) {
    int remotePe = warpId;
    size_t destByteOffset = myPe * bytesPerPeer;
    size_t srcByteOffset = 0;
    size_t sendBytes = bytesPerPeer;
#if 1
    application::SymmMemObjPtr dest = dstMemObj;
    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(inputData) + srcByteOffset;
    uint8_t* dstPtr =
        reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset + destByteOffset;
    anvil::SdmaQueueDeviceHandle** devicehandles =
        dest->deviceHandles_d + remotePe * dest->sdmaNumQueue;
    HSAuint64* signals = dest->signalPtrs + remotePe * dest->sdmaNumQueue;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * dest->sdmaNumQueue;
    core::SdmaPutThread(srcPtr, dstPtr, sendBytes, devicehandles, signals, expectedSignals,
                        dest->sdmaNumQueue, 0);
#endif
  }

  if (warpId < npes && laneId == 0) {
    int remotePe = warpId;
    SubGroupSdmaDrainPe(remotePe, dstMemObj);
    shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
        flagsMemObj, static_cast<size_t>(myPe) * sizeof(uint64_t), &flagVal, 8,
        core::atomicType::AMO_SET, remotePe, 0);
  }
  __syncthreads();

  // Completion wait: distribute the npes-1 peer-flag spins across the block's first
  // npes threads (thread t polls peer t), then one block barrier + one tail system
  // fence. Every peer flag is still seq-cst SYSTEM-acquired before the barrier
  // releases the block, so the acquire contract is unchanged.
  {
    int sender = static_cast<int>(threadIdx.x);
    if (sender < npes && sender != myPe) {
      // No finite spin threshold: a false timeout would let the kernel proceed
      // before the peer's data is ready.
      int spinCount = 0;
      bool warned = false;
      // Relaxed system spin + one seq-cst SYSTEM acquire on exit: relaxed is still
      // cross-agent-visible (monotonic flags) so it exits identically, and the trailing
      // seq-cst load performs the same acquire as a seq-cst spin's last iteration.
      while (core::AtomicLoadRelaxedSystem(flags + sender) < flagVal) {
        ++spinCount;
        if (!warned && spinCount > 10000000) {
          printf("PE %d: Slow wait for data from peer %d (still waiting)\n", myPe, sender);
          warned = true;
        }
      }
      (void)core::AtomicLoadSeqCstSystem(flags + sender);
    }
    __syncthreads();
    if (threadLinearId == 0) __threadfence_system();
    __syncthreads();
  }

  // Monotonic generation flags; no reset needed.
}

template <typename T>
__global__ void OneShotAllGatherSdmaKernel(int myPe, int npes, T* input,
                                           const application::SymmMemObjPtr srcMemObj,
                                           const application::SymmMemObjPtr dstMemObj,
                                           const application::SymmMemObjPtr flagsMemObj,
                                           size_t elementCount, size_t dstBaseOffset = 0,
                                           uint64_t flagVal = 1) {
  OneShotAllGatherSdmaKernel_body<T>(myPe, npes, input, srcMemObj, dstMemObj, flagsMemObj,
                                     elementCount, dstBaseOffset, flagVal);
}

// ---------------------------------------------------------------------------
// Sub-group intra-node SDMA AllGather (nearest-neighbor pipelined ring)
// ---------------------------------------------------------------------------
// Intra-node phase of the hierarchical AllGather. The group is the arithmetic set
// of global PEs ``{peBase, peBase+peStride, ..., peBase+(groupSize-1)*peStride}``;
// this PE is at position ``groupPos``. Each member SDMA-writes its shard into slot
// ``groupPos`` of every member's buffer, so after the flag handshake every member
// holds all ``groupSize`` shards in group-position order. Flags are indexed by group
// position, so a per-call ``flagVal`` token keeps successive calls race-free without
// a reset. (Flat whole-world gather above is the special case groupSize=npes,
// groupPos=myPe, peBase=0, peStride=1.)
// ``blockLocal`` indexes threads off ``threadIdx.x`` ALONE so this body can run in a
// single block of a larger fused grid (NIC||XGMI overlap; see all_gather.hpp override
// note + ccl_kernel_args.hpp CclFusedRingLocalGatherArgs); default false => grid-wide
// thread id, byte-identical to the single-block launch.
// ``flagBase`` (default 0) offsets this gather's flag slots to ``[flagBase, flagBase+
// groupSize)`` so multiple sub-group gathers can run concurrently without racing on
// the shared slots; 0 keeps the historical single-region layout byte-for-byte.
// Ring: a flat G-way push incasts G-1 concurrent writes the SDMA engines cannot
// saturate; the nearest-neighbor ring keeps one flow per XGMI link (no incast) and
// chunk-pipelines each shard (C ~4MiB chunks). Completion: fencedFlag=1 uses
// SdmaPutCopySignalThread (copy + signal on the engine, no remote op on the copy
// engine -- the only reliable model on this HW); fencedFlag=0 uses per-chunk
// SdmaPutThread+quiet+AMO. Flag (c,k) at flagBase+c*G+k; same final bytes, only the
// copy schedule differs.
template <typename T>
__device__ void OneShotAllGatherSdmaSubGroupRingKernel_body(
    int myPe, int npes, int groupSize, int groupPos, int peBase, int peStride, T* input,
    const application::SymmMemObjPtr dstMemObj, const application::SymmMemObjPtr flagsMemObj,
    size_t elementCount, size_t dstBaseOffset = 0, size_t dstSlotStrideBytes = 0,
    uint64_t flagVal = 1, bool blockLocal = false, size_t flagBase = 0, int fencedFlag = 0) {
  (void)npes;
  if (elementCount == 0 || groupSize <= 0) {
    return;
  }

  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
  const size_t threadLinearId =
      blockLocal ? static_cast<size_t>(threadIdx.x)
                 : static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const int warpId = static_cast<int>(threadLinearId / warpSize);
  const int laneId = threadIdx.x % warpSize;

  const int G = groupSize;
  const int g = groupPos;
  const size_t bytesPerPeer = elementCount * sizeof(T);
  const size_t slotStride = dstSlotStrideBytes != 0 ? dstSlotStrideBytes : bytesPerPeer;

  application::SymmMemObjPtr dest = dstMemObj;
  const int nq = dest->sdmaNumQueue > 0 ? static_cast<int>(dest->sdmaNumQueue) : 1;
  uint8_t* selfDstBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[myPe]) + dstBaseOffset;

  const size_t kChunkTarget = 4u * 1024u * 1024u;  // ~4 MiB per chunk
  int C = static_cast<int>((bytesPerPeer + kChunkTarget - 1) / kChunkTarget);
  if (C < 1) C = 1;
  // Cap ring chunk count at 16: 32 chunks fault ("write to a read-only page") in the
  // relay/self-fill addressing at the 32-chunk boundary. C<=16 keeps every size safe
  // (this is a chunk-count limit, not a per-size buffer edge).
  if (C > 16) C = 16;
  const size_t align = 16;
  size_t chunkBytes = (bytesPerPeer + static_cast<size_t>(C) - 1) / static_cast<size_t>(C);
  chunkBytes = ((chunkBytes + align - 1) / align) * align;
  if (chunkBytes == 0) chunkBytes = bytesPerPeer;

  // Multi-queue submission: the C chunk-pipelines are independent, so stripe them across
  // Q=min(nq,C,numWarps) warps, warp w owning queue w and chunks {c : c%Q==w}, one flow
  // per queue (no incast); each warp drains only its own queue. Per-(c,k) cross-PE
  // AMO_SET on queues>0 can drop flag updates (only queue 0 is reliable); the monotonic
  // per-link step-counter below avoids this by replacing per-chunk atomics with one
  // accumulating counter per link.
  const int numWarps =
      static_cast<int>((static_cast<size_t>(blockDim.x) + warpSize - 1) / warpSize);
  int Q = nq < C ? nq : C;
  if (Q > numWarps) Q = numWarps;
  if (Q < 1) Q = 1;
  // fencedFlag bit1: pin to a single SDMA queue (Q=1), warp0 pipelines all C chunks.
  if ((fencedFlag & 2) != 0) Q = 1;
  // fencedFlag bit2: 2x4 hierarchical half-ring. Split G=8 into two 4-PE halves; phase1 =
  // a 4-ring within each half (incast-free), phase2 = pairwise cross-half exchange
  // (g <-> g+H). Ring diameter 7->3, stays Q=1. Flags on the ring's private region:
  // phase1 counter[c]=flagBase+c, phase2 counter[c]=flagBase+C+c. Same final bytes;
  // default OFF (bit2 clear) => G-ring path unchanged.
  // fencedFlag bit9: flat multi-warp broadcast for small messages (incast not bandwidth-
  // limiting, latency dominates). Each PE self-fills slot g then broadcasts to the G-1
  // others, one submitter warp per dest PE (=distinct engine). Counter: each of the G-1
  // senders +1 to flags[flagBase+c] => want = (flagVal-1)*(G-1)+(G-1), accumulating and
  // order-independent, so final bytes identical. Default OFF (bit9 clear).
  const bool flatMW = ((fencedFlag & 512) != 0) && (G > 1);
  // fencedFlag bit10 (fuseSelf, requires flatMW): fold the self-copy into the broadcast
  // doorbell fan -- warp t copies slot g to PE t (t==g => local) and +1 to PE t's
  // counter, no serial self-fill front. Receiver now sees G writers => want =
  // (flagVal-1)*G+G. Self slot still from input, ADD64-accumulated. Default OFF (bit10).
  const bool fuseSelf = flatMW && ((fencedFlag & 1024) != 0);
  // fencedFlag bit11 (batchSelf, requires flatMW, excl. fuseSelf): defer the self-fill
  // drain -- submit all C self-chunks (pipelined on queue 0) then one quiet. Self slot
  // still drained+fenced before return; only the completion-wait moves out of the loop.
  // Default OFF (bit11).
  const bool batchSelf = flatMW && !fuseSelf && ((fencedFlag & 2048) != 0);
  // fencedFlag bit12 (overlapSelf, requires flatMW, excl. fuseSelf/batchSelf, needs warp
  // G-1): run the self-fill on the free warp (G-1) concurrently with the broadcast
  // (independent: local vs remote engines, broadcast sources from input), deferring the
  // self drain+fence past the broadcast barrier. Self keeps its own signal/quiet path so
  // the G-1-writer barrier is untouched; self slot still drained+fenced before return.
  // Default OFF (bit12).
  const bool overlapSelf =
      flatMW && !fuseSelf && !batchSelf && ((fencedFlag & 4096) != 0) && ((G - 1) < numWarps);
  if (fuseSelf) {
    const uint64_t fWant = (flagVal - 1) * static_cast<uint64_t>(G) + static_cast<uint64_t>(G);
    // warp t copies my owned slot g to PE t (t==g => local self-fill), ADD64 to PE t's counter.
    if (warpId < G && laneId == 0) {
      const int destPe = peBase + warpId * peStride;  // t==g => myPe (local copy)
      anvil::SdmaQueueDeviceHandle** dHandles = dest->deviceHandles_d + destPe * nq;
      uint8_t* dDstBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[destPe]) + dstBaseOffset;
      uint64_t* destF = reinterpret_cast<uint64_t*>(flagsMemObj->peerPtrs[destPe]);
      for (int c = 0; c < C; ++c) {
        size_t off = static_cast<size_t>(c) * chunkBytes;
        if (off >= bytesPerPeer) break;
        size_t len = chunkBytes;
        if (off + len > bytesPerPeer) len = bytesPerPeer - off;
        const size_t myCtr = flagBase + static_cast<size_t>(c);
        uint8_t* src = reinterpret_cast<uint8_t*>(input) + off;
        uint8_t* dstPtr = dDstBase + static_cast<size_t>(g) * slotStride + off;
        void* peerCtr = static_cast<void*>(destF + myCtr);
        core::SdmaPutCopyRemoteAddThread(src, dstPtr, len, dHandles, 0, peerCtr);
      }
    }
    __syncthreads();
    // single completion barrier: my slot region filled by all G writers (incl. my own self-copy).
    for (int c = static_cast<int>(threadIdx.x); c < C; c += static_cast<int>(blockDim.x)) {
      if (static_cast<size_t>(c) * chunkBytes >= bytesPerPeer) continue;
      const size_t myCtr = flagBase + static_cast<size_t>(c);
      long spin = 0;
      while (core::AtomicLoadRelaxedSystem(flags + myCtr) < fWant) {
        if (++spin > 20000000L) break;
      }
      (void)core::AtomicLoadSeqCstSystem(flags + myCtr);
    }
    __syncthreads();
    if (threadLinearId == 0) __threadfence_system();
    __syncthreads();
    return;
  }
  if (flatMW) {
    const uint64_t kk8MB = 8u * 1024u * 1024u;
    const uint64_t fWant =
        (flagVal - 1) * static_cast<uint64_t>(G - 1) + static_cast<uint64_t>(G - 1);
    // (1) self-fill slot g from input. overlapSelf runs it on the free warp G-1 (drain+fence
    //     deferred to step 3a) to overlap the broadcast; else warp0, serial front.
    const int selfWarp = overlapSelf ? (G - 1) : 0;
    if (warpId == selfWarp && laneId == 0) {
      anvil::SdmaQueueDeviceHandle** selfHandles = dest->deviceHandles_d + myPe * nq;
      HSAuint64* selfSignals = dest->signalPtrs + myPe * nq;
      HSAuint64* selfExpected = dest->expectSignalsPtr + myPe * nq;
      for (int c = 0; c < C; ++c) {
        size_t off = static_cast<size_t>(c) * chunkBytes;
        if (off >= bytesPerPeer) break;
        size_t len = chunkBytes;
        if (off + len > bytesPerPeer) len = bytesPerPeer - off;
        int sN = static_cast<int>((len + (kk8MB - 1)) / kk8MB);
        if (sN < 1) sN = 1;
        uint8_t* selfSlotP = selfDstBase + static_cast<size_t>(g) * slotStride + off;
        core::SdmaPutThread(reinterpret_cast<uint8_t*>(input) + off, selfSlotP, len, selfHandles,
                            selfSignals, selfExpected, nq, 0, sN);
        // batchSelf and overlapSelf both defer the drain; the plain path drains per chunk.
        if (!batchSelf && !overlapSelf) core::SdmaQueitThread(selfSignals + 0, selfExpected + 0, 1);
      }
      if (batchSelf) {
        core::SdmaQueitThread(selfSignals + 0, selfExpected + 0, 1);
        __threadfence_system();
      } else if (!overlapSelf) {
        __threadfence_system();
      }
      // overlapSelf: NO drain/fence here -- deferred to step (3a) so self overlaps the broadcast.
    }
    // Only the serial-front paths gate the broadcast on self-fill; overlapSelf does not.
    if (!overlapSelf) __syncthreads();
    // (2) warp t broadcasts my owned slot to the t-th OTHER PE (skip self g); distinct engines.
    if (warpId < G - 1 && laneId == 0) {
      const int t = warpId;
      const int destPos = (t < g) ? t : (t + 1);       // skip my own position g
      const int destPe = peBase + destPos * peStride;  // distinct dest PE = distinct engine
      anvil::SdmaQueueDeviceHandle** dHandles = dest->deviceHandles_d + destPe * nq;
      uint8_t* dDstBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[destPe]) + dstBaseOffset;
      uint64_t* destF = reinterpret_cast<uint64_t*>(flagsMemObj->peerPtrs[destPe]);
      for (int c = 0; c < C; ++c) {
        size_t off = static_cast<size_t>(c) * chunkBytes;
        if (off >= bytesPerPeer) break;
        size_t len = chunkBytes;
        if (off + len > bytesPerPeer) len = bytesPerPeer - off;
        const size_t myCtr = flagBase + static_cast<size_t>(c);
        uint8_t* src = reinterpret_cast<uint8_t*>(input) + off;
        uint8_t* dstPtr = dDstBase + static_cast<size_t>(g) * slotStride + off;
        void* peerCtr = static_cast<void*>(destF + myCtr);
        core::SdmaPutCopyRemoteAddThread(src, dstPtr, len, dHandles, 0, peerCtr);
      }
    }
    __syncthreads();
    // (3a) overlapSelf: drain the deferred self-fill so the self slot is landed before the fence.
    if (overlapSelf && warpId == selfWarp && laneId == 0) {
      HSAuint64* selfSignals = dest->signalPtrs + myPe * nq;
      HSAuint64* selfExpected = dest->expectSignalsPtr + myPe * nq;
      core::SdmaQueitThread(selfSignals + 0, selfExpected + 0, 1);
    }
    // (3) single completion barrier: my slot region filled by all G-1 senders.
    for (int c = static_cast<int>(threadIdx.x); c < C; c += static_cast<int>(blockDim.x)) {
      if (static_cast<size_t>(c) * chunkBytes >= bytesPerPeer) continue;
      const size_t myCtr = flagBase + static_cast<size_t>(c);
      long spin = 0;
      while (core::AtomicLoadRelaxedSystem(flags + myCtr) < fWant) {
        if (++spin > 20000000L) break;
      }
      (void)core::AtomicLoadSeqCstSystem(flags + myCtr);
    }
    __syncthreads();
    if (threadLinearId == 0) __threadfence_system();
    __syncthreads();
    return;
  }
  const bool half2x4 = ((fencedFlag & 4) != 0) && (G % 2 == 0) && (G >= 4);
  if (half2x4) {
    const int H = G / 2;
    const int half = g / H;                             // 0 or 1
    const int pos = g % H;                              // 0..H-1 within my half
    const int halfBase = peBase + half * H * peStride;  // first PE of my half
    const uint64_t p1Base = (flagVal - 1) * static_cast<uint64_t>(H - 1);
    const uint64_t p1Want = p1Base + static_cast<uint64_t>(H - 1);
    const uint64_t p2Base = (flagVal - 1) * static_cast<uint64_t>(H);
    const uint64_t p2Want = p2Base + static_cast<uint64_t>(H);
    const uint64_t k8MB = 8u * 1024u * 1024u;

    // fencedFlag bit3 (fusedOverlap): fuse phase1/phase2 -- the moment a my-half slot lands
    // (self-filled or relayed), submit both its ring forward (phase1) and its cross-half send
    // (phase2) on the same queue, so phase2 fills phase1's bubbles and the full-half barrier
    // vanishes. Same final bytes and accumulating counters (p1 at flagBase+c, p2 at
    // flagBase+C+c); stays Q=1.
    const bool fusedOverlap = ((fencedFlag & 8) != 0);
    if (fusedOverlap) {
      if (warpId == 0 && laneId == 0) {
        const int nextPos = (pos + 1) % H;
        const int nextPe = halfBase + nextPos * peStride;
        anvil::SdmaQueueDeviceHandle** nextHandles = dest->deviceHandles_d + nextPe * nq;
        uint8_t* nextDstBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[nextPe]) + dstBaseOffset;
        uint64_t* nextP1 = reinterpret_cast<uint64_t*>(flagsMemObj->peerPtrs[nextPe]);
        const int partnerPe = peBase + ((1 - half) * H + pos) * peStride;
        anvil::SdmaQueueDeviceHandle** pHandles = dest->deviceHandles_d + partnerPe * nq;
        uint8_t* pDstBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[partnerPe]) + dstBaseOffset;
        uint64_t* partnerP2 = reinterpret_cast<uint64_t*>(flagsMemObj->peerPtrs[partnerPe]);
        anvil::SdmaQueueDeviceHandle** selfHandles = dest->deviceHandles_d + myPe * nq;
        HSAuint64* selfSignals = dest->signalPtrs + myPe * nq;
        HSAuint64* selfExpected = dest->expectSignalsPtr + myPe * nq;
        const int mySlot = half * H + pos;
        // (1) self-fill all C chunks of my own slot (drained), then one fence.
        for (int c = 0; c < C; ++c) {
          size_t off = static_cast<size_t>(c) * chunkBytes;
          if (off >= bytesPerPeer) break;
          size_t len = chunkBytes;
          if (off + len > bytesPerPeer) len = bytesPerPeer - off;
          int sN = static_cast<int>((len + (k8MB - 1)) / k8MB);
          if (sN < 1) sN = 1;
          uint8_t* selfSlotP = selfDstBase + static_cast<size_t>(mySlot) * slotStride + off;
          core::SdmaPutThread(reinterpret_cast<uint8_t*>(input) + off, selfSlotP, len, selfHandles,
                              selfSignals, selfExpected, nq, 0, sN);
          core::SdmaQueitThread(selfSignals + 0, selfExpected + 0, 1);
        }
        __threadfence_system();
        // (2) fused relay(phase1) + cross-half send(phase2): walk the H ring steps; each
        //     landed slot is forwarded (s<H-1) AND shipped cross-half (all s).
        for (int c = 0; c < C; ++c) {
          size_t off = static_cast<size_t>(c) * chunkBytes;
          if (off >= bytesPerPeer) break;
          size_t len = chunkBytes;
          if (off + len > bytesPerPeer) len = bytesPerPeer - off;
          const size_t myCtr = flagBase + static_cast<size_t>(c);
          const size_t p2Ctr = flagBase + static_cast<size_t>(C) + static_cast<size_t>(c);
          void* fwdCtr = static_cast<void*>(nextP1 + myCtr);
          void* xhCtr = static_cast<void*>(partnerP2 + p2Ctr);
          for (int s = 0; s < H; ++s) {
            int gslot;
            uint8_t* src;
            if (s == 0) {
              gslot = mySlot;
              src = reinterpret_cast<uint8_t*>(input) + off;  // mySlot: source straight from input
            } else {
              const uint64_t want = p1Base + static_cast<uint64_t>(s);
              long spin = 0;
              while (core::AtomicLoadRelaxedSystem(flags + myCtr) < want) {
                if (++spin > 20000000L) break;
              }
              (void)core::AtomicLoadSeqCstSystem(flags + myCtr);
              const int kk = (pos - s + H) % H;
              gslot = half * H + kk;
              src = selfDstBase + static_cast<size_t>(gslot) * slotStride + off;  // relayed slot
            }
            // phase1: forward this slot to the ring successor (H-1 forwards; last slot not fwd'd).
            if (s < H - 1) {
              uint8_t* fwdDst = nextDstBase + static_cast<size_t>(gslot) * slotStride + off;
              core::SdmaPutCopyRemoteAddThread(src, fwdDst, len, nextHandles, 0, fwdCtr);
            }
            // phase2: cross-half send of this slot to the positional partner (all H slots).
            uint8_t* xhDst = pDstBase + static_cast<size_t>(gslot) * slotStride + off;
            core::SdmaPutCopyRemoteAddThread(src, xhDst, len, pHandles, 0, xhCtr);
          }
        }
      }
      __syncthreads();
      // block-level completion: all my-half slots relayed in (p1) AND partner's half landed (p2).
      for (int c = static_cast<int>(threadIdx.x); c < C; c += static_cast<int>(blockDim.x)) {
        if (static_cast<size_t>(c) * chunkBytes >= bytesPerPeer) continue;
        const size_t myCtr = flagBase + static_cast<size_t>(c);
        const size_t p2Ctr = flagBase + static_cast<size_t>(C) + static_cast<size_t>(c);
        long spin = 0;
        while (core::AtomicLoadRelaxedSystem(flags + myCtr) < p1Want) {
          if (++spin > 20000000L) break;
        }
        spin = 0;
        while (core::AtomicLoadRelaxedSystem(flags + p2Ctr) < p2Want) {
          if (++spin > 20000000L) break;
        }
        (void)core::AtomicLoadSeqCstSystem(flags + myCtr);
        (void)core::AtomicLoadSeqCstSystem(flags + p2Ctr);
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      __syncthreads();
      return;
    }

    // fencedFlag bit6 (spreadP1): flat within-half broadcast instead of the Q=1 4-ring relay
    // -- each PE sends its owned slot (from input, no relay dep) to the H-1 other half-PEs =
    // H-1 distinct engines on queue 0. Counter: H-1 senders +1 => want = p1Base+(H-1),
    // identical to the ring. Default OFF (bit6).
    const bool spreadP1 = ((fencedFlag & 64) != 0);
    // fencedFlag bit7 (multiWarp): give each dest its own submitter warp (warp t -> dest
    // position t = distinct engine) so all H doorbells fire together instead of serialized
    // from warp0. Queue-0-only per dest, distinct dest per warp (no same-queue race);
    // counters unchanged. Default OFF (bit7).
    const bool multiWarp = ((fencedFlag & 128) != 0);
    // fencedFlag bit8 (concP, requires multiWarp): run phase1 (warps [0,H)) and phase2 (warps
    // [H,2H)) concurrently -- both source from input independently -- dropping the intermediate
    // p1 barrier and waiting p1Want and p2Want once at the end. Peak incast rises to H-1+H
    // (helps small, may throttle large). Default OFF.
    const bool concP = ((fencedFlag & 256) != 0) && multiWarp;
    const int mySlotP1mw = half * H + pos;
    // ---- PHASE 1: self-fill my own slot, then relay/broadcast my half's H slots.
    if (spreadP1 && multiWarp && H > 1) {
      if (warpId == 0 && laneId == 0) {
        anvil::SdmaQueueDeviceHandle** selfHandles = dest->deviceHandles_d + myPe * nq;
        HSAuint64* selfSignals = dest->signalPtrs + myPe * nq;
        HSAuint64* selfExpected = dest->expectSignalsPtr + myPe * nq;
        for (int c = 0; c < C; ++c) {
          size_t off = static_cast<size_t>(c) * chunkBytes;
          if (off >= bytesPerPeer) break;
          size_t len = chunkBytes;
          if (off + len > bytesPerPeer) len = bytesPerPeer - off;
          int sN = static_cast<int>((len + (k8MB - 1)) / k8MB);
          if (sN < 1) sN = 1;
          uint8_t* selfSlotP = selfDstBase + static_cast<size_t>(mySlotP1mw) * slotStride + off;
          core::SdmaPutThread(reinterpret_cast<uint8_t*>(input) + off, selfSlotP, len, selfHandles,
                              selfSignals, selfExpected, nq, 0, sN);
          core::SdmaQueitThread(selfSignals + 0, selfExpected + 0, 1);
        }
        __threadfence_system();
      }
      __syncthreads();
      // warp t (t != pos) broadcasts my owned slot to same-half PE at position t (distinct engine).
      if (warpId < H && warpId != pos && laneId == 0) {
        const int t = warpId;
        const int destPe = halfBase + t * peStride;
        anvil::SdmaQueueDeviceHandle** dHandles = dest->deviceHandles_d + destPe * nq;
        uint8_t* dDstBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[destPe]) + dstBaseOffset;
        uint64_t* destP1 = reinterpret_cast<uint64_t*>(flagsMemObj->peerPtrs[destPe]);
        for (int c = 0; c < C; ++c) {
          size_t off = static_cast<size_t>(c) * chunkBytes;
          if (off >= bytesPerPeer) break;
          size_t len = chunkBytes;
          if (off + len > bytesPerPeer) len = bytesPerPeer - off;
          const size_t myCtr = flagBase + static_cast<size_t>(c);
          uint8_t* src = reinterpret_cast<uint8_t*>(input) + off;
          uint8_t* dstPtr = dDstBase + static_cast<size_t>(mySlotP1mw) * slotStride + off;
          void* peerCtr = static_cast<void*>(destP1 + myCtr);
          core::SdmaPutCopyRemoteAddThread(src, dstPtr, len, dHandles, 0, peerCtr);
        }
      }
    } else if (warpId == 0 && laneId == 0) {
      const int nextPos = (pos + 1) % H;
      const int nextPe = halfBase + nextPos * peStride;
      anvil::SdmaQueueDeviceHandle** nextHandles = dest->deviceHandles_d + nextPe * nq;
      uint8_t* nextDstBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[nextPe]) + dstBaseOffset;
      anvil::SdmaQueueDeviceHandle** selfHandles = dest->deviceHandles_d + myPe * nq;
      HSAuint64* selfSignals = dest->signalPtrs + myPe * nq;
      HSAuint64* selfExpected = dest->expectSignalsPtr + myPe * nq;
      uint64_t* nextP1 = reinterpret_cast<uint64_t*>(flagsMemObj->peerPtrs[nextPe]);
      const int mySlot = half * H + pos;  // global slot this PE owns
      for (int c = 0; c < C; ++c) {
        size_t off = static_cast<size_t>(c) * chunkBytes;
        if (off >= bytesPerPeer) break;
        size_t len = chunkBytes;
        if (off + len > bytesPerPeer) len = bytesPerPeer - off;
        int sN = static_cast<int>((len + (k8MB - 1)) / k8MB);
        if (sN < 1) sN = 1;
        uint8_t* selfSlotP = selfDstBase + static_cast<size_t>(mySlot) * slotStride + off;
        core::SdmaPutThread(reinterpret_cast<uint8_t*>(input) + off, selfSlotP, len, selfHandles,
                            selfSignals, selfExpected, nq, 0, sN);
        core::SdmaQueitThread(selfSignals + 0, selfExpected + 0, 1);
      }
      __threadfence_system();
      if (spreadP1 && H > 1) {
        // flat within-half broadcast: my owned slot -> the H-1 other PEs of my half.
        for (int c = 0; c < C; ++c) {
          size_t off = static_cast<size_t>(c) * chunkBytes;
          if (off >= bytesPerPeer) break;
          size_t len = chunkBytes;
          if (off + len > bytesPerPeer) len = bytesPerPeer - off;
          const size_t myCtr = flagBase + static_cast<size_t>(c);
          uint8_t* src = reinterpret_cast<uint8_t*>(input) + off;  // owned slot from input
          for (int t = 0; t < H; ++t) {
            if (t == pos) continue;                      // skip self
            const int destPe = halfBase + t * peStride;  // each = distinct engine
            anvil::SdmaQueueDeviceHandle** dHandles = dest->deviceHandles_d + destPe * nq;
            uint8_t* dDstBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[destPe]) + dstBaseOffset;
            uint64_t* destP1 = reinterpret_cast<uint64_t*>(flagsMemObj->peerPtrs[destPe]);
            uint8_t* dstPtr = dDstBase + static_cast<size_t>(mySlot) * slotStride + off;
            void* peerCtr = static_cast<void*>(destP1 + myCtr);
            core::SdmaPutCopyRemoteAddThread(src, dstPtr, len, dHandles, 0, peerCtr);
          }
        }
      } else if (H > 1) {
        for (int c = 0; c < C; ++c) {
          size_t off = static_cast<size_t>(c) * chunkBytes;
          if (off >= bytesPerPeer) break;
          size_t len = chunkBytes;
          if (off + len > bytesPerPeer) len = bytesPerPeer - off;
          const size_t myCtr = flagBase + static_cast<size_t>(c);
          for (int s = 0; s < H - 1; ++s) {
            const int kk = (pos - s + H) % H;  // within-half slot to forward
            const int gslot = half * H + kk;   // global slot index
            uint8_t* src;
            if (s == 0) {
              src = reinterpret_cast<uint8_t*>(input) + off;
            } else {
              const uint64_t want = p1Base + static_cast<uint64_t>(s);
              long spin = 0;
              while (core::AtomicLoadRelaxedSystem(flags + myCtr) < want) {
                if (++spin > 20000000L) break;
              }
              (void)core::AtomicLoadSeqCstSystem(flags + myCtr);
              src = selfDstBase + static_cast<size_t>(gslot) * slotStride + off;
            }
            uint8_t* dstPtr = nextDstBase + static_cast<size_t>(gslot) * slotStride + off;
            void* peerCtr = static_cast<void*>(nextP1 + myCtr);
            core::SdmaPutCopyRemoteAddThread(src, dstPtr, len, nextHandles, 0, peerCtr);
          }
        }
      }
    }
    __syncthreads();
    // concP: skip the intermediate p1-completion barrier -- phase2 doesn't depend on phase1;
    // both are waited once at the final barrier below.
    if (!concP) {
      if (H > 1) {
        for (int c = static_cast<int>(threadIdx.x); c < C; c += static_cast<int>(blockDim.x)) {
          if (static_cast<size_t>(c) * chunkBytes >= bytesPerPeer) continue;
          const size_t myCtr = flagBase + static_cast<size_t>(c);
          long spin = 0;
          while (core::AtomicLoadRelaxedSystem(flags + myCtr) < p1Want) {
            if (++spin > 20000000L) break;
          }
          (void)core::AtomicLoadSeqCstSystem(flags + myCtr);
        }
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      __syncthreads();
    }

    // ---- PHASE 2: pairwise cross-half exchange. Send my half's H slots to partner g^H;
    // partner symmetrically sends its half's H slots to me. One accumulating counter per
    // chunk (partner ADDs +1 after each of my H slots lands => want = base + H).
    // fencedFlag bit4 (parallelP2): phase2's H cross-half slot copies are independent (same
    // partner, disjoint dst slots, no relay dep), so stripe them across QP=min(H,nq) warps,
    // warp w owning queue w and slots {j : j%QP==w} (one warp/queue, no same-queue race).
    // Accumulating +1 per slot => want=p2Base+H. Default OFF (bit4).
    // fencedFlag bit5 (spreadP2): engine=f(src,dst), so a 2nd queue index to the same partner
    // is a 2nd channel to one engine (stalls). Spread across destinations instead: each PE
    // broadcasts its owned slot (from input) to the other half's H PEs = H distinct engines on
    // queue 0. One +1 per sender => want=p2Base+H, unchanged. Default OFF (bit5).
    const bool spreadP2 = ((fencedFlag & 32) != 0);
    const bool parallelP2 = ((fencedFlag & 16) != 0) && nq > 1;
    if (spreadP2 && multiWarp) {
      // multiWarp dest-spread phase2: warp t sends my owned slot to the other-half PE at
      // position t (H distinct engines, queue 0). With concP, phase2 on warps [H,2H) to
      // overlap phase1 (warps [0,H)).
      const int p2lo = concP ? H : 0;
      if (warpId >= p2lo && warpId < p2lo + H && laneId == 0) {
        const int otherBase = peBase + ((1 - half) * H) * peStride;
        const int mySlot = half * H + pos;
        const int t = warpId - p2lo;
        const int destPe = otherBase + t * peStride;
        anvil::SdmaQueueDeviceHandle** dHandles = dest->deviceHandles_d + destPe * nq;
        uint8_t* dDstBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[destPe]) + dstBaseOffset;
        uint64_t* destP2 = reinterpret_cast<uint64_t*>(flagsMemObj->peerPtrs[destPe]);
        for (int c = 0; c < C; ++c) {
          size_t off = static_cast<size_t>(c) * chunkBytes;
          if (off >= bytesPerPeer) break;
          size_t len = chunkBytes;
          if (off + len > bytesPerPeer) len = bytesPerPeer - off;
          const size_t p2Ctr = flagBase + static_cast<size_t>(C) + static_cast<size_t>(c);
          uint8_t* src = reinterpret_cast<uint8_t*>(input) + off;
          uint8_t* dstPtr = dDstBase + static_cast<size_t>(mySlot) * slotStride + off;
          void* peerCtr = static_cast<void*>(destP2 + p2Ctr);
          core::SdmaPutCopyRemoteAddThread(src, dstPtr, len, dHandles, 0, peerCtr);
        }
      }
    } else if (spreadP2) {
      if (warpId == 0 && laneId == 0) {
        const int otherBase = peBase + ((1 - half) * H) * peStride;  // first PE of other half
        const int mySlot = half * H + pos;                           // global slot this PE owns
        for (int c = 0; c < C; ++c) {
          size_t off = static_cast<size_t>(c) * chunkBytes;
          if (off >= bytesPerPeer) break;
          size_t len = chunkBytes;
          if (off + len > bytesPerPeer) len = bytesPerPeer - off;
          const size_t p2Ctr = flagBase + static_cast<size_t>(C) + static_cast<size_t>(c);
          uint8_t* src = reinterpret_cast<uint8_t*>(input) + off;  // owned slot from input
          for (int t = 0; t < H; ++t) {
            const int destPe = otherBase + t * peStride;  // each = distinct engine
            anvil::SdmaQueueDeviceHandle** dHandles = dest->deviceHandles_d + destPe * nq;
            uint8_t* dDstBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[destPe]) + dstBaseOffset;
            uint64_t* destP2 = reinterpret_cast<uint64_t*>(flagsMemObj->peerPtrs[destPe]);
            uint8_t* dstPtr = dDstBase + static_cast<size_t>(mySlot) * slotStride + off;
            void* peerCtr = static_cast<void*>(destP2 + p2Ctr);
            core::SdmaPutCopyRemoteAddThread(src, dstPtr, len, dHandles, 0, peerCtr);
          }
        }
      }
    } else if (parallelP2) {
      const int QP = H < nq ? H : nq;
      if (warpId < QP && laneId == 0) {
        const int w = warpId;
        const int partnerPe = peBase + ((1 - half) * H + pos) * peStride;
        anvil::SdmaQueueDeviceHandle** pHandles = dest->deviceHandles_d + partnerPe * nq;
        uint8_t* pDstBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[partnerPe]) + dstBaseOffset;
        uint64_t* partnerP2 = reinterpret_cast<uint64_t*>(flagsMemObj->peerPtrs[partnerPe]);
        for (int c = 0; c < C; ++c) {
          size_t off = static_cast<size_t>(c) * chunkBytes;
          if (off >= bytesPerPeer) break;
          size_t len = chunkBytes;
          if (off + len > bytesPerPeer) len = bytesPerPeer - off;
          const size_t p2Ctr = flagBase + static_cast<size_t>(C) + static_cast<size_t>(c);
          for (int j = w; j < H; j += QP) {
            const int gslot = half * H + j;  // my half's slots -> partner needs
            uint8_t* src = selfDstBase + static_cast<size_t>(gslot) * slotStride + off;
            uint8_t* dstPtr = pDstBase + static_cast<size_t>(gslot) * slotStride + off;
            void* peerCtr = static_cast<void*>(partnerP2 + p2Ctr);
            core::SdmaPutCopyRemoteAddThread(src, dstPtr, len, pHandles, static_cast<uint32_t>(w),
                                             peerCtr);
          }
        }
      }
    } else if (warpId == 0 && laneId == 0) {
      const int partnerPe = peBase + ((1 - half) * H + pos) * peStride;
      anvil::SdmaQueueDeviceHandle** pHandles = dest->deviceHandles_d + partnerPe * nq;
      uint8_t* pDstBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[partnerPe]) + dstBaseOffset;
      uint64_t* partnerP2 = reinterpret_cast<uint64_t*>(flagsMemObj->peerPtrs[partnerPe]);
      for (int c = 0; c < C; ++c) {
        size_t off = static_cast<size_t>(c) * chunkBytes;
        if (off >= bytesPerPeer) break;
        size_t len = chunkBytes;
        if (off + len > bytesPerPeer) len = bytesPerPeer - off;
        const size_t p2Ctr = flagBase + static_cast<size_t>(C) + static_cast<size_t>(c);
        for (int j = 0; j < H; ++j) {
          const int gslot = half * H + j;  // my half's slots -> partner needs
          uint8_t* src = selfDstBase + static_cast<size_t>(gslot) * slotStride + off;
          uint8_t* dstPtr = pDstBase + static_cast<size_t>(gslot) * slotStride + off;
          void* peerCtr = static_cast<void*>(partnerP2 + p2Ctr);
          core::SdmaPutCopyRemoteAddThread(src, dstPtr, len, pHandles, 0, peerCtr);
        }
      }
    }
    __syncthreads();
    for (int c = static_cast<int>(threadIdx.x); c < C; c += static_cast<int>(blockDim.x)) {
      if (static_cast<size_t>(c) * chunkBytes >= bytesPerPeer) continue;
      const size_t p2Ctr = flagBase + static_cast<size_t>(C) + static_cast<size_t>(c);
      long spin = 0;
      while (core::AtomicLoadRelaxedSystem(flags + p2Ctr) < p2Want) {
        if (++spin > 20000000L) break;
      }
      (void)core::AtomicLoadSeqCstSystem(flags + p2Ctr);
      // concP: the intermediate p1 barrier was skipped, so also wait phase1 here.
      if (concP && H > 1) {
        const size_t myCtr = flagBase + static_cast<size_t>(c);
        spin = 0;
        while (core::AtomicLoadRelaxedSystem(flags + myCtr) < p1Want) {
          if (++spin > 20000000L) break;
        }
        (void)core::AtomicLoadSeqCstSystem(flags + myCtr);
      }
    }
    __syncthreads();
    if (threadLinearId == 0) __threadfence_system();
    __syncthreads();
    return;
  }
  if (warpId < Q && laneId == 0) {
    const int q = warpId;  // this warp's dedicated SDMA queue / channel
    const int nextPos = (g + 1) % G;
    const int nextPe = peBase + nextPos * peStride;
    anvil::SdmaQueueDeviceHandle** nextHandles = dest->deviceHandles_d + nextPe * nq;
    HSAuint64* nextSignals = dest->signalPtrs + nextPe * nq;
    HSAuint64* nextExpected = dest->expectSignalsPtr + nextPe * nq;
    uint8_t* nextDstBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[nextPe]) + dstBaseOffset;

    anvil::SdmaQueueDeviceHandle** selfHandles = dest->deviceHandles_d + myPe * nq;
    HSAuint64* selfSignals = dest->signalPtrs + myPe * nq;
    HSAuint64* selfExpected = dest->expectSignalsPtr + myPe * nq;
    uint8_t* selfSlot = selfDstBase + static_cast<size_t>(g) * slotStride;

    // Self-fill own slot g from input (SDMA self-copy), this warp's chunks on queue q.
    for (int c = q; c < C; c += Q) {
      size_t off = static_cast<size_t>(c) * chunkBytes;
      if (off >= bytesPerPeer) break;
      size_t len = chunkBytes;
      if (off + len > bytesPerPeer) len = bytesPerPeer - off;
      // Cap each SDMA copy descriptor at <=8 MiB (CDNA3 microcode faults on a single >8MB
      // COPY_LINEAR): split len>8MB into ceil(len/8MB) sub-descriptors on the same queue,
      // one trailing atomic + drain. len<=8MB unchanged.
      int selfNdesc = static_cast<int>((len + (8u * 1024u * 1024u - 1)) / (8u * 1024u * 1024u));
      if (selfNdesc < 1) selfNdesc = 1;
      core::SdmaPutThread(reinterpret_cast<uint8_t*>(input) + off, selfSlot + off, len, selfHandles,
                          selfSignals, selfExpected, nq, static_cast<uint32_t>(q), selfNdesc);
      core::SdmaQueitThread(selfSignals + q, selfExpected + q, 1);
    }
    __threadfence_system();

    if (G > 1) {
      // Monotonic per-link step-counter completion, replacing per-(c,k) cross-PE AMO_SET
      // flags that drop under multi-warp load. Each PE has one incoming ring link delivering
      // slots in strict step order; keep one counter per chunk at flagBase+c that the
      // predecessor atomic-adds +1 to after each drained step -- accumulating and monotone,
      // so no update can be lost. Per-op base = (flagVal-1)*(G-1); step s waits counter >=
      // base+s.
      const uint64_t ringBase = (flagVal - 1) * static_cast<uint64_t>(G - 1);
      uint64_t oneAdd = 1;
      // Each chunk is a self-contained pipeline over the ring's G-1 steps, run on this warp's
      // queue q so the Q warps advance Q chunks concurrently.
      for (int c = q; c < C; c += Q) {
        size_t off = static_cast<size_t>(c) * chunkBytes;
        if (off >= bytesPerPeer) break;
        size_t len = chunkBytes;
        if (off + len > bytesPerPeer) len = bytesPerPeer - off;
        const size_t myCtr = flagBase + static_cast<size_t>(c);  // my incoming-link counter
        for (int s = 0; s < G - 1; ++s) {
          const int k = (g - s + G) % G;
          uint8_t* src;
          if (s == 0) {
            src = reinterpret_cast<uint8_t*>(input) + off;  // own shard from input
          } else {
            // Wait until my predecessor has delivered >= s slots to me on this link.
            const uint64_t want = ringBase + static_cast<uint64_t>(s);
            long spin = 0;
            bool stuck = false;
            while (core::AtomicLoadRelaxedSystem(flags + myCtr) < want) {
              if (++spin > 20000000L) {
                stuck = true;
                break;
              }
            }
            if (stuck) {
              printf("[RINGSTUCK] SEND myPe=%d g=%d s=%d c=%d ctr=%llu got=%llu want=%llu\n", myPe,
                     g, s, c, static_cast<unsigned long long>(myCtr),
                     static_cast<unsigned long long>(core::AtomicLoadRelaxedSystem(flags + myCtr)),
                     static_cast<unsigned long long>(want));
            }
            (void)core::AtomicLoadSeqCstSystem(flags + myCtr);
            src = selfDstBase + static_cast<size_t>(k) * slotStride + off;
          }
          uint8_t* dstPtr = nextDstBase + static_cast<size_t>(k) * slotStride + off;
          // Fused copy + remote counter +1 on the same SDMA queue, one doorbell. A CU P2P
          // atomic on flagsMemObj->peerPtrs faults (engine-addressed peerPtrs) and spins
          // forever, so the SDMA engine posts the ADD64: COPY_LINEAR then SDMA_PKT_ATOMIC
          // ADD64(+1) to nextPe's counter, FIFO-ordered so the peer sees the +1 strictly
          // after the bytes land (flag-after-bytes). ADD accumulates so no +1 is lost.
          // peerCtr = nextPe's copy of my incoming-link counter slot.
          uint64_t* nextCtrBase = reinterpret_cast<uint64_t*>(flagsMemObj->peerPtrs[nextPe]);
          void* peerCtr = static_cast<void*>(nextCtrBase + myCtr);
          (void)oneAdd;
          core::SdmaPutCopyRemoteAddThread(src, dstPtr, len, nextHandles, static_cast<uint32_t>(q),
                                           peerCtr);
        }
      }
    }
  }
  __syncthreads();

  if (G > 1) {
    // Completion: each PE waits until its incoming-link counter reaches base+(G-1) for
    // every chunk (predecessor delivered ALL G-1 foreign slots => my whole buffer is
    // full; slot g I self-filled above). C counters, not C*G flags.
    const uint64_t ringBase = (flagVal - 1) * static_cast<uint64_t>(G - 1);
    const uint64_t want = ringBase + static_cast<uint64_t>(G - 1);
    for (int c = static_cast<int>(threadIdx.x); c < C; c += static_cast<int>(blockDim.x)) {
      if (static_cast<size_t>(c) * chunkBytes >= bytesPerPeer) continue;
      const size_t myCtr = flagBase + static_cast<size_t>(c);
      long spin = 0;
      bool stuck = false;
      while (core::AtomicLoadRelaxedSystem(flags + myCtr) < want) {
        if (++spin > 20000000L) {
          stuck = true;
          break;
        }
      }
      if (stuck) {
        printf("[RINGSTUCK] COMPL myPe=%d g=%d c=%d ctr=%llu got=%llu want=%llu\n", myPe, g, c,
               static_cast<unsigned long long>(myCtr),
               static_cast<unsigned long long>(core::AtomicLoadRelaxedSystem(flags + myCtr)),
               static_cast<unsigned long long>(want));
      }
      (void)core::AtomicLoadSeqCstSystem(flags + myCtr);
    }
    __syncthreads();
    if (threadLinearId == 0) __threadfence_system();
    __syncthreads();
  }
}

template <typename T>
__device__ void OneShotAllGatherSdmaSubGroupKernel_body(
    int myPe, int npes, int groupSize, int groupPos, int peBase, int peStride, T* input,
    const application::SymmMemObjPtr dstMemObj, const application::SymmMemObjPtr flagsMemObj,
    size_t elementCount, size_t dstBaseOffset = 0, size_t dstSlotStrideBytes = 0,
    uint64_t flagVal = 1, bool blockLocal = false, size_t flagBase = 0,
    int pushPeerLo = 0, int pushPeerHi = -1) {
  (void)npes;
  if (elementCount == 0 || groupSize <= 0) {
    return;
  }
  // Restrict the PUSH half to peer-columns [pushPeerLo, pHi). The completion WAIT below is
  // UNCHANGED (full [0,G)) -- it gates on senders INTO this rank, independent of push
  // targets. Default (0, <0) => full-mesh push.
  const int pLo = (pushPeerLo > 0) ? pushPeerLo : 0;
  const int pHi = (pushPeerHi >= 0 && pushPeerHi < groupSize) ? pushPeerHi : groupSize;
  T* __restrict__ inputData = input;
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);

  const size_t threadLinearId =
      blockLocal ? static_cast<size_t>(threadIdx.x)
                 : static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;

  const size_t bytesPerElement = sizeof(T);
  const size_t bytesPerPeer = elementCount * bytesPerElement;
  // Per-peer destination slot stride. Default (0) packs slots contiguously (stride ==
  // copy size), preserving the original layout; non-zero enables the chunked pipeline.
  const size_t slotStride = dstSlotStrideBytes != 0 ? dstSlotStrideBytes : bytesPerPeer;

  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;

  // Each member warp pushes this PE's shard into slot ``groupPos`` of the warpId-th
  // member's buffer (SDMA over XGMI). G warps drive G distinct peer links; one queue per
  // peer is already bandwidth-bound, so per-peer multi-queue gives no speedup.
  if (warpId >= pLo && warpId < pHi && laneId == 0) {
    int remotePe = peBase + warpId * peStride;
    size_t destByteOffset = static_cast<size_t>(groupPos) * slotStride;
    application::SymmMemObjPtr dest = dstMemObj;
    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(inputData);
    uint8_t* dstPtr =
        reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset + destByteOffset;
    anvil::SdmaQueueDeviceHandle** devicehandles =
        dest->deviceHandles_d + remotePe * dest->sdmaNumQueue;
    HSAuint64* signals = dest->signalPtrs + remotePe * dest->sdmaNumQueue;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * dest->sdmaNumQueue;
    core::SdmaPutThread(srcPtr, dstPtr, bytesPerPeer, devicehandles, signals, expectedSignals,
                        dest->sdmaNumQueue, 0);
  }

  // Per-peer completion tail: each warp drains its own peer's queue then bumps that peer's
  // flag -- parallel across warps.
  if (warpId >= pLo &&
      warpId < pHi && laneId == 0) {
    int remotePe = peBase + warpId * peStride;
    SubGroupSdmaDrainPe(remotePe, dstMemObj);
    // Sender-side completion fence: ShmemQuietThread drains the submission queue but does
    // not order the raw SDMA data writes vs the flag AMO at system scope. Without it the
    // receiver can observe the flag before the pushed bytes are globally ordered.
    __threadfence_system();
    shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
        flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
        core::atomicType::AMO_SET, remotePe, 0);
  }
  __syncthreads();

  // Completion wait: thread t seq-cst SYSTEM-acquires sender-position t's flag (agent-wide
  // acquire => the peer's SDMA writes become coherently visible), then one block barrier +
  // one thread-0 __threadfence_system. System-scope acquire is required: the flag and its
  // data are written by a remote peer GPU (a different HSA agent) via SDMA, so an
  // agent-scope relaxed load gives no cross-agent happens-before.
  {
    int senderPos = static_cast<int>(threadIdx.x);
    if (senderPos < groupSize && senderPos != groupPos) {
      int spinCount = 0;
      bool warned = false;
      // Light-spin: relaxed system spin + one seq-cst SYSTEM acquire on exit (see above).
      while (core::AtomicLoadRelaxedSystem(flags + flagBase + senderPos) < flagVal) {
        ++spinCount;
        if (!warned && spinCount > 10000000) {
          printf("PE %d: Slow wait for sub-group pos %d (still waiting)\n", myPe, senderPos);
          warned = true;
        }
      }
      (void)core::AtomicLoadSeqCstSystem(flags + flagBase + senderPos);
    }
    __syncthreads();
    if (threadLinearId == 0) __threadfence_system();
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Push-only sub-group reassembly (deadlock-free, parallel)
// ---------------------------------------------------------------------------
// The remote-block reassembly is an intra-node sub-group gather where the output slots
// are DISJOINT per rank -- each rank only WRITES its own column, never READS a peer's --
// so the cross-rank flag WAIT the collective gather does is not needed for the push;
// coupling push+wait inside every (block,remote) gather is what deadlocks when several
// reassembly blocks run concurrently (each waits on peers whose matching block may not
// be co-resident => circular stall).
//
// This does the WRITE half ONLY: warp ``w`` SDMA-pushes this rank's slice into slot
// ``groupPos`` of member ``w``'s output, quiets ONLY its own queue ``qId`` (never
// drains-all, so concurrent blocks on other queues are unaffected), then bumps flag
// ``flagBase+groupPos`` on member ``w``. It NEVER waits on a peer, so any number run
// concurrently without deadlock; a SINGLE completion reader later waits every flag AFTER
// all pushes issue. ``qId`` MUST be distinct across blocks targeting the same peer so the
// per-queue signal counter is never raced.
template <typename T>
__device__ void OneShotSubGroupPushOnly_body(
    int groupSize, int groupPos, int peBase, int peStride, T* input,
    const application::SymmMemObjPtr dstMemObj, const application::SymmMemObjPtr flagsMemObj,
    size_t elementCount, size_t dstBaseOffset, size_t dstSlotStrideBytes, uint64_t flagVal,
    size_t flagBase, int qId, int deepSqPhase = 0,
    int pushPeerLo = 0, int pushPeerHi = -1, int pushFlag = 1) {
  // Intra reassembly deep-SQ phase split (see HierReasmDeepSqOn). deepSqPhase==0 = single-
  // shot (submit->drain->fence->flag). ==1 = submit-only (push copy + bump expectedSignals[q],
  // no drain/fence/flag) so a worker feeds the copy engine back-to-back. ==2 = drain+flag-only:
  // one quiet to the accumulated expected count covers every phase-1 submit on this queue
  // (SQ FIFO, signal monotonic), then fence + flag after the drain, so the flag never precedes
  // its bytes.
  if (elementCount == 0 || groupSize <= 0) {
    return;
  }
  const size_t threadLinearId = static_cast<size_t>(threadIdx.x);  // block-local
  const size_t bytesPerElement = sizeof(T);
  const size_t bytesPerPeer = elementCount * bytesPerElement;
  const size_t slotStride = dstSlotStrideBytes != 0 ? dstSlotStrideBytes : bytesPerPeer;
  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;
  // Restrict this push to peer-columns [pLoP, pHiP). Default (0, -1) => full [0,groupSize).
  const int pLoP = (pushPeerLo > 0) ? pushPeerLo : 0;
  const int pHiP = (pushPeerHi >= 0 && pushPeerHi < groupSize) ? pushPeerHi : groupSize;
  const int effWarp = warpId;

  if (warpId >= pLoP && warpId < pHiP && laneId == 0) {
    int remotePe = peBase + effWarp * peStride;
    size_t destByteOffset = static_cast<size_t>(groupPos) * slotStride;
    application::SymmMemObjPtr dest = dstMemObj;
    const int nq = dest->sdmaNumQueue > 0 ? static_cast<int>(dest->sdmaNumQueue) : 1;
    const int q = (qId >= 0) ? (qId % nq) : 0;
    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(input);
    uint8_t* dstPtr =
        reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset + destByteOffset;
    anvil::SdmaQueueDeviceHandle** devicehandles = dest->deviceHandles_d + remotePe * nq;
    HSAuint64* signals = dest->signalPtrs + remotePe * nq;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * nq;
    // Push this rank's column on its OWN queue ``q``. deepSqPhase==2 is DRAIN-only
    // (the copy was already submitted in phase 1).
    if (deepSqPhase != 2) {
      core::SdmaPutThread(srcPtr, dstPtr, bytesPerPeer, devicehandles, signals, expectedSignals,
                          nq, q);
    }
    // deepSqPhase==1 is SUBMIT-only: skip drain/fence/flag.
    if (deepSqPhase != 1) {
      // Drain ONLY queue ``q`` so the copy has landed before the flag fires, without
      // touching other blocks' queues. In phase 2 expectedSignals[q] holds the accumulated
      // phase-1 count so this one wait covers them all.
      core::SdmaQueitThread(signals + q, expectedSignals + q, 1);
      // Sender-side fence (see collective body above): order the pushed bytes BEFORE the
      // flag AMO becomes peer-visible.
      __threadfence_system();
      // Flag AMO is a DIRECT peer-memory CAS (P2P), not an SDMA-queue signal, so it never
      // races the per-queue counter. pushFlag==0 (non-last tile of the read-coalescing tile
      // loop) suppresses only the flag -- the copy is still drained+fenced, and the flag
      // fires only on the last tile after all bytes drained.
      if (pushFlag) {
        shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
            flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal,
            8, core::atomicType::AMO_SET, remotePe, 0);
      }
    }
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// Fused hierarchical param-contiguous SubGroup gather (ONE launch)
// ---------------------------------------------------------------------------
// Warp ``w`` drives member ``w``: this PE (position ``g``) pushes, for every node block
// ``m`` and every param split ``s``, its E_s-element sub-slice into the member's output
// at param-contiguous element offset ``O_s*W + (m*G+g)*E_s``. Same subgroup flags as the
// direct gather: bump slot ``g`` on each member, then wait all G members.
template <typename T>
__device__ void OneShotAllGatherSdmaSubGroupParamContiguousKernel_body(
    int myPe, int npes, int groupSize, int groupPos, int peBase, int peStride, int numBlocks,
    int firstBlock, T* input, const application::SymmMemObjPtr dstMemObj,
    const application::SymmMemObjPtr flagsMemObj, size_t blockStrideElems, size_t worldSize,
    size_t dstBaseOffset, uint64_t flagVal, const size_t* splitSizes, const size_t* splitOffsets,
    size_t splitCount) {
  (void)npes;
  if (groupSize <= 0 || numBlocks <= 0 || splitCount == 0 || splitSizes == nullptr ||
      splitOffsets == nullptr) {
    return;
  }

  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;
  const size_t bytesPerElement = sizeof(T);
  const size_t G = static_cast<size_t>(groupSize);
  const size_t g = static_cast<size_t>(groupPos);

  // One warp per destination member; loop node blocks then param splits, all
  // written to the same param-contiguous offset (constant across members).
  if (warpId < groupSize && laneId == 0) {
    int remotePe = peBase + warpId * peStride;
    application::SymmMemObjPtr dest = dstMemObj;
    anvil::SdmaQueueDeviceHandle** devicehandles =
        dest->deviceHandles_d + remotePe * dest->sdmaNumQueue;
    HSAuint64* signals = dest->signalPtrs + remotePe * dest->sdmaNumQueue;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * dest->sdmaNumQueue;
    uint8_t* dstBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset;

    for (int i = 0; i < numBlocks; ++i) {
      const int m = firstBlock + i;                     // global node block
      const size_t r = static_cast<size_t>(m) * G + g;  // global rank
      uint8_t* blkSrc = reinterpret_cast<uint8_t*>(input) +
                        static_cast<size_t>(i) * blockStrideElems * bytesPerElement;
      for (size_t s = 0; s < splitCount; ++s) {
        size_t E = splitSizes[s];
        if (E == 0) {
          continue;
        }
        size_t O = splitOffsets[s];
        size_t outElemOffset = O * worldSize + r * E;
        uint8_t* srcPtr = blkSrc + O * bytesPerElement;
        uint8_t* dstPtr = dstBase + outElemOffset * bytesPerElement;
        core::SdmaPutThread(srcPtr, dstPtr, E * bytesPerElement, devicehandles, signals,
                            expectedSignals, dest->sdmaNumQueue, 0);
      }
    }
  }
  // Fence all warps' SDMA puts before any warp quiets + bumps its flag.
  __syncthreads();

  if (warpId < groupSize && laneId == 0) {
    int remotePe = peBase + warpId * peStride;
    SubGroupSdmaDrainPe(remotePe, dstMemObj);
    shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
        flagsMemObj, static_cast<size_t>(groupPos) * sizeof(uint64_t), &flagVal, 8,
        core::atomicType::AMO_SET, remotePe, 0);
  }
  __syncthreads();

  // Completion wait (see subgroup gather note). Thread 0 acquires every peer flag.
  if (threadLinearId == 0) {
    for (int senderPos = 0; senderPos < groupSize; ++senderPos) {
      if (senderPos == groupPos) {
        continue;
      }
      // Light-spin: relaxed system spin + one seq-cst SYSTEM acquire on exit.
      while (core::AtomicLoadRelaxedSystem(flags + senderPos) < flagVal) {
      }
      (void)core::AtomicLoadSeqCstSystem(flags + senderPos);
    }
    __threadfence_system();
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// Sub-group intra-node SDMA broadcast
// ---------------------------------------------------------------------------
// Root (group position 0 == global PE ``peBase``) SDMA-copies its whole
// ``elementCount``-lane buffer (over XGMI) into every member's ``dstMemObj`` --
// including itself -- so every member of the arithmetic sub-group
// ``{peBase, peBase+peStride, ..., peBase+(groupSize-1)*peStride}`` ends with the full
// buffer. Intra-node placement phase of the leader-only hierarchical AllGather (leader
// rings inter-node into staging, then broadcasts, crossing the NIC once per node-block).
// Root warp ``w`` handles member ``w``: one SDMA put + quiet + single-slot flag bump;
// non-root members spin on flag slot 0. Single flag slot with a per-call token => calls
// stay race-free without a reset.
template <typename T>
__device__ void OneShotBroadcastSdmaSubGroupKernel_body(
    int myPe, int groupSize, int groupPos, int peBase, int peStride, T* input,
    const application::SymmMemObjPtr dstMemObj, const application::SymmMemObjPtr flagsMemObj,
    size_t elementCount, size_t dstBaseOffset = 0, uint64_t flagVal = 1) {
  if (elementCount == 0 || groupSize <= 0) {
    return;
  }

  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  const size_t bytesTotal = elementCount * sizeof(T);

  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;

  if (groupPos == 0) {
    // Root: push the whole buffer to every member (including self), one warp
    // per member over a distinct peer XGMI link.
    if (warpId < groupSize && laneId == 0) {
      int remotePe = peBase + warpId * peStride;
      application::SymmMemObjPtr dest = dstMemObj;
      uint8_t* srcPtr = reinterpret_cast<uint8_t*>(input);
      uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset;
      anvil::SdmaQueueDeviceHandle** devicehandles =
          dest->deviceHandles_d + remotePe * dest->sdmaNumQueue;
      HSAuint64* signals = dest->signalPtrs + remotePe * dest->sdmaNumQueue;
      HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * dest->sdmaNumQueue;
      core::SdmaPutThread(srcPtr, dstPtr, bytesTotal, devicehandles, signals, expectedSignals,
                          dest->sdmaNumQueue, 0);
      SubGroupSdmaDrainPe(remotePe, dstMemObj);
      shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
          flagsMemObj, 0, &flagVal, 8, core::atomicType::AMO_SET, remotePe, 0);
    }
    __syncthreads();
  } else {
    // Non-root: wait until the root has written our buffer and bumped flag 0.
    if (threadLinearId == 0) {
      int spinCount = 0;
      bool warned = false;
      // Light-spin: relaxed system spin + one seq-cst SYSTEM acquire on exit.
      while (core::AtomicLoadRelaxedSystem(flags + 0) < flagVal) {
        ++spinCount;
        if (!warned && spinCount > 10000000) {
          printf("PE %d: Slow wait for broadcast root (still waiting)\n", myPe);
          warned = true;
        }
      }
      (void)core::AtomicLoadSeqCstSystem(flags + 0);
      __threadfence_system();
    }
    __syncthreads();
  }
}

template <typename T>
__device__ void OneShotAllGatherSdmaParamContiguousKernel_body(
    int myPe, int npes, T* input, const application::SymmMemObjPtr srcMemObj,
    const application::SymmMemObjPtr dstMemObj, const application::SymmMemObjPtr flagsMemObj,
    size_t elementCount, size_t dstBaseOffset, uint64_t flagVal, const size_t* splitSizes,
    const size_t* splitOffsets, size_t splitCount) {
  if (elementCount == 0 || npes <= 0 || splitCount == 0 || splitSizes == nullptr ||
      splitOffsets == nullptr) {
    return;
  }

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;
  const size_t bytesPerElement = sizeof(T);

  if (warpId < npes && laneId == 0) {
    int remotePe = warpId;
    application::SymmMemObjPtr dest = dstMemObj;
    anvil::SdmaQueueDeviceHandle** devicehandles =
        dest->deviceHandles_d + remotePe * dest->sdmaNumQueue;
    HSAuint64* signals = dest->signalPtrs + remotePe * dest->sdmaNumQueue;
    HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * dest->sdmaNumQueue;

    for (size_t split = 0; split < splitCount; ++split) {
      size_t splitElems = splitSizes[split];
      if (splitElems == 0) {
        continue;
      }
      size_t inputElemOffset = splitOffsets[split];
      if (inputElemOffset > elementCount || splitElems > elementCount - inputElemOffset) {
        continue;
      }
      size_t outputElemOffset =
          inputElemOffset * static_cast<size_t>(npes) + static_cast<size_t>(myPe) * splitElems;
      uint8_t* srcPtr = reinterpret_cast<uint8_t*>(input) + inputElemOffset * bytesPerElement;
      uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset +
                        outputElemOffset * bytesPerElement;
      core::SdmaPutThread(srcPtr, dstPtr, splitElems * bytesPerElement, devicehandles, signals,
                          expectedSignals, dest->sdmaNumQueue, 0);
    }
  }

  __syncthreads();

  if (warpId < npes && laneId == 0) {
    int remotePe = warpId;
    SubGroupSdmaDrainPe(remotePe, dstMemObj);
    shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
        flagsMemObj, static_cast<size_t>(myPe) * sizeof(uint64_t), &flagVal, 8,
        core::atomicType::AMO_SET, remotePe, 0);
  }
  __syncthreads();

  // Completion wait (see subgroup gather note). Thread 0 acquires every peer flag.
  if (threadLinearId == 0) {
    uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
    for (int sender = 0; sender < npes; ++sender) {
      if (sender == myPe) {
        continue;
      }
      int spinCount = 0;
      bool warned = false;
      // Light-spin: relaxed system spin + one seq-cst SYSTEM acquire on exit.
      while (core::AtomicLoadRelaxedSystem(flags + sender) < flagVal) {
        ++spinCount;
        if (!warned && spinCount > 10000000) {
          printf("PE %d: Slow wait for param-contiguous data from peer %d (still waiting)\n", myPe,
                 sender);
          warned = true;
        }
      }
      (void)core::AtomicLoadSeqCstSystem(flags + sender);
    }
    __threadfence_system();
  }
  __syncthreads();
}

template <typename T>
__global__ void OneShotAllGatherSdmaParamContiguousKernel(
    int myPe, int npes, T* input, const application::SymmMemObjPtr srcMemObj,
    const application::SymmMemObjPtr dstMemObj, const application::SymmMemObjPtr flagsMemObj,
    size_t elementCount, size_t dstBaseOffset = 0, uint64_t flagVal = 1,
    const size_t* splitSizes = nullptr, const size_t* splitOffsets = nullptr,
    size_t splitCount = 0) {
  OneShotAllGatherSdmaParamContiguousKernel_body<T>(myPe, npes, input, srcMemObj, dstMemObj,
                                                    flagsMemObj, elementCount, dstBaseOffset,
                                                    flagVal, splitSizes, splitOffsets, splitCount);
}
}  // namespace collective
}  // namespace mori
