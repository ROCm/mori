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

// SDMA completion drain for sub-group gathers. Index by GLOBAL pe (`pe*nq`) to
// match the put side; `pe%8` under-indexes for peBase>=8 and drains unarmed slots.
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

  // Each peer's whole shard is pushed as one COPY_LINEAR on a single SDMA queue
  // (one copy engine saturates one XGMI link).
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

  // Completion wait: thread t seq-cst SYSTEM-acquires peer t's flag, then one
  // block barrier + one tail system fence.
  {
    int sender = static_cast<int>(threadIdx.x);
    if (sender < npes && sender != myPe) {
      // Keep waiting for the peer completion flag. A finite spin threshold can
      // produce false timeouts under heavy traffic and cause incorrect forward
      // progress (kernel continues before data is actually ready).
      int spinCount = 0;
      bool warned = false;
      // Light-spin: relaxed system load, then one seq-cst SYSTEM acquire on exit
      // (monotonic flags => same acquire, less fabric contention).
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
// Sub-group intra-node SDMA AllGather
// ---------------------------------------------------------------------------
// Intra-node phase of the hierarchical AllGather. Group is the arithmetic PE set
// ``{peBase + i*peStride : i in [0,groupSize)}``; this PE is at ``groupPos``. Each
// member writes its shard into slot ``groupPos`` of every member's dst; after the
// flag handshake every member holds all ``groupSize`` shards in position order.
// Flags indexed by group position with a per-call ``flagVal`` token (no reset). The
// flat whole-world gather above is ``groupSize=npes, groupPos=myPe, peBase=0,
// peStride=1``.
// ``blockLocal``: index threads off ``threadIdx.x`` alone so this body runs in one
// block of a larger fused grid (NIC||XGMI overlap; see all_gather.hpp,
// ccl_kernel_args.hpp CclFusedRingLocalGatherArgs). Default false is byte-identical.
// ``flagBase`` (default 0): offset flag slots to ``[flagBase, flagBase+groupSize)``
// so independent sub-group gathers run concurrently without racing shared slots.
// Nearest-neighbor pipelined SDMA ring (no incast): rank g self-fills slot g, then
// over G-1 steps forwards slot k=(g-s+G)%G to nextPeer, chunk-pipelined (C chunks).
// Flag (c,k) at flagBase + c*G + k; fencedFlag selects the copy schedule.
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
  // Cap chunk count at 16: C==32 faults (write to read-only page) in the relay
  // addressing at the 32-chunk boundary.
  if (C > 16) C = 16;
  const size_t align = 16;
  size_t chunkBytes = (bytesPerPeer + static_cast<size_t>(C) - 1) / static_cast<size_t>(C);
  chunkBytes = ((chunkBytes + align - 1) / align) * align;
  if (chunkBytes == 0) chunkBytes = bytesPerPeer;

  // Flat multi-warp broadcast (fencedFlag bit9, small-message regime): each PE self-
  // fills slot g then broadcasts to all G-1 peers, one submitter warp per dest engine.
  // Counter: G-1 senders +1 => want=(flagVal-1)*(G-1)+(G-1). Default OFF.
  const bool flatMW = ((fencedFlag & 512) != 0) && (G > 1);
  // Batch self-fill (fencedFlag bit11): submit all C self-chunks then one trailing quiet
  // (still drained+fenced before return). Requires flatMW. Default OFF.
  const bool batchSelf = flatMW && ((fencedFlag & 2048) != 0);
  if (flatMW) {
    const uint64_t fWant =
        (flagVal - 1) * static_cast<uint64_t>(G - 1) + static_cast<uint64_t>(G - 1);
    // (1) self-fill my own slot g from input on warp0 as a serial front, then batch-drain
    //     after all C chunks (one trailing quiet + fence before return).
    if (warpId == 0 && laneId == 0) {
      anvil::SdmaQueueDeviceHandle** selfHandles = dest->deviceHandles_d + myPe * nq;
      HSAuint64* selfSignals = dest->signalPtrs + myPe * nq;
      HSAuint64* selfExpected = dest->expectSignalsPtr + myPe * nq;
      for (int c = 0; c < C; ++c) {
        size_t off = static_cast<size_t>(c) * chunkBytes;
        if (off >= bytesPerPeer) break;
        size_t len = chunkBytes;
        if (off + len > bytesPerPeer) len = bytesPerPeer - off;
        uint8_t* selfSlotP = selfDstBase + static_cast<size_t>(g) * slotStride + off;
        core::SdmaPutThread(reinterpret_cast<uint8_t*>(input) + off, selfSlotP, len, selfHandles,
                            selfSignals, selfExpected, nq, 0);
      }
      // batchSelf: all C self-chunks submitted; one trailing quiet + fence before return.
      core::SdmaQueitThread(selfSignals + 0, selfExpected + 0, 1);
      __threadfence_system();
    }
    // Gate the broadcast on the self-fill completing.
    __syncthreads();
    // (2) warp t (t in [0,G-1)) broadcasts my owned slot to the t-th OTHER PE (skip self g).
    //     Each dest is a distinct engine; all G-1 doorbells fire near-simultaneously.
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
  // Restrict the PUSH half to peer-columns [pushPeerLo, pHi); the WAIT below stays full
  // [0,G) (gates on senders INTO this rank). Default (0, <0) => full-mesh push.
  const int pLo = (pushPeerLo > 0) ? pushPeerLo : 0;
  const int pHi = (pushPeerHi >= 0 && pushPeerHi < groupSize) ? pushPeerHi : groupSize;
  T* __restrict__ inputData = input;
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);

  const size_t threadLinearId =
      blockLocal ? static_cast<size_t>(threadIdx.x)
                 : static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;

  const size_t bytesPerElement = sizeof(T);
  const size_t bytesPerPeer = elementCount * bytesPerElement;
  // Per-peer dst slot stride. Default (0) packs slots contiguously (stride == copy size);
  // a non-zero stride lands each chunk at its strided position for the chunked pipeline.
  const size_t slotStride = dstSlotStrideBytes != 0 ? dstSlotStrideBytes : bytesPerPeer;

  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;

  // Warp w pushes this PE's shard into slot ``groupPos`` of member w's dst (single-queue
  // put over XGMI; G warps drive G distinct peer links in parallel).
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

  // Per-peer completion tail: each warp drains its peer's queue then bumps its flag.
  if (warpId >= pLo &&
      warpId < pHi && laneId == 0) {
    int remotePe = peBase + warpId * peStride;
    SubGroupSdmaDrainPe(remotePe, dstMemObj);
    // Sender-side completion fence: system-scope order the pushed bytes before the flag
    // AMO becomes peer-visible, else the receiver can observe the flag ahead of the data.
    __threadfence_system();
    shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
        flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
        core::atomicType::AMO_SET, remotePe, 0);
  }
  __syncthreads();

  // Completion wait: thread t seq-cst SYSTEM-acquires sender-position t's flag, then one
  // block barrier + one tail __threadfence_system. System-scope acquire is required
  // because the flag and its guarded data come from a remote peer GPU (another HSA agent)
  // via SDMA -- an agent-scope load gives no cross-agent happens-before.
  {
    int senderPos = static_cast<int>(threadIdx.x);
    if (senderPos < groupSize && senderPos != groupPos) {
      int spinCount = 0;
      bool warned = false;
      // Light-spin: relaxed system spin + one seq-cst SYSTEM acquire on exit.
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
// WRITE half of the sub-group gather ONLY (no peer wait): warp ``w`` SDMA-pushes this
// rank's slice into slot ``groupPos`` of member ``w``'s output, quiets ONLY its own SDMA
// queue ``qId``, then bumps flag ``flagBase+groupPos`` on member ``w``. Output slots are
// disjoint per rank (each rank only WRITES its own column), so dropping the coupled wait
// is safe and any number of these run concurrently without deadlock; a single completion
// reader waits every flag afterward. ``qId`` MUST be distinct across blocks targeting the
// same peer (per-queue signal counter expectedSignals[qId] must not be raced).
template <typename T>
__device__ void OneShotSubGroupPushOnly_body(
    int groupSize, int groupPos, int peBase, int peStride, T* input,
    const application::SymmMemObjPtr dstMemObj, const application::SymmMemObjPtr flagsMemObj,
    size_t elementCount, size_t dstBaseOffset, size_t dstSlotStrideBytes, uint64_t flagVal,
    size_t flagBase, int qId, int deepSqPhase = 0,
    int pushPeerLo = 0, int pushPeerHi = -1, int pushFlag = 1) {
  // Deep-SQ phase split (see HierReasmDeepSqOn). 0 = single-shot (submit->drain->fence->
  // flag). 1 = submit-only (copy + bump expectedSignals[q], no drain/fence/flag) to feed
  // the engine back-to-back. 2 = drain+flag-only (one quiet covers all phase-1 submits on
  // this queue, then fence + flag).
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
    // Push this rank's column on its OWN queue ``q``. Phase 2 skips the submit.
    if (deepSqPhase != 2) {
      core::SdmaPutThread(srcPtr, dstPtr, bytesPerPeer, devicehandles, signals, expectedSignals,
                          nq, q);
    }
    // Phase 1 skips drain/fence/flag to feed the next copy without a drain round-trip.
    if (deepSqPhase != 1) {
      // Drain ONLY queue ``q`` (phase 2: expectedSignals[q] covers all phase-1 submits).
      core::SdmaQueitThread(signals + q, expectedSignals + q, 1);
      // Sender-side fence: order pushed bytes before the flag AMO becomes peer-visible.
      __threadfence_system();
      // Flag AMO is a direct P2P CAS (never races the per-queue counter). pushFlag==0
      // (non-last tile) suppresses only the flag; the copy is still drained+fenced.
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
// Single-launch fused param-contiguous SubGroup gather. Warp ``w`` drives member ``w``;
// this PE (group position ``g``) pushes, for every node block ``m`` and param split ``s``,
// its E_s-element sub-slice into the member's output at param-contiguous element offset
// ``O_s*W + (m*G+g)*E_s``. Same subgroup flags: bump slot ``g`` once, wait all G members.
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
  // Fence all warps' SDMA puts before any warp quiets + bumps its completion
  // flag (mirrors the flat OneShotAllGatherSdmaParamContiguousKernel_body).
  __syncthreads();

  if (warpId < groupSize && laneId == 0) {
    int remotePe = peBase + warpId * peStride;
    SubGroupSdmaDrainPe(remotePe, dstMemObj);
    shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
        flagsMemObj, static_cast<size_t>(groupPos) * sizeof(uint64_t), &flagVal, 8,
        core::atomicType::AMO_SET, remotePe, 0);
  }
  __syncthreads();

  // Completion wait: one block barrier + one system fence (see subgroup gather note).
  // Thread 0 acquires every peer flag first.
  if (threadLinearId == 0) {
    for (int senderPos = 0; senderPos < groupSize; ++senderPos) {
      if (senderPos == groupPos) {
        continue;
      }
      // LIGHT-SPIN completion (see subgroup gather): relaxed system spin + one
      // seq-cst SYSTEM acquire on exit -> same acquire, less fabric contention.
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
// Root (group position 0 == PE ``peBase``) SDMA-copies its whole ``input`` buffer into
// the ``dstMemObj`` of every member of the sub-group
// ``{peBase + i*peStride : i in [0,groupSize)}`` (including itself). Intra-node placement
// phase of the leader-only hierarchical AllGather.
// Root warp ``w`` handles member ``w``: one SDMA put + quiet + single-slot flag bump.
// Each non-root member spins on flag slot 0. Single flag slot with a per-call token
// (no reset).
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
      // LIGHT-SPIN completion (see subgroup gather): relaxed system spin + one
      // seq-cst SYSTEM acquire on exit -> same acquire, less fabric contention.
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

  // Completion wait: one block barrier + one system fence (see subgroup gather note).
  // Thread 0 acquires every peer flag first.
  if (threadLinearId == 0) {
    uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
    for (int sender = 0; sender < npes; ++sender) {
      if (sender == myPe) {
        continue;
      }
      int spinCount = 0;
      bool warned = false;
      // Light-spin: relaxed system spin + one seq-cst SYSTEM acquire on exit (bit-exact).
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
