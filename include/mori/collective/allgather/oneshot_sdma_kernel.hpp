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
  //  T* __restrict__ src = reinterpret_cast<T*>(srcMemObj->localPtr);
  //  T* __restrict__ dst = reinterpret_cast<T*>(dstMemObj->localPtr);
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);

  const size_t threadLinearId =
      static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;
  //  const size_t threadsPerGrid = static_cast<size_t>(blockDim.x) *
  //  static_cast<size_t>(gridDim.x); const size_t stride = threadsPerGrid > 0 ? threadsPerGrid : 1;

  const size_t bytesPerElement = sizeof(T);
  const size_t bytesPerPeer = elementCount * bytesPerElement;
  const size_t elemsPerPeer = elementCount;

  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;

  // Each peer's whole shard is pushed as one COPY_LINEAR on a single SDMA queue.
  // A single copy engine already saturates one XGMI link, so this intra gather is
  // per-link throughput-bound, not engine-count-bound: splitting a peer's shard
  // across more SDMA engines does not raise bandwidth (and adds per-descriptor
  // overhead), so the single-queue push is used.
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
    shmem::ShmemQuietThread(remotePe, dstMemObj);
    shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
        flagsMemObj, static_cast<size_t>(myPe) * sizeof(uint64_t), &flagVal, 8,
        core::atomicType::AMO_SET, remotePe, 0);
  }
  __syncthreads();

  // Completion wait: one block barrier + one system fence at the tail instead of
  // one per sender. The npes-1 peer-flag spins are distributed across the block's
  // first npes threads (thread t polls peer t) rather than polled serially by
  // thread 0. Every peer flag is still seq-cst SYSTEM-acquired before the barrier
  // releases the block, so the acquire contract is unchanged.
  {
    int sender = static_cast<int>(threadIdx.x);
    if (sender < npes && sender != myPe) {
      // Keep waiting for the peer completion flag. A finite spin threshold can
      // produce false timeouts under heavy traffic and cause incorrect forward
      // progress (kernel continues before data is actually ready).
      int spinCount = 0;
      bool warned = false;
      // Busy-wait on a relaxed system load, then take one seq-cst SYSTEM acquire on
      // exit. A seq-cst load in a tight spin forces a full cross-agent coherence
      // round-trip every iteration, whose traffic steals fabric bandwidth from the
      // SDMA copy engines. The relaxed load is still cross-agent-visible (monotonic
      // flags) so it exits at the same moment, and the trailing seq-cst load performs
      // the same acquire the plain seq-cst spin would on its last iteration.
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
// This is the intra-node phase of the hierarchical cross-node AllGather: the
// ``G`` local ranks of one node gather their ``G`` shards over the SDMA copy
// engines (XGMI), producing each rank's contiguous node-block. The flat
// whole-world gather above is the special case
// ``groupSize=npes, groupPos=myPe, peBase=0, peStride=1``.
//
// The group is the arithmetic set of global PEs
// ``{peBase, peBase+peStride, ..., peBase+(groupSize-1)*peStride}`` and this
// PE is at position ``groupPos`` within it. Each member SDMA-writes its own
// shard into slot ``groupPos`` of every member's destination buffer; after
// the cross-set flag handshake every member holds all ``groupSize`` shards
// concatenated in group-position order. Flags are indexed by group position
// (not global PE), so a per-call ``flagVal`` token keeps successive calls
// race-free without a reset.
// ``blockLocal`` makes this body index its threads off
// ``threadIdx.x`` ALONE (ignoring ``blockIdx.x``) so it can run inside a SINGLE
// designated block of a larger FUSED grid while the OTHER blocks run the
// inter-node RDMA ring concurrently (the fused recv+reassemble / NIC||XGMI
// overlap path; see all_gather.hpp
// override note + ccl_kernel_args.hpp CclFusedRingLocalGatherArgs). The gather
// only ever needs ``groupSize`` warps (G<=warpsPerBlock), so one block
// suffices. Default false keeps the historical grid-wide thread id ->
// BYTE-FOR-BYTE identical to the shipped single-block (1,)/(512,) launch; inert
// until a fused launcher sets it.
// ``flagBase`` (default 0, INERT for every existing caller) offsets the flag
// slots this gather uses from ``[0, groupSize)`` to ``[flagBase, flagBase+
// groupSize)``. It lets MULTIPLE independent sub-group gathers run CONCURRENTLY
// (different blocks of one fused grid, e.g. the Phase-4 chunked remote-block
// reassembly where sub-range j uses flagBase = j*groupSize) without racing on the
// shared flag slots. 0 keeps the historical single-region layout byte-for-byte.
// Intra-node nearest-neighbor pipelined SDMA ring for the multi-node local block.
// A flat G-way push makes every receiver take a G-1 concurrent-write incast that the
// SDMA engines cannot saturate; a nearest-neighbor ring keeps only one flow across
// each XGMI link at a time (no incast). Rank g self-fills its own slot, then over
// G-1 steps forwards slot k=(g-s+G)%G to nextPeer, chunk-pipelined (each shard split
// into C ~4MiB chunks so chunk c+1 is in flight while c relays => critical path
// ~shardTime + (G-1)*chunkTime, all G links busy).
// Completion: fencedFlag=1 uses SdmaPutCopySignalThread (copy + local signal on the
// engine, per-step drain, then P2P AMO_SET the C landing flags -- no remote op on
// the copy engine, the only reliable model on this HW); fencedFlag=0 uses the plain
// per-chunk SdmaPutThread+ShmemQuietThread+AMO. Flag (c,k) at flagBase + c*G + k.
// Same final slot layout and bytes; only the copy schedule differs.
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
  // Cap the ring chunk count at 16. A chunk count of 32 triggers a memory fault
  // ("write access to a read-only page") in the relay/self-fill addressing at the
  // 32-chunk boundary; C<=16 keeps every size at a safe chunk count (128MB -> 8MB
  // chunks, 256MB -> 16MB chunks). The individual chunk sizes are valid copy sizes,
  // so this is a chunk-count limit, not a per-size buffer edge.
  if (C > 16) C = 16;
  // This single-queue (Q=1) ring is fully pipelined but its single-warp serial
  // submission cannot keep one SDMA queue full as the op grows, so at the largest
  // sizes the flat crown (which issues concurrent COPY_LINEARs and is incast-bound)
  // can be faster. Parallel submission across ring-local queues (nq>1) is the path
  // to close that gap.
  const size_t align = 16;
  size_t chunkBytes = (bytesPerPeer + static_cast<size_t>(C) - 1) / static_cast<size_t>(C);
  chunkBytes = ((chunkBytes + align - 1) / align) * align;
  if (chunkBytes == 0) chunkBytes = bytesPerPeer;

  // Multi-queue ring submission. Running the whole relay on one warp/one SDMA queue is
  // submission-bound even though the schedule is incast-free; XGMI is saturated by
  // driving many channels concurrently, not one engine. Each chunk-pipeline c is
  // independent (its own ring of per-(c,k) flags), so stripe the C chunks across
  // Q=min(nq,C,numWarps) submitter warps, warp w owning SDMA queue w and chunks
  // {c : c%Q==w}. Q queues => Q engines busy in parallel, one flow per queue => still
  // no incast. Each warp drains only its own queue so the submitters never serialize on
  // a shared quiet. Identical final slot bytes and per-(c,k) flag protocol; only the
  // submit fan-out changes.
  //
  // Note: per-(c,k) cross-PE AMO_SET flag delivery on queues>0 is not robust under
  // sustained multi-warp concurrency (flag updates can be lost); queue 0 alone is
  // reliable. The monotonic per-link tail step-counter used below avoids this by
  // replacing the per-chunk cross-agent atomics with one accumulating counter per link.
  const int numWarps =
      static_cast<int>((static_cast<size_t>(blockDim.x) + warpSize - 1) / warpSize);
  int Q = nq < C ? nq : C;
  if (Q > numWarps) Q = numWarps;
  if (Q < 1) Q = 1;
  // crownRing bit1 forces single-queue (Q=1): warp0 pipelines all C chunks on queue 0.
  // Queue index 1 stalls its chunks under sustained load regardless of the flag
  // mechanism (the second SDMA engine/queue is the common factor, not the protocol),
  // so Q=1 gives a reliably completing ring; multi-queue fan-out is enabled once the
  // q>0 delivery is resolved. Still fully pipelined (C chunks overlapped on one queue).
  if ((fencedFlag & 2) != 0) Q = 1;
  // 2x4 hierarchical half-ring (fencedFlag bit2). Split the G=8 intra gather into two
  // 4-PE halves: PHASE 1 = a 4-ring within each half (H-1=3 hops, incast-free, Q=1
  // fused COPY+ADD64) so every PE gathers its own half's H slots; PHASE 2 = a single
  // pairwise cross-half exchange (PE g <-> g+H) of that H-slot block. The ring diameter
  // drops 7->3, reducing per-step serialization at the large sizes where the Q=1 8-ring
  // is submission-bound. Stays Q=1 (single SDMA queue), so it does not depend on the
  // unresolved q>0 second-engine stall. Same final slot bytes; only the copy schedule
  // and completion-flag regions differ. Flags on the ring's private region: phase1
  // counter[c]=flagBase+c, phase2 counter[c]=flagBase+C+c. Default OFF (bit2 clear) =>
  // existing G-ring path unchanged.
  // The halved diameter also completes at the largest sizes where the Q=1 8-ring faults
  // on pipeline depth. Because phase1 and phase2 are sequential and phase2 is itself a
  // single-queue serial block copy, the submission ceiling relocates into phase2 at mid
  // sizes (addressed by the fused-overlap path below).
  // Flat multi-warp broadcast for the small-message regime (fencedFlag bit9). At small
  // sizes the transfer is tiny so the 7-way incast the 2x4 path avoids is not bandwidth-
  // limiting; there fixed orchestration latency dominates and the cheapest schedule wins.
  // Each PE self-fills its own slot g, then broadcasts g to all G-1 other PEs with one
  // submitter warp per distinct dest PE (= distinct engine, engine=f(src,dst)) so all
  // G-1 doorbells fire together, and one completion barrier -- two fewer round-trips than
  // the 2x4. Distinct from the default flat crown, which submits the G-1 copies serially
  // from one warp. Counter: each of the G-1 senders adds +1 to my flags[flagBase+c] =>
  // want = (flagVal-1)*(G-1) + (G-1), accumulating and order-independent, so the final
  // slot bytes are identical and only the schedule differs. Default OFF (bit9 clear).
  const bool flatMW = ((fencedFlag & 512) != 0) && (G > 1);
  // Fused self-fill (fencedFlag bit10). In the plain flatMW path the self-fill (self
  // SDMA copy + drain + fence) is a serial front on the critical path before the
  // broadcast issues, yet the broadcast sources from `input`, not the self-filled slot,
  // so it does not depend on it. Fold the self-copy into the same parallel doorbell fan:
  // warp t in [0,G) copies my slot g to PE t (t==g => a local self-copy) and adds +1 to
  // PE t's flags[flagBase+c], identical to a peer send. All G doorbells fire together
  // with no serial front. The receiver now sees G writers => want=(flagVal-1)*G + G. The
  // self slot is still filled from input, only ADD64-accumulated (order-independent)
  // instead of drained. Requires flatMW. Default OFF (bit10 clear).
  const bool fuseSelf = flatMW && ((fencedFlag & 1024) != 0);
  // Batch self-fill (fencedFlag bit11). The plain flatMW self-fill drains after each of
  // its C self-chunks, so they run as C serial round-trips before the broadcast issues.
  // batchSelf keeps flatMW's G-1-writer barrier but defers the drain: submit all C self-
  // chunks first (they pipeline on queue 0), then one quiet at the end. The self slot is
  // still drained+fenced before return; only the completion-wait moves out of the loop.
  // Requires flatMW, mutually exclusive with fuseSelf. Default OFF (bit11 clear).
  const bool batchSelf = flatMW && !fuseSelf && ((fencedFlag & 2048) != 0);
  // Overlap self-fill (fencedFlag bit12). batchSelf still runs the self-fill as a serial
  // front on warp0 before the broadcast (the __syncthreads after step 1 gates the
  // broadcast on warp0's self submit). The local block is G=8 launched with 512 threads
  // = 8 warps, so warp (G-1) is free (the broadcast uses warps [0,G-1)). Run the self-
  // fill on that free warp and drop the front __syncthreads so self and broadcast submit
  // concurrently (independent: self is a local engine, broadcast is the G-1 remote
  // engines, and the broadcast sources from input, not the self slot), deferring the
  // self drain+fence to after the broadcast barrier. Self keeps the separate signal/quiet
  // path (not the counter) so the G-1-writer barrier is untouched. The self slot is still
  // drained+fenced before return. Requires flatMW, mutually exclusive with fuseSelf/
  // batchSelf, needs warp (G-1) to exist.
  const bool overlapSelf = flatMW && !fuseSelf && !batchSelf &&
                           ((fencedFlag & 4096) != 0) && ((G - 1) < numWarps);
  if (fuseSelf) {
    const uint64_t fWant =
        (flagVal - 1) * static_cast<uint64_t>(G) + static_cast<uint64_t>(G);
    // warp t (t in [0,G)) copies my owned slot g to PE t (t==g => local self-fill), ADD64 to
    // PE t's counter. All G distinct-engine doorbells fire together; no serial self-fill front.
    if (warpId < G && laneId == 0) {
      const int destPe = peBase + warpId * peStride;       // t==g => myPe (local copy)
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
    // (1) self-fill my own slot g from input (drained), one fence.
    //     overlapSelf: run on the free warp (G-1), submit only (drain+fence deferred to step 3a)
    //     so it overlaps the broadcast; else run on warp0 as a serial front (batchSelf/plain).
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
    // (2) warp t (t in [0,G-1)) broadcasts my owned slot to the t-th OTHER PE (skip self g).
    //     Each dest is a distinct engine; all G-1 doorbells fire near-simultaneously.
    if (warpId < G - 1 && laneId == 0) {
      const int t = warpId;
      const int destPos = (t < g) ? t : (t + 1);           // skip my own position g
      const int destPe = peBase + destPos * peStride;      // distinct dest PE = distinct engine
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
    // (3a) overlapSelf: drain the deferred self-fill now -- it ran concurrently with the
    //      broadcast on the free warp; wait it here so the self slot is landed before the fence.
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
    const int half = g / H;                              // 0 or 1
    const int pos = g % H;                               // 0..H-1 within my half
    const int halfBase = peBase + half * H * peStride;   // first PE of my half
    const uint64_t p1Base = (flagVal - 1) * static_cast<uint64_t>(H - 1);
    const uint64_t p1Want = p1Base + static_cast<uint64_t>(H - 1);
    const uint64_t p2Base = (flagVal - 1) * static_cast<uint64_t>(H);
    const uint64_t p2Want = p2Base + static_cast<uint64_t>(H);
    const uint64_t k8MB = 8u * 1024u * 1024u;

    // ===== Fused phase1/phase2 overlap (fencedFlag bit3) =====
    // In the sequential 2x4 the phases run back-to-back with a full-half barrier, and
    // phase2 is a single-queue serial block of H copies, so the Q=1 submission ceiling
    // relocates into phase2; meanwhile phase1's warp0 spins idle between ring steps,
    // leaving the SDMA queue empty during those bubbles. Fuse the two phases: the moment
    // a my-half slot lands (self-filled mySlot, or relayed from the predecessor),
    // immediately submit both its ring forward (phase1, to next) and its cross-half send
    // (phase2, to partner) on the same queue, so phase2 fills phase1's bubbles and the
    // full-half barrier vanishes. Identical final slot bytes and accumulating flag
    // counters (p1 at flagBase+c, p2 at flagBase+C+c); only the submit order interleaves.
    // Stays Q=1 (single warp/queue).
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
        // (1) self-fill all C chunks of my own slot (drained), then one fence -- keeps the
        //     self-copy drains out of the fused relay/exchange stream below.
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
        // (2) fused relay(phase1) + cross-half send(phase2). For each chunk, walk the H ring
        //     steps; each landed slot is forwarded (s<H-1) AND shipped cross-half (all s).
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

    // Destination-spread phase1 (fencedFlag bit6). When phase2 is made H-engine-parallel,
    // the intra-half 4-ring relay (Q=1 single-engine, running sequentially before phase2)
    // becomes the bottleneck. Make phase1 a flat within-half broadcast instead: each PE
    // sends its own slot (from input, no relay dependency) to the H-1 other PEs of its
    // half = H-1 distinct dest engines, all on queue 0. Receiver gathers its half's H
    // slots (self-fill + H-1 peers), peak incast per phase stays <=H rather than the 7-way
    // flat incast. Counter: each of the H-1 senders +1 to the receiver's flags[flagBase+c]
    // => want = p1Base+(H-1), identical to the ring (single predecessor increments H-1
    // times). Default OFF (bit6 clear).
    const bool spreadP1 = ((fencedFlag & 64) != 0);
    // Multi-warp dest-spread submission (fencedFlag bit7). The parallelism axis is
    // destination engines (engine=f(src,dst)), not queue index, but the single-warp dest-
    // spread still submits all H copies serially from warp0, so the engines start
    // staggered by warp0's per-copy submit latency. Give each destination its own
    // submitter warp (warp t -> dest position t = distinct engine) so all H doorbells ring
    // near-simultaneously. Stays queue-0-only per dest (no q>0 same-dst stall); each warp
    // targets a distinct dest PE (no same-queue enqueue race). Counters unchanged (each
    // dest still receives one +1 from this PE). Default OFF (bit7 clear).
    const bool multiWarp = ((fencedFlag & 128) != 0);
    // Concurrent phase1/phase2 (fencedFlag bit8). With multiWarp, phase1 and phase2 both
    // use warps [0,H) separated by the p1-completion barrier; at small sizes that barrier
    // bubble dominates the short transfer. Both phases source from input independently
    // (phase1 broadcasts my owned slot within-half, phase2 cross-half), so run them
    // concurrently: phase1 on warps [0,H), phase2 on warps [H,2H), drop the intermediate
    // p1 barrier, and wait both p1Want and p2Want once at the end. Peak incast rises to
    // H-1+H, which helps small (latency-bound) sizes but may throttle large (incast-bound)
    // ones. 8 warps in the 512-thread block => warps [H,2H) valid. Default OFF.
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
      const int mySlot = half * H + pos;                 // global slot this PE owns
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
            if (t == pos) continue;                                // skip self
            const int destPe = halfBase + t * peStride;            // each = distinct engine
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
            const int kk = (pos - s + H) % H;            // within-half slot to forward
            const int gslot = half * H + kk;             // global slot index
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
    // For concurrent phase1/phase2, skip the intermediate p1-completion barrier: phase2
    // (warps [H,2H)) sources from input and does not depend on phase1, so it runs
    // alongside it; both are waited once at the final barrier below.
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
    // Parallel phase2 (fencedFlag bit4). A single warp serial-submitting phase2's H*C
    // fused copies to one SDMA queue cannot saturate XGMI. Phase2's H cross-half slot
    // copies are independent (same partner PE, disjoint dst slots, no cross-hop relay
    // dependency), so they are a safe place to add parallel submission. Stripe the H slots
    // across QP=min(H,nq) submitter warps: warp w owns SDMA queue w and slots {j : j%QP==w},
    // exactly one warp per queue (no concurrent same-queue enqueue race). Each fused
    // COPY+ADD64 rides its own queue; the accumulating +1 per slot is order-independent so
    // the receiver still waits want=p2Base+H. Stays on the 2x4 base. Default OFF (bit4 clear).
    // Destination-spread phase2 (fencedFlag bit5). engine = f(src,dst) on this HW, so a
    // second queue index to the same partner is a second channel to the same engine, which
    // stalls; the pairwise phase2 (all H slots -> one partner) then uses only one engine per
    // PE and is single-link bound. Spread across destinations instead of queue indices: each
    // PE broadcasts its own slot (from input, no phase1 dependency) to the H PEs of the other
    // half = H different dest PEs = H different SDMA engines = H-way parallelism on queue 0.
    // Receiver PE gathers the other half's H owned slots (one +1 per sender => want =
    // p2Base + H, unchanged). Stays queue-0-only (no q>0 stall). Default OFF (bit5 clear).
    const bool spreadP2 = ((fencedFlag & 32) != 0);
    const bool parallelP2 = ((fencedFlag & 16) != 0) && nq > 1;
    if (spreadP2 && multiWarp) {
      // Multi-warp dest-spread phase2: warp t sends my owned slot to the other-half PE at
      // position t (H distinct dest engines, one submitter warp each, all queue 0). All H
      // doorbells fire together instead of serialized behind warp0. With concP, phase2 runs
      // on warps [H,2H) so it overlaps phase1 (warps [0,H)) -- both submit concurrently.
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
          uint8_t* src = reinterpret_cast<uint8_t*>(input) + off;    // owned slot from input
          for (int t = 0; t < H; ++t) {
            const int destPe = otherBase + t * peStride;             // each = distinct engine
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
            const int gslot = half * H + j;              // my half's slots -> partner needs
            uint8_t* src = selfDstBase + static_cast<size_t>(gslot) * slotStride + off;
            uint8_t* dstPtr = pDstBase + static_cast<size_t>(gslot) * slotStride + off;
            void* peerCtr = static_cast<void*>(partnerP2 + p2Ctr);
            core::SdmaPutCopyRemoteAddThread(src, dstPtr, len, pHandles,
                                             static_cast<uint32_t>(w), peerCtr);
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
          const int gslot = half * H + j;                // my half's slots -> partner needs
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
      // Cap each SDMA copy descriptor at <=8 MiB (CDNA3 microcode faults on a single
      // >8MB COPY_LINEAR). For len>8MB, split the self-fill into ceil(len/8MB) equal sub-
      // descriptors on the same queue with one trailing atomic and one drain; len<=8MB is
      // unchanged.
      int selfNdesc = static_cast<int>((len + (8u * 1024u * 1024u - 1)) / (8u * 1024u * 1024u));
      if (selfNdesc < 1) selfNdesc = 1;
      core::SdmaPutThread(reinterpret_cast<uint8_t*>(input) + off, selfSlot + off, len, selfHandles,
                          selfSignals, selfExpected, nq, static_cast<uint32_t>(q), selfNdesc);
      core::SdmaQueitThread(selfSignals + q, selfExpected + q, 1);
    }
    __threadfence_system();

    if (G > 1) {
      // Monotonic per-link tail step-counter completion, replacing per-(c,k) cross-PE
      // AMO_SET flags that can be dropped under multi-warp load. Each PE has exactly one
      // incoming ring link (its predecessor), which for chunk c delivers slots in strict
      // step order s'=0..G-2. Instead of C*G distinct SET flags (lossy if one is dropped),
      // keep one counter per chunk at flagBase+c that the predecessor atomic-adds +1 to
      // after each drained step. The receiver only needs the count, since slots land in a
      // fixed order; ADD is accumulating and monotone so no update can be lost. Across reps
      // the counter grows by G-1 per op, so the per-op base is (flagVal-1)*(G-1); step s
      // waits counter >= base+s.
      const uint64_t ringBase = (flagVal - 1) * static_cast<uint64_t>(G - 1);
      uint64_t oneAdd = 1;
      // Each chunk c is a self-contained pipeline through the ring's G-1 steps; run the
      // whole chunk on this warp's queue q so the Q warps advance Q chunks concurrently.
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
              printf("[RINGSTUCK] SEND myPe=%d g=%d s=%d c=%d ctr=%llu got=%llu want=%llu\n",
                     myPe, g, s, c, static_cast<unsigned long long>(myCtr),
                     static_cast<unsigned long long>(core::AtomicLoadRelaxedSystem(flags + myCtr)),
                     static_cast<unsigned long long>(want));
            }
            (void)core::AtomicLoadSeqCstSystem(flags + myCtr);
            src = selfDstBase + static_cast<size_t>(k) * slotStride + off;
          }
          uint8_t* dstPtr = nextDstBase + static_cast<size_t>(k) * slotStride + off;
          // Fused copy + remote tail-counter +1 on the same SDMA queue in one doorbell.
          // A CU P2P atomic on flagsMemObj->peerPtrs faults here (the intra flag object's
          // peerPtrs are engine-addressed) and the P2P AMO helper spins forever on that
          // memory, so the SDMA engine posts the ADD64 instead: SdmaPutCopyRemoteAddThread
          // enqueues COPY_LINEAR then SDMA_PKT_ATOMIC ADD64(+1) to nextPe's counter, FIFO-
          // ordered so the peer sees the +1 strictly after the bytes land (flag-after-
          // bytes). This keeps both the per-step send-drain and the separate flag publish
          // off the critical path -- the receiver's counter poll is the sole flow control.
          // ADD accumulates so no +1 can be lost. peerCtr = nextPe's copy of my incoming-
          // link counter slot (flagsMemObj->peerPtrs[nextPe] + myCtr), XGMI P2P engine-mapped.
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
        printf("[RINGSTUCK] COMPL myPe=%d g=%d c=%d ctr=%llu got=%llu want=%llu\n",
               myPe, g, c, static_cast<unsigned long long>(myCtr),
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
    uint64_t flagVal = 1, bool blockLocal = false, size_t flagBase = 0, int multiQueue = 0,
    int qFlag = 0, int ringPhased = 0, int batchFence = 0, int h2x4 = 0, int pushPeerLo = 0,
    int pushPeerHi = -1, int putNdesc = 1, int poleSqDepth = 1, int polePull = 0,
    int putCacheHint = 0, int intra2 = 0) {
  (void)npes;
  if (elementCount == 0 || groupSize <= 0) {
    return;
  }
  // FIRST-LAND idle-engine reclamation (see HierLocalOffload): restrict the PUSH half
  // to peer-columns [pushPeerLo, pHi) so a helper CTA can issue the complementary
  // range. The completion WAIT below is UNCHANGED (full [0,G)) -- it gates on senders
  // INTO this rank (set by peers' pushes), which is independent of which targets this
  // rank pushes to. Default (pushPeerLo==0, pushPeerHi<0) => pHi==groupSize == the
  // byte-identical shipped full-mesh push.
  const int pLo = (pushPeerLo > 0) ? pushPeerLo : 0;
  const int pHi = (pushPeerHi >= 0 && pushPeerHi < groupSize) ? pushPeerHi : groupSize;
  // COPY-ENGINE (LL-style) INLINE-FLAG COMPLETION (see HierQFlagOn / MORI_HIER_QFLAG,
  // ported here from the push-only body's qFlag path -- T4/T7). The shipped crown
  // gather delivers each peer's completion flag via THREE per-peer critical-path
  // round-trips: ShmemQuietThread (send-CQ drain) + __threadfence_system (system
  // flush) + a SEPARATE direct P2P AMO store. When qFlagActive, instead ride the
  // ready-flag on the SAME SDMA queue as its data copy (COPY_LINEAR + FENCE packet,
  // FIFO-ordered) so "flag visible => bytes visible" holds with ZERO extra ops on
  // the completion critical path -- the copy-engine completion model. Only the
  // single-queue path is eligible (multiQueue splits across engines and keeps its
  // own drain-before-AMO accounting). BIT-EXACT by construction: the fence packet
  // is FIFO-ordered strictly after the peer's copied bytes on one engine, so the
  // reader's existing seq-cst SYSTEM acquire + __threadfence_system never observes
  // the flag ahead of the data. Default 0 => byte-identical shipped crown path.
  const bool qFlagActive = (qFlag != 0 && multiQueue == 0);
  // Staggered-permutation ring schedule (see HierRingPhasedOn / MORI_HIER_RING). The
  // plain path fires all G-1 peer copies concurrently (full mesh), bursting every
  // source->dest pair into the XGMI crossbar at once. When ringPhasedActive, issue this
  // GPU's copies in G-1 rotated phases so at phase p every rank targets partner
  // (r+1+p) mod G (a permutation when phases align) with a block barrier between phases.
  // Only the plain single-queue path is eligible (multiQueue owns its own engine-split
  // accounting; qFlag rides its own tail).
  const bool ringPhasedActive = (ringPhased != 0 && multiQueue == 0 && !qFlagActive);
  // Batched sender-side completion fence (see HierBatchFence / MORI_HIER_BATCH_FENCE).
  // Only the plain per-peer completion tail (not qFlag, which rides its flag inline on
  // the SDMA queue) is eligible. When active, the G concurrent __threadfence_system in
  // the tail collapse to one (drains + flag-AMOs stay parallel per peer).
  const bool batchFenceActive = (batchFence != 0 && !qFlagActive);
  // Hierarchical 2x4 intra gather (see HierH2x4 / MORI_HIER_H2X4). Splits the node's G
  // ranks into two sub-groups of H=G/2 and gathers in 2 phases so peak concurrent
  // outbound copies per GPU drop from G-1 to H (the width at which the SDMA engines
  // saturate; the flat G-way push does not). Only eligible on the plain single-queue
  // path (the other levers own their own submit+tail), for even G that is 4 or 8 (H a
  // power of two so partnerPos = groupPos ^ H is the same sub-group position in the other
  // sub-group). When active, this branch does the full submit + per-phase completion
  // signaling; the shared tail below is skipped and the final completion wait (unchanged)
  // still waits all G-1 sender flags.
  const bool h2x4Active = (h2x4 != 0 && multiQueue == 0 && !qFlagActive && !ringPhasedActive &&
                           !batchFenceActive && (groupSize == 4 || groupSize == 8));
  // 2x4 stacked-flat-body intra (see HierIntra2 / MORI_HIER_INTRA2). Issue the flat push
  // in ceil(G/W) drained W-regular-matching waves (concurrent egress + incast G-1 -> W).
  // Only the plain single-queue path is eligible (the other levers own their own submit+
  // tail). Its own drain/fence/flag => skips the shared tail below.
  const bool intra2Active = (intra2 > 0 && multiQueue == 0 && !qFlagActive && !ringPhasedActive &&
                             !batchFenceActive && !h2x4Active);

  T* __restrict__ inputData = input;
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);

  const size_t threadLinearId =
      blockLocal ? static_cast<size_t>(threadIdx.x)
                 : static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;

  const size_t bytesPerElement = sizeof(T);
  const size_t bytesPerPeer = elementCount * bytesPerElement;
  // Per-peer destination slot stride. Default (0) packs slots contiguously (stride ==
  // copy size), preserving the original layout exactly. A non-zero stride lets a chunk
  // (bytesPerPeer) land at its strided position inside a full-size block, enabling the
  // chunked pipeline.
  const size_t slotStride = dstSlotStrideBytes != 0 ? dstSlotStrideBytes : bytesPerPeer;

  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;

  // ===================== Pull local gather =====================
  // See HierPolePull. The default crown is a push all-gather (read local input, write
  // shard into slot groupPos of all peers; XGMI-egress-heavy) and is XGMI-egress-bound.
  // This flips the direction: each rank's own copy engines read the peer shards over
  // XGMI into local output; completion is a self-drain (no per-peer cross-node data-
  // landing fence). Only a cheap staged flag (local stage of the own shard, no data
  // move) crosses PEs.
  //   (1) stage own shard input -> own output slot (self SDMA copy), drain, fence.
  //   (2) AMO_SET flags[flagBase+groupPos] on every peer  ("my slot is staged").
  //   (3) wait flags[flagBase+0..G) (every peer staged its own slot).
  //   (4) PULL slot j (j!=groupPos) from peer j output[j] -> local output[j].
  //   (5) drain own pull queues, fence, sync.
  // Bit-exact: peer j staged shard j into its output slot j; this rank pulls that exact
  // block into its own slot j -> identical final layout. Only the single-queue,
  // no-other-lever crown local block is eligible. polePull==0 => byte-identical crown.
  const bool polePullActive = (polePull != 0 && multiQueue == 0 && !qFlagActive &&
                               !ringPhasedActive && !batchFenceActive && !h2x4Active);
  if (polePullActive) {
    application::SymmMemObjPtr dest = dstMemObj;
    const int nq = dest->sdmaNumQueue > 0 ? dest->sdmaNumQueue : 1;
    uint8_t* localOut = reinterpret_cast<uint8_t*>(dest->localPtr) + dstBaseOffset;
    uint8_t* myIn = reinterpret_cast<uint8_t*>(inputData);
    // (1) stage own shard into own output slot via the self SDMA queue.
    if (warpId == 0 && laneId == 0) {
      const int selfPe = peBase + groupPos * peStride;
      const size_t selfOff = static_cast<size_t>(groupPos) * slotStride;
      anvil::SdmaQueueDeviceHandle** dh = dest->deviceHandles_d + selfPe * nq;
      HSAuint64* sig = dest->signalPtrs + selfPe * nq;
      HSAuint64* esig = dest->expectSignalsPtr + selfPe * nq;
      core::SdmaPutThread(myIn, localOut + selfOff, bytesPerPeer, dh, sig, esig, nq, 0, 1);
      shmem::ShmemQuietThread(selfPe, dest);
    }
    __syncthreads();
    if (threadLinearId == 0) __threadfence_system();
    __syncthreads();
    // (2) signal "staged" to every peer at my slot.
    if (warpId < groupSize && laneId == 0) {
      int remotePe = peBase + warpId * peStride;
      shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
          flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
          core::atomicType::AMO_SET, remotePe, 0);
    }
    // (3) wait all peers staged (parallel poll over the first G threads).
    {
      int senderPos = static_cast<int>(threadIdx.x);
      if (senderPos < groupSize && senderPos != groupPos) {
        while (core::AtomicLoadRelaxedSystem(flags + flagBase + senderPos) < flagVal) {
        }
        (void)core::AtomicLoadSeqCstSystem(flags + flagBase + senderPos);
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      __syncthreads();
    }
    // (4) PULL each other slot from its owner's output over XGMI into local output.
    if (warpId < groupSize && warpId != groupPos && laneId == 0) {
      int remotePe = peBase + warpId * peStride;
      const size_t off = static_cast<size_t>(warpId) * slotStride;
      uint8_t* srcPtr =
          reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset + off;
      uint8_t* dstPtr = localOut + off;
      anvil::SdmaQueueDeviceHandle** dh = dest->deviceHandles_d + remotePe * nq;
      HSAuint64* sig = dest->signalPtrs + remotePe * nq;
      HSAuint64* esig = dest->expectSignalsPtr + remotePe * nq;
      core::SdmaPutThread(srcPtr, dstPtr, bytesPerPeer, dh, sig, esig, nq, 0, putNdesc);
    }
    // (5) drain own pull queues (self-completion; no cross-node landing flag needed).
    if (warpId < groupSize && warpId != groupPos && laneId == 0) {
      int remotePe = peBase + warpId * peStride;
      shmem::ShmemQuietThread(remotePe, dest);
    }
    __syncthreads();
    if (threadLinearId == 0) __threadfence_system();
    __syncthreads();
    return;
  }

  // Each member warp pushes this PE's shard into slot ``groupPos`` of the
  // warpId-th group member's destination buffer (SDMA over XGMI / P2P).
  // A single copy engine already saturates one XGMI link (~108 GB/s/link
  // measured), and the G warps drive G distinct peer links in parallel, so
  // one queue per peer is already bandwidth-bound — splitting a peer's shard
  // across multiple SDMA queues (SdmaPutWarp) gives no speedup and is
  // marginally slower (verified NC=1/2/4 via test_intra_subgroup_sdma --bench).
  // Keep the proven single-queue put (multiQueue==0).
  //
  // Multi-engine per-link (see HierIntraMultiQueue / MORI_INTRA_MQ). anvil::connect()
  // spreads sdmaNumQueue channels across the KFD-recommended engine mask per peer link
  // (typically 2 XGMI SDMA engines per link on MI300X), but the plain put drives only
  // queue 0 -> only one of the link's engines. multiQueue!=0 splits this peer's column
  // across all sdmaNumQueue queues (SdmaPutWarp: lane k drives queue k over a disjoint
  // contiguous sub-range), engaging every recommended engine on the link to raise per-
  // link fill. Disjoint sub-ranges of the same bytes, and the existing
  // ShmemQuietThread(remotePe) below already drains all sdmaNumQueue queues before the
  // flag AMO, so the flag still never precedes any sub-copy.
  if (h2x4Active) {
    // Hierarchical 2x4 sub-group broadcast. See HierH2x4 for the rationale.
    // H=G/2 sub-groups; each rank at (sgId,sgPos). partnerPos = groupPos ^ H is the
    // same sgPos in the other sub-group. Two phases (peak H concurrent copies each):
    //   P1: push MY shard to my H-1 sub-group peers + my partner (+ self-copy own
    //       slot, unsignalled) and signal flags[groupPos] on the H peer targets.
    //   (local wait for my partner's shard: flags[partnerPos])
    //   P2: forward my partner's LANDED shard to my H-1 sub-group peers and signal
    //       flags[partnerPos] on them.
    // Coverage: every shard reaches all G-1 non-owners, each receiver's flags[X]
    // (X != receiver) set exactly once; the final wait below is unchanged. Bit-exact.
    const int H = groupSize / 2;
    const int sgId = groupPos / H;
    const int sgPos = groupPos % H;
    const int partnerPos = groupPos ^ H;
    const size_t myShardDstOff = static_cast<size_t>(groupPos) * slotStride;
    const size_t partnerDstOff = static_cast<size_t>(partnerPos) * slotStride;
    uint8_t* myInput = reinterpret_cast<uint8_t*>(inputData);
    uint8_t* myLocalBuf = reinterpret_cast<uint8_t*>(dstMemObj->localPtr) + dstBaseOffset;
    application::SymmMemObjPtr dest = dstMemObj;
    const int nq = dest->sdmaNumQueue;
    if (h2x4 == 3) {
      // ---- Asymmetric single-relay 2x4 ----
      // Sub-group sgId occupies output slots {sgId*H .. sgId*H+H-1}; each SG's relay is
      // its sgPos==0 member. No circular cross-rank dependency: (P1) intra-SG 4-way
      // gather (mutual, non-circular), barrier via flags; (P2) the two relays exchange
      // their full H-shard block over the single relay0<->relay1 link (one-directional
      // writes); (P3) each relay broadcasts the other SG's H shards to its H-1 sub-group
      // peers. Cross-half XGMI is carried only by the relay link. Each shard s reaches
      // every rank g!=owner, flags[flagBase+s] set exactly once per receiver; the shared
      // final wait below is unchanged.
      const bool amRelay = (sgPos == 0);            // CTA-uniform (whole block = 1 rank)
      const int myRelay = sgId * H;                 // my SG's relay slot
      const int otherRelay = (sgId ^ 1) * H;        // the other SG's relay slot
      uint8_t* outBuf = myLocalBuf;                 // my assembled output base
      const bool contig = (slotStride == bytesPerPeer);

      // ---- PHASE 1: intra sub-group 4-way gather (ALL ranks) ----
      // warp w<H-1 -> w-th sub-group peer (skip self); warp H-1 -> self-slot copy.
      int a1pos = -1;
      if (warpId < H - 1) {
        int j = (warpId < sgPos) ? warpId : warpId + 1;
        a1pos = sgId * H + j;
      } else if (warpId == H - 1) {
        a1pos = groupPos;  // own slot
      }
      if (warpId <= H - 1 && laneId == 0 && a1pos >= 0) {
        int remotePe = peBase + a1pos * peStride;
        uint8_t* dstPtr =
            reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset + myShardDstOff;
        anvil::SdmaQueueDeviceHandle** dh = dest->deviceHandles_d + remotePe * nq;
        HSAuint64* sig = dest->signalPtrs + remotePe * nq;
        HSAuint64* esig = dest->expectSignalsPtr + remotePe * nq;
        core::SdmaPutThread(myInput, dstPtr, bytesPerPeer, dh, sig, esig, nq, 0);
      }
      __syncthreads();
      if (warpId < H - 1 && laneId == 0 && a1pos >= 0) {
        shmem::ShmemQuietThread(peBase + a1pos * peStride, dest);
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      __syncthreads();
      if (warpId < H - 1 && laneId == 0 && a1pos >= 0) {
        shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
            flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
            core::atomicType::AMO_SET, peBase + a1pos * peStride, 0);
      }
      __syncthreads();
      // local-wait my H-1 sub-group peers' shards (warp w<H-1 waits its a1pos shard)
      if (warpId < H - 1 && laneId == 0 && a1pos >= 0) {
        while (core::AtomicLoadRelaxedSystem(flags + flagBase + a1pos) < flagVal) {
        }
        (void)core::AtomicLoadSeqCstSystem(flags + flagBase + a1pos);
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      __syncthreads();

      // ---- PHASE 2: the two relays EXCHANGE their H-shard block (single relay link) ----
      if (amRelay) {
        int remotePe = peBase + otherRelay * peStride;
        anvil::SdmaQueueDeviceHandle** dh = dest->deviceHandles_d + remotePe * nq;
        HSAuint64* sig = dest->signalPtrs + remotePe * nq;
        HSAuint64* esig = dest->expectSignalsPtr + remotePe * nq;
        uint8_t* peerBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset;
        // ONE warp does the forward (all H shards to the SAME peer => no same-peer race).
        if (warpId == 0 && laneId == 0) {
          if (contig) {
            size_t off = static_cast<size_t>(myRelay) * slotStride;
            core::SdmaPutThread(outBuf + off, peerBase + off,
                                static_cast<size_t>(H) * bytesPerPeer, dh, sig, esig, nq, 0);
          } else {
            for (int k = 0; k < H; ++k) {
              size_t off = static_cast<size_t>(myRelay + k) * slotStride;
              core::SdmaPutThread(outBuf + off, peerBase + off, bytesPerPeer, dh, sig, esig, nq, 0);
            }
          }
          shmem::ShmemQuietThread(remotePe, dest);
        }
        __syncthreads();
        if (threadLinearId == 0) __threadfence_system();
        __syncthreads();
        if (warpId < H && laneId == 0) {
          int shardPos = myRelay + warpId;
          shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
              flagsMemObj, (flagBase + static_cast<size_t>(shardPos)) * sizeof(uint64_t), &flagVal,
              8, core::atomicType::AMO_SET, remotePe, 0);
        }
        __syncthreads();
        // relay local-waits the OTHER SG's H shards
        if (warpId < H && laneId == 0) {
          int shardPos = otherRelay + warpId;
          while (core::AtomicLoadRelaxedSystem(flags + flagBase + shardPos) < flagVal) {
          }
          (void)core::AtomicLoadSeqCstSystem(flags + flagBase + shardPos);
        }
        __syncthreads();
        if (threadLinearId == 0) __threadfence_system();
        __syncthreads();
      }

      // ---- PHASE 3: relay BROADCASTS the other SG's H shards to its H-1 peers ----
      if (amRelay) {
        // warp w<H-1 -> peer at pos sgId*H+(w+1); each warp targets a DISTINCT peer.
        if (warpId < H - 1 && laneId == 0) {
          int peerPos = sgId * H + (warpId + 1);
          int remotePe = peBase + peerPos * peStride;
          anvil::SdmaQueueDeviceHandle** dh = dest->deviceHandles_d + remotePe * nq;
          HSAuint64* sig = dest->signalPtrs + remotePe * nq;
          HSAuint64* esig = dest->expectSignalsPtr + remotePe * nq;
          uint8_t* peerBase = reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset;
          if (contig) {
            size_t off = static_cast<size_t>(otherRelay) * slotStride;
            core::SdmaPutThread(outBuf + off, peerBase + off,
                                static_cast<size_t>(H) * bytesPerPeer, dh, sig, esig, nq, 0);
          } else {
            for (int k = 0; k < H; ++k) {
              size_t off = static_cast<size_t>(otherRelay + k) * slotStride;
              core::SdmaPutThread(outBuf + off, peerBase + off, bytesPerPeer, dh, sig, esig, nq, 0);
            }
          }
          shmem::ShmemQuietThread(remotePe, dest);
        }
        __syncthreads();
        if (threadLinearId == 0) __threadfence_system();
        __syncthreads();
        if (warpId < H - 1 && laneId == 0) {
          int peerPos = sgId * H + (warpId + 1);
          int remotePe = peBase + peerPos * peStride;
          for (int k = 0; k < H; ++k) {
            int shardPos = otherRelay + k;
            shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
                flagsMemObj, (flagBase + static_cast<size_t>(shardPos)) * sizeof(uint64_t), &flagVal,
                8, core::atomicType::AMO_SET, remotePe, 0);
          }
        }
        __syncthreads();
      } else {
        // non-relay: local-wait the other SG's H shards (delivered by my relay in P3)
        if (warpId < H && laneId == 0) {
          int shardPos = otherRelay + warpId;
          while (core::AtomicLoadRelaxedSystem(flags + flagBase + shardPos) < flagVal) {
          }
          (void)core::AtomicLoadSeqCstSystem(flags + flagBase + shardPos);
        }
        __syncthreads();
        if (threadLinearId == 0) __threadfence_system();
        __syncthreads();
      }
    } else {
    // Overlapped mode (h2x4==2): pipeline P2 under P1. P2 rides queue min(1,nq-1) so its
    // forward SDMA executes on a distinct link queue from P1 (queue 0), pipelining with
    // P1's drain instead of being FIFO-serialized behind it. The partner-wait also moves
    // before the P1 drain so it overlaps P1 flight rather than stacking after it. Same
    // bytes/slots/flags as the serial mode. Serial mode (h2x4==1) is unchanged (qP2==0,
    // partner-wait after drain).
    const bool olap = (h2x4 == 2);
    const int qP2 = (olap && nq > 1) ? 1 : 0;

    // Per-warp phase-1 target position: warp w<H-1 => w-th sub-group peer (skip self),
    // warp H-1 => partner, warp H => self (own slot copy).
    int p1pos = -1;
    if (warpId < H - 1) {
      int j = (warpId < sgPos) ? warpId : warpId + 1;
      p1pos = sgId * H + j;
    } else if (warpId == H - 1) {
      p1pos = partnerPos;
    } else if (warpId == H) {
      p1pos = groupPos;  // self
    }
    // ---- PHASE 1 submit ----
    if (warpId <= H && laneId == 0 && p1pos >= 0) {
      int remotePe = peBase + p1pos * peStride;
      uint8_t* dstPtr =
          reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset + myShardDstOff;
      anvil::SdmaQueueDeviceHandle** dh = dest->deviceHandles_d + remotePe * nq;
      HSAuint64* sig = dest->signalPtrs + remotePe * nq;
      HSAuint64* esig = dest->expectSignalsPtr + remotePe * nq;
      core::SdmaPutThread(myInput, dstPtr, bytesPerPeer, dh, sig, esig, nq, 0);
    }
    __syncthreads();
    // ---- PHASE 2 target position (computed here; used by both schedules) ----
    int p2pos = -1;
    if (warpId < H - 1) {
      int j = (warpId < sgPos) ? warpId : warpId + 1;
      p2pos = sgId * H + j;
    }
    if (olap) {
      // Deadlock-free "overlap" variant. Moving the partner-wait before the P1 signal
      // would deadlock: rank A's partner-wait needs partner B's P1 signal, which needs B
      // past its barrier, which needs B's own partner-wait on A's P1 signal -- a circular
      // cross-rank wait. So the P1 signal must precede the partner-wait (partners release
      // each other first), which is the serial ordering; the two phases are too tightly
      // cross-coupled to overlap the drain/wait. The only residual overlap lever is
      // routing P2 onto a distinct queue qP2 (its forward SDMA runs on a different link
      // queue than P1's queue 0). Same bytes/slots/flags; ShmemQuietThread drains all nq
      // queues. Note: routing the local-peer P2 onto queue 1 can stall under the known
      // queue-1 signal-counter liveness hazard (see ccl_kernels.hip reasm qId note), so
      // this variant is disabled by default.
      if (warpId <= H && laneId == 0 && p1pos >= 0) {
        shmem::ShmemQuietThread(peBase + p1pos * peStride, dest);
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      __syncthreads();
      if (warpId < H && laneId == 0 && p1pos >= 0) {
        shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
            flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
            core::atomicType::AMO_SET, peBase + p1pos * peStride, 0);
      }
      __syncthreads();
      if (threadLinearId == 0) {
        while (core::AtomicLoadRelaxedSystem(flags + flagBase + partnerPos) < flagVal) {
        }
        (void)core::AtomicLoadSeqCstSystem(flags + flagBase + partnerPos);
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      __syncthreads();
      if (warpId < H - 1 && laneId == 0 && p2pos >= 0) {
        int remotePe = peBase + p2pos * peStride;
        uint8_t* srcPtr = myLocalBuf + partnerDstOff;
        uint8_t* dstPtr =
            reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset + partnerDstOff;
        anvil::SdmaQueueDeviceHandle** dh = dest->deviceHandles_d + remotePe * nq;
        HSAuint64* sig = dest->signalPtrs + remotePe * nq;
        HSAuint64* esig = dest->expectSignalsPtr + remotePe * nq;
        core::SdmaPutThread(srcPtr, dstPtr, bytesPerPeer, dh, sig, esig, nq, qP2);
      }
      __syncthreads();
      if (warpId < H - 1 && laneId == 0 && p2pos >= 0) {
        shmem::ShmemQuietThread(peBase + p2pos * peStride, dest);
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      __syncthreads();
      if (warpId < H - 1 && laneId == 0 && p2pos >= 0) {
        shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
            flagsMemObj, (flagBase + static_cast<size_t>(partnerPos)) * sizeof(uint64_t), &flagVal, 8,
            core::atomicType::AMO_SET, peBase + p2pos * peStride, 0);
      }
      __syncthreads();
    } else {
      // Serial (h2x4==1): drain all phase-1 queues (incl. self), CTA fence, signal the
      // H peer targets (self carries no flag: the final wait skips groupPos), local-wait
      // partner shard, fence, then phase 2.
      if (warpId <= H && laneId == 0 && p1pos >= 0) {
        shmem::ShmemQuietThread(peBase + p1pos * peStride, dest);
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      __syncthreads();
      if (warpId < H && laneId == 0 && p1pos >= 0) {
        shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
            flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
            core::atomicType::AMO_SET, peBase + p1pos * peStride, 0);
      }
      __syncthreads();
      // ---- local wait for partner's shard ----
      if (threadLinearId == 0) {
        while (core::AtomicLoadRelaxedSystem(flags + flagBase + partnerPos) < flagVal) {
        }
        (void)core::AtomicLoadSeqCstSystem(flags + flagBase + partnerPos);
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      __syncthreads();
      // ---- PHASE 2: forward partner's shard to my H-1 sub-group peers ----
      if (warpId < H - 1 && laneId == 0 && p2pos >= 0) {
        int remotePe = peBase + p2pos * peStride;
        uint8_t* srcPtr = myLocalBuf + partnerDstOff;
        uint8_t* dstPtr =
            reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset + partnerDstOff;
        anvil::SdmaQueueDeviceHandle** dh = dest->deviceHandles_d + remotePe * nq;
        HSAuint64* sig = dest->signalPtrs + remotePe * nq;
        HSAuint64* esig = dest->expectSignalsPtr + remotePe * nq;
        core::SdmaPutThread(srcPtr, dstPtr, bytesPerPeer, dh, sig, esig, nq, 0);
      }
      __syncthreads();
      if (warpId < H - 1 && laneId == 0 && p2pos >= 0) {
        shmem::ShmemQuietThread(peBase + p2pos * peStride, dest);
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      __syncthreads();
      if (warpId < H - 1 && laneId == 0 && p2pos >= 0) {
        shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
            flagsMemObj, (flagBase + static_cast<size_t>(partnerPos)) * sizeof(uint64_t), &flagVal, 8,
            core::atomicType::AMO_SET, peBase + p2pos * peStride, 0);
      }
      __syncthreads();
    }
    }
  } else if (ringPhasedActive) {
    // Width-W permutation issue schedule (ringPhased carries the width W). The plain
    // crown (the else branches below) submits all G peer copies at once, so every
    // receiver takes a G-1 concurrent XGMI write incast. This issues the G copies in
    // ceil(G/W) rotated phases of W concurrent links each, with a CTA barrier between
    // phases: warp w (peer w) belongs to phase ((w-groupPos+G)%G)/W, so across all ranks
    // each phase is a W-regular matching (every receiver takes exactly W concurrent
    // writers per phase) -- the middle ground between full incast (W>=G-1) and full
    // serialization (W==1). The per-peer completion (drain + __threadfence_system + AMO)
    // is deferred to the shared tail below and runs identically for every warp, so this
    // changes only the submission stagger: same bytes to slot groupPos on the same
    // queue 0, same expectedSignals accounting, same tail acquire.
    int W = ringPhased;
    if (W < 1) W = 1;
    if (W > groupSize) W = groupSize;
    const int numPhases = (groupSize + W - 1) / W;
    size_t destByteOffset = static_cast<size_t>(groupPos) * slotStride;
    application::SymmMemObjPtr dest = dstMemObj;
    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(inputData);
    // Per-warp phase index (only meaningful for the active issuing warps).
    int myPhase = (warpId < groupSize) ? (((warpId - groupPos + groupSize) % groupSize) / W) : -1;
    for (int ph = 0; ph < numPhases; ++ph) {
      if (warpId < groupSize && laneId == 0 && myPhase == ph) {
        int remotePe = peBase + warpId * peStride;
        uint8_t* dstPtr =
            reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset + destByteOffset;
        anvil::SdmaQueueDeviceHandle** devicehandles =
            dest->deviceHandles_d + remotePe * dest->sdmaNumQueue;
        HSAuint64* signals = dest->signalPtrs + remotePe * dest->sdmaNumQueue;
        HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * dest->sdmaNumQueue;
        core::SdmaPutThread(srcPtr, dstPtr, bytesPerPeer, devicehandles, signals, expectedSignals,
                            dest->sdmaNumQueue, 0);
      }
      __syncthreads();
    }
  } else if (intra2Active) {
    // ============ 2x4 stacked-flat-body intra ============
    // See HierIntra2. The plain crown fires all G-1 peer copies concurrently (flat full
    // mesh: per-receiver incast G-1, per-PE egress G-1), which the SDMA engines cannot
    // saturate (the 4-way width does). This issues the same flat push in ceil(G/W)
    // sequential waves of a W-regular rotated matching: warp w (peer w) is active in wave
    // ((w-groupPos+G)%G)/W, so across ranks each wave is a perfect W-matching (every
    // receiver has exactly W concurrent writers this wave). Unlike the width-W ringPhased
    // path (which defers completion to the shared tail so all G flows overlap in flight),
    // each wave drains to completion (per-peer ShmemQuiet + system fence + flag AMO)
    // before the next wave submits, so at most W XGMI egress links are in flight at once
    // (per-PE egress and per-receiver incast reduced G-1 -> W). W==4 at G==8 is the 2x4.
    // Same bytes into slot groupPos of the same G peers, same per-peer drain+fence+flag;
    // only the issue is serialized into drained W-wide waves. The final completion wait
    // (full [0,G)) below is unchanged.
    int W = intra2;
    if (W < 1) W = 1;
    if (W > groupSize) W = groupSize;
    const int numPhases = (groupSize + W - 1) / W;
    const size_t destByteOffset = static_cast<size_t>(groupPos) * slotStride;
    application::SymmMemObjPtr dest = dstMemObj;
    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(inputData);
    const int nq = dest->sdmaNumQueue > 0 ? static_cast<int>(dest->sdmaNumQueue) : 1;
    const int myPhase =
        (warpId < groupSize) ? (((warpId - groupPos + groupSize) % groupSize) / W) : -1;
    for (int ph = 0; ph < numPhases; ++ph) {
      // ---- SUBMIT wave ph (W-regular matching) ----
      if (warpId < groupSize && laneId == 0 && myPhase == ph) {
        int remotePe = peBase + warpId * peStride;
        uint8_t* dstPtr =
            reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset + destByteOffset;
        anvil::SdmaQueueDeviceHandle** dh = dest->deviceHandles_d + remotePe * nq;
        HSAuint64* sig = dest->signalPtrs + remotePe * nq;
        HSAuint64* esig = dest->expectSignalsPtr + remotePe * nq;
        core::SdmaPutThread(srcPtr, dstPtr, bytesPerPeer, dh, sig, esig, nq, 0, putNdesc);
      }
      __syncthreads();
      // ---- DRAIN wave ph to completion BEFORE the next wave submits ----
      if (warpId < groupSize && laneId == 0 && myPhase == ph) {
        int remotePe = peBase + warpId * peStride;
        shmem::ShmemQuietThread(remotePe, dest);
      }
      __syncthreads();
      if (threadLinearId == 0) __threadfence_system();
      __syncthreads();
      if (warpId < groupSize && laneId == 0 && myPhase == ph) {
        int remotePe = peBase + warpId * peStride;
        shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
            flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
            core::atomicType::AMO_SET, remotePe, 0);
      }
      __syncthreads();
    }
  } else if (multiQueue != 0 && warpId < groupSize && dstMemObj->sdmaNumQueue > 1) {
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
    // Whole warp active: SdmaPutWarp uses lanes [0, sdmaNumQueue) to split the copy.
    core::SdmaPutWarp(srcPtr, dstPtr, bytesPerPeer, devicehandles, signals, expectedSignals,
                      dest->sdmaNumQueue);
  } else if (warpId >= pLo && warpId < pHi && laneId == 0) {
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
    if (qFlagActive) {
      // Push this peer's column AND its completion flag as ONE FIFO-ordered SDMA
      // queue sequence (COPY_LINEAR + FENCE) -- no send-CQ drain, no system fence,
      // no separate P2P AMO. The flag lands strictly after its bytes on the same
      // engine (bit-exact). Skips the drain/fence/AMO tail block below for this peer.
      uint64_t* pf =
          reinterpret_cast<uint64_t*>(flagsMemObj->peerPtrs[remotePe]) + (flagBase + groupPos);
      core::SdmaPutFencedFlagThread(srcPtr, dstPtr, bytesPerPeer, devicehandles, signals,
                                    expectedSignals, dest->sdmaNumQueue, 0, pf,
                                    static_cast<uint32_t>(flagVal));
    } else {
      // Descriptor pipelining on the local-gather pole. putNdesc>1 splits the per-peer
      // COPY_LINEAR into contiguous back-to-back sub-descriptors in one doorbell so the
      // engine's descriptor-fetch latency is hidden across sub-copies -- a probe of
      // whether the pole is descriptor-latency-bound vs XGMI-egress-bound. putNdesc==1
      // leaves the path unchanged.
      // Pole SQ depth. poleSqDepth>1 issues K independent back-to-back copies (K
      // doorbells / K COPY+ATOMIC work items) on this peer's queue 0 -- a genuine SQ
      // occupancy of K in-flight whole-copies so the engine never idles between a
      // completed COPY_LINEAR and the next fetch+launch. Distinct from putNdesc (one
      // doorbell, K sub-descriptors of one copy). Disjoint contiguous chunks of the same
      // bytes, each bumps expectedSignals so the shared ShmemQuiet tail drains all K
      // before the flag AMO. poleSqDepth==1 leaves the path unchanged.
      if (poleSqDepth > 1) {
        size_t K = static_cast<size_t>(poleSqDepth);
        const size_t unit = 16;
        const size_t nU = (bytesPerPeer + unit - 1) / unit;
        if (K > nU) K = (nU < 1) ? 1 : nU;
        const size_t uPerC = (nU + K - 1) / K;
        for (size_t c = 0; c < K; ++c) {
          size_t s = c * uPerC * unit;
          if (s >= bytesPerPeer) break;
          size_t e = s + uPerC * unit;
          if (e > bytesPerPeer) e = bytesPerPeer;
          core::SdmaPutThread(srcPtr + s, dstPtr + s, e - s, devicehandles, signals,
                              expectedSignals, dest->sdmaNumQueue, 0, 1);
        }
      } else {
        // Forward the COPY_LINEAR DW2 coherence hint (putCacheHint) onto the local-gather
        // pole push. putCacheHint==0 leaves the path unchanged.
        core::SdmaPutThread(srcPtr, dstPtr, bytesPerPeer, devicehandles, signals, expectedSignals,
                            dest->sdmaNumQueue, 0, putNdesc, putCacheHint);
      }
    }
  }

  // Shared per-peer completion tail: runs for both the plain full-mesh issue and the
  // width-W phased issue (only the qFlag path rode its flag inline on the SDMA queue and
  // skips this). Each warp drains its own peer's queue then bumps that peer's completion
  // flag -- parallel across warps, identical for either issue schedule (the phased path
  // already CTA-barriered all submissions above).
  if (!qFlagActive && !batchFenceActive && !h2x4Active && !intra2Active && warpId >= pLo &&
      warpId < pHi && laneId == 0) {
    int remotePe = peBase + warpId * peStride;
    shmem::ShmemQuietThread(remotePe, dstMemObj);
    // Sender-side completion fence: ShmemQuietThread drains the SDMA submission queue,
    // but the peer-visible ordering of the raw SDMA data writes vs the subsequent flag
    // AMO is not otherwise guaranteed at system scope. Without it the receiver can
    // observe the flag (and race past its own threadfence_system) before the pushed
    // bytes are globally ordered.
    __threadfence_system();
    shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
        flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
        core::atomicType::AMO_SET, remotePe, 0);
  } else if (batchFenceActive) {
    // Batched sender fence. Split the per-peer tail into three CTA-wide phases so the G
    // concurrent __threadfence_system collapse to one:
    //   (1) every warp drains its own peer's SDMA queue (parallel, unchanged),
    //   (2) CTA barrier, then one thread issues a single __threadfence_system,
    //   (3) CTA barrier, then every warp publishes its own peer's flag AMO.
    // Strictly stronger ordering than the per-peer path (the one fence happens after all
    // drains and before all AMOs).
    if (warpId < groupSize && laneId == 0) {
      int remotePe = peBase + warpId * peStride;
      shmem::ShmemQuietThread(remotePe, dstMemObj);
    }
    __syncthreads();
    if (threadLinearId == 0) __threadfence_system();
    __syncthreads();
    if (warpId < groupSize && laneId == 0) {
      int remotePe = peBase + warpId * peStride;
      shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
          flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
          core::atomicType::AMO_SET, remotePe, 0);
    }
  }
  __syncthreads();

  // Completion wait. Thread t acquires sender-position t's flag concurrently (a seq-cst
  // SYSTEM load = agent-wide acquire, so once observed the peer's SDMA data writes are
  // coherently visible to this GPU agent), then one block barrier publishes those
  // acquired flags happens-before all consumers and thread 0 issues a single
  // __threadfence_system. Every peer flag is still system-scope acquired before the
  // barrier releases the block; only the redundant per-sender barriers/fences (G-1 -> 1)
  // and the serial post-arrival loads (collapsed to one concurrent round) are removed.
  // System-scope acquire is required because the flag and the data it guards are written
  // by a remote peer GPU (a different HSA agent) via SDMA; an agent-scope relaxed load
  // gives no cross-agent happens-before, so a system acquire + system fence is what makes
  // the peer's data visible without a host sync.
  {
    int senderPos = static_cast<int>(threadIdx.x);
    if (senderPos < groupSize && senderPos != groupPos) {
      int spinCount = 0;
      bool warned = false;
      // LIGHT-SPIN completion (see peer-wait above): relaxed system spin + one
      // seq-cst SYSTEM acquire on exit -> same acquire contract, far less
      // per-iteration coherence traffic competing with the copy engines.
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
// PHASE 4: pipelined relay ring sub-group reassembly
// ---------------------------------------------------------------------------
// The push-only reassembly below is a flat G-way scatter: every rank writes its own
// column into slot ``groupPos`` of all G members, so each receiver is written by G
// senders at once (a G-way XGMI incast). The copy engines cannot saturate the mesh
// under G-way incast, while a ring's perfect-matching-per-step pattern can. This
// primitive reassembles one remote block's sub-range with the non-contending ring
// pattern instead:
//   - Ring order: successor = (groupPos+1)%G, predecessor = (groupPos-1+G)%G.
//   - Rank g starts holding its own shard s0 = groupPos (staged into the output
//     self-slot; for direct-land it is already there via the inter-node RDMA WRITE).
//   - For step k = 0..G-2: send the shard s = (groupPos-k+G)%G that this rank now
//     holds (in its own output slot m*G+s) to the SUCCESSOR's same output slot, then
//     AMO_SET flag[flagBase+s] on the successor. Before step k>0 wait on THIS rank's
//     own flag[flagBase+s] (the predecessor delivered s in its step k-1).
// Each step is a perfect matching (1 outbound + 1 inbound per rank) => no incast,
// full per-link BW. Deadlock-free: rank g trails its predecessor by exactly one
// step (a time-DAG), and worker j runs the SAME f on every GPU so the predecessor's
// setter always exists. Coverage: shard s originates at rank s and is relayed to all
// G-1 non-owners, so every receiver R ends with flag[flagBase+s] set for all s!=R
// (via the ring) and s==R (set locally) -> the SAME G completion flags the reader
// waits for. Bit-exact final layout (output[m*G+s] holds shard s). Single-thread
// driver (threadIdx.x==0); spins are BOUNDED (timeout->break => mismatch, never a
// cluster-wedging hang). ``ringSelfSrc`` != nullptr stages the self shard from the
// ring buffer (non-direct-land); nullptr => self slot already holds it (direct-land).
template <typename T>
__device__ void OneShotSubGroupPushRing_body(
    int groupSize, int groupPos, int peBase, int peStride,
    const application::SymmMemObjPtr dstMemObj, const application::SymmMemObjPtr flagsMemObj,
    size_t blockBaseBytes, size_t subOff, size_t subBytes, size_t slotStride, uint64_t flagVal,
    size_t flagBase, int qId, const T* ringSelfSrc) {
  if (threadIdx.x != 0 || subBytes == 0 || groupSize <= 0) return;
  const int G = groupSize;
  const int myPos = groupPos;
  const int succPos = (myPos + 1) % G;
  const int succPe = peBase + succPos * peStride;
  application::SymmMemObjPtr dest = dstMemObj;
  const int nq = dest->sdmaNumQueue > 0 ? static_cast<int>(dest->sdmaNumQueue) : 1;
  const int q = (qId >= 0) ? (qId % nq) : 0;
  uint64_t* myFlags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
  uint8_t* myOut = reinterpret_cast<uint8_t*>(dest->localPtr);
  uint8_t* succOut = reinterpret_cast<uint8_t*>(dest->peerPtrs[succPe]);
  anvil::SdmaQueueDeviceHandle** dhS = dest->deviceHandles_d + succPe * nq;
  HSAuint64* sigS = dest->signalPtrs + succPe * nq;
  HSAuint64* esigS = dest->expectSignalsPtr + succPe * nq;
  // Stage my own shard (s == myPos) into my output self-slot if it isn't there yet.
  if (ringSelfSrc != nullptr) {
    const size_t selfByte = blockBaseBytes + static_cast<size_t>(myPos) * slotStride + subOff;
    const int selfPe = peBase + myPos * peStride;
    anvil::SdmaQueueDeviceHandle** dhSelf = dest->deviceHandles_d + selfPe * nq;
    HSAuint64* sigSelf = dest->signalPtrs + selfPe * nq;
    HSAuint64* esigSelf = dest->expectSignalsPtr + selfPe * nq;
    core::SdmaPutThread(reinterpret_cast<uint8_t*>(const_cast<T*>(ringSelfSrc)), myOut + selfByte,
                        subBytes, dhSelf, sigSelf, esigSelf, nq, q);
    core::SdmaQueitThread(sigSelf + q, esigSelf + q, 1);
  }
  __threadfence_system();
  core::AtomicStoreSeqCstSystem(myFlags + flagBase + myPos, flagVal);  // self shard present
  for (int k = 0; k < G - 1; ++k) {
    const int s = ((myPos - k) % G + G) % G;
    if (k > 0) {
      long spin = 0;
      while (core::AtomicLoadSeqCstSystem(myFlags + flagBase + s) < flagVal) {
        if (++spin > 200000000L) break;  // BOUNDED: timeout -> mismatch, never a hang
      }
      __threadfence_system();
    }
    const size_t off = blockBaseBytes + static_cast<size_t>(s) * slotStride + subOff;
    core::SdmaPutThread(myOut + off, succOut + off, subBytes, dhS, sigS, esigS, nq, q);
    core::SdmaQueitThread(sigS + q, esigS + q, 1);
    __threadfence_system();
    shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
        flagsMemObj, (flagBase + static_cast<size_t>(s)) * sizeof(uint64_t), &flagVal, 8,
        core::atomicType::AMO_SET, succPe, 0);
  }
}

// ---------------------------------------------------------------------------
// PHASE 4: PUSH-ONLY sub-group reassembly (deadlock-free, parallel)
// ---------------------------------------------------------------------------
// The remote-block reassembly of the hierarchical AllGather is an intra-node
// sub-group gather among the ``G`` local ranks: rank ``groupPos`` owns one
// column (its own ring-buffer slice of block ``m``) and must place it into slot
// ``groupPos`` of block ``m`` in EVERY member's output. Crucially the output
// slots are DISJOINT per rank, so no rank ever READS another rank's data --
// each rank only WRITES its own column. That means the cross-rank flag WAIT that
// the collective gather (OneShotAllGatherSdmaSubGroupKernel_body) performs is not
// needed for the push itself; it only serves as a completion barrier. Coupling
// push+wait inside every (block,remote) gather is exactly what dead-locks when
// several reassembly blocks run concurrently (each block's wait spins on peers
// whose matching block may not be co-resident -> circular stall).
//
// This primitive does the WRITE half ONLY: warp ``w`` SDMA-pushes this rank's
// ``elementCount``-lane slice into slot ``groupPos`` of member ``w``'s output at
// ``dstBaseOffset + groupPos*slotStride``, quiets ONLY its own SDMA queue
// ``qId`` (never drains-all, so concurrent blocks on other queues are
// unaffected), then bumps completion flag ``flagBase+groupPos`` on member ``w``.
// It NEVER waits on a peer, so any number of these run concurrently without
// dead-lock. A SINGLE completion reader (the fused kernel's local-block CTA)
// later waits for every (block,remote,sender) flag AFTER all pushes are issued.
// ``qId`` MUST be distinct across blocks that target the same peer so the
// per-queue signal counter (expectedSignals[qId]) is never raced; callers clamp
// the number of concurrent reassembly blocks to the peer's SDMA queue count.
template <typename T>
__device__ void OneShotSubGroupPushOnly_body(int groupSize, int groupPos, int peBase, int peStride,
                                             T* input, const application::SymmMemObjPtr dstMemObj,
                                             const application::SymmMemObjPtr flagsMemObj,
                                             size_t elementCount, size_t dstBaseOffset,
                                             size_t dstSlotStrideBytes, uint64_t flagVal,
                                             size_t flagBase, int qId, int deepSqPhase = 0,
                                             int qSplit = 0, int qStride = 1, int fuseFence = 0,
                                             int ndesc = 1, int pushRotate = 0, int qFlag = 0,
                                             int multiQueue = 0, int pushPhased = 0,
                                             int pushPeerLo = 0, int pushPeerHi = -1,
                                             int pushFlag = 1, int skipSelfCopy = 0) {
  // Intra reassembly deep-SQ phase split (see HierReasmDeepSqOn). deepSqPhase==0
  // (default) = the single-shot path (submit -> drain -> fence -> flag). ==1 =
  // submit-only: push the SDMA copy + bump expectedSignals[q] but don't drain/fence/
  // fire the flag, so a worker can enqueue all its owned channels' copies back-to-back
  // and keep the copy engine continuously fed instead of drain-idling per channel. ==2 =
  // drain+flag-only: no submit; a single quiet to the accumulated expected count covers
  // every phase-1 back-to-back submit on this queue (SQ FIFO, signal monotonic), then
  // fence + fire the flag after the drain, so the output flag never precedes its bytes.
  if (elementCount == 0 || groupSize <= 0) {
    return;
  }
  const size_t threadLinearId = static_cast<size_t>(threadIdx.x);  // block-local
  const size_t bytesPerElement = sizeof(T);
  const size_t bytesPerPeer = elementCount * bytesPerElement;
  const size_t slotStride = dstSlotStrideBytes != 0 ? dstSlotStrideBytes : bytesPerPeer;
  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;
  // FIRST-LAND idle-engine reclamation (see HierLocalOffload): restrict this push to
  // target peer-columns [pLoP, pHiP). Default (0, -1) => full [0,groupSize) ==
  // byte-identical. Used ONLY by the reasm-CTA local-offload helper (plain path).
  const int pLoP = (pushPeerLo > 0) ? pushPeerLo : 0;
  const int pHiP = (pushPeerHi >= 0 && pushPeerHi < groupSize) ? pushPeerHi : groupSize;
  // native ring-stagger of the warp->peer map (see HierPushRotateOn). Pure
  // permutation: warp w targets peer effWarp = (w+groupPos) % groupSize, so each
  // rank starts on a different destination -> spreads the per-cycle XGMI port load.
  // pushRotate==0 => effWarp==warpId => byte-identical peer assignment.
  const int effWarp = (pushRotate != 0) ? ((warpId + groupPos) % groupSize) : warpId;

  // Multi-engine per-link put (see HierReasmMultiQueueOn / MORI_INTRA_MQ). Drive all
  // sdmaNumQueue XGMI SDMA engines per link for this peer's column with the full warp
  // (lane k -> queue k over a disjoint contiguous sub-range) instead of the single-lane
  // single-queue put. Only the single-shot completion path is eligible (no deepSq /
  // qSplit / qFlag / fuseFence -- those keep their own queue accounting). Disjoint sub-
  // ranges of the same bytes, and all nqTop queues drained before the flag AMO fires so
  // the flag never precedes any sub-copy.
  const int nqTop = dstMemObj->sdmaNumQueue > 0 ? static_cast<int>(dstMemObj->sdmaNumQueue) : 1;
  const bool mqActive = (multiQueue != 0 && nqTop > 1 && deepSqPhase == 0 && qSplit == 0 &&
                         qFlag == 0 && fuseFence == 0);
  // PHASED-PERMUTATION PUSH (see HierPushPhasedOn): serialise the groupSize peer
  // copies into groupSize ROTATED phases -- in phase p push ONLY to peer
  // (groupPos+1+p)%groupSize, with a CTA barrier between phases -- so across ranks
  // every phase is a perfect matching (each receiver has exactly one writer this
  // phase), killing the all-to-all XGMI incast on the dominant reasm leg. Single-shot
  // completion path only. Block-uniform gate => the per-phase __syncthreads() is not
  // divergent. Bit-exact: same column bytes into slot groupPos of the same G peers
  // with the same per-peer drain+fence+flag; only issue order/overlap changes.
  const bool phasedActive = (pushPhased != 0 && deepSqPhase == 0 && qSplit == 0 &&
                             qFlag == 0 && fuseFence == 0 && !mqActive);
  if (phasedActive) {
    const int nq = dstMemObj->sdmaNumQueue > 0 ? static_cast<int>(dstMemObj->sdmaNumQueue) : 1;
    application::SymmMemObjPtr dest = dstMemObj;
    uint8_t* srcPtr = reinterpret_cast<uint8_t*>(input);
    const size_t destByteOffset = static_cast<size_t>(groupPos) * slotStride;
    for (int p = 0; p < groupSize; ++p) {
      const int peer = (groupPos + 1 + p) % groupSize;
      if (warpId == 0 && laneId == 0) {
        const int remotePe = peBase + peer * peStride;
        const int q = (peer % nq + nq) % nq;
        uint8_t* dstPtr =
            reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset + destByteOffset;
        anvil::SdmaQueueDeviceHandle** devicehandles = dest->deviceHandles_d + remotePe * nq;
        HSAuint64* signals = dest->signalPtrs + remotePe * nq;
        HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * nq;
        core::SdmaPutThread(srcPtr, dstPtr, bytesPerPeer, devicehandles, signals, expectedSignals,
                            nq, q, ndesc);
        core::SdmaQueitThread(signals + q, expectedSignals + q, 1);
        __threadfence_system();
        shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
            flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
            core::atomicType::AMO_SET, remotePe, 0);
      }
      __syncthreads();
    }
    return;
  }
  if (mqActive) {
    if (warpId < groupSize) {
      int remotePe = peBase + effWarp * peStride;
      size_t destByteOffset = static_cast<size_t>(groupPos) * slotStride;
      application::SymmMemObjPtr dest = dstMemObj;
      uint8_t* srcPtr = reinterpret_cast<uint8_t*>(input);
      uint8_t* dstPtr =
          reinterpret_cast<uint8_t*>(dest->peerPtrs[remotePe]) + dstBaseOffset + destByteOffset;
      anvil::SdmaQueueDeviceHandle** devicehandles = dest->deviceHandles_d + remotePe * nqTop;
      HSAuint64* signals = dest->signalPtrs + remotePe * nqTop;
      HSAuint64* expectedSignals = dest->expectSignalsPtr + remotePe * nqTop;
      // Full warp active: SdmaPutWarp uses lanes [0, nqTop) to split the copy across
      // every recommended engine on the link.
      core::SdmaPutWarp(srcPtr, dstPtr, bytesPerPeer, devicehandles, signals, expectedSignals,
                        nqTop);
      if (laneId == 0) {
        // Drain ALL nqTop queues (every sub-copy landed), fence, then fire the ONE
        // completion flag -- the flag never precedes any sub-copy (bit-exact).
        core::SdmaQueitThread(signals, expectedSignals, nqTop);
        __threadfence_system();
        shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
            flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
            core::atomicType::AMO_SET, remotePe, 0);
      }
    }
  } else if (warpId >= pLoP && warpId < pHiP && laneId == 0) {
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
    // PER-PEER QUEUE SPLIT (see HierReasmQSplitOn): single-shot path only. Split
    // this column across this worker's OWN disjoint per-peer queue class
    // {k in [0,nq): k % qs == base} (base = qId % qs, qs = qStride = effReasm) so
    // idle per-peer queues (nq>effReasm) join the copy. Worker-disjoint queue set
    // => no cross-worker same-queue race; ALL owned queues drained before the flag
    // => the flag never precedes its bytes (bit-exact). qSplit==0 => shipped path.
    if (qFlag != 0 && deepSqPhase == 0 && qSplit == 0 && fuseFence == 0) {
      // COPY-ENGINE FLAG DELIVERY (see SdmaPutFencedFlagThread / MORI_HIER_QFLAG):
      // push this column AND its peer completion flag as one FIFO-ordered queue
      // sequence (COPY_LINEAR + FENCE) -- NO per-peer send-CQ drain, NO
      // __threadfence_system, NO separate direct P2P AMO. The flag write is
      // FIFO-ordered after its data on the same engine so the completion reader
      // never sees the flag ahead of the bytes (bit-exact). This strips the
      // per-peer drain/system-fence/AMO round-trips from the 8x7 all-to-all
      // completion critical path -- the native copy-engine completion model.
      uint64_t* peerFlag =
          reinterpret_cast<uint64_t*>(flagsMemObj->peerPtrs[remotePe]) + (flagBase + groupPos);
      core::SdmaPutFencedFlagThread(srcPtr, dstPtr, bytesPerPeer, devicehandles, signals,
                                    expectedSignals, nq, q, peerFlag,
                                    static_cast<uint32_t>(flagVal));
      // fall through past the legacy single-queue path below (no drain/fence/flag).
    } else if (qSplit != 0 && nq > 1 && deepSqPhase == 0) {
      const int qs = (qStride > 0) ? qStride : 1;
      const int base = ((qId % qs) + qs) % qs;
      int nOwned = 0;
      for (int k = base; k < nq; k += qs) ++nOwned;
      if (nOwned < 1) nOwned = 1;
      const size_t unit = 16;
      const size_t nU = (bytesPerPeer + unit - 1) / unit;
      const size_t uPerQ = (nU + static_cast<size_t>(nOwned) - 1) / static_cast<size_t>(nOwned);
      // Submit all owned-queue sub-copies back-to-back (feed the engines), then
      // drain every owned queue, fence, and fire the single completion flag.
      int idx = 0;
      for (int k = base; k < nq; k += qs) {
        const size_t s = static_cast<size_t>(idx) * uPerQ * unit;
        size_t e = s + uPerQ * unit;
        if (e > bytesPerPeer) e = bytesPerPeer;
        if (s < e) {
          core::SdmaPutThread(srcPtr + s, dstPtr + s, e - s, devicehandles, signals,
                              expectedSignals, nq, k);
        }
        ++idx;
      }
      for (int k = base; k < nq; k += qs) {
        core::SdmaQueitThread(signals + k, expectedSignals + k, 1);
      }
      __threadfence_system();
      shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
          flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
          core::atomicType::AMO_SET, remotePe, 0);
      // fall through past the legacy single-queue path below.
    } else if (fuseFence != 0 && deepSqPhase == 0) {
      // BATCHED SENDER-QUIET (see HierFuseSenderFence): push + drain THIS warp's
      // peer copy here, but DEFER the __threadfence_system + completion flag AMO
      // to a second phase AFTER a block barrier. The shipped path interleaves
      // drain->fence->flag inside each per-peer warp so the G peer copies quiet
      // in warp order with a fence between every one; batching lets all G copies
      // drain first, then a single completion phase fences+fires every flag --
      // the "8x7 in-flight quiet" the completion-model study named. Bit-exact BY
      // CONSTRUCTION: every copy is drained before ANY flag fires, and each warp
      // still issues its own __threadfence_system before its flag => the flag
      // never precedes its bytes; only the drain/fence/flag ORDER across peers
      // changes, not which bytes any flag guards.
      core::SdmaPutThread(srcPtr, dstPtr, bytesPerPeer, devicehandles, signals, expectedSignals,
                          nq, q);
      core::SdmaQueitThread(signals + q, expectedSignals + q, 1);
      // fence + flag deferred to the post-barrier completion phase below.
    } else {
    // DIRECT-LAND self-column skip (see skipSelfCopy): the self output slot was
    // filled by the inter-node RDMA WRITE (data landed + fenced before this worker
    // ran), so this warp does NO SDMA copy/drain -- it only fences + fires the self
    // completion flag so the reader still sees source==groupPos delivered. Non-self
    // warps copy normally (their source is now the output self-slot, same bytes).
    const bool selfSkip = (skipSelfCopy != 0 && effWarp == groupPos);
    // Push this rank's column on its OWN queue ``q`` (distinct per block).
    // deepSqPhase==2 is DRAIN-only (the copy was already submitted in phase 1).
    if (deepSqPhase != 2 && !selfSkip) {
      // ndesc>1 (single-shot crown path only) pipelines the per-peer copy across
      // several back-to-back descriptors on queue q (see HierPutNdesc). Safe for
      // deepSqPhase 0/1: the accumulated signal count still equals one bump/push.
      core::SdmaPutThread(srcPtr, dstPtr, bytesPerPeer, devicehandles, signals, expectedSignals,
                          nq, q, ndesc);
    }
    // deepSqPhase==1 is SUBMIT-only: skip the drain/fence/flag so the next
    // channel's copy is fed without a per-descriptor drain round-trip.
    if (deepSqPhase != 1) {
      // Drain ONLY queue ``q`` (single-queue quiet) so the copy has landed before
      // the flag fires, WITHOUT touching other blocks' queues. In deep-SQ phase 2
      // expectedSignals[q] already holds the accumulated count of every phase-1
      // submit on this queue, so this single wait covers them all.
      if (!selfSkip) core::SdmaQueitThread(signals + q, expectedSignals + q, 1);
      // SENDER-SIDE completion fence (see collective body above): system-scope
      // order the pushed SDMA bytes BEFORE the flag AMO becomes peer-visible, so
      // the completion reader cannot see the flag ahead of the data. Closes the
      // copy-engine visibility gap that previously needed a host stream sync.
      __threadfence_system();
      // The flag AMO is a DIRECT peer-memory CAS (P2P), not an SDMA-queue signal,
      // so it never races the per-queue counter. Bump slot ``flagBase+groupPos``.
      // pushFlag==0 (non-last tile of the read-coalescing tile loop) suppresses only the
      // flag AMO -- the copy is still drained+fenced this tile, and the flag fires only on
      // the last tile after all bytes have drained.
      if (pushFlag) {
        shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
            flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
            core::atomicType::AMO_SET, remotePe, 0);
      }
    }
    }  // end legacy single-queue path (else of qSplit)
  }
  // BATCHED SENDER-QUIET completion phase (fuseFence): after the barrier every
  // warp's peer copy has drained; now each peer warp fences once + fires its flag.
  // Uniform across the block (fuseFence/deepSqPhase are block-uniform) so the
  // barrier is not divergent. No-op when fuseFence==0 => byte-identical path.
  if (fuseFence != 0 && deepSqPhase == 0) {
    __syncthreads();
    if (warpId < groupSize && laneId == 0) {
      int remotePe = peBase + effWarp * peStride;
      __threadfence_system();
      shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
          flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
          core::atomicType::AMO_SET, remotePe, 0);
    }
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// Fused hierarchical param-contiguous SubGroup gather (ONE launch)
// ---------------------------------------------------------------------------
// Replaces HierAllGather.enqueue_param_contiguous's N_nodes*N_params separate
// SubGroup launches with a single launch: warp ``w`` drives destination member
// ``w``; this PE (group position ``g == groupPos``) pushes, for every node block
// ``m`` and every param split ``s``, its E_s-element sub-slice from the Phase-A
// collection into the member's registered output at param-contiguous element
// offset ``O_s*W + (m*G+g)*E_s``. Same subgroup flags as the per-slot direct
// gather: bump slot ``g`` on each member once, then wait for all G members.
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
      uint8_t* blkSrc =
          reinterpret_cast<uint8_t*>(input) + static_cast<size_t>(i) * blockStrideElems * bytesPerElement;
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
  // flag (mirrors the proven flat OneShotAllGatherSdmaParamContiguousKernel_body).
  __syncthreads();

  if (warpId < groupSize && laneId == 0) {
    int remotePe = peBase + warpId * peStride;
    shmem::ShmemQuietThread(remotePe, dstMemObj);
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
// One source ("root", group position 0 == global PE ``peBase``) holds a full
// buffer of ``elementCount`` u32 lanes in ``input``; this kernel SDMA-copies
// that whole buffer (over XGMI / P2P copy engines) into the ``dstMemObj`` of
// every member of the arithmetic sub-group
// ``{peBase, peBase+peStride, ..., peBase+(groupSize-1)*peStride}`` -- including
// the root itself, so every member ends with the full buffer in ``dstMemObj``.
//
// This is the intra-node *placement* phase of the hierarchical AllGather's
// leader-only variant (DESIGN.md's primary suggestion): the node leader
// (local_rank 0) runs the inter-node RDMA ring into a staging buffer, then
// broadcasts the full ``N*G`` output to the node's ``G`` local ranks via the
// SDMA copy engines. Compared to the "every-rank-direct" decomposition (where
// all ``G`` local ranks independently ring their node-block over the NIC, i.e.
// ``G x`` redundant inter-node traffic), the leader-only ring + this broadcast
// crosses the NIC only once per node-block, cutting NIC traffic ~``G x`` at the
// price of one extra fast XGMI hop.
//
// Root warp ``w`` handles member ``w`` (remotePe = peBase + w*peStride): a
// single SDMA put of the whole buffer, then quiet + a single-slot flag bump on
// that member. Each non-root member spins on flag slot 0 until the root's
// monotonic token arrives. The flag is a single slot (one source) with a
// per-call token, so successive calls stay race-free without a reset.
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
      shmem::ShmemQuietThread(remotePe, dstMemObj);
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
    shmem::ShmemQuietThread(remotePe, dstMemObj);
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
      // LIGHT-SPIN completion (see subgroup gather): busy-wait on a RELAXED system
      // load, then ONE seq-cst SYSTEM acquire on exit. Relaxed is still cross-agent
      // visible (monotonic flags), the trailing seq-cst load reproduces the exact
      // acquire the shipped seq-cst spin took on its last iteration, and the
      // block-level __threadfence_system below still publishes it => bit-exact.
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
