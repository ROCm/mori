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

  for (int sender = 0; sender < npes; ++sender) {
    if (sender == myPe) {
      continue;
    }

    if (threadLinearId == 0) {
      // Keep waiting for the peer completion flag. A finite spin threshold can
      // produce false timeouts under heavy traffic and cause incorrect forward
      // progress (kernel continues before data is actually ready).
      int spinCount = 0;
      bool warned = false;
      // SYSTEM-scope acquire (see subgroup gather): remote-agent SDMA producer.
      while (core::AtomicLoadSeqCstSystem(flags + sender) < flagVal) {
        ++spinCount;
        if (!warned && spinCount > 10000000) {
          printf("PE %d: Slow wait for data from peer %d (still waiting)\n", myPe, sender);
          warned = true;
        }
      }
      __threadfence_system();
    }
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
// overlap lever -- the un-refuted path to RCCL parity this work proved out, ported
// here; see all_gather.hpp
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
template <typename T>
__device__ void OneShotAllGatherSdmaSubGroupKernel_body(
    int myPe, int npes, int groupSize, int groupPos, int peBase, int peStride, T* input,
    const application::SymmMemObjPtr dstMemObj, const application::SymmMemObjPtr flagsMemObj,
    size_t elementCount, size_t dstBaseOffset = 0, size_t dstSlotStrideBytes = 0,
    uint64_t flagVal = 1, bool blockLocal = false, size_t flagBase = 0) {
  (void)npes;
  if (elementCount == 0 || groupSize <= 0) {
    return;
  }

  T* __restrict__ inputData = input;
  uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);

  const size_t threadLinearId =
      blockLocal ? static_cast<size_t>(threadIdx.x)
                 : static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + threadIdx.x;

  const size_t bytesPerElement = sizeof(T);
  const size_t bytesPerPeer = elementCount * bytesPerElement;
  // M5: the per-peer destination slot stride. Default (0) packs slots
  // contiguously (stride == copy size), preserving the original layout exactly.
  // A non-zero stride lets a chunk (bytesPerPeer) land at its strided position
  // inside a full-size block -- the chunked-pipeline enabler.
  const size_t slotStride = dstSlotStrideBytes != 0 ? dstSlotStrideBytes : bytesPerPeer;

  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;

  // Each member warp pushes this PE's shard into slot ``groupPos`` of the
  // warpId-th group member's destination buffer (SDMA over XGMI / P2P).
  // A single copy engine already saturates one XGMI link (~108 GB/s/link
  // measured), and the G warps drive G distinct peer links in parallel, so
  // one queue per peer is already bandwidth-bound — splitting a peer's shard
  // across multiple SDMA queues (SdmaPutWarp) gives no speedup and is
  // marginally slower (verified NC=1/2/4 via test_intra_subgroup_sdma --bench).
  // Keep the proven single-queue put.
  if (warpId < groupSize && laneId == 0) {
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

  if (warpId < groupSize && laneId == 0) {
    int remotePe = peBase + warpId * peStride;
    shmem::ShmemQuietThread(remotePe, dstMemObj);
    // SENDER-SIDE completion fence: ShmemQuietThread drains the SDMA submission
    // queue, but the peer-visible ORDERING of the raw SDMA data writes vs the
    // subsequent flag AMO is not otherwise guaranteed at SYSTEM scope. Without
    // this the receiver can observe the flag (and race past its own
    // threadfence_system) before the pushed bytes are globally ordered -- the
    // copy-engine completion gap that only a host stream.synchronize masked
    // (every prior fix was RECEIVER-side; this is the missing SENDER fence).
    __threadfence_system();
    shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
        flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
        core::atomicType::AMO_SET, remotePe, 0);
  }
  __syncthreads();

  for (int senderPos = 0; senderPos < groupSize; ++senderPos) {
    if (senderPos == groupPos) {
      continue;
    }
    if (threadLinearId == 0) {
      int spinCount = 0;
      bool warned = false;
      // SYSTEM-scope acquire: the flag (and the data it guards) is written by a
      // REMOTE peer GPU (a different HSA agent) via SDMA. An AGENT-scope relaxed
      // load establishes no cross-agent happens-before, so observing the flag
      // does NOT guarantee the peer's data writes are coherently visible to this
      // agent's subsequent copy-OUT -> occasional stale bytes under FSDP rapid
      // reuse (loss nondeterminism). A system-scope acquire + system threadfence
      // makes the peer's prior data writes visible without a host sync.
      while (core::AtomicLoadSeqCstSystem(flags + flagBase + senderPos) < flagVal) {
        ++spinCount;
        if (!warned && spinCount > 10000000) {
          printf("PE %d: Slow wait for sub-group pos %d (still waiting)\n", myPe, senderPos);
          warned = true;
        }
      }
      __threadfence_system();
    }
    __syncthreads();
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
                                             size_t flagBase, int qId) {
  if (elementCount == 0 || groupSize <= 0) {
    return;
  }
  const size_t threadLinearId = static_cast<size_t>(threadIdx.x);  // block-local
  const size_t bytesPerElement = sizeof(T);
  const size_t bytesPerPeer = elementCount * bytesPerElement;
  const size_t slotStride = dstSlotStrideBytes != 0 ? dstSlotStrideBytes : bytesPerPeer;
  int warpId = threadLinearId / warpSize;
  const int laneId = threadIdx.x % warpSize;

  if (warpId < groupSize && laneId == 0) {
    int remotePe = peBase + warpId * peStride;
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
    // Push this rank's column on its OWN queue ``q`` (distinct per block).
    core::SdmaPutThread(srcPtr, dstPtr, bytesPerPeer, devicehandles, signals, expectedSignals, nq, q);
    // Drain ONLY queue ``q`` (single-queue quiet) so the copy has landed before
    // the flag fires, WITHOUT touching other blocks' queues.
    core::SdmaQueitThread(signals + q, expectedSignals + q, 1);
    // SENDER-SIDE completion fence (see collective body above): system-scope
    // order the pushed SDMA bytes BEFORE the flag AMO becomes peer-visible, so
    // the completion reader cannot see the flag ahead of the data. Closes the
    // copy-engine visibility gap that previously needed a host stream sync.
    __threadfence_system();
    // The flag AMO is a DIRECT peer-memory CAS (P2P), not an SDMA-queue signal,
    // so it never races the per-queue counter. Bump slot ``flagBase+groupPos``.
    shmem::ShmemAtomicSizeNonFetchThreadKernel<application::TransportType::SDMA>(
        flagsMemObj, (flagBase + static_cast<size_t>(groupPos)) * sizeof(uint64_t), &flagVal, 8,
        core::atomicType::AMO_SET, remotePe, 0);
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

  for (int senderPos = 0; senderPos < groupSize; ++senderPos) {
    if (senderPos == groupPos) {
      continue;
    }
    if (threadLinearId == 0) {
      // SYSTEM-scope acquire (see subgroup gather): remote-agent SDMA producer.
      while (core::AtomicLoadSeqCstSystem(flags + senderPos) < flagVal) {
      }
      __threadfence_system();
    }
    __syncthreads();
  }
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
      // SYSTEM-scope acquire (see subgroup gather): remote-agent SDMA producer.
      while (core::AtomicLoadSeqCstSystem(flags + 0) < flagVal) {
        ++spinCount;
        if (!warned && spinCount > 10000000) {
          printf("PE %d: Slow wait for broadcast root (still waiting)\n", myPe);
          warned = true;
        }
      }
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

  for (int sender = 0; sender < npes; ++sender) {
    if (sender == myPe) {
      continue;
    }

    if (threadLinearId == 0) {
      int spinCount = 0;
      bool warned = false;
      uint64_t* __restrict__ flags = reinterpret_cast<uint64_t*>(flagsMemObj->localPtr);
      // SYSTEM-scope acquire (see subgroup gather): remote-agent SDMA writes the
      // flag+data; AGENT-scope relaxed gives no cross-agent visibility ordering.
      while (core::AtomicLoadSeqCstSystem(flags + sender) < flagVal) {
        ++spinCount;
        if (!warned && spinCount > 10000000) {
          printf("PE %d: Slow wait for param-contiguous data from peer %d (still waiting)\n", myPe,
                 sender);
          warned = true;
        }
      }
      __threadfence_system();
    }
    __syncthreads();
  }
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
