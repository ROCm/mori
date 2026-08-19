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

// ===========================================================================
// all_reduce_kernels.hpp
//
// Fused "push" all-reduce built on top of the reduce-scatter push kernel. The
// kernel runs the reduce-scatter phases (fused SDMA scatter into staging +
// receiver-side completion wait + grid-strided vectorized reduce) to produce
// THIS PE's reduced shard, sliced into S = 1<<logS slice-groups. Each slice-
// group then broadcasts ITS slice to every peer independently the moment that
// slice is reduced -- a pipelined per-slice broadcast that overlaps with the
// reduce tail of the other slices (no device-wide barrier). After the
// collective every PE holds the full reduced vector.
//
// This header reuses the reduce-scatter building blocks (ReduceVecGroup, the Op
// functors, ReduceComputeType) and the shared StartSdmaScatter. Phase 1-3 (the
// scatter + completion wait + acquire + grid-strided reduce) are the shared
// ReduceScatterPushBody, so all-reduce and reduce-scatter run the identical
// front; only the post-reduce tail (broadcast vs. group reset) differs.
// ===========================================================================
#pragma once

#include <array>
#include <cstdint>

#include "mori/collective/XLA/reduce_scatter_kernels.hpp"

using namespace mori::core;
using namespace mori::shmem;
using namespace mori::application;
using namespace mori::collective;

// ---------------------------------------------------------------------------
// Fused all-reduce kernel ("push").
//
//   input         : raw symmetric-heap pointer, N = npes*chunkElems elements
//   staging       : raw symmetric-heap pointer, (npes-1) slots of chunkElems
//   output        : raw symmetric-heap pointer, N = npes*chunkElems elements
//                   (the FULL reduced vector; on exit every slot p holds the
//                    reduction over all PEs of shard p)
//   groupCounters : plain device buffer (>= S uint32): per-slice-group arrival
//                   counters. Zeroed once by the host; each group's last block
//                   self-resets its slot. Used to detect the last block of each
//                   slice-group so exactly one block broadcasts that slice.
//
// Phase 1-3 are the reduce-scatter push algorithm (see ReduceScatterPushKernel):
// block 0 SDMA-scatters each peer's source slice into that peer's staging slot,
// every block waits on its per-slice completion counter, then the grid reduces
// the npes contributions of MY shard into output[myPe] (= myShard). The reduce
// grid is split into S = 1<<logS slice-groups, so each group pipelines on its
// own slice.
//
// Phase 4 (pipelined per-slice broadcast): as soon as slice-group g finishes
// reducing its slice, every block in the group does a system fence (publishing
// its reduce stores) and bumps groupCounters[g]. The block that reads the full
// group count G is the group's last block -- it knows slice g is complete -- and
// broadcasts JUST slice g into every peer's output[myPe] slice via a multi-
// producer-safe SDMA scatter (CAS reserve, since several groups' last blocks may
// share a per-peer queue). Each copy trails an ADD32(1) into the receiver's
// DEDICATED per-slice broadcast counter signalPtrs[kBcastSlot+g] -- distinct from
// the reduce-scatter slice counters [0..S-1] AND from other slices' broadcast
// counters, so no ADD can be miscounted by any other wait. The last block waits
// until all npes-1 peers' copies of slice g have landed, acquires, and resets its
// slice + broadcast counters for the next launch. Groups run independently, so
// each broadcast overlaps with the reduce tail of the other slices (no barrier).
//
// Because several groups' last blocks may target the same per-peer SDMA queue,
// all issuing blocks must be co-resident (host caps the grid to the SM count) so
// the CAS-reserved ring slots are committed in order without deadlock.
//
// input/staging/output MUST live in the symmetric static heap (ShmemMalloc).
// Limited to npes <= 8 (SDMA fast path).
// ---------------------------------------------------------------------------

// Base signalPtrs slot for the per-slice broadcast completion counters. Placed
// just past the reduce-scatter slice counters [0..S-1] (S <= kRSPushMaxSlices), so
// slice g's broadcast counter is signalPtrs[kBcastSlot+g] and never shares a
// counter with a reduce-scatter Phase-2 wait or another slice's broadcast. The
// signal buffer reserves room for these slots (see symmetric_memory.cpp).
static constexpr int kBcastSlot = kRSPushMaxSlices;

template <int NumVecs, class ReduceOp, class T = typename ReduceOp::Type>
__global__ void __launch_bounds__(256, 1)
AllReducePushKernel(int myPe, int npes, int logS, const T* __restrict__ input,
                    T* __restrict__ staging, T* __restrict__ output,
                    uint32_t* __restrict__ groupCounters, size_t chunkElems) {
  // My reduced shard lives in output slot myPe.
  T* __restrict__ myShard = output + static_cast<size_t>(myPe) * chunkElems;

  // === Phase 1-3: shared reduce-scatter push front (scatter + completion wait +
  // acquire + grid-strided reduce) into MY shard slot within the full output. ===
  ReduceScatterPushBody<NumVecs, ReduceOp, T>(myPe, npes, logS, input, staging, myShard,
                                              chunkElems);

  // === Phase 4: pipelined per-slice broadcast =================================
  // Recompute this block's slice-group g and its slice extent [sOfs, sOfs+sCnt)
  // (identical to the slicing in ReduceScatterPushBody).
  const uint32_t G = gridDim.x >> logS, g = blockIdx.x / G;
  constexpr int vecSize = VecBytes / sizeof(T);
  const size_t sliceLen = ((chunkElems >> logS) / vecSize) * vecSize;
  const size_t sOfs = g * sliceLen;
  const size_t sCnt = (g == (1u << logS) - 1) ? (chunkElems - sOfs) : sliceLen;

  // Publish this block's reduce stores to slice g before signaling group arrival,
  // so when group g's counter hits G all its stores are globally visible to the
  // last block's DMA read of the slice.
  __threadfence_system();
  __shared__ bool isGroupLast;
  if (threadIdx.x == 0) {
    isGroupLast = (atomicAdd(&groupCounters[g], 1u) + 1 == G);
  }
  __syncthreads();

  if (isGroupLast) {
    auto* heapObj = GetGlobalGpuStatesPtr()->heapObj;
    if (threadIdx.x == 0) {
      // Every block in group g has passed its Phase-2 read of signalPtrs[g] (they
      // all reached here), so slice g's RS counter is dead -- reset it now, along
      // with the group counter, overlapping the store latency with the broadcast.
      heapObj->signalPtrs[g] = 0;   // reset slice g's reduce-scatter counter
      groupCounters[g] = 0;         // reset group g's local counter
    }
    // Destination slot is myPe on every peer, so the byte offset is constant; the
    // source is my reduced slice (same for all peers). Self already holds it.
    const size_t heapBase = GetGlobalGpuStatesPtr()->heapBaseAddr;
    const size_t dstOff = reinterpret_cast<uintptr_t>(output) - heapBase +
                          (myPe * chunkElems + sOfs) * sizeof(T);
    // Warp 0 broadcasts slice g to all peers, trailing an ADD32(1) into each peer's
    // DEDICATED per-slice broadcast counter signalPtrs[kBcastSlot+g]. Groups spread
    // across queues (qId = g % numSdmaQ) to reduce contention; the CAS reserve keeps
    // it correct even when two groups share a queue.
    //const int numSdmaQ = static_cast<int>(heapObj->sdmaNumQueue);
    if (threadIdx.x < warpSize) {
      SdmaBroadcastSliceWarp(myShard + sOfs, dstOff, sCnt * sizeof(T), myPe, npes,
                                kBcastSlot + static_cast<int>(g), /*qId=*/0);
    }
    if (threadIdx.x == 0) {
      // Wait for every peer's copy of slice g to land in my output[myPe] slice.
      const uint32_t want = static_cast<uint32_t>(npes - 1);  // no self-copy
      // 32-bit ADD into the low dword -> poll 32 bits of this slice's counter.
      auto* addr =
          Tglobal(reinterpret_cast<uint32_t*>(&heapObj->signalPtrs[kBcastSlot + g]));
      while (__hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < want) {
      }
      heapObj->signalPtrs[kBcastSlot + g] = 0;  // reset this slice's bcast counter
    }
    __syncthreads();
    // One system-scope full fence does both jobs: ACQUIRE so peers' SDMA writes to
    // output (over XGMI) become visible, and RELEASE so this PE's signalPtrs resets
    // are published for the next launch's peer SDMA. The reset stores (thread 0,
    // above) are ordered before this fence by the preceding __syncthreads().
    __threadfence_system();
  }
}
