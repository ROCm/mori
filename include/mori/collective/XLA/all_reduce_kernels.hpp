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
// THIS PE's reduced shard, sliced into S = 1<<logS slices. Each slice is
// broadcast to every peer the moment the grid finishes reducing it -- a
// pipelined per-slice broadcast that overlaps with the reduce of the later
// slices (no device-wide barrier). After the collective every PE holds the full
// reduced vector.
//
// This header reuses the reduce-scatter building blocks (ReduceVecGroup, the Op
// functors, ReduceComputeType) and the shared StartSdmaScatter. Phase 1-3 (the
// scatter + completion wait + acquire + grid-strided reduce) are exactly what
// reduce-scatter does: every block loops all S slices itself and calls the shared
// per-slice reduce, detail::WaitAndReduceSlice. This kernel appends Phase 4 to
// each iteration of that loop, so the two collectives share the per-slice reduce
// and neither constrains gridDim.x.
// ===========================================================================
#pragma once

#include <array>
#include <cstdint>

#include "mori/collective/XLA/reduce_scatter_kernels.hpp"

namespace mori {
namespace collective {
// ---------------------------------------------------------------------------
// Fused all-reduce kernel ("push").
//
//   input         : raw symmetric-heap pointer, N = npes*chunkElems elements
//   staging       : raw symmetric-heap pointer, (npes-1) slots of chunkElems
//   output        : raw symmetric-heap pointer, N = npes*chunkElems elements
//                   (the FULL reduced vector; on exit every slot p holds the
//                    reduction over all PEs of shard p)
//   groupCounters : plain device buffer (>= S uint32): per-slice arrival
//                   counters. Zeroed once by the host; each slice's elected block
//                   self-resets its slot. Used to detect the last block to finish
//                   a slice, so exactly one block broadcasts that slice.
//
// Phase 1-3 are the reduce-scatter push algorithm: block 0 SDMA-scatters each
// peer's source slice into that peer's staging slot, then every block loops the S
// slices itself and calls the shared per-slice reduce
// (detail::WaitAndReduceSlice), which waits on that slice's completion counter and
// reduces it with the whole grid into output[myPe] (= myShard).
//
// Phase 4 (pipelined per-slice broadcast) is the rest of that same loop iteration.
// As soon as a block finishes reducing slice s it does a system fence (publishing
// its reduce stores) and bumps groupCounters[s]. The block that reads the full count
// gridDim.x is the last one out of slice s -- it knows the slice is complete -- and
// ISSUES a broadcast of JUST slice s into every peer's output[myPe] slice via a
// multi-producer-safe SDMA scatter (CAS reserve, since several slices' elected
// blocks may share a per-peer queue). Each copy trails an ADD32(1) into the
// receiver's DEDICATED per-slice broadcast counter signalPtrs[kBcastSlot+s] --
// distinct from the reduce-scatter slice counters [0..S-1] AND from other slices'
// broadcast counters, so no ADD can be miscounted by any other wait.
static constexpr int kBcastSlot = kRSPushMaxSlices;

template <int NumVecs, class ReduceOp, class T = typename ReduceOp::Type>
__global__ void __launch_bounds__(256, 1)
AllReducePushKernel(int myPe, int npes, int logS, const T* __restrict__ input,
                    T* __restrict__ output, uint32_t* __restrict__ groupCounters,
                    size_t chunkElems, mori::cco::ccoDevComm devComm,
                    T* __restrict__ staging, mori::cco::ccoWindow_t heapWin) {

  // Phase 1: scatter the input to the staging buffer
  if (blockIdx.x == 0) {
    // reduce-scatter: per-peer source slice (stride=chunkElems), dst=staging,
    // no self-copy (self is folded in by the Phase-3 reduce reading local input).
    // Staging is packed densely (no self hole): slot = myPe<peer?myPe:myPe-1, a
    // bijection over the npes-1 non-self peers. 
    const size_t chunkBytes = chunkElems * sizeof(T);
    StartSdmaScatter<sizeof(T)>(
        devComm.sdma, npes, logS, chunkElems,
        [=](int peer) { return peer != myPe; },
        [=](int peer) -> const uint8_t* {
          return reinterpret_cast<const uint8_t*>(input + peer * chunkElems);
        },
        [=](int peer) -> uint8_t* {
          const int slot = (myPe < peer ? myPe : myPe - 1);   // my slot in peer's staging
          int32_t diff = (peer - myPe)*static_cast<int32_t>(heapWin->stride4G);
          return reinterpret_cast<uint8_t*>(staging + slot * chunkElems) + 
             (static_cast<uint64_t>(diff)<<32);
        });
  }

  // My reduced shard lives in output slot myPe.
  T* __restrict__ myShard = output + myPe * chunkElems;
  uint64_t* __restrict__ signalBuf = devComm.sdma.signalBuf;

  // Slices this block broadcast, and therefore owes a completion wait. Deferring
  // that wait to the tail below keeps the elected block reducing the remaining
  // slices instead of stalling on XGMI in the middle of the loop.
  uint32_t ownedMask = 0;
  __shared__ bool isSliceLast;

  constexpr int vecSize = VecBytes / sizeof(T);
  const uint32_t S = 1u << logS;
  // vecSize-aligned slice length; the last slice absorbs the remainder.
  const size_t sliceLen = ((chunkElems >> logS) / vecSize) * vecSize;
  const uint32_t lstride = FORCE_SGPR(gridDim.x * blockDim.x);  // loop-invariant

  for (uint32_t s = 0; s < S; s++) {
    // Slice s covers [sOfs, sOfs+sCnt) of my shard -- Phase 3 reduces that range
    // and Phase 4 broadcasts exactly the same range.
    const size_t sOfs = FORCE_SGPR(s * sliceLen);
    const size_t sCnt = FORCE_SGPR((s == S - 1) ? (chunkElems - sOfs) : sliceLen);

    // === Phase 2-3: shared per-slice front (completion wait + acquire +
    // grid-strided reduce) into MY shard slot within the full output. ===
    detail::WaitAndReduceSlice<NumVecs, ReduceOp>(signalBuf, s, myPe, npes, input, staging,
                                                 myShard, sOfs, sCnt, chunkElems, lstride);

    // === Phase 4: this slice is reduced -- elect one block to broadcast it =====
    // Publish this block's reduce stores to slice s before announcing arrival, so
    // once the counter hits gridDim.x every block's stores are globally visible to
    // the elected block's DMA read of the slice.
    __threadfence_system();
    if (threadIdx.x == 0) {
      isSliceLast = (atomicAdd(&groupCounters[s], 1u) + 1 == gridDim.x);
    }
    __syncthreads();

    if (isSliceLast) {  // block-uniform: isSliceLast is shared
      if (threadIdx.x == 0) {
        StreamStore<ESystemScope, sizeof(uint64_t)>(&signalBuf[s], 0);
        groupCounters[s] = 0;  // reset slice s's arrival counter
      }
      // That reset must have landed before any doorbell below rings: once a
      // broadcast lands, its receiver can finish this collective and its NEXT
      // launch's Phase-1 SDMA starts ADDing to my signalBuf[s]. __syncthreads()
      // carries an s_waitcnt vmcnt(0) that drains thread 0's store, and it orders
      // that store before the doorbells of ALL lanes, not just thread 0's.
      __syncthreads();

      // Destination slot is myPe on every peer, so the source (my reduced slice)
      // and the symmetric destination address are the same for all peers. Self
      // already holds it.
      // Warp 0 issues slice s to all peers, each copy trailing an ADD32(1) into
      // that peer's DEDICATED per-slice broadcast counter signalBuf[kBcastSlot+s].
      // The CAS reserve keeps it correct even when two slices' elected blocks
      // share a queue. No wait here -- see the tail.
      if (threadIdx.x < warpSize) {
        SdmaBroadcastSliceWarp(
            devComm.sdma, myShard + sOfs,
            [=](int peer) -> uint8_t* {
              int32_t diff = (peer - myPe) * static_cast<int32_t>(heapWin->stride4G);
              return reinterpret_cast<uint8_t*>(myShard + sOfs) +
                     (static_cast<uint64_t>(diff) << 32);
            },
            sCnt * sizeof(T), myPe, npes, kBcastSlot + static_cast<int>(s), /*qId=*/0);
      }
      ownedMask |= (1u << s);
    }
  }

  // === Tail: collect the broadcasts this block issued ==========================
  // No fence closes the kernel. The broadcast-counter resets are system-scope
  // stores, so they need no writeback, and nothing beyond the kernel boundary
  // races them: unlike the Phase-2 counters, this slot is only touched again by a
  // peer's Phase 4, which cannot run before my NEXT launch's scatter. Nor is an
  // acquire needed -- no thread here reads the slices peers DMA'd into output, and
  // whoever consumes them acquires at its own kernel boundary.
  if (threadIdx.x == 0 && ownedMask != 0) {
    const uint32_t want = static_cast<uint32_t>(npes - 1);  // no self-copy
    for (uint32_t s = 0; s < S; s++) {
      if ((ownedMask & (1u << s)) == 0) continue;
      // Wait for every peer's copy of slice s to land in my output[myPe] slice.
      // 32-bit ADD into the low dword -> poll 32 bits of this slice's counter.
      auto* addr = Iglobal(reinterpret_cast<uint32_t*>(&signalBuf[kBcastSlot + s]));
      while (__hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < want) {
      }
      // System scope again: peers ADD into this slot from their Phase 4.
      StreamStore<ESystemScope, sizeof(uint64_t)>(&signalBuf[kBcastSlot + s], 0);
    }
  }
}
} // namespace collective
} // namespace mori
