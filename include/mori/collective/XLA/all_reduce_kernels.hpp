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
// THIS PE's reduced shard, sliced into S = 1<<logS slices. The shard is then
// broadcast to every peer in two copies: slices [0, S-1) as one copy once slice
// S-2 is reduced (overlapping the reduce of the last slice), then the last slice.
// After the collective every PE holds the full reduced vector.
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
//   groupCounters : plain device buffer (>= kBcastFlagIdx + 2 uint32), zeroed once
//                   by the host. [0, S): per-slice arrival counters; each slice's
//                   elected block self-resets its slot. Used to detect the last
//                   block to finish a slice, so exactly one block broadcasts that
//                   slice. [kBcastFlagIdx + b]: broadcast b's release flag.
//   barrierCtr    : symmetric-heap uint64 for PushEntryBarrier (block 0, before
//                   Phase 1)
//
// Phase 1-3 are the reduce-scatter push algorithm: block 0 SDMA-scatters each
// peer's source slice into that peer's staging slot, then every block loops the S
// slices itself and calls the shared per-slice reduce
// (detail::WaitAndReduceSlice), which waits on that slice's completion counter and
// reduces it with the whole grid into output[myPe] (= myShard).
//
// Phase 4 (two-copy broadcast) is the rest of that same loop iteration. As soon
// as a block finishes reducing slice s it does a system fence (publishing its
// reduce stores) and bumps groupCounters[s]. The block that reads the full count
// gridDim.x is the last one out of slice s and resets that slice's counters.
// Both broadcast copies (b=0: slices [0, S-1), b=1: the last slice) into every
// peer's output[myPe] are pre-posted by block 0 in Phase 1, right behind the
// scatter, each preceded by a POLL_REGMEM (padded to 64B with NOPs) on a local
// flag groupCounters[kBcastFlagIdx+b]. Phase 4 then only has to store the flag:
// the elected block of slice S-2 releases b=0 and that of slice S-1 releases b=1
// -- no packet build or doorbell on the critical path. Each copy trails an
// ADD64(1 << 8b) into byte b of the receiver's DEDICATED packed broadcast counter
// signalPtrs[kBcastSlot] -- distinct from the packed reduce-scatter slice
// counter, so no ADD can be miscounted by any other wait.
//
// Why not one broadcast per slice: all packets to a peer share one SDMA queue
// (one engine per peer pair), so no broadcast can start before the S scatter
// packets to that peer drain, by which time slices [0, S-1) are normally reduced.
// Per-slice copies only added packets; the scatter and broadcast also share the
// same link direction, so they could not overlap usefully anyway.
//
// The release flags live in groupCounters, not signalBuf: they are plain local
// stores, and sharing a uint64 with a counter that peers' SDMA ADD64s rewrite in
// full could lose a flag update. Only this PE's own SDMA engine polls them.
// The flags are epochs, never reset: the queue polls for the value Phase 1 read
// + 1. kBcastSlot / kBcastFlagIdx are defined in collectives_common.hpp.

template <int NumVecs, class ReduceOp, class T = typename ReduceOp::Type>
__global__ void __launch_bounds__(256, 1)
AllReducePushKernel(int myPe, int npes, int logS, const T* __restrict__ input,
                    T* __restrict__ output, uint32_t* __restrict__ groupCounters,
                    size_t chunkElems, mori::cco::ccoDevComm devComm,
                    T* __restrict__ staging, mori::cco::ccoWindow_t heapWin,
                    uint64_t* __restrict__ barrierCtr) {

  // My reduced shard lives in output slot myPe.
  T* __restrict__ myShard = output + myPe * chunkElems;
  uint64_t* __restrict__ signalBuf = devComm.sdma.signalBuf;
  auto bcastFlag = [=](uint32_t b) { return &groupCounters[kBcastFlagIdx + b]; };

  constexpr int vecSize = VecBytes / sizeof(T);
  const uint32_t S = 1u << logS;
  // vecSize-aligned slice length; the last slice absorbs the remainder.
  const size_t sliceLen = ((chunkElems >> logS) / vecSize) * vecSize;

  // Phase 1: scatter the input to the staging buffer
  if (blockIdx.x == 0) {
    __shared__ uint32_t flagEpoch[2];
    if (threadIdx.x == 0) {
      // Read the flags BEFORE arriving: no block of this launch can bump one until
      // peers scatter to me, which needs my arrival. The previous launch's bump is
      // visible across the kernel boundary.
      for (uint32_t b = 0; b < 2; b++) {
        flagEpoch[b] = __hip_atomic_load(cco::impl::global(bcastFlag(b)), __ATOMIC_RELAXED,
                                         __HIP_MEMORY_SCOPE_SYSTEM) + 1;
      }
      // Back-to-back all-reduce is safe on its own (a peer finishes only after our
      // broadcasts, i.e. after we are done with staging and reset the slice counter),
      // but staging/signalBuf are shared with the other push collectives, whose
      // completion implies no such thing. Gate on every PE entering this launch.
      PushEntryBarrier(barrierCtr, myPe, npes, heapWin->stride4G);
    }
    __syncthreads();
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

    // Pre-post both broadcasts behind the scatter, each parked on its flag until
    // Phase 4 bumps it. b=0: slices [0, S-1); b=1: the last slice (the whole
    // shard at S==1, where b=0 does not exist).
    // Both broadcasts go out per peer as one reservation and one doorbell.
    const bool hasB0 = S > 1;
    const uint32_t epoch0 = flagEpoch[0], epoch1 = flagEpoch[1];
    SdmaPostPerPeer(devComm.sdma, npes, myPe, hasB0 ? 4 : 2,
        [=](int peer, int sub, uint32_t* q, uint64_t base) {
          HSAuint64* sig = devComm.sdma.peerSignalPtrs[peer] + kBcastSlot;
          
          const size_t lastOfs = (S - 1) * sliceLen; 
          int32_t diff = (peer - myPe) * static_cast<int32_t>(heapWin->stride4G);
          auto *peerDst = (uint8_t*)myShard + (static_cast<uint64_t>(diff) << 32);
          if (hasB0) {
            WriteGatedBroadcast(sub, q, base, myShard, peerDst,
                          lastOfs * sizeof(T), sig, SliceSignalInc(0),
                          bcastFlag(0), epoch0);
            base += 2 * kSDMACopyAtomicPktSize;
          }
          WriteGatedBroadcast(sub, q, base, myShard + lastOfs,
                              peerDst + lastOfs * sizeof(T),
                             (chunkElems - lastOfs) * sizeof(T),
                              sig, SliceSignalInc(1), bcastFlag(1), epoch1);
        });
  }

  // Whether this block released b=1 and so owes the completion wait for BOTH
  // broadcasts. Deferring that wait to the tail below keeps the elected block
  // reducing the remaining slices instead of stalling on XGMI mid-loop.
  bool ownsTail = false;
  __shared__ bool isSliceLast;

  const uint32_t lstride = FORCE_SGPR(gridDim.x * blockDim.x);  // loop-invariant
  uint64_t seen = 0;

  for (uint32_t s = 0; s < S; s++) {
    // Slice s covers [sOfs, sOfs+sCnt) of my shard -- Phase 3 reduces that range.
    const size_t sOfs = FORCE_SGPR(s * sliceLen);
    const size_t sCnt = FORCE_SGPR((s == S - 1) ? (chunkElems - sOfs) : sliceLen);

    // === Phase 2-3: shared per-slice front (completion wait + acquire +
    // grid-strided reduce) into MY shard slot within the full output. ===
    detail::WaitAndReduceSlice<NumVecs, ReduceOp>(signalBuf, seen, s, myPe, npes, input,
                                                 staging, myShard, sOfs, sCnt, chunkElems,
                                                 lstride);

    // === Phase 4: this slice is reduced -- elect its last block ================
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
        groupCounters[s] = 0;  // reset slice s's arrival counter
        // The packed slice counter is shared by every slice, so it can only be
        // cleared once no add is still in flight: the last block out of slice
        // S-1 is that point (every block has passed every slice's wait). It must
        // land before the b=1 release below, which lets peers finish and start
        // their next launch's scatter into this counter.
        if (s == S - 1) {
          StreamStore<ESystemScope, sizeof(uint64_t)>(&signalBuf[kSliceSignalSlot], 0);
        }
      }
      // Only slices S-2 and S-1 release a broadcast; the elected blocks of slices
      // 0..S-3 just reset their arrival counter.
      if (s + 2 >= S) {
        // b=0 (elected by S-2): slices [0, S-1). Every block reduces slices in
        // order and fences before counting in, so the last block into S-2 knows
        // all of [0, S-1) is reduced and published. b=1 (elected by S-1): the
        // last slice. At S==1 only b=1 exists and covers the whole shard.
        const uint32_t b = (s == S - 1) ? 1u : 0u;

        // The copies were pre-posted in Phase 1 behind a POLL on this flag; bump
        // it to the epoch they wait for. This block is the flag's only writer in
        // the launch, so load+store is enough. Release orders thread 0's resets
        // above before it: once a broadcast lands, its receiver can finish and
        // its NEXT launch's scatter starts ADDing to my packed slice counter.
        if (threadIdx.x == 0) {
          auto* flag = cco::impl::global(bcastFlag(b));
          const uint32_t e = __hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          __hip_atomic_store(flag, e + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        ownsTail = (b == 1);
      }
    }
  } // for s

  // === Tail: collect every peer's broadcasts into my output ===================
  // Both broadcasts share one packed counter, so it can only be cleared once
  // both bytes are complete; the b=1 owner waits for both (b=0 does not exist at
  // S==1). Each peer's queue carries b=0 ahead of b=1, so byte 0 is normally
  // complete by the time byte 1 is.
  //
  // No fence closes the kernel. The reset is a system-scope store, so it needs no
  // writeback, and nothing beyond the kernel boundary races it: this slot is only
  // touched again by a peer's Phase 4, which cannot run before my NEXT launch's
  // scatter. Nor is an acquire needed -- no thread here reads the slices peers
  // DMA'd into output, and whoever consumes them acquires at its own kernel
  // boundary.
  if (threadIdx.x == 0 && ownsTail) {
    const uint32_t want = static_cast<uint32_t>(npes - 1);  // no self-copy
    auto* addr = cco::impl::global(&signalBuf[kBcastSlot]);
    auto done = [=](uint64_t v) {
      return SliceSignalCount(v, 1) >= want && (S == 1 || SliceSignalCount(v, 0) >= want);
    };
    while (!done(__hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))) {
      __builtin_amdgcn_s_sleep(1);
    }
    // System scope again: peers ADD into this slot from their Phase 4.
    StreamStore<ESystemScope, sizeof(uint64_t)>(&signalBuf[kBcastSlot], 0);
  }
}

// ===========================================================================
// Fused all-reduce ("pull"). No SDMA, no staging: every byte moves as a plain
// vector load or store off the LSA flat VA.
//
// The reduce and the redistribution are a single pass. Each PE reduces shard
// myPe by reading it out of every peer's input, and the finished vector goes
// from the accumulator registers straight into my own output[myPe] AND, over
// XGMI, into every peer's output[myPe]:
//
//   output_q[ myPe*chunkElems + j ] = REDUCE_p( input_p[ myPe*chunkElems + j ] )
//                                                                for every q
// Fusing is only legal because the reduce and the redistribution visit the same
// indices in the same thread. PullReduceShard is grid-strided (thread t owns
// {gtid + n*gstride}), so the value a thread is about to publish is one it just
// computed -- there is no cross-block dependency, and therefore no mid-kernel
// barrier of any kind.
//
// That leaves one synchronization, at the very end: I must not return until
// every peer has finished writing into MY output. `syncFlags` is a symmetric
// array of one uint64 per PE, slot p owned by producer p, and the whole
// handshake runs in the single block that closes out the grid -- kernel
// completion already implies every block retired, so one waiter is enough to
// make "kernel done" mean "output complete". Every other block returns the
// moment it has counted in, freeing its CU.
// ===========================================================================

// Stored into a peer's flag slot once I have finished writing into that peer's
// output. Any non-zero value works: the slots are zeroed again at the end of
// every launch.
static constexpr uint64_t kARPullReady = 1;

//   input     : raw symmetric-heap pointer, N = npes*chunkElems elements
//   output    : raw symmetric-heap pointer, N = npes*chunkElems elements (on
//               exit every slot p holds the reduction over all PEs of shard p)
//   syncFlags : raw symmetric-heap pointer, >= npes uint64 slots, zero on entry
//   groupCounters : plain device buffer (>= 1 uint32), zeroed once by the host.
//               [0] elects the block that runs the closing handshake, and is
//               reset by that same block.
template <int NumVecs, int NPES, class ReduceOp, class T = typename ReduceOp::Type>
__global__ void __launch_bounds__(256, 1)
AllReducePullKernel(int myPe, const T* __restrict__ input, T* __restrict__ output,
                    uint64_t* __restrict__ syncFlags, uint32_t* __restrict__ groupCounters,
                    size_t chunkElems, mori::cco::ccoWindow_t heapWin) {
  const uint32_t stride4G = heapWin->stride4G;
  // My reduced shard lives in output slot myPe, exactly as in the push kernel.
  T* __restrict__ myShard = output + static_cast<size_t>(myPe) * chunkElems;

  // === Reduce + fan out, in one pass =========================================
  // Every finished position goes to my own slot and to all NPES-1 peers' copies
  // of that same slot. Peer p's copy of a local heap pointer sits at that
  // pointer + (p - myPe)*stride4G<<32, the same rank delta the loads use.
  //
  // Local store is agent scope: nobody reads my shard remotely any more (they
  // are handed it), so it may sit in L2 for whoever consumes the output next.
  // Remote stores are system scope, and are what carries the collective.
  auto store = [myShard, myPe, stride4G](size_t elemIdx, auto v) {
    constexpr int B = sizeof(v);
    T* __restrict__ mine = myShard + elemIdx;
    StreamStore<EAgentScope, B>(mine, v);
#pragma unroll
    for (int k = 1; k < NPES; k++) {
      // Peer (myPe + k) % NPES, written as a delta so the modulo stays out of
      // the address math. Loop-invariant, so it hoists out of the grid stride.
      const int32_t delta = (k < NPES - myPe) ? k : k - NPES;
      const int32_t diff = delta * static_cast<int32_t>(stride4G);
      StreamStore<ESystemScope, B>(
          reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(mine) +
                               (static_cast<uint64_t>(diff) << 32)),
          v);
    }
  };
  detail::PullReduceShard<NumVecs, NPES, ReduceOp>(myPe, stride4G, input, store, chunkElems);

  // === Closing handshake =====================================================
  // Release my remote stores, then count in. Everyone but the last block out is
  // finished and gives its CU back; the last block alone signals, waits and
  // clears, which is enough because the kernel cannot complete until it exits.
  __threadfence_system();
  __shared__ bool isGridLast;
  if (threadIdx.x == 0) {
    isGridLast = (atomicAdd(&groupCounters[0], 1u) + 1 == gridDim.x);
  }
  __syncthreads();
  if (!isGridLast) return;  // block-uniform: isGridLast is shared

  {
    const int peer = static_cast<int>(threadIdx.x);
    const bool isPeerLane = peer < NPES && peer != myPe;
    // Tell peer p I am done writing into it: one lane per peer, one 8-byte
    // remote store each.
    if (isPeerLane) {
      const int32_t diff = (peer - myPe) * static_cast<int32_t>(stride4G);
      auto* slot = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(syncFlags + myPe) +
                                               (static_cast<uint64_t>(diff) << 32));
      StreamStore<ESystemScope, sizeof(uint64_t)>(cco::impl::global(slot), kARPullReady);
    }
    // Push those tokens out before parking on the peers', or every PE could sit
    // waiting on a signal still held in its own write path.
    __threadfence_system();
    // Then wait for the same from everyone. Signalling strictly precedes
    // waiting on every PE, so this cannot deadlock however the grids interleave.
    // The poll is local (peers write my slots over XGMI); system scope because
    // the writer is another device.
    if (isPeerLane) {
      auto* addr = cco::impl::global(&syncFlags[peer]);
      while (__hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == 0) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
    // Acquire half only (not the full seq_cst __threadfence_system): we publish
    // nothing after this, we only need the peers' stores visible to my consumer.
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");

    // Clear for the next launch. The tokens are dead by now -- every peer that
    // wrote one is past its own wait -- so zeroing cannot drop an unseen signal.
    // System scope: peers write these slots from another device, so a plain
    // store would leave a dirty line that could evict on top of a later token.
    //
    // As with AllGatherPushKernel this clears at kernel END and so relies on the
    // caller barriering between launches (the benchmark does hipStreamSynchronize
    // + ccoBarrierAll per iteration); without one, a peer's NEXT launch token can
    // arrive before this store and be overwritten by it, no matter the scope.
    if (peer < NPES) {
      StreamStore<ESystemScope, sizeof(uint64_t)>(&syncFlags[peer], 0);
    }
  }
  // Re-arm the arrival counter. Safe here: isGridLast means every block of this
  // grid has already counted, so no arrival can be lost.
  if (threadIdx.x == 0) groupCounters[0] = 0;
}
} // namespace collective
} // namespace mori
