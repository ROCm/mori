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
//   groupCounters : plain device buffer (kGroupCounterCount uint32), zeroed once
//                   by the host. [kBcastFlagIdx + b]: broadcast b's release
//                   counter (every block counts in once per launch, never
//                   reset). [kBcastTargetIdx]: this launch's b=1 release target,
//                   which elects the tail owner.
//   barrierCtr    : symmetric-heap uint64[npes] gen-barrier flags shared by every
//                   collective (GenBarrier in block 0, before Phase 1)
//
// Phase 1-3 are the reduce-scatter push algorithm: block 0 SDMA-scatters each
// peer's source slice into that peer's staging slot, then every block loops the S
// slices itself and calls the shared per-slice reduce
// (detail::WaitAndReduceSlice), which waits on that slice's completion counter and
// reduces it with the whole grid into output[myPe] (= myShard).
//
// Phase 4 (two-copy broadcast) is the rest of that same loop iteration. Both
// broadcast copies (b=0: slices [0, S-1), b=1: the last slice) into every peer's
// output[myPe] are pre-posted by block 0 in Phase 1, right behind the scatter,
// each preceded by a POLL_REGMEM (padded to 64B with NOPs) on a local counter
// groupCounters[kBcastFlagIdx+b]. Once a block has reduced slice S-2 (resp.
// S-1) it does a system fence (publishing its reduce stores) and adds 1 to
// counter b=0 (resp. b=1); the add that brings the counter to its target
// releases the copy -- no election, packet build or doorbell on the critical
// path. Slices 0..S-3 need no count. Each copy trails an
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
// The release counters live in groupCounters, not signalBuf: they are local
// atomics, and sharing a uint64 with a counter that peers' SDMA ADD64s rewrite
// in full could lose an update. Only this PE's own SDMA engine polls them.
// They are never reset: the queue polls for == (value Phase 1 read + gridDim.x),
// exact modulo 2^32. A reset could not be ordered after the engine's last poll
// sample, and an equality poll that misses its value hangs. The counter cannot
// overshoot its target while polled: the next launch's adds need peers' next
// scatters, which follow the entry barrier, which follows this launch's b=1
// (and, in queue order, b=0) landing at every peer.
// kBcastSlot / kBcastFlagIdx / kBcastTargetIdx are defined in
// collectives_common.hpp.

template <int NumVecs, class ReduceOp, class T = typename ReduceOp::Type>
__global__ void __launch_bounds__(256, 1)
AllReducePushKernel(int myPe, int npes, int logS, const T* __restrict__ input,
                    T* __restrict__ output, uint32_t* __restrict__ groupCounters,
                    size_t strideElems, size_t totalElems, cco::ccoDevComm devComm,
                    T* __restrict__ staging, cco::ccoWindow_t heapWin,
                    uint64_t* __restrict__ barrierCtr) {

  // 16B-aligned uniform shard STRIDE; the last active PE's shard is shorter so
  // the shards tile [0, totalElems) exactly (see allReduceImpl). shardLen(p) is
  // peer p's length; my shard sits at the uniform offset myPe*strideElems.
  auto shardLen = [=](int p) -> size_t {
    const size_t ofs = p * strideElems;
    return min(totalElems - ofs, strideElems);
  };
  uint64_t* __restrict__ signalBuf = devComm.sdma.signalBuf;
  auto bcastFlag = [=](uint32_t b) { return &groupCounters[kBcastFlagIdx + b]; };

  constexpr int vecSize = VecBytes / sizeof(T);
  const uint32_t S = 1u << logS;

  // Phase 1: scatter the input to the staging buffer
  if (blockIdx.x == 0) {
    __shared__ uint32_t flagEpoch[2];
    __shared__ uint64_t scatterGen;
    if (threadIdx.x == 0) {
      // Read the counters BEFORE arriving: no block of this launch can count in
      // until peers scatter to me, which needs my arrival. The previous launch's
      // counts are visible across the kernel boundary.
      for (uint32_t b = 0; b < 2; b++) {
        flagEpoch[b] = AtomicLoad<ESystemScope>(bcastFlag(b)) + gridDim.x;
      }
      // Blocks of this launch read the b=1 target only after their last slice
      // wait, which is causally behind the barrier below; the release fence
      // orders this store before my arrival.
      AtomicStore<EAgentScope>(&groupCounters[kBcastTargetIdx], flagEpoch[1]);
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    }
    // Publish entry to every PE, but do not wait for a global rendezvous:
    // staging/signalBuf are shared with the other push collectives, so each
    // destination queue will POLL_REGMEM its own peer's generation before its
    // first scatter copy. Lane 0 enters after its stores above, in program order.
    if (threadIdx.x < warpSize) {
      // Every lane reads my slot before lane myPe overwrites it (same wave).
      const uint64_t gen = AtomicLoad<EAgentScope>(&barrierCtr[myPe]) & ~1ULL;
      const uint64_t next = gen + 2;
      if (threadIdx.x == 0) scatterGen = next;
      PublishGen(barrierCtr, next, myPe, npes, heapWin->stride4G);
    }
    __syncthreads();
    // reduce-scatter: per-peer source slice (stride=chunkElems), dst=staging,
    // no self-copy (self is folded in by the Phase-3 reduce reading local input).
    // Staging is packed densely (no self hole): slot = myPe<peer?myPe:myPe-1, a
    // bijection over the npes-1 non-self peers. 
    // Per-peer copy length = destination shard's length (shardLen(peer)); self
    // is skipped. The dst slot stride is the DESTINATION peer's length (its
    // staging slots are that size). 
    StartSdmaScatter<sizeof(T)>(
        devComm.sdma, npes, logS,
        [=](int peer) { return shardLen(peer); },
        [=](int peer) { return peer != myPe; },
        [=](int peer) -> const uint8_t* {
          return reinterpret_cast<const uint8_t*>(input + peer * strideElems);
        },
        [=](int peer) -> uint8_t* {
          const int slot = (myPe < peer ? myPe : myPe - 1);   // my slot in peer's staging
          int32_t diff = (peer - myPe)*static_cast<int32_t>(heapWin->stride4G);
          return reinterpret_cast<uint8_t*>(staging + slot * shardLen(peer)) +
             (static_cast<uint64_t>(diff)<<32);
        },
        barrierCtr, static_cast<uint32_t>(scatterGen));

    // Pre-post both broadcasts behind the scatter, each parked on its counter
    // until every block has counted in at Phase 4. b=0: slices [0, S-1); b=1: the last slice (the whole
    // shard at S==1, where b=0 does not exist).
    // Both broadcasts go out per peer as one reservation and one doorbell.
    const bool hasB0 = S > 1;
    const uint32_t epoch0 = flagEpoch[0], epoch1 = flagEpoch[1];
    SdmaPostPerPeer(devComm.sdma, npes, myPe, hasB0 ? 4 : 2,
        [=](int peer, int sub, uint32_t* q, uint64_t base) {
          HSAuint64* sig = devComm.sdma.peerSignalPtrs[peer] + kBcastSlot;
          
          // My reduced shard lives in output slot myPe (at the uniform stride offset).
          T* __restrict__ myShard = output + myPe * strideElems;
          const size_t myElems = shardLen(myPe),
                       sliceLen = ((myElems >> logS) / vecSize) * vecSize,
                       lastOfs = (S - 1) * sliceLen; 
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
                             (myElems - lastOfs) * sizeof(T),
                              sig, SliceSignalInc(1), bcastFlag(1), epoch1);
        });
  }

  // Whether this block's count released b=1, so it owes the slice-counter reset
  // and the completion wait for BOTH broadcasts. Only meaningful in thread 0.
  bool ownsTail = false;

  const uint32_t lstride = FORCE_SGPR(gridDim.x * blockDim.x);  // loop-invariant
  uint64_t seen = 0;

  for (uint32_t s = 0; s < S; s++) {
    // My reduced shard lives in output slot myPe (at the uniform stride offset).
    T* __restrict__ myShard = output + myPe * strideElems;
    const size_t myElems = shardLen(myPe),
                 sliceLen = ((myElems >> logS) / vecSize) * vecSize;
    // Slice s covers [sOfs, sOfs+sCnt) of my shard -- Phase 3 reduces that range.
    const size_t sOfs = FORCE_SGPR(s * sliceLen);
    const size_t sCnt = FORCE_SGPR(s == S - 1 ? myElems - sOfs : sliceLen);

    // === Phase 2-3: shared per-slice front (completion wait + acquire +
    // grid-strided reduce) into MY shard slot within the full output. ===
    // input base uses the uniform stride; staging slots use my own length.
    detail::WaitAndReduceSlice<NumVecs, ReduceOp>(signalBuf, seen, s, myPe, npes, input,
                                                 staging, myShard, sOfs, sCnt, strideElems,
                                                 myElems, lstride);

    // === Phase 4: count in for slices S-2 (b=0) and S-1 (b=1) ==================
    // b=0 covers slices [0, S-1): every block reduces slices in order, so its
    // count at S-2 vouches for all of them. b=1 covers the last slice (the whole
    // shard at S==1, where b=0 does not exist). Slices 0..S-3 need no count.
    if (s + 2 >= S) {
      const uint32_t b = (s == S - 1) ? 1u : 0u;
      // Every wave publishes its reduce stores before thread 0 counts the block
      // in, so once the counter reaches its target all blocks' stores are
      // visible to the SDMA read of the broadcast.
      __threadfence_system();
      __syncthreads();
      // The SDMA engine polls this counter for == target directly, so the last
      // count releases the pre-posted copy with no further step. System scope:
      // the SDMA engine must observe the RMW.
      if (threadIdx.x == 0) {
        const uint32_t old = __hip_atomic_fetch_add(cco::impl::global(bcastFlag(b)), 1u,
                                                    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        if (b == 1) {
          ownsTail = (old + 1 == AtomicLoad<EAgentScope>(&groupCounters[kBcastTargetIdx]));
        }
      }
    }
  } // for s

  // === Tail: reset the slice counter, collect every peer's broadcasts =========
  // The block whose count completed b=1 is the last one out of every slice wait,
  // so no add to the packed slice counter is still unseen and it may clear it.
  // That need not precede the b=1 release: peers only ADD into it again from
  // their next launch's scatter, behind the entry GenBarrier, which needs my next
  // launch to have started.
  //
  // Both broadcasts share one packed counter, so it can only be cleared once
  // both bytes are complete; the tail owner waits for both (b=0 does not exist
  // at S==1). Each peer's queue carries b=0 ahead of b=1, so byte 0 is normally
  // complete by the time byte 1 is.
  //
  // No fence closes the kernel. The resets are system-scope stores, so they need
  // no writeback, and nothing beyond the kernel boundary races them: these slots
  // are only touched again by a peer's next scatter or broadcast, which cannot
  // run before my NEXT launch's entry barrier. Nor is an acquire needed -- no
  // thread here reads the slices peers DMA'd into output, and whoever consumes
  // them acquires at its own kernel boundary.
  if (threadIdx.x == 0 && ownsTail) {
    AtomicStore<ESystemScope>(&signalBuf[kSliceSignalSlot], 0UL);
    const uint32_t want = static_cast<uint32_t>(npes - 1);  // no self-copy
    auto* addr = &signalBuf[kBcastSlot];
    auto done = [=](uint64_t v) {
      return SliceSignalCount(v, 1) >= want && (S == 1 || SliceSignalCount(v, 0) >= want);
    };
    while (!done(AtomicLoad<EAgentScope>(addr))) {
      __builtin_amdgcn_s_sleep(1);
    }
    // System scope again: peers ADD into this slot from their Phase 4.
    AtomicStore<ESystemScope>(&signalBuf[kBcastSlot], 0UL);
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
//   output_q[ myPe*strideElems + j ] = REDUCE_p( input_p[ myPe*strideElems + j ] )
//                                                                for every q
// Fusing is only legal because the reduce and the redistribution visit the same
// indices in the same thread. PullReduceShard is grid-strided (thread t owns
// {gtid + n*gstride}), so the value a thread is about to publish is one it just
// computed -- there is no cross-block dependency, and therefore no mid-kernel
// barrier of any kind.
//
// Synchronization is the same two GenBarrier rounds as ReduceScatterPullKernel
// (entry posts gen+1, exit gen+2, gen = arrived[myPe] & ~1), so back-to-back
// launches need no host barrier:
//   * entry: no block may read a peer's input or write a peer's output before
//     that peer has entered this launch (its producers of input and consumers
//     of output have retired). Block 0 posts; every block waits on the local
//     slots itself, so blocks never wait on each other.
//   * exit: the last block out waits until every peer has finished reading my
//     input and writing my output, so "kernel done" means "output complete and
//     input reusable". Every other block returns the moment it has counted in.
// ===========================================================================
// The last active PE's myElems is rounded up to a whole ComputeT pack by the
// facade; the straddling pad pack lands in the allocation's 4B padding and
// never crosses into another shard (it stays inside this PE's 16B stride slot).
// Tiny messages that would leave some PE with an empty shard are fine here: an
// idle PE gets myElems == 0 and simply runs the two GenBarrier rounds. (The push
// path cannot represent idle PEs, which is why the facade routes those to pull.)
//
//   input         : raw symmetric-heap pointer (npes shards of strideElems)
//   output        : raw symmetric-heap pointer; on exit slot p holds the
//                   reduction over all PEs of shard p
//   strideElems   : uniform shard stride (ComputeT units), base = myPe*strideElems
//   myElems       : THIS PE's shard length (ComputeT units); 0 == idle PE
//   groupCounters : plain device buffer, zeroed once by the host. [0] elects the
//                   last block; atomicInc wraps it back to 0, so no reset.
//   arrived       : symmetric-heap uint64[npes] gen-barrier flags shared by every
//                   collective (see GenBarrier).
template <int NumVecs, int NPES, class ReduceOp, class T = typename ReduceOp::Type>
__global__ void __launch_bounds__(256, 1)
AllReducePullKernel(int myPe, cco::ccoWindow_t heapWin,
                       const T* __restrict__ input, T* __restrict__ output,
                       size_t strideElems, size_t myElems,
                       uint32_t* __restrict__ groupCounters,
                       uint64_t* __restrict__ arrived) {
  const uint32_t stride4G = heapWin->stride4G;
  // My reduced shard sits at the uniform offset myPe*strideElems in every peer.
  const size_t baseElems = static_cast<size_t>(myPe) * strideElems;
  T* __restrict__ myShard = output + baseElems;

  // In block 0's first wave every lane loads this before lane myPe posts.
  const uint64_t gen = AtomicLoad<EAgentScope>(&arrived[myPe]) & ~1ULL;
  if (threadIdx.x < warpSize) GenBarrier<true>(arrived, gen + 1, myPe, NPES, stride4G);
  __syncthreads();

  // Each finished position goes to my own slot and to all NPES-1 peers' copies
  // of that same slot (peer p's copy is mine + (p-myPe)*stride4G<<32).
  auto store = [myShard, myPe, stride4G](size_t elemIdx, auto v) {
    constexpr int B = sizeof(v);
    T* __restrict__ mine = myShard + elemIdx;
    StreamStore<EAgentScope, B>(mine, v);
#pragma unroll
    for (int k = 1; k < NPES; k++) {
      const int32_t delta = (k < NPES - myPe) ? k : k - NPES;
      const int32_t diff = delta * static_cast<int32_t>(stride4G);
      StreamStore<ESystemScope, B>(
          reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(mine) +
                               (static_cast<uint64_t>(diff) << 32)),
          v);
    }
  };
  detail::PullReduceShard<NumVecs, NPES, ReduceOp>(myPe, stride4G, input, store,
                                                   baseElems, myElems);

  // Exit barrier: the remote stores ARE the payload, so every wave publishes
  // before its block counts in (the barrier release fence only covers the last).
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");   // system-scope release only
  __syncthreads();
  if (threadIdx.x < warpSize) {
    uint32_t last = 0;
    if (threadIdx.x == 0) {
      last = atomicInc(&groupCounters[kArrivalIdx], gridDim.x - 1) == gridDim.x - 1;
    }
    if (BroadcastWarp(last, 0)) GenBarrier(arrived, gen + 2, myPe, NPES, stride4G);
  }
}
} // namespace collective
} // namespace mori
