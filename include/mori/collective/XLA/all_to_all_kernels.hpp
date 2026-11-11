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
// all_to_all_kernels.hpp
//
// SDMA-based "push" all-to-all. Mirrors RCCL's per-rank send/recv loop: PE myPe
// sends srcPtrs[p] to peer p, and peer p stores it in its recv slot for sender
// myPe (dstPtrs[myPe]). Self (p==myPe) is included via an SDMA self-copy, like
// all-gather. Compute-free: SDMA writes every peer's recv slot directly and the
// kernel only waits for completion.
//
// Reuses the shared fused SDMA routine StartSdmaScatter (collectives_common.hpp)
// via call-site lambdas: active=true (self included), per-peer source is an
// arbitrary pointer srcPtrs[peer], and the destination offset is constant
// (dstPtrs[myPe] - heapBase; every receiver stores our chunk at its own
// dstPtrs[myPe]-equivalent symmetric-heap offset).
// ===========================================================================
#pragma once

#include <cstdint>
#include "mori/collective/XLA/collectives_common.hpp"
#include "mori/collective/XLA/all_gather_kernels.hpp" // for detail::PullCopyInterleaved
namespace mori {
namespace ccl {

// Single-block, compute-free all-to-all.
//   pairs      : npes endpoints, by value; pairs.p[p].source (local) is sent to
//                peer p, pairs.p[p].dest (symmetric heap) receives peer p's chunk.
//   chunkBytes : size in bytes of each per-peer chunk.
template <int Flags>
__global__ void AllToAllPushKernel(int myPe, int npes, const AddressPairs pairs,
                                   size_t chunkBytes, cco::ccoDevComm devComm,
                                   cco::ccoWindow_t heapWin,
                                   uint64_t* __restrict__ barrierCtr,
                                   uint64_t* __restrict__ arrived) {
  const uint32_t stride4G = heapWin->stride4G;
  // Every lane of the first wave reads my slot before lane myPe posts. A
  // kAsyncDone launch finds the start's pushGen - 1 there, so it gets the same.
  const uint64_t pushGen = LoadPushGen(barrierCtr, myPe);

  if constexpr (Flags & kAsyncStart) {
    // A peer finishing its previous launch only proves our chunk reached it; each
    // peer's queue polls for its arrival in this launch before writing its recv slot.
    PostArrival(barrierCtr, pushGen - 1, myPe, npes, stride4G);
    __syncthreads();

    // Push each send slot p to peer p's recv slot for sender myPe (self
    // included). My recv slot has the SAME byte offset within the heap window on
    // every peer (all ranks Allocate identically), so peer p's copy of my recv
    // slot is just my local recv pointer shifted by the flat-VA rank delta
    // (pe - myPe)*stride4G<<32 -- the same inline LSA addressing the
    // reduce-scatter/all-reduce push kernels use. winBase cancels out, so this
    // needs no ccoGetLocalPtr/ccoGetLsaPeerPtr round trip. The source is an
    // arbitrary per-peer local pointer (SDMA reads local VA directly). The lambda
    // captures a pointer into the kernel arguments, not a per-thread copy of them.
    const AddressPair* const pp = pairs.p;
    uint8_t* const localDst = reinterpret_cast<uint8_t*>(pp[myPe].dest);
    StartSdmaScatter(
        devComm.sdma, npes, /*logS=*/0, pushGen,
        [=](int peer) {
          auto [dstP, sigP] = PeerPtrs(localDst, arrived + myPe * kArrivedStride,
                                       peer, myPe, stride4G);
          return ScatterPeer{true, chunkBytes,
                             reinterpret_cast<const uint8_t*>(pp[peer].source),
                             dstP, sigP,
                             peer != myPe ? BarrierGate(barrierCtr, peer) : nullptr,
                             static_cast<uint32_t>(pushGen - 1)};
        });
  }
  if constexpr ((Flags & kAsyncDone) == 0) return;

  // === Wait until all npes senders (self-copy included) signalled, ===
  // and until my chunks landed at every peer, so the sources are reusable on return.
  if (threadIdx.x < warpSize) {
    WaitArrivedAndDelivered(arrived, /*slot=*/0, pushGen, myPe, npes, stride4G);
  }
  __syncthreads();
  // System-scope ACQUIRE: recv slots were written by peers' SDMA over XGMI.
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");

  // Recv buffers already fully written by SDMA. The block read pushGen at
  // entry, so it can advance the generation for the next launch now.
  if (threadIdx.x == 0) {
    AtomicStore<ESystemScope>(&barrierCtr[myPe], pushGen);
  }
}

// Single-block, compute-free ragged all-to-all (XLA ragged-all-to-all
// semantics). For peer p and update u (idx = p*U + u) PE myPe copies rows
// [inOffs[idx], inOffs[idx] + sendSizes[idx]) of `input` to rows starting at
// outOffs[idx] of peer p's `output`; outOffs index the RECEIVER's buffer. The
// metadata (int64 rows) stays in device memory and is read by the SDMA issue
// lanes. `output` must sit at the same symmetric-heap offset on every PE;
// `input` may be any local buffer. Completion and generations are exactly
// AllToAllPushKernel's: every peer gets one signal in slot 0.
template <int Flags>
__global__ void RaggedAllToAllPushKernel(int myPe, int npes, const void* input, void* output,
                                         const int64_t* __restrict__ inOffs,
                                         const int64_t* __restrict__ sendSizes,
                                         const int64_t* __restrict__ outOffs, int U,
                                         size_t rowBytes, cco::ccoDevComm devComm,
                                         cco::ccoWindow_t heapWin,
                                         uint64_t* __restrict__ barrierCtr,
                                         uint64_t* __restrict__ arrived) {
  const uint32_t stride4G = heapWin->stride4G;
  const uint64_t pushGen = LoadPushGen(barrierCtr, myPe);

  if constexpr (Flags & kAsyncStart) {
    PostArrival(barrierCtr, pushGen - 1, myPe, npes, stride4G);
    __syncthreads();

    auto* const localDst = static_cast<uint8_t*>(output);
    auto* const src = static_cast<const uint8_t*>(input);
    uint64_t* const myRow = arrived + myPe * kArrivedStride;
    StartSdmaRaggedScatter(
        devComm.sdma, npes, U, inOffs, sendSizes, outOffs, rowBytes, pushGen,
        [=](int peer) {
          auto [dstP, rowP] = PeerPtrs(localDst, myRow, peer, myPe, stride4G);
          return RaggedPeer{src, dstP, rowP, rowP + kRaggedScratch,
                            peer != myPe ? BarrierGate(barrierCtr, peer) : nullptr,
                            static_cast<uint32_t>(pushGen - 1)};
        });
  }
  if constexpr ((Flags & kAsyncDone) == 0) return;

  if (threadIdx.x < warpSize) {
    WaitArrivedAndDelivered(arrived, /*slot=*/0, pushGen, myPe, npes, stride4G);
  }
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  if (threadIdx.x == 0) {
    AtomicStore<ESystemScope>(&barrierCtr[myPe], pushGen);
  }
}

// Multi-block, SDMA-free all-to-all: every PE reads, from each peer p, the chunk
// p sends to it and stores it into its own recv slot for sender p.
//   pairs      : npes endpoints, by value; pairs.p[p].dest (local) receives
//                peer p's chunk. Unlike the push kernel, the SEND slots must also
//                be symmetric: peer p's chunk for me sits at the heap offset of my
//                own pairs.p[myPe].source, so it is read at PeerPtr(that, p).
//   chunkBytes : size in bytes of each per-peer chunk
//   NPES       : compile-time peer count (see PullCopyInterleaved).
//
// Peer-interleaved: the copy reads all NPES peers' send slots concurrently, 
// so every XGMI read link is busy at once instead of one
// peer at a time. Synchronization is the two GenBarrier rounds: entry 
// (peers' send slots are ready) and exit by the last block (peers are done 
// reading my send slots).
template <int NPES, int NumVecs>
__global__ void __launch_bounds__(256, 1)
AllToAllPullKernel(int myPe, int npes, cco::ccoWindow_t heapWin,
                    const AddressPairs pairs, size_t chunkBytes,
                    uint32_t* __restrict__ groupCtr, uint64_t* __restrict__ arrived) {
  const uint32_t stride4G = heapWin->stride4G;
  // Hoist the per-thread-indexed dests into LDS once per block, so the copy
  // loop's statically indexed dsts[j] needs no kernel-argument reloads.
  __shared__ uint8_t* dsts[NPES];
  if (threadIdx.x < NPES) {
    dsts[threadIdx.x] = static_cast<uint8_t*>(pairs.p[threadIdx.x].dest);
  }
  const auto* src = static_cast<const uint8_t*>(pairs.p[myPe].source);

  // In block 0's first wave every lane loads this before lane myPe posts.
  const uint64_t gen = AtomicLoad<EAgentScope>(&arrived[myPe]) & ~1ULL;
  GenBarrier<true>(arrived, gen + 1, myPe, npes, stride4G);
  __syncthreads();  // dsts visible to every thread before the copy

  detail::PullCopyInterleaved<NPES, NumVecs>(
      chunkBytes, [&](int j) { 
        return std::pair{PeerPtr(src, j, myPe, stride4G), // peer j's send slot for me
                          dsts[j]};  // my recv slot for sender j
  }); 

  // A block counts in only once all its waves are done reading peer send slots.
  __syncthreads();
  if (LastBlockOut(groupCtr)) GenBarrier(arrived, gen + 2, myPe, npes, stride4G);
}
} // namespace ccl
} // namespace mori
