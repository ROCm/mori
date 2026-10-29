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
// all_gather_kernels.hpp
//
// SDMA-based "push" all-gather. All-gather is the mirror of reduce-scatter's
// scatter step with NO reduction: each PE pushes its single local shard into
// every peer's output[myPe] slot -- including itself (SDMA self-copy). Since
// there is no GPU compute to overlap, this uses a single block and logS=0
// (S=1, one slice, one completion flag), reusing the generalized fused SDMA
// scatter (StartSdmaScatter, byte-based) from collectives_common.hpp.
// ===========================================================================
#pragma once

#include <cstdint>
#include "mori/collective/XLA/collectives_common.hpp"

namespace mori {
namespace collective {

namespace detail {

// Peer-interleaved pull copy. Unlike PullCopySegments (peer-outer: the whole grid
// drains one peer before the next, so only one XGMI read link is busy at a time),
// this issues NumVecs b128 loads spread across all NPES peers at once -- V =
// NumVecs/NPES units per peer -- so NPES read links run concurrently and pull
// bandwidth scales past a single link. NPES is a compile-time constant so the
// peer bases live in SGPRs and the AA[] batch is statically indexed; requires
// NPES to divide NumVecs. srcOf(j)/dstOf(j) return peer j's chunk base (src = a
// peer pointer, dst = a local slot), both indexed by the same unit offset.
// Loads are system scope (sources are peers' buffers); stores use StoreScope.
template <int NPES, int NumVecs, StreamScope StoreScope = ESystemScope,
          class SrcDstFn>
__device__ __forceinline__ void PullCopyInterleaved(size_t bytes, SrcDstFn srcDstOf) {
  static_assert(NumVecs % NPES == 0, "NPES must divide NumVecs");
  constexpr int V = NumVecs / NPES;  // units in flight per peer
  using Vec = TVecType<VecBytes>;
  const size_t N = bytes / VecBytes;  // whole 16B units per peer chunk
  const size_t gstride = static_cast<size_t>(blockDim.x) * gridDim.x;
  const size_t gtid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  // Hoist the per-peer bases once (uniform across the block -> SGPRs).
  const uint8_t* __restrict__ src[NPES];
  uint8_t* __restrict__ dst[NPES];
#pragma unroll
  for (int j = 0; j < NPES; j++) {
    auto [s, d] = srcDstOf(j);
    src[j] = s;
    dst[j] = d;
  }

  // Main batch: each thread holds V units for every peer => NumVecs loads in
  // flight across NPES links before any store. Advance by V*gstride per step.
  const size_t step = V * gstride;
  size_t u0 = gtid;
  for (; u0 + (V - 1) * gstride < N; u0 += step) {
    Vec aa[NumVecs];
#pragma unroll
    for (int j = 0; j < NPES; j++) {
#pragma unroll
      for (int i = 0; i < V; i++) {
        aa[j * V + i] =
            StreamLoad<ESystemScope>(src[j] + (u0 + i * gstride) * VecBytes);
      }
    }
#pragma unroll
    for (int j = 0; j < NPES; j++) {
#pragma unroll
      for (int i = 0; i < V; i++) {
        StreamStore<StoreScope>(dst[j] + (u0 + i * gstride) * VecBytes,
                                                aa[j * V + i]);
      }
    }
  }
  // Remainder units (still interleaved across peers to keep links busy).
  for (size_t u = u0; u < N; u += gstride) {
#pragma unroll
    for (int j = 0; j < NPES; j++) {
      StreamStore<StoreScope>(
          dst[j] + u * VecBytes,
          StreamLoad<ESystemScope>(src[j] + u * VecBytes));
    }
  }
  // Byte tail for a final < 16B per peer (empty when bytes % 16 == 0).
  for (size_t i = N * VecBytes + gtid; i < bytes; i += gstride) {
#pragma unroll
    for (int j = 0; j < NPES; j++) {
      StreamStore<StoreScope, 1>(dst[j] + i, 
        StreamLoad<ESystemScope, 1>(src[j] + i));
    }
  }
}

} // namespace detail

// Single-block, compute-free all-gather.
//   input  : raw symmetric-heap pointer, my shard of chunkBytes bytes
//   output : raw symmetric-heap pointer, npes*chunkBytes bytes (all shards)
//
// Phase 1: SDMA scatters my shard into every peer's output[myPe] slot (self
// included), each copy trailed by an atomic swap storing the push generation into slot
// 0 of my row of the receiver's `arrived`. Phase 2: the first wave waits until
// every row (self included) reached the generation, then an acquire fence makes
// the peer SDMA writes to output visible. No Phase 3 -- output is already fully
// written by SDMA. Finally thread 0 advances the generation.
//
// blockDim.x must be >= npes for block 0's fast-path packet build; launch with
// npes <= 8 (SDMA fast path) and a comfortable blockDim (256).
__global__ void AllGatherPushKernel(int myPe, int npes, const void* __restrict__ input,
                                    void* __restrict__ output, size_t chunkBytes,
                                    cco::ccoDevComm devComm,
                                    cco::ccoWindow_t heapWin,
                                    uint64_t* __restrict__ barrierCtr,
                                    uint64_t* __restrict__ arrived) {
  const uint32_t stride4G = heapWin->stride4G;
  // Every lane of the first wave reads my slot before lane myPe posts.
  const uint64_t pushGen = LoadPushGen(barrierCtr, myPe);
  // A peer finishing its previous launch only proves our shard reached it; gate
  // our writes into its output until every PE entered this one.
  GenBarrier(barrierCtr, pushGen - 1, myPe, npes, stride4G);
  __syncthreads();

  // Single block: push my shard to every peer's output[myPe] slot (+ self). The
  // destination slot is myPe for every peer, so my local output[myPe] pointer
  // maps to peer p's copy by the flat-VA rank delta (pe - myPe)*stride4G<<32 --
  // the same inline LSA addressing the reduce-scatter/all-reduce push kernels
  // use. winBase cancels out, so this needs no ccoGetLocalPtr/ccoGetLsaPeerPtr
  // round trip and keeps no "output at heap offset 0" assumption. The source is
  // our single shard (same for all peers).
  StartSdmaScatter(
      devComm.sdma, npes, /*logS=*/0, pushGen,
      [=](int peer) {
        auto *localDst = reinterpret_cast<uint8_t*>(output) + myPe * chunkBytes;
        auto [dstP, sigP] = PeerPtrs(localDst, arrived + myPe * kArrivedStride,
                                     peer, myPe, stride4G);
        return ScatterPeer{true, chunkBytes, reinterpret_cast<const uint8_t*>(input),
                           dstP, sigP};
      });

  // === Phase 2: wait until all npes senders (self-copy included) signalled. ===
  if (threadIdx.x < warpSize) {
    WaitArrived(arrived, /*slot=*/0, pushGen, myPe, npes, /*skipSelf=*/false);
  }
  __syncthreads();
  // System-scope ACQUIRE: output was written by peers' SDMA over XGMI, so
  // invalidate against another device's writes before any reader trusts output.
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");

  // No Phase 3: output already fully written by SDMA. The block read pushGen at
  // entry, so it can advance the generation for the next launch now.
  if (threadIdx.x == 0) {
    AtomicStore<ESystemScope>(&barrierCtr[myPe], pushGen);
  }
}

// Multi-block, SDMA-free all-gather: every PE reads each peer's shard straight
// out of the peer's `input` over XGMI and stores it into its own output[peer].
//   input  : raw symmetric-heap pointer, my shard of chunkBytes bytes
//   output : local pointer, npes*chunkBytes bytes (all shards)
//   chunkBytes    : bytes per shard.
//   groupCtr : plain device buffer, [kArrivalIdx] elects the last block.
//   arrived       : symmetric-heap uint64[npes] gen-barrier flags of this communicator
//                   (see GenBarrier).
//   NPES          : compile-time peer count (see PullCopyInterleaved).
//
// Peer-interleaved: the copy spreads its in-flight loads across all NPES peers at
// once (PullCopyInterleaved) so NPES XGMI read links run concurrently instead of
// one peer at a time. Two GenBarrier rounds: entry (gen+1) so no block reads a
// peer's input before that peer entered this launch, and exit (gen+2) by the last
// block so "kernel done" means "input reusable". The self shard is just peer myPe
// (a local copy), so there is no in-place skip -- an in-place gather would
// redundantly self-copy output[myPe], which is harmless.
template <int NPES, int NumVecs>
__global__ void __launch_bounds__(256, 1)
AllGatherPullKernel(int myPe, int npes, cco::ccoWindow_t heapWin,
                     const void* __restrict__ input, void* __restrict__ output,
                     size_t chunkBytes, uint32_t* __restrict__ groupCtr,
                     uint64_t* __restrict__ arrived) {
  const uint32_t stride4G = heapWin->stride4G;
  // In block 0's first wave every lane loads this before lane myPe posts.
  const uint64_t gen = AtomicLoad<EAgentScope>(&arrived[myPe]) & ~1ULL;
  GenBarrier<true>(arrived, gen + 1, myPe, npes, stride4G);
  __syncthreads();

  const auto* in = static_cast<const uint8_t*>(input);
  auto* out = static_cast<uint8_t*>(output);
  // System scope: the caller may hand output to a remote reader after the kernel.
  detail::PullCopyInterleaved<NPES, NumVecs>(
      chunkBytes, [=](int j) { 
          return std::pair{PeerPtr(in, j, myPe, stride4G), 
                            out + j * chunkBytes}; });

  // A block counts in only once all its waves are done reading peer inputs.
  __syncthreads();
  if (LastBlockOut(groupCtr)) GenBarrier(arrived, gen + 2, myPe, npes, stride4G);
}
} // namespace collective
} // namespace mori
