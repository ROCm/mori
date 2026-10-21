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
// collective_permute_kernels.hpp
//
// SDMA-based "push" collective-permute for at most one destination. PE myPe copies
// sendBuf into peer dstPe's recvBuf (self-copy when dstPe == myPe; dstPe < 0 means
// no outgoing) and waits for srcPe's copy to land in our recvBuf (srcPe < 0 means
// no incoming).
//
// Compute-free: one block, logS=0, reusing StartSdmaScatter from
// collectives_common.hpp. sendBuf and recvBuf must be symmetric-heap pointers
// so the address-based SDMA put can translate local->peer (offset from
// heapBaseAddr).
// ===========================================================================
#pragma once

#include <cstdint>
#include "mori/collective/XLA/collectives_common.hpp"

namespace mori {
namespace collective {

// Single-block, compute-free collective-permute (one destination).
//   sendBuf  : raw symmetric-heap pointer, numBytes to send
//   recvBuf  : raw symmetric-heap pointer, numBytes receive slot
//   dstPe    : peer that receives our sendBuf (may be myPe), or -1 if nobody does
//   srcPe    : peer that sends into our recvBuf, or -1 if nobody does
//
//   ready    : symmetric-heap uint64[>= npes] ready counts, zeroed once, never
//              reset. Slot p of my copy is written only by PE p.
//   cnt      : local uint64[2*kRSPushMaxPeers], zeroed once, thread 0 only.
//              cnt[p] counts my permutes sending to p, cnt[kRSPushMaxPeers+p]
//              my permutes receiving from p.
//
// Phase 0: pairwise handshake, no global barrier -- only the PEs of a send/recv
// pair synchronize, so PEs that do not launch this kernel are never waited on.
// Each side counts the permutes it shares with its partner: whenever I send to
// dstPe, dstPe receives from me, so my k-th send pairs with its k-th receive.
// The receiver posts its count k into its source's ready[myPe]; the sender
// waits until its destination's slot reaches its own count k. The slot only
// grows, so nothing is consumed or cleared.
// Phase 1: SDMA pushes sendBuf into dstPe's recvBuf (self included when
// dstPe == myPe), trailed by an ADD64 of 1 into the receiver's signalPtrs[0].
// Phase 2: if srcPe >= 0, thread 0 waits until the counter reaches 1, then an
// acquire fence makes the peer SDMA write visible. Finally thread 0 resets the
// counter for the next launch.
//
// With dstPe == -1 no peer is active, so Phase 1 issues no packets.
//
// Launch <<<1, 64>>> with npes <= 8 (SDMA fast path).
__global__ void CollectivePermutePushKernel(int npes, int dstPe, int srcPe,
                                            const void* __restrict__ sendBuf,
                                            void* __restrict__ recvBuf, size_t numBytes,
                                            cco::ccoDevComm devComm,
                                            cco::ccoWindow_t heapWin,
                                            uint64_t* __restrict__ ready,
                                            uint64_t* __restrict__ cnt) {
  const int myLsaRank = heapWin->lsaRank;
  const uint32_t stride4G = heapWin->stride4G;
  // dstPe reaching count k proves it has entered this launch, so by its stream
  // order its previous kernel (of any collective) is done with recvBuf and has
  // cleared signalBuf[0]. Without it, my copy could overwrite recvBuf before
  // dstPe consumed it, or my ADD could land before its end-of-launch reset and be
  // lost. Every PE posts before waiting, so a ring cannot deadlock. A self-send
  // needs no handshake: stream order covers it.
  //
  // No fences: the post only says "I have entered", and everything before it in
  // my stream is ordered by the kernel boundary; after the wait nothing reads
  // data dstPe wrote. `>=` cannot miss: dstPe posts k+1 for me only after my
  // k-th copy lands, which needs this wait to pass first.
  if (threadIdx.x == 0) {
    if (srcPe >= 0 && srcPe != myLsaRank) {
      const uint64_t k = ++cnt[kRSPushMaxPeers + srcPe];
      const int32_t diff = (srcPe - myLsaRank) * static_cast<int32_t>(stride4G);
      auto* tok = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(&ready[myLsaRank]) +
                                              (static_cast<uint64_t>(diff) << 32));
      AtomicStore<ESystemScope>(tok, k);
    }
    if (dstPe >= 0 && dstPe != myLsaRank) {
      const uint64_t k = ++cnt[dstPe];
      while (AtomicLoad<ESystemScope>(&ready[dstPe]) < k) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
  }
  __syncthreads();

  // recvBuf's byte offset within the heap window is identical on every rank
  // (symmetric layout), so my local recvBuf pointer maps to peer p's copy by the
  // flat-VA rank delta (pe - lsaRank)*stride4G<<32 -- the same inline LSA
  // addressing the reduce-scatter/all-reduce push kernels use. winBase cancels
  // out, so this needs no ccoGetLocalPtr/ccoGetLsaPeerPtr round trip.
  uint8_t* const localDst = reinterpret_cast<uint8_t*>(recvBuf);
  StartSdmaScatter(
      devComm.sdma, npes, /*logS=*/0, numBytes,
      [=](int peer) { return peer == dstPe; },
      [=](int) -> const uint8_t* { return reinterpret_cast<const uint8_t*>(sendBuf); },
      [=](int peer) -> uint8_t* {
        int32_t diff = (peer - myLsaRank) * static_cast<int32_t>(stride4G);
        return localDst + (static_cast<uint64_t>(diff) << 32);
      });

  uint64_t* __restrict__ signalBuf = devComm.sdma.signalBuf;
  const uint32_t want = (srcPe >= 0) ? 1u : 0u;

  if (threadIdx.x == 0 && want) {
    auto* addr = reinterpret_cast<uint32_t*>(&signalBuf[0]);  // S=1
    while (AtomicLoad<EAgentScope>(addr) < want) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  if (want) {
    // System-scope ACQUIRE: recvBuf was written by srcPe's SDMA over XGMI.
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  }

  // Reset the counter for reuse. SYSTEM scope (sc0 sc1: write-through past L2)
  // because srcPe ADDs into this slot by SDMA straight into HBM -- a plain store
  // would leave a dirty line that could evict on top of a later ADD and swallow
  // it. That is also why no release fence follows: there is nothing left in L2 to
  // write back.
  //
  // Clearing at kernel END is safe: srcPe's next ADD only follows my next
  // post to it, which I make only once my next launch has started.
  if (threadIdx.x == 0) {
    AtomicStore<ESystemScope>(&signalBuf[0], 0UL);
  }
}

// Multi-block, SDMA-free collective-permute (one destination): every PE reads
// srcPe's sendBuf straight over XGMI into its own recvBuf. Same arguments as
// CollectivePermutePushKernel, plus:
//   sendReady : symmetric-heap uint64[>= npes], zeroed once, never reset. Slot p
//               of my copy is written only by PE p: the number of permutes in
//               which p has entered to send to me.
//   readDone  : symmetric-heap uint64[>= npes], same rules: the number of
//               permutes in which p has finished reading my sendBuf.
//   groupCounters : plain device buffer, [kArrivalIdx] elects the last block.
//
// Pull reverses the push dependencies: the receiver may read only once the
// sender has entered (its producers of sendBuf have retired), and the sender may
// complete only once the receiver is done reading sendBuf. Both sides number
// their shared permutes with the same cnt[] as the push kernel (k-th send to p
// pairs with p's k-th receive from me), so push and pull launches can be mixed:
//   * entry, block 0: post k into dstPe's sendReady[me]; also post k into
//     srcPe's ready[me], the push kernel's "receiver has entered" token.
//   * every block of a receiver waits for sendReady[srcPe] >= k, then copies.
//   * exit, last block: post k into srcPe's readDone[me], then, if sending, wait
//     for readDone[dstPe] >= k; finally bump cnt.
// Every block reads cnt at entry and only the last block, after every block has
// counted in, writes it, so no block of this launch can see the new value. Every
// PE posts before it waits, so rings and 2-cycles cannot deadlock (the grid is
// co-resident). dstPe == srcPe == myPe is a plain local copy without handshake.
__global__ void __launch_bounds__(256, 1)
CollectivePermutePullKernel(int npes, int dstPe, int srcPe, cco::ccoWindow_t heapWin,
                            const void* __restrict__ sendBuf, void* __restrict__ recvBuf,
                            size_t numBytes, uint64_t* __restrict__ ready,
                            uint64_t* __restrict__ sendReady, uint64_t* __restrict__ readDone,
                            uint64_t* __restrict__ cnt, uint32_t* __restrict__ groupCounters) {
  const int me = heapWin->lsaRank;
  const uint32_t stride4G = heapWin->stride4G;
  const bool sends = dstPe >= 0 && dstPe != me;
  const bool recvs = srcPe >= 0 && srcPe != me;
  const uint64_t kSend = sends ? AtomicLoad<EAgentScope>(&cnt[dstPe]) + 1 : 0;
  const uint64_t kRecv = recvs ? AtomicLoad<EAgentScope>(&cnt[kRSPushMaxPeers + srcPe]) + 1 : 0;

  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) {
      // Orders the stores of sendBuf's producers ahead of the post.
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
      if (sends) AtomicStore<ESystemScope>(PeerPtr(&sendReady[me], dstPe, me, stride4G), kSend);
      if (recvs) AtomicStore<ESystemScope>(PeerPtr(&ready[me], srcPe, me, stride4G), kRecv);
    }
    if (recvs) {
      while (AtomicLoad<ESystemScope>(&sendReady[srcPe]) < kRecv) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
  }
  __syncthreads();
  if (recvs) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");

  if (srcPe >= 0) {
    const auto* src = PeerPtr(static_cast<const uint8_t*>(sendBuf), srcPe, me, stride4G);
    auto* dst = static_cast<uint8_t*>(recvBuf);
    if (src != dst) {
      PullCopySegments(1, numBytes, [=](int) { return src; }, [=](int) { return dst; });
    }
  }

  // A block counts in only once all its waves are done reading sendBuf.
  __syncthreads();
  if (threadIdx.x < warpSize && LastBlockOut(groupCounters) && threadIdx.x == 0) {
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "");
    if (recvs) AtomicStore<ESystemScope>(PeerPtr(&readDone[me], srcPe, me, stride4G), kRecv);
    if (sends) {
      while (AtomicLoad<ESystemScope>(&readDone[dstPe]) < kSend) {
        __builtin_amdgcn_s_sleep(1);
      }
      cnt[dstPe] = kSend;
    }
    if (recvs) cnt[kRSPushMaxPeers + srcPe] = kRecv;
  }
}
} // namespace collective
} // namespace mori
