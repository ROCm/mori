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
//   permuteCtr : symmetric-heap uint64[kPermuteFlagSlots][kCCLMaxPeers]
//              handshake flags, zeroed once, never reset. Slot [idx][p] of my
//              copy is written only by PE p (permSlot(idx, p)); the push
//              kernel uses only the kReadyFlag row.
//   permuteNum      : local uint64[2*kCCLMaxPeers], zeroed once, thread 0 only.
//              permuteNum[p] counts my permutes sending to p, permuteNum[kCCLMaxPeers+p]
//              my permutes receiving from p.
//   arrived  : symmetric-heap SDMA completion slots (see kArrivedStride), global
//              PE rows; this kernel uses column kPermuteArrived only.
//
// Phase 0: pairwise handshake, no global barrier -- only the PEs of a send/recv
// pair synchronize, so PEs that do not launch this kernel are never waited on.
// Each side counts the permutes it shares with its partner: whenever I send to
// dstPe, dstPe receives from me, so my k-th send pairs with its k-th receive.
// The receiver posts its count k into its source's kReadyFlag[myPe]; the sender
// waits until its destination's slot reaches its own count k. The slot only
// grows, so nothing is consumed or cleared.
// Phase 1: SDMA pushes sendBuf into dstPe's recvBuf (self included when
// dstPe == myPe), trailed by an atomic swap storing my send count k into my row's
// kPermuteArrived slot of the receiver's `arrived`.
// Phase 2: if srcPe >= 0, thread 0 waits until srcPe's slot reaches my receive
// count k, then an acquire fence makes the peer SDMA write visible. The slot only
// grows, so nothing is reset.
//
// A self-permute (dstPe == srcPe == myPe) counts in permuteNum[myPe] and
// permuteNum[kCCLMaxPeers+myPe] too, so the self send and self receive stay paired;
// the pull kernel never counts self-permutes, so mixing push and pull keeps them
// in step.
//
// With dstPe == -1 no peer is active, so Phase 1 issues no packets.
//
// Launch <<<1, 64>>> with npes <= 8 (SDMA fast path).
__global__ void CollectivePermutePushKernel(int npes, int dstPe, int srcPe,
                                            const void* __restrict__ sendBuf,
                                            void* __restrict__ recvBuf, size_t numBytes,
                                            cco::ccoDevComm devComm,
                                            cco::ccoWindow_t heapWin,
                                            uint64_t* __restrict__ permuteCtr,
                                            uint64_t* __restrict__ permuteNum,
                                            uint64_t* __restrict__ arrived) {
  const int myLsa = heapWin->lsaRank;
  const uint32_t stride4G = heapWin->stride4G;
  __shared__ uint64_t sendK, recvK;

  // Flat offset of flag `idx`'s slot for peer `peer` in the permuteCtr buffer.
  auto permSlot = [=](PermuteFlagIdx idx, int pe) -> uint64_t* { 
    return permuteCtr + idx * kCCLMaxPeers + pe; 
  };

  // dstPe reaching count k proves it has entered this launch, so by its stream
  // order its previous kernel (of any collective) is done with recvBuf. Without
  // it, my copy could overwrite recvBuf before dstPe consumed it. Every PE posts
  // before waiting, so a ring cannot deadlock. A self-send needs no handshake:
  // stream order covers it.
  //
  // No fences: the post only says "I have entered", and everything before it in
  // my stream is ordered by the kernel boundary; after the wait nothing reads
  // data dstPe wrote. `>=` cannot miss: dstPe posts k+1 for me only after my
  // k-th copy lands, which needs this wait to pass first.
  if (threadIdx.x == 0) {
    if (srcPe >= 0) {
      const uint64_t k = ++permuteNum[kCCLMaxPeers + srcPe];
      recvK = k;
      if (srcPe != myLsa) {
        AtomicStore<ESystemScope>(
            PeerPtr(permSlot(kReadyFlag, myLsa), srcPe, myLsa, stride4G),
            k);
      }
    }
    if (dstPe >= 0) {
      const uint64_t k = ++permuteNum[dstPe];
      sendK = k;
      if (dstPe != myLsa) {
        while (AtomicLoad<ESystemScope>(permSlot(kReadyFlag, dstPe)) < k) {
          __builtin_amdgcn_s_sleep(1);
        }
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
      devComm.sdma, npes, /*logS=*/0, sendK,
      [=](int peer) {
        auto [dstP, sigP] = PeerPtrs(localDst,
                    arrived + myLsa * kArrivedStride + kPermuteArrived,
                    peer, myLsa, stride4G);
        return ScatterPeer{
            peer == dstPe, numBytes, reinterpret_cast<const uint8_t*>(sendBuf),
            dstP, sigP};
      });

  if (srcPe >= 0) {
    if (threadIdx.x == 0) {
      const uint64_t* addr = arrived + srcPe * kArrivedStride + kPermuteArrived;
      while (AtomicLoad<ESystemScope>(addr) < recvK) {
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
    // System-scope ACQUIRE: recvBuf was written by srcPe's SDMA over XGMI.
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  }
}

// Multi-block, SDMA-free collective-permute (one destination): every PE reads
// srcPe's sendBuf straight over XGMI into its own recvBuf. Same arguments as
// CollectivePermutePushKernel. It shares the same permuteCtr buffer and uses
// three of its rows (slot [idx][p] written only by PE p):
//   kReadyFlag     : posted into srcPe's row as the "receiver has entered" token.
//   kSendReadyFlag : the number of permutes in which p has entered to send to me.
//   kReadDoneFlag  : the number of permutes in which p has finished reading my
//                    sendBuf.
//   groupCtr : plain device buffer, [kArrivalIdx] elects the last block.
//
// Pull reverses the push dependencies: the receiver may read only once the
// sender has entered (its producers of sendBuf have retired), and the sender may
// complete only once the receiver is done reading sendBuf. Both sides number
// their shared permutes with the same permuteNum[] as the push kernel (k-th send to p
// pairs with p's k-th receive from me), so push and pull launches can be mixed:
//   * entry, block 0: post k into dstPe's kSendReadyFlag[me]; also post k into
//     srcPe's kReadyFlag[me], the push kernel's "receiver has entered" token.
//   * every block of a receiver waits for kSendReadyFlag[srcPe] >= k, then copies.
//   * exit, last block: post k into srcPe's kReadDoneFlag[me], then, if sending,
//     wait for kReadDoneFlag[dstPe] >= k; finally bump permuteNum.
// Every block reads permuteNum at entry and only the last block, after every block has
// counted in, writes it, so no block of this launch can see the new value. Every
// PE posts before it waits, so rings and 2-cycles cannot deadlock (the grid is
// co-resident). dstPe == srcPe == myPe is a plain local copy without handshake.
template <int NumVecs>
__global__ void __launch_bounds__(256, 1)
CollectivePermutePullKernel(int npes, int dstPe, int srcPe, cco::ccoWindow_t heapWin,
                            const void* __restrict__ sendBuf, void* __restrict__ recvBuf,
                            size_t numBytes, uint64_t* __restrict__ permuteCtr,
                            uint64_t* __restrict__ permuteNum, uint32_t* __restrict__ groupCtr) {
  const int me = heapWin->lsaRank;
  const uint32_t stride4G = heapWin->stride4G;
  const bool sends = dstPe >= 0 && dstPe != me;
  const bool recvs = srcPe >= 0 && srcPe != me;
  const uint64_t kSend = sends ? AtomicLoad<EAgentScope>(&permuteNum[dstPe]) + 1 : 0;
  const uint64_t kRecv = recvs ? AtomicLoad<EAgentScope>(&permuteNum[kCCLMaxPeers + srcPe]) + 1 : 0;

  // Flat offset of flag `idx`'s slot for peer `peer` in the permuteCtr buffer.
  auto permSlot = [=](PermuteFlagIdx idx, int pe) -> uint64_t* { 
    return permuteCtr + idx * kCCLMaxPeers + pe; 
  };

  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) {
      // Orders the stores of sendBuf's producers ahead of the post.
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
      if (sends)
        AtomicStore<ESystemScope>(
            PeerPtr(permSlot(kSendReadyFlag, me), dstPe, me, stride4G), kSend);
      if (recvs)
        AtomicStore<ESystemScope>(
            PeerPtr(permSlot(kReadyFlag, me), srcPe, me, stride4G), kRecv);
    }
    if (recvs) {
      while (AtomicLoad<ESystemScope>(permSlot(kSendReadyFlag, srcPe)) < kRecv) {
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
      PullCopySegments<NumVecs>(1, numBytes, [=](int) { return src; }, [=](int) { return dst; });
    }
  }

  // A block counts in only once all its waves are done reading sendBuf.
  __syncthreads();
  if (threadIdx.x < warpSize && LastBlockOut(groupCtr) && threadIdx.x == 0) {
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "");
    if (recvs)
      AtomicStore<ESystemScope>(
          PeerPtr(permSlot(kReadDoneFlag, me), srcPe, me, stride4G), kRecv);
    if (sends) {
      while (AtomicLoad<ESystemScope>(permSlot(kReadDoneFlag, dstPe)) < kSend) {
        __builtin_amdgcn_s_sleep(1);
      }
      permuteNum[dstPe] = kSend;
    }
    if (recvs) permuteNum[kCCLMaxPeers + srcPe] = kRecv;
  }
}
} // namespace collective
} // namespace mori
