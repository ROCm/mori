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
//
// DEVICE ONLY. Included by the generated TU, never by a host source.
//
// EP intranode dispatch + combine on cco-LSA, ported from
// src/ops/dispatch_combine/intranode.hpp (the bf16 gather path). Mechanical:
// `memObj->GetAs<T*>(pe)` became EpPeer<T>(), `GetAs<T*>()` became EpLocal<T>(),
// both offsets into one cco window. The algorithm is unchanged.
//
// mori/shmem is deliberately not included: its WaitUntil* helpers are plain spin
// loops, reproduced here as EpWaitEq/EpWaitGt, so this TU carries no device
// globals and needs no per-module init.

#pragma once

#include <hip/hip_runtime.h>

#include <type_traits>

#include "mori/cco/cco.hpp"
#include "mori/core/transport/p2p/device_primitives.hpp"
#include "mori/ops/dispatch_combine_v2/ep_cfg.hpp"
#include "src/ops/dispatch_combine_v2/ep_scatter_quant.hpp"

namespace mori {
namespace ops {
namespace v2 {

// ---------------------------------------------------------------------------
// Arena addressing. Replaces SymmMemObjPtr::GetAs -- a two dependent-load table
// lookup -- with the flat-VA formula. `off` is a launch argument: one uniform
// scalar load, versus two dependent loads for the table.
//
// Intranode only: peer indices here are LSA ranks, which equal world ranks
// while the whole world is one LSA team. EpCfgIsValid keeps worldSize inside a
// wavefront, which is the same single-node assumption.
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ T* EpPeer(unsigned long long win, int peer, unsigned long long off) {
  return reinterpret_cast<T*>(::mori::cco::ccoGetLsaPeerPtr(
      reinterpret_cast<::mori::cco::ccoWindow_t>(win), peer, static_cast<size_t>(off)));
}

template <typename T>
__device__ __forceinline__ T* EpLocal(unsigned long long win, unsigned long long off) {
  return reinterpret_cast<T*>(::mori::cco::ccoGetLocalPtr(
      reinterpret_cast<::mori::cco::ccoWindow_t>(win), static_cast<size_t>(off)));
}

struct EpWinC {
  unsigned long long h;
};

using EpWinConstPtr = const __attribute__((address_space(4))) ::mori::cco::ccoWindowDevice*;

template <typename T>
__device__ __forceinline__ T* EpPeer(EpWinC win, int peer, unsigned long long off) {
  const EpWinConstPtr w = reinterpret_cast<EpWinConstPtr>(win.h);
  return reinterpret_cast<T*>(w->winBase + ((static_cast<uint64_t>(peer) * w->stride4G) << 32) +
                              static_cast<size_t>(off));
}

template <typename T>
__device__ __forceinline__ T* EpLocal(EpWinC win, unsigned long long off) {
  const EpWinConstPtr w = reinterpret_cast<EpWinConstPtr>(win.h);
  return reinterpret_cast<T*>(w->winBase +
                              ((static_cast<uint64_t>(w->lsaRank) * w->stride4G) << 32) +
                              static_cast<size_t>(off));
}

// Spin helpers. SYSTEM scope is load-bearing: the dispatch notify loop spins on
// a *peer's* signal word, and AGENT scope there hangs.
template <typename T>
__device__ __forceinline__ void EpWaitEq(T* addr, T val) {
  while (__hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != val) {
  }
}

template <typename T>
__device__ __forceinline__ T EpWaitGt(T* addr, T val) {
  T got;
  do {
    got = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  } while (got <= val);
  return got;
}

// (pe, localTokId) <-> flat index. The stride must be at least every PUBLISHED
// slot index, NOT the per-rank send capacity; dispatch publishes slots below
// EpMaxRecv only (a capped config drops the rest), so the capacity is the stride.
// A smaller stride aliases peer p slot s onto peer p+1 slot s-stride.
template <EpCfg kCfg>
__device__ __forceinline__ int EpFlatStride() {
  return EpMaxRecv(kCfg);
}
template <EpCfg kCfg>
__device__ __forceinline__ int EpFlatIndex(int pe, int localTokId) {
  return pe * EpFlatStride<kCfg>() + localTokId;
}
// The reverse map ("tis": recv slot -> global source token id) is a PUBLIC output
// and its stride is maxTokPerRank, NOT the forward stride above: forward encodes
// (destPe, recv slot) over maxRecv, reverse (srcPe, source token) over
// maxTokPerRank. FlyDSL publishes rank*maxTokPerRank + srcTok, so this must match
// or one handle decodes differently per backend.
template <EpCfg kCfg>
__device__ __forceinline__ int EpSrcTokIndex(int pe, int srcTokId) {
  return pe * kCfg.maxTokPerRank + srcTokId;
}
template <EpCfg kCfg>
__device__ __forceinline__ int EpPeFromFlat(int flat) {
  return flat / EpFlatStride<kCfg>();
}
template <EpCfg kCfg>
__device__ __forceinline__ int EpLocalTokFromFlat(int flat) {
  return flat % EpFlatStride<kCfg>();
}
// "No destination": decodes to pe == worldSize, which combine reads as a null
// source. Must use the SAME stride as the encoder -- with a mismatched stride it
// decodes to a real peer and combine silently folds in someone else's token.
template <EpCfg kCfg>
__device__ __forceinline__ int EpNullFlat() {
  return kCfg.worldSize * EpFlatStride<kCfg>();
}

// Partitions numItems x hiddenDim across the grid's warps: several warps split
// one token when warps outnumber tokens, one warp takes several otherwise.
struct EpMultiWarpIter {
  int warpsPerItem;
  size_t dimPerWarp;
  size_t dimSize;

  __device__ EpMultiWarpIter(int globalWarpNum, int numItems, size_t dim) : dimSize(dim) {
    warpsPerItem = (globalWarpNum + numItems - 1) / numItems;
    if (warpsPerItem < 1) warpsPerItem = 1;
    dimPerWarp = (dimSize + warpsPerItem - 1) / warpsPerItem;
  }

  __device__ void Decode(int i, int& itemId, int& inItemPartId, size_t& dimOffset,
                         size_t& dimChunk) const {
    itemId = i / warpsPerItem;
    inItemPartId = i % warpsPerItem;
    dimOffset = static_cast<size_t>(inItemPartId) * dimPerWarp;
    dimChunk = (dimOffset < dimSize) ? min(dimSize - dimOffset, dimPerWarp) : size_t{0};
  }
};

/* ------------------------------------------------------------------------- */
/*                                  Dispatch                                  */
/* ------------------------------------------------------------------------- */
// Dispatch's tail, shared by both bodies: every block reports in, then this rank's
// per-destination counts go out and everyone else's come back.
//
// Tickets on the arrival counter rather than a spin on it, after the gfx125x body
// (ep_intranode_1250x.hpp). Every block draws one; the FIRST arriver drains the
// peers' signal slots and then draws a second; the HIGHEST draw publishes the
// counts. Holding the highest ticket implies both preconditions of that store --
// every block in, so destPeTokenCounter is a complete sum, and the drain observed.
//
// Two things this buys over spinning in block 0:
//   * the drain -- waiting for each peer to have consumed the PREVIOUS call's
//     signal -- is a fabric round trip on uncached peer memory, and its address
//     depends only on destPe, so it needs nothing the barrier gives. On the first
//     arriver it is spent while the other blocks are still copying; after the
//     barrier it is exposed.
//   * the warp that waits is whichever block got here first, not block 0, which
//     may arrive early and then spin for the whole of everyone else's work.
template <EpCfg kCfg>
__device__ __forceinline__ void EpDispatchNotify(EpArgs args) {
  constexpr int kNpes = kCfg.worldSize;
  constexpr unsigned kNoTicket = ~0u;
  const int myPe = args.rank;
  const int thdId = threadIdx.x;
  const int laneId = threadIdx.x & (kCfg.waveSize - 1);
  const int warpId = thdId / kCfg.waveSize;
  const unsigned long long win = args.window;

  // A block's peer stores must be COMPLETE and WRITTEN BACK before it arrives, as
  // the gfx125x body releases before its own arrival. __syncthreads() does not wait
  // on vmcnt, so every warp drains its stores into this XCD's L2 first; on a cached
  // window they are not visible to a peer until that L2 is written back, and the
  // publisher's release reaches the L2 of the XCD IT runs on, not the other seven.
  // One warp issues it because buffer_wbl2 is cache-wide and the barrier has already
  // put every warp's stores in the cache it writes back.
  __builtin_amdgcn_s_waitcnt(0);
  __syncthreads();

  unsigned arriveTicket = 0;
  if (warpId == 0) {
    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    if (laneId == 0) arriveTicket = atomicAdd(args.gridBarrier, 1u);
    arriveTicket = (unsigned)__shfl((int)arriveTicket, 0);
  }
  const bool isFirstArriver = (warpId == 0) && (arriveTicket == 0u);

  unsigned drainTicket = kNoTicket;
  if (isFirstArriver) {
    for (int destPe = laneId; destPe < kNpes; destPe += kCfg.waveSize) {
      EpWaitEq(EpPeer<int>(win, destPe, args.offRecvNum) + myPe, 0);
    }
    if (laneId == 0) drainTicket = atomicAdd(args.gridBarrier, 1u);
    drainTicket = (unsigned)__shfl((int)drainTicket, 0);
  }

  // atomicAdd returns the pre-increment value, so gridDim.x + 1 draws read
  // 0 .. gridDim.x: the highest TICKET, not the counter's final value.
  const unsigned kHighestTicket = gridDim.x;
  if ((warpId == 0) && (arriveTicket == kHighestTicket || drainTicket == kHighestTicket)) {
    // A release, not __threadfence_system(): the signal must not pass the payload,
    // but nothing here reads what a peer wrote, so the acquire half -- a full cache
    // invalidate -- is bought for nothing. Hoisted out of the loop, which wide EP
    // multi-iterates.
    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    for (int destPe = laneId; destPe < kNpes; destPe += kCfg.waveSize) {
      // +1 so a zero-token destination still sees a distinct "signal arrived".
      const int numTokenSignal = __hip_atomic_load(args.destPeTokenCounter + destPe,
                                                   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) +
                                 1;
      __hip_atomic_store(EpPeer<int>(win, destPe, args.offRecvNum) + myPe, numTokenSignal,
                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      // Cleared here, by the warp that just read it: the inbound loop below runs on
      // the first arriver, which may reach it before this warp has read the sum.
      args.destPeTokenCounter[destPe] = 0;
    }
    __hip_atomic_store(args.gridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }

  // Inbound rides the first arriver so it is already parked when the peers' signals
  // land. It cannot move above the outbound store on the same warp: this rank would
  // wait on a peer that is waiting on this rank, with neither having sent. Separate
  // tickets are what break that cycle.
  if (isFirstArriver) {
    int myRecv = 0;
    int* recvTokenNums = EpLocal<int>(win, args.offRecvNum);
    for (int srcPe = laneId; srcPe < kNpes; srcPe += kCfg.waveSize) {
      int* signal = recvTokenNums + srcPe;
      myRecv += EpWaitGt(signal, 0) - 1;
      __hip_atomic_store(signal, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    // One store instead of an atomicAdd a source -- and the total no longer depends
    // on the body having zeroed it, which block 0 did and block 0 is not necessarily
    // the warp that lands here.
    for (int off = kCfg.waveSize / 2; off > 0; off >>= 1) myRecv += __shfl_down(myRecv, off);
    if (laneId == 0) {
      *args.totalRecvTokenNum = myRecv;
      EpLocal<int>(win, args.offTokOff)[0] = 0;
    }
  }
}

/* ------------------------------------------------------------------------- */
/*                            Intranode dispatch                            */
/* ------------------------------------------------------------------------- */
// Ported from EP v1's EpDispatchIntraNodeLLKernel_body (intranode_ll.hpp), and since
// then the only intranode dispatch off gfx125x. One block owns one token at a time and its warps
// split by role:
//
//   warp 0       control plane: route the NEXT token, then forward the current
//                token's expert ids and weights to its destinations
//   warp 1       the scale row, when Cfg.scaleBytes > 0
//   the others   data plane: load a slice of the payload once into registers and
//                store the same registers to every destination
//
// The route sits in LDS, double buffered, so the control warp's atomics for token
// N+1 overlap the copy of token N; one block barrier a token. A copy warp's slice
// is not a runtime division as in v1: warpPerBlock, hiddenDim and the dtype are Cfg
// constants, so EpRow* fix how many rows each warp takes and the compiler unrolls the
// row loop to exactly that.

template <int kMaxDests>
struct EpRoute {
  int numDests;
  int srcTok;
  int destRank[kMaxDests];
  int recvSlot[kMaxDests];
};

// v_readlane of a 64-bit pointer: the result is uniform, so the address stays in
// SGPRs for the stores that follow.
template <typename P>
__device__ __forceinline__ P* EpReadLanePtr(P* ptr, int srcLane) {
  const unsigned long long v = reinterpret_cast<unsigned long long>(ptr);
  const uint32_t lo = __builtin_amdgcn_readlane(static_cast<uint32_t>(v), srcLane);
  const uint32_t hi = __builtin_amdgcn_readlane(static_cast<uint32_t>(v >> 32), srcLane);
  return reinterpret_cast<P*>((static_cast<unsigned long long>(hi) << 32) | lo);
}

// Warp 0 only. Lane k < topk names the rank of expert slot k; lane p < worldSize
// then reads all topk slots through readlane and learns whether (and through which
// slot) the token goes to rank p. The loop runs topk times, a Cfg constant: v1
// unrolled a fixed 8, which dropped the slots past it. The lowest slot that targets a
// rank owns the copy, as in the portable body.
template <EpCfg kCfg>
__device__ __forceinline__ void EpDispatchComputeRoute(const EpArgs& args, int tokenId,
                                                       EpRoute<EpMaxDests(kCfg)>& route,
                                                       int& didAlloc, int& slotForMe, int& mySlot) {
  constexpr int kTopk = kCfg.numExpertPerToken;
  constexpr int kNpes = kCfg.worldSize;
  const int laneId = threadIdx.x & (kCfg.waveSize - 1);

  route.srcTok = tokenId;

  int targetPe = -1;
  if (laneId < kTopk) {
    const int expert = args.tokenIndices[tokenId * kTopk + laneId];
    const int pe = expert >= 0 ? expert / kCfg.numExpertPerRank : -1;
    targetPe = pe < kNpes ? pe : -1;
  }

  slotForMe = -1;
#pragma unroll
  for (int k = kTopk - 1; k >= 0; --k) {
    const int peK = __builtin_amdgcn_readlane(targetPe, k);
    slotForMe = (peK == laneId) ? k : slotForMe;
  }

  didAlloc = 0;
  mySlot = 0;
  if (slotForMe >= 0) {
    didAlloc = 1;
    mySlot = __hip_atomic_fetch_add(EpPeer<int>(args.window, laneId, args.offTokOff), 1,
                                    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    // Under a receive cap a slot past it is dropped. The offset is still consumed,
    // as the portable body consumes it: the claim is what orders the ranks, and a
    // dropped claim must not renumber the ones that fit. Dropping before the ballot
    // keeps the destination out of the route, so no copy warp sends to it, and
    // EpDispatchWriteMap leaves its map slot null.
    if constexpr (EpRecvCapped(kCfg)) {
      if (mySlot >= EpMaxRecv(kCfg)) didAlloc = 0;
    }
  }

  const unsigned long long active = __ballot(didAlloc);
  if (didAlloc) {
    const int pos = __popcll(active & ((1ull << laneId) - 1));
    route.destRank[pos] = laneId;
    route.recvSlot[pos] = mySlot;
  }
  if (laneId == 0) route.numDests = __popcll(active);
}

// Replay: the handle's dest map already names every slot, so nothing is claimed and
// nothing is written back. Lane k reads top-k slot k rather than owning destination
// k, so the rank is decoded instead of being the lane -- which is why destPe is an
// output here and not simply laneId. Dedup happened when the map was written: a
// duplicate slot is already EpNullFlat, so the surviving slots are exactly the
// distinct destinations the route wants.
template <EpCfg kCfg>
__device__ __forceinline__ void EpDispatchReplayRoute(const EpArgs& args, int tokenId,
                                                      EpRoute<EpMaxDests(kCfg)>& route,
                                                      int& didAlloc, int& destPe, int& mySlot) {
  constexpr int kTopk = kCfg.numExpertPerToken;
  constexpr int kNpes = kCfg.worldSize;
  const int laneId = threadIdx.x & (kCfg.waveSize - 1);

  route.srcTok = tokenId;
  didAlloc = 0;
  destPe = -1;
  mySlot = 0;
  if (laneId < kTopk) {
    const int flat = args.dispDestTokIdMap[tokenId * kTopk + laneId];
    const int pe = EpPeFromFlat<kCfg>(flat);
    if (flat >= 0 && pe < kNpes) {
      didAlloc = 1;
      destPe = pe;
      mySlot = EpLocalTokFromFlat<kCfg>(flat);
    }
  }

  const unsigned long long active = __ballot(didAlloc);
  if (didAlloc) {
    const int pos = __popcll(active & ((1ull << laneId) - 1));
    route.destRank[pos] = destPe;
    route.recvSlot[pos] = mySlot;
  }
  if (laneId == 0) route.numDests = __popcll(active);
}

// Deferred past the block barrier so the copy warps are not held up by it.
template <EpCfg kCfg>
__device__ __forceinline__ void EpDispatchWriteMap(const EpArgs& args, int tokenId, int didAlloc,
                                                   int slotForMe, int mySlot) {
  constexpr int kTopk = kCfg.numExpertPerToken;
  const int laneId = threadIdx.x & (kCfg.waveSize - 1);

  // Which rank this lane speaks for: its own in the routing pass, the one it
  // decoded in replay.
  const int destPe = kCfg.replay ? slotForMe : laneId;
  if constexpr (!kCfg.replay) {
    if (laneId < kTopk) args.dispDestTokIdMap[tokenId * kTopk + laneId] = EpNullFlat<kCfg>();
  }
  if (didAlloc) {
    atomicAdd(args.destPeTokenCounter + destPe, 1);
    EpPeer<int>(args.window, destPe, args.offRecvToSrc)[mySlot] =
        EpSrcTokIndex<kCfg>(args.rank, tokenId);
    if constexpr (!kCfg.replay) {
      args.dispDestTokIdMap[tokenId * kTopk + slotForMe] = EpFlatIndex<kCfg>(laneId, mySlot);
    }
  }
}

// The routing pass or the replay pass, whichever this Cfg renders. slotForMe carries
// the map slot to write in the first and the decoded rank in the second; EpDispatchWriteMap
// reads it the same way.
template <EpCfg kCfg>
__device__ __forceinline__ void EpDispatchRoute(const EpArgs& args, int tokenId,
                                                EpRoute<EpMaxDests(kCfg)>& route, int& didAlloc,
                                                int& slotForMe, int& mySlot) {
  if constexpr (kCfg.replay) {
    EpDispatchReplayRoute<kCfg>(args, tokenId, route, didAlloc, slotForMe, mySlot);
  } else {
    EpDispatchComputeRoute<kCfg>(args, tokenId, route, didAlloc, slotForMe, mySlot);
  }
}

template <EpCfg kCfg, typename T>
__device__ void EpDispatchBody(EpArgs args) {
  constexpr int kTopk = kCfg.numExpertPerToken;
  constexpr int kCtrl = EpCtrlWarps(kCfg);
  constexpr int kCopyWarps = EpCopyWarps(kCfg);
  constexpr int kRows = EpRowsPerToken(kCfg);
  constexpr int kMaxRows = EpMaxRowCount(kCfg);
  constexpr bool kEvenRows = (kRows % kCopyWarps) == 0;
  constexpr bool kPartialRow = EpLastRowVecs(kCfg) < kCfg.waveSize;
  constexpr size_t kRowBytes = EpRowBytes(kCfg);
  constexpr size_t kTokBytes = EpTokenBytes(kCfg);
  constexpr int kChunk = kMaxRows < 4 ? kMaxRows : 4;
  // No warp has a short chunk: every row of every chunk exists.
  constexpr bool kRowsAlwaysLive = kEvenRows && (kMaxRows % kChunk) == 0;

  const int thdId = threadIdx.x;
  const int laneId = thdId & (kCfg.waveSize - 1);
  const int warpId = thdId / kCfg.waveSize;
  const unsigned long long win = args.window;
  const int numTokens = args.numTokens;

  using Vec = typename core::VecTypeSelector<16>::dataType;

  __shared__ EpRoute<EpMaxDests(kCfg)> routeBuf[2];

  const int copyWarp = warpId - kCtrl;
  const int rowCount = kEvenRows ? kRows / kCopyWarps : EpRowCount(kCfg, copyWarp);
  const int rowBegin = EpRowBegin(kCfg, copyWarp);
  const size_t laneOff = static_cast<size_t>(laneId) * 16;
  Vec pre[kChunk];

  if (args.tokenIndices && args.inpTokenBuf) {
    // Prologue: warp 0 routes the first token this block owns -- a remote atomic a
    // destination, and nothing else in the block can start until it lands.
    //
    // So the copy warps read that token's payload here, BEFORE the barrier. A token's
    // source address is blockIdx.x alone; only the DESTINATION needs the route. The
    // barrier waits on lgkmcnt only, so these loads stay in flight across it and the
    // atomic's latency is spent on them. Where a block holds one token -- every call
    // with no more tokens than blocks -- this prologue IS the kernel.
    int nextDid = 0, nextSlotK = -1, nextSlot = 0;
    if (warpId == 0 && blockIdx.x < numTokens) {
      EpDispatchRoute<kCfg>(args, blockIdx.x, routeBuf[0], nextDid, nextSlotK, nextSlot);
    } else if (warpId >= kCtrl && blockIdx.x < numTokens && rowCount > 0) {
      const uint8_t* src = reinterpret_cast<const uint8_t*>(args.inpTokenBuf) +
                           static_cast<size_t>(blockIdx.x) * kTokBytes +
                           static_cast<size_t>(rowBegin) * kRowBytes;
#pragma unroll
      for (int u = 0; u < kChunk; ++u) {
        bool live = kRowsAlwaysLive || u < rowCount;
        if constexpr (kPartialRow)
          live = live && !(rowBegin + u == kRows - 1 && laneId >= EpLastRowVecs(kCfg));
        if (live) pre[u] = core::load<16>(src + u * kRowBytes + laneOff);
      }
    }
    __syncthreads();
    if (warpId == 0 && blockIdx.x < numTokens) {
      EpDispatchWriteMap<kCfg>(args, blockIdx.x, nextDid, nextSlotK, nextSlot);
    }

    for (int tokenIdx = blockIdx.x; tokenIdx < numTokens; tokenIdx += gridDim.x) {
      const int pingPong = (tokenIdx / gridDim.x) & 1;
      const auto& route = routeBuf[pingPong];

      if (warpId == 0) {
        const int nextTok = tokenIdx + gridDim.x;
        if (nextTok < numTokens) {
          EpDispatchRoute<kCfg>(args, nextTok, routeBuf[1 - pingPong], nextDid, nextSlotK,
                                nextSlot);
          EpDispatchWriteMap<kCfg>(args, nextTok, nextDid, nextSlotK, nextSlot);
        }
      } else if (kCfg.scaleBytes > 0 && warpId == 1) {
        if constexpr (kCfg.scaleBytes > 0) {
          constexpr int kSrcDw = kCfg.scaleBytes / 4;
          constexpr int kDstDw = EpScaleStride(kCfg) / 4;
          constexpr int kIters = (kDstDw + kCfg.waveSize - 1) / kCfg.waveSize;
          const int numDests = route.numDests;
          if (numDests > 0 && args.scalesBuf) {
            const unsigned int* src = reinterpret_cast<const unsigned int*>(args.scalesBuf) +
                                      static_cast<size_t>(route.srcTok) * kSrcDw;
            unsigned int v[kIters];
#pragma unroll
            for (int i = 0; i < kIters; ++i) {
              const int e = laneId + i * kCfg.waveSize;
              v[i] = e < kSrcDw ? src[e] : 0u;
            }
            for (int d = 0; d < numDests; ++d) {
              const int rank = __builtin_amdgcn_readfirstlane(route.destRank[d]);
              const int slot = __builtin_amdgcn_readfirstlane(route.recvSlot[d]);
              unsigned int* dst = EpPeer<unsigned int>(win, rank, args.offOutScales) +
                                  static_cast<size_t>(slot) * kDstDw;
#pragma unroll
              for (int i = 0; i < kIters; ++i) {
                const int e = laneId + i * kCfg.waveSize;
                if (e < kDstDw) dst[e] = v[i];
              }
            }
          }
        }
      } else {
        const int numDests = route.numDests;

        // The expert ids and weights go out from here too, a destination to a copy
        // warp: on warp 0 they were a serial loop over every destination, and at one
        // token a block that loop WAS the kernel (ablation at 1 token a rank, EP8,
        // hidden 7168: 10.7us with no token work at all, 15.5 with the metadata on
        // warp 0 and no payload, 17.1 complete).
        if (numDests > 0) {
          const int srcTok = route.srcTok;
          float wt = 0.0f;
          int idx = 0;
          if (laneId < kTopk) {
            if constexpr (kCfg.useWeights) {
              if (args.weightsBuf) wt = args.weightsBuf[srcTok * kTopk + laneId];
            }
            idx = args.tokenIndices[srcTok * kTopk + laneId];
          }
          for (int d = copyWarp; d < numDests; d += kCopyWarps) {
            const int rank = __builtin_amdgcn_readfirstlane(route.destRank[d]);
            const int slot = __builtin_amdgcn_readfirstlane(route.recvSlot[d]);
            if (laneId < kTopk) {
              if constexpr (kCfg.useWeights) {
                if (args.weightsBuf) {
                  EpPeer<float>(win, rank, args.offOutWts)[slot * kTopk + laneId] = wt;
                }
              }
              EpPeer<int>(win, rank, args.offOutIdx)[slot * kTopk + laneId] = idx;
            }
          }
        }

        if (numDests > 0 && rowCount > 0) {
          // Lane i holds destination i's address; the warps alternate the order so
          // neighbours do not all start on the same peer.
          uint8_t* dstReg = nullptr;
          if (laneId < numDests) {
            const int di = (copyWarp & 1) ? numDests - 1 - laneId : laneId;
            dstReg = EpPeer<uint8_t>(win, route.destRank[di], args.offDispOut) +
                     static_cast<size_t>(route.recvSlot[di]) * kTokBytes +
                     static_cast<size_t>(rowBegin) * kRowBytes;
          }
          const uint8_t* src = reinterpret_cast<const uint8_t*>(args.inpTokenBuf) +
                               static_cast<size_t>(route.srcTok) * kTokBytes +
                               static_cast<size_t>(rowBegin) * kRowBytes;
          // The prologue already read chunk 0 of this block's first token.
          bool pre_ok = tokenIdx == blockIdx.x;

          for (int c0 = 0; c0 < rowCount; c0 += kChunk) {
            Vec v[kChunk];
#pragma unroll
            for (int u = 0; u < kChunk; ++u) {
              const int row = c0 + u;
              bool live = kRowsAlwaysLive || row < rowCount;
              if constexpr (kPartialRow)
                live = live && !(rowBegin + row == kRows - 1 && laneId >= EpLastRowVecs(kCfg));
              if (live)
                v[u] =
                    (pre_ok && c0 == 0) ? pre[u] : core::load<16>(src + row * kRowBytes + laneOff);
            }
            for (int d = 0; d < numDests; ++d) {
              uint8_t* dst = EpReadLanePtr(dstReg, d);
#pragma unroll
              for (int u = 0; u < kChunk; ++u) {
                const int row = c0 + u;
                bool live = kRowsAlwaysLive || row < rowCount;
                if constexpr (kPartialRow)
                  live = live && !(rowBegin + row == kRows - 1 && laneId >= EpLastRowVecs(kCfg));
                if (live) core::store<16>(dst + row * kRowBytes + laneOff, v[u]);
              }
            }
          }
        }
      }
      __syncthreads();
    }
  }

  EpDispatchNotify<kCfg>(args);
}

/* ------------------------------------------------------------------------- */
/*                            Cross-device barrier                            */
/* ------------------------------------------------------------------------- */
// Two stages: a grid-wide rendezvous inside this device, then a monotone epoch
// published to every peer's slot and awaited locally. The epoch comes from a
// host-owned counter so it never repeats across launches.
template <EpCfg kCfg>
__device__ __forceinline__ void EpCrossDeviceBarrier(EpArgs args, unsigned long long flag) {
  constexpr int kNpes = kCfg.worldSize;
  const int thdId = threadIdx.x;
  const int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long win = args.window;

  __syncthreads();
  // Release THIS block's staging writes before announcing arrival. v1 calls this
  // unnecessary given the acquire below; without it exactly half the tokens came
  // back wrong at 80 blocks. Both are kept -- that failure was found with neither
  // present, so which one alone sufficed was never isolated.
  __threadfence_system();
  unsigned _gridArvl = 0;
  if (thdId == 0) _gridArvl = atomicAdd(args.gridBarrier, 1u);

  if constexpr (!EpIsWideEp(kCfg)) {
    // Narrow path: all participating threads are in one warp, no multi-warp race.
    if (globalThdId < kNpes) {
      EpWaitEq(args.gridBarrier, static_cast<unsigned int>(gridDim.x));
      __hip_atomic_store(args.gridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

      __threadfence_system();
      __hip_atomic_store(EpPeer<unsigned long long>(win, globalThdId, args.offXdb) + args.rank,
                         flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  } else {
    // Wide path: last-arrival resets to avoid the inter-block race where a late
    // block misses the transient gridDim.x value and deadlocks on 0.
    if (thdId == 0) {
      if (_gridArvl == static_cast<unsigned>(gridDim.x) - 1)
        __hip_atomic_store(args.gridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      else
        EpWaitEq(args.gridBarrier, 0u);
    }
    __syncthreads();

    if (globalThdId < kNpes) {
      __threadfence_system();
      __hip_atomic_store(EpPeer<unsigned long long>(win, globalThdId, args.offXdb) + args.rank,
                         flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }

  if (globalThdId == 0) atomicAdd(args.xdbFlag, 1ull);

  unsigned long long* localBarrier = EpLocal<unsigned long long>(win, args.offXdb);
  if (thdId < kNpes) {
    while (__hip_atomic_load(localBarrier + thdId, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) !=
           flag) {
    }
    __threadfence_system();
  }
  __syncthreads();
}

/* ------------------------------------------------------------------------- */
/*                                   Combine                                  */
/* ------------------------------------------------------------------------- */
template <EpCfg kCfg, typename T>
__device__ void EpCombineBody(EpArgs args) {
  constexpr int kNpes = kCfg.worldSize;
  constexpr int kTopk = kCfg.numExpertPerToken;
  constexpr size_t kHidden = static_cast<size_t>(kCfg.hiddenDim);

  const int thdId = threadIdx.x;
  const int laneId = threadIdx.x & (kCfg.waveSize - 1);
  const int warpId = thdId / kCfg.waveSize;
  const int warpNum = kCfg.warpPerBlock;
  const int globalWarpId = blockIdx.x * warpNum + warpId;
  const int globalWarpNum = gridDim.x * warpNum;

  const unsigned long long win = args.window;
  const unsigned long long flag = args.xdbFlag[0];
  const int totalRecv = args.totalRecvTokenNum[0];

  // Stage the post-expert tokens into the arena so peers can gather them --
  // unless the caller already produced them there, which an expert op writing
  // straight into combine_in_view does. Skipping is not a micro-optimisation:
  // at 4k tokens x 7168 the copy is ~300 MB of pure self-copy.
  T* const stage = EpLocal<T>(win, args.offOutTok);
  if (reinterpret_cast<const T*>(args.inpTokenBuf) != stage) {
    for (int i = globalWarpId; i < totalRecv; i += globalWarpNum) {
      core::WarpCopy(stage + i * kHidden,
                     reinterpret_cast<const T*>(args.inpTokenBuf) + i * kHidden, kHidden);
    }
  }

  EpCrossDeviceBarrier<kCfg>(args, flag);

  if (args.numTokens == 0) return;

  // Per-warp pointer arrays: [srcPtrs][srcWeightPtrs]. Sized by
  // EpCombineSharedBytes, which the host uses for the launch too.
  extern __shared__ char epSharedMem[];
  T** srcPtrs = reinterpret_cast<T**>(epSharedMem) + warpId * kTopk;
  float** srcWeightPtrs = nullptr;
  if constexpr (kCfg.useWeights) {
    srcWeightPtrs = reinterpret_cast<float**>(epSharedMem) + warpNum * kTopk + warpId * kTopk;
  }

  EpMultiWarpIter mwIter(globalWarpNum, args.numTokens, kHidden);

  for (int i = globalWarpId; i < args.numTokens * mwIter.warpsPerItem; i += globalWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenOffset, hiddenSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenOffset, hiddenSize);

    for (int j = laneId; j < kTopk; j += kCfg.waveSize) {
      const int flat = args.dispDestTokIdMap[tokenId * kTopk + j];
      const int destPe = EpPeFromFlat<kCfg>(flat);

      if (destPe < kNpes) {
        const int destLocalTokId = EpLocalTokFromFlat<kCfg>(flat);
        srcPtrs[j] =
            EpPeer<T>(win, destPe, args.offOutTok) + destLocalTokId * kHidden + hiddenOffset;
        if constexpr (kCfg.useWeights) {
          srcWeightPtrs[j] = EpPeer<float>(win, destPe, args.offOutWts) + destLocalTokId * kTopk;
        }
      } else {
        srcPtrs[j] = nullptr;
        if constexpr (kCfg.useWeights) srcWeightPtrs[j] = nullptr;
      }
    }

    T* outPtr = reinterpret_cast<T*>(args.outTokenBuf) + tokenId * kHidden + hiddenOffset;

    // WarpAccum stops at the first null, so dedup/drop holes have to be
    // compacted to the front first. Only worth the ballot at small worldSize,
    // where holes are common (v1 uses the same <= 4 cutoff).
    int validAccumCount = kTopk;
    if constexpr (kNpes <= 4) {
      int isValid = 0;
      T* myTokPtr = nullptr;
      if (laneId < kTopk) {
        myTokPtr = srcPtrs[laneId];
        isValid = (myTokPtr != nullptr) ? 1 : 0;
      }
      const unsigned long long validMask = __ballot(isValid);
      validAccumCount = __popcll(validMask);
      if (validAccumCount < kTopk && isValid) {
        const int myPos = __popcll(validMask & ((1ULL << laneId) - 1));
        srcPtrs[myPos] = myTokPtr;
      }
    }

    core::WarpAccum<T, 4>(outPtr, srcPtrs, nullptr, validAccumCount, hiddenSize);

    if constexpr (kCfg.useWeights) {
      if (args.outWeightsBuf && inTokenPartId == mwIter.warpsPerItem - 1) {
        core::WarpAccum<float, 4>(args.outWeightsBuf + tokenId * kTopk, srcWeightPtrs, nullptr,
                                  kTopk, kTopk);
      }
    }
  }
}

/* ------------------------------------------------------------------------- */
/*                  Scatter combine, quantized or not                         */
/* ------------------------------------------------------------------------- */
// 16 B of T (eight bf16 or four f32) as f32, and back.
template <typename T>
__device__ __forceinline__ void EpVecToF32(uint4 v, float (&f)[16 / sizeof(T)]) {
  if constexpr (sizeof(T) == 2) {
    EpBf16x8ToF32(v, f);
  } else {
    f[0] = __uint_as_float(v.x);
    f[1] = __uint_as_float(v.y);
    f[2] = __uint_as_float(v.z);
    f[3] = __uint_as_float(v.w);
  }
}

template <typename T>
__device__ __forceinline__ uint4 EpF32ToVec(const float (&f)[16 / sizeof(T)]) {
  if constexpr (sizeof(T) == 2) {
    unsigned pk[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const unsigned lo = __builtin_bit_cast(unsigned short, T(f[2 * i]));
      const unsigned hi = __builtin_bit_cast(unsigned short, T(f[2 * i + 1]));
      pk[i] = lo | (hi << 16);
    }
    return make_uint4(pk[0], pk[1], pk[2], pk[3]);
  } else {
    return make_uint4(__float_as_uint(f[0]), __float_as_uint(f[1]), __float_as_uint(f[2]),
                      __float_as_uint(f[3]));
  }
}

// What a scatter lane puts in its landing row for kVec elements, and reads back:
//   Put  stage 1, called by every lane (the quantized codec reduces across a group)
//   Get  stage 3, one row's share for this lane
//   Add  stage 3, that share into the f32 sum
// The quantized codec: bf16 in, fp8 or e2m1 plus a scale per group on the wire.
template <EpCfg kCfg, typename T, bool kPlain = kCfg.combineScatterPlain>
struct EpScatterCodec {
  static_assert(sizeof(T) == 2, "the quantized scatter takes bf16 rows");
#if !defined(__gfx950__)
  static_assert(!kCfg.combineScatterFp4, "the e2m1 scatter combine is gfx950's");
#endif
  static constexpr bool kFp4 = kCfg.combineScatterFp4;
  static constexpr int kGroup = kCfg.combineScatterQGroup;
  static constexpr float kMax = kFp4 ? kEpScatterFp4Max : kEpScatterFp8Max;
  static constexpr int kVec = kEpScatterLaneElems;
  static constexpr int kLanesPerGroup = kGroup / kVec;
  // Wire bytes an element: one for fp8, half for e2m1 (the shift halves an offset).
  static constexpr int kPayShift = kFp4 ? 1 : 0;
  static constexpr int kPayB = EpScatterPayloadBytes(kCfg);
  struct Lane {
    EpScatterWire<kFp4> q;
    float scale;
  };

  __device__ static void Put(char* row, int e, uint4 raw, int laneId, bool active) {
    float f[kVec];
    EpBf16x8ToF32(raw, f);
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kVec; ++i) amax = fmaxf(amax, fabsf(f[i]));
#pragma unroll
    for (int off = 1; off < kLanesPerGroup; off <<= 1) amax = fmaxf(amax, __shfl_xor(amax, off));
    float scale, inv;
    unsigned e8m0;
    EpScatterScale<kGroup, kMax>(amax, scale, inv, e8m0);
    if (!active) return;
    EpScatterPack<kFp4>(row + (e >> kPayShift), f, inv);
    if (laneId % kLanesPerGroup == 0) {
      if constexpr (kGroup == 32) {
        reinterpret_cast<unsigned char*>(row + kPayB)[e / kGroup] = (unsigned char)e8m0;
      } else {
        reinterpret_cast<float*>(row + kPayB)[e / kGroup] = scale;
      }
    }
  }

  __device__ static Lane Get(const char* row, int e) {
    Lane l;
    l.q = *reinterpret_cast<const EpScatterWire<kFp4>*>(row + (e >> kPayShift));
    if constexpr (kGroup == 32) {
      const unsigned b = reinterpret_cast<const unsigned char*>(row + kPayB)[e / kGroup];
      l.scale = __uint_as_float(b << 23);
    } else {
      l.scale = reinterpret_cast<const float*>(row + kPayB)[e / kGroup];
    }
    return l;
  }

  __device__ static void Add(const Lane& l, float (&acc)[kVec]) {
    EpScatterAcc<kFp4>(l.q, l.scale, acc);
  }
};

// The unquantized codec: the T row as it is (bf16 or fp32), 16 B a lane.
template <EpCfg kCfg, typename T>
struct EpScatterCodec<kCfg, T, true> {
  static constexpr int kVec = 16 / sizeof(T);
  using Lane = uint4;

  __device__ static void Put(char* row, int e, uint4 raw, int, bool active) {
    if (active) *reinterpret_cast<uint4*>(row + (size_t)e * sizeof(T)) = raw;
  }

  __device__ static Lane Get(const char* row, int e) {
    return *reinterpret_cast<const uint4*>(row + (size_t)e * sizeof(T));
  }

  __device__ static void Add(const Lane& l, float (&acc)[kVec]) {
    float f[kVec];
    EpVecToF32<T>(l, f);
#pragma unroll
    for (int i = 0; i < kVec; ++i) acc[i] += f[i];
  }
};

// Each computing rank puts its post-expert rows, through the codec, into the origin
// rank's landing row (myPe, origin token); after the cross-device barrier the origin
// rank sums its tokens' rows locally in f32, in top-k order as the gather does.
// Weights travel in the row, so the fold reads them there.
template <EpCfg kCfg, typename T>
__device__ void EpCombineScatterBody(EpArgs args) {
  using Codec = EpScatterCodec<kCfg, T>;
  constexpr int kVec = Codec::kVec;
  constexpr int kNpes = kCfg.worldSize;
  constexpr int kTopk = kCfg.numExpertPerToken;
  constexpr int kHidden = kCfg.hiddenDim;
  constexpr int kM = kCfg.maxTokPerRank;
  constexpr int kChunk = kCfg.waveSize * kVec;
  constexpr int kSteps = (kHidden + kChunk - 1) / kChunk;
  constexpr bool kFull = kHidden % kChunk == 0;
  constexpr int kRowB = EpScatterRowBytes(kCfg);
  // the weights follow the payload and its scales
  constexpr int kWtsOff = EpScatterPayloadBytes(kCfg) + EpScatterScaleBytes(kCfg);
  constexpr size_t kLandOff = EpScatterLandOff(kCfg);

  const int myPe = args.rank;
  const int laneId = threadIdx.x & (kCfg.waveSize - 1);
  const int warpId = threadIdx.x / kCfg.waveSize;
  const int globalWarpId = blockIdx.x * kCfg.warpPerBlock + warpId;
  const int globalWarpNum = gridDim.x * kCfg.warpPerBlock;

  const unsigned long long win = args.window;
  const unsigned long long flag = args.xdbFlag[0];
  const int totalRecv = args.totalRecvTokenNum[0];
  const T* const inp = reinterpret_cast<const T*>(args.inpTokenBuf);
  int* const recvToSrc = EpLocal<int>(win, args.offRecvToSrc);
  float* const recvWts = EpLocal<float>(win, args.offOutWts);

  // Stage 1: one warp per (recv slot, step), kBatch items a pass with every load
  // issued before the first is used.
  constexpr int kBatch = 2;
  const int items = totalRecv * kSteps;
  for (int u0 = globalWarpId; u0 < items; u0 += kBatch * globalWarpNum) {
    int slot[kBatch], origin[kBatch], e[kBatch];
    uint4 raw[kBatch];
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      const int u = min(u0 + b * globalWarpNum, items - 1);
      slot[b] = u / kSteps;
      e[b] = (u % kSteps) * kChunk + laneId * kVec;
      // Written by the senders' dispatch; read at system scope.
      origin[b] =
          __hip_atomic_load(recvToSrc + slot[b], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      raw[b] = make_uint4(0, 0, 0, 0);
      if (kFull || e[b] < kHidden)
        raw[b] = *reinterpret_cast<const uint4*>(inp + (size_t)slot[b] * kHidden + e[b]);
    }
#pragma unroll
    for (int b = 0; b < kBatch; ++b) {
      if (u0 + b * globalWarpNum >= items) break;
      char* const row = EpPeer<char>(win, origin[b] / kM, args.offOutTok) + kLandOff +
                        ((size_t)myPe * kM + origin[b] % kM) * kRowB;
      Codec::Put(row, e[b], raw[b], laneId, kFull || e[b] < kHidden);
      if constexpr (kCfg.useWeights) {
        if (e[b] < kChunk && laneId < kTopk) {
          reinterpret_cast<float*>(row + kWtsOff)[laneId] =
              __hip_atomic_load(recvWts + (size_t)slot[b] * kTopk + laneId, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
    }
  }

  EpCrossDeviceBarrier<kCfg>(args, flag);

  if (args.numTokens == 0) return;

  // Stage 3: sum each token's landed rows, one warp per (token, step); every row's
  // loads go out before the first is summed.
  const char* const land = EpLocal<char>(win, args.offOutTok) + kLandOff;
  T* const out = reinterpret_cast<T*>(args.outTokenBuf);
  for (int u = globalWarpId; u < args.numTokens * kSteps; u += globalWarpNum) {
    const int tok = u / kSteps;
    const int step = u % kSteps;
    const int e = step * kChunk + laneId * kVec;
    const bool active = kFull || e < kHidden;
    int pe[kTopk];
#pragma unroll
    for (int j = 0; j < kTopk; ++j)
      pe[j] = EpPeFromFlat<kCfg>(args.dispDestTokIdMap[tok * kTopk + j]);
    typename Codec::Lane q[kTopk];
    float w[kTopk];
#pragma unroll
    for (int j = 0; j < kTopk; ++j) {
      // An unrouted slot reads PE 0's row and is dropped below.
      const char* const row = land + ((size_t)(pe[j] < kNpes ? pe[j] : 0) * kM + tok) * kRowB;
      q[j] = {};
      w[j] = 0.0f;
      if (active) q[j] = Codec::Get(row, e);
      if constexpr (kCfg.useWeights) {
        if (step == 0 && laneId < kTopk)
          w[j] = reinterpret_cast<const float*>(row + kWtsOff)[laneId];
      }
    }
    float acc[kVec] = {};
    float wacc = 0.0f;
#pragma unroll
    for (int j = 0; j < kTopk; ++j) {
      if (pe[j] >= kNpes) continue;
      Codec::Add(q[j], acc);
      wacc += w[j];
    }
    if (active) *reinterpret_cast<uint4*>(out + (size_t)tok * kHidden + e) = EpF32ToVec<T>(acc);
    if constexpr (kCfg.useWeights) {
      if (args.outWeightsBuf && step == 0 && laneId < kTopk)
        args.outWeightsBuf[tok * kTopk + laneId] = wacc;
    }
  }
}

}  // namespace v2
}  // namespace ops
}  // namespace mori
