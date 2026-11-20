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

#include "mori/cco/cco.hpp"
#include "mori/core/transport/p2p/device_primitives.hpp"
#include "mori/ops/dispatch_combine_v2/ep_cfg.hpp"

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

// (pe, localTokId) <-> flat index. The stride must be at least the recv-slot
// capacity, NOT the per-rank send capacity: the local id being encoded is a slot
// index handed out by the destination's tokOff counter, which runs to EpMaxRecv.
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
// Dispatch's tail, shared by both bodies: every block arrives at the grid barrier,
// then one warp announces this rank's per-destination counts and reads back everyone
// else's.
template <EpCfg kCfg>
__device__ __forceinline__ void EpDispatchNotify(EpArgs args) {
  constexpr int kNpes = kCfg.worldSize;
  const int myPe = args.rank;
  const int thdId = threadIdx.x;
  const int laneId = threadIdx.x & (kCfg.waveSize - 1);
  const int globalWarpId = blockIdx.x * kCfg.warpPerBlock + thdId / kCfg.waveSize;
  const unsigned long long win = args.window;

  __syncthreads();
  if (thdId == 0) atomicAdd(args.gridBarrier, 1u);

  // Phase 2: one warp announces this rank's per-destination counts, then reads
  // back everyone else's. The grid barrier is hoisted before the peer loop so
  // that wide EP (worldSize > waveSize) multi-iterates safely.
  if (globalWarpId == 0) {
    EpWaitEq(args.gridBarrier, static_cast<unsigned int>(gridDim.x));
    __hip_atomic_store(args.gridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    for (int destPe = laneId; destPe < kNpes; destPe += kCfg.waveSize) {
      // +1 so a zero-token destination still sees a distinct "signal arrived".
      const int numTokenSignal = __hip_atomic_load(args.destPeTokenCounter + destPe,
                                                   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) +
                                 1;
      int* signal = EpPeer<int>(win, destPe, args.offRecvNum) + myPe;
      EpWaitEq(signal, 0);
      __threadfence_system();
      __hip_atomic_store(signal, numTokenSignal, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }

    int* recvTokenNums = EpLocal<int>(win, args.offRecvNum);
    for (int srcPe = laneId; srcPe < kNpes; srcPe += kCfg.waveSize) {
      int* signal = recvTokenNums + srcPe;
      const int recvTokenNum = EpWaitGt(signal, 0) - 1;
      __hip_atomic_store(signal, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);
      args.destPeTokenCounter[srcPe] = 0;
    }

    if (laneId == 0) EpLocal<int>(win, args.offTokOff)[0] = 0;
  }
}

template <EpCfg kCfg, typename T>
__device__ void EpDispatchBody(EpArgs args) {
  constexpr int kNpes = kCfg.worldSize;
  const int myPe = args.rank;
  constexpr int kTopk = kCfg.numExpertPerToken;
  constexpr int kExpertPerRank = kCfg.numExpertPerRank;
  constexpr size_t kHidden = static_cast<size_t>(kCfg.hiddenDim);
  const int kNullFlat = EpNullFlat<kCfg>();

  const int thdId = threadIdx.x;
  const int laneId = threadIdx.x & (kCfg.waveSize - 1);
  const int warpId = thdId / kCfg.waveSize;
  const int warpNum = kCfg.warpPerBlock;
  const int globalWarpId = blockIdx.x * warpNum + warpId;
  const int globalWarpNum = gridDim.x * warpNum;

  const unsigned long long win = args.window;

  // Reset the count Phase 2 accumulates into. Ordered before those adds by the
  // __syncthreads() below (same warp); combine only reads it.
  if (globalWarpId == 0 && laneId == 0) *args.totalRecvTokenNum = 0;

  // Phase 1: route and send. One warp per (token, top-k slot).
  if (args.tokenIndices && args.inpTokenBuf) {
    for (int i = globalWarpId; i < args.numTokens * kTopk; i += globalWarpNum) {
      const int srcTokId = i / kTopk;
      int destPe;
      int destTokId = 0;

      const int destExpert = args.tokenIndices[i];
      // A negative expert id is the caller's "drop this slot" sentinel; the
      // null flat index makes combine treat the slot as a nullptr source.
      if (destExpert < 0) {
        if (laneId == 0) args.dispDestTokIdMap[i] = kNullFlat;
        continue;
      }
      destPe = destExpert / kExpertPerRank;
      // Out-of-range expert ids would index a peer pointer out of bounds. Drop
      // through the same sentinel; the whole warp skips coherently because
      // destPe is warp-uniform here.
      if (destPe < 0 || destPe >= kNpes) {
        if (laneId == 0) args.dispDestTokIdMap[i] = kNullFlat;
        continue;
      }

      // Dedup: if an earlier top-k slot of this token already targets destPe,
      // the payload is already on its way.
      int condition = 0;
      if (laneId < (i % kTopk)) {
        const int otherExpert = args.tokenIndices[srcTokId * kTopk + laneId];
        condition = (otherExpert >= 0) && (destPe == (otherExpert / kExpertPerRank));
      }
      if (__any(condition)) {
        if (laneId == 0) args.dispDestTokIdMap[i] = kNullFlat;
        continue;
      }

      if (laneId == 0) {
        // Claim a slot in the destination's recv buffer. Remote atomic on the
        // peer's copy of tokOff; SYSTEM scope because it crosses devices.
        destTokId = __hip_atomic_fetch_add(EpPeer<int>(win, destPe, args.offTokOff), 1,
                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        atomicAdd(args.destPeTokenCounter + destPe, 1);
        args.dispDestTokIdMap[i] = EpFlatIndex<kCfg>(destPe, destTokId);
        // Tell the destination which global source token owns that slot, so
        // combine can route the reduction back.
        EpPeer<int>(win, destPe, args.offRecvToSrc)[destTokId] =
            EpSrcTokIndex<kCfg>(myPe, srcTokId);
      }
      destTokId = __shfl(destTokId, 0);

      if (laneId < kTopk) {
        if constexpr (kCfg.useWeights) {
          if (args.weightsBuf) {
            EpPeer<float>(win, destPe, args.offOutWts)[destTokId * kTopk + laneId] =
                args.weightsBuf[srcTokId * kTopk + laneId];
          }
        }
        EpPeer<int>(win, destPe, args.offOutIdx)[destTokId * kTopk + laneId] =
            args.tokenIndices[srcTokId * kTopk + laneId];
      }

      core::WarpCopy(EpPeer<T>(win, destPe, args.offDispOut) + destTokId * kHidden,
                     reinterpret_cast<const T*>(args.inpTokenBuf) + srcTokId * kHidden, kHidden);
      // The scale row follows its token to the same slot. No staging detour like the
      // gfx1250 body needs: this body already copies straight to the peer per token,
      // and on these parts a peer vector store is not the slow path TDM exists for.
      // Laid out at EpScaleStride even here, where the alignment buys nothing:
      // one layout per consumer, whatever arch produced it.
      if constexpr (kCfg.scaleBytes > 0) {
        constexpr int kSrcDw = kCfg.scaleBytes / 4;
        constexpr int kDstDw = EpScaleStride(kCfg) / 4;
        if (args.scalesBuf) {
          unsigned int* dstS =
              EpPeer<unsigned int>(win, destPe, args.offOutScales) + (size_t)destTokId * kDstDw;
          core::WarpCopy(
              dstS,
              reinterpret_cast<const unsigned int*>(args.scalesBuf) + (size_t)srcTokId * kSrcDw,
              kSrcDw);
          for (int e = kSrcDw + laneId; e < kDstDw; e += kCfg.waveSize) dstS[e] = 0u;
        }
      }
    }
  }

  EpDispatchNotify<kCfg>(args);
}

/* ------------------------------------------------------------------------- */
/*                         LL dispatch (Cfg.llDispatch)                       */
/* ------------------------------------------------------------------------- */
// Ported from EP v1's EpDispatchIntraNodeLLKernel_body (intranode_ll.hpp). One block
// owns one token at a time and its warps split by role:
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
// constants, so EpLl* fix how many rows each warp takes and the compiler unrolls the
// row loop to exactly that.

template <int kMaxDests>
struct EpLlRoute {
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
__device__ __forceinline__ void EpLlComputeRoute(const EpArgs& args, int tokenId,
                                                 EpLlRoute<EpLlMaxDests(kCfg)>& route,
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
  }

  const unsigned long long active = __ballot(didAlloc);
  if (didAlloc) {
    const int pos = __popcll(active & ((1ull << laneId) - 1));
    route.destRank[pos] = laneId;
    route.recvSlot[pos] = mySlot;
  }
  if (laneId == 0) route.numDests = __popcll(active);
}

// Deferred past the block barrier so the copy warps are not held up by it.
template <EpCfg kCfg>
__device__ __forceinline__ void EpLlWriteMap(const EpArgs& args, int tokenId, int didAlloc,
                                             int slotForMe, int mySlot) {
  constexpr int kTopk = kCfg.numExpertPerToken;
  const int laneId = threadIdx.x & (kCfg.waveSize - 1);

  if (laneId < kTopk) args.dispDestTokIdMap[tokenId * kTopk + laneId] = EpNullFlat<kCfg>();
  if (didAlloc) {
    atomicAdd(args.destPeTokenCounter + laneId, 1);
    EpPeer<int>(args.window, laneId, args.offRecvToSrc)[mySlot] =
        EpSrcTokIndex<kCfg>(args.rank, tokenId);
    args.dispDestTokIdMap[tokenId * kTopk + slotForMe] = EpFlatIndex<kCfg>(laneId, mySlot);
  }
}

template <EpCfg kCfg, typename T>
__device__ void EpDispatchLLBody(EpArgs args) {
  constexpr int kTopk = kCfg.numExpertPerToken;
  constexpr int kCtrl = EpLlCtrlWarps(kCfg);
  constexpr int kCopyWarps = EpLlCopyWarps(kCfg);
  constexpr int kRows = EpLlRows(kCfg);
  constexpr int kMaxRows = EpLlMaxRowCount(kCfg);
  constexpr bool kEvenRows = (kRows % kCopyWarps) == 0;
  constexpr bool kPartialRow = EpLlLastRowVecs(kCfg) < kCfg.waveSize;
  constexpr size_t kRowBytes = EpLlRowBytes(kCfg);
  constexpr size_t kTokBytes = EpTokenBytes(kCfg);
  constexpr int kChunk = kMaxRows < 4 ? kMaxRows : 4;
  // No warp has a short chunk: every row of every chunk exists.
  constexpr bool kRowsAlwaysLive = kEvenRows && (kMaxRows % kChunk) == 0;

  const int thdId = threadIdx.x;
  const int laneId = thdId & (kCfg.waveSize - 1);
  const int warpId = thdId / kCfg.waveSize;
  const int globalWarpId = blockIdx.x * kCfg.warpPerBlock + warpId;
  const unsigned long long win = args.window;
  const int numTokens = args.numTokens;

  using Vec = typename core::VecTypeSelector<16>::dataType;

  __shared__ EpLlRoute<EpLlMaxDests(kCfg)> routeBuf[2];

  // Reset the count the notify step accumulates into; same warp, so ordered before it.
  if (globalWarpId == 0 && laneId == 0) *args.totalRecvTokenNum = 0;

  const int copyWarp = warpId - kCtrl;
  const int rowCount = kEvenRows ? kRows / kCopyWarps : EpLlRowCount(kCfg, copyWarp);
  const int rowBegin = EpLlRowBegin(kCfg, copyWarp);
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
      EpLlComputeRoute<kCfg>(args, blockIdx.x, routeBuf[0], nextDid, nextSlotK, nextSlot);
    } else if (warpId >= kCtrl && blockIdx.x < numTokens && rowCount > 0) {
      const uint8_t* src = reinterpret_cast<const uint8_t*>(args.inpTokenBuf) +
                           static_cast<size_t>(blockIdx.x) * kTokBytes +
                           static_cast<size_t>(rowBegin) * kRowBytes;
#pragma unroll
      for (int u = 0; u < kChunk; ++u) {
        bool live = kRowsAlwaysLive || u < rowCount;
        if constexpr (kPartialRow)
          live = live && !(rowBegin + u == kRows - 1 && laneId >= EpLlLastRowVecs(kCfg));
        if (live) pre[u] = core::load<16>(src + u * kRowBytes + laneOff);
      }
    }
    __syncthreads();
    if (warpId == 0 && blockIdx.x < numTokens) {
      EpLlWriteMap<kCfg>(args, blockIdx.x, nextDid, nextSlotK, nextSlot);
    }

    for (int tokenIdx = blockIdx.x; tokenIdx < numTokens; tokenIdx += gridDim.x) {
      const int pingPong = (tokenIdx / gridDim.x) & 1;
      const auto& route = routeBuf[pingPong];

      if (warpId == 0) {
        const int nextTok = tokenIdx + gridDim.x;
        if (nextTok < numTokens) {
          EpLlComputeRoute<kCfg>(args, nextTok, routeBuf[1 - pingPong], nextDid, nextSlotK,
                                 nextSlot);
          EpLlWriteMap<kCfg>(args, nextTok, nextDid, nextSlotK, nextSlot);
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
                live = live && !(rowBegin + row == kRows - 1 && laneId >= EpLlLastRowVecs(kCfg));
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
                  live = live && !(rowBegin + row == kRows - 1 && laneId >= EpLlLastRowVecs(kCfg));
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

}  // namespace v2
}  // namespace ops
}  // namespace mori
