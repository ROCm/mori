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
// One routed slot's weights, expert ids, payload and scale row, sent to its
// destination slot. The routing pass and the replay pass share it.
template <EpCfg kCfg, typename T>
__device__ __forceinline__ void EpPublishSlot(EpArgs args, int laneId, int destPe, int destTokId,
                                              int srcTokId) {
  constexpr int kTopk = kCfg.numExpertPerToken;
  constexpr size_t kHidden = static_cast<size_t>(kCfg.hiddenDim);
  const unsigned long long win = args.window;
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
          dstS, reinterpret_cast<const unsigned int*>(args.scalesBuf) + (size_t)srcTokId * kSrcDw,
          kSrcDw);
      for (int e = kSrcDw + laneId; e < kDstDw; e += kCfg.waveSize) dstS[e] = 0u;
    }
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
  if constexpr (kCfg.replay) {
    // Replay: the handle's dest map names every slot, so nothing is claimed.
    if (args.tokenIndices && args.inpTokenBuf) {
      for (int i = globalWarpId; i < args.numTokens * kTopk; i += globalWarpNum) {
        const int flat = args.dispDestTokIdMap[i];
        const int destPe = EpPeFromFlat<kCfg>(flat);
        if (flat < 0 || destPe >= kNpes) continue;
        const int destTokId = EpLocalTokFromFlat<kCfg>(flat);
        const int srcTokId = i / kTopk;
        if (laneId == 0) {
          atomicAdd(args.destPeTokenCounter + destPe, 1);
          EpPeer<int>(win, destPe, args.offRecvToSrc)[destTokId] =
              EpSrcTokIndex<kCfg>(myPe, srcTokId);
        }
        EpPublishSlot<kCfg, T>(args, laneId, destPe, destTokId, srcTokId);
      }
    }
  } else if (args.tokenIndices && args.inpTokenBuf) {
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
        // Under a recv cap a slot past it is dropped: null in the map, nothing sent.
        if (!EpRecvCapped(kCfg) || destTokId < EpMaxRecv(kCfg)) {
          atomicAdd(args.destPeTokenCounter + destPe, 1);
          args.dispDestTokIdMap[i] = EpFlatIndex<kCfg>(destPe, destTokId);
          // Tell the destination which global source token owns that slot, so
          // combine can route the reduction back.
          EpPeer<int>(win, destPe, args.offRecvToSrc)[destTokId] =
              EpSrcTokIndex<kCfg>(myPe, srcTokId);
        } else {
          args.dispDestTokIdMap[i] = kNullFlat;
          destTokId = -1;
        }
      }
      destTokId = __shfl(destTokId, 0);
      if constexpr (EpRecvCapped(kCfg)) {
        if (destTokId < 0) continue;
      }

      EpPublishSlot<kCfg, T>(args, laneId, destPe, destTokId, srcTokId);
    }
  }

  // Drain this block's peer stores before it arrives, as FlyDSL does: the signal
  // fence below covers warp 0's stores only. A drain is enough because the window
  // is uncached (cco's default); a cached window would need a release per block.
  __builtin_amdgcn_s_waitcnt(0);
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
// The arch's own e4m3: fnuz on gfx94x, OCP elsewhere.
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
constexpr float kEpScatterFp8Max = 240.0f;
#else
constexpr float kEpScatterFp8Max = 448.0f;
#endif

// Elements a lane handles per step: one 16 B load of bf16.
constexpr int kEpScatterLaneElems = 8;

__device__ __forceinline__ void EpBf16x8ToF32(uint4 v, float (&f)[8]) {
  const unsigned w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}

// e2m1's largest finite value.
constexpr float kEpScatterFp4Max = 6.0f;

// The group's scale and its reciprocal. 1x32: E8M0, the smallest power of two that
// keeps amax within the element max (RoundUp, as the gfx125x push and aiter pick it).
// 1x128: amax / max exactly, so every block uses the whole element range; a block
// whose amax is below max * 2^-126 keeps scale 1, since its scale would be subnormal
// and max / amax can overflow (0 * inf = NaN), as FlyDSL's _block_scale does.
template <int kGroup, float kMax>
__device__ __forceinline__ void EpScatterScale(float amax, float& scale, float& inv,
                                               unsigned& e8m0) {
  if constexpr (kGroup == 32) {
    const unsigned bits = __float_as_uint(amax * (1.0f / kMax));
    unsigned e = ((bits & 0x7fffffffu) + 0x7fffffu) >> 23;
    e = e < 1u ? 1u : (e > 254u ? 254u : e);
    e8m0 = e;
    scale = __uint_as_float(e << 23);
    inv = __uint_as_float((254u - e) << 23);
  } else {
    const bool tiny = !(amax >= kMax * 0x1p-126f);
    scale = tiny ? 1.0f : amax * (1.0f / kMax);
    inv = tiny ? 1.0f : kMax / amax;
    e8m0 = 0;
  }
}

// Eight elements at 1/inv scale into the wire, and back. fp8: 8 bytes; e2m1: 4 bytes,
// element 2i in the low nibble.
template <bool kFp4>
__device__ __forceinline__ void EpScatterPack(char* dst, const float (&f)[8], float inv) {
  if constexpr (kFp4) {
#if defined(__gfx950__)
    float q[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      q[i] = __builtin_amdgcn_fmed3f(f[i] * inv, kEpScatterFp4Max, -kEpScatterFp4Max);
    unsigned u = 0;
    u = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u, q[0], q[1], 1.0f, 0);
    u = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u, q[2], q[3], 1.0f, 1);
    u = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u, q[4], q[5], 1.0f, 2);
    u = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u, q[6], q[7], 1.0f, 3);
    *reinterpret_cast<unsigned*>(dst) = u;
#endif
  } else {
    float q[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      q[i] = __builtin_amdgcn_fmed3f(f[i] * inv, kEpScatterFp8Max, -kEpScatterFp8Max);
    unsigned lo = __builtin_amdgcn_cvt_pk_fp8_f32(q[0], q[1], 0, false);
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(q[2], q[3], lo, true);
    unsigned hi = __builtin_amdgcn_cvt_pk_fp8_f32(q[4], q[5], 0, false);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(q[6], q[7], hi, true);
    *reinterpret_cast<uint2*>(dst) = make_uint2(lo, hi);
  }
}

// What a lane loads per step: eight fp8 or eight e2m1.
template <bool kFp4>
using EpScatterWire = std::conditional_t<kFp4, unsigned, uint2>;

template <bool kFp4>
__device__ __forceinline__ void EpScatterAcc(EpScatterWire<kFp4> q, float scale, float (&acc)[8]) {
  if constexpr (kFp4) {
#if defined(__gfx950__)
    const auto a = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(q, 1.0f, 0);
    const auto b = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(q, 1.0f, 1);
    const auto c = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(q, 1.0f, 2);
    const auto d = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(q, 1.0f, 3);
    acc[0] += a[0] * scale;
    acc[1] += a[1] * scale;
    acc[2] += b[0] * scale;
    acc[3] += b[1] * scale;
    acc[4] += c[0] * scale;
    acc[5] += c[1] * scale;
    acc[6] += d[0] * scale;
    acc[7] += d[1] * scale;
#endif
  } else {
    const unsigned w[2] = {q.x, q.y};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const auto a = __builtin_amdgcn_cvt_pk_f32_fp8(w[h], false);
      const auto b = __builtin_amdgcn_cvt_pk_f32_fp8(w[h], true);
      acc[4 * h + 0] += a[0] * scale;
      acc[4 * h + 1] += a[1] * scale;
      acc[4 * h + 2] += b[0] * scale;
      acc[4 * h + 3] += b[1] * scale;
    }
  }
}

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
