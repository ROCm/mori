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
// Decoders for the reverse map. The PUSH combine needs both halves: a rank
// holding an expert result reads its own recvToSrc entry to learn which rank
// owns the token and which of that rank's tokens it is, which is exactly the
// (pe, localTok) pair a push slot is addressed by.
template <EpCfg kCfg>
__device__ __forceinline__ int EpPeFromSrcTok(int srcTok) {
  return srcTok / kCfg.maxTokPerRank;
}
template <EpCfg kCfg>
__device__ __forceinline__ int EpTokFromSrcTok(int srcTok) {
  return srcTok % kCfg.maxTokPerRank;
}
// A push slot, in slots (multiply by EpCombinePushSlotBytes). Same shape as the
// reverse map's encoding on purpose: the sender writes (myPe, destination token)
// and the owner reads (source pe, its own token), so one formula serves both and
// there is no second stride to keep in step. v1 spells this SendBufSlotOffset.
template <EpCfg kCfg>
__device__ __forceinline__ int EpPushSlot(int pe, int localTokId) {
  return pe * kCfg.maxTokPerRank + localTokId;
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

// CORRECT EITHER WAY. Rounds each warp's slice of the hidden dim up to 128
// elements so the slice OFFSETS stay aligned. Without it the offsets are
// multiples of ceil(hidden / warpsPerItem), which is aligned only when
// warpsPerItem divides the hidden dim -- and when it does not, the combine
// reduce's own alignment test fails and every element goes through its scalar
// tail. Measured on gfx1250, EP4, hidden 7168, push combine at the shipped
// 64x16 (so 1024 warps), 3 alternating rounds, CHECK=1:
//
// Combine us, push fp8 with the gate off and on, pull bf16 for scale:
//
//   ct     pull   off     on
//   256    33.7   34.6   34.6    warpsPerItem 4, slice 1792, already aligned
//   342    48.1   68.3   41.0    warpsPerItem 3, slice 2390 -> 2432
//   384    48.6   74.0   44.4
//   448    58.2   77.5   47.7
//   511    49.6   79.6   49.8
//   512    45.8   46.7   46.8    warpsPerItem 2, slice 3584, already aligned
//
// Nothing from 576 to 4096 moves by more than 0.1. warpsPerItem is 3 for every
// ct in [342, 511] and no vector width divides 2390, so two of every three
// warps ran the scalar tail and the band cost up to 30 us more than ct=512 does
// with more work to do. Pull does not show it: at 64x8 it has 512 warps, so
// warpsPerItem is 2 across that band and 3584 is aligned. See §23.15n.
//
// 0 turns it off. 1 means the default 128. Any larger value is used as the
// alignment itself, which is how the gate gets checked: at hidden 7168 and
// warpsPerItem 3, an alignment of 3584 leaves the third part with no work at
// all, so a run that sets it and does not move is a run where the flag never
// reached the compile -- and that has to be ruled out before a null result
// here is read as "the offsets were not the problem".
#ifndef MORI_COMB_PARTALIGN
#define MORI_COMB_PARTALIGN 1
#endif

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
    // 128 elements, not 128 bytes: this struct does not know the element type,
    // and 128 elements covers the widest requirement any caller has (a 128 B
    // TDM row of 1-byte wire) while staying a multiple of every narrower one.
    // Guarded on the dim being large enough that every part still gets work --
    // rounding up is safe for correctness either way, since Decode already
    // hands back a zero-length chunk for a part that starts past the end, but
    // below this bound it would idle warps for nothing.
    constexpr size_t kPartAlign = (MORI_COMB_PARTALIGN > 1) ? (size_t)MORI_COMB_PARTALIGN : 128;
    if (MORI_COMB_PARTALIGN != 0 && dimSize >= kPartAlign * (size_t)warpsPerItem)
      dimPerWarp = (dimPerWarp + kPartAlign - 1) & ~(kPartAlign - 1);
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
  if (thdId == 0) atomicAdd(args.gridBarrier, 1u);

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
    // Wide path: peers span multiple warps — single-thread wait+reset avoids the
    // race where warp 0 resets the barrier before warp 1 reads gridDim.x.
    if (thdId == 0) {
      EpWaitEq(args.gridBarrier, static_cast<unsigned int>(gridDim.x));
      __hip_atomic_store(args.gridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
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

  // Same expression the slot stride is built from, so the weights sit where the
  // other side looks for them.
  constexpr size_t kHiddenB = (size_t)EpCombineWireBytes(kCfg);
  constexpr int kSlotB = EpCombinePushSlotBytes(kCfg);
  // Wire quant lives in the gfx1250 body, where the narrow can happen inside the
  // TDM tile and pay for itself. Implementing it here too would mean shipping a
  // second version that no machine on hand can check, so hip_backend does not
  // offer quant off gfx1250 and this catches it if that ever drifts.
  static_assert(kCfg.combineQuant == 0,
                "combineQuant is implemented in the gfx125x combine body only");

  if constexpr (kCfg.combinePush) {
    // PUSH: send each expert result straight into the owner's slot, so the
    // owner's reduce reads only its own memory. No TDM here -- on these parts a
    // peer vector store is not the slow path TDM exists for, which is the same
    // reason dispatch above copies straight to the peer per token.
    //
    // recvToSrc[slot] is (owner rank, that rank's token id), which is exactly
    // the pair a push slot is keyed by. The weights come from the LOCAL
    // forwarded copy: offOutWts is indexed by recv slot, the same index this
    // loop walks.
    const int* const recvToSrc = EpLocal<int>(win, args.offRecvToSrc);
    const T* const myToks = reinterpret_cast<const T*>(args.inpTokenBuf);
    const float* const myWts = kCfg.useWeights ? EpLocal<float>(win, args.offOutWts) : nullptr;
    for (int i = globalWarpId; i < totalRecv; i += globalWarpNum) {
      const int srcTok = recvToSrc[i];
      const int destPe = EpPeFromSrcTok<kCfg>(srcTok);
      const int destLocalTok = EpTokFromSrcTok<kCfg>(srcTok);
      // Holds for any slot dispatch wrote. Checked because the arena starts
      // zeroed, so a stale slot would decode to (rank 0, token 0) and corrupt a
      // live token instead of faulting.
      if (destPe >= kNpes || destLocalTok >= kCfg.maxTokPerRank) continue;
      uint8_t* const dstSlot = EpPeer<uint8_t>(win, destPe, args.offCombPush) +
                               (size_t)EpPushSlot<kCfg>(args.rank, destLocalTok) * kSlotB;
      core::WarpCopy(reinterpret_cast<T*>(dstSlot), myToks + (size_t)i * kHidden, kHidden);
      if constexpr (kCfg.useWeights) {
        if (myWts) {
          core::WarpCopy(reinterpret_cast<float*>(dstSlot + kHiddenB), myWts + (size_t)i * kTopk,
                         kTopk);
        }
      }
    }
  } else {
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
  }

  EpCrossDeviceBarrier<kCfg>(args, flag);

  *args.totalRecvTokenNum = 0;
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
        if constexpr (kCfg.combinePush) {
          // All local: destPe wrote its result for our token here before the
          // barrier, keyed by (source rank, our token) -- the mirror of what the
          // sender wrote, (its own rank, our token id on us).
          uint8_t* const slot = EpLocal<uint8_t>(win, args.offCombPush) +
                                (size_t)EpPushSlot<kCfg>(destPe, tokenId) * kSlotB;
          srcPtrs[j] = reinterpret_cast<T*>(slot) + hiddenOffset;
          if constexpr (kCfg.useWeights) {
            srcWeightPtrs[j] = reinterpret_cast<float*>(slot + kHiddenB);
          }
        } else {
          const int destLocalTokId = EpLocalTokFromFlat<kCfg>(flat);
          srcPtrs[j] =
              EpPeer<T>(win, destPe, args.offOutTok) + destLocalTokId * kHidden + hiddenOffset;
          if constexpr (kCfg.useWeights) {
            srcWeightPtrs[j] = EpPeer<float>(win, destPe, args.offOutWts) + destLocalTokId * kTopk;
          }
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
