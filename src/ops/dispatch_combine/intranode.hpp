// Copyright 뿯½ Advanced Micro Devices, Inc. All rights reserved.
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
#pragma once

#include <type_traits>

#include "mori/core/core.hpp"
#include "mori/core/profiler/constants.hpp"
#include "mori/core/profiler/kernel_profiler.hpp"
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "mori/shmem/shmem.hpp"
#include "src/ops/dispatch_combine/common.hpp"
#include "src/ops/dispatch_combine/convert.hpp"
#ifdef ENABLE_PROFILER
#include "mori/profiler/profiler.hpp"
#endif
#if defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
#include <hip/amd_detail/amd_gfx1250_TDM.h>
// Experimental: send the dispatch token payload cross-card via the gfx1250 TDM
// (tensor global<->LDS DMA) engine instead of core::WarpCopy. Single-buffer: one
// LDS tile per warp. Per committed token the warp issues an async LOAD src->tile,
// runs the remote slot atomic + metadata (overlapping the load), waits the load,
// issues the STORE tile->peer, and waits the store (frees the tile). gfx1250 has
// 320KB LDS/CU so a 14KB bf16 tile keeps ~22 warps/CU resident -> ~22-way TDM
// in-flight per CU hides each warp's store drain. Wave-scoped only (no block
// barrier). TdmShape/TdmIssueLoad/TdmIssueStore are the descriptor primitives.
namespace mori {
namespace moe {
// Fill a GROUP1 (shape) descriptor for a 1D hiddenDim-element token payload.
template <typename T>
__device__ __forceinline__ gfx1250_TDM_GROUP1 TdmShape(int hiddenDim) {
  gfx1250_TDM_GROUP1 g1;
  g1.dataSize(sizeof(T) == 2 ? 1 : (sizeof(T) == 4 ? 2 : 0));
  g1.tensorDim0(hiddenDim); g1.tensorDim1(1);
  g1.tensorDim0Stride(hiddenDim); g1.tensorDim1Stride(1);
  g1.tileDim0(hiddenDim); g1.tileDim1(1);
  return g1;
}
// Issue an async TDM load global->LDS (does NOT wait for completion).
template <typename T>
__device__ __forceinline__ void TdmIssueLoad(T* ldsTile, const T* src, const gfx1250_TDM_GROUP1& g1) {
  typedef int _tdm_v4i __attribute__((ext_vector_type(4)));
  typedef int _tdm_v8i __attribute__((ext_vector_type(8)));
  gfx1250_TDM_GROUP0 g0; g0.ldsAddr((uintptr_t)ldsTile); g0.globalAddr((uintptr_t)src);
  _tdm_v4i z4{0, 0, 0, 0}; _tdm_v8i z8{0, 0, 0, 0, 0, 0, 0, 0};
  __builtin_amdgcn_tensor_load_to_lds(g0.m_bitfield, g1.m_bitfield, z4, z4, z8, 0);
}
// Issue an async TDM store LDS->global (does NOT wait for completion).
template <typename T>
__device__ __forceinline__ void TdmIssueStore(T* dst, T* ldsTile, const gfx1250_TDM_GROUP1& g1) {
  typedef int _tdm_v4i __attribute__((ext_vector_type(4)));
  typedef int _tdm_v8i __attribute__((ext_vector_type(8)));
  gfx1250_TDM_GROUP0 g0; g0.ldsAddr((uintptr_t)ldsTile); g0.globalAddr((uintptr_t)dst);
  _tdm_v4i z4{0, 0, 0, 0}; _tdm_v8i z8{0, 0, 0, 0, 0, 0, 0, 0};
  __builtin_amdgcn_tensor_store_from_lds(g0.m_bitfield, g1.m_bitfield, z4, z4, z8, 0);
}
}  // namespace moe
}  // namespace mori
#endif

namespace mori {
namespace moe {

#define MAX_GPUS_PER_NODE 8

/* ---------------------------------------------------------------------------------------------- */
/*                                          BarrierKernel                                         */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ void CrossDeviceBarrierIntraNodeKernel(EpDispatchCombineArgs<T> args,
                                                         const uint64_t crossDeviceBarrierFlag) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;

  __syncthreads();
  if (thdId == 0) atomicAdd(args.combineGridBarrier, 1);

  if (globalThdId < args.config.worldSize) {
    // Set remote flag after all copies are done
    shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, gridDim.x);
    __hip_atomic_store(args.combineGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    __threadfence_system();
    core::AtomicStoreRelaxedSystem(
        args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>(globalThdId) + args.config.rank,
        crossDeviceBarrierFlag);
  }
#if defined(MORI_CNT_STEP) && (MORI_CNT_STEP == 13)
  // DBG: return after WAIT A (all local combine blocks arrived) + peer signal,
  // BEFORE WAIT B (cross-device peer wait). All ranks do this symmetrically, so
  // no rank waits on a peer. Hang here => WAIT A (local grid barrier) is the
  // culprit; clean here => hang is in WAIT B (cross-device).
  return;
#endif

  if (globalThdId == 0) atomicAdd(args.crossDeviceBarrierFlag, 1);

  uint64_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>();
#if defined(MORI_CNT_STEP) && (MORI_CNT_STEP == 14)
  // DBG: bounded WAIT B. If it can't complete, print which peer's flag is missing
  // (rank, waited-peer, value seen vs wanted) and BREAK -> no permanent hang, so
  // the printf flushes. Direct evidence of WAIT B being the hang + the reason.
  if (thdId < args.config.worldSize) {
    unsigned long long _spin = 0;
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != crossDeviceBarrierFlag) {
      if (++_spin > 40000000ull) {
        printf("[WAITB-STUCK] rank=%d waitPeer=%d saw=%llu want=%llu\n", args.config.rank, thdId,
               (unsigned long long)core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId),
               (unsigned long long)crossDeviceBarrierFlag);
        break;
      }
    }
  }
#else
  if (thdId < args.config.worldSize) {
    // Backoff in the cross-device wait: the empty tight spin livelocks the cco/xGMI
    // fabric under CNT2's timing and never re-observes the peer's flag write ->
    // combine hangs (plain's slower timing happens to dodge it). s_sleep throttles
    // the poll (matches GridBarrier's spin) and lets the peer flag become visible.
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != crossDeviceBarrierFlag) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
#endif
  __syncthreads();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpDispatchIntraNodeKernel                                   */
/* ---------------------------------------------------------------------------------------------- */

template <typename T, bool EnableStdMoE = false>
__device__ void EpDispatchIntraNodeKernel_body(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;

  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;

  int myPe = config.rank;
  int npes = config.worldSize;
  size_t hiddenDim = config.HiddenDimSz();

#if defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
  // Single-buffer TDM token payload. Per-warp ONE LDS tile (dynamic shared, sized
  // warpNum*hiddenDim*sizeof(T) by LaunchDispatch). gfx1250 has 320KB LDS/CU, so a
  // 14KB bf16 tile (hidden=7168) lets ~22 warps/CU stay resident -> ~22-way TDM
  // in-flight per CU hides the store's remote drain via warp-level parallelism.
  // Per committed token the warp does: issue async LOAD src->tile, run the remote
  // slot atomic + metadata (overlapping the load), wait the load, issue STORE
  // tile->peer, wait the store (releases the tile for the next token).
  const gfx1250_TDM_GROUP1 _tdmG1 = TdmShape<T>(static_cast<int>(hiddenDim));
  extern __shared__ char _tdmDispSmem[];
  T* _tdmTile = reinterpret_cast<T*>(_tdmDispSmem) + (size_t)warpId * hiddenDim;
  const T* _tdmSrc = nullptr;   // src of the token whose load is in flight this iter
#endif

#if defined(MORI_DISP_NOTIFY)
  // ===================== NOTIFY slot pre-assignment (signal-based) =============
  // Two passes with intra-GPU grid barriers, using DIRECTIONAL cross-device
  // signals (no symmetric cross-device barrier) to keep the fixed sync overhead
  // low (that overhead, not the per-token atomic, was the reason the earlier
  // barrier-based NOTIFY was slow):
  //  (A) count committed tokens per dest PE into localCnt;
  //  (B) exchange the count matrix via parity-encoded per-cell signals, then
  //      compute each source rank's contiguous slot base (prefix sum);
  //  (C) send payload with a LOCAL (agent-scope) slot atomic inside the region;
  //  (D) signal-based completion (RELEASE fence + per-destPe signal; receiver
  //      ACQUIRE + poll) so combine observes every peer's writes.
  // All blocks are co-resident at the tuned geometry (block_num <= CU), which the
  // GridBarrier and the parity/signal protocols rely on.
  {
    const int globalThdId = blockIdx.x * blockDim.x + thdId;
    const int totalThreads = gridDim.x * blockDim.x;
    index_t* localCnt = args.dispTokOffsetMemObj->template GetAs<index_t*>();     // [npes]
    index_t* baseArr = args.destPeTokenCounter;                                   // [npes] (reused)
    index_t* matLocal = args.dispCountMatrixMemObj->template GetAs<index_t*>();   // [npes*npes]
    // parity flips every dispatch (combine advances crossDeviceBarrierFlag by 1
    // per round) and is identical across ranks, so it distinguishes this launch's
    // count-matrix cells from the previous launch's without any reset.
    const index_t parity = (index_t)(args.crossDeviceBarrierFlag[0] & 1ull);

    // Each of the 3 consecutive intra-GPU GridBarriers below uses a DISTINCT word
    // in dispatchGridBarrier[] (+0 here, +2 after Pass A, +3 after prefix; +1 is
    // the completion counter). GridBarrier self-resets its word via a winner CAS
    // (bar==gridDim.x ? 0), which is NOT safe to reuse back-to-back: with tiny
    // inputs each pass is near-instant, so a fast block laps into the next barrier
    // and atomicAdd's the SAME word between the resetting block's two CAS attempts
    // -> that block never sees gridDim.x again -> spins forever -> global hang
    // (same root cause as the completion fix in commit 521e11b6; 4K hides it
    // because the passes are slow enough that no block laps). Separate words break
    // the lap. (worldSize>=4 gives >=4 words; intra-node EP4 is the target.)
    unsigned long long _pt[9]; const bool _pw = (blockIdx.x == 0 && thdId == 0);
    if (_pw) _pt[0] = clock64();
    for (int d = globalThdId; d < npes; d += totalThreads) localCnt[d] = 0;
    GridBarrier(args.dispatchGridBarrier + 0);
    if (_pw) _pt[1] = clock64();
#if defined(MORI_CNT_STEP) && (MORI_CNT_STEP == 1)
    return;  // DBG stop-stage 1: after barrier1 (dispatch-only bisect)
#endif

    // ---- Pass A: count committed (token -> destPe) ----
#if defined(MORI_DISP_NOTIFY_CNT2)
    // Warp-per-TOKEN count with BLOCK-LEVEL LDS aggregation (validated standalone in
    // tools/count_repro.cc KIND=3). Per warp handles ONE token: lane l reads expert
    // slot l, computes destPe; dedup per (token,destPe) via ONE __match_any_sync
    // instead of the topk-iteration __shfl loop. The match_any MUST be called
    // UNIFORMLY by all lanes (divergent warp-sync builtins fault gfx1250 with HSA
    // 0x1016), so inactive lanes pass a sentinel that can't equal any real destPe.
    // Kept lanes bump a per-BLOCK LDS histogram; each block then flushes only `npes`
    // atomics into the uncached symmetric localCnt (was O(committed tokens)).
    {
      constexpr int kCnt2MaxNpes = 64;  // intra-node world size fits easily
      __shared__ index_t smemCnt[kCnt2MaxNpes];
      for (int p = thdId; p < npes; p += blockDim.x) smemCnt[p] = 0;
      __syncthreads();
#if defined(MORI_CNT_STEP) && (MORI_CNT_STEP == 2)
      return;  // DBG stop-stage 2: after smemCnt zero + __syncthreads A (pre count loop)
#endif
      if (args.tokenIndices && args.inpTokenBuf && !args.replayMode && npes <= kCnt2MaxNpes) {
        const int _topk = config.numExpertPerToken;
        for (int tok = globalWarpId; tok < args.curRankNumToken; tok += globalWarpNum) {
          int base = tok * _topk;
          index_t expert = (laneId < _topk) ? args.tokenIndices[base + laneId] : (index_t)-1;
          int destPe = -1;
          if (expert >= 0) {
            index_t dp = expert / config.numExpertPerRank;
            if (dp >= 0 && dp < config.worldSize) destPe = static_cast<int>(dp);
          }
          // Uniform (non-divergent) match_any dedup: ALL lanes call it; inactive
          // lanes use a sentinel (0xFFFFFFFF) distinct from any real destPe. Lowest
          // lane in each destPe group keeps; dropped/dup slots get the null sentinel.
          unsigned mv = (destPe >= 0) ? static_cast<unsigned>(destPe) : 0xFFFFFFFFu;
          unsigned long long grp = __match_any_sync(0xFFFFFFFFFFFFFFFFull, mv);
          int keep = (destPe >= 0 && laneId == (__ffsll((long long)grp) - 1)) ? 1 : 0;
          if (laneId < _topk && !keep)
            args.dispDestTokIdMap[base + laneId] = FlatTokenIndex(config, config.worldSize, 0);
          if (keep) atomicAdd(&smemCnt[destPe], 1);
        }
      }
#if defined(MORI_CNT_STEP) && (MORI_CNT_STEP == 3)
      return;  // DBG stop-stage 3: after count loop (match_any+smemCnt), pre __syncthreads B
#endif
      __syncthreads();
#if defined(MORI_CNT_STEP) && (MORI_CNT_STEP == 4)
      return;  // DBG stop-stage 4: after __syncthreads B, before localCnt flush
#endif
      for (int p = thdId; p < npes; p += blockDim.x) {
        index_t v = smemCnt[p];
        if (v) atomicAdd(&localCnt[p], v);
      }
    }
#else
    if (args.tokenIndices && args.inpTokenBuf && !args.replayMode) {
      for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
           i += globalWarpNum) {
        index_t srcTokId = i / config.numExpertPerToken;
        index_t destExpert = args.tokenIndices[i];
        if (destExpert < 0) {
          if (laneId == 0) args.dispDestTokIdMap[i] = FlatTokenIndex(config, config.worldSize, 0);
          continue;
        }
        index_t destPe = destExpert / config.numExpertPerRank;
        if (destPe < 0 || destPe >= config.worldSize) {
          if (laneId == 0) args.dispDestTokIdMap[i] = FlatTokenIndex(config, config.worldSize, 0);
          continue;
        }
        int condition = 0;
        if (laneId < (i % config.numExpertPerToken)) {
          index_t otherExpert = args.tokenIndices[srcTokId * config.numExpertPerToken + laneId];
          condition = (otherExpert >= 0) && (destPe == (otherExpert / config.numExpertPerRank));
        }
        if (__any(condition)) {
          if (laneId == 0) args.dispDestTokIdMap[i] = FlatTokenIndex(config, config.worldSize, 0);
          continue;
        }
        if (laneId == 0) atomicAdd(&localCnt[destPe], 1);
      }
    }
#endif
#if defined(MORI_CNT_STEP) && (MORI_CNT_STEP == 5)
    return;  // DBG stop-stage 5: after full Pass A (incl localCnt flush)
#endif
    if (_pw) _pt[2] = clock64();
    GridBarrier(args.dispatchGridBarrier + 2);
    if (_pw) _pt[3] = clock64();
#if defined(MORI_CNT_STEP) && (MORI_CNT_STEP == 6)
    return;  // DBG stop-stage 6: after barrier2
#endif

    // ---- Exchange: write my count row into every peer's matrix as a parity-coded
    // signal cell = ((count+1) << 1) | parity (system-scope store). ----
    if (blockIdx.x == 0) {
      // ONE-SHOT, block-wide: map each thread of block0 to a distinct (peer,d)
      // cell so all npes*npes cross-device stores ISSUE in parallel (1 RTT),
      // instead of npes serial stores per lane. Scales to EP64: EP4 = 16 cells
      // (1 round, 16 threads), EP64 = 4096 cells (few rounds across blockDim).
      // block0-only => exactly one writer per cell (no duplicate stores).
      for (int c = thdId; c < npes * npes; c += blockDim.x) {
        int p = c / npes, d = c - p * npes;
        index_t* peerMat = args.dispCountMatrixMemObj->template GetAs<index_t*>(p);
        core::AtomicStoreRelaxedSystem(&peerMat[myPe * npes + d],
                                       (index_t)(((localCnt[d] + 1) << 1) | parity));
      }
    }
    if (_pw) _pt[8] = clock64();
    // ---- Prefix: base[d] = sum_{s<myPe} M[s][d]; total recv = sum_s M[s][myPe].
    // Poll each needed cell until its parity matches this launch, then decode. ----
    if (globalThdId < npes) {
      int d = globalThdId;
      index_t b = 0;
      for (int s = 0; s < myPe; ++s) {
        index_t v;
        while (((v = core::AtomicLoadRelaxedSystem(&matLocal[s * npes + d])) & 1) != parity ||
               v == 0) {
        }
        b += (v >> 1) - 1;
      }
      baseArr[d] = b;
      localCnt[d] = 0;
    }
    if (globalThdId == 0) {
      index_t tot = 0;
      for (int s = 0; s < npes; ++s) {
        index_t v;
        while (((v = core::AtomicLoadRelaxedSystem(&matLocal[s * npes + myPe])) & 1) != parity ||
               v == 0) {
        }
        tot += (v >> 1) - 1;
      }
      *args.totalRecvTokenNum = tot;
    }
#if defined(MORI_CNT_STEP) && (MORI_CNT_STEP == 7)
    return;  // DBG stop-stage 7: after exchange+prefix
#endif
    if (_pw) _pt[4] = clock64();
    GridBarrier(args.dispatchGridBarrier + 3);
    if (_pw) _pt[5] = clock64();
#if defined(MORI_CNT_STEP) && (MORI_CNT_STEP == 8)
    return;  // DBG stop-stage 8: after barrier3
#endif

    // ---- Pass B: send payload; LOCAL slot atomic inside [baseArr[destPe], ...) ----
    if (args.tokenIndices && args.inpTokenBuf && !args.replayMode) {
      for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
           i += globalWarpNum) {
        index_t srcTokId = i / config.numExpertPerToken;
        index_t destExpert = args.tokenIndices[i];
        if (destExpert < 0) continue;
        index_t destPe = destExpert / config.numExpertPerRank;
        if (destPe < 0 || destPe >= config.worldSize) continue;
        int condition = 0;
        if (laneId < (i % config.numExpertPerToken)) {
          index_t otherExpert = args.tokenIndices[srcTokId * config.numExpertPerToken + laneId];
          condition = (otherExpert >= 0) && (destPe == (otherExpert / config.numExpertPerRank));
        }
        if (__any(condition)) continue;

        index_t destTokId = 0;
        if (laneId == 0) {
          index_t slot = atomicAdd(&localCnt[destPe], 1);
          destTokId = baseArr[destPe] + slot;
          args.dispDestTokIdMap[i] = FlatTokenIndex(config, destPe, destTokId);
          args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] =
              FlatTokenIndex(config, myPe, srcTokId);
        }
        destTokId = __shfl(destTokId, 0);

        if (laneId < config.numExpertPerToken) {
          if (args.weightsBuf) {
            args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(
                destPe)[destTokId * config.numExpertPerToken + laneId] =
                args.weightsBuf[srcTokId * config.numExpertPerToken + laneId];
          }
          args.shmemOutIndicesMemObj->template GetAs<index_t*>(
              destPe)[destTokId * config.numExpertPerToken + laneId] =
              args.tokenIndices[srcTokId * config.numExpertPerToken + laneId];
        }
        if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
          size_t destScaleOffset = (size_t)destTokId * config.scaleDim * config.scaleTypeSize;
          size_t srcScaleOffset = (size_t)srcTokId * config.scaleDim * config.scaleTypeSize;
          core::WarpCopy(
              args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + destScaleOffset,
              args.scalesBuf + srcScaleOffset, config.scaleDim * config.scaleTypeSize);
        }
        size_t srcTokOffset = srcTokId * hiddenDim;
        size_t destTokOffset = destTokId * hiddenDim;
        core::WarpCopy<T, 8>(
            args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) + destTokOffset,
            args.inpTokenBuf + srcTokOffset, hiddenDim);
      }
    }

#if defined(MORI_CNT_STEP) && (MORI_CNT_STEP == 9)
    return;  // DBG stop-stage 9: after Pass B
#endif
    // ---- Signal-based completion (replaces the symmetric cross-device barrier):
    // all local blocks arrive, warp0 RELEASE-fences its P2P writes and posts a
    // per-peer signal, then waits for every peer to signal this rank. ----
    // Use a SEPARATE barrier word (+1) for the completion arrival counter, NOT
    // dispatchGridBarrier[0] which the last GridBarrier (prefix) just used. Reusing
    // [0] races the GridBarrier self-reset: with tiny inputs Pass B is near-instant,
    // so a fast block reaches this atomicAdd and bumps [0] between the resetting
    // block's two atomicCAS(bar,gridDim.x,0) attempts -> that block never sees
    // gridDim.x again and spins forever -> global hang. (4K hides it: Pass B is slow
    // enough that every block finishes the reset before any reaches here.)
    if (_pw) _pt[6] = clock64();
    auto* complBar = args.dispatchGridBarrier + 1;
    if (thdId == 0) atomicAdd(complBar, 1);
    index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
    if (globalWarpId == 0) {
      for (int destPe = laneId; destPe < npes; destPe += warpSize) {
        shmem::ShmemUint32WaitUntilEquals(complBar, gridDim.x);
        __hip_atomic_store(complBar, 0u, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
        index_t numTokenSignal = core::AtomicLoadRelaxed(&localCnt[destPe]) + 1;
        index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
        shmem::ShmemInt32WaitUntilEquals(signal, 0);
        __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
        core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
      }
    }
    if (globalWarpId == 0) {
      for (int srcPe = laneId; srcPe < npes; srcPe += warpSize) {
        index_t* signal = recvTokenNums + srcPe;
        shmem::ShmemInt32WaitUntilGreaterThan(signal, 0);
        __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
        core::AtomicStoreRelaxedSystem(signal, 0);
      }
    }
    if (_pw) _pt[7] = clock64();
#if defined(MORI_CNT_STEP)
    // DBG phase breakdown (gated; dead-code eliminated when MORI_CNT_STEP unset).
    if (_pw && myPe == 0) {
      const double c2u = 1.0 / 2400.0;  // gfx1250 ~2.4GHz: cycles -> us
      printf("[PHASE] bar0=%.1f passA=%.1f bar2=%.1f exch=%.1f prefix=%.1f bar3=%.1f passB=%.1f compl=%.1f tot=%.1f (us)\n",
             (_pt[1] - _pt[0]) * c2u, (_pt[2] - _pt[1]) * c2u, (_pt[3] - _pt[2]) * c2u,
             (_pt[8] - _pt[3]) * c2u, (_pt[4] - _pt[8]) * c2u, (_pt[5] - _pt[4]) * c2u,
             (_pt[6] - _pt[5]) * c2u, (_pt[7] - _pt[6]) * c2u, (_pt[7] - _pt[0]) * c2u);
    }
#endif
#ifdef ENABLE_STANDARD_MOE_ADAPT
    if constexpr (EnableStdMoE) {
      InvokeConvertDispatchOutput<T>(args, myPe);
    }
#endif
#if defined(MORI_CNT_STEP)
    // DBG: dump a per-rank signature of dispDestTokIdMap (the SEND-side map combine
    // reads in its accumulate loop; check_dispatch_result does NOT cover it). Run
    // CNT2 vs plain dispatch-only and diff: differing cksum => Pass A produced a
    // different map; oob>0 => a destLocalTokId >= recv capacity => combine would OOB.
    // Only rank 0 prints so the 4-process stdout does not interleave -> the single
    // line is directly comparable across runs. _sumLoc is an order-independent
    // invariant (unaffected by Pass B atomic race ordering); _ck is order-sensitive.
    if (blockIdx.x == 0 && thdId == 0 && myPe == 0) {
      const int _n = args.curRankNumToken * config.numExpertPerToken;
      const index_t _cap = (index_t)config.MaxNumTokensToRecv();
      unsigned long long _ck = 1469598103934665603ull;
      long long _sumLoc = 0;
      int _valid = 0, _oob = 0;
      index_t _maxLoc = -1;
      for (int _i = 0; _i < _n; ++_i) {
        index_t _dt = args.dispDestTokIdMap[_i];
        _ck = (_ck ^ (unsigned long long)(unsigned)_dt) * 1099511628211ull;
        index_t _pe = PeFromFlatTokenIndex(config, _dt);
        if (_pe >= 0 && _pe < config.worldSize) {
          index_t _lt = LocalTokIdFromFlatTokenIndex(config, _dt);
          _valid++;
          _sumLoc += (long long)_lt;
          if (_lt > _maxLoc) _maxLoc = _lt;
          if (_lt >= _cap) _oob++;
        }
      }
      printf("[DDT] rank0 n=%d valid=%d sumLoc=%lld maxLocalTok=%d recvCap=%d oob=%d cksum=%llu\n",
             _n, _valid, _sumLoc, (int)_maxLoc, (int)_cap, _oob, _ck);
    }
#endif
    return;
  }
#endif  // MORI_DISP_NOTIFY

  IF_ENABLE_PROFILER(
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);
  MORI_TRACE_NEXT(seq, Slot::DispatchSendTokens);

  if (args.tokenIndices && args.inpTokenBuf) {
    // Phase1: send token
    // Each warp compute token offset on destinition PE
    for (int i = globalWarpId; i < args.curRankNumToken * config.numExpertPerToken;
         i += globalWarpNum) {
      index_t srcTokId = i / config.numExpertPerToken;
      index_t destPe;
      index_t destTokId = 0;

      if (!args.replayMode) {
        // Cache routing: decide where this (token, top-k) pair goes via
        // atomicAdd-based slot assignment. Records the routing into dispDestTokIdMap
        // (and the symmetric local view via dispTokIdToSrcTokIdMemObj on the
        // destination PE) so a later replay-routing dispatch / combine can reuse
        // the same layout deterministically.
        index_t destExpert = args.tokenIndices[i];
        // Routing sentinel: a negative expert id means "drop this top-k slot".
        // Skip the dispatch entirely and write the existing combine-side null sentinel
        // (PE == worldSize) into dispDestTokIdMap so combine treats this slot as nullptr.
        if (destExpert < 0) {
          if (laneId == 0) args.dispDestTokIdMap[i] = FlatTokenIndex(config, config.worldSize, 0);
          continue;
        }
        destPe = destExpert / config.numExpertPerRank;
        // Out-of-range expert id guard: destPe is warp-uniform here (one
        // token-expert per warp) and indexes GetAs(destPe) / destPeTokenCounter
        // below. An out-of-range id (e.g. an EPLB physical id
        // >= worldSize*numExpertPerRank) would index those out of bounds (the
        // assert at dispatch is stripped under NDEBUG) -> HSA page fault. Drop it
        // via the same overflow sentinel the dedup path uses; the whole warp
        // skips coherently.
        if (destPe < 0 || destPe >= config.worldSize) {
          if (laneId == 0) args.dispDestTokIdMap[i] = FlatTokenIndex(config, config.worldSize, 0);
          continue;
        }

        // Deduplicate
        assert(config.numExpertPerToken < warpSize);
        int condition = 0;
        if (laneId < (i % config.numExpertPerToken)) {
          index_t otherExpert = args.tokenIndices[srcTokId * config.numExpertPerToken + laneId];
          condition = (otherExpert >= 0) && (destPe == (otherExpert / config.numExpertPerRank));
        }
        if (__any(condition)) {
          // Indicate that this token is already sent to the destination PE by setting an overflow
          // token index
          if (laneId == 0) args.dispDestTokIdMap[i] = FlatTokenIndex(config, config.worldSize, 0);
          continue;
        }

        {
          // Fine-grained timing: slot assignment = remote returning atomic on
          // dispTokOffset[destPe] + the two remote metadata writes.
          MORI_TRACE_SPAN(profiler, Slot::DispSlotAssign);
          if (laneId == 0) {
            // decide token id in dest pe
            // Cross-GPU slot allocation: the offset counter lives on the destination
            // PE, so the fetch-add MUST be SYSTEM-scoped. Plain atomicAdd is agent
            // (device) scope and is NOT atomic across GPUs over the cco/LSA fabric,
            // so concurrent senders can get the same destTokId -> slot collision ->
            // corrupt dispatch (map/payload disagree). Matches v2's system-scope
            // atomic_add_global.
#if defined(MORI_CNT_STEP)
            // DBG: time each remote SYSTEM-scope slot atomic on global warp 0 (real
            // EP4 concurrency: all other warps hammer their dest counters too).
            unsigned long long _al0 = (globalWarpId == 0) ? clock64() : 0ull;
#endif
            destTokId = __hip_atomic_fetch_add(
                args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1,
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
            // NATURAL data-dependency timing (no asm/volatile): this REAL map store
            // reads destTokId, so for correctness the compiler MUST s_waitcnt for the
            // atomic's return before it -> the end-clock right after captures the true
            // round-trip completion. Only extra vs the bare atomic = FlatTokenIndex +
            // one LOCAL store (~tens of ns). If this ~matches the asm-barrier number,
            // it proves the ~3.5us completion latency without any asm trick.
            args.dispDestTokIdMap[i] = FlatTokenIndex(config, destPe, destTokId);
#if defined(MORI_CNT_STEP)
            if (globalWarpId == 0) {
              unsigned long long _al1 = clock64();
              printf("[ATOMLAT-NAT] warp0 pair_i=%d destPe=%d cyc=%llu ~%.0fns\n", i, (int)destPe,
                     _al1 - _al0, (double)(_al1 - _al0) / 2.4);
            }
#endif
            assert(destTokId < config.MaxNumTokensToRecv() &&
                   "Total recv token overflow: increase maxTotalRecvTokens");
            atomicAdd(args.destPeTokenCounter + destPe, 1);
            args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] =
                FlatTokenIndex(config, myPe, srcTokId);
          }
          destTokId = __shfl(destTokId, 0);
        }
      } else {
        // Replay routing: caller already supplied a populated dispDestTokIdMap
        // from a matching cache-routing dispatch. Recover (destPe, destTokId) directly
        // and skip CAS / dedup / cross-rank src-id writes. The sentinel slot
        // (destPe == worldSize) means the original cache-routing dispatch dropped or deduped
        // this top-k slot, so we skip transmitting payload as well.
        index_t flat = args.dispDestTokIdMap[i];
        destPe = PeFromFlatTokenIndex(config, flat);
        if (destPe >= config.worldSize) continue;
        destTokId = LocalTokIdFromFlatTokenIndex(config, flat);
      }

      // Write weights and indices
      {
        MORI_TRACE_SPAN(profiler, Slot::DispWriteMeta);
        if (laneId < config.numExpertPerToken) {
          if (args.weightsBuf) {
            args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(
                destPe)[destTokId * config.numExpertPerToken + laneId] =
                args.weightsBuf[srcTokId * config.numExpertPerToken + laneId];
          }
          args.shmemOutIndicesMemObj->template GetAs<index_t*>(
              destPe)[destTokId * config.numExpertPerToken + laneId] =
              args.tokenIndices[srcTokId * config.numExpertPerToken + laneId];
        }
      }

      // Write scales
      if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
        size_t destScaleOffset = (size_t)destTokId * config.scaleDim * config.scaleTypeSize;
        size_t srcScaleOffset = (size_t)srcTokId * config.scaleDim * config.scaleTypeSize;
        core::WarpCopy(
            args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) + destScaleOffset,
            args.scalesBuf + srcScaleOffset, config.scaleDim * config.scaleTypeSize);
      }

      size_t srcTokOffset = srcTokId * hiddenDim;
      size_t destTokOffset = destTokId * hiddenDim;

      {
        // Fine-grained timing: the actual hidden-dim payload WarpCopy to the
        // destination PE (the bulk P2P write). Unroll=8 issues 8 in-flight 16B
        // loads before the stores (memory-level parallelism) to hide the
        // higher per-access latency of the CCO peer path; matches the v2/FlyDSL
        // multi-stream copy that keeps dispatch fast.
        MORI_TRACE_SPAN(profiler, Slot::DispTokenCopy);
#if defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
        // Inline TDM copy (exact tdm_ep4_dispatch pattern): build the descriptor
        // locally each token, LOAD src->tile, wait, wave barrier, STORE tile->peer, wait.
        {
          typedef int _v4i __attribute__((ext_vector_type(4)));
          typedef int _v8i __attribute__((ext_vector_type(8)));
          const int _D = (int)hiddenDim;
          gfx1250_TDM_GROUP1 g1;
          g1.dataSize(1);
          g1.tensorDim0(_D); g1.tensorDim1(1);
          g1.tensorDim0Stride(_D); g1.tensorDim1Stride(1);
          g1.tileDim0(_D); g1.tileDim1(1);
          gfx1250_TDM_GROUP0 g0; g0.ldsAddr((uintptr_t)_tdmTile);
          _v4i z4{0,0,0,0}; _v8i z8{0,0,0,0,0,0,0,0};
          g0.globalAddr((uintptr_t)(args.inpTokenBuf + srcTokOffset));
          __builtin_amdgcn_tensor_load_to_lds(g0.m_bitfield, g1.m_bitfield, z4, z4, z8, 0);
          __builtin_amdgcn_s_wait_tensorcnt(0);
          __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
          __builtin_amdgcn_wave_barrier();
          __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
          g0.globalAddr((uintptr_t)(args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) +
                                    destTokOffset));
          __builtin_amdgcn_tensor_store_from_lds(g0.m_bitfield, g1.m_bitfield, z4, z4, z8, 0);
          __builtin_amdgcn_s_wait_tensorcnt(0);
        }
#else
        core::WarpCopy<T, 8>(
            args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) + destTokOffset,
            args.inpTokenBuf + srcTokOffset, hiddenDim);
#endif
      }
    }
  }
  __syncthreads();
  if (thdId == 0) atomicAdd(args.dispatchGridBarrier, 1);

  // Send token num & token to expert mapping to other ranks
  MORI_TRACE_NEXT(seq, Slot::DispatchNotifyPeer);
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      // Wait until all tokens are sent
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, gridDim.x);
      __hip_atomic_store(args.dispatchGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

      // Add 1 so that when token number == 0, receiver side still know the signal is sent
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      // System release fence: make this rank's remote payload/index/weight/tis
      // writes to destPe globally visible BEFORE the readiness signal lands, so a
      // peer that observes the signal is guaranteed to see the data (matches v2's
      // fence_system_release). Without it, on the cco/LSA fabric the signal can be
      // observed before the P2P writes drain -> peer reads stale -> wrong dispatch.
      __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
      core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
    }
  }

  // Phase 2: recv token
  // Each warp wait until sender finished by waiting token number signal
  MORI_TRACE_NEXT(seq, Slot::DispatchWaitPeerToken);
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      index_t* signal = recvTokenNums + destPe;
      index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      // System acquire fence: pair with the sender's release so that after we see
      // the signal, this rank's subsequent reads observe the peer's P2P writes
      // (matches v2's fence_system_acquire).
      __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
      core::AtomicStoreRelaxedSystem(signal, 0);
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);

      // reset local counter
      args.destPeTokenCounter[destPe] = 0;
    }

    // reset counter
    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
    }
  }

#ifdef ENABLE_STANDARD_MOE_ADAPT
  if constexpr (EnableStdMoE) {
    InvokeConvertDispatchOutput<T>(args, myPe);
  }
#endif
}

template <typename T, bool EnableStdMoE = false>
__global__ void EpDispatchIntraNodeKernel(EpDispatchCombineArgs<T> args) {
  EpDispatchIntraNodeKernel_body<T, EnableStdMoE>(args);
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpCombineIntraNodeKernel                                    */
/* ---------------------------------------------------------------------------------------------- */
template <typename T, bool UseP2PRead = true, bool EnableStdMoE = false,
          bool UseFp8DirectCast = false, bool UseFp8BlockwiseQuant = false, bool UseWeights = true,
          int Vec8Top8BlockElems = 0, int Vec8AccumNum = 8, bool UseFp4Combine = false>
__device__ __forceinline__ void EpCombineIntraNodeKernel_body(EpDispatchCombineArgs<T> args) {
  using TokT =
      std::conditional_t<UseFp8DirectCast || UseFp8BlockwiseQuant, core::CombineInternalFp8, T>;
  // UseFp4Combine reuses the FP8-blockwise staging/scale layout but transports each element as
  // packed FP4 (E2M1, 2/byte -> half the combine bytes). It is a variant of blockwise combine.
  static_assert(!UseFp4Combine || UseFp8BlockwiseQuant,
                "UseFp4Combine builds on the FP8-blockwise combine path");
  static_assert(!(UseFp8DirectCast && UseFp8BlockwiseQuant),
                "Fp8 direct cast and blockwise quant are mutually exclusive");
  static_assert((!UseFp8DirectCast && !UseFp8BlockwiseQuant) || std::is_same_v<T, hip_bfloat16>,
                "Fp8 combine quant currently only supports bf16 input");
  static_assert((Vec8Top8BlockElems & (Vec8Top8BlockElems - 1)) == 0,
                "Vec8Top8BlockElems must be 0 or a power of two");
  const EpDispatchCombineConfig& config = args.config;
  int thdId = threadIdx.x;
  int thdNum = blockDim.x;

  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;

  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;
  int globalThdNum = gridDim.x * warpNum * warpSize;

  int myPe = config.rank;
  int npes = config.worldSize;

  IF_ENABLE_PROFILER(
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);
  MORI_TRACE_NEXT(seq, Slot::CombineStageInput);

  const uint64_t crossDeviceBarrierFlag = args.crossDeviceBarrierFlag[0];
  // Copy input to shmem registered buffer so that other GPUs can access directly
  index_t totalRecvTokenNum = args.totalRecvTokenNum[0];
#if defined(MORI_CNT_STEP)
  if (blockIdx.x == 0 && thdId == 0)
    printf("[CMB C0] rank=%d recv=%d xdevFlag=%llu enter\n", myPe, (int)totalRecvTokenNum,
           (unsigned long long)crossDeviceBarrierFlag);
#endif
  // When TokT != T (e.g. fp8 combine), staging layout uses TokT-sized tokens. FP4 blockwise packs
  // two E2M1 values per byte, so its token region is half the FP8 one -- keep this in sync with
  // EpDispatchCombineConfig::CombineTokenRegionBytes() used by the host staging allocator.
  const size_t hiddenDim = config.HiddenDimSz();
  const size_t hiddenBytes =
      UseFp4Combine ? ((hiddenDim + 1) / 2) * sizeof(TokT) : hiddenDim * sizeof(TokT);
  const size_t weightBytes =
      (UseWeights && args.weightsBuf != nullptr) ? config.numExpertPerToken * sizeof(float) : 0;
  const size_t scaleBytes =
      UseFp8BlockwiseQuant ? static_cast<size_t>(args.fp8BlockwiseCombineScaleDim) * sizeof(float)
                           : 0;
  const size_t combXferBytes = hiddenBytes + scaleBytes + weightBytes;

  if constexpr (EnableStdMoE) {
#ifdef ENABLE_STANDARD_MOE_ADAPT
    InvokeConvertCombineInput<T, UseP2PRead>(args, myPe);
#endif
  } else if constexpr (UseP2PRead) {
    if (args.config.useExternalInpBuffer) {
      for (int i = globalWarpId; i < totalRecvTokenNum; i += globalWarpNum) {
        if constexpr (UseFp8BlockwiseQuant) {
          core::WarpQuantizeToFp8Blockwise<core::CombineInternalFp8>(
              args.intraNodeTokBufs.combineInp->template GetAs<TokT*>() + i * hiddenDim,
              args.shmemInpScalesMemObj->template GetAs<float*>() +
                  i * args.fp8BlockwiseCombineScaleDim,
              args.inpTokenBuf + i * hiddenDim, hiddenDim, args.fp8BlockwiseCombineScaleDim);
        } else if constexpr (!std::is_same_v<T, TokT> &&
                             std::is_same_v<TokT, core::CombineInternalFp8>) {
          core::WarpCastBf16ToCombineInternalFp8<T>(
              args.intraNodeTokBufs.combineInp->template GetAs<TokT*>() + i * hiddenDim,
              args.inpTokenBuf + i * hiddenDim, hiddenDim, laneId);
        } else {
          core::WarpCopy(args.intraNodeTokBufs.combineInp->template GetAs<T*>() + i * hiddenDim,
                         args.inpTokenBuf + i * hiddenDim, hiddenDim);
        }
      }
    }
    if constexpr (UseWeights) {
      MORI_TRACE_NEXT(seq, Slot::CombineCopyWeights);
      if (args.weightsBuf) {
        for (int i = globalWarpId; i < totalRecvTokenNum; i += globalWarpNum) {
          core::WarpCopy(
              args.shmemInpWeightsMemObj->template GetAs<float*>() + i * config.numExpertPerToken,
              args.weightsBuf + i * config.numExpertPerToken, config.numExpertPerToken);
        }
      }
    }
  } else {
    // When the caller passes a routing handle, args.dispTokIdToSrcTokIdLocal
    // holds a per-call snapshot of the symmetric local view. Otherwise fall
    // back to the shared symmetric buffer.
    const index_t* localSrcMap =
        args.dispTokIdToSrcTokIdLocal != nullptr
            ? args.dispTokIdToSrcTokIdLocal
            : args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(myPe);
#ifdef ENABLE_PROFILER
    for (int tokenIdx = globalWarpId; tokenIdx < totalRecvTokenNum; tokenIdx += globalWarpNum) {
      index_t destTokId = localSrcMap[tokenIdx];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);
      index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
      uint8_t* destStagingPtr = args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(destPe) +
                                SendBufSlotOffset(config, myPe, destLocalTokId) * combXferBytes;
      if constexpr (UseFp8BlockwiseQuant) {
        core::WarpQuantizeToCombineBlockwise<UseFp4Combine, core::CombineInternalFp8>(
            reinterpret_cast<core::CombineInternalFp8*>(destStagingPtr),
            reinterpret_cast<float*>(destStagingPtr + hiddenBytes),
            args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim, args.fp8BlockwiseCombineScaleDim);
      } else if constexpr (!std::is_same_v<T, TokT> &&
                           std::is_same_v<TokT, core::CombineInternalFp8>) {
        core::WarpCastBf16ToCombineInternalFp8<T>(reinterpret_cast<TokT*>(destStagingPtr),
                                                  args.inpTokenBuf + tokenIdx * hiddenDim,
                                                  hiddenDim, laneId);
      } else {
        core::WarpCopy(reinterpret_cast<T*>(destStagingPtr),
                       args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim);
      }
    }
    if constexpr (UseWeights) {
      MORI_TRACE_NEXT(seq, Slot::CombineCopyWeights);
      if (args.weightsBuf) {
        for (int tokenIdx = globalWarpId; tokenIdx < totalRecvTokenNum; tokenIdx += globalWarpNum) {
          index_t destTokId = localSrcMap[tokenIdx];
          index_t destPe = PeFromFlatTokenIndex(config, destTokId);
          index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
          uint8_t* destStagingPtr =
              args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(destPe) +
              SendBufSlotOffset(config, myPe, destLocalTokId) * combXferBytes;
          core::WarpCopy(reinterpret_cast<float*>(destStagingPtr + hiddenBytes + scaleBytes),
                         args.weightsBuf + tokenIdx * config.numExpertPerToken,
                         config.numExpertPerToken);
        }
      }
    }
#else
    for (int tokenIdx = globalWarpId; tokenIdx < totalRecvTokenNum; tokenIdx += globalWarpNum) {
      index_t destTokId = localSrcMap[tokenIdx];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);
      index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
      uint8_t* destStagingPtr = args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(destPe) +
                                SendBufSlotOffset(config, myPe, destLocalTokId) * combXferBytes;
      if constexpr (UseFp8BlockwiseQuant) {
        core::WarpQuantizeToCombineBlockwise<UseFp4Combine, core::CombineInternalFp8>(
            reinterpret_cast<core::CombineInternalFp8*>(destStagingPtr),
            reinterpret_cast<float*>(destStagingPtr + hiddenBytes),
            args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim, args.fp8BlockwiseCombineScaleDim);
      } else if constexpr (!std::is_same_v<T, TokT> &&
                           std::is_same_v<TokT, core::CombineInternalFp8>) {
        core::WarpCastBf16ToCombineInternalFp8<T>(reinterpret_cast<TokT*>(destStagingPtr),
                                                  args.inpTokenBuf + tokenIdx * hiddenDim,
                                                  hiddenDim, laneId);
      } else {
        core::WarpCopy(reinterpret_cast<T*>(destStagingPtr),
                       args.inpTokenBuf + tokenIdx * hiddenDim, hiddenDim);
      }
      if constexpr (UseWeights) {
        if (args.weightsBuf) {
          core::WarpCopy(reinterpret_cast<float*>(destStagingPtr + hiddenBytes + scaleBytes),
                         args.weightsBuf + tokenIdx * config.numExpertPerToken,
                         config.numExpertPerToken);
        }
      }
    }
#endif
  }

  // Make sure copy on all GPUs are finished
  MORI_TRACE_NEXT(seq, Slot::CombineBarrier);
#if defined(MORI_CNT_STEP)
  if (blockIdx.x == 0 && thdId == 0) printf("[CMB C1] rank=%d pre-xdevbar\n", myPe);
#endif
#if defined(MORI_CNT_STEP) && (MORI_CNT_STEP == 11)
  return;  // DBG combine stop-stage 11: after staging, BEFORE CrossDeviceBarrier
           // (all ranks return symmetrically -> no one waits -> clean if staging ok)
#endif
  CrossDeviceBarrierIntraNodeKernel(args, crossDeviceBarrierFlag);
#if defined(MORI_CNT_STEP) && (MORI_CNT_STEP == 12)
  return;  // DBG combine stop-stage 12: after CrossDeviceBarrier
#endif
#if defined(MORI_CNT_STEP)
  if (blockIdx.x == 0 && thdId == 0) printf("[CMB C2] rank=%d post-xdevbar\n", myPe);
#endif
  // With a routing handle, the caller owns this tensor (it may still be alive in autograd ctx),
  // so we skip the reset. The next dispatch will allocate or replay its own.
  if (args.dispTokIdToSrcTokIdLocal == nullptr) {
    *args.totalRecvTokenNum = 0;
  }
  if (args.curRankNumToken == 0) return;

  MORI_TRACE_NEXT(seq, Slot::CombineAccumSetup);
  extern __shared__ char sharedMem[];
  // Layout: [srcPtrs] [srcWeightsPtr if UseWeights] [srcScalePtrs if UseFp8BlockwiseQuant];
  // host-side combine_shared_mem() must use the same flags.
  TokT** srcPtrs = reinterpret_cast<TokT**>(sharedMem) + warpId * config.numExpertPerToken;
  float** srcWeightsPtr = nullptr;
  if constexpr (UseWeights) {
    srcWeightsPtr = reinterpret_cast<float**>(sharedMem) + warpNum * config.numExpertPerToken +
                    warpId * config.numExpertPerToken;
  }
  float** srcScalePtrs = nullptr;
  if constexpr (UseFp8BlockwiseQuant) {
    constexpr int scalePtrArrayOffset = UseWeights ? 2 : 1;
    srcScalePtrs = reinterpret_cast<float**>(sharedMem) +
                   scalePtrArrayOffset * warpNum * config.numExpertPerToken +
                   warpId * config.numExpertPerToken;
  }

  MultiWarpIter mwIter(globalWarpNum, args.curRankNumToken, hiddenDim);
#if defined(MORI_CNT_STEP)
  if (blockIdx.x == 0 && thdId == 0)
    printf("[CMB C2b] rank=%d pre-accum curTok=%d\n", myPe, (int)args.curRankNumToken);
#endif

  assert(config.numExpertPerToken < warpSize);
  for (int i = globalWarpId; i < (args.curRankNumToken * mwIter.warpsPerItem); i += globalWarpNum) {
    int tokenId, inTokenPartId;
    size_t hiddenDimOffset, hiddenDimSize;
    mwIter.Decode(i, tokenId, inTokenPartId, hiddenDimOffset, hiddenDimSize);

    // Prepare data pointers on different GPUs
    MORI_TRACE_NEXT(seq, Slot::CombinePreparePtrs);
    for (int j = laneId; j < config.numExpertPerToken; j += warpSize) {
      index_t destTokId = args.dispDestTokIdMap[tokenId * config.numExpertPerToken + j];
      index_t destPe = PeFromFlatTokenIndex(config, destTokId);

      if (destPe < config.worldSize) {
        if constexpr (UseP2PRead) {
          index_t destLocalTokId = LocalTokIdFromFlatTokenIndex(config, destTokId);
          srcPtrs[j] = args.intraNodeTokBufs.combineInp->template GetAs<TokT*>(destPe) +
                       destLocalTokId * hiddenDim + hiddenDimOffset;
          if constexpr (UseWeights) {
            srcWeightsPtr[j] = args.shmemInpWeightsMemObj->template GetAs<float*>(destPe) +
                               destLocalTokId * config.numExpertPerToken;
          }
          if constexpr (UseFp8BlockwiseQuant) {
            float* scalePtr = args.shmemInpScalesMemObj->template GetAs<float*>(destPe) +
                              destLocalTokId * args.fp8BlockwiseCombineScaleDim;
            srcScalePtrs[j] = (scalePtr[0] < 0.0f) ? scalePtr : nullptr;
          }
        } else {
          srcPtrs[j] = reinterpret_cast<TokT*>(
                           args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(myPe) +
                           SendBufSlotOffset(config, destPe, tokenId) * combXferBytes) +
                       hiddenDimOffset;
          if constexpr (UseWeights) {
            srcWeightsPtr[j] = reinterpret_cast<float*>(
                args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(myPe) +
                SendBufSlotOffset(config, destPe, tokenId) * combXferBytes + hiddenBytes +
                scaleBytes);
          }
          if constexpr (UseFp8BlockwiseQuant) {
            float* scalePtr = reinterpret_cast<float*>(
                args.intraNodeTokBufs.combineInp->template GetAs<uint8_t*>(myPe) +
                SendBufSlotOffset(config, destPe, tokenId) * combXferBytes + hiddenBytes);
            srcScalePtrs[j] = (scalePtr[0] < 0.0f) ? scalePtr : nullptr;
          }
        }
      } else {
        srcPtrs[j] = nullptr;
        if constexpr (UseWeights) {
          srcWeightsPtr[j] = nullptr;
        }
        if constexpr (UseFp8BlockwiseQuant) {
          srcScalePtrs[j] = nullptr;
        }
      }
    }

    T* outPtr = args.intraNodeTokBufs.combineOut->template GetAs<T*>() + tokenId * hiddenDim +
                hiddenDimOffset;

    int validAccumCount = config.numExpertPerToken;
    if (config.worldSize <= 4) {
      {
        int isValid = 0;
        TokT* myTokPtr = nullptr;
        float* myScalePtr = nullptr;
        if (laneId < config.numExpertPerToken) {
          myTokPtr = srcPtrs[laneId];
          if constexpr (UseFp8BlockwiseQuant) {
            myScalePtr = srcScalePtrs[laneId];
          }
          isValid = (myTokPtr != nullptr) ? 1 : 0;
        }
        unsigned long long validMask = __ballot(isValid);
        validAccumCount = __popcll(validMask);
        if (validAccumCount < config.numExpertPerToken && isValid) {
          int myPos = __popcll(validMask & ((1ULL << laneId) - 1));
          srcPtrs[myPos] = myTokPtr;
          if constexpr (UseFp8BlockwiseQuant) {
            srcScalePtrs[myPos] = myScalePtr;
          }
        }
      }
    }

    if constexpr (UseFp8BlockwiseQuant) {
      MORI_TRACE_NEXT(seq, Slot::CombineDequantAccum);
      if constexpr (Vec8Top8BlockElems != 0) {
        if (mwIter.warpsPerItem == 1) {
          core::WarpAccumCombineDequantFullBlockVec8Top8<UseFp4Combine, T, core::CombineInternalFp8,
                                                         Vec8Top8BlockElems, Vec8AccumNum>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), hiddenDim);
        } else if ((hiddenDimOffset & 0x7) == 0 && (hiddenDimSize & 0x7) == 0) {
          core::WarpAccumCombineDequantSegmentBlockVec8Top8<
              UseFp4Combine, T, core::CombineInternalFp8, Vec8Top8BlockElems, Vec8AccumNum>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), hiddenDimOffset, hiddenDimSize);
        } else {
          // Misaligned segment: vec8 helper would fault on the load. Tiny scalar fallback.
          core::WarpAccumCombineDequantSegmentScalarTop8<UseFp4Combine, T, core::CombineInternalFp8,
                                                         Vec8Top8BlockElems, Vec8AccumNum>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), hiddenDimOffset, hiddenDimSize,
              hiddenDim, args.fp8BlockwiseCombineScaleDim);
        }
      } else {
        if (mwIter.warpsPerItem == 1) {
          core::WarpAccumCombineDequantFull<UseFp4Combine, T, core::CombineInternalFp8>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), validAccumCount, hiddenDim,
              args.fp8BlockwiseCombineScaleDim);
        } else {
          core::WarpAccumCombineDequantSegment<UseFp4Combine, T, core::CombineInternalFp8>(
              outPtr, reinterpret_cast<const core::CombineInternalFp8* const*>(srcPtrs),
              reinterpret_cast<const float* const*>(srcScalePtrs), validAccumCount, hiddenDimOffset,
              hiddenDimSize, hiddenDim, args.fp8BlockwiseCombineScaleDim);
        }
      }
    } else if constexpr (!std::is_same_v<T, TokT> &&
                         std::is_same_v<TokT, core::CombineInternalFp8>) {
      MORI_TRACE_NEXT(seq, Slot::CombineDequantAccum);
      core::WarpAccumCombineInternalFp8ToBf16(outPtr, reinterpret_cast<const TokT* const*>(srcPtrs),
                                              validAccumCount, laneId, hiddenDimSize);
    } else {
      MORI_TRACE_NEXT(seq, Slot::CombineDequantAccum);
      // 16B vec load + load-first/unroll gather (v2-style): keep AccumNum*Unroll
      // remote peer reads in flight to hide CCO/xGMI latency (gfx1250 combine).
      core::WarpAccumLF<T, 16>(outPtr, srcPtrs, nullptr, validAccumCount, hiddenDimSize);
    }

    if constexpr (UseWeights) {
      MORI_TRACE_NEXT(seq, Slot::CombineAccumWeights);
      if (args.weightsBuf && inTokenPartId == mwIter.warpsPerItem - 1) {
        core::WarpAccum<float, 4>(args.shmemCombineOutWeightsMemObj->template GetAs<float*>() +
                                      tokenId * config.numExpertPerToken,
                                  srcWeightsPtr, nullptr, config.numExpertPerToken,
                                  config.numExpertPerToken);
      }
    }
  }
#if defined(MORI_CNT_STEP)
  if (blockIdx.x == 0 && thdId == 0) printf("[CMB C3] rank=%d combine done\n", myPe);
#endif
}

template <typename T, bool UseP2PRead = true, bool EnableStdMoE = false,
          bool UseFp8DirectCast = false, bool UseFp8BlockwiseQuant = false, bool UseWeights = true,
          int Vec8Top8BlockElems = 0, int Vec8AccumNum = 8, bool UseFp4Combine = false>
__global__ void EpCombineIntraNodeKernel(EpDispatchCombineArgs<T> args) {
  EpCombineIntraNodeKernel_body<T, UseP2PRead, EnableStdMoE, UseFp8DirectCast, UseFp8BlockwiseQuant,
                                UseWeights, Vec8Top8BlockElems, Vec8AccumNum, UseFp4Combine>(args);
}

}  // namespace moe
}  // namespace mori
