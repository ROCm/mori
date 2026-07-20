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
                                                         const uint64_t crossDeviceBarrierFlag,
                                                         long long* tLocalWait = nullptr,
                                                         long long* tPeerWait = nullptr) {
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int globalThdId = blockIdx.x * blockDim.x + threadIdx.x;

  int warpNum = blockDim.x / warpSize;
  int globalWarpNum = gridDim.x * warpNum;
#if defined(MORI_DISP_TIMING)
  const bool _xt = (args.config.rank == 0 && blockIdx.x == 0 && thdId == 0);
#endif

  __syncthreads();
  if (thdId == 0) atomicAdd(args.combineGridBarrier, 1);

  if (globalThdId < args.config.worldSize) {
#if defined(MORI_DISP_TIMING)
    long long _w0 = (_xt && tLocalWait) ? wall_clock64() : 0;
#endif
    // Set remote flag after all copies are done
    shmem::ShmemUint32WaitUntilEquals(args.combineGridBarrier, gridDim.x);
#if defined(MORI_DISP_TIMING)
    if (_xt && tLocalWait) *tLocalWait = wall_clock64() - _w0;
#endif
    __hip_atomic_store(args.combineGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);

    __threadfence_system();
    core::AtomicStoreRelaxedSystem(
        args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>(globalThdId) + args.config.rank,
        crossDeviceBarrierFlag);
  }

  if (globalThdId == 0) atomicAdd(args.crossDeviceBarrierFlag, 1);

  uint64_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>();
#if defined(MORI_DISP_TIMING)
  long long _p0 = (_xt && tPeerWait) ? wall_clock64() : 0;
#endif
  if (thdId < args.config.worldSize) {
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != crossDeviceBarrierFlag) {
    }
  }
#if defined(MORI_DISP_TIMING)
  if (_xt && tPeerWait) *tPeerWait = wall_clock64() - _p0;
#endif
  __syncthreads();
}

#if defined(MORI_DISP_NOTIFY)
// Monotonic reusable intra-GPU grid barrier over a local uint32 counter. The
// counter must be 0 at kernel entry and is left non-zero until reset by the
// caller; each call advances the arrival total by gridDim.x and blocks until it
// reaches `expected` (= barrierIndex * gridDim.x). Correct ONLY when every block
// is co-resident (block_num <= CU) -- guaranteed by the tuned dispatch geometry
// (192 <= 256 CU); a non-resident block would never arrive and deadlock.
__device__ __forceinline__ void NotifyGridBarrier(uint32_t* bar, uint32_t expected) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __hip_atomic_fetch_add(bar, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    while (__hip_atomic_load(bar, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < expected) {
    }
  }
  __syncthreads();
}
#endif

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
  // ===================== NOTIFY-based slot pre-assignment =====================
  // Two passes with an intra-GPU grid barrier between them (all blocks co-resident
  // at the tuned geometry). Eliminates the per-token cross-GPU (system-scope) slot
  // atomic: (A) count committed tokens per dest PE locally, (B) exchange the count
  // matrix + cross-device barrier, compute each source rank's contiguous slot base,
  // (C) send with a LOCAL (agent-scope) slot atomic inside the pre-assigned region.
  {
    const int globalThdId = blockIdx.x * blockDim.x + thdId;
    const int totalThreads = gridDim.x * blockDim.x;
    index_t* localCnt = args.dispTokOffsetMemObj->template GetAs<index_t*>();     // [npes]
    index_t* baseArr = args.destPeTokenCounter;                                   // [npes] (reused)
    index_t* matLocal = args.dispCountMatrixMemObj->template GetAs<index_t*>();   // [npes*npes]
    const uint64_t xdevFlag = args.crossDeviceBarrierFlag[0];
    uint32_t barExp = 0;

#if defined(MORI_DISP_TIMING)
    // Fine-grained in-kernel wall_clock64 breakdown of NOTIFY dispatch (rank0/
    // block0/thd0). Each grid barrier, cross-device barrier and threadfence is
    // timed separately to pinpoint where the ~1ms goes.
    const bool _tThd = (myPe == 0 && blockIdx.x == 0 && thdId == 0);
    long long _tp[11];
    int _tn = 0;
    long long _nbCopy = 0, _nbAtom = 0;  // Pass B copy / local-atomic sums (warp0)
    int _nbCopyN = 0, _nbAtomN = 0;
    long long _xb1L = 0, _xb1P = 0, _xb2L = 0, _xb2P = 0;  // xdev barrier local/peer waits
#define TSTAMP() do { if (_tThd) _tp[_tn++] = wall_clock64(); } while (0)
#else
#define TSTAMP()
#endif
    TSTAMP();  // [0] entry

    // zero local per-destPe counter
    for (int d = globalThdId; d < npes; d += totalThreads) localCnt[d] = 0;
    barExp += gridDim.x;
    NotifyGridBarrier(args.dispatchGridBarrier, barExp);
    TSTAMP();  // [1] after zero-counter + setup grid barrier

    // ---- Pass A: count committed (token -> destPe) into localCnt (agent atomic) ----
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
    TSTAMP();  // [2] after Pass A count loop (pre grid barrier)
    barExp += gridDim.x;
    NotifyGridBarrier(args.dispatchGridBarrier, barExp);
    TSTAMP();  // [3] after grid barrier #1

    // ---- Exchange: block 0 writes my count row to every peer's matrix[myPe][*] ----
    if (blockIdx.x == 0 && warpId == 0) {
      for (int p = laneId; p < npes; p += warpSize) {
        index_t* peerMat = args.dispCountMatrixMemObj->template GetAs<index_t*>(p);
        for (int d = 0; d < npes; ++d) peerMat[myPe * npes + d] = localCnt[d];
      }
    }
    __threadfence_system();
    TSTAMP();  // [4] after matrix exchange write + threadfence (pre xdev barrier)
#if defined(MORI_DISP_TIMING)
    CrossDeviceBarrierIntraNodeKernel(args, xdevFlag, &_xb1L, &_xb1P);
#else
    CrossDeviceBarrierIntraNodeKernel(args, xdevFlag);
#endif
    TSTAMP();  // [5] after cross-device barrier #1

    // ---- Compute base[destPe] = sum_{s<myPe} M[s][destPe]; recvCount; reset localCnt ----
    if (globalThdId < npes) {
      int d = globalThdId;
      index_t b = 0;
      for (int s = 0; s < myPe; ++s) b += matLocal[s * npes + d];
      baseArr[d] = b;
      localCnt[d] = 0;
    }
    if (globalThdId == 0) {
      index_t tot = 0;
      for (int s = 0; s < npes; ++s) tot += matLocal[s * npes + myPe];
      *args.totalRecvTokenNum = tot;
    }
    TSTAMP();  // [6] after prefix-sum base compute (pre grid barrier)
    barExp += gridDim.x;
    NotifyGridBarrier(args.dispatchGridBarrier, barExp);
    TSTAMP();  // [7] after grid barrier #2

    // ---- Pass B: send payload with LOCAL slot assignment inside the region ----
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
#if defined(MORI_DISP_TIMING)
          long long _b0 = _tThd ? wall_clock64() : 0;
#endif
          index_t _slot = atomicAdd(&localCnt[destPe], 1);
#if defined(MORI_DISP_TIMING)
          if (_tThd) { asm volatile("" ::"v"(_slot)); _nbAtom += wall_clock64() - _b0; _nbAtomN++; }
#endif
          destTokId = baseArr[destPe] + _slot;
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
#if defined(MORI_DISP_TIMING)
        long long _bc0 = _tThd ? wall_clock64() : 0;
#endif
        core::WarpCopy<T, 8>(
            args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) + destTokOffset,
            args.inpTokenBuf + srcTokOffset, hiddenDim);
#if defined(MORI_DISP_TIMING)
        if (_tThd) { _nbCopy += wall_clock64() - _bc0; _nbCopyN++; }
#endif
      }
    }

    TSTAMP();  // [8] after Pass B payload send

    // Make all payload/metadata writes visible to peers, then cross-device barrier
    // so combine (next kernel) observes the dispatched data.
    __threadfence_system();
    TSTAMP();  // [9] after threadfence_system (pre xdev barrier #2)
#if defined(MORI_DISP_TIMING)
    CrossDeviceBarrierIntraNodeKernel(args, xdevFlag + 1, &_xb2L, &_xb2P);
#else
    CrossDeviceBarrierIntraNodeKernel(args, xdevFlag + 1);
#endif
    if (globalThdId == 0) *args.dispatchGridBarrier = 0;  // reset for next launch
    TSTAMP();  // [10] after cross-device barrier #2
#if defined(MORI_DISP_TIMING)
    if (_tThd) {
      long long tk = _tp[10] - _tp[0];
      printf(
          "[NOTIFY-TIMING] tot=%lld tk | setup=%lld countLoop=%lld gridbar1=%lld "
          "exch+fence=%lld xdevbar1=%lld(L=%lld P=%lld) prefix=%lld gridbar2=%lld sendB=%lld "
          "(copySum=%lld atomSum=%lld) fence2=%lld xdevbar2=%lld(L=%lld P=%lld) tk\n",
          tk, _tp[1] - _tp[0], _tp[2] - _tp[1], _tp[3] - _tp[2], _tp[4] - _tp[3],
          _tp[5] - _tp[4], _xb1L, _xb1P, _tp[6] - _tp[5], _tp[7] - _tp[6], _tp[8] - _tp[7],
          _nbCopy, _nbAtom, _tp[9] - _tp[8], _tp[10] - _tp[9], _xb2L, _xb2P);
    }
#endif
#undef TSTAMP
#ifdef ENABLE_STANDARD_MOE_ADAPT
    if constexpr (EnableStdMoE) {
      InvokeConvertDispatchOutput<T>(args, myPe);
    }
#endif
    return;
  }
#endif  // MORI_DISP_NOTIFY

  IF_ENABLE_PROFILER(
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);
  MORI_TRACE_NEXT(seq, Slot::DispatchSendTokens);

#if defined(MORI_DISP_TIMING)
  // Accumulate, on ONE representative warp lane (rank0/globalWarp0/lane0), the
  // total wall time spent in the per-token SYSTEM-scope remote slot atomic vs the
  // hidden-dim payload copy, to weigh "atomics vs data movement" in the legacy path.
  const bool _tThd = (myPe == 0 && globalWarpId == 0 && laneId == 0);
  long long _atomAcc = 0, _copyAcc = 0, _loopT0 = 0, _loopT1 = 0, _pN = 0, _pR = 0;
  int _atomCnt = 0, _copyCnt = 0;
  long long _atomDs[32];  // per-atomic wall delta (critical-path latency)
  // BURST TEST (MORI_BW_DUMMY_ATOMICS==1000): force ALL blocks to start the send
  // loop simultaneously via a grid barrier on combineGridBarrier (unused by legacy
  // dispatch; reset to 0 for the later combine kernel across the kernel boundary).
  // If this alone slows the send ~2x, "synchronized start -> congestion" is real.
  if (args.bwDummyAtomics == 1000) {
    const int _gtid = blockIdx.x * blockDim.x + thdId;
    __syncthreads();
    if (thdId == 0) __hip_atomic_fetch_add(args.combineGridBarrier, 1u, __ATOMIC_ACQ_REL,
                                           __HIP_MEMORY_SCOPE_AGENT);
    if (thdId == 0)
      while (__hip_atomic_load(args.combineGridBarrier, __ATOMIC_ACQUIRE,
                               __HIP_MEMORY_SCOPE_AGENT) < gridDim.x) {
      }
    __syncthreads();
    if (_gtid == 0)
      __hip_atomic_store(args.combineGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  if (_tThd) _loopT0 = wall_clock64();
#endif
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
#if defined(MORI_DISP_TIMING)
            long long _a0 = _tThd ? wall_clock64() : 0;
#endif
            destTokId = __hip_atomic_fetch_add(
                args.dispTokOffsetMemObj->template GetAs<index_t*>(destPe), 1,
                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#if defined(MORI_DISP_TIMING)
            if (_tThd) {
              // Force the atomic's return value into a register (materialize the
              // s_wait) BEFORE the second timestamp, so we measure the true
              // issue->return critical-path latency, not just the issue.
              asm volatile("" ::"v"(destTokId));
              long long _a1 = wall_clock64();
              if (_atomCnt < 32) _atomDs[_atomCnt] = _a1 - _a0;
              _atomAcc += _a1 - _a0;
              _atomCnt++;
            }
#endif
            assert(destTokId < config.MaxNumTokensToRecv() &&
                   "Total recv token overflow: increase maxTotalRecvTokens");
            atomicAdd(args.destPeTokenCounter + destPe, 1);
            // In dispDestTokIdMap, record the destination slot for this token-expert pair (flat
            // index into the dest PE's recv buffer) In dispTokIdToSrcTokIdMemObj on the dest PE,
            // record which global source token occupies this slot (for combine-phase routing)
            args.dispDestTokIdMap[i] = FlatTokenIndex(config, destPe, destTokId);
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

      if (myPe == 3 && (srcTokId == 108 || srcTokId == 0) && laneId == 0 &&
          (i % config.numExpertPerToken) == 0) {
        const unsigned* sp = (const unsigned*)(args.inpTokenBuf + srcTokOffset);
        printf("SND r3 tok=%d destPe=%d destTokId=%d inpBase=%p srcOff=%llu s0=%08x s1=%08x smid=%08x\n",
               (int)srcTokId, destPe, (int)destTokId, (void*)args.inpTokenBuf,
               (unsigned long long)srcTokOffset, sp[0], sp[1],
               ((const unsigned*)(args.inpTokenBuf + srcTokOffset + hiddenDim / 2))[0]);
      }

      {
        // Fine-grained timing: the actual hidden-dim payload WarpCopy to the
        // destination PE (the bulk P2P write). Unroll=8 issues 8 in-flight 16B
        // loads before the stores (memory-level parallelism) to hide the
        // higher per-access latency of the CCO peer path; matches the v2/FlyDSL
        // multi-stream copy that keeps dispatch fast.
        MORI_TRACE_SPAN(profiler, Slot::DispTokenCopy);
#if defined(MORI_DISP_TIMING)
        long long _c0 = _tThd ? wall_clock64() : 0;
#endif
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
#if defined(MORI_DISP_TIMING)
        if (_tThd) { _copyAcc += wall_clock64() - _c0; _copyCnt++; }
#endif
      }
    }
  }
#if defined(MORI_DISP_TIMING)
  if (_tThd) _loopT1 = wall_clock64();  // block0 send loop done (pre-syncthreads)
#endif
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
#if defined(MORI_DISP_TIMING)
  if (_tThd) _pN = wall_clock64();  // after grid barrier wait + notify peers
#endif

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

#if defined(MORI_DISP_TIMING)
  if (_tThd) {
    _pR = wall_clock64();
    int _np = _atomCnt < 16 ? _atomCnt : 16;
    printf(
        "[ATOMIC-EACH] warp0 n=%d each_tk= %lld %lld %lld %lld %lld %lld %lld %lld\n",
        _atomCnt, _np > 0 ? _atomDs[0] : -1, _np > 1 ? _atomDs[1] : -1,
        _np > 2 ? _atomDs[2] : -1, _np > 3 ? _atomDs[3] : -1, _np > 4 ? _atomDs[4] : -1,
        _np > 5 ? _atomDs[5] : -1, _np > 6 ? _atomDs[6] : -1, _np > 7 ? _atomDs[7] : -1);
    // Grid-wide phase breakdown (block0/thd0), same measurement scope as NOTIFY.
    printf(
        "[LEGACY-TIMING] tot=%lld tk | sendLoop=%lld (atomicSum=%lld copySum=%lld) "
        "gridbar+notify=%lld recvWait=%lld tk\n",
        _pR - _loopT0, _loopT1 - _loopT0, _atomAcc, _copyAcc, _pN - _loopT1, _pR - _pN);
  }
#endif
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
  CrossDeviceBarrierIntraNodeKernel(args, crossDeviceBarrierFlag);
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
