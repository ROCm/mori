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

  if (globalThdId == 0) atomicAdd(args.crossDeviceBarrierFlag, 1);

  uint64_t* localBarrierPtr = args.crossDeviceBarrierMemObj->template GetAs<uint64_t*>();
  if (thdId < args.config.worldSize) {
    // Backoff in the cross-device wait: the empty tight spin livelocks the cco/xGMI
    // fabric under CNT2's timing and never re-observes the peer's flag write ->
    // combine hangs (plain's slower timing happens to dodge it). s_sleep throttles
    // the poll (matches GridBarrier's spin) and lets the peer flag become visible.
    while (core::AtomicLoadRelaxedSystem(localBarrierPtr + thdId) != crossDeviceBarrierFlag) {
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
}

/* ---------------------------------------------------------------------------------------------- */
/*                                    EpDispatchIntraNodeKernel                                   */
/* ---------------------------------------------------------------------------------------------- */

/* ====================== NOTIFY (CNT2) dispatch kernel ======================
 * Signal-based two-pass slot pre-assignment (NO symmetric cross-device barrier;
 * that barrier, not the per-token atomic, was why the old NOTIFY was slow):
 *   (A) CNT2 count committed tokens per dest PE into localCnt (LDS histogram);
 *   (B) exchange the count matrix via parity-encoded per-cell signals + prefix sum;
 *   (C) send payload with a LOCAL (agent-scope) slot atomic inside the region;
 *   (D) signal-based completion (RELEASE fence + per-destPe signal; ACQUIRE poll).
 * All blocks are co-resident at the tuned geometry (block_num <= CU), which the
 * GridBarrier and the parity/signal protocols rely on. Payload copy uses TDM when
 * MORI_DISP_TDM is compiled in (gfx1250), else WarpCopy.
 * ========================================================================== */
template <typename T, bool EnableStdMoE = false>
__device__ void EpDispatchIntraNodeNotifyKernel_body(EpDispatchCombineArgs<T> args) {
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
  // Per-warp LDS tile for TDM payload staging (dynamic shared, sized
  // warpNum*hiddenDim*sizeof(T) by the launcher). gfx1250 has 320KB LDS/CU, so a
  // 14KB bf16 tile (hidden=7168) lets ~22 warps/CU stay resident.
  extern __shared__ char _tdmDispSmem[];
#endif

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
#if defined(MORI_DISP_TIMING)
    // Diagnostic phase breakdown (block0/warp0 only, single-CU timeline; ratios are
    // clock-frequency independent). Answers "where does dispatch time go" -> whether
    // the count phase (Pass A / CNT2 target) is worth optimizing. See count-opt §8.13.
    long long _pt[9];
    // rank0 only: avoid 4 ranks interleaving printf into garbled lines.
    const bool _ptOn = (myPe == 0 && blockIdx.x == 0 && thdId == 0);
#define _PTS(i) do { if (_ptOn) _pt[i] = clock64(); } while (0)
#else
#define _PTS(i) do {} while (0)
#endif
    for (int d = globalThdId; d < npes; d += totalThreads) localCnt[d] = 0;
    _PTS(0);
    GridBarrier(args.dispatchGridBarrier + 0);
    _PTS(1);  // <- bar0 (zero localCnt + GridBarrier)

    // ---- Pass A (CNT2): count committed (token -> destPe) ----
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
      __syncthreads();
      for (int p = thdId; p < npes; p += blockDim.x) {
        index_t v = smemCnt[p];
        if (v) atomicAdd(&localCnt[p], v);
      }
    }
    _PTS(2);  // <- passA (count committed tokens; CNT2 optimizes THIS phase)
    GridBarrier(args.dispatchGridBarrier + 2);
    _PTS(3);  // <- bar2 (GridBarrier)

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
    _PTS(4);  // <- exch (block0 issues all npes*npes cross-device count stores)
    // ---- Prefix: base[d] = sum_{s<myPe} M[s][d]; total recv = sum_s M[s][myPe].
    // Poll each needed cell until its parity matches this launch, then decode.
    // block0-only, one thread per dest column d (d<npes fits in block0's blockDim
    // for all intra-node world sizes, npes<=64). The total (sum over all sources
    // to me) is computed IN PARALLEL across those npes threads via an LDS reduce,
    // instead of a single thread serially polling npes cells (that serial poll was
    // the EP64 bottleneck: 1 thread doing npes cross-device reads back-to-back). ----
    if (blockIdx.x == 0) {
      // A __syncthreads is REQUIRED here: Exchange (above) reads localCnt[d] to
      // build its cross-device store value, but different block0 threads read
      // localCnt than the ones that zero it below. For npes*npes <= warpSize (EP4)
      // warp reconvergence masks this, but for EP>4 the exchange loop spans several
      // warps and a fast warp could zero localCnt before another warp reads it.
      __syncthreads();
      __shared__ index_t s_tot;
      if (thdId == 0) s_tot = 0;
      __syncthreads();
      if (thdId < npes) {
        int d = thdId;
        index_t b = 0;
        for (int s = 0; s < myPe; ++s) {
          index_t v;
          while (((v = core::AtomicLoadRelaxedSystem(&matLocal[s * npes + d])) & 1) != parity ||
                 v == 0) {
            __builtin_amdgcn_s_sleep(1);  // backoff: tight system-scope spin is slow
          }                               // to re-observe peer stores on gfx1250 (see
          b += (v >> 1) - 1;              // combine WAIT B fix, count-opt §8.2)
        }
        baseArr[d] = b;
        localCnt[d] = 0;
        // Parallel total: thread d contributes source d's count destined to me
        // (cell M[d][myPe]); the LDS atomicAdd reduces across all npes threads.
        index_t vt;
        while (((vt = core::AtomicLoadRelaxedSystem(&matLocal[d * npes + myPe])) & 1) != parity ||
               vt == 0) {
          __builtin_amdgcn_s_sleep(1);
        }
        atomicAdd(&s_tot, (vt >> 1) - 1);
      }
      __syncthreads();
      if (thdId == 0) *args.totalRecvTokenNum = s_tot;
    }
    _PTS(5);  // <- prefix (poll parity cells, prefix sum + parallel total)
    GridBarrier(args.dispatchGridBarrier + 3);
    _PTS(6);  // <- bar3 (GridBarrier)

    // ---- Pass B: send payload; LOCAL slot atomic inside [baseArr[destPe], ...) ----
    if (args.tokenIndices && args.inpTokenBuf && !args.replayMode) {
#if defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
#ifndef MORI_DISP_TDM_NCHUNK
#define MORI_DISP_TDM_NCHUNK 2   // chunks/token: NCHUNK=2 => 7KB tile => wpb32 fits LDS
#endif
      const int _topk = config.numExpertPerToken;
      const int _Npair = args.curRankNumToken * _topk;
      const int _nInt = (int)(hiddenDim * sizeof(T) / 4);
      const int _NCH = MORI_DISP_TDM_NCHUNK;
      if ((_nInt % (64 * _NCH)) == 0) {
        // ===== CONTINUOUS CROSS-TOKEN TDM PIPELINE =====
        // Flatten every committed token's payload into one stream of chunks and run a
        // single load/store pipeline across the WHOLE stream (not per-token). Each
        // iteration issues load(C_t) + store(C_{t-1}) BEFORE one s_wait_tensorcnt(0),
        // so the two drain together (overlap) -- and the overlap now spans token
        // boundaries, removing the per-token drain bubble the old code had. Uses the
        // safe wait(0) discipline (never relies on tensorcnt op ordering): C_{t-1} was
        // loaded and fully drained in the previous iteration before its store here.
        // Routing + metadata (atomic slot, weights/indices/scales) is done inside the
        // producer when a token's first chunk is emitted -- warp-collective, called
        // uniformly by all lanes.
        // Chunk tile = 1D (dim1=1) contiguous burst. On XGMI a 2D tile (dim0=28) is
        // fragmented into tiny per-row descriptors -> ~3x slower cross-card (measured
        // standalone: 1D 1x3584 = 1355 GB/s vs 2D 56x64 = 424 GB/s dev0->dev1). Local
        // HBM hides it (both ~4 TB/s), but dispatch is cross-card, so use 1D.
        const int _ci = _nInt / _NCH;            // int words per chunk
        const int _D1 = 1, _D0 = _ci;            // 1D contiguous burst per chunk (xGMI-efficient)
        int* _bbase = reinterpret_cast<int*>(_tdmDispSmem) + (size_t)warpId * _nInt;
        int* _buf[2] = {_bbase, _bbase + _ci};   // ping-pong of chunk-sized tiles
        typedef int _v4 __attribute__((ext_vector_type(4)));
        typedef int _v8 __attribute__((ext_vector_type(8)));
        gfx1250_TDM_GROUP1 _g1; _g1.dataSize(2);
        _g1.tensorDim0(_D0); _g1.tensorDim1(_D1);
        _g1.tensorDim0Stride(_D0); _g1.tensorDim1Stride(_D1);
        _g1.tileDim0(_D0); _g1.tileDim1(_D1);
        gfx1250_TDM_GROUP0 _g0;
        _v4 _z4{0, 0, 0, 0}; _v8 _z8{0, 0, 0, 0, 0, 0, 0, 0};

        // producer state (all lanes hold identical scalars; control flow warp-uniform)
        int _i = globalWarpId, _chunk = 0; bool _have = false;
        const int* _sInt = nullptr; int* _dInt = nullptr;
        auto produce = [&](const int*& cs, int*& cd) -> bool {
          if (_have && _chunk < _NCH) {          // still emitting current token's chunks
            cs = _sInt + _chunk * _ci; cd = _dInt + _chunk * _ci; ++_chunk; return true;
          }
          for (; _i < _Npair; _i += globalWarpNum) {
            index_t destExpert = args.tokenIndices[_i];
            if (destExpert < 0) continue;
            index_t destPe = destExpert / config.numExpertPerRank;
            if (destPe < 0 || destPe >= config.worldSize) continue;
            index_t srcTokId = _i / _topk;
            int condition = 0;
            if (laneId < (_i % _topk)) {
              index_t otherExpert = args.tokenIndices[srcTokId * _topk + laneId];
              condition = (otherExpert >= 0) && (destPe == (otherExpert / config.numExpertPerRank));
            }
            if (__any(condition)) continue;      // dedup: earlier slot already hit this destPe

            index_t destTokId = 0;
            if (laneId == 0) {
              index_t slot = atomicAdd(&localCnt[destPe], 1);
              destTokId = baseArr[destPe] + slot;
              args.dispDestTokIdMap[_i] = FlatTokenIndex(config, destPe, destTokId);
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
            _sInt = reinterpret_cast<const int*>(args.inpTokenBuf + (size_t)srcTokId * hiddenDim);
            _dInt = reinterpret_cast<int*>(
                args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) +
                (size_t)destTokId * hiddenDim);
            _have = true; _chunk = 1;            // emit chunk 0 now; next call emits chunk 1..
            _i += globalWarpNum;                 // step cursor past this token
            cs = _sInt; cd = _dInt; return true;
          }
          _have = false; return false;
        };

        const int* _cs; int* _cd;
        if (produce(_cs, _cd)) {
          _g0.ldsAddr((uintptr_t)_buf[0]); _g0.globalAddr((uintptr_t)_cs);
          __builtin_amdgcn_tensor_load_to_lds(_g0.m_bitfield, _g1.m_bitfield, _z4, _z4, _z8, 0);
          __builtin_amdgcn_s_wait_tensorcnt(0);  // prologue: C_0 resident (only bubble)
          __builtin_amdgcn_wave_barrier();
          int _pb = 0; int* _pd = _cd;           // pending chunk: buffer + dest
          for (;;) {
            const int* _ns; int* _nd;
            bool _more = produce(_ns, _nd);
            if (_more) {                          // load C_t into the other buffer
              _g0.ldsAddr((uintptr_t)_buf[_pb ^ 1]); _g0.globalAddr((uintptr_t)_ns);
              __builtin_amdgcn_tensor_load_to_lds(_g0.m_bitfield, _g1.m_bitfield, _z4, _z4, _z8, 0);
            }
            _g0.ldsAddr((uintptr_t)_buf[_pb]); _g0.globalAddr((uintptr_t)_pd);  // store C_{t-1}
            __builtin_amdgcn_tensor_store_from_lds(_g0.m_bitfield, _g1.m_bitfield, _z4, _z4, _z8, 0);
            __builtin_amdgcn_s_wait_tensorcnt(0); // load(C_t) + store(C_{t-1}) drain together
            __builtin_amdgcn_wave_barrier();
            if (!_more) break;
            _pb ^= 1; _pd = _nd;
          }
        }
      } else {
        // geometry not TDM-friendly (int-word count not a multiple of 64*NCHUNK):
        // fall back to the per-token WarpCopy loop.
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
          core::WarpCopy<T, 8>(
              args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) +
                  (size_t)destTokId * hiddenDim,
              args.inpTokenBuf + (size_t)srcTokId * hiddenDim, hiddenDim);
        }
      }
#else
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
        core::WarpCopy<T, 8>(
            args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) +
                (size_t)destTokId * hiddenDim,
            args.inpTokenBuf + (size_t)srcTokId * hiddenDim, hiddenDim);
      }
#endif
    }
    _PTS(7);  // <- passB (payload copy: WarpCopy/TDM + per-token local slot atomic)

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
#if defined(MORI_DISP_TIMING)
    _PTS(8);  // <- compl (cross-device signal send + recv)
    if (_ptOn) {
      long long bar0 = _pt[1] - _pt[0], passA = _pt[2] - _pt[1];
      long long bar2 = _pt[3] - _pt[2], exch = _pt[4] - _pt[3];
      long long prefix = _pt[5] - _pt[4], bar3 = _pt[6] - _pt[5];
      long long passB = _pt[7] - _pt[6], cmpl = _pt[8] - _pt[7];
      long long tot = _pt[8] - _pt[0];
      double t = (double)(tot ? tot : 1);
      printf("[PHASE] cyc: bar0=%lld passA=%lld bar2=%lld exch=%lld prefix=%lld "
             "bar3=%lld passB=%lld compl=%lld tot=%lld\n",
             bar0, passA, bar2, exch, prefix, bar3, passB, cmpl, tot);
      printf("[PHASE] pct: bar0=%.2f passA(CNT2)=%.2f bar2=%.2f exch=%.2f "
             "prefix=%.2f bar3=%.2f passB=%.2f compl=%.2f\n",
             100.0 * bar0 / t, 100.0 * passA / t, 100.0 * bar2 / t,
             100.0 * exch / t, 100.0 * prefix / t, 100.0 * bar3 / t,
             100.0 * passB / t, 100.0 * cmpl / t);
    }
#endif
#undef _PTS
#ifdef ENABLE_STANDARD_MOE_ADAPT
    if constexpr (EnableStdMoE) {
      InvokeConvertDispatchOutput<T>(args, myPe);
    }
#endif
    return;
  }
}

/* ======================== LEGACY dispatch kernel ==========================
 * Cross-GPU returning atomic slot assignment: each committed (token, expert) does
 * a SYSTEM-scope fetch_add on the destination PE's dispTokOffset to claim its slot
 * inline during the send (no separate count pass). Payload copy is ALWAYS WarpCopy
 * -- this kernel is the baseline used only to compare against NOTIFY; it never uses
 * TDM (the TDM payload path lives in EpDispatchIntraNodeNotifyKernel_body).
 * ========================================================================== */
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
  // TDM (1D whole-token burst) payload for the LEGACY remote-atomic path. Per-warp LDS
  // tile (launcher sizes >= warpNum*hiddenDim*sizeof(T)). Order per token: issue async
  // LOAD src->tile, run the REMOTE fetch_add slot atomic (its ~us cross-GPU round-trip
  // overlaps the local HBM load), wait the load, then STORE tile->peer. Hides the
  // atomic latency under the load; 1D avoids the xGMI fragmentation of a 2D tile.
  extern __shared__ char _tdmLegacySmem[];
  T* _tdmTile = reinterpret_cast<T*>(_tdmLegacySmem) + (size_t)warpId * hiddenDim;
  const gfx1250_TDM_GROUP1 _tdmG1 = TdmShape<T>(static_cast<int>(hiddenDim));
#endif

#if defined(MORI_DISP_TIMING)
  // LEGACY phase breakdown (block0/warp0/rank0), same clock64 basis as the NOTIFY
  // kernel so the two are directly comparable. sendloop = remote-atomic slot assign +
  // payload WarpCopy (fused; the NOTIFY analogue is passA(count)+passB(copy) with a
  // LOCAL atomic). atomClk = warp0's cumulative REMOTE fetch_add completion time -- if
  // atomClk << sendloop the cross-GPU atomic is overlapped behind the copy, so NOTIFY
  // swapping it for a local atomic cannot buy wall-clock (see count-opt §8.4/§8.7).
  long long _lpt[4];
  long long _atomClk = 0;
  const bool _lptOn = (myPe == 0 && blockIdx.x == 0 && thdId == 0);
#define _LPTS(i) do { if (_lptOn) _lpt[i] = clock64(); } while (0)
#else
#define _LPTS(i) do {} while (0)
#endif

  IF_ENABLE_PROFILER(
      INTRANODE_PROFILER_INIT_CONTEXT(profiler, args.profilerConfig, globalWarpId, laneId));
  MORI_TRACE_SEQ(seq, profiler);
  MORI_TRACE_NEXT(seq, Slot::DispatchSendTokens);

  _LPTS(0);  // <- start of Phase1 send loop
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

#if defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
        // LOAD first: prefetch this token's payload into the per-warp LDS tile BEFORE the
        // remote fetch_add below, so the atomic's cross-GPU round-trip overlaps the load.
        TdmIssueLoad<T>(_tdmTile, args.inpTokenBuf + (size_t)srcTokId * hiddenDim, _tdmG1);
#endif

        {
          // Fine-grained timing: slot assignment = remote returning atomic on
          // dispTokOffset[destPe] + the two remote metadata writes.
          MORI_TRACE_SPAN(profiler, Slot::DispSlotAssign);
          if (laneId == 0) {
#if defined(MORI_DISP_TIMING)
            long long _a0 = _lptOn ? clock64() : 0;  // warp0-lane0: measure the atomic
#endif
            // decide token id in dest pe
            // Cross-GPU slot allocation: the offset counter lives on the destination
            // PE, so the fetch-add MUST be SYSTEM-scoped. Plain atomicAdd is agent
            // (device) scope and is NOT atomic across GPUs over the cco/LSA fabric,
            // so concurrent senders can get the same destTokId -> slot collision ->
            // corrupt dispatch (map/payload disagree). Matches v2's system-scope
            // atomic_add_global.
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
            assert(destTokId < config.MaxNumTokensToRecv() &&
                   "Total recv token overflow: increase maxTotalRecvTokens");
            atomicAdd(args.destPeTokenCounter + destPe, 1);
            args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] =
                FlatTokenIndex(config, myPe, srcTokId);
#if defined(MORI_DISP_TIMING)
            // End clock AFTER the destTokId-dependent stores -> captures the remote
            // fetch_add's true round-trip completion (natural data dependency).
            if (_lptOn) _atomClk += clock64() - _a0;
#endif
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
#if defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
        TdmIssueLoad<T>(_tdmTile, args.inpTokenBuf + (size_t)srcTokId * hiddenDim, _tdmG1);
#endif
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
        // STORE last: the LOAD was issued before the remote atomic (overlapping its
        // round-trip). Wait the load, then issue the 1D burst store tile->peer.
        __builtin_amdgcn_s_wait_tensorcnt(0);
        __builtin_amdgcn_wave_barrier();
        TdmIssueStore<T>(
            args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) + destTokOffset,
            _tdmTile, _tdmG1);
        __builtin_amdgcn_s_wait_tensorcnt(0);
#else
        // LEGACY WarpCopy baseline (used to compare against NOTIFY).
        core::WarpCopy<T, 8>(
            args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) + destTokOffset,
            args.inpTokenBuf + srcTokOffset, hiddenDim);
#endif
      }
    }
  }
  __syncthreads();
  _LPTS(1);  // <- sendloop done (remote-atomic slot assign + payload copy + meta)
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
  _LPTS(2);  // <- grid barrier wait + per-peer release-signal (sigsend)

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
  _LPTS(3);  // <- Phase2 recv (wait peer signals + acquire)
  if (_lptOn) {
    long long sendloop = _lpt[1] - _lpt[0], sigsend = _lpt[2] - _lpt[1];
    long long sigrecv = _lpt[3] - _lpt[2], tot = _lpt[3] - _lpt[0];
    double t = (double)(tot ? tot : 1);
    printf("[LPHASE] cyc: sendloop=%lld sigsend=%lld sigrecv=%lld tot=%lld atomClk(warp0)=%lld\n",
           sendloop, sigsend, sigrecv, tot, _atomClk);
    printf("[LPHASE] pct: sendloop=%.2f sigsend=%.2f sigrecv=%.2f | atomClk/sendloop=%.2f%%\n",
           100.0 * sendloop / t, 100.0 * sigsend / t, 100.0 * sigrecv / t,
           sendloop ? 100.0 * (double)_atomClk / (double)sendloop : 0.0);
  }
#endif
#undef _LPTS

#ifdef ENABLE_STANDARD_MOE_ADAPT
  if constexpr (EnableStdMoE) {
    InvokeConvertDispatchOutput<T>(args, myPe);
  }
#endif
}

/* ---------------------------------------------------------------------------------------------- */
/*                     EpDispatchIntraNodeBatchKernel (block-local batched slot)                   */
/* ---------------------------------------------------------------------------------------------- */
// Block-local exact-count + batched remote reservation. Avoids BOTH:
//  (1) NOTIFY's remote count-matrix exchange/prefix + grid barriers -- everything here is
//      block-local, so phase transitions are just __syncthreads (no grid barrier), and
//  (2) legacy's per-token REMOTE fetch_add -- here each block does ONE remote fetch_add(N)
//      per destPe, N = the block's EXACT committed count (counted locally first) so there
//      is no over-reservation / no holes (the concern that rules out a blind batch atomic).
// Per token the slot is a fast LDS atomic; payload via clean 1D TDM. Remote atomics drop
// from O(committed tokens) to npes*numBlocks. Completion tail identical to legacy.
template <typename T, bool EnableStdMoE = false>
__device__ void EpDispatchIntraNodeBatchKernel_body(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int globalWarpNum = gridDim.x * warpNum;
  int myPe = config.rank;
  int npes = config.worldSize;
  size_t hiddenDim = config.HiddenDimSz();
  const int topk = config.numExpertPerToken;
  const int Npair = args.curRankNumToken * topk;

#if defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
  extern __shared__ char _tdmBatchSmem[];
  T* _tdmTile = reinterpret_cast<T*>(_tdmBatchSmem) + (size_t)warpId * hiddenDim;
  const gfx1250_TDM_GROUP1 _tdmG1 = TdmShape<T>(static_cast<int>(hiddenDim));
#endif

  constexpr int kMaxNpes = MAX_GPUS_PER_NODE;
  __shared__ index_t s_N[kMaxNpes];     // block's committed count per destPe
  __shared__ index_t s_base[kMaxNpes];  // reserved contiguous base slot on destPe
  __shared__ index_t s_run[kMaxNpes];   // block-local running distribution index

#if defined(MORI_DISP_TIMING)
  long long _pt[6];
  const bool _ptOn = (myPe == 0 && blockIdx.x == 0 && thdId == 0);
#define _BPTS(i) do { if (_ptOn) _pt[i] = clock64(); } while (0)
#else
#define _BPTS(i) do {} while (0)
#endif
  _BPTS(0);

  // ---- Phase 1: block-local count committed tokens per destPe (+ drop sentinels) ----
  for (int p = thdId; p < npes; p += blockDim.x) { s_N[p] = 0; s_run[p] = 0; }
  __syncthreads();
  if (args.tokenIndices && args.inpTokenBuf && !args.replayMode) {
#if defined(MORI_DISP_PERTOK) && defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
    // PER-TOKEN count: one warp per token. Lanes 0..topk-1 read the token's experts;
    // dedup to distinct destPe via ONE __match_any_sync (CNT2 rule, lowest lane kept);
    // count each kept lane's destPe; write drop-sentinels for the non-kept slots. Same
    // committed set as Phase 3 (identical keep rule) so s_N matches the store count.
    for (int tok = globalWarpId; tok < args.curRankNumToken; tok += globalWarpNum) {
      index_t myExpert = (laneId < topk) ? args.tokenIndices[(size_t)tok * topk + laneId] : (index_t)-1;
      int myDestPe = -1;
      if (myExpert >= 0) { int d = (int)(myExpert / config.numExpertPerRank);
                           if (d >= 0 && d < config.worldSize) myDestPe = d; }
      unsigned mv = (myDestPe >= 0) ? (unsigned)myDestPe : 0xFFFFFFFFu;
      unsigned long long grp = __match_any_sync(0xFFFFFFFFFFFFFFFFull, mv);
      int keep = (myDestPe >= 0 && laneId == (__ffsll((long long)grp) - 1)) ? 1 : 0;
      if (laneId < topk && !keep)
        args.dispDestTokIdMap[(size_t)tok * topk + laneId] = FlatTokenIndex(config, config.worldSize, 0);
      if (keep) atomicAdd(&s_N[myDestPe], 1);
    }
#else
    for (int i = globalWarpId; i < Npair; i += globalWarpNum) {
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
      index_t srcTokId = i / topk;
      int condition = 0;
      if (laneId < (i % topk)) {
        index_t otherExpert = args.tokenIndices[srcTokId * topk + laneId];
        condition = (otherExpert >= 0) && (destPe == (otherExpert / config.numExpertPerRank));
      }
      if (__any(condition)) {
        if (laneId == 0) args.dispDestTokIdMap[i] = FlatTokenIndex(config, config.worldSize, 0);
        continue;
      }
      if (laneId == 0) atomicAdd(&s_N[destPe], 1);
    }
#endif
  }
  __syncthreads();

  _BPTS(1);  // <- phase1 count (block-local histogram)
  // ---- Phase 2: reserve N contiguous slots per destPe with ONE remote atomic each ----
  for (int p = thdId; p < npes; p += blockDim.x) {
    if (s_N[p] > 0) {
      s_base[p] = __hip_atomic_fetch_add(
          args.dispTokOffsetMemObj->template GetAs<index_t*>(p), s_N[p], __ATOMIC_RELAXED,
          __HIP_MEMORY_SCOPE_SYSTEM);
      atomicAdd(args.destPeTokenCounter + p, s_N[p]);
    }
  }
  __syncthreads();

  _BPTS(2);  // <- phase2 reserve (npes remote atomicAdd(N))
  // ---- Phase 3: distribute LOCAL slots + send payload (clean 1D TDM, no remote atomic) ----
  if (args.tokenIndices && args.inpTokenBuf && !args.replayMode) {
#if defined(MORI_DISP_PERTOK) && defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
    // PER-TOKEN 1 load : N store. One warp per token: LOAD the token ONCE into the LDS
    // tile, then STORE it to each DISTINCT destPe from that same tile (stores back-to-back,
    // load amortized -> higher store duty, like all-to-all's 1-load:N-store). Slot is a
    // fast LDS atomic; dedup via __match_any_sync (same keep rule as Phase 1).
    for (int tok = globalWarpId; tok < args.curRankNumToken; tok += globalWarpNum) {
      index_t myExpert = (laneId < topk) ? args.tokenIndices[(size_t)tok * topk + laneId] : (index_t)-1;
      int myDestPe = -1;
      if (myExpert >= 0) { int d = (int)(myExpert / config.numExpertPerRank);
                           if (d >= 0 && d < config.worldSize) myDestPe = d; }
      unsigned mv = (myDestPe >= 0) ? (unsigned)myDestPe : 0xFFFFFFFFu;
      unsigned long long grp = __match_any_sync(0xFFFFFFFFFFFFFFFFull, mv);
      int keep = (myDestPe >= 0 && laneId == (__ffsll((long long)grp) - 1)) ? 1 : 0;
      if (!__any(keep)) continue;   // token routed nowhere valid -> skip (no load)

      TdmIssueLoad<T>(_tdmTile, args.inpTokenBuf + (size_t)tok * hiddenDim, _tdmG1);
      bool loadWaited = false;
      for (int l = 0; l < topk; ++l) {
        if (!__shfl(keep, l)) continue;         // fixed l -> uniform shfl
        int d = __shfl(myDestPe, l);
        index_t destTokId = 0;
        if (laneId == 0) {
          index_t j = atomicAdd(&s_run[d], 1);
          destTokId = s_base[d] + j;
          args.dispDestTokIdMap[(size_t)tok * topk + l] = FlatTokenIndex(config, d, destTokId);
          args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(d)[destTokId] =
              FlatTokenIndex(config, myPe, tok);
        }
        destTokId = __shfl(destTokId, 0);
        if (laneId < config.numExpertPerToken) {
          if (args.weightsBuf) {
            args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(
                d)[destTokId * config.numExpertPerToken + laneId] =
                args.weightsBuf[(size_t)tok * config.numExpertPerToken + laneId];
          }
          args.shmemOutIndicesMemObj->template GetAs<index_t*>(
              d)[destTokId * config.numExpertPerToken + laneId] =
              args.tokenIndices[(size_t)tok * config.numExpertPerToken + laneId];
        }
        if (args.scalesBuf && (config.scaleDim > 0) && (config.scaleTypeSize > 0)) {
          size_t dso = (size_t)destTokId * config.scaleDim * config.scaleTypeSize;
          size_t sso = (size_t)tok * config.scaleDim * config.scaleTypeSize;
          core::WarpCopy(args.shmemOutScalesMemObj->template GetAs<uint8_t*>(d) + dso,
                         args.scalesBuf + sso, config.scaleDim * config.scaleTypeSize);
        }
        if (!loadWaited) { __builtin_amdgcn_s_wait_tensorcnt(0); loadWaited = true; }  // load done -> tile valid
        TdmIssueStore<T>(args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(d) +
                             (size_t)destTokId * hiddenDim,
                         _tdmTile, _tdmG1);
      }
      __builtin_amdgcn_s_wait_tensorcnt(0);   // drain all N stores before reusing tile next token
    }
#else
    for (int i = globalWarpId; i < Npair; i += globalWarpNum) {
      index_t destExpert = args.tokenIndices[i];
      if (destExpert < 0) continue;
      index_t destPe = destExpert / config.numExpertPerRank;
      if (destPe < 0 || destPe >= config.worldSize) continue;
      index_t srcTokId = i / topk;
      int condition = 0;
      if (laneId < (i % topk)) {
        index_t otherExpert = args.tokenIndices[srcTokId * topk + laneId];
        condition = (otherExpert >= 0) && (destPe == (otherExpert / config.numExpertPerRank));
      }
      if (__any(condition)) continue;

#if defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
      // LOAD first (overlaps the local LDS slot atomic + metadata below).
      TdmIssueLoad<T>(_tdmTile, args.inpTokenBuf + (size_t)srcTokId * hiddenDim, _tdmG1);
#endif
      index_t destTokId = 0;
      if (laneId == 0) {
        index_t j = atomicAdd(&s_run[destPe], 1);       // fast LDS slot (was remote)
        destTokId = s_base[destPe] + j;
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
      size_t destTokOffset = (size_t)destTokId * hiddenDim;
#if defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
      __builtin_amdgcn_s_wait_tensorcnt(0);
      __builtin_amdgcn_wave_barrier();
      TdmIssueStore<T>(
          args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) + destTokOffset,
          _tdmTile, _tdmG1);
      __builtin_amdgcn_s_wait_tensorcnt(0);
#else
      core::WarpCopy<T, 8>(
          args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) + destTokOffset,
          args.inpTokenBuf + (size_t)srcTokId * hiddenDim, hiddenDim);
#endif
    }
#endif  // MORI_DISP_PERTOK
  }
  __syncthreads();
  _BPTS(3);  // <- phase3 payload copy (Part B: 1D TDM)

  // ---- Completion (identical to legacy): all blocks arrive, then per-peer release-signal ----
  if (thdId == 0) atomicAdd(args.dispatchGridBarrier, 1);
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, gridDim.x);
      __hip_atomic_store(args.dispatchGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
      core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
    }
  }
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      index_t* signal = recvTokenNums + destPe;
      index_t recvTokenNum = shmem::ShmemInt32WaitUntilGreaterThan(signal, 0) - 1;
      __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
      core::AtomicStoreRelaxedSystem(signal, 0);
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);
      args.destPeTokenCounter[destPe] = 0;
    }
    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
    }
  }
#if defined(MORI_DISP_TIMING)
  _BPTS(4);  // <- completion (cross-device signal send + recv)
  if (_ptOn) {
    long long p1 = _pt[1] - _pt[0], p2 = _pt[2] - _pt[1];
    long long pB = _pt[3] - _pt[2], pc = _pt[4] - _pt[3];
    long long tot = _pt[4] - _pt[0];
    double t = (double)(tot ? tot : 1);
    printf("[BPHASE] cyc: count=%lld reserve=%lld partB=%lld compl=%lld tot=%lld\n", p1, p2, pB, pc, tot);
    printf("[BPHASE] pct: count=%.2f reserve=%.2f partB=%.2f compl=%.2f\n",
           100.0 * p1 / t, 100.0 * p2 / t, 100.0 * pB / t, 100.0 * pc / t);
  }
#endif
#undef _BPTS
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

template <typename T, bool EnableStdMoE = false>
__global__ void EpDispatchIntraNodeNotifyKernel(EpDispatchCombineArgs<T> args) {
  EpDispatchIntraNodeNotifyKernel_body<T, EnableStdMoE>(args);
}

template <typename T, bool EnableStdMoE = false>
__global__ void EpDispatchIntraNodeBatchKernel(EpDispatchCombineArgs<T> args) {
  EpDispatchIntraNodeBatchKernel_body<T, EnableStdMoE>(args);
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
