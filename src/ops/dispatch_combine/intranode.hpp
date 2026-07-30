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
// 2D meta tile (dataSize=2 -> 4B elems). Both dims must be >= 2 (no 1xN wedge on gfx1250).
__device__ __forceinline__ gfx1250_TDM_GROUP1 TdmShape2D(int dim0, int dim1) {
  gfx1250_TDM_GROUP1 g1;
  g1.dataSize(2);
  g1.tensorDim0(dim0);
  g1.tensorDim1(dim1);
  g1.tensorDim0Stride(dim0);
  g1.tensorDim1Stride(dim1);
  g1.tileDim0(dim0);
  g1.tileDim1(dim1);
  return g1;
}
// gfx1250 TDM tensor_load_to_lds fast-dim row must be >= 128B (evidence: _ct_real.sh TW=112
// bf16 row=224B -> ~500 GB/s vs TW=128 row=256B -> ~1500; for dataSize=2 meta ints that is
// tensorDim0>=32). Among legal (d0,d1) factor pairs, picks the one CLOSEST TO SQUARE (min
// |d0-d1|) rather than the smallest d0 -- only square shapes (10x10..64x64) are validated per
// TDM_USAGE.md, and a smallest-d0 tie-break always collapsed to the 128B floor regardless of
// nElems (e.g. scale's 4096 landed on 32x128 instead of the square, wider-row 64x64). Ties
// keep the larger d0 (wider row). Returns dim1=0 if nElems cannot form a legal tile (caller
// direct-writes).
__device__ __forceinline__ gfx1250_TDM_GROUP1 TdmShapeMeta(int nElems, int preferDim0) {
  constexpr int kMinFastDim = 32;  // 32 x 4B = 128B minimum LOAD row
  int bestD0 = 0, bestD1 = 0;
  auto tryPair = [&](int d0, int d1) {
    if (d0 < 2 || d1 < 2 || d0 * d1 != nElems || d0 < kMinFastDim) return;
    if (!bestD0) { bestD0 = d0; bestD1 = d1; return; }
    int curGap = (bestD0 > bestD1) ? (bestD0 - bestD1) : (bestD1 - bestD0);
    int newGap = (d0 > d1) ? (d0 - d1) : (d1 - d0);
    if (newGap < curGap || (newGap == curGap && d0 > bestD0)) {
      bestD0 = d0;
      bestD1 = d1;
    }
  };
  if (preferDim0 >= kMinFastDim) tryPair(preferDim0, nElems / preferDim0);
  tryPair(kMinFastDim, nElems / kMinFastDim);
  // Search outward from sqrt(nElems) instead of the O(nElems) linear scan this replaced: for
  // d0 <= sqrt(nElems), gap=|d0-nElems/d0| increases monotonically as d0 decreases, and for
  // d0 > sqrt(nElems) it increases monotonically as d0 increases -- so the first divisor found
  // scanning outward in each direction is that direction's best, and tryPair's existing gap
  // comparison (unchanged above) picks the same global winner an exhaustive scan would. Verified
  // byte-for-byte equivalent to the old scan for nElems in [2, 200000] x preferDim0 in {8,16,32,64}
  // (host-side brute-force comparison, see _verify_shapemeta.cpp).
  int sqLo = 1, sqHi = nElems;
  while (sqLo < sqHi) {
    int mid = sqLo + (sqHi - sqLo + 1) / 2;
    if ((long long)mid * mid <= nElems) sqLo = mid; else sqHi = mid - 1;
  }
  int lo = sqLo;
  while (lo >= kMinFastDim) {
    if (nElems % lo == 0) { tryPair(lo, nElems / lo); break; }
    --lo;
  }
  int hi = sqLo + 1;
  if (hi < kMinFastDim) hi = kMinFastDim;
  while (hi <= nElems / 2) {
    if (nElems % hi == 0) { tryPair(hi, nElems / hi); break; }
    ++hi;
  }
  if (!bestD0) return TdmShape2D(2, 2);  // unreachable if TdmMetaTileOk(nElems)
  return TdmShape2D(bestD0, bestD1);
}
// 128B-ALIGNED split for a contiguous run of 4B elements, for the meta path where the run start
// is a remote-atomic-derived slot index and therefore has an arbitrary 128B phase.
//
// TdmShapeMeta picks the factor pair closest to square, which is right when only the tile's total
// size matters. It is WRONG here: scale is 128B/token so its destination (base + ab*128) is always
// 128B-aligned, yet closest-to-square turns cc=58 tokens (1856 elems) into a 58x32 tile whose rows
// are 232B -- neither 128B-wide nor 128B-apart, throwing away the one field that was aligned by
// construction. This instead peels a scalar head so the TDM body starts on a 128B boundary and
// makes every row exactly 32 elems = 128B, so every row start is aligned too.
//
// `phase` is the run start's element offset within its 128B-aligned array base; dst and src share
// it (both are base + ab*K), so one split serves both sides. Returns all-scalar (body=0) when the
// aligned remainder cannot form the >=2 rows a legal tile needs.
struct TdmSplit128 {
  int head;  // leading elements to copy scalar (until 128B-aligned)
  int body;  // elements covered by the TDM tile (whole 128B rows)
  int rows;  // body / 32
};
__device__ __forceinline__ TdmSplit128 TdmAlignSplit128(size_t phase, int nElems) {
  constexpr int P = 32;  // 32 x 4B = 128B
  int head = (int)((P - (phase & (size_t)(P - 1))) & (size_t)(P - 1));
  if (head > nElems) head = nElems;
  int rows = (nElems - head) / P;
  if (rows < 2) return TdmSplit128{nElems, 0, 0};
  return TdmSplit128{head, rows * P, rows};
}

__device__ __forceinline__ bool TdmMetaTileOk(int nElems) {
  constexpr int kMinFastDim = 32;
  if (nElems < kMinFastDim * 2) return false;  // need 32x2=64 elems minimum
  if (nElems % kMinFastDim == 0 && nElems / kMinFastDim >= 2) return true;
  for (int d0 = nElems / 2; d0 >= kMinFastDim; --d0)
    if (nElems % d0 == 0 && nElems / d0 >= 2) return true;
  return false;
}

// Legal whole-run tile geometry by closed form: try tensorDim1 = 8, 4, 2 (largest first, so the row
// stays as narrow as the >=128B floor allows) and take the first exact divisor whose row reaches the
// 32-element floor. Returns 0 when no legal pair exists. Three masks, no search -- TdmShapeMeta's
// closest-to-square divisor scan runs on the GPU and measured 8.5us of pure ALU at EP4-4K, more than
// the remainders it was there to remove.
__device__ __forceinline__ int TdmCheapDim1(int nElems) {
  if ((nElems & 7) == 0 && (nElems >> 3) >= 32) return 8;
  if ((nElems & 3) == 0 && (nElems >> 2) >= 32) return 4;
  if ((nElems & 1) == 0 && (nElems >> 1) >= 32) return 2;
  return 0;
}
// Cover the WHOLE run with ONE tile so it carries no scalar head/tail at all. Its rows are then not
// 128B apart (idx at cc=29 tokens, the measured mode, is 232 elems -> 58x4, a 232B row), which costs
// streaming bandwidth -- but a meta run is at most tokCapM*perTok (~2.2KB per field), so its cost is
// dominated by per-op overhead plus the remainders, not bandwidth: at EP4-4K the peel's remainders
// measured htIdx 9.4us + htWt 14.1us, which the whole-run tile drops to 0.1/0.0.
//
// Measured against the aligned-peel form it replaced, paired (only this changing), noTIMING: +5.9% at
// the default geometry, +11.4%/+5.3% at METASPLIT 4/1, and +11.0%/+9.7% at DBN 48/32 where the run
// hits the tokCapM=70 ceiling and chunks. The gain tracks how many remainders are removed, not run
// size, so there is no crossover to fall back at within the reachable cc range (cc <= tokCapM).
// Runs too short for a legal pair keep the peel: cc*topk < 64 makes TdmCheapDim1 fail, which is why
// this is a fallback rather than an unconditional whole-run tile.
__device__ __forceinline__ TdmSplit128 TdmWholeOrSplit128(size_t phase, int nElems) {
  const TdmSplit128 sp = TdmAlignSplit128(phase, nElems);
  // A run that is already 128B-phased and a whole number of 32-element rows has no remainder to
  // remove, and its aligned rows are 128B wide AND 128B apart -- strictly better, keep it. This is
  // always the case for scale (both sides are base + ab*sBytesM), whose htSc is already ~0.1us.
  if (sp.head == 0 && sp.body == nElems) return sp;
  if (TdmCheapDim1(nElems)) return TdmSplit128{0, nElems, 0};  // rows==0 && body>0 => whole run
  return sp;  // no legal pair: srcmap only reaches one at cc>=64, idx/wt only below cc*topk=64
}
// Warp-cooperative copy of a contiguous run of 4B elements to a PEER, 16B per store where the
// alignment allows it. The cross-GPU write path is priced per TRANSACTION, not per byte -- plain
// stores sustain only ~54 GB/s there (MORI_DISP_METAVEC) against TDM's ~1600 -- so a 4B-at-a-time
// loop pays four times the transactions it needs to.
//
// This exists for srcmap, the one meta field that never gets a TDM body: it is 1 element per slot,
// so a (block, peer) run is cc ~= 58 elements at DBN=64, and a legal tile needs 64 (TdmCheapDim1's
// 32x2), while TdmAlignSplit128's aligned remainder reaches only 1 row of the 2 it needs. So the
// whole run lands in the scalar peel, and htSrc measured 20.1us of a 36.7us metasend (TIMING).
//
// uint4 requires both sides on the same 16B phase. They normally are (both are base + ab), but the
// two bases are independent allocations, so this checks instead of assuming and falls back per run.
template <typename E>
__device__ __forceinline__ void WarpCopyRun4B(E* d, const E* s, int n, int laneId) {
  static_assert(sizeof(E) == 4, "WarpCopyRun4B is for 4B elements");
  if (n <= 0) return;
  if (((((uintptr_t)d) ^ ((uintptr_t)s)) & 15u) == 0u) {
    int pre = (int)((((16u - (((uintptr_t)d) & 15u)) & 15u)) >> 2);
    if (pre > n) pre = n;
    for (int i = laneId; i < pre; i += warpSize) d[i] = s[i];
    const int nv = (n - pre) >> 2;
    uint4* dv = reinterpret_cast<uint4*>(d + pre);
    const uint4* sv = reinterpret_cast<const uint4*>(s + pre);
    for (int i = laneId; i < nv; i += warpSize) dv[i] = sv[i];
    for (int i = pre + (nv << 2) + laneId; i < n; i += warpSize) d[i] = s[i];
    return;
  }
  for (int i = laneId; i < n; i += warpSize) d[i] = s[i];
}
// Shape for a split's TDM body. rows==0 marks a whole-run tile (see TdmWholeOrSplit128).
__device__ __forceinline__ gfx1250_TDM_GROUP1 TdmSplitShape(const TdmSplit128& sp, int nElems) {
  if (sp.rows == 0) {
    const int d1 = TdmCheapDim1(nElems);
    if (d1 <= 0) return TdmShape2D(32, 2);  // unreachable: rows==0 only when TdmCheapDim1 succeeded
    return TdmShape2D(nElems / d1, d1);
  }
  return TdmShape2D(32, sp.rows);
}
}  // namespace moe
}  // namespace mori
#endif

namespace mori {
namespace moe {

#define MAX_GPUS_PER_NODE 8

// Vectorized (16B-coalesced) warp copy of the small per-token scale blob. core::WarpCopy
// only vectorizes runs >= warpSize*16 B and otherwise falls back to BYTE-AT-A-TIME; the
// dispatch scale blob is just scaleDim*scaleTypeSize (=128B here), so it hit the byte path
// -> 128 separate UNCACHED cross-card byte stores per (token,peer), the single biggest
// Part-B metadata cost (bisected: dropping scales alone lifts EP4-4K dispatch 1131->1368).
// This does it as 16B stores (8 lanes -> one coalesced 128B write). dst/src are destTokId/
// tok * blob-size aligned (blob is a multiple of 16B here) so the uint4 path stays aligned.
__device__ __forceinline__ void WarpScaleCopy(uint8_t* dst, const uint8_t* src, int nbytes) {
  int laneId = threadIdx.x & (warpSize - 1);
  int nvec = nbytes >> 4;  // number of 16B chunks
  for (int c = laneId; c < nvec; c += warpSize)
    reinterpret_cast<uint4*>(dst)[c] = reinterpret_cast<const uint4*>(src)[c];
  for (int b = (nvec << 4) + laneId; b < nbytes; b += warpSize) dst[b] = src[b];  // tail bytes
}

// Coalesced warp copy with the LOADS BATCHED 4 deep. A plain `for (i = lane; i < n; i += warpSize)
// dst[i] = src[i]` cannot be software-pipelined: dst and src are both plain pointers, so the
// compiler must assume dst[i] aliases src[i + warpSize], and the copy pays a full HBM round trip per
// 32-element step. Issuing all four loads of a step before any of its stores keeps them in flight.
// The tail uses clamped indices rather than a stride loop so it stays batched too (and never reads
// out of bounds); callers must pass n >= 1.
template <typename V>
__device__ __forceinline__ void WarpBatchCopy4(V* dst, const V* src, int n, int laneId) {
  if (n <= 0) return;
  const int w = warpSize;
  int i = laneId;
  for (; i + 3 * w < n; i += 4 * w) {
    V v0 = src[i], v1 = src[i + w], v2 = src[i + 2 * w], v3 = src[i + 3 * w];
    dst[i] = v0;
    dst[i + w] = v1;
    dst[i + 2 * w] = v2;
    dst[i + 3 * w] = v3;
  }
  const int i1 = i + w, i2 = i + 2 * w;
  V t0 = src[i < n ? i : n - 1], t1 = src[i1 < n ? i1 : n - 1], t2 = src[i2 < n ? i2 : n - 1];
  if (i < n) dst[i] = t0;
  if (i1 < n) dst[i1] = t1;
  if (i2 < n) dst[i2] = t2;
}

#if defined(MORI_DISP_METADIAG)
// [METASHAPE] Geometry of the metadata idx run: run length (cc), which tile kind it took, and the
// run start's 128B phase. Deliberately NOT under MORI_DISP_TIMING -- these are data-flow values, so
// they are identical in the shipping untimed build, and reading them there avoids the clock64 probes
// that were already shown to distort the meta phase. Whether the whole-run tile's rows are 128B
// aligned is decided by d0 = cc*topk/d1 (aligned iff d0 % 32 == 0), so cc is what settles it.
__device__ unsigned int _meta_ccHist[128] = {};
__device__ unsigned int _meta_kindHist[3] = {};  // 0=whole-run tile, 1=aligned split, 2=all-scalar
__device__ unsigned int _meta_phaseHist[32] = {};
__device__ unsigned long long _meta_diag_call_idx = 0;
#endif

#if defined(MORI_DISP_TIMING)
// [CUSPLIT] separate call counter: the very first (cold) non-replay launch has atypical
// cross-rank skew (process/ctx startup, first-touch allocs) that swamps the steady-state
// completion-wait signal. Skip a few cold launches and sample a later, warmer one instead.
__device__ unsigned long long _cusplit_timing_call_idx = 0;
// [PEERCNT] separate call counter for the per-peer in/out token count diag print (all ranks print).
__device__ unsigned long long _cusplit_diag_call_idx = 0;
// [BPHASE] same fix as _cusplit_timing_call_idx above: sample the 4th non-replay launch
// (index 3) instead of racing all launches for "first one under a cycle threshold" -- the
// old race let an atypical/cold launch win the print for a newly-tried dbn/wpb config.
__device__ unsigned long long _bphase_timing_call_idx = 0;
// Cross-block Part-B accounting. A single-thread (block0) probe is biased: block0 copies
// a light share then spins at the grid barrier, so its "compl" swallows other blocks' copy.
// Instead every block's thd0 atomically min/max its Part-B start/end clock -> the WALL span
// of the concurrent copy phase (_pb_hi-_pb_lo). Dividing by the whole-kernel wall span
// (blk0 end - global min start) gives a dimensionless fraction (no clock-freq needed):
//   PartB algo-BW = dispatch algo-BW / frac   (same 'algorithm BW' basis as the a2a bench).
// clock64() shares one counter domain across all CUs on a GPU, so cross-block compare is valid.
// Max per-block Part-B DURATION (each block's own clock64 end-start diff). Blocks copy
// concurrently, so the busiest block's duration ~= the copy-phase wall time. Using per-block
// DIFFS (not absolute clocks) avoids clock-domain / cross-launch / launch-interleave races.
// Cross-launch max is safe: replay launches skip the copy so their tiny dur never wins.
__device__ unsigned long long _pb_maxdur = 0ull;
// [DIAG] busiest-WARP pure meta-send duration (clock64 span of the per-block meta loop), to prove
// whether meta itself stalls (ms => real bug) vs the stall being purely in cwait. NOTE: every meta
// warp contributes (not just globalWarpId 0) -- a single-warp probe cannot see a slow warp
// elsewhere in the grid, which is exactly the question this global exists to answer.
__device__ unsigned long long _meta_blk_maxdur = 0ull;
// [MSPLIT] Where the meta phase's ~21us actually goes, and where FINALIZE's does. Both phases are
// coupled through the HBM staging arrays (_cusplit_stg*): FINALIZE gathers into them, the meta
// phase TDM-reads them straight back. Moving that staging to LDS can only ever remove the read
// half (mld) plus FINALIZE's staging stores (fstg), so these buckets decide whether that is worth
// doing at all. Each warp accumulates its own cycles and atomicMaxes at the end, same convention
// as _meta_blk_maxdur -- so these are maxima over warps, NOT a decomposition of one warp: they are
// only additive if the warps are homogeneous (check their sum against _meta_blk_maxdur).
//   mIssue = issuing the 4 TdmIssueLoad; mHT = the head/tail regular global->peer copies (issued
//   before the wait, so they overlap the loads); mLd = residual wait for those loads;
//   mSt = issue the 4 TDM stores + wait for them.
// The meta store wait is deferred past the meta/payload barrier, so mSt measures store ISSUE only
// and the wait lands in mDrain, charged where it is actually paid: before the tile is overwritten.
__device__ unsigned long long _meta_issue_maxdur = 0ull;
__device__ unsigned long long _meta_ht_maxdur = 0ull;
__device__ unsigned long long _meta_ld_maxdur = 0ull;
__device__ unsigned long long _meta_st_maxdur = 0ull;
__device__ unsigned long long _meta_drain_maxdur = 0ull;
// [MSPLIT] Same decomposition for the PAYLOAD phase, which until now was only ever measured as one
// ~117us own3b bucket. The question these answer: the single-buffer design leans on ~22 warps/CU
// being TDM-in-flight to hide each warp's store drain (see the TdmIssueLoad header comment), so is
// that drain actually still exposed? pLd = wait for the token's load; pStI = issuing its N stores;
// pDrain = the wait that frees the tile for the next token. Per-warp sums, atomicMax like the meta
// buckets, so they are maxima over warps and only additive if the warps are homogeneous.
__device__ unsigned long long _pay_ld_maxdur = 0ull;
__device__ unsigned long long _pay_sti_maxdur = 0ull;
__device__ unsigned long long _pay_drain_maxdur = 0ull;
// mHT split by field (idx / wt / scale / srcmap): srcmap's run is shorter than 2 x 128B rows so it
// never gets a TDM body, while scale is always 128B-aligned and never has a remainder.
__device__ unsigned long long _meta_ht_f[4] = {0ull, 0ull, 0ull, 0ull};
//   fRoute = FINALIZE's routing (re-read tokenIndices, dedup, s_run atomic, dispDestTokIdMap store);
//   fStg   = FINALIZE's warp-cooperative gather into the HBM staging arrays.
__device__ unsigned long long _fin_route_maxdur = 0ull;
__device__ unsigned long long _fin_stg_maxdur = 0ull;
#endif

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
/*        EpDispatchIntraNodeKernel_clean_body (legacy wide-grid, block-local batched slot)         */
/* ---------------------------------------------------------------------------------------------- */
// LEGACY high-bandwidth body, selected with -DMORI_DISP_CLEAN. Default launch geometry is
// 256 blocks x 16 warps (see _resolve_launch_params in python/mori/ops/dispatch_combine.py):
// it interleaves metadata with the payload per token and relies on a wide grid to hide that,
// unlike EpDispatchIntraNodeKernel_body below which batches metadata and runs at 64x8.
//
// Block-local exact-count + batched remote reservation. Avoids BOTH:
//  (1) NOTIFY's remote count-matrix exchange/prefix + grid barriers -- everything here is
//      block-local, so phase transitions are just __syncthreads (no grid barrier), and
//  (2) legacy's per-token REMOTE fetch_add -- here each block does ONE remote fetch_add(N)
//      per destPe, N = the block's EXACT committed count (counted locally first) so there
//      is no over-reservation / no holes (the concern that rules out a blind batch atomic).
// Per token the slot is a fast LDS atomic; payload via clean 1D TDM. Remote atomics drop
// from O(committed tokens) to npes*numBlocks. Completion tail identical to legacy.
template <typename T, bool EnableStdMoE = false>
__device__ void EpDispatchIntraNodeKernel_clean_body(EpDispatchCombineArgs<T> args) {
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
  long long _pt[8];
  long long _pbStart = 0;  // thd0 Part-B start clock (per-block register, for duration diff)
  const bool _ptOn = (myPe == 0 && blockIdx.x == 0 && thdId == 0);
#define _BPTS(i) do { if (_ptOn) _pt[i] = clock64(); } while (0)
#else
#define _BPTS(i) do {} while (0)
#endif
  _BPTS(0);  // kernel entry

  // ---- Phase 1: block-local count committed tokens per destPe (+ drop sentinels) ----
  for (int p = thdId; p < npes; p += blockDim.x) { s_N[p] = 0; s_run[p] = 0; }
  __syncthreads();
  if (args.tokenIndices && args.inpTokenBuf && !args.replayMode) {
#if defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))  // PER-TOKEN 1-load:N-store is now the DEFAULT batch path (gfx125x+TDM)
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
#if defined(MORI_DISP_TIMING)
  if (thdId == 0) _pbStart = clock64();  // per-block Part-B start (register)
#endif
  // ---- Phase 3: distribute LOCAL slots + send payload (clean 1D TDM, no remote atomic) ----
  if (args.tokenIndices && args.inpTokenBuf && !args.replayMode) {
#if defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
    // PER-TOKEN 1 load : N store. One warp per token: LOAD the token ONCE into the LDS tile,
    // then STORE it to each DISTINCT destPe from that same tile (load amortized, like a2a's
    // 1-load:N-store). Slot is a fast block-local LDS atomic on the Phase-2 reserved base;
    // dedup via __match_any_sync (same keep rule as Phase 1). Per kept peer we also scatter
    // the routing metadata: dispDestTokIdMap + reverse map (dispTokIdToSrcTokId) + weights +
    // indices + scales. scales go through the 16B-coalesced WarpScaleCopy (byte-path fix).
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
          WarpScaleCopy(args.shmemOutScalesMemObj->template GetAs<uint8_t*>(d) + dso,
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
    // Fallback ONLY for non-TDM / non-gfx125x builds: per-pair N-load:N-store WarpCopy.
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
      core::WarpCopy<T, 8>(
          args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) + destTokOffset,
          args.inpTokenBuf + (size_t)srcTokId * hiddenDim, hiddenDim);
    }
#endif  // TDM+gfx125x PER-TOKEN default vs WarpCopy fallback
  }
  __syncthreads();
  _BPTS(3);  // <- phase3 payload copy (Part B: 1D TDM)
#if defined(MORI_DISP_TIMING)
  if (thdId == 0) atomicMax(&_pb_maxdur, (unsigned long long)(clock64() - _pbStart));  // per-block Part-B duration
#endif

  // ---- Completion (identical to legacy): all blocks arrive, then per-peer release-signal ----
  if (thdId == 0) atomicAdd(args.dispatchGridBarrier, 1);
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      shmem::ShmemUint32WaitUntilEquals(args.dispatchGridBarrier, gridDim.x);
      _BPTS(4);  // <- grid barrier satisfied (all local blocks arrived)
      __hip_atomic_store(args.dispatchGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      shmem::ShmemInt32WaitUntilEquals(signal, 0);
      __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
      core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
    }
    _BPTS(5);  // <- all per-peer completion signals sent
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
    _BPTS(6);  // <- all peers' signals received (completion done)
  }
#if defined(MORI_DISP_TIMING)
  if (_ptOn && !args.replayMode) {  // print ONCE, on a REAL (non-replay) launch that actually copied
    __threadfence();  // see all blocks' atomicMax(_pb_maxdur) before reading it
    long long _totChk = _pt[6] - _pt[0];
    unsigned long long _callIdx = atomicAdd(&_bphase_timing_call_idx, 1ull);
    if (_totChk > 0 && _totChk < 20000000LL && _callIdx == 3ull) {
      long long p1 = _pt[1] - _pt[0], p2 = _pt[2] - _pt[1];
      long long pB = _pt[3] - _pt[2], pc = _pt[6] - _pt[3];
      // completion sub-phases: grid barrier wait / per-peer signal send / wait peer recv
      long long cbar = _pt[4] - _pt[3], csig = _pt[5] - _pt[4], cwait = _pt[6] - _pt[5];
      long long tot = _pt[6] - _pt[0];  // blk0 finishes last (does the x-card signal) => whole-kernel wall
      double t = (double)(tot ? tot : 1);
      // block0-local view (BIASED: blk0 copies few tokens then spins at the barrier).
      printf("[BPHASE] blk0 cyc: count=%lld reserve=%lld partB=%lld compl=%lld tot=%lld\n", p1, p2, pB, pc, tot);
      printf("[BPHASE] blk0 pct: count=%.2f reserve=%.2f partB=%.2f compl=%.2f\n",
             100.0 * p1 / t, 100.0 * p2 / t, 100.0 * pB / t, 100.0 * pc / t);
      printf("[BPHASE] compl cyc: barrier=%lld sigsend=%lld waitpeer=%lld\n", cbar, csig, cwait);
      printf("[BPHASE] compl pct: barrier=%.2f sigsend=%.2f waitpeer=%.2f\n",
             100.0 * cbar / t, 100.0 * csig / t, 100.0 * cwait / t);
      // ACCURATE Part-B: busiest-block copy DURATION / whole-kernel wall (both same clock64 domain,
      // both DIFFS). frac dimensionless -> PartB algo-BW = dispatch algo-BW / frac (a2a-aligned).
      double frac = (double)_pb_maxdur / t;
      printf("[BPHASE] PARTB(xblk): partB_maxdur=%llu kernel=%lld frac=%.4f  =>  PartB_algoBW = dispatch_algoBW / %.4f\n",
             _pb_maxdur, tot, frac, frac);
    }
  }
#endif
#undef _BPTS
#ifdef ENABLE_STANDARD_MOE_ADAPT
  if constexpr (EnableStdMoE) {
    InvokeConvertDispatchOutput<T>(args, myPe);
  }
#endif
}

/* ---------------------------------------------------------------------------------------------- */
/*             EpDispatchIntraNodeKernel_body (DEFAULT: narrow grid, batched metadata)              */
/* ---------------------------------------------------------------------------------------------- */
// Default dispatch body. Default launch geometry is 64 blocks x 8 warps (see
// _resolve_launch_params in python/mori/ops/dispatch_combine.py). Versus the legacy clean body it
//   (1) gathers each token's idx/weights/scale/srcmap into peer-local, destTokId-ordered staging
//       during FINALIZE, then ships each (block, peer) run as ONE batched 4-field TDM copy,
//       instead of interleaving scattered per-token metadata stores with the payload, and
//   (2) sends metadata BEFORE the payload so the payload phase drains meta's cross-GPU writes
//       before the completion signal has to cross the fabric.
// Every block is self-contained and every block does the same thing: it counts, reserves,
// finalizes, then sends the metadata and payload for exactly the tokens it owns, so phase
// transitions are plain __syncthreads and no device-wide grid barrier is needed anywhere before
// completion. Token work is strided over the whole grid (gridDim.x * warpNum warps).

// ---- Metadata staging scratch (JIT-side __device__ globals -> NO C++ lib rebuild). Sizes are
// fixed for the EP4/4096-token config. destTokId < worldSize*maxInpTokenPerRank =
// 16384; grid <= CUSPLIT_MAX_BLOCKS; npes <= MAX_GPUS_PER_NODE.
#define CUSPLIT_MAX_SLOTS_PER_PEER 16384
#define CUSPLIT_MAX_BLOCKS 512
// GATHER-FUSED staging: FINALIZE gathers each token's metadata into these per-peer, destTokId-
// ordered SoA arrays (sequential reads of tokenIndices/weights/scales by srcTok, sequential writes
// by destTokId). The meta phase then does a PURE TDM copy staging -> peer (no scattered
// gather). Layout mirrors the peer's dest buffers so a contiguous [destTokId] chunk is TDM-able.
//   _cusplit_stgIdx[peer * CAP*MAXTK + destTokId*tk + e]   = tokenIndices[srcTok*tk + e]
//   _cusplit_stgWt [peer * CAP*MAXTK + destTokId*tk + e]   = weightsBuf  [srcTok*tk + e]
//   _cusplit_stgSc [peer * CAP*MAXSB + destTokId*sBytes+b] = scalesBuf   [srcTok*sBytes + b]
#define CUSPLIT_MAX_TOPK 16
#define CUSPLIT_MAX_SCALE_BYTES 128
__device__ index_t _cusplit_stgIdx[MAX_GPUS_PER_NODE * CUSPLIT_MAX_SLOTS_PER_PEER * CUSPLIT_MAX_TOPK];
__device__ float _cusplit_stgWt[MAX_GPUS_PER_NODE * CUSPLIT_MAX_SLOTS_PER_PEER * CUSPLIT_MAX_TOPK];
__device__ uint8_t _cusplit_stgSc[MAX_GPUS_PER_NODE * CUSPLIT_MAX_SLOTS_PER_PEER * CUSPLIT_MAX_SCALE_BYTES];
// Staging for dispTokIdToSrcTokId. FINALIZE would otherwise write this field with one CROSS-GPU
// scattered 4B store per (token, destPe) -- measured at 20.8us of FINALIZE's 52.4us, more than the
// whole idx/wt/scale staging copy, for 4 bytes of payload. Staging it locally makes it a 4th meta
// field, sent as one contiguous [base, base+cnt) run per (block, peer) like the other three, so
// scattered remote stores become coalesced ones. It also makes the meta item count divide evenly
// by the warp count (4 fields x npes vs 3), which squared the per-warp work distribution.
__device__ index_t _cusplit_stgSrc[MAX_GPUS_PER_NODE * CUSPLIT_MAX_SLOTS_PER_PEER];
#if defined(MORI_DISP_DBLRESERVE)
// Keeps the doubled RESERVE atomic from being optimized away as dead; never actually incremented.
__device__ unsigned _cusplit_dbl_sink;
#endif
#if defined(MORI_DISP_GRIDFLAG)
// Per-block arrival flag, replacing the single contended counter the grid barrier uses. See the
// completion section for why. Padded to 32 index_t (128B) per block so no two blocks share a line.
#define CUSPLIT_ARRIVE_PAD 32
__device__ index_t _cusplit_blkArrive[CUSPLIT_MAX_BLOCKS * CUSPLIT_ARRIVE_PAD];
#endif
// Per-(srcBlock, peer) contiguous remote slot range, written in Phase 2 (per-block RESERVE) and
// read by the meta phase: _cusplit_blkBase[block*npes+peer] = this block's remote base on the
// peer, _cusplit_blkCount = its token count (0 if none). Every block indexes its own row, so this
// caps the launch: gridDim.x must be <= CUSPLIT_MAX_BLOCKS (512, i.e. 2x the 256 CUs on gfx1250).
__device__ index_t _cusplit_blkBase[CUSPLIT_MAX_BLOCKS * MAX_GPUS_PER_NODE];
__device__ index_t _cusplit_blkCount[CUSPLIT_MAX_BLOCKS * MAX_GPUS_PER_NODE];
// The four staged fields moved per (block, peer) run: idx, weights, scale, srcmap.
constexpr int kMetaFields = 4;

// ---- Completion-wait primitives, optionally throttled (MORI_DISP_COMPL_BACKOFF=N, default OFF).
// The shmem WaitUntil* helpers (shmem_device_api.hpp) are backoff-free tight spins on
// __ATOMIC_RELAXED/__HIP_MEMORY_SCOPE_SYSTEM loads. CrossDeviceBarrierIntraNodeKernel above
// documents that this exact pattern livelocks the cco/xGMI fabric so a peer's flag write is never
// re-observed (combine hung until s_sleep was added); dispatch's completion still uses the
// unthrottled form. With the gate OFF these are byte-identical to the shmem originals.
#ifdef MORI_DISP_COMPL_BACKOFF
#define _CUSPLIT_SPIN_PAUSE() __builtin_amdgcn_s_sleep(MORI_DISP_COMPL_BACKOFF)
#else
#define _CUSPLIT_SPIN_PAUSE() ((void)0)
#endif
template <typename T>
__device__ __forceinline__ void _CusplitWaitEq(T* addr, T val) {
  while (core::AtomicLoadRelaxedSystem(addr) != val) {
    _CUSPLIT_SPIN_PAUSE();
  }
}
template <typename T>
__device__ __forceinline__ T _CusplitWaitGt(T* addr, T val) {
  T got;
  do {
    got = core::AtomicLoadRelaxedSystem(addr);
    if (got <= val) _CUSPLIT_SPIN_PAUSE();
  } while (got <= val);
  return got;
}

template <typename T, bool EnableStdMoE = false>
__device__ void EpDispatchIntraNodeKernel_body(EpDispatchCombineArgs<T> args) {
  const EpDispatchCombineConfig& config = args.config;
  int thdId = threadIdx.x;
  int laneId = threadIdx.x & (warpSize - 1);
  int warpId = thdId / warpSize;
  int warpNum = blockDim.x / warpSize;
  int globalWarpId = blockIdx.x * warpNum + warpId;
  int myPe = config.rank;
  int npes = config.worldSize;
  size_t hiddenDim = config.HiddenDimSz();
  const int topk = config.numExpertPerToken;
  // ALL data-parallel work (count / reserve / finalize / meta / payload) runs on EVERY block, and
  // each token is counted, reserved, finalized and sent by the SAME owning block, so nothing is
  // dropped. One partition, shared by all three token loops: warp aWarp of aWarps.
  int aWarp = globalWarpId;
  int aWarps = (int)gridDim.x * warpNum;

  // Tokens processed per warp iteration. warpSize/topk lets COUNT's tokenIndices read use all
  // warpSize lanes (a full 128B coalesced burst) instead of only topk of them (8/32 here => a 32B
  // load). Falls back to 1 token/iteration when topk does not divide the warp. COUNT, FINALIZE and
  // the payload loop must all use the SAME partition: s_N sizes the block's reservation (so a block
  // has to FINALIZE exactly the tokens it counted) and the payload phase reads back only the
  // dispDestTokIdMap entries its own block wrote (only a __syncthreads separates them, not a grid
  // barrier), so a partition mismatch races across blocks and lands payloads in the wrong slots.
  const int _tpi = (topk > 0 && topk <= warpSize && (warpSize % topk) == 0) ? (warpSize / topk) : 1;
  const int _sLane = (_tpi > 1) ? (laneId / topk) : 0;  // which token of the batch this lane serves
  const int _eLane = (_tpi > 1) ? (laneId - _sLane * topk) : laneId;
  const bool _laneAct = (_tpi > 1) ? (_sLane < _tpi) : (laneId < topk);

#if defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
  extern __shared__ char _tdmBatchSmem[];
  T* _tdmTile = reinterpret_cast<T*>(_tdmBatchSmem) + (size_t)warpId * hiddenDim;
  const gfx1250_TDM_GROUP1 _tdmG1 = TdmShape<T>(static_cast<int>(hiddenDim));
#if defined(MORI_DISP_PAYSPLIT) && (MORI_DISP_PAYSPLIT > 1)
  // Payload segmentation: the SAME LDS tile, described as MORI_DISP_PAYSPLIT consecutive segments
  // instead of one. The point is in-flight count, not bytes -- see the payload loop.
  constexpr int _pnS = MORI_DISP_PAYSPLIT;
  const int _pSeg = static_cast<int>(hiddenDim) / _pnS;
  // A TDM row under 128B collapses bandwidth (~500 GB/s at 224B vs ~1500 at 256B), and an uneven
  // split would drop elements, so fall back to the single-segment form rather than be wrong or slow.
  const bool _pSplitOk =
      (_pSeg * _pnS == static_cast<int>(hiddenDim)) && (_pSeg * (int)sizeof(T) >= 128);
  const gfx1250_TDM_GROUP1 _tdmGS = TdmShape<T>(_pSplitOk ? _pSeg : static_cast<int>(hiddenDim));
#endif
#endif

  constexpr int kMaxNpes = MAX_GPUS_PER_NODE;

#if defined(MORI_DISP_TIMING)
  long long _pt[8];
  long long _pbStart = 0;  // thd0 Part-B start clock (per-block register, for duration diff)
  const bool _ptOn = (blockIdx.x == 0 && thdId == 0);
#define _BPTS(i) do { if (_ptOn) _pt[i] = clock64(); } while (0)
#else
#define _BPTS(i) do {} while (0)
#endif
  _BPTS(0);  // kernel entry

#if defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
  // ==== Phases (TDM-only, decentralized): Phase 1 block-local COUNT (LDS histogram, like CLEAN);
  // Phase 2 per-block RESERVE (each block one remote atomic per peer -> its own contiguous slot
  // range on the peer, s_base) -- fully decentralized, NO grid barrier; FINALIZE assigns
  // destTokId = s_base + block-local running index (s_run) and gathers the four metadata fields
  // into peer-local staging; then each block TDM-sends its own metadata runs, and finally streams
  // its own tokens' payload via TDM. Phase transitions are plain __syncthreads. ----
  __shared__ index_t s_N[kMaxNpes];     // block-local committed count per destPe
  __shared__ index_t s_base[kMaxNpes];  // this block's REMOTE contiguous slot base on the peer
  __shared__ index_t s_run[kMaxNpes];   // block-local running distribution index (Phase 3)
  for (int p = thdId; p < npes; p += blockDim.x) { s_N[p] = 0; s_run[p] = 0; }
  __syncthreads();

  // ---- Phase 1: block-local count (LDS atomic -- no cross-block contention) ----
#if defined(MORI_DISP_DBLCOUNT)
  // Cost isolation by DOUBLING instead of deleting. Deleting this phase would hand RESERVE a garbage
  // s_N and let the payload store outside its reserved slot range (memory corruption, not just a
  // wrong answer), so the phase runs a SECOND time with its only real side effect -- the atomicAdd --
  // redirected to a scratch histogram. The dispDestTokIdMap store in the else branch writes the same
  // sentinel both passes, so it is idempotent and stays as is. The result therefore remains correct:
  // ACC must still PASS, which is what proves kernel(2x) - kernel(1x) measured a working kernel.
  __shared__ index_t s_Ndbl[kMaxNpes];
  for (int p = thdId; p < npes; p += blockDim.x) s_Ndbl[p] = 0;
  __syncthreads();
#endif
  if (args.tokenIndices && args.inpTokenBuf && !args.replayMode) {
#if defined(MORI_DISP_DBLCOUNT)
   // The pass loop lives entirely inside the gate so the default build is byte-identical to the
   // ungated form -- a trip-count-1 loop left in the default path would put a measurement artifact
   // into the shipping path. Deliberately not unrolled: two copies of the body raise register
   // pressure enough to slow the WHOLE kernel (payload included), and then the doubling difference
   // measures the compiler's reaction instead of the phase.
#pragma clang loop unroll(disable)
   for (int _cp = 0; _cp < 2; ++_cp) {
#endif
    for (int tokBase = aWarp * _tpi; tokBase < args.curRankNumToken; tokBase += aWarps * _tpi) {
      int tok = tokBase + _sLane;
      bool act = _laneAct && (tok < args.curRankNumToken);
      index_t myExpert = act ? args.tokenIndices[(size_t)tok * topk + _eLane] : (index_t)-1;
      int myDestPe = -1;
      if (myExpert >= 0) { int d = (int)(myExpert / config.numExpertPerRank);
                           if (d >= 0 && d < config.worldSize) myDestPe = d; }
      // Composite match key. With several tokens in flight per iteration, matching on destPe alone
      // would merge lanes of DIFFERENT tokens into one group and keep only one of them, undercounting
      // s_N. At _tpi == 1 the _sLane term is 0 and this is the plain destPe-only key.
      unsigned mv = (myDestPe >= 0) ? (((unsigned)_sLane << 8) | (unsigned)myDestPe) : 0xFFFFFFFFu;
      unsigned long long grp = __match_any_sync(0xFFFFFFFFFFFFFFFFull, mv);
      int keep = (myDestPe >= 0 && laneId == (__ffsll((long long)grp) - 1)) ? 1 : 0;
      if (act) {
        if (keep) {
#if defined(MORI_DISP_DBLCOUNT)
          // Two branches on distinct static arrays, NOT one runtime-selected pointer: a ternary over
          // two LDS arrays still leaves the compiler unable to prove the address space and degrades
          // the atomic to flat, which slows BOTH passes and makes the difference meaningless. (It did:
          // the pointer form measured 988.8 GB/s, implying an absurd 48.8us for a 3.7us phase.)
          if (_cp == 0) atomicAdd(&s_N[myDestPe], 1);
          else atomicAdd(&s_Ndbl[myDestPe], 1);
#else
          atomicAdd(&s_N[myDestPe], 1);
#endif
        } else {
          args.dispDestTokIdMap[(size_t)tok * topk + _eLane] = FlatTokenIndex(config, config.worldSize, 0);
        }
      }
    }
#if defined(MORI_DISP_DBLCOUNT)
   }
#endif
  }
  __syncthreads();  // all warps in this block done counting before pushing s_N to global
  _BPTS(1);  // <- phase1 count (block-local LDS histogram)

  // ---- Phase 2: per-block RESERVE. Each block does ONE remote atomic per active peer against
  // dispTokOffsetMemObj[p], the returned old value is this block's own contiguous slot base on
  // that peer (s_base[p]) -- fully decentralized like CLEAN, so NO grid barrier is needed here
  // (barrierA/barrierB removed). Also: (a) local atomicAdd into destPeTokenCounter[p] for the
  // recv-count report at completion, and (b) record this (block,peer) range in the global
  // _cusplit_blkBase/_cusplit_blkCount so the metadata group can iterate per-(block,peer) spans
  // (per-block reserve => this rank's slots on a peer are one contiguous run PER BLOCK, not one
  // run for the whole rank). blkCount is written even when 0 to overwrite stale prior-launch
  // values, and each block only ever touches its own row.
  for (int p = thdId; p < npes; p += blockDim.x) {
    index_t n = s_N[p];
    _cusplit_blkCount[(size_t)blockIdx.x * npes + p] = n;
    if (n > 0) {
#if defined(MORI_DISP_DBLRESERVE)
      // Second round trip to the SAME remote counter, adding 0: identical cross-GPU atomic latency
      // and identical contention (all gridDim.x blocks hit one counter per peer), but the slot
      // allocation is untouched, so the result stays correct and ACC still PASSes. The sink keeps the
      // atomic from being elided as dead -- the branch is never taken, a slot base is never negative.
      index_t _rdbl = __hip_atomic_fetch_add(args.dispTokOffsetMemObj->template GetAs<index_t*>(p),
                                             (index_t)0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      if (_rdbl < (index_t)0) atomicAdd(&_cusplit_dbl_sink, 1u);
#endif
      s_base[p] = __hip_atomic_fetch_add(args.dispTokOffsetMemObj->template GetAs<index_t*>(p), n,
                                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      _cusplit_blkBase[(size_t)blockIdx.x * npes + p] = s_base[p];
      atomicAdd(&args.destPeTokenCounter[p], n);
    }
  }
  __syncthreads();  // s_base visible to all threads in this block
  _BPTS(2);         // <- reserve bucket: per-block remote atomic (no barrier)

  // ---- FINALIZE: recompute routing (cheap ALU); destTokId = this block's remote base (s_base)
  // plus a block-local running index (s_run, LDS atomic). No cross-block collision: each block
  // owns a disjoint [s_base, s_base+s_N) range carved out by its own remote atomic above. ----
  const int sBytesF = config.scaleDim * config.scaleTypeSize;
  const bool doScaleF = (args.scalesBuf && config.scaleDim > 0 && config.scaleTypeSize > 0);
#if defined(MORI_DISP_METALDS)
  // ---- META STAGED IN LDS INSTEAD OF HBM. FINALIZE's gather and the meta send are the same 2.9MB
  // seen twice: the gather writes it to the global staging arrays (measured 5.45us: 166.0 -> 160.55us
  // with MORI_DISP_NOSTG) and the meta phase TDM-loads it straight back into LDS to have something a
  // TDM store can send (measured 6.95us for the whole phase, of which only ~1.8us is engine time for
  // 2.9MB). Gathering directly into that LDS tile deletes the HBM round trip and the TDM load with it.
  // Vector stores to the peer are NOT an option -- MORI_DISP_METAVEC measures 995 GB/s (213.4us),
  // i.e. cross-GPU vector writes sustain only ~54 GB/s against TDM's ~1600 -- so meta must stay on TDM,
  // and TDM can only source LDS. This is the same tile the payload phase uses (_tdmBatchSmem, whole
  // block), so the meta stores must be drained BEFORE the separating __syncthreads(), unlike the
  // per-warp-private tile whose drain is deferred past it: here a warp's payload load overwrites LDS
  // that OTHER warps' meta stores are still reading.
  //
  // Layout: one 128B-aligned region per (peer, field), sized from this block's own s_N. Each region
  // starts `phase & 31` elements past its aligned base, giving the LDS copy the SAME 128B phase as
  // the peer-side destination -- so TdmWholeOrSplit128's peel aligns both sides at once, for any
  // split part or chunk offset, exactly as it does when the source is staging.
  __shared__ int s_mOff[kMaxNpes * 4];  // element offsets into _tdmBatchSmem, per (peer, field)
  __shared__ int s_mOk;                 // 0 => block does not fit in LDS, fall back to staging
  if (thdId == 0) {
    const int _sBL = config.scaleDim * config.scaleTypeSize;
    const int _sVL = _sBL >> 2;  // scale 4B-elements per slot
    const int _avail = (int)((size_t)warpNum * hiddenDim * sizeof(T) / 4);
    int off = 0, ok = 1;
    // The meta send's degenerate branch (tile too small for even one token) reads staging, so LDS
    // staging has to be off whenever that branch can be taken.
    const int _perTokL = topk * 4 + topk * 4 + _sBL + 4;
    if (_perTokL <= 0 || ((int)(hiddenDim * sizeof(T)) - 512) / _perTokL <= 0) ok = 0;
    for (int p = 0; p < npes; ++p) {
      for (int f = 0; f < 4; ++f) {
        s_mOff[p * 4 + f] = 0;
        const int ePS = (f < 2) ? topk : ((f == 2) ? _sVL : 1);
        if (f == 1 && !args.weightsBuf) continue;
        if (f == 2 && !(args.scalesBuf && _sBL > 0)) continue;
        const int nEl = (int)s_N[p] * ePS;
        if (nEl <= 0) continue;
        const size_t phase = (size_t)s_base[p] * ePS;
        const TdmSplit128 sp = TdmWholeOrSplit128(phase, nEl);
        // A peeled body starts `sp.head` elements in, so the region has to carry the phase for that
        // to land on a 128B boundary. A whole-run tile stores from element 0, so it wants none.
        const int d = (sp.rows > 0 && sp.head > 0) ? (int)(phase & 31) : 0;
        off = (off + 31) & ~31;
        s_mOff[p * 4 + f] = off + d;
        off += d + nEl;
        if (off > _avail) ok = 0;
      }
    }
    s_mOk = ok;
  }
  __syncthreads();
#endif
#if defined(MORI_DISP_TIMING)
  unsigned long long _fRoute = 0ull, _fStg = 0ull;
#endif
  if (args.tokenIndices && args.inpTokenBuf && !args.replayMode) {
    // ---- Lane-parallel FINALIZE. The TOKEN PARTITION is untouched: this walks exactly the tokens
    // the per-token form it replaced did, so COUNT and the payload loop stay as they are and the
    // "each block reads back only its own dispDestTokIdMap" invariant still holds. Only the
    // INTRA-WARP lane assignment changes, in two places:
    //
    // (1) Routing runs in COUNT's layout -- lane -> (token tokBase+_sLane, expert _eLane) -- so the
    //     tokenIndices read is one full-warp 128B burst per _tpi tokens instead of _tpi separate
    //     topk-lane 32B reads, and the serial chain (dependent load -> dedup -> s_run LDS atomic)
    //     is walked once per _tpi tokens instead of once per token. The per-token form walked it 8
    //     times per warp at DBN=64/wpb=8 (8 tokens/warp), nothing overlapping across iterations.
    // (2) The staging gather uses LANE GROUPS: gsz lanes serve ONE (token, peer) destination, so
    //     ngrp destinations are gathered per pass with all warpSize lanes busy. The per-token form
    //     had the whole warp cooperate on one destination at a time with only topk (or nSvF) lanes
    //     active -- 8 of 32 here -- and walked the ~3.6 kept destinations of each token serially.
    //
    // The KEPT LANE SET is provably identical to COUNT's: within one token the composite match key
    // groups the same experts, and the group's lowest lane is the lowest expert index routed to that
    // peer. So the dispDestTokIdMap entries written are the same ones; only which slot id each lands
    // on can differ, and that was already nondeterministic (the s_run atomics race).
    const int nSvF = sBytesF >> 4;
    // gsz = lanes per destination, rounded up to a power of two so laneId splits by shift/mask.
    // Capped at warpSize, which degenerates to "whole warp on one destination" for topk or nSvF
    // wider than a warp.
    int _gszReq = (topk > nSvF) ? topk : nSvF;
    if (_gszReq < 1) _gszReq = 1;
    int _gszP2 = 1;
    while (_gszP2 < _gszReq) _gszP2 <<= 1;
    const int gsz = (_gszP2 <= warpSize) ? _gszP2 : warpSize;
    const int ngrp = warpSize / gsz;
    const int myGrp = laneId / gsz;
    const int myE = laneId - myGrp * gsz;
    for (int tokBase = aWarp * _tpi; tokBase < args.curRankNumToken; tokBase += aWarps * _tpi) {
#if defined(MORI_DISP_TIMING)
      long long _f0 = clock64();
#endif
      int tok = tokBase + _sLane;
      bool act = _laneAct && (tok < args.curRankNumToken);
      index_t myExpert = act ? args.tokenIndices[(size_t)tok * topk + _eLane] : (index_t)-1;
      int myDestPe = -1;
      if (myExpert >= 0) { int d = (int)(myExpert / config.numExpertPerRank);
                           if (d >= 0 && d < config.worldSize) myDestPe = d; }
      // Composite key, identical to COUNT's: without the _sLane term lanes of DIFFERENT tokens that
      // share a destPe collapse into one group and only one of them gets a slot.
      unsigned mv = (myDestPe >= 0) ? (((unsigned)_sLane << 8) | (unsigned)myDestPe) : 0xFFFFFFFFu;
      unsigned long long grp = __match_any_sync(0xFFFFFFFFFFFFFFFFull, mv);
      int keep = (act && myDestPe >= 0 && laneId == (__ffsll((long long)grp) - 1)) ? 1 : 0;
      index_t myDestTokId = -1;
      if (keep) {
        index_t j = atomicAdd(&s_run[myDestPe], 1);
        myDestTokId = s_base[myDestPe] + j;
        args.dispDestTokIdMap[(size_t)tok * topk + _eLane] =
            FlatTokenIndex(config, myDestPe, myDestTokId);
        // srcmap goes to local staging (4th meta field) rather than a cross-GPU scattered 4B store.
#if defined(MORI_DISP_METALDS)
        const int _lsS = (int)(myDestTokId - s_base[myDestPe]);
        if (s_mOk && _lsS >= 0 && _lsS < (int)s_N[myDestPe]) {
          (reinterpret_cast<index_t*>(_tdmBatchSmem) + s_mOff[myDestPe * 4 + 3])[_lsS] =
              FlatTokenIndex(config, myPe, tok);
        } else if (myDestTokId < CUSPLIT_MAX_SLOTS_PER_PEER) {
          _cusplit_stgSrc[(size_t)myDestPe * CUSPLIT_MAX_SLOTS_PER_PEER + myDestTokId] =
              FlatTokenIndex(config, myPe, tok);
        }
#elif !defined(MORI_DISP_NOSTG)
        if (myDestTokId < CUSPLIT_MAX_SLOTS_PER_PEER)
          _cusplit_stgSrc[(size_t)myDestPe * CUSPLIT_MAX_SLOTS_PER_PEER + myDestTokId] =
              FlatTokenIndex(config, myPe, tok);
#endif
      }
#if defined(MORI_DISP_TIMING)
      long long _f1 = clock64();
      _fRoute += (unsigned long long)(_f1 - _f0);
#endif
#if !defined(MORI_DISP_NOSTG)
      // Hand out the kept destinations ngrp at a time. keepMask is warp-uniform, so the loop trip
      // count is uniform and the per-group `continue` below only masks lanes -- it cannot diverge
      // the loop itself.
      unsigned long long keepMask = __ballot(keep);
      while (keepMask) {
        int srcLane = -1;
        unsigned long long t = keepMask;
        for (int g = 0; g < ngrp; ++g) {
          if (!t) break;
          int l = __ffsll((long long)t) - 1;
          t &= t - 1;
          if (g == myGrp) srcLane = l;
        }
        keepMask = t;  // consumed exactly the (up to ngrp) lanes handed out above
        // __shfl is warp-wide: groups that got no destination this pass must still execute it, so
        // read lane 0 and drop the result below rather than skipping the shuffle.
        int sl = (srcLane < 0) ? 0 : srcLane;
        int d = __shfl(myDestPe, sl);
        index_t dt = __shfl(myDestTokId, sl);
        int gTok = __shfl(tok, sl);
        if (srcLane < 0) continue;
        if (dt < 0 || dt >= CUSPLIT_MAX_SLOTS_PER_PEER) continue;
#if defined(MORI_DISP_METALDS)
        // Same gather, LDS destination. Kept as a SEPARATE path rather than swapping the base
        // pointers below: a pointer that is assigned an LDS address on one path and a global address
        // on the other loses its address space, and the whole gather degrades to flat accesses.
        // The block-local slot is dt - s_base[d], < s_N[d] by construction (dt = s_base[d] + s_run++).
        const int _lg = (int)(dt - s_base[d]);
        if (s_mOk && _lg >= 0 && _lg < (int)s_N[d]) {
          int* _lb = reinterpret_cast<int*>(_tdmBatchSmem);
          index_t* lIdx = reinterpret_cast<index_t*>(_lb + s_mOff[d * 4 + 0]) + _lg * topk;
          for (int e = myE; e < topk; e += gsz) lIdx[e] = args.tokenIndices[(size_t)gTok * topk + e];
          if (args.weightsBuf) {
            float* lWt = reinterpret_cast<float*>(_lb + s_mOff[d * 4 + 1]) + _lg * topk;
            for (int e = myE; e < topk; e += gsz) lWt[e] = args.weightsBuf[(size_t)gTok * topk + e];
          }
          if (doScaleF) {
            uint4* lSc = reinterpret_cast<uint4*>(_lb + s_mOff[d * 4 + 2]) + (size_t)_lg * nSvF;
            const uint4* srcSc =
                reinterpret_cast<const uint4*>(args.scalesBuf + (size_t)gTok * sBytesF);
            for (int c = myE; c < nSvF; c += gsz) lSc[c] = srcSc[c];
          }
          continue;
        }
#endif
        index_t* sIdx = _cusplit_stgIdx +
                        (size_t)d * CUSPLIT_MAX_SLOTS_PER_PEER * CUSPLIT_MAX_TOPK + (size_t)dt * topk;
        float* sWt = _cusplit_stgWt +
                     (size_t)d * CUSPLIT_MAX_SLOTS_PER_PEER * CUSPLIT_MAX_TOPK + (size_t)dt * topk;
        uint8_t* sSc = _cusplit_stgSc +
                       (size_t)d * CUSPLIT_MAX_SLOTS_PER_PEER * CUSPLIT_MAX_SCALE_BYTES +
                       (size_t)dt * sBytesF;
        for (int e = myE; e < topk; e += gsz) sIdx[e] = args.tokenIndices[(size_t)gTok * topk + e];
        if (args.weightsBuf) {
          for (int e = myE; e < topk; e += gsz) sWt[e] = args.weightsBuf[(size_t)gTok * topk + e];
        }
        if (doScaleF) {
          const uint8_t* srcSc = args.scalesBuf + (size_t)gTok * sBytesF;
          for (int c = myE; c < nSvF; c += gsz)
            reinterpret_cast<uint4*>(sSc)[c] = reinterpret_cast<const uint4*>(srcSc)[c];
        }
      }
#endif  // !MORI_DISP_NOSTG
#if defined(MORI_DISP_TIMING)
      _fStg += (unsigned long long)(clock64() - _f1);
#endif
    }
  }
#if defined(MORI_DISP_TIMING)
  if (laneId == 0) {
    atomicMax(&_fin_route_maxdur, _fRoute);
    atomicMax(&_fin_stg_maxdur, _fStg);
  }
#endif
  _BPTS(7);  // <- finalize done (all blocks), right before payload/meta grid barrier
  // ---- No grid barrier here: each block is self-contained -- it routes its own tokens (FINALIZE)
  // then sends only those tokens' meta+payload, reading only its OWN dispDestTokIdMap / staging /
  // blkBase / blkCount (same aWarps stride). So a block-level __syncthreads (make this block's
  // FINALIZE cross-warp writes visible to its meta/payload warps) suffices -- no cross-block
  // dependency, no grid-barrier cost, no all-blocks-co-resident requirement. ----
  __syncthreads();
#endif  // MORI_DISP_TDM && gfx125x (this body is a TDM-only path)

// META FIRST, THEN PAYLOAD: the payload phase that follows (~116-133us) serves as the DRAIN WINDOW
// for meta's cross-GPU writes, so by the time the completion cross-rank signal fires, meta fabric
// traffic is long gone and no longer queues ahead of the (small) signal atomic on the sender's
// outbound fabric -- which is what made cwait spin ~ms when meta trailed payload into completion.
#if defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
  // ---- Phase 3a-meta: PER-BLOCK meta send, BEFORE payload. Each warp TDM-copies one (peer,
  // sub-range) run of THIS block's own gathered staging to the peer. Nothing is read across blocks,
  // so the __syncthreads() already performed after FINALIZE suffices and no device-wide grid
  // barrier is needed -- the grid-cooperative variant that did read other blocks' ranges needed one
  // and it cost ~40us of pure structural overhead per launch for no distribution benefit.
  //
  // Geometry at DBN=64/wpb=8, CONFIRMED by the [GEOM] print (MORI_DISP_TIMING): gridDim=64
  // blockDim=256 warpSize=32 warpNum=8 aWarps=512 numToken=4096 topk=8 npes=4, i.e. 8 tokens
  // per warp. Do NOT re-derive this from launch.cpp's `block_x = WARP_SIZE(64) * wpb`: that is the
  // C++ LaunchDispatch path, which the python benchmarks never take. The python JIT path launches
  // dispatch_combine.py `block = (self._warp_size * actual_wpb,)` with the DEVICE warp size
  // (32 on gfx1250), so blockDim = 32*8 = 256 and globalWarpNum = 512 -- not 512*2.
  // Per block there are npes*kMetaFields = 16 items, which divides evenly by the 8 warps. With the
  // pre-srcmap 3 fields it did not (12/8), leaving a 2:1 split where a quarter of the warps carried
  // half the work -- adding srcmap as the 4th field squared the distribution.
  // ----
#if defined(MORI_DISP_TIMING)
  long long _mt0b = clock64();
#endif
  // This warp has meta TDM stores still reading its LDS tile. The tile is per-warp private
  // (_m4 and the payload's _tdmTile are both _tdmBatchSmem + warpId * mtileBytesM), so only this
  // warp can invalidate them, and only by overwriting the tile: either the next run's TDM load
  // below, or the payload phase's first load. Declared outside the meta `if` so the drain can sit
  // after the __syncthreads() that separates the two phases, where the barrier's own wait for the
  // block's slowest meta warp absorbs it. mSt therefore measures store ISSUE only, with the wait
  // reported separately as mDrain.
  //
  // Unconditional: this only moves WHERE the wait is paid, every byte still travels by TDM, so it
  // cannot hit the ~54 GB/s cross-GPU plain-store wall that made METAVEC a loss. Measured with ACC
  // PASS both times: +0.65% at DBN=64 (1268.6 -> 1276.8 GB/s) and +1.2% at DBN=128 (1342.0 -> 1357.7).
  bool _mPend = false;
#if defined(MORI_DISP_TIMING)
  unsigned long long _mDrain = 0ull;
#endif
#if defined(MORI_DISP_METALDS)
  // Meta stores that read the block-shared LDS tile. Drained just before the barrier that hands the
  // tile to the payload phase -- see the store site for why this one cannot be deferred past it.
  bool _lPend = false;
#endif
#if defined(MORI_DISP_NOMETA) || defined(MORI_DISP_METAFUSE)
  // MORI_DISP_METAFUSE moves this phase's work into the payload loop, so the phase itself goes away.
  //
  // MORI_DISP_NOMETA is DIAGNOSTIC ONLY, PRODUCES WRONG RESULTS. Compiles the whole meta send away while keeping the
  // launch geometry and LDS reservation identical, so kernel(full) - kernel(NOMETA) is the meta
  // phase's real cost WITHOUT the clock64() probes that inflate it under MORI_DISP_TIMING.
  if (false) {
#else
  if (args.tokenIndices && args.inpTokenBuf && !args.replayMode) {
#endif
    const int tkM = config.numExpertPerToken;
    const int sBytesM = config.scaleDim * config.scaleTypeSize;
    const int sVecM = sBytesM >> 4;
    const bool doScaleM = (args.scalesBuf && config.scaleDim > 0 && config.scaleTypeSize > 0);
    const index_t recvCapM = (index_t)config.MaxNumTokensToRecv();
    // One warp owns a whole (peer, sub-range) run and moves ALL FOUR fields through one LDS tile
    // with a SINGLE load-wait / store-wait pair; splitting the work per field instead makes every
    // field pay its own full LOAD -> s_wait_tensorcnt -> STORE -> s_wait_tensorcnt round trip.
    // The launch reserves warpNum * hiddenDim * sizeof(T) of dynamic LDS (14336B/warp at hidden
    // 7168 bf16) -- enough for ~73 tokens x 196B/token, past the ~58 a (block,peer) run holds.
    const int mtileBytesM = (int)(hiddenDim * sizeof(T));
    const int perTokM = tkM * 4 + tkM * 4 + sBytesM + 4;
    // 512B of slack covers rounding each of the 4 field regions up to a 128B LDS boundary.
    const int tokCapM = (perTokM > 0) ? ((mtileBytesM - 512) / perTokM) : 0;
#if defined(MORI_DISP_TIMING)
    unsigned long long _mIssue = 0ull, _mHT = 0ull, _mLd = 0ull, _mSt = 0ull;
    unsigned long long _mF[4] = {0ull, 0ull, 0ull, 0ull};  // mHT split per field: idx/wt/scale/srcmap
#define _MHTS(i)                                                  \
  do {                                                            \
    long long _t = clock64();                                     \
    _mF[i] += (unsigned long long)(_t - _mFp);                    \
    _mFp = _t;                                                    \
  } while (0)
#else
#define _MHTS(i) do {} while (0)
#endif
    if (tokCapM > 0) {
      uint8_t* _m4 = reinterpret_cast<uint8_t*>(_tdmBatchSmem) + (size_t)warpId * mtileBytesM;
#if defined(MORI_DISP_METAFIELD)
      // ---- One warp per (peer, field): npes*4 = 16 items over 8 warps, so a warp issues 2 TDM
      // stores instead of 4 and the block's meta op count halves (32 -> 16) while every warp still
      // works. Premise: with the deferred drain, mSt is 21.4us of pure store ISSUE (mDrain is 0.0), so the
      // cost looked like TDM engine queueing, making op count the lever.
      //
      // MEASURED A LOSS (-4.3%: 1276.8 -> 1222.2 GB/s at EP4-4K, acc PASS), kept only so nobody
      // retries it. With METASPLIT=1 also losing (-1.5%, halves op count by idling warps instead),
      // the conclusion is that the meta TDM cost tracks BYTES, not op count. Two costs added here
      // outweigh the halved op count: srcmap is the only field that never gets a TDM body (its run
      // is cc 4B elements, under the 128B row floor), so per-field assignment concentrates all of
      // its per-lane remote stores into npes warps while the rest go idle -- and the __syncthreads()
      // before the payload phase waits for those stragglers; and a warp's two items share one tile,
      // so the second pays a drain the per-run form never had.
      //
      // The per-run form's reason to keep all four fields in one warp was the shared
      // load-wait/store-wait pair; the deferred drain already removed the store wait, so a field now costs
      // one load wait. Each item also carries the peer's WHOLE run (no sub-range cut), which is the
      // fewest 128B remainders possible -- the same property METASPLIT=1 was after, without idling
      // warps to get it.
      //
      // The tile holds ONE field here, not four, so its token capacity is per-field and far above
      // the run length (444 idx/wt, 111 scale, 3552 srcmap tokens at 14336B/warp vs cntAll ~58):
      // the chunk loop is kept for the guarantee, not because it iterates.
      const int nItems = npes * kMetaFields;
      for (int it = warpId; it < nItems; it += warpNum) {
        const int peer = it / kMetaFields;
        const int field = it - peer * kMetaFields;  // 0=idx, 1=wt, 2=scale, 3=srcmap
        if (field == 1 && !args.weightsBuf) continue;
        if (field == 2 && !doScaleM) continue;
        // s_N/s_base: this block's own Phase-2 reserve result, still in LDS (see the per-run form).
        index_t cntAll = s_N[peer];
        if (cntAll <= 0) continue;
        index_t baseAll = s_base[peer];
        // 4B elements per token for this field. Doubles as the run's 128B phase multiplier, since
        // every field's src and dst are both base + ab * ePerTok elements.
        const int ePerTok = (field == 2) ? (sVecM * 4) : ((field == 3) ? 1 : tkM);
        const int capTok = (mtileBytesM - 128) / (ePerTok * 4);
        for (index_t cs = 0; cs < cntAll; cs += capTok) {
          const int cc = (int)((cs + capTok <= cntAll) ? capTok : (cntAll - cs));
          const index_t ab = baseAll + cs;
          if (ab + cc > recvCapM) continue;  // OOB guard (peer slot capacity)
          const int nEl = cc * ePerTok;
          int* src;
          int* dst;
          if (field == 0) {
            src = reinterpret_cast<int*>(
                _cusplit_stgIdx + (size_t)peer * CUSPLIT_MAX_SLOTS_PER_PEER * CUSPLIT_MAX_TOPK +
                (size_t)ab * tkM);
            dst = reinterpret_cast<int*>(
                args.shmemOutIndicesMemObj->template GetAs<index_t*>(peer) + (size_t)ab * tkM);
          } else if (field == 1) {
            src = reinterpret_cast<int*>(
                _cusplit_stgWt + (size_t)peer * CUSPLIT_MAX_SLOTS_PER_PEER * CUSPLIT_MAX_TOPK +
                (size_t)ab * tkM);
            dst = reinterpret_cast<int*>(
                args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(peer) + (size_t)ab * tkM);
          } else if (field == 2) {
            src = reinterpret_cast<int*>(
                _cusplit_stgSc + (size_t)peer * CUSPLIT_MAX_SLOTS_PER_PEER * CUSPLIT_MAX_SCALE_BYTES +
                (size_t)ab * sBytesM);
            dst = reinterpret_cast<int*>(
                args.shmemOutScalesMemObj->template GetAs<uint8_t*>(peer) + (size_t)ab * sBytesM);
          } else {
            src = reinterpret_cast<int*>(_cusplit_stgSrc +
                                         (size_t)peer * CUSPLIT_MAX_SLOTS_PER_PEER + (size_t)ab);
            dst = reinterpret_cast<int*>(
                args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(peer) + (size_t)ab);
          }
          const TdmSplit128 sp = TdmWholeOrSplit128((size_t)ab * ePerTok, nEl);
          int* tile = reinterpret_cast<int*>(_m4);
          gfx1250_TDM_GROUP1 g{};
          // A previous item's stores may still be reading this tile (each warp owns 2 items).
          if (_mPend) {
#if defined(MORI_DISP_TIMING)
            long long _md0 = clock64();
#endif
            __builtin_amdgcn_s_wait_tensorcnt(0);
            _mPend = false;
#if defined(MORI_DISP_TIMING)
            _mDrain += (unsigned long long)(clock64() - _md0);
#endif
          }
#if defined(MORI_DISP_TIMING)
          long long _m0 = clock64();
#endif
          if (sp.body) {
            g = TdmSplitShape(sp, sp.body);
            TdmIssueLoad<int>(tile, src + sp.head, g);
          }
#if defined(MORI_DISP_TIMING)
          long long _m1 = clock64();
          _mIssue += (unsigned long long)(_m1 - _m0);
#endif
          // Unaligned head/tail (and a field too small for 2 rows -- srcmap, whose run is cc 4B
          // elements and so never reaches the 128B row floor) go straight global->global, issued
          // here so they overlap the TDM load already in flight.
          for (int i = laneId; i < sp.head; i += warpSize) dst[i] = src[i];
          for (int i = sp.head + sp.body + laneId; i < nEl; i += warpSize) dst[i] = src[i];
#if defined(MORI_DISP_TIMING)
          long long _m2 = clock64();
          _mHT += (unsigned long long)(_m2 - _m1);




          
          _mF[field] += (unsigned long long)(_m2 - _m1);
#endif
          if (sp.body) {
            __builtin_amdgcn_s_wait_tensorcnt(0);
#if defined(MORI_DISP_TIMING)
            long long _m3 = clock64();
            _mLd += (unsigned long long)(_m3 - _m2);
#endif
            TdmIssueStore<int>(dst + sp.head, tile, g);
            _mPend = true;  // deferred drain, same as the per-run form below
#if defined(MORI_DISP_TIMING)
            _mSt += (unsigned long long)(clock64() - _m3);
#endif
          }
        }
      }
#else
      // Only npes runs exist per block but there are warpNum warps, so cut each peer's run into
      // warpNum/npes contiguous sub-ranges -- every warp keeps exactly one run, one round trip.
      // -DMORI_DISP_METASPLIT=N overrides the cut. Splitting buys warp parallelism but every extra
      // sub-range brings its OWN 128B head/tail remainder, and those remainders do not go through
      // TDM at all -- they are per-lane remote 4B stores, measured at mHT ~32.6k cyc vs mSt ~9.2k
      // for the TDM bodies. So a coarser cut trades parallelism for fewer scalar fragments and more
      // bytes per TDM op; N=1 is one warp per peer, the fewest fragments possible.
#if defined(MORI_DISP_METASPLIT)
      const int split = (MORI_DISP_METASPLIT) > 0 ? (MORI_DISP_METASPLIT) : 1;
#else
      const int split = (npes > 0 && warpNum >= npes) ? (warpNum / npes) : 1;
#endif
      const int nRuns = npes * split;
      for (int r = warpId; r < nRuns; r += warpNum) {
        int peer = r / split;
        int part = r - peer * split;
        // s_N/s_base are this block's own Phase-2 reserve result, still live in LDS -- they are
        // exactly what _cusplit_blkCount/_cusplit_blkBase were written from and no other block ever
        // reads this block's row. Reading the global copies instead costs two SERIALIZED HBM loads at
        // the head of every warp's run: cntAll gates the `continue`, so baseAll cannot issue until it
        // lands. (_cusplit_blkCount/Base are still written there for the degenerate-LDS path below.)
        index_t cntAll = s_N[peer];
        if (cntAll <= 0) continue;
        index_t baseAll = s_base[peer];
        index_t q = cntAll / split, rm = cntAll - q * split;
        index_t myBeg = (index_t)part * q + ((part < rm) ? part : rm);
        index_t myCnt = q + ((part < rm) ? 1 : 0);
        for (index_t cs = 0; cs < myCnt; cs += tokCapM) {
          int cc = (int)((cs + tokCapM <= myCnt) ? tokCapM : (myCnt - cs));
          index_t ab = baseAll + myBeg + cs;
          if (ab + cc > recvCapM) continue;  // OOB guard (peer slot capacity)
          const int nIdxB = cc * tkM, nScIB = cc * sVecM * 4, nWtB = cc * tkM;
          index_t* sI = _cusplit_stgIdx +
                        (size_t)peer * CUSPLIT_MAX_SLOTS_PER_PEER * CUSPLIT_MAX_TOPK + (size_t)ab * tkM;
          float* sW = _cusplit_stgWt +
                      (size_t)peer * CUSPLIT_MAX_SLOTS_PER_PEER * CUSPLIT_MAX_TOPK + (size_t)ab * tkM;
          uint8_t* sS = _cusplit_stgSc +
                        (size_t)peer * CUSPLIT_MAX_SLOTS_PER_PEER * CUSPLIT_MAX_SCALE_BYTES +
                        (size_t)ab * sBytesM;
          index_t* sR = _cusplit_stgSrc + (size_t)peer * CUSPLIT_MAX_SLOTS_PER_PEER + (size_t)ab;
          index_t* dI = args.shmemOutIndicesMemObj->template GetAs<index_t*>(peer) + (size_t)ab * tkM;
          float* dW = args.weightsBuf ? (args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(peer) +
                                         (size_t)ab * tkM)
                                      : nullptr;
          uint8_t* dS = doScaleM ? (args.shmemOutScalesMemObj->template GetAs<uint8_t*>(peer) +
                                    (size_t)ab * sBytesM)
                                 : nullptr;
          index_t* dR = args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(peer) + (size_t)ab;
#if defined(MORI_DISP_METAVEC)
          // ---- Direct staging -> peer copy, NO LDS BOUNCE AND NO DRAIN WAIT. The TDM form below
          // has to s_wait_tensorcnt(0) before the payload phase can reuse the LDS tile, and that
          // drain is the meta phase's largest bucket (mSt ~21us at EP4-4K vs mSt+mHT ~30us total).
          // It is not a bandwidth cost: meta is only ~2.9MB, i.e. ~97 GB/s, against the payload's
          // ~1600 GB/s -- it is the round trip to the peer. Plain stores own no tile, so nothing
          // has to wait for them here and they drain across the payload phase instead, which is the
          // window the meta-before-payload ordering exists to provide (see the DRAIN WINDOW note
          // above). The warp-level drain that makes them visible before this rank signals
          // completion is the __threadfence_system() after the payload loop.
#if defined(MORI_DISP_TIMING)
          long long _m1 = clock64();
          long long _mFp = _m1;
#endif
          WarpBatchCopy4<int>(reinterpret_cast<int*>(dI), reinterpret_cast<const int*>(sI), nIdxB,
                              laneId);
          _MHTS(0);
          if (dW)
            WarpBatchCopy4<int>(reinterpret_cast<int*>(dW), reinterpret_cast<const int*>(sW), nWtB,
                                laneId);
          _MHTS(1);
          // scale is sBytesM per token and both sides are base + ab*sBytesM, so 16B-aligned by
          // construction -- copy it as uint4 to move 4x the bytes per in-flight load.
          if (dS)
            WarpBatchCopy4<uint4>(reinterpret_cast<uint4*>(dS), reinterpret_cast<const uint4*>(sS),
                                  cc * sVecM, laneId);
          _MHTS(2);
          WarpBatchCopy4<index_t>(dR, sR, cc, laneId);
          _MHTS(3);
#if defined(MORI_DISP_TIMING)
          _mHT += (unsigned long long)(clock64() - _m1);
#endif
#else
          // Per-field 128B-aligned split. Each field's LDS region is padded up to a 128B multiple
          // so the tile's LDS side is aligned too (tokCapM already reserves the slack).
          const TdmSplit128 spI = TdmWholeOrSplit128((size_t)ab * tkM, nIdxB);
          const TdmSplit128 spW = (dW != nullptr) ? spI : TdmSplit128{0, 0, 0};
          const TdmSplit128 spS = (dS != nullptr)
                                      ? TdmWholeOrSplit128((size_t)ab * sVecM * 4, nScIB)
                                      : TdmSplit128{0, 0, 0};
          const TdmSplit128 spR = TdmWholeOrSplit128((size_t)ab, cc);
          int* tI = reinterpret_cast<int*>(_m4);
          int* tW = tI + ((spI.body + 31) & ~31);
          int* tS = tW + ((spW.body + 31) & ~31);
          int* tR = tS + ((spS.body + 31) & ~31);
          gfx1250_TDM_GROUP1 gI{}, gW{}, gS{}, gR{};
#if defined(MORI_DISP_METALDS)
          // FINALIZE already gathered this run into LDS with the peer-side 128B phase, so the tile
          // is the SOURCE here and there is nothing to load. Each field's chunk starts at its own
          // block-local slot, which keeps the phase identical for every split part and chunk.
          // Computed unconditionally so every one of these stays an LDS pointer -- mixing in a
          // global (or null) alternative would sink the address space to generic.
          const bool _useL = (s_mOk != 0);
          int* _lb = reinterpret_cast<int*>(_tdmBatchSmem);
          const size_t _sl = (size_t)(myBeg + cs);
          int* lI = _lb + s_mOff[peer * 4 + 0] + _sl * tkM;
          int* lW = _lb + s_mOff[peer * 4 + 1] + _sl * tkM;
          int* lS = _lb + s_mOff[peer * 4 + 2] + _sl * sVecM * 4;
          int* lR = _lb + s_mOff[peer * 4 + 3] + _sl;
#else
          constexpr bool _useL = false;
          int* lI = tI; int* lW = tI; int* lS = tI; int* lR = tI;
#endif
#if defined(MORI_DISP_METADIAG)
          if (laneId == 0 && !args.replayMode) {
            atomicAdd(&_meta_ccHist[(cc < 128) ? cc : 127], 1u);
            atomicAdd(&_meta_kindHist[(spI.body == 0) ? 2 : ((spI.rows == 0) ? 0 : 1)], 1u);
            atomicAdd(&_meta_phaseHist[(int)(((size_t)ab * tkM) & 31)], 1u);
          }
#endif
          // The loads below overwrite the tile, so a previous run's stores must be done reading it
          // first. At the default geometry each warp owns a single run and this never fires; it only
          // pays off when a warp carries several runs or a chunked run (cc > tokCapM).
          if (_mPend) {
#if defined(MORI_DISP_TIMING)
            long long _md0 = clock64();
#endif
            __builtin_amdgcn_s_wait_tensorcnt(0);
            _mPend = false;
#if defined(MORI_DISP_TIMING)
            _mDrain += (unsigned long long)(clock64() - _md0);
#endif
          }
#if defined(MORI_DISP_TIMING)
          long long _m0 = clock64();
#endif
          if (spI.body) gI = TdmSplitShape(spI, spI.body);
          if (spW.body) gW = TdmSplitShape(spW, spW.body);
          if (spS.body) gS = TdmSplitShape(spS, spS.body);
          if (spR.body) gR = TdmSplitShape(spR, spR.body);
          if (!_useL) {
            if (spI.body) TdmIssueLoad<int>(tI, reinterpret_cast<int*>(sI + spI.head), gI);
            if (spW.body) TdmIssueLoad<int>(tW, reinterpret_cast<int*>(sW + spW.head), gW);
            if (spS.body) TdmIssueLoad<int>(tS, reinterpret_cast<int*>(sS) + spS.head, gS);
            if (spR.body) TdmIssueLoad<int>(tR, reinterpret_cast<int*>(sR + spR.head), gR);
          }
          // Unaligned head/tail (and any field too small for 2 rows) go straight global->global,
          // issued here so they overlap the TDM loads already in flight instead of serializing.
#if defined(MORI_DISP_TIMING)
          long long _m1 = clock64();
          long long _mFp = _m1;
          _mIssue += (unsigned long long)(_m1 - _m0);
#endif
          // Head/tail remainders, one branch per address space for the same reason as the gather:
          // a single source pointer fed from both LDS and global would make every access flat.
#define _MHT_REM(dstp, ldsp, glbp, hd, bd, ntot)                                   \
  do {                                                                             \
    if (_useL) {                                                                   \
      for (int i = laneId; i < (hd); i += warpSize) (dstp)[i] = (ldsp)[i];          \
      for (int i = (hd) + (bd) + laneId; i < (ntot); i += warpSize)                 \
        (dstp)[i] = (ldsp)[i];                                                      \
    } else {                                                                       \
      for (int i = laneId; i < (hd); i += warpSize) (dstp)[i] = (glbp)[i];          \
      for (int i = (hd) + (bd) + laneId; i < (ntot); i += warpSize)                 \
        (dstp)[i] = (glbp)[i];                                                      \
    }                                                                              \
  } while (0)
          _MHT_REM(reinterpret_cast<int*>(dI), lI, reinterpret_cast<int*>(sI), spI.head, spI.body,
                   nIdxB);
          _MHTS(0);
          if (dW)
            _MHT_REM(reinterpret_cast<int*>(dW), lW, reinterpret_cast<int*>(sW), spW.head, spW.body,
                     nWtB);
          _MHTS(1);
          if (dS)
            _MHT_REM(reinterpret_cast<int*>(dS), lS, reinterpret_cast<int*>(sS), spS.head, spS.body,
                     nScIB);
          _MHTS(2);
#if defined(MORI_DISP_SRCVEC)
          // body==0 means the run got no TDM tile at all and the "peel" is the whole run -- always
          // the case for srcmap at cc < 64. That is the one place worth vectorising; when a body does
          // exist the peel is under 32 elements and the plain loop is fine.
          if (spR.body == 0) {
            if (_useL)
              WarpCopyRun4B(dR, reinterpret_cast<const index_t*>(lR), cc, laneId);
            else
              WarpCopyRun4B(dR, sR, cc, laneId);
          } else
#endif
          _MHT_REM(dR, reinterpret_cast<index_t*>(lR), sR, spR.head, spR.body, cc);
          _MHTS(3);
#undef _MHT_REM
#if defined(MORI_DISP_TIMING)
          long long _m2 = clock64();
          _mHT += (unsigned long long)(_m2 - _m1);
#endif
          if (spI.body || spW.body || spS.body || spR.body) {
            // Nothing to wait for when LDS is the source: FINALIZE's gather plus the __syncthreads()
            // after it already made every slot visible to every warp in the block.
            if (!_useL) __builtin_amdgcn_s_wait_tensorcnt(0);
#if defined(MORI_DISP_TIMING)
            long long _m3 = clock64();
            _mLd += (unsigned long long)(_m3 - _m2);
#endif
            if (spI.body)
              TdmIssueStore<int>(reinterpret_cast<int*>(dI + spI.head), _useL ? (lI + spI.head) : tI, gI);
            if (spW.body)
              TdmIssueStore<int>(reinterpret_cast<int*>(dW + spW.head), _useL ? (lW + spW.head) : tW, gW);
            if (spS.body)
              TdmIssueStore<int>(reinterpret_cast<int*>(dS) + spS.head, _useL ? (lS + spS.head) : tS, gS);
            if (spR.body)
              TdmIssueStore<int>(reinterpret_cast<int*>(dR + spR.head), _useL ? (lR + spR.head) : tR, gR);
#if defined(MORI_DISP_METALDS)
            // The LDS these stores read is shared by the WHOLE BLOCK, so the drain cannot be deferred
            // past the separating __syncthreads() the way the per-warp-private tile's is: after that
            // barrier any warp's payload load may overwrite another warp's meta run.
            // It is deferred to just BEFORE that barrier instead (see _lPend below), which still lets
            // all four stores of all chunks overlap.
            if (_useL) _lPend = true; else
#endif
            // Do NOT wait here. Nothing this warp does between here and the payload phase touches
            // the tile, and the __syncthreads() in between already makes every warp wait for the
            // slowest meta warp in the block -- so the drain is paid out of time that is otherwise
            // spent idle at that barrier. mSt therefore measures store ISSUE only.
            _mPend = true;
#if defined(MORI_DISP_TIMING)
            _mSt += (unsigned long long)(clock64() - _m3);
#endif
          }
#endif  // MORI_DISP_METAVEC
        }
      }
#endif  // MORI_DISP_METAFIELD
#if defined(MORI_DISP_TIMING)
      if (laneId == 0) {
        atomicMax(&_meta_issue_maxdur, _mIssue);
        atomicMax(&_meta_ht_maxdur, _mHT);
        atomicMax(&_meta_ld_maxdur, _mLd);
        atomicMax(&_meta_st_maxdur, _mSt);
        for (int _q = 0; _q < 4; ++_q) atomicMax(&_meta_ht_f[_q], _mF[_q]);
      }
#endif
    } else {
      // Degenerate LDS budget: hiddenDim * sizeof(T) cannot hold even one token's four fields, so
      // there is no tile to bounce through. Copy global->global instead, one (peer, field) item per
      // warp. Correctness fallback only -- no shipped EP config reaches it (hidden 7168 bf16 fits
      // ~73 tokens per warp tile), so it is deliberately kept simple rather than tuned.
      const int nItems = npes * kMetaFields;
      for (int item = warpId; item < nItems; item += warpNum) {
        int peer = item / kMetaFields;
        int field = item - peer * kMetaFields;  // 0=idx, 1=wt, 2=scale, 3=srcmap
        if (field == 1 && !args.weightsBuf) continue;
        if (field == 2 && !doScaleM) continue;
        index_t cnt = _cusplit_blkCount[(size_t)blockIdx.x * npes + peer];
        if (cnt <= 0) continue;
        index_t ab = _cusplit_blkBase[(size_t)blockIdx.x * npes + peer];
        if (ab + cnt > recvCapM) continue;  // OOB guard (peer slot capacity)
        if (field == 0) {
          index_t* src = _cusplit_stgIdx +
                         (size_t)peer * CUSPLIT_MAX_SLOTS_PER_PEER * CUSPLIT_MAX_TOPK + (size_t)ab * tkM;
          index_t* dst = args.shmemOutIndicesMemObj->template GetAs<index_t*>(peer) + (size_t)ab * tkM;
          for (int i = laneId; i < (int)cnt * tkM; i += warpSize) dst[i] = src[i];
        } else if (field == 1) {
          float* src = _cusplit_stgWt +
                       (size_t)peer * CUSPLIT_MAX_SLOTS_PER_PEER * CUSPLIT_MAX_TOPK + (size_t)ab * tkM;
          float* dst = args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(peer) + (size_t)ab * tkM;
          for (int i = laneId; i < (int)cnt * tkM; i += warpSize) dst[i] = src[i];
        } else if (field == 2) {
          uint8_t* src = _cusplit_stgSc +
                         (size_t)peer * CUSPLIT_MAX_SLOTS_PER_PEER * CUSPLIT_MAX_SCALE_BYTES +
                         (size_t)ab * sBytesM;
          uint8_t* dst = args.shmemOutScalesMemObj->template GetAs<uint8_t*>(peer) + (size_t)ab * sBytesM;
          for (int c = laneId; c < (int)cnt * sVecM; c += warpSize)
            reinterpret_cast<uint4*>(dst)[c] = reinterpret_cast<uint4*>(src)[c];
        } else {
          index_t* src = _cusplit_stgSrc + (size_t)peer * CUSPLIT_MAX_SLOTS_PER_PEER + (size_t)ab;
          index_t* dst = args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(peer) + (size_t)ab;
          for (int i = laneId; i < (int)cnt; i += warpSize) dst[i] = src[i];
        }
      }
    }
  }
#if defined(MORI_DISP_TIMING)
  // EVERY warp reports: the busiest meta warp may live in any block, and a single-warp probe
  // (the previous globalWarpId==0 form) reports a duration that is not the grid maximum.
  if (laneId == 0) atomicMax(&_meta_blk_maxdur, (unsigned long long)(clock64() - _mt0b));
#endif
#if defined(MORI_DISP_METALDS)
  // BEFORE the barrier, not after: these stores read LDS that belongs to the whole block, so once any
  // warp is past the barrier and issuing payload loads, another warp's still-reading meta store would
  // see overwritten data. All four fields of all this warp's runs are in flight by now, so this waits
  // on the last one only.
  if (_lPend) {
    __builtin_amdgcn_s_wait_tensorcnt(0);
    _lPend = false;
  }
#endif
  __syncthreads();   // all meta warps done before reusing _tdmBatchSmem for the payload tile
  // Pay whatever is left of the deferred drain, before the payload phase's first TdmIssueLoad
  // overwrites the tile these stores are still reading. Everything that ran between the store issue
  // and here -- the rest of this warp's runs, and the barrier's wait for the slowest meta warp in
  // the block -- came for free. It stays inside the metasend bucket (_pbStart is stamped after this
  // point), so [CUSPLIT] metasend remains comparable to the immediate-wait form.
  if (_mPend) {
#if defined(MORI_DISP_TIMING)
    long long _md1 = clock64();
#endif
    __builtin_amdgcn_s_wait_tensorcnt(0);
#if defined(MORI_DISP_TIMING)
    _mDrain += (unsigned long long)(clock64() - _md1);
#endif
  }
#if defined(MORI_DISP_TIMING)
  if (laneId == 0) atomicMax(&_meta_drain_maxdur, _mDrain);
#endif
#endif  // MORI_DISP_TDM && gfx125x (per-block meta send)

#if defined(MORI_DISP_TIMING)
  if (thdId == 0) _pbStart = clock64();  // Part-B (payload send) start = right before payload -> isolates token-send BW
#endif

#if defined(MORI_DISP_TDM) && (defined(__gfx1250__) || defined(__gfx1251__))
  // ---- Phase 3b: payload copy, driven by the slot map (dispDestTokIdMap, own-block). ----
#if defined(MORI_DISP_NOPAY)
  // DIAGNOSTIC ONLY, PRODUCES WRONG RESULTS. See MORI_DISP_NOMETA: kernel(full) - kernel(NOPAY)
  // is the payload phase's real cost, which is what the 1.3TB/s budget has to be measured against.
  if (false) {
#else
  if (args.tokenIndices && args.inpTokenBuf && !args.replayMode) {
#endif
    // Reuses aWarp/aWarps rather than recomputing them: the block-level __syncthreads() above
    // stands in for a grid barrier ONLY because a block reads back exactly the dispDestTokIdMap
    // entries it wrote itself, so this loop must walk the same token set COUNT and FINALIZE did.
    // A different partition here races on slot ids written by other blocks and lands payloads in
    // the wrong slots (it silently did, until acc_check caught it).
#if defined(MORI_DISP_TIMING)
    unsigned long long _pLd = 0ull, _pStI = 0ull, _pDr = 0ull;
#endif
#if defined(MORI_DISP_PAYDYN)
    // ---- Block-internal DYNAMIC work distribution.
    //
    // The kernel ends when the slowest block ends, and a block ends when its slowest warp ends. Every
    // warp gets the same NUMBER of tokens, but a token goes to 1..4 destinations depending on routing,
    // so equal token counts are not equal bytes and the unluckiest of the 8 warps holds its whole
    // block back. That is pure loss: the other 7 warps sit drained while it finishes.
    //
    // This does NOT touch the partition rule above. What must be preserved is which BLOCK owns which
    // tokens -- a block reads back exactly the dispDestTokIdMap entries it wrote itself, which is the
    // only reason a plain __syncthreads() can stand in for a grid barrier here. WHICH WARP inside the
    // block takes a given token is free, because the map is indexed by token, not by warp. So the
    // block's token set is bit-for-bit the same set the static form walked; only the assignment of
    // units to warps within the block becomes demand-driven.
    //
    // Unit = _tpi consecutive tokens, keeping the lane->token mapping identical. Unit u maps back to
    // the static form's (warp slot, round) pair, so the union over u is exactly the union over the
    // block's warps and rounds. An LDS atomic costs tens of cycles and is claimed nUnits times per
    // block (16 at DBN=64/wpb=8), against a phase that runs ~125us.
    __shared__ int s_payNext;
    if (thdId == 0) s_payNext = 0;
    __syncthreads();
    const int _blkStride = aWarps * _tpi;                  // distance to this block's next round
    const int _blkBase = blockIdx.x * warpNum * _tpi;      // first token of this block's round 0
    const int _nUnits =
        ((args.curRankNumToken + _blkStride - 1) / _blkStride) * warpNum;
    for (;;) {
      int _u = 0;
      if (laneId == 0) _u = atomicAdd(&s_payNext, 1);
      _u = __shfl(_u, 0);
      if (_u >= _nUnits) break;
      const int tokBase = _blkBase + (_u % warpNum) * _tpi + (_u / warpNum) * _blkStride;
      // Rounds do not divide evenly in general, so a unit past the end is skipped, not a loop exit --
      // other warp slots in the same round may still have work.
      if (tokBase >= args.curRankNumToken) continue;
#else
    for (int tokBase = aWarp * _tpi; tokBase < args.curRankNumToken; tokBase += aWarps * _tpi) {
#endif
     for (int _sub = 0; _sub < _tpi; ++_sub) {
      int tok = tokBase + _sub;
      if (tok >= args.curRankNumToken) break;
      index_t flatMe = (laneId < topk)
                           ? args.dispDestTokIdMap[(size_t)tok * topk + laneId]
                           : FlatTokenIndex(config, config.worldSize, 0);
      index_t peMe = PeFromFlatTokenIndex(config, flatMe);
      int validMe = (laneId < topk && peMe < (index_t)npes) ? 1 : 0;
      if (!__any(validMe)) continue;  // token routed nowhere -> no load
#if defined(MORI_DISP_PAYSPLIT) && (MORI_DISP_PAYSPLIT > 1)
      // ---- One token, issued as _pnS segments of the same LDS tile.
      //
      // The payload phase is bounded by how many TDM operations are in flight, not by bytes. The
      // evidence is the geometry sweep: 64x8 (512 warps) gives 1278 GB/s, and BOTH 64x16 and 128x8
      // (1024 warps either way, one buying it with LDS, the other with CUs) land on 1366 -- the same
      // ceiling from the same in-flight count, reached by two different resources. The unsegmented
      // form leaves each warp with 1 load and then ~3.6 stores in flight (one per destination of one
      // token), and the next token cannot start until the drain, because its load would overwrite the
      // tile the stores are reading.
      //
      // Splitting the token multiplies that in-flight count by _pnS while the tile stays exactly
      // hiddenDim elements: same 14336B per warp, same 114KB per block, same 64 blocks. What it costs
      // is _pnS x the TDM operations for the same bytes, so it only pays if per-op overhead is small
      // next to a segment -- at _pnS=2 a segment is still 7168B, 56x the 128B minimum row.
      if (_pSplitOk) {
        for (int _g = 0; _g < _pnS; ++_g)
          TdmIssueLoad<T>(_tdmTile + _g * _pSeg,
                          args.inpTokenBuf + (size_t)tok * hiddenDim + _g * _pSeg, _tdmGS);
      } else {
        TdmIssueLoad<T>(_tdmTile, args.inpTokenBuf + (size_t)tok * hiddenDim, _tdmG1);
      }
#else
      TdmIssueLoad<T>(_tdmTile, args.inpTokenBuf + (size_t)tok * hiddenDim, _tdmG1);
#endif
      bool loadWaited = false;
#if defined(MORI_DISP_TIMING)
      long long _p0 = clock64();
      long long _p1 = _p0;
#endif
      for (int l = 0; l < topk; ++l) {
        if (!__shfl(validMe, l)) continue;            // fixed l -> uniform shfl
        index_t flat = __shfl(flatMe, l);
        index_t destPe = PeFromFlatTokenIndex(config, flat);
        index_t destTokId = LocalTokIdFromFlatTokenIndex(config, flat);
        if (!loadWaited) {
          __builtin_amdgcn_s_wait_tensorcnt(0);
          loadWaited = true;
#if defined(MORI_DISP_TIMING)
          _p1 = clock64();
          _pLd += (unsigned long long)(_p1 - _p0);
#endif
        }
#if defined(MORI_DISP_PAYSPLIT) && (MORI_DISP_PAYSPLIT > 1)
        if (_pSplitOk) {
          T* _dB = args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) +
                   (size_t)destTokId * hiddenDim;
          for (int _g = 0; _g < _pnS; ++_g)
            TdmIssueStore<T>(_dB + _g * _pSeg, _tdmTile + _g * _pSeg, _tdmGS);
        } else {
          TdmIssueStore<T>(args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) +
                               (size_t)destTokId * hiddenDim,
                           _tdmTile, _tdmG1);
        }
#else
        TdmIssueStore<T>(args.intraNodeTokBufs.dispatchOut->template GetAs<T*>(destPe) +
                             (size_t)destTokId * hiddenDim,
                         _tdmTile, _tdmG1);
#endif
#if defined(MORI_DISP_METAFUSE)
        // ---- Meta, fused into the payload loop, replacing the whole meta phase (which this gate
        // also compiles away, together with its staging gather under MORI_DISP_NOSTG).
        //
        // The payload TDM store just issued owns the warp's LDS tile until the drain below, so the
        // warp has nothing to do but spin in s_wait_tensorcnt -- these 196B/token of plain
        // cross-GPU stores are paid out of those cycles. That matters because the meta phase costs
        // ~7us (kernel 166.0us -> 159.05us under MORI_DISP_NOMETA) to move 2.9MB, only ~1.8us of
        // which is engine time; the rest is the staging round trip, the LDS bounce and the TDM
        // latency, none of which survive here.
        //
        // destPe/destTokId are warp-uniform (both arrive via __shfl), so the whole warp cooperates
        // on ONE destination and each field is a contiguous burst, same as the staging gather did.
        // The source is the ORIGINAL input, not staging: the payload loop already walks exactly the
        // (token, destination) pairs FINALIZE kept, so staging has no information this does not.
        // Visibility before this rank's completion signal comes from the __threadfence_system()
        // after the loop, the same one the METAVEC path relies on.
        if (destTokId < (index_t)config.MaxNumTokensToRecv()) {
          index_t* _fI =
              args.shmemOutIndicesMemObj->template GetAs<index_t*>(destPe) + (size_t)destTokId * topk;
          for (int e = laneId; e < topk; e += warpSize)
            _fI[e] = args.tokenIndices[(size_t)tok * topk + e];
          if (args.weightsBuf) {
            float* _fW = args.shmemDispatchOutWeightsMemObj->template GetAs<float*>(destPe) +
                         (size_t)destTokId * topk;
            for (int e = laneId; e < topk; e += warpSize)
              _fW[e] = args.weightsBuf[(size_t)tok * topk + e];
          }
          const int _fSB = config.scaleDim * config.scaleTypeSize;
          if (args.scalesBuf && config.scaleDim > 0 && config.scaleTypeSize > 0) {
            // Both sides are base + destTokId * _fSB with _fSB a multiple of 16 at every supported
            // scale config, so uint4 stores are aligned by construction (staging relied on this too).
            uint4* _fS = reinterpret_cast<uint4*>(
                args.shmemOutScalesMemObj->template GetAs<uint8_t*>(destPe) +
                (size_t)destTokId * _fSB);
            const uint4* _fSs =
                reinterpret_cast<const uint4*>(args.scalesBuf + (size_t)tok * _fSB);
            for (int c = laneId; c < (_fSB >> 4); c += warpSize) _fS[c] = _fSs[c];
          }
          if (laneId == 0)
            args.dispTokIdToSrcTokIdMemObj->template GetAs<index_t*>(destPe)[destTokId] =
                FlatTokenIndex(config, myPe, tok);
        }
#endif
      }
#if defined(MORI_DISP_TIMING)
      long long _p2 = clock64();
      _pStI += (unsigned long long)(_p2 - _p1);
#endif
      __builtin_amdgcn_s_wait_tensorcnt(0);   // drain all N stores before reusing tile
#if defined(MORI_DISP_TIMING)
      _pDr += (unsigned long long)(clock64() - _p2);
#endif
     }
    }
#if defined(MORI_DISP_TIMING)
    if (laneId == 0) {
      atomicMax(&_pay_ld_maxdur, _pLd);
      atomicMax(&_pay_sti_maxdur, _pStI);
      atomicMax(&_pay_drain_maxdur, _pDr);
    }
#endif
  }
#if defined(MORI_DISP_METAVEC) || defined(MORI_DISP_METAFUSE)
  // The meta phase left its cross-GPU stores in flight on purpose (no drain wait there). This is
  // where they are made visible, per warp, before any block reports arrival at the completion
  // barrier -- by now the payload phase has given them ~116us to land, so the drain is free.
  __threadfence_system();
#endif
#endif  // MORI_DISP_TDM && gfx125x (payload group)
  __syncthreads();
  _BPTS(3);  // <- phase3 payload copy (Part B: 1D TDM)
#if defined(MORI_DISP_TIMING)
  if (thdId == 0) atomicMax(&_pb_maxdur, (unsigned long long)(clock64() - _pbStart));  // per-block Part-B duration
#endif

  // ---- Completion (identical to legacy): all blocks arrive, then per-peer release-signal ----
#if defined(MORI_DISP_GRIDFLAG)
  // Grid barrier as gridDim.x independent flags instead of one counter.
  //
  // The counter form has every block's thread 0 atomicAdd the SAME address, so the increments
  // serialize on one cacheline while block 0's warp spins reading that very line -- reader and all
  // gridDim.x writers ping-pong it. Here each block writes its own 128B-separated flag (no
  // contention) and the waiter reads gridDim.x of them 32 lanes at a time, 2 rounds at DBN=64.
  //
  // Clearing inside the same launch is safe: a block sets its flag exactly once, after its payload,
  // and the clear runs only once every flag is set, so no block can still be racing to write one.
  if (thdId == 0)
    __hip_atomic_store(&_cusplit_blkArrive[(size_t)blockIdx.x * CUSPLIT_ARRIVE_PAD], (index_t)1,
                       __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
  if (thdId == 0) atomicAdd(args.dispatchGridBarrier, 1);
#endif
  index_t* recvTokenNums = args.recvTokenNumMemObj->template GetAs<index_t*>();
#if defined(MORI_DISP_TIMING)
  index_t _diagOutCnt = 0;  // this rank's OUTGOING per-peer token count (destPe = laneId)
  index_t _diagInCnt = 0;   // this rank's INCOMING per-peer token count (destPe = laneId)
#endif
  if (globalWarpId == 0) {
#if defined(MORI_DISP_GRIDFLAG)
    // All 32 lanes poll, so DBN=64 is 2 loads per sweep instead of one lane re-reading one line.
    const unsigned _nb = gridDim.x;
    for (;;) {
      bool _allIn = true;
      for (unsigned _b = laneId; _b < _nb; _b += warpSize)
        if (__hip_atomic_load(&_cusplit_blkArrive[(size_t)_b * CUSPLIT_ARRIVE_PAD], __ATOMIC_ACQUIRE,
                              __HIP_MEMORY_SCOPE_AGENT) == (index_t)0)
          _allIn = false;
      if (__all(_allIn)) break;
    }
    for (unsigned _b = laneId; _b < _nb; _b += warpSize)
      __hip_atomic_store(&_cusplit_blkArrive[(size_t)_b * CUSPLIT_ARRIVE_PAD], (index_t)0,
                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    _BPTS(4);  // <- grid barrier satisfied (all local blocks arrived)
#endif
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
#if !defined(MORI_DISP_GRIDFLAG)
      _CusplitWaitEq(args.dispatchGridBarrier, (unsigned)gridDim.x);
      _BPTS(4);  // <- grid barrier satisfied (all local blocks arrived)
      __hip_atomic_store(args.dispatchGridBarrier, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
      index_t numTokenSignal = core::AtomicLoadRelaxed(args.destPeTokenCounter + destPe) + 1;
#if defined(MORI_DISP_TIMING)
      _diagOutCnt = numTokenSignal - 1;
#endif
      index_t* signal = args.recvTokenNumMemObj->template GetAs<index_t*>(destPe) + myPe;
      _CusplitWaitEq(signal, (index_t)0);
      __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
      core::AtomicStoreRelaxedSystem(signal, numTokenSignal);
    }
    _BPTS(5);  // <- all per-peer completion signals sent
  }
  if (globalWarpId == 0) {
    for (int destPe = laneId; destPe < npes; destPe += warpSize) {
      index_t* signal = recvTokenNums + destPe;
      index_t recvTokenNum = _CusplitWaitGt(signal, (index_t)0) - 1;
      __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
      core::AtomicStoreRelaxedSystem(signal, 0);
      atomicAdd(args.totalRecvTokenNum, recvTokenNum);
      args.destPeTokenCounter[destPe] = 0;
#if defined(MORI_DISP_TIMING)
      _diagInCnt = recvTokenNum;
#endif
    }
    if (laneId == 0) {
      args.dispTokOffsetMemObj->template GetAs<index_t*>()[0] = 0;
    }
    _BPTS(6);  // <- all peers' signals received (completion done)
#if defined(MORI_DISP_TIMING)
    if (!args.replayMode) {
      index_t out0 = __shfl(_diagOutCnt, 0), out1 = __shfl(_diagOutCnt, 1);
      index_t out2 = __shfl(_diagOutCnt, 2), out3 = __shfl(_diagOutCnt, 3);
      index_t in0 = __shfl(_diagInCnt, 0), in1 = __shfl(_diagInCnt, 1);
      index_t in2 = __shfl(_diagInCnt, 2), in3 = __shfl(_diagInCnt, 3);
      if (laneId == 0) {
        unsigned long long _cc = atomicAdd(&_cusplit_diag_call_idx, 1ull);
        if (_cc >= 2ull && _cc < 13ull)
          printf("[PEERCNT] rank=%d call=%llu out(0..3)=%d,%d,%d,%d in(0..3)=%d,%d,%d,%d\n", myPe, _cc,
                 (int)out0, (int)out1, (int)out2, (int)out3, (int)in0, (int)in1, (int)in2, (int)in3);
      }
    }
#endif
  }
#if defined(MORI_DISP_METADIAG)
  if (blockIdx.x == 0 && thdId == 0 && !args.replayMode) {
    __threadfence();
    // Histograms accumulate across launches; every steady-state launch has the same geometry, so
    // sampling one later call is enough and no reset is needed.
    if (atomicAdd(&_meta_diag_call_idx, 1ull) == 4ull) {
      printf("[METASHAPE] rank=%d topk=%d whole=%u split=%u scalar=%u\n", myPe,
             (int)config.numExpertPerToken, _meta_kindHist[0], _meta_kindHist[1],
             _meta_kindHist[2]);
      for (int _q = 0; _q < 128; ++_q)
        if (_meta_ccHist[_q])
          printf("[METASHAPE] rank=%d cc=%d n=%u\n", myPe, _q, _meta_ccHist[_q]);
      for (int _q = 0; _q < 32; ++_q)
        if (_meta_phaseHist[_q])
          printf("[METASHAPE] rank=%d phase=%d n=%u\n", myPe, _q, _meta_phaseHist[_q]);
    }
  }
#endif
#if defined(MORI_DISP_TIMING)
  // [CUSPLIT] Part-B (payload send) isolation: partB_maxdur = busiest block's payload-phase
  // duration (post-barrier). frac = partB / kernel-wall -> PARTB_BW = dispatch_BW / frac.
  if (blockIdx.x == 0 && thdId == 0 && !args.replayMode) {
    __threadfence();
    long long tot = _pt[6] - _pt[0];
    unsigned long long _callIdx = atomicAdd(&_cusplit_timing_call_idx, 1ull);
    if (_callIdx == 2ull)  // launch geometry, once -- settles warpNum/_tpi/tokens-per-warp questions
      printf("[GEOM] rank=%d gridDim=%d blockDim=%d warpSize=%d warpNum=%d aWarps=%d numToken=%d topk=%d npes=%d eprk=%d tpi=%d tokPerWarp=%.2f\n",
             myPe, (int)gridDim.x, (int)blockDim.x, (int)warpSize, warpNum, aWarps,
             (int)args.curRankNumToken, topk, npes, config.numExpertPerRank, _tpi,
             (double)args.curRankNumToken / (double)aWarps);
    if (_callIdx >= 2ull && _callIdx < 13ull)  // [DIAG] print regardless of tot (completion may be slow)
      printf("[DIAG] rank=%d call=%llu partB=%.1fus metablk=%.1fus cbar=%.1fus csig=%.1fus cwait=%.1fus tot=%.1fus cap=%d\n",
             myPe, _callIdx, _pb_maxdur / 2270.0,
             _meta_blk_maxdur / 2270.0, (_pt[4] - _pt[3]) / 2270.0,
             (_pt[5] - _pt[4]) / 2270.0, (_pt[6] - _pt[5]) / 2270.0, tot / 2270.0,
             config.MaxNumTokensToRecv());
    if (tot > 0 && tot < 20000000LL && _callIdx >= 3ull && _callIdx < 13ull) {
      long long p1 = _pt[1] - _pt[0], p2 = _pt[2] - _pt[1];
      long long cpl = _pt[6] - _pt[3];
      // Same completion sub-phase breakdown as the legacy BPHASE block above (barrier wait /
      // per-peer signal send / wait-for-peer-recv) -- _pt[3..6] are already stamped by the
      // identical _BPTS(3..6) call sites, this just reuses them instead of new instrumentation.
      long long cbar = _pt[4] - _pt[3], csig = _pt[5] - _pt[4], cwait = _pt[6] - _pt[5];
      // assign = FINALIZE across all blocks; metasend = the span from FINALIZE's end to the start of
      // this block's payload phase (i.e. the per-block meta send); own3b = block0's own Part-B span.
      long long p3assign = _pt[7] - _pt[2], p3meta = _pbStart - _pt[7], p3own = _pt[3] - _pbStart;
      double frac = (double)_pb_maxdur / (double)tot;
      printf("[CUSPLIT] rank=%d count=%lld reserve=%lld assign=%lld metasend=%lld own3b=%lld partB_maxdur=%llu compl=%lld kernel=%lld frac=%.4f => PARTB_BW=dispatch_BW/%.4f | compl(barrier=%lld sigsend=%lld waitpeer=%lld) | meta(send_maxdur=%llu)\n",
             myPe, p1, p2, p3assign, p3meta, p3own, _pb_maxdur, cpl, tot, frac, frac,
             cbar, csig, cwait, _meta_blk_maxdur);
      // Maxima over warps of each sub-bucket -- see the _meta_issue_maxdur comment: additive only
      // if the warps are homogeneous, so mIssue+mHT+mLd+mSt is printed against send_maxdur above
      // and fRoute+fStg against the assign bucket, as the homogeneity check.
      printf("[MSPLIT] rank=%d mIssue=%llu mHT=%llu mLd=%llu mSt=%llu mDrain=%llu | fRoute=%llu fStg=%llu | htIdx=%llu htWt=%llu htSc=%llu htSrc=%llu | pLd=%llu pStI=%llu pDrain=%llu\n",
             myPe, _meta_issue_maxdur, _meta_ht_maxdur, _meta_ld_maxdur, _meta_st_maxdur,
             _meta_drain_maxdur, _fin_route_maxdur, _fin_stg_maxdur, _meta_ht_f[0], _meta_ht_f[1],
             _meta_ht_f[2], _meta_ht_f[3], _pay_ld_maxdur, _pay_sti_maxdur, _pay_drain_maxdur);
    }
    // Without these resets the atomicMax globals are running maxima over EVERY launch since
    // module load, so one cold launch pins them forever and no per-call value can be read out.
    // There is no grid barrier in this body, so a straggler block can still atomicMax after this
    // reset and have its duration attributed to the next call -- accepted, since these two are
    // MORI_DISP_TIMING-only reporting and the printed value is a max over many calls anyway.
    _pb_maxdur = 0ull;
    _meta_blk_maxdur = 0ull;
    _meta_issue_maxdur = 0ull;
    _meta_ht_maxdur = 0ull;
    _meta_ld_maxdur = 0ull;
    _meta_st_maxdur = 0ull;
    _meta_drain_maxdur = 0ull;
    _pay_ld_maxdur = 0ull;
    _pay_sti_maxdur = 0ull;
    _pay_drain_maxdur = 0ull;
    _fin_route_maxdur = 0ull;
    _fin_stg_maxdur = 0ull;
    for (int _q = 0; _q < 4; ++_q) _meta_ht_f[_q] = 0ull;
  }
#endif
#undef _BPTS
#ifdef ENABLE_STANDARD_MOE_ADAPT
  if constexpr (EnableStdMoE) {
    InvokeConvertDispatchOutput<T>(args, myPe);
  }
#endif
}

// Body selector. The extern-C launch symbol EpDispatchIntraNodeBatchKernel_<dtype> calls *_body
// directly (see WRAP_BOOL in ep_common.hip) and bypasses the __global__ wrapper below, so the
// switch has to live here. Default is EpDispatchIntraNodeKernel_body (64 blocks x 8 warps);
// -DMORI_DISP_CLEAN selects the legacy EpDispatchIntraNodeKernel_clean_body (256 x 16).
template <typename T, bool EnableStdMoE = false>
__device__ void EpDispatchIntraNodeBatchKernel_body(EpDispatchCombineArgs<T> args) {
#ifdef MORI_DISP_CLEAN
  EpDispatchIntraNodeKernel_clean_body<T, EnableStdMoE>(args);
#else
  EpDispatchIntraNodeKernel_body<T, EnableStdMoE>(args);
#endif
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
