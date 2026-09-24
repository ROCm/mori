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

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define MORI_TDM_FN __host__ __device__ __forceinline__
// Keeps the whole-run planner out of the hot path. That path sits between the chunk loop's
// s_wait_tensorcnt(0) and the next issue, so scalar work there delays every transfer in a
// serialized chain. This cuts ep_dispatch from 193 extra instructions to 51; it is NOT what the
// 4us in the MORI_TDM_WMAX note came from, see there.
#define MORI_TDM_COLD __host__ __device__ __attribute__((noinline))
#else
#define MORI_TDM_FN inline
#define MORI_TDM_COLD inline
#endif

namespace mori {
namespace tdm {

constexpr int kTdmRowBytes = 128;
constexpr int kTdmRowElems4B = kTdmRowBytes / 4;

// Setting this to 0 restores the head/body/tail-only planner exactly: every run that has no
// 128B-legal body goes back to being copied by scalar cross-card stores. It is the A/B switch
// for the whole-run tile, and it sits on the single decision point every metadata field of
// every run passes through, so an arm built with 0 really is the old code.
#ifndef MORI_TDM_WHOLE
#define MORI_TDM_WHOLE 1
#endif

// Longest run still eligible for a whole-run tile. This is an EMPIRICAL boundary, not a derived
// one: runs a little longer than this are faster left on the scalar path.
//
// Measured on f01-1 2026-09-13, ct=16384 topk=9 bf16, one round per cap, W1 against the same tree
// with the tile off:
//
//   cap 48        -2.5us   ct=512 topk=6 keeps its full -10.7us
//   cap 63        +4.2us
//   cap 64        +4.3us
//   cap 94        +4.0us
//   no cap        +4.8us
//
// So some run between 49 and 63 elements long is the whole difference at ct=16384. Which one is
// not known -- do not repeat the guess that it is the 64-element source-map run, that is what
// caps 63 and 64 were built to test and both of them lost.
//
// What it is NOT: the planner's own code. Three versions of this file put 134, 193 and 51 extra
// instructions in ep_dispatch and all three measured the same +4us, so instruction count and the
// timing move independently here.
#ifndef MORI_TDM_WMAX
#define MORI_TDM_WMAX 48
#endif

struct TdmSplit128 {
  int head;
  int body;
  // 0 with body > 0 marks a whole-run tile: one descriptor covering the entire run at whatever
  // 128B phase it starts on. > 0 is the row count of a 128B-aligned body.
  int rows;
  // The tile geometry, decided once here rather than recomputed at each issue site. Deriving it
  // there instead costs 134 instructions in ep_dispatch: the issue site cannot see which kind of
  // split it holds, so the row width stops being the constant 128B/4 it is for every row-addressed
  // body and the address arithmetic around all eight sites stops folding. Measured 2026-09-13 on
  // f01-1: 1791 instructions with the derivation, 1657 with it hoisted here.
  int dim0;
  int dim1;
};

MORI_TDM_FN TdmSplit128 TdmScalarOnly(int nElems) {
  return TdmSplit128{nElems > 0 ? nElems : 0, 0, 0, 0, 0};
}

MORI_TDM_FN TdmSplit128 TdmRowSplit(int head, int rows) {
  return TdmSplit128{head, rows * kTdmRowElems4B, rows, kTdmRowElems4B, rows};
}

// Whole-run tile geometry, by closed form. Try dim1 = 8, 4, 2 largest first so the row stays as
// narrow as the 128B bandwidth floor allows, and take the first exact divisor whose row still
// reaches 32 elements. Returns 0 when no divisor clears the floor.
MORI_TDM_FN int TdmCheapDim1(int nElems) {
  if ((nElems & 7) == 0 && (nElems >> 3) >= kTdmRowElems4B) return 8;
  if ((nElems & 3) == 0 && (nElems >> 2) >= kTdmRowElems4B) return 4;
  if ((nElems & 1) == 0 && (nElems >> 1) >= kTdmRowElems4B) return 2;
  return 0;
}

// True when nElems can be covered by ONE tile whose two dims are both >= 2 and whose product is
// exactly nElems. Both conditions matter: >= 2 because gfx1250 has no 1xN wedge, and exact
// because a tile even one element larger writes past the end of the run, into whichever field
// the pool laid down next.
MORI_TDM_FN bool TdmWholeFits(int nElems) {
  if (TdmCheapDim1(nElems) > 0) return true;
  return nElems >= 4 && (nElems & 1) == 0;
}

MORI_TDM_COLD TdmSplit128 TdmPlanWhole(int nElems) {
  if (nElems <= 0) return TdmSplit128{0, 0, 0, 0, 0};
  if (MORI_TDM_WHOLE != 0 && nElems <= MORI_TDM_WMAX) {
    // A tile is a rectangle, so an odd run cannot be one. Hand the tile nElems - 1 and leave the
    // single odd element to the scalar tail `_MHT_REM` already walks. One scalar cross-card store
    // is not what costs the 6us at ct=512; doing all nElems of them is. This matters more than it
    // looks: the run length is cc for the source map and cc * topk for idx and weights, so an odd
    // cc makes all three odd at once.
    const int cover = (nElems & 1) ? (nElems - 1) : nElems;
    if (TdmWholeFits(cover)) {
      const int c1 = TdmCheapDim1(cover);
      const int d1 = (c1 > 0) ? c1 : 2;
      return TdmSplit128{0, cover, 0, cover / d1, d1};
    }
  }
  return TdmScalarOnly(nElems);
}

// The 128B row floor this splits on is a BANDWIDTH result, not a legality one: 224B rows measure
// ~500 GB/s against ~1500 for 256B rows. A metadata field at ct=512 is 64B..512B, so half
// bandwidth on it is worth nothing measurable, while dropping it off the TDM path costs the whole
// pipeline -- a warp with only one op to issue before its s_wait_tensorcnt(0) exposes both the
// load latency and the cross-card store completion. So when no 128B-legal body exists, cover the
// run with one tile instead of giving up and copying it with scalar cross-card stores.
MORI_TDM_FN TdmSplit128 TdmPlanRun(size_t phase, int nElems) {
  constexpr int P = kTdmRowElems4B;
  if (nElems <= 0) return TdmSplit128{0, 0, 0, 0, 0};
  int head = (int)((P - (phase & (size_t)(P - 1))) & (size_t)(P - 1));
  if (head > nElems) head = nElems;
  const int rows = (nElems - head) / P;
  if (rows >= 2) return TdmRowSplit(head, rows);
  return TdmPlanWhole(nElems);
}

MORI_TDM_FN int TdmSplitDim1(const TdmSplit128& sp) { return sp.dim1; }

MORI_TDM_FN int TdmSplitDim0(const TdmSplit128& sp) { return sp.dim0; }

MORI_TDM_FN bool TdmSplitOk(size_t phase, int nElems, const TdmSplit128& sp) {
  if (sp.head < 0 || sp.body < 0 || sp.rows < 0) return false;
  if (sp.head + sp.body > nElems) return false;
  if (sp.body == 0) return sp.rows == 0 && sp.head == (nElems > 0 ? nElems : 0);
  if (sp.rows == 0) {
    // Whole-run tile. It owes nothing to the 128B grid, but it has to start where the run starts,
    // it may leave at most the one odd element to the scalar tail, and its two dims must multiply
    // out to exactly what it covers -- a tile even one element larger writes past the end of the
    // run, into whichever field the pool laid down next.
    const int d0 = TdmSplitDim0(sp), d1 = TdmSplitDim1(sp);
    if (sp.head != 0 || nElems - sp.body > 1) return false;
    if (d0 < 2 || d1 < 2) return false;
    return d0 * d1 == sp.body;
  }
  if (((phase + (size_t)sp.head) & (size_t)(kTdmRowElems4B - 1)) != 0) return false;
  if ((sp.body % kTdmRowElems4B) != 0) return false;
  if (sp.rows != sp.body / kTdmRowElems4B) return false;
  // The dims are carried in the struct now, so check the stored pair rather than re-deriving it:
  // a planner that files the wrong geometry would otherwise be validated against its own mistake.
  const int d0 = TdmSplitDim0(sp), d1 = TdmSplitDim1(sp);
  if (d0 != kTdmRowElems4B || d1 != sp.rows) return false;
  if (d0 * d1 != sp.body) return false;
  if (sp.head >= kTdmRowElems4B) return false;
  if (nElems - sp.head - sp.body >= kTdmRowElems4B) return false;
  return true;
}

MORI_TDM_FN bool TdmAddrOnRow(const void* p) {
  return ((uintptr_t)p & (uintptr_t)(kTdmRowBytes - 1)) == 0;
}

MORI_TDM_FN TdmSplit128 TdmPlanXfer4B(const void* src, const void* dst, int nElems) {
  if (nElems <= 0) return TdmSplit128{0, 0, 0, 0, 0};
  const uintptr_t s = (uintptr_t)src, d = (uintptr_t)dst;
  if (((s | d) & (uintptr_t)3) != 0) return TdmScalarOnly(nElems);
  // Congruence mod 128 is what the head/body/tail split needs: its body is addressed as whole
  // 128B rows on BOTH sides, so the two sides have to reach the grid after the same head. A
  // whole-run tile has no row semantics on either side, so it is still available here. The
  // kernel's source is the staging pool and its destination is an offset into the peer window;
  // that these two happen to share a phase today is a coincidence, not something to plan on.
  if (((s ^ d) & (uintptr_t)(kTdmRowBytes - 1)) != 0) return TdmPlanWhole(nElems);
  return TdmPlanRun((size_t)((s & (uintptr_t)(kTdmRowBytes - 1)) >> 2), nElems);
}

MORI_TDM_FN bool TdmXferOk(const void* src, const void* dst, int nElems, const TdmSplit128& sp) {
  if (src == nullptr || dst == nullptr) return sp.head == 0 && sp.body == 0;
  const uintptr_t s = (uintptr_t)src, d = (uintptr_t)dst;
  if (sp.body != 0) {
    if (((s | d) & (uintptr_t)3) != 0) return false;
    // Same asymmetry as in TdmPlanXfer4B: only the row-addressed body needs the two sides to
    // share a 128B phase.
    if (sp.rows > 0 && ((s ^ d) & (uintptr_t)(kTdmRowBytes - 1)) != 0) return false;
  }
  return TdmSplitOk((size_t)((s & (uintptr_t)(kTdmRowBytes - 1)) >> 2), nElems, sp);
}

}  // namespace tdm
}  // namespace mori
