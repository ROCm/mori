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
#else
#define MORI_TDM_FN inline
#endif

namespace mori {
namespace tdm {

constexpr int kTdmRowBytes = 128;
constexpr int kTdmRowElems4B = kTdmRowBytes / 4;

struct TdmSplit128 {
  int head;
  int body;
  int rows;
};

MORI_TDM_FN TdmSplit128 TdmPlanRun(size_t phase, int nElems) {
  constexpr int P = kTdmRowElems4B;
  if (nElems <= 0) return TdmSplit128{0, 0, 0};
  int head = (int)((P - (phase & (size_t)(P - 1))) & (size_t)(P - 1));
  if (head > nElems) head = nElems;
  const int rows = (nElems - head) / P;
  if (rows < 2) return TdmSplit128{nElems, 0, 0};
  return TdmSplit128{head, rows * P, rows};
}

MORI_TDM_FN int TdmSplitDim0(const TdmSplit128& sp) {
  return (sp.rows > 0) ? kTdmRowElems4B : 0;
}

MORI_TDM_FN int TdmSplitDim1(const TdmSplit128& sp) { return (sp.rows > 0) ? sp.rows : 0; }

MORI_TDM_FN bool TdmSplitOk(size_t phase, int nElems, const TdmSplit128& sp) {
  if (sp.head < 0 || sp.body < 0 || sp.rows < 0) return false;
  if (sp.head + sp.body > nElems) return false;
  if (sp.body == 0) return sp.rows == 0 && sp.head == (nElems > 0 ? nElems : 0);
  if (((phase + (size_t)sp.head) & (size_t)(kTdmRowElems4B - 1)) != 0) return false;
  if ((sp.body % kTdmRowElems4B) != 0) return false;
  if (sp.rows != sp.body / kTdmRowElems4B) return false;
  const int d0 = TdmSplitDim0(sp), d1 = TdmSplitDim1(sp);
  if (d0 <= 0 || d1 <= 0) return false;
  if ((d0 % kTdmRowElems4B) != 0) return false;
  if (d0 * d1 != sp.body) return false;
  if (sp.head >= kTdmRowElems4B) return false;
  if (nElems - sp.head - sp.body >= kTdmRowElems4B) return false;
  return true;
}

MORI_TDM_FN bool TdmAddrOnRow(const void* p) {
  return ((uintptr_t)p & (uintptr_t)(kTdmRowBytes - 1)) == 0;
}

MORI_TDM_FN TdmSplit128 TdmPlanXfer4B(const void* src, const void* dst, int nElems) {
  const int allScalar = (nElems > 0) ? nElems : 0;
  if (nElems <= 0) return TdmSplit128{0, 0, 0};
  const uintptr_t s = (uintptr_t)src, d = (uintptr_t)dst;
  if (((s | d) & (uintptr_t)3) != 0) return TdmSplit128{allScalar, 0, 0};
  if (((s ^ d) & (uintptr_t)(kTdmRowBytes - 1)) != 0) return TdmSplit128{allScalar, 0, 0};
  return TdmPlanRun((size_t)((s & (uintptr_t)(kTdmRowBytes - 1)) >> 2), nElems);
}

MORI_TDM_FN bool TdmXferOk(const void* src, const void* dst, int nElems, const TdmSplit128& sp) {
  if (src == nullptr || dst == nullptr) return sp.head == 0 && sp.body == 0;
  const uintptr_t s = (uintptr_t)src, d = (uintptr_t)dst;
  if (sp.body != 0) {
    if (((s | d) & (uintptr_t)3) != 0) return false;
    if (((s ^ d) & (uintptr_t)(kTdmRowBytes - 1)) != 0) return false;
  }
  return TdmSplitOk((size_t)((s & (uintptr_t)(kTdmRowBytes - 1)) >> 2), nElems, sp);
}

}  // namespace tdm
}  // namespace mori
