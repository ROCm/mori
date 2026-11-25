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
// The scatter combine's fp8 / e2m1 wire codec: per-group scale, pack and
// dequantize-accumulate, eight bf16 elements a lane. Shared by the intranode
// scatter combine and the AsyncLL quantized combine.

#pragma once

#include <hip/hip_runtime.h>

#include <type_traits>

namespace mori {
namespace ops {
namespace v2 {

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

}  // namespace v2
}  // namespace ops
}  // namespace mori
