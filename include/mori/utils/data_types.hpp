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

#include <hip/hip_ext_ocp.h>
#include <hip/hip_fp8.h>

#include <hip/amd_detail/amd_hip_ocp_host.hpp>

namespace mori {

#if defined(HIP_FP8_TYPE_FNUZ) && HIP_FP8_TYPE_FNUZ == 1
#define MORI_FP8_TYPE_FNUZ_ENABLED
#endif

#if defined(HIP_FP8_TYPE_OCP) && HIP_FP8_TYPE_OCP == 1
#define MORI_FP8_TYPE_OCP_ENABLED
#endif

/* ---------------------------------------------------------------------------------------------- */
/*                                               FP4                                              */
/* ---------------------------------------------------------------------------------------------- */
// TODO(ditian12): due to hip header file <hip/hip_fp4.h> bug, we redefine fp4 here, remove this
// definition once ROCM resolved the bug. Bug details:
//  In file included from /opt/rocm/include/hip/hip_fp4.h:29:
//   /opt/rocm/include/hip/amd_detail/amd_hip_fp4.h:232:86: error: no matching constructor for
//   initialization of '__half2'
//     232 |   u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(u.ui32,
//     internal::half2_to_f16x2(__half2{x, 0}),
//         |                                                                                      ^
//         ~~~~~~
//   /opt/rocm/include/hip/amd_detail/amd_hip_fp16.h:301:29: note: candidate constructor not viable:
//   no known conversion from 'int' to 'const __half' for 2nd argument
//     301 |   __HOST_DEVICE__ constexpr __half2(const __half& xx, const __half& yy) : x(xx), y(yy)
//     {}

typedef uint8_t mori_fp4_storage;
typedef uint8_t mori_fp4x2_storage;
typedef uint16_t mori_fp4x4_storage;

__device__ static mori_fp4_storage bfloat16_to_fp4_e2m1(const __hip_bfloat16 x) {
  union {
    uint32_t ui32;
    mori_fp4_storage fp4[4];
  } u{0};
#if defined(HIP_ENABLE_GFX950_OCP_BUILTINS) && HIP_ENABLE_GFX950_OCP_BUILTINS == 1
  u.ui32 =
      __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(u.ui32, __hip_bfloat162{x, 0}, 1.0f /* scale */, 0);
#else
  u.ui32 = fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E2M1, true>(x, 0 /* scale */);
#endif
  return u.fp4[0];
}

__device__ static mori_fp4x2_storage bfloat162_to_fp4x2_e2m1(const __hip_bfloat162 x) {
  union {
    uint32_t ui32;
    mori_fp4x2_storage fp4x2[4];
  } u{0};
#if defined(HIP_ENABLE_GFX950_OCP_BUILTINS) && HIP_ENABLE_GFX950_OCP_BUILTINS == 1
  u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(u.ui32, x, 1.0f /* scale */, 0);
#else
  u.ui32 |= fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E2M1, true>(x.y, 0 /*scale*/);
  u.ui32 <<= 4;
  u.ui32 |= fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E2M1, true>(x.x, 0 /*scale*/);
#endif
  return u.fp4x2[0];
}

__device__ static mori_fp4_storage float_to_fp4_e2m1(const float x) {
  union {
    uint32_t ui32;
    mori_fp4_storage fp4[4];
  } u{0};
#if defined(HIP_ENABLE_GFX950_OCP_BUILTINS) && HIP_ENABLE_GFX950_OCP_BUILTINS == 1
  u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u.ui32, x, 0.0f, 1.0f /* scale */, 0);
#else
  u.ui32 = fcbx::from_float<float, fcbx::Encoding::E2M1, true>(x, 0 /*scale*/);
#endif
  return u.fp4[0];
}

__device__ static mori_fp4x2_storage float2_to_fp4x2_e2m1(const float2 x) {
  union {
    uint32_t ui32;
    mori_fp4x2_storage fp4x2[4];
  } u{0};
#if defined(HIP_ENABLE_GFX950_OCP_BUILTINS) && HIP_ENABLE_GFX950_OCP_BUILTINS == 1
  u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u.ui32, x.x, x.y, 1.0f /* scale */, 0);
#else
  u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M1, true>(x.y, 0 /*scale*/);
  u.ui32 <<= 4;
  u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M1, true>(x.x, 0 /*scale*/);
#endif
  return u.fp4x2[0];
}

struct mori_fp4_e2m1 {
  mori_fp4_storage x;

 public:
  __device__ mori_fp4_e2m1() = default;

  __device__ explicit mori_fp4_e2m1(const __hip_bfloat16 f) : x(bfloat16_to_fp4_e2m1(f)) {}

  __device__ explicit mori_fp4_e2m1(const float f) : x(float_to_fp4_e2m1(f)) {}

  __device__ operator __hip_bfloat16() const {
    union {
      __hip_bfloat16 bf162[2];
      __amd_bf16x2_storage_t bf16x2;
    } u{0};
#if defined(HIP_ENABLE_GFX950_OCP_BUILTINS) && HIP_ENABLE_GFX950_OCP_BUILTINS == 1
    u.bf16x2 = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(x, 1.0f /* scale */, 0);
#else
    using namespace fcbx;
    u.bf16x2 =
        __amd_bf16x2_storage_t{to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(x & 0xFu, 0),
                               to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(x >> 4, 0)};
#endif
    return u.bf162[0];
  }

  __device__ operator float() const {
#if defined(HIP_ENABLE_GFX950_OCP_BUILTINS) && HIP_ENABLE_GFX950_OCP_BUILTINS == 1
    auto ret = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(x, 1.0f /* scale */, 0);
#else
    using namespace fcbx;
    __amd_floatx2_storage_t ret{to_float<float, Encoding::E2M1, true>(x & 0xFu, 0),
                                to_float<float, Encoding::E2M1, true>(x >> 4, 0)};
#endif
    return ret[0];
  }
};

struct mori_fp4x2_e2m1 {
  mori_fp4x2_storage x;

 public:
  __device__ mori_fp4x2_e2m1() = default;

  __device__ explicit mori_fp4x2_e2m1(const __hip_bfloat162 f) : x(bfloat162_to_fp4x2_e2m1(f)) {}

  __device__ explicit mori_fp4x2_e2m1(const float2 f) : x(float2_to_fp4x2_e2m1(f)) {}

  __device__ operator __hip_bfloat162() const {
    union {
      __bf16 __attribute__((vector_size(4))) raw;
      __hip_bfloat162 bf162;
    } u{0};
#if defined(HIP_ENABLE_GFX950_OCP_BUILTINS) && HIP_ENABLE_GFX950_OCP_BUILTINS == 1
    u.raw = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(x, 1.0f /* scale */, 0);
#else
    using namespace fcbx;
    u.bf162 =
        __amd_bf16x2_storage_t{to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(x & 0xFu, 0),
                               to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(x >> 4, 0)};
#endif
    return u.bf162;
  }

  __device__ operator float2() const {
#if defined(HIP_ENABLE_GFX950_OCP_BUILTINS) && HIP_ENABLE_GFX950_OCP_BUILTINS == 1
    auto fp32x2 = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(x, 1.0f /* scale */, 0);
#else
    using namespace fcbx;
    auto fp32x2 = __amd_floatx2_storage_t{to_float<float, Encoding::E2M1, true>(x & 0xFu, 0),
                                          to_float<float, Encoding::E2M1, true>(x >> 4, 0)};
#endif
    return float2(fp32x2[0], fp32x2[1]);
  }
};

struct mori_fp4x4_e2m1 {
  mori_fp4x4_storage x;

 public:
  __device__ mori_fp4x4_e2m1() = default;

  __device__ explicit mori_fp4x4_e2m1(const __hip_bfloat162 low, const __hip_bfloat162 high)
      : x(bfloat162_to_fp4x2_e2m1(high) << 8 | bfloat162_to_fp4x2_e2m1(low)) {}

  __device__ explicit mori_fp4x4_e2m1(const float4 f)
      : x(float2_to_fp4x2_e2m1(float2(f.z, f.w)) << 8 | float2_to_fp4x2_e2m1(float2(f.x, f.y))) {}

  __device__ operator float4() const {
#if defined(HIP_ENABLE_GFX950_OCP_BUILTINS) && HIP_ENABLE_GFX950_OCP_BUILTINS == 1
    auto fp32x2_1 = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(x & 0xFFu, 1.0f /* scale */, 0);
    auto fp32x2_2 = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(x >> 8, 1.0f /* scale */, 0);
#else
    using namespace fcbx;
    auto fp32x2_1 =
        __amd_floatx2_storage_t{to_float<float, Encoding::E2M1, true>(x & 0xFu, 0),
                                to_float<float, Encoding::E2M1, true>((x >> 4) & 0xFu, 0)};
    auto fp32x2_2 =
        __amd_floatx2_storage_t{to_float<float, Encoding::E2M1, true>((x >> 8) & 0xFu, 0),
                                to_float<float, Encoding::E2M1, true>(x >> 12, 0)};
#endif
    return float4(fp32x2_1[0], fp32x2_1[1], fp32x2_2[0], fp32x2_2[1]);
  }
};

}  // namespace mori
