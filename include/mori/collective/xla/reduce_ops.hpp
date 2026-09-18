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

// ===========================================================================
// reduce_ops.hpp
//
// Reduction-op functors and the storage -> compute type map used by the
// reduce-scatter / all-reduce kernels. The accumulator stays in the element
// type T (so bf16 reduces in bf16-width registers -- half the VGPRs of a float
// accumulator and no spilling for the 8x8 register tile). Numerical accuracy is
// handled inside the Op functor: packed-bf16 ops promote each lane to float,
// apply the op, and round back to bf16 (per-op rounding, not a float running
// sum). Packed-f16 ops stay in native f16 (V_PK_ADD/MUL/MIN/MAX_F16).
// ===========================================================================
#pragma once

#include <algorithm>
#include <cstdint>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

#define FACADE_REDUCE_USE_ALL_TYPES 0

#if defined(__HIPCC__) || defined(__HIP__)
#include "mori/core/transport/p2p/device_primitives.hpp"  // Bf16BitsToF32
#endif // __HIPCC__ || __HIP__

namespace mori {
namespace collective {

#define COLLECTIVE_DATA_TYPE_LIST(V) \
  V(F8E5M2, __hip_fp8_e5m2) \
  V(F8E4M3FN, __hip_fp8_e4m3) \
  V(F16, __half) \
  V(BF16, hip_bfloat16) \
  V(S8,  int8_t) \
  V(U8,  uint8_t) \
  V(S32, int32_t) \
  V(U32, uint32_t) \
  V(S64, int64_t) \
  V(U64, uint64_t) \
  V(F32, float) \
  V(F64, double)

// Numeric element type of the reduce collectives, expressed as a plain enum so
// the facade public API stays non-templated. The type/op -> kernel dispatch
// happens inside the (device) Run* definitions.
enum class DataType {
  #define DECLARE_ENUM(enum_name, ...) enum_name,
    COLLECTIVE_DATA_TYPE_LIST(DECLARE_ENUM)
  #undef DECLARE_ENUM
};

#define COLLECTIVE_REDUCE_OP_LIST(V) \
  V(SUM) \
  V(PRODUCT) \
  V(MIN) \
  V(MAX)

// Reduction operation for the reduce collectives.
enum class ReduceOpKind {
  #define DECLARE_ENUM(enum_name) enum_name,
    COLLECTIVE_REDUCE_OP_LIST(DECLARE_ENUM)
  #undef DECLARE_ENUM
};

namespace detail {

inline const char *DataTypeName(DataType dt) {
#define ITEM(name, ...) case DataType::name: return #name;
  switch (dt) {
    COLLECTIVE_DATA_TYPE_LIST(ITEM)
    default:
      return "?";
  }
#undef ITEM
}
  
inline const char* ReduceOpName(ReduceOpKind op) {
#define ITEM(x) case ReduceOpKind::x: return #x;
  switch (op) {
    COLLECTIVE_REDUCE_OP_LIST(ITEM)
    default:
      return "?";
  }
#undef ITEM
}

template <DataType dt>
struct ReduceTypeMap;  // primary left undefined: unmapped dt -> compile error
#define ITEM(name, ctype) \
  template <> struct ReduceTypeMap<DataType::name> { using type = ctype; };
  COLLECTIVE_DATA_TYPE_LIST(ITEM)
#undef ITEM

template <DataType dt>
using ReduceType = typename ReduceTypeMap<dt>::type;

#if defined(__HIPCC__) || defined(__HIP__)

#define COLL_HOST_DEVICE __host__ __device__
template <typename T>
struct SumOp {
  using Type = T;
  COLL_HOST_DEVICE T operator()(T a, T b) { return a + b; }
};
template <class T>
struct MaxOp {
  using Type = T;
  COLL_HOST_DEVICE T operator()(T a, T b) { return std::max(a, b); }
};
template <class T>
struct MinOp {
  using Type = T;
  COLL_HOST_DEVICE T operator()(T a, T b) { return std::min(a, b); }
};
template <class T>
struct ProdOp {
  using Type = T;
  COLL_HOST_DEVICE T operator()(T a, T b) { return a * b; }
};

// Maps a storage element type to the type used to INSTANTIATE the reduce kernels.
// float/s32 reduce as themselves; s64 is 8B so vecSize is 2; bf16/f16 reduce as a
// packed pair so vecSize stays at fp32 parity and the 8x8 accumulator tile does
// not spill. Data buffers stay physically ElemT; host code reinterpret_casts them
// to ComputeT and passes counts in packs (kPack = sizeof(ComputeT)/sizeof(ElemT)).
template <class T>
struct ReduceComputeType {
  using type = T;
};

template <class F>
COLL_HOST_DEVICE hipError_t DispatchReduceType(DataType dt, F&& func) {
  switch (dt) {
    case DataType::F32:
      return std::forward<F>(func)(float{});
#if FACADE_REDUCE_USE_ALL_TYPES
    case DataType::BF16:
      return std::forward<F>(func)(hip_bfloat16{});
    case DataType::F16:
      return std::forward<F>(func)(__half{});
    case DataType::S32:
      return std::forward<F>(func)(int32_t{});
    case DataType::S64:
      return std::forward<F>(func)(int64_t{});
#endif
    default:
      return hipErrorNotSupported;
  }
}

template <class T, class F>
COLL_HOST_DEVICE hipError_t DispatchReduceOp(ReduceOpKind op, F&& func) {
  using C = typename ReduceComputeType<T>::type;
  switch (op) {
    case ReduceOpKind::SUM:
      return std::forward<F>(func)(SumOp<C>{});
#if FACADE_REDUCE_USE_ALL_TYPES
    case ReduceOpKind::PRODUCT:
      return std::forward<F>(func)(ProdOp<C>{});
    case ReduceOpKind::MIN:
      return std::forward<F>(func)(MinOp<C>{});
    case ReduceOpKind::MAX:
      return std::forward<F>(func)(MaxOp<C>{});
#endif
    default:
      return hipErrorNotSupported;
  }
}

template <class T>
struct AccumulatorType {
  using type = T;
};

// Generic up/down cast (identity for float; specialize for fp16/bf16 if needed).
template <typename T>
__device__ __forceinline__ typename AccumulatorType<T>::type UpcastF(T v) {
  return static_cast<typename AccumulatorType<T>::type>(v);
}

template <typename T>
__device__ __forceinline__ T DowncastF(typename AccumulatorType<T>::type v) {
  return static_cast<T>(v);
}

struct alignas(4) BF16Pack {
  hip_bfloat16 x, y;
};
static_assert(sizeof(BF16Pack) == 4 && alignof(BF16Pack) == 4);

struct alignas(4) F16Pack {
  __half x, y;
};
static_assert(sizeof(F16Pack) == 4 && alignof(F16Pack) == 4);
template <>
struct ReduceComputeType<hip_bfloat16> {
  using type = BF16Pack;
};
template <>
struct ReduceComputeType<__half> {
  using type = F16Pack;
};

__device__ __forceinline__ float2 UnpackBf16Pack(BF16Pack a) {
  const uint32_t u = __builtin_bit_cast(uint32_t, a);
  return float2{mori::core::Bf16BitsToF32(static_cast<uint16_t>(u)),
                mori::core::Bf16BitsToF32(static_cast<uint16_t>(u >> 16))};
}

__device__ __forceinline__ BF16Pack PackBf16Pack(float x, float y) {
  const auto r = static_cast<__hip_bfloat162_raw>(__float22bfloat162_rn(float2{x, y}));
  const auto packed = static_cast<uint32_t>(r.x) | (static_cast<uint32_t>(r.y) << 16);
  return __builtin_bit_cast(BF16Pack, packed);
}

__device__ __forceinline__ __half2 ToHalf2(F16Pack a) { 
  return __builtin_bit_cast(__half2, a); 
}

__device__ __forceinline__ F16Pack FromHalf2(__half2 v) { 
  return __builtin_bit_cast(F16Pack, v); 
}

template <>
struct SumOp<BF16Pack> {
  using Type = BF16Pack;
  __device__ Type operator()(Type a, Type b) {
    const float2 xa = UnpackBf16Pack(a);
    const float2 xb = UnpackBf16Pack(b);
    return PackBf16Pack(xa.x + xb.x, xa.y + xb.y);
  }
};

template <>
struct ProdOp<BF16Pack> {
  using Type = BF16Pack;
  __device__ Type operator()(Type a, Type b) {
    const float2 xa = UnpackBf16Pack(a);
    const float2 xb = UnpackBf16Pack(b);
    return PackBf16Pack(xa.x * xb.x, xa.y * xb.y);
  }
};

template <>
struct MinOp<BF16Pack> {
  using Type = BF16Pack;
  __device__ Type operator()(Type a, Type b) {
    const float2 xa = UnpackBf16Pack(a);
    const float2 xb = UnpackBf16Pack(b);
    return PackBf16Pack(fminf(xa.x, xb.x), fminf(xa.y, xb.y));
  }
};

template <>
struct MaxOp<BF16Pack> {
  using Type = BF16Pack;
  __device__ Type operator()(Type a, Type b) {
    const float2 xa = UnpackBf16Pack(a);
    const float2 xb = UnpackBf16Pack(b);
    return PackBf16Pack(fmaxf(xa.x, xb.x), fmaxf(xa.y, xb.y));
  }
};

template <>
struct SumOp<F16Pack> {
  using Type = F16Pack;
  __device__ Type operator()(Type a, Type b) {
    return FromHalf2(__hadd2(ToHalf2(a), ToHalf2(b)));
  }
};

template <>
struct ProdOp<F16Pack> {
  using Type = F16Pack;
  __device__ Type operator()(Type a, Type b) {
    return FromHalf2(__hmul2(ToHalf2(a), ToHalf2(b)));
  }
};

template <>
struct MinOp<F16Pack> {
  using Type = F16Pack;
  __device__ Type operator()(Type a, Type b) {
    // HIP AMD has no __hmin2; per-lane __hmin (plan fallback).
    const __half2 ha = ToHalf2(a);
    const __half2 hb = ToHalf2(b);
    return FromHalf2(__half2{__hmin(ha.x, hb.x), __hmin(ha.y, hb.y)});
  }
};

template <>
struct MaxOp<F16Pack> {
  using Type = F16Pack;
  __device__ Type operator()(Type a, Type b) {
    // HIP AMD has no __hmax2; per-lane __hmax (plan fallback).
    const __half2 ha = ToHalf2(a);
    const __half2 hb = ToHalf2(b);
    return FromHalf2(__half2{__hmax(ha.x, hb.x), __hmax(ha.y, hb.y)});
  }
};

#endif // __HIPCC__ || __HIP__
}  // namespace detail

}  // namespace collective
}  // namespace mori
