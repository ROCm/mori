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

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>

#include "mori/core/utils.hpp"
#include "mori/utils/data_types.hpp"
namespace mori {
namespace core {

#if defined(MORI_FP8_TYPE_OCP_ENABLED)
using CombineInternalFp8 = __hip_fp8_e4m3;
using CombineInternalFp8x4 = __hip_fp8x4_e4m3;
#elif defined(MORI_FP8_TYPE_FNUZ_ENABLED)
using CombineInternalFp8 = __hip_fp8_e4m3_fnuz;
using CombineInternalFp8x4 = __hip_fp8x4_e4m3_fnuz;
#else
using CombineInternalFp8 = uint8_t;
#endif

/* ---------------------------------------------------------------------------------------------- */
/*                                        Type Definitions                                        */
/* ---------------------------------------------------------------------------------------------- */
template <int VecBytes>
struct VecTypeSelector {
  using type = void;
};

template <>
struct VecTypeSelector<1> {
  using dataType = uint8_t;
};

template <>
struct VecTypeSelector<2> {
  using dataType = uint16_t;
};

template <>
struct VecTypeSelector<4> {
  using dataType = uint32_t;
};

template <>
struct VecTypeSelector<8> {
  using dataType = uint64_t;
};

template <>
struct VecTypeSelector<16> {
  using dataType = ulong2;
};

template <typename T, int VecSize>
struct VecTypeAdaptor {
  using type = void;
};

template <>
struct VecTypeAdaptor<float, 1> {
  using dataType = float;
};

template <>
struct VecTypeAdaptor<float, 2> {
  using dataType = float2;
};

template <>
struct VecTypeAdaptor<float, 4> {
  using dataType = float4;
};

#ifdef MORI_FP8_TYPE_OCP_ENABLED
template <>
struct VecTypeAdaptor<mori_fp4_e2m1, 2> {
  using dataType = mori_fp4x2_e2m1;
};

template <>
struct VecTypeAdaptor<mori_fp4_e2m1, 4> {
  using dataType = mori_fp4x4_e2m1;
};
#endif

/* ---------------------------------------------------------------------------------------------- */
/*                                           Load/Store                                           */
/* ---------------------------------------------------------------------------------------------- */
#define USE_BUILDIN_LD 1
#define USE_BUILDIN_ST 1

#if USE_BUILDIN_LD
template <int VecBytes>
__device__ __forceinline__ typename VecTypeSelector<VecBytes>::dataType load(const void* addr);

template <>
__device__ __forceinline__ typename VecTypeSelector<1>::dataType load<1>(const void* addr) {
  return __builtin_nontemporal_load((uint8_t*)addr);
}

template <>
__device__ __forceinline__ typename VecTypeSelector<2>::dataType load<2>(const void* addr) {
  return __builtin_nontemporal_load((uint16_t*)addr);
}

template <>
__device__ __forceinline__ typename VecTypeSelector<4>::dataType load<4>(const void* addr) {
  return __builtin_nontemporal_load((uint32_t*)addr);
}

template <>
__device__ __forceinline__ typename VecTypeSelector<8>::dataType load<8>(const void* addr) {
  return __builtin_nontemporal_load((uint64_t*)addr);
}

template <>
__device__ __forceinline__ typename VecTypeSelector<16>::dataType load<16>(const void* addr) {
  ulong2 result;
  result.x = __builtin_nontemporal_load((uint64_t*)addr);
  result.y = __builtin_nontemporal_load(((uint64_t*)addr) + 1);
  return result;
}
#else
template <int VecBytes>
__device__ __forceinline__ typename VecTypeSelector<VecBytes>::dataType load(const void* addr);

template <>
__device__ __forceinline__ typename VecTypeSelector<1>::dataType load<1>(const void* addr) {
  return *static_cast<const uint8_t*>(addr);
}

template <>
__device__ __forceinline__ typename VecTypeSelector<2>::dataType load<2>(const void* addr) {
  return *static_cast<const uint16_t*>(addr);
}

template <>
__device__ __forceinline__ typename VecTypeSelector<4>::dataType load<4>(const void* addr) {
  return *static_cast<const uint32_t*>(addr);
}

template <>
__device__ __forceinline__ typename VecTypeSelector<8>::dataType load<8>(const void* addr) {
  return *static_cast<const uint64_t*>(addr);
}

template <>
__device__ __forceinline__ typename VecTypeSelector<16>::dataType load<16>(const void* addr) {
  const uint64_t* ptr = static_cast<const uint64_t*>(addr);
  ulong2 result;
  result.x = ptr[0];
  result.y = ptr[1];
  return result;
}
#endif

#if USE_BUILDIN_ST
template <int VecBytes>
__device__ __forceinline__ void store(void* addr,
                                      typename VecTypeSelector<VecBytes>::dataType value);

template <>
__device__ __forceinline__ void store<1>(void* addr, typename VecTypeSelector<1>::dataType value) {
  __builtin_nontemporal_store(value, (uint8_t*)addr);
}

template <>
__device__ __forceinline__ void store<2>(void* addr, typename VecTypeSelector<2>::dataType value) {
  __builtin_nontemporal_store(value, (uint16_t*)addr);
}

template <>
__device__ __forceinline__ void store<4>(void* addr, typename VecTypeSelector<4>::dataType value) {
  __builtin_nontemporal_store(value, (uint32_t*)addr);
}

template <>
__device__ __forceinline__ void store<8>(void* addr, typename VecTypeSelector<8>::dataType value) {
  __builtin_nontemporal_store(value, (uint64_t*)addr);
}

template <>
__device__ __forceinline__ void store<16>(void* addr,
                                          typename VecTypeSelector<16>::dataType value) {
  __builtin_nontemporal_store(value.x, (uint64_t*)addr);
  __builtin_nontemporal_store(value.y, ((uint64_t*)addr) + 1);
}
#else
template <int VecBytes>
__device__ __forceinline__ void store(void* addr,
                                      typename VecTypeSelector<VecBytes>::dataType value);

template <>
__device__ __forceinline__ void store<1>(void* addr, typename VecTypeSelector<1>::dataType value) {
  *((uint8_t*)addr) = value;
}

template <>
__device__ __forceinline__ void store<2>(void* addr, typename VecTypeSelector<2>::dataType value) {
  *((uint16_t*)addr) = value;
}

template <>
__device__ __forceinline__ void store<4>(void* addr, typename VecTypeSelector<4>::dataType value) {
  *((uint32_t*)addr) = value;
}

template <>
__device__ __forceinline__ void store<8>(void* addr, typename VecTypeSelector<8>::dataType value) {
  *((uint64_t*)addr) = value;
}

template <>
__device__ __forceinline__ void store<16>(void* addr,
                                          typename VecTypeSelector<16>::dataType value) {
  *((uint64_t*)addr) = value.x;
  *(((uint64_t*)addr) + 1) = value.y;
}
#endif

/* ---------------------------------------------------------------------------------------------- */
/*                                              Copy                                              */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ void ThreadCopy(T* dst, T* src, size_t nelems) {
  constexpr int vecSize = 16 / sizeof(T);
  int offset = 0;

  while ((offset + vecSize) <= nelems) {
    reinterpret_cast<uint4*>(dst + offset)[0] = reinterpret_cast<uint4*>(src + offset)[0];
    offset += vecSize;
  }

  while (offset < nelems) {
    dst[offset] = src[offset];
    offset += 1;
  }
}

template <typename T, int Unroll>
inline __device__ void WarpCopyImpl(T* __restrict__ dst, const T* __restrict__ src, size_t& offset,
                                    size_t nelems) {
  constexpr int VecBytes = 16;
  constexpr int vecSize = VecBytes / sizeof(T);
  int laneId = threadIdx.x & (warpSize - 1);
  using DataType = typename VecTypeSelector<VecBytes>::dataType;

  const int elemsPerWarp = Unroll * warpSize * vecSize;
  const size_t numIters = (nelems - offset) / elemsPerWarp;
  for (size_t iter = 0; iter < numIters; iter++) {
    DataType vec[Unroll];
#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      vec[u] = load<VecBytes>(src + offset + (laneId + u * warpSize) * vecSize);
    }

#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      store<VecBytes>(dst + offset + (laneId + u * warpSize) * vecSize, vec[u]);
    }

    offset += elemsPerWarp;
  }
}

template <typename T, int Unroll = 1>
inline __device__ void WarpCopy(T* __restrict__ dst, const T* __restrict__ src, size_t nelems) {
  int laneId = threadIdx.x & (warpSize - 1);

  size_t offset = 0;
  WarpCopyImpl<T, Unroll>(dst, src, offset, nelems);
  if constexpr (Unroll > 1) {
    WarpCopyImpl<T, 1>(dst, src, offset, nelems);
  }

  offset += laneId;
  while (offset < nelems) {
    dst[offset] = src[offset];
    offset += warpSize;
  }
}

template <typename T, int N>
inline __device__ void WarpCopy(T* dst, T* src) {
  constexpr int vecSize = 16 / sizeof(T);
  int laneId = threadIdx.x & (warpSize - 1);

  for (int i = laneId * vecSize; (i + vecSize) <= N; i += warpSize * vecSize) {
    reinterpret_cast<uint4*>(dst + i)[0] = reinterpret_cast<uint4*>(src + i)[0];
  }

  if constexpr ((N % vecSize) != 0) {
    int offset = N / vecSize * vecSize;
    for (int i = offset + laneId; i < N; i += warpSize) dst[i] = src[i];
  }
}

/* ---------------------------------------------------------------------------------------------- */
/*                                             Reduce                                             */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ T WarpReduceSum(T val) {
  int laneId = threadIdx.x & (warpSize - 1);
  for (int delta = (warpSize >> 1); delta > 0; delta = (delta >> 1)) {
    val += __shfl_down(val, delta);
  }
  return val;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                             Prefix                                             */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ T WarpPrefixSum(T val, size_t laneNum) {
  assert(laneNum <= warpSize);
  int laneId = WarpLaneId();
  uint32_t prefixSum = 0;
  if (laneId < laneNum) {
    for (int i = 0; i <= laneId; i++) {
      uint32_t targetLaneVal = __shfl(val, i);
      if (laneId > i) prefixSum += targetLaneVal;
    }
  }
  return prefixSum;
}

// TODO: fix bugs
template <typename T>
inline __device__ T BlockPrefixSum(T val, size_t thdNum) {
  int blockSize = FlatBlockSize();
  assert(thdNum <= blockSize);

  int warpId = FlatBlockWarpId();

  int firstThd = warpId * DeviceWarpSize();
  int lastThd = std::min(firstThd + DeviceWarpSize(), blockSize);
  int thisWarpSize = lastThd - firstThd;

  T prefixSum = WarpPrefixSum(val, thisWarpSize);

  __shared__ T warpPrefixSum[32];  // max warp num is 32

  if (WarpLaneId() == (DeviceWarpSize() - 1)) warpPrefixSum[warpId] = prefixSum + val;
  __syncthreads();

  for (int i = 0; i < warpId; i++) {
    prefixSum += warpPrefixSum[i];
  }

  return prefixSum;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                        WarpAccumulation                                        */
/* ---------------------------------------------------------------------------------------------- */
template <typename T>
inline __device__ void WarpAccum(T* accum, T* src, size_t nelems) {
  constexpr int vecSize = 16 / sizeof(T);
  int laneId = threadIdx.x & (warpSize - 1);
  int offset = laneId * vecSize;

  while ((offset + vecSize) <= nelems) {
    uint4 srcVal = reinterpret_cast<uint4*>(src + offset)[0];
    uint4 accumVal = reinterpret_cast<uint4*>(accum + offset)[0];
    for (int i = 0; i < vecSize; i++) {
      reinterpret_cast<T*>(&accumVal)[i] += reinterpret_cast<T*>(&srcVal)[i];
    }
    reinterpret_cast<uint4*>(accum + offset)[0] = accumVal;
    offset += warpSize * vecSize;
  }

  while (offset < nelems) {
    accum[offset] += src[offset];
    offset += 1;
  }
}

template <typename T, int VecBytes>
__forceinline__ __device__ void WarpAccumDynamic(T* __restrict__ dest, T* const* __restrict__ srcs,
                                                 const float* __restrict__ srcScales,
                                                 size_t accumNum, size_t nelems) {
  static_assert((VecBytes <= 16) && (VecBytes >= 4) && IsPowerOf2(VecBytes));

  constexpr int vecSize = VecBytes / sizeof(T);
  const int laneId = threadIdx.x & (warpSize - 1);
  size_t offset = 0;

  using DataType = typename VecTypeSelector<VecBytes>::dataType;
  const int elemsPerWarp = warpSize * vecSize;
  const size_t numIters = (nelems - offset) / elemsPerWarp;
  const size_t laneOffset = laneId * vecSize;

  using AccumFp32Type = std::conditional_t<std::is_same_v<T, mori_fp4x2_e2m1>, float2, float>;

  for (size_t iter = 0; iter < numIters; ++iter) {
    AccumFp32Type accumValFp32[vecSize] = {AccumFp32Type{0}};
#pragma unroll
    for (int i = 0; i < accumNum; ++i) {
      if (srcs[i] == nullptr) continue;
      DataType srcVal = load<VecBytes>(srcs[i] + offset + laneOffset);
      float srcScale = (srcScales == nullptr) ? 1.0f : srcScales[i];
#pragma unroll
      for (int j = 0; j < vecSize; ++j) {
        accumValFp32[j] += AccumFp32Type(reinterpret_cast<const T*>(&srcVal)[j]) * srcScale;
      }
    }

    union {
      DataType accumVec;
      T accumVal[vecSize];
    };
#pragma unroll
    for (int j = 0; j < vecSize; ++j) {
      accumVal[j] = T(accumValFp32[j]);
    }
    store<VecBytes>(dest + offset + laneOffset, accumVec);

    offset += elemsPerWarp;
  }

  // remaining size
  offset += laneId;
  while (offset < nelems) {
    AccumFp32Type accumValFp32 = AccumFp32Type{0};
    for (int i = 0; i < accumNum; ++i) {
      const T* srcPtr = srcs[i];
      if (srcPtr == nullptr) continue;

      float srcScale = (srcScales == nullptr) ? 1.0f : srcScales[i];
      accumValFp32 += AccumFp32Type(srcPtr[offset]) * srcScale;
    }
    dest[offset] = T(accumValFp32);
    offset += warpSize;
  }
}

template <typename T, int VecBytes, int AccumNum, int Unroll>
__forceinline__ __device__ void WarpAccumImpl(T* __restrict__ dest, T* const* __restrict__ srcs,
                                              const float* __restrict__ srcScales, size_t& offset,
                                              size_t nelems) {
  constexpr int vecSize = VecBytes / sizeof(T);
  using DataType = typename VecTypeSelector<VecBytes>::dataType;

  const int elemsPerWarp = Unroll * warpSize * vecSize;
  const size_t numIters = (nelems - offset) / elemsPerWarp;
#if 0
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("numIters=%zu nelems=%zu offset=%zu elemsPerWarp=%d\n", numIters, nelems, offset,
           elemsPerWarp);
  }
#endif
  const int laneId = threadIdx.x & (warpSize - 1);
  const size_t laneOffset = laneId * vecSize;

  using AccumFp32Type = std::conditional_t<std::is_same_v<T, mori_fp4x2_e2m1>, float2, float>;

  for (size_t iter = 0; iter < numIters; iter++) {
    AccumFp32Type accumValFp32[Unroll][vecSize] = {0};

#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      const T* srcPtr = srcs[i];
      if (srcPtr == nullptr) continue;

#pragma unroll Unroll
      for (int u = 0; u < Unroll; u++) {
        DataType srcVals = load<VecBytes>(srcPtr + offset + laneOffset + u * warpSize * vecSize);
        float srcScale = (srcScales == nullptr) ? 1.0f : srcScales[i];
#pragma unroll vecSize
        for (int j = 0; j < vecSize; ++j) {
          accumValFp32[u][j] += AccumFp32Type(reinterpret_cast<const T*>(&srcVals)[j]) * srcScale;
        }
      }
    }

    union {
      DataType accumVec[Unroll];
      T accumVal[Unroll][vecSize];
    };
#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
#pragma unroll vecSize
      for (int j = 0; j < vecSize; ++j) {
        accumVal[u][j] = T(accumValFp32[u][j]);
      }
      store<VecBytes>(dest + offset + laneOffset + u * warpSize * vecSize, accumVec[u]);
    }

    offset += elemsPerWarp;
  }
}

template <typename T, int VecBytes, int AccumNum>
__forceinline__ __device__ void WarpAccumImpl(T* __restrict__ dest, T* const* __restrict__ srcs,
                                              const float* __restrict__ srcScales, size_t& offset,
                                              size_t nelems) {
  constexpr int vecSize = VecBytes / sizeof(T);
  using DataType = typename VecTypeSelector<VecBytes>::dataType;

  const int elemsPerWarp = warpSize * vecSize;
  const size_t numIters = (nelems - offset) / elemsPerWarp;

  const int laneId = threadIdx.x & (warpSize - 1);
  const size_t laneOffset = laneId * vecSize;

  float scales[AccumNum];
  const T* cached_srcs[AccumNum];
#pragma unroll AccumNum
  for (int i = 0; i < AccumNum; ++i) {
    scales[i] = (srcScales == nullptr) ? 1.0f : srcScales[i];
    cached_srcs[i] = srcs[i];
  }

  using AccumFp32Type = std::conditional_t<std::is_same_v<T, mori_fp4x2_e2m1>, float2, float>;

  for (size_t iter = 0; iter < numIters; ++iter) {
    AccumFp32Type accumValFp32[vecSize] = {AccumFp32Type{0}};

    DataType srcVals[AccumNum];
#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      if (cached_srcs[i] != nullptr)
        srcVals[i] = load<VecBytes>(cached_srcs[i] + offset + laneOffset);
    }

#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      if (cached_srcs[i] != nullptr) {
#pragma unroll vecSize
        for (int j = 0; j < vecSize; ++j) {
          accumValFp32[j] += AccumFp32Type(reinterpret_cast<const T*>(srcVals + i)[j]) * scales[i];
        }
      }
    }

    union {
      DataType accumVec;
      T accumVal[vecSize];
    };
#pragma unroll vecSize
    for (int j = 0; j < vecSize; ++j) {
      accumVal[j] = T(accumValFp32[j]);
    }
    store<VecBytes>(dest + offset + laneOffset, accumVec);

    offset += elemsPerWarp;
  }
}

#if 0
template <typename T, int VecBytes, int AccumNum>
__forceinline__ __device__ void WarpAccumPipelineImpl(T* __restrict__ dest,
                                                      T* const* __restrict__ srcs,
                                                      const float* __restrict__ srcScales,
                                                      size_t& offset, size_t nelems) {
  constexpr int vecSize = VecBytes / sizeof(T);
  using DataType = typename VecTypeSelector<VecBytes>::dataType;

  const int elemsPerWarp = warpSize * vecSize;
  const size_t numIters = (nelems - offset) / elemsPerWarp;

  const int laneId = threadIdx.x & (warpSize - 1);
  const size_t laneOffset = laneId * vecSize;

  float scales[AccumNum];
#pragma unroll AccumNum
  for (int i = 0; i < AccumNum; ++i) {
    scales[i] = (srcScales == nullptr) ? 1.0f : srcScales[i];
  }

  for (size_t iter = 0; iter < numIters; ++iter) {
    float accumValFp32[vecSize];
    DataType srcVals[AccumNum];

    if (srcs[0] != nullptr) srcVals[0] = load<VecBytes>(srcs[0] + offset + laneOffset);
    for (int j = 0; j < vecSize; ++j) {
      accumValFp32[j] = float(reinterpret_cast<const T*>(srcVals)[j]);
    }

    DataType tmp1, tmp2;
    if (srcs[1] != nullptr) tmp1 = load<VecBytes>(srcs[1] + offset + laneOffset);
    bool tail = true;

    // #pragma unroll AccumNum
    for (int i = 2; i < AccumNum; i += 2) {
      if (srcs[i] != nullptr) tmp2 = load<VecBytes>(srcs[i] + offset + laneOffset);

      if (srcs[i - 1] != nullptr) {
        // #pragma unroll vecSize
        for (int j = 0; j < vecSize; ++j) {
          accumValFp32[j] += float(reinterpret_cast<const T*>(tmp1)[j]) * scales[i - 1];
        }
      }

      if (i + 1 < AccumNum) {
        if (srcs[i + 1] != nullptr) tmp1 = load<VecBytes>(srcs[i + 1] + offset + laneOffset);
      } else {
        tail = false;
      }

      if (srcs[i] != nullptr) {
        // #pragma unroll vecSize
        for (int j = 0; j < vecSize; ++j) {
          accumValFp32[j] += float(reinterpret_cast<const T*>(tmp2)[j]) * scales[i];
        }
      }
    }

    if (tail) {
      if (srcs[AccumNum - 1] != nullptr) {
        // #pragma unroll vecSize
        for (int j = 0; j < vecSize; ++j) {
          accumValFp32[j] += float(reinterpret_cast<const T*>(tmp1)[j]) * scales[AccumNum - 1];
        }
      }
    }

    union {
      DataType accumVec;
      T accumVal[vecSize];
    };
#pragma unroll vecSize
    for (int j = 0; j < vecSize; ++j) {
      accumVal[j] = T(accumValFp32[j]);
    }
    store<VecBytes>(dest + offset + laneOffset, accumVec);
    offset += elemsPerWarp;
  }
}
#endif

template <typename T, int VecBytes, int AccumNum, int Unroll>
__forceinline__ __device__ void WarpAccum(T* __restrict__ dest, T* const* __restrict__ srcs,
                                          const float* __restrict__ srcScales, size_t nelems) {
  static_assert((VecBytes <= 16) && (VecBytes >= 4) && IsPowerOf2(VecBytes));

  constexpr int vecSize = VecBytes / sizeof(T);
  const int laneId = threadIdx.x & (warpSize - 1);
  size_t offset = 0;

  // WarpAccumImpl<T, VecBytes, AccumNum, Unroll>(dest, srcs, srcScales, offset, nelems);
  // WarpAccumImpl<T, VecBytes, AccumNum, 1>(dest, srcs, srcScales, offset, nelems);

  WarpAccumImpl<T, VecBytes, AccumNum>(dest, srcs, srcScales, offset, nelems);

  // remaining size

  using AccumFp32Type = std::conditional_t<std::is_same_v<T, mori_fp4x2_e2m1>, float2, float>;

  offset += laneId;
  while (offset < nelems) {
    AccumFp32Type accumValFp32 = AccumFp32Type{0};
#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      const T* srcPtr = srcs[i];
      if (srcPtr == nullptr) continue;

      float srcScale = (srcScales == nullptr) ? 1.0f : srcScales[i];
      accumValFp32 += AccumFp32Type(srcPtr[offset]) * srcScale;
    }
    dest[offset] = T(accumValFp32);
    offset += warpSize;
  }
}

#ifndef WARP_ACCUM_UNROLL
#define WARP_ACCUM_UNROLL 2
#endif

template <typename T, int VecBytes>
__forceinline__ __device__ void WarpAccum(T* __restrict__ dest, T* const* __restrict__ srcs,
                                          const float* __restrict__ srcScales, size_t accumNum,
                                          size_t nelems) {
#define WARP_ACCUM_CASE(AccumNum)                                                       \
  case AccumNum:                                                                        \
    WarpAccum<T, VecBytes, AccumNum, WARP_ACCUM_UNROLL>(dest, srcs, srcScales, nelems); \
    break;

  switch (accumNum) {
    WARP_ACCUM_CASE(1)
    WARP_ACCUM_CASE(2)
    WARP_ACCUM_CASE(4)
    WARP_ACCUM_CASE(6)
    WARP_ACCUM_CASE(8)
    WARP_ACCUM_CASE(10)
    default:
      WarpAccumDynamic<T, VecBytes>(dest, srcs, srcScales, accumNum, nelems);
      break;
  }

#undef WARP_ACCUM_CASE
}

template <typename T>
__forceinline__ __device__ void WarpCastBf16ToCombineInternalFp8(
    CombineInternalFp8* __restrict__ dst, const T* __restrict__ src, int hiddenDim, int laneId) {
#if defined(MORI_FP8_TYPE_OCP_ENABLED) || defined(MORI_FP8_TYPE_FNUZ_ENABLED)

  if constexpr (std::is_same_v<T, hip_bfloat16>) {
    using Fp8T = CombineInternalFp8;
    using Fp8x4T = CombineInternalFp8x4;
    constexpr int kVec8 = 8;
    constexpr int kVec4 = 4;

    const uintptr_t srcAddr = reinterpret_cast<uintptr_t>(src);
    const uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dst);
    const bool canVec8 = ((srcAddr & 0x7) == 0) && ((dstAddr & 0x7) == 0);
    const bool canVec4 = ((srcAddr & 0x3) == 0) && ((dstAddr & 0x3) == 0);

    const int vecEnd8 = (hiddenDim / kVec8) * kVec8;
    const int vecEnd4 = (hiddenDim / kVec4) * kVec4;

    if (canVec8) {
      const auto* __restrict__ srcAligned =
          static_cast<const hip_bfloat16*>(__builtin_assume_aligned(src, 8));
      auto* __restrict__ dstAligned =
          static_cast<CombineInternalFp8*>(__builtin_assume_aligned(dst, 8));

#pragma unroll 4
      for (int j = laneId * kVec8; j < vecEnd8; j += warpSize * kVec8) {
        union {
          ulong2 u64x2;
          uint32_t u32[4];
        } in;
        in.u64x2 = load<16>(srcAligned + j);

        const __hip_bfloat162_raw bf01{static_cast<unsigned short>(in.u32[0]),
                                       static_cast<unsigned short>(in.u32[0] >> 16)};
        const __hip_bfloat162_raw bf23{static_cast<unsigned short>(in.u32[1]),
                                       static_cast<unsigned short>(in.u32[1] >> 16)};
        const __hip_bfloat162_raw bf45{static_cast<unsigned short>(in.u32[2]),
                                       static_cast<unsigned short>(in.u32[2] >> 16)};
        const __hip_bfloat162_raw bf67{static_cast<unsigned short>(in.u32[3]),
                                       static_cast<unsigned short>(in.u32[3] >> 16)};

        const __hip_fp8x2_storage_t fp01 = __hip_cvt_bfloat16raw2_to_fp8x2(
            bf01, Fp8x4T::__default_saturation, Fp8x4T::__default_interpret);
        const __hip_fp8x2_storage_t fp23 = __hip_cvt_bfloat16raw2_to_fp8x2(
            bf23, Fp8x4T::__default_saturation, Fp8x4T::__default_interpret);
        const __hip_fp8x2_storage_t fp45 = __hip_cvt_bfloat16raw2_to_fp8x2(
            bf45, Fp8x4T::__default_saturation, Fp8x4T::__default_interpret);
        const __hip_fp8x2_storage_t fp67 = __hip_cvt_bfloat16raw2_to_fp8x2(
            bf67, Fp8x4T::__default_saturation, Fp8x4T::__default_interpret);

        const uint32_t packed0 = static_cast<uint32_t>(fp01) | (static_cast<uint32_t>(fp23) << 16);
        const uint32_t packed1 = static_cast<uint32_t>(fp45) | (static_cast<uint32_t>(fp67) << 16);
        const uint64_t packed01 =
            static_cast<uint64_t>(packed0) | (static_cast<uint64_t>(packed1) << 32);

        store<8>(dstAligned + j, packed01);
      }

#pragma unroll 2
      for (int j = vecEnd8 + laneId * kVec4; j < vecEnd4; j += warpSize * kVec4) {
        const __hip_bfloat162 low = *reinterpret_cast<const __hip_bfloat162*>(srcAligned + j);
        const __hip_bfloat162 high = *reinterpret_cast<const __hip_bfloat162*>(srcAligned + j + 2);
        const Fp8x4T packed(high, low);
        *reinterpret_cast<__hip_fp8x4_storage_t*>(dstAligned + j) = packed.__x;
      }
    } else if (canVec4) {
#pragma unroll 2
      for (int j = laneId * kVec4; j < vecEnd4; j += warpSize * kVec4) {
        const __hip_bfloat162 low = *reinterpret_cast<const __hip_bfloat162*>(src + j);
        const __hip_bfloat162 high = *reinterpret_cast<const __hip_bfloat162*>(src + j + 2);
        const Fp8x4T packed(high, low);
        *reinterpret_cast<__hip_fp8x4_storage_t*>(dst + j) = packed.__x;
      }
    }

    if (canVec8 || canVec4) {
      for (int j = vecEnd4 + laneId; j < hiddenDim; j += warpSize) {
        dst[j] = Fp8T(src[j]);
      }
    } else {
      for (int j = laneId; j < hiddenDim; j += warpSize) {
        dst[j] = Fp8T(src[j]);
      }
    }
  }
  // Note: when T != hip_bfloat16, this function is a no-op.
  // Callers should guard with if constexpr or ensure T is hip_bfloat16.
#else
  static_assert(!sizeof(T*), "WarpCastBf16ToCombineInternalFp8 requires FP8 type support "
                              "(MORI_FP8_TYPE_OCP_ENABLED or MORI_FP8_TYPE_FNUZ_ENABLED)");
#endif
}

#if defined(MORI_FP8_TYPE_OCP_ENABLED) || defined(MORI_FP8_TYPE_FNUZ_ENABLED)
namespace detail {
using CombineInternalFp8T = CombineInternalFp8;
using CombineInternalFp8x4T = CombineInternalFp8x4;

template <int AccumNum>
__forceinline__ __device__ void WarpAccumCombineInternalFp8ToBf16Fixed(
    hip_bfloat16* __restrict__ out, const CombineInternalFp8T* const* __restrict__ srcPtrs,
    int laneId, int hiddenDimSize) {
  static_assert(AccumNum > 0, "AccumNum must be positive");

  using Fp8T = CombineInternalFp8T;
  using Fp8x4T = CombineInternalFp8x4T;
  constexpr int kVec8 = 8;
  constexpr int kVec4 = 4;

  const uintptr_t outAddr = reinterpret_cast<uintptr_t>(out);
  bool canVec8 = ((outAddr & 0x7) == 0);
  bool canVec4 = true;
#pragma unroll
  for (int n = 0; n < AccumNum; n++) {
    const Fp8T* src = srcPtrs[n];
    if (src == nullptr) continue;
    const uintptr_t srcAddr = reinterpret_cast<uintptr_t>(src);
    canVec8 &= ((srcAddr & 0x7) == 0);
    canVec4 &= ((srcAddr & 0x3) == 0);
  }

  const int vecEnd8 = (hiddenDimSize / kVec8) * kVec8;
  const int vecEnd4 = (hiddenDimSize / kVec4) * kVec4;

  if (canVec8) {
    auto* __restrict__ outAligned = static_cast<hip_bfloat16*>(__builtin_assume_aligned(out, 8));

#pragma unroll 4
    for (int j = laneId * kVec8; j < vecEnd8; j += warpSize * kVec8) {
      float4 sumLo = {0.0f, 0.0f, 0.0f, 0.0f};
      float4 sumHi = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int n = 0; n < AccumNum; n++) {
        const Fp8T* src = srcPtrs[n];
        if (src == nullptr) continue;
        const auto* srcAligned = static_cast<const Fp8T*>(__builtin_assume_aligned(src, 8));
        const uint64_t a = load<8>(srcAligned + j);
        Fp8x4T v0;
        v0.__x = static_cast<__hip_fp8x4_storage_t>(a);
        const float4 f0 = static_cast<float4>(v0);
        Fp8x4T v1;
        v1.__x = static_cast<__hip_fp8x4_storage_t>(a >> 32);
        const float4 f1 = static_cast<float4>(v1);
        sumLo.x += f0.x;
        sumLo.y += f0.y;
        sumLo.z += f0.z;
        sumLo.w += f0.w;
        sumHi.x += f1.x;
        sumHi.y += f1.y;
        sumHi.z += f1.z;
        sumHi.w += f1.w;
      }

      const __hip_bfloat162 bf01 = __float22bfloat162_rn(float2{sumLo.x, sumLo.y});
      const __hip_bfloat162 bf23 = __float22bfloat162_rn(float2{sumLo.z, sumLo.w});
      const __hip_bfloat162 bf45 = __float22bfloat162_rn(float2{sumHi.x, sumHi.y});
      const __hip_bfloat162 bf67 = __float22bfloat162_rn(float2{sumHi.z, sumHi.w});

      const __hip_bfloat162_raw bf01r = static_cast<__hip_bfloat162_raw>(bf01);
      const __hip_bfloat162_raw bf23r = static_cast<__hip_bfloat162_raw>(bf23);
      const __hip_bfloat162_raw bf45r = static_cast<__hip_bfloat162_raw>(bf45);
      const __hip_bfloat162_raw bf67r = static_cast<__hip_bfloat162_raw>(bf67);

      const uint32_t u01 = static_cast<uint32_t>(bf01r.x) | (static_cast<uint32_t>(bf01r.y) << 16);
      const uint32_t u23 = static_cast<uint32_t>(bf23r.x) | (static_cast<uint32_t>(bf23r.y) << 16);
      const uint32_t u45 = static_cast<uint32_t>(bf45r.x) | (static_cast<uint32_t>(bf45r.y) << 16);
      const uint32_t u67 = static_cast<uint32_t>(bf67r.x) | (static_cast<uint32_t>(bf67r.y) << 16);

      const ulong2 packedOut{(static_cast<uint64_t>(u01) | (static_cast<uint64_t>(u23) << 32)),
                             (static_cast<uint64_t>(u45) | (static_cast<uint64_t>(u67) << 32))};
      store<16>(outAligned + j, packedOut);
    }

    if (vecEnd8 < vecEnd4) {
#pragma unroll 2
      for (int j = vecEnd8 + laneId * kVec4; j < vecEnd4; j += warpSize * kVec4) {
        float4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int n = 0; n < AccumNum; n++) {
          const Fp8T* src = srcPtrs[n];
          if (src == nullptr) continue;
          Fp8x4T v;
          v.__x = *reinterpret_cast<const __hip_fp8x4_storage_t*>(src + j);
          const float4 f = static_cast<float4>(v);
          sum4.x += f.x;
          sum4.y += f.y;
          sum4.z += f.z;
          sum4.w += f.w;
        }
        out[j + 0] = hip_bfloat16(sum4.x);
        out[j + 1] = hip_bfloat16(sum4.y);
        out[j + 2] = hip_bfloat16(sum4.z);
        out[j + 3] = hip_bfloat16(sum4.w);
      }
    }
  } else if (canVec4) {
#pragma unroll 2
    for (int j = laneId * kVec4; j < vecEnd4; j += warpSize * kVec4) {
      float4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int n = 0; n < AccumNum; n++) {
        const Fp8T* src = srcPtrs[n];
        if (src == nullptr) continue;
        Fp8x4T v;
        v.__x = *reinterpret_cast<const __hip_fp8x4_storage_t*>(src + j);
        const float4 f = static_cast<float4>(v);
        sum4.x += f.x;
        sum4.y += f.y;
        sum4.z += f.z;
        sum4.w += f.w;
      }
      out[j + 0] = hip_bfloat16(sum4.x);
      out[j + 1] = hip_bfloat16(sum4.y);
      out[j + 2] = hip_bfloat16(sum4.z);
      out[j + 3] = hip_bfloat16(sum4.w);
    }
  }

  const int scalarStart = (canVec8 || canVec4) ? vecEnd4 : 0;
  for (int j = scalarStart + laneId; j < hiddenDimSize; j += warpSize) {
    float sum = 0.0f;
#pragma unroll
    for (int n = 0; n < AccumNum; n++) {
      const Fp8T* src = srcPtrs[n];
      if (src == nullptr) continue;
      sum += float(src[j]);
    }
    out[j] = hip_bfloat16(sum);
  }
}

__forceinline__ __device__ void WarpAccumCombineInternalFp8ToBf16Dynamic(
    hip_bfloat16* __restrict__ out, const CombineInternalFp8T* const* __restrict__ srcPtrs,
    int accumNum, int laneId, int hiddenDimSize) {
  using Fp8T = CombineInternalFp8T;
  using Fp8x4T = CombineInternalFp8x4T;

  constexpr int kVec4 = 4;
  const int vecEnd = (hiddenDimSize / kVec4) * kVec4;

  bool canVec4 = true;
#pragma unroll 4
  for (int n = 0; n < accumNum; n++) {
    const Fp8T* src = srcPtrs[n];
    if (src == nullptr) continue;
    canVec4 &= ((reinterpret_cast<uintptr_t>(src) & 0x3) == 0);
  }

  if (canVec4) {
    for (int j = laneId * kVec4; j < vecEnd; j += warpSize * kVec4) {
      float4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
      for (int n = 0; n < accumNum; n++) {
        const Fp8T* src = srcPtrs[n];
        if (src == nullptr) continue;
        Fp8x4T v;
        v.__x = *reinterpret_cast<const __hip_fp8x4_storage_t*>(src + j);
        const float4 f = static_cast<float4>(v);
        sum4.x += f.x;
        sum4.y += f.y;
        sum4.z += f.z;
        sum4.w += f.w;
      }
      out[j + 0] = hip_bfloat16(sum4.x);
      out[j + 1] = hip_bfloat16(sum4.y);
      out[j + 2] = hip_bfloat16(sum4.z);
      out[j + 3] = hip_bfloat16(sum4.w);
    }
  }

  const int scalarStart = canVec4 ? vecEnd : 0;
  for (int j = scalarStart + laneId; j < hiddenDimSize; j += warpSize) {
    float sum = 0.0f;
#pragma unroll 4
    for (int n = 0; n < accumNum; n++) {
      const Fp8T* src = srcPtrs[n];
      if (src == nullptr) continue;
      sum += float(src[j]);
    }
    out[j] = hip_bfloat16(sum);
  }
}

}  // namespace detail
#endif

template <typename T>
__forceinline__ __device__ void WarpAccumCombineInternalFp8ToBf16(
    T* __restrict__ out, const CombineInternalFp8* const* __restrict__ srcPtrs, int accumNum,
    int laneId, int hiddenDimSize) {
#if defined(MORI_FP8_TYPE_OCP_ENABLED) || defined(MORI_FP8_TYPE_FNUZ_ENABLED)
  if constexpr (std::is_same_v<T, hip_bfloat16>) {
    switch (accumNum) {
      case 2:
        detail::WarpAccumCombineInternalFp8ToBf16Fixed<2>(
            reinterpret_cast<hip_bfloat16*>(out),
            reinterpret_cast<const detail::CombineInternalFp8T* const*>(srcPtrs), laneId,
            hiddenDimSize);
        break;
      case 4:
        detail::WarpAccumCombineInternalFp8ToBf16Fixed<4>(
            reinterpret_cast<hip_bfloat16*>(out),
            reinterpret_cast<const detail::CombineInternalFp8T* const*>(srcPtrs), laneId,
            hiddenDimSize);
        break;
      case 8:
        detail::WarpAccumCombineInternalFp8ToBf16Fixed<8>(
            reinterpret_cast<hip_bfloat16*>(out),
            reinterpret_cast<const detail::CombineInternalFp8T* const*>(srcPtrs), laneId,
            hiddenDimSize);
        break;
      default:
        detail::WarpAccumCombineInternalFp8ToBf16Dynamic(
            reinterpret_cast<hip_bfloat16*>(out),
            reinterpret_cast<const detail::CombineInternalFp8T* const*>(srcPtrs), accumNum, laneId,
            hiddenDimSize);
        break;
    }
  }
  // Note: when T != hip_bfloat16, this function is a no-op.
  // Callers should guard with if constexpr or ensure T is hip_bfloat16.
#else
  static_assert(!sizeof(T*), "WarpAccumCombineInternalFp8ToBf16 requires FP8 type support "
                              "(MORI_FP8_TYPE_OCP_ENABLED or MORI_FP8_TYPE_FNUZ_ENABLED)");
#endif
}

}  // namespace core
}  // namespace mori
