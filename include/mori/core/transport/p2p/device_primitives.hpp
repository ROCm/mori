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

#include <cstdint>

#include "mori/core/utils.hpp"

#ifndef HIP_FP8_CVT_FAST_PATH
#if defined(__AMDGCN__)
#define HIP_FP8_CVT_FAST_PATH 1
#else
#define HIP_FP8_CVT_FAST_PATH 0
#endif
#endif
namespace mori {
namespace core {

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

template <typename T>
inline __device__ void ThreadCopy(T* dst, T* src, size_t nelems) {
  constexpr int VecBytes = 16;
  using DataType = typename VecTypeSelector<VecBytes>::dataType;
  constexpr int vecSize = VecBytes / sizeof(T);
  int offset = 0;

  while ((offset + vecSize) <= nelems) {
    reinterpret_cast<uint4*>(dst + offset)[0] = reinterpret_cast<uint4*>(src + offset)[0];
    // store<VecBytes>(dst + offset, reinterpret_cast<DataType*>(src + offset)[0]);
    offset += vecSize;
  }

  while (offset < nelems) {
    dst[offset] = src[offset];
    // store<sizeof(T)>(dst + offset, src[offset]);
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

// template <typename T>
// inline __device__ void WarpCopy(T* dst, T* src, size_t nelems) {
//   constexpr int vecSize = 16 / sizeof(T);
//   int laneId = threadIdx.x & (warpSize - 1);
//   int offset = laneId * vecSize;

//   while ((offset + vecSize) <= nelems) {
//     reinterpret_cast<uint4*>(dst + offset)[0] = reinterpret_cast<uint4*>(src + offset)[0];
//     offset += warpSize * vecSize;
//   }

//   offset = offset - laneId * vecSize + laneId;
//   while (offset < nelems) {
//     dst[offset] = src[offset];
//     offset += warpSize;
//   }
// }

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

template <typename T>
inline __device__ T WarpReduceSum(T val) {
  int laneId = threadIdx.x & (warpSize - 1);
  for (int delta = (warpSize >> 1); delta > 0; delta = (delta >> 1)) {
    val += __shfl_down(val, delta);
  }
  return val;
}

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
  for (size_t iter = 0; iter < numIters; ++iter) {
    float accumValFp32[vecSize] = {0};
#pragma unroll
    for (int i = 0; i < accumNum; ++i) {
      if (srcs[i] == nullptr) continue;
      DataType srcVal = load<VecBytes>(srcs[i] + offset + laneOffset);
      float srcScale = (srcScales == nullptr) ? 1.0f : srcScales[i];
#pragma unroll
      for (int j = 0; j < vecSize; ++j) {
        accumValFp32[j] += float(reinterpret_cast<const T*>(&srcVal)[j]) * srcScale;
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
    float accumValFp32 = 0;
    for (int i = 0; i < accumNum; ++i) {
      const T* srcPtr = srcs[i];
      if (srcPtr == nullptr) continue;

      float srcScale = (srcScales == nullptr) ? 1.0f : srcScales[i];
      accumValFp32 += float(srcPtr[offset]) * srcScale;
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

  for (size_t iter = 0; iter < numIters; iter++) {
    float accumValFp32[Unroll][vecSize] = {0};

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
          accumValFp32[u][j] += float(reinterpret_cast<const T*>(&srcVals)[j]) * srcScale;
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

  for (size_t iter = 0; iter < numIters; ++iter) {
    float accumValFp32[vecSize] = {0};

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
          accumValFp32[j] += float(reinterpret_cast<const T*>(srcVals + i)[j]) * scales[i];
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
  offset += laneId;
  while (offset < nelems) {
    float accumValFp32 = 0;
#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      const T* srcPtr = srcs[i];
      if (srcPtr == nullptr) continue;

      float srcScale = (srcScales == nullptr) ? 1.0f : srcScales[i];
      accumValFp32 += float(srcPtr[offset]) * srcScale;
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

#if defined(HIP_FP8_TYPE_FNUZ) && HIP_FP8_TYPE_FNUZ == 1
using CombineInternalFp8T = __hip_fp8_e4m3_fnuz;
static constexpr float kCombineInternalFp8MaxFinite = 240.0f;
#elif defined(HIP_FP8_TYPE_OCP) && HIP_FP8_TYPE_OCP == 1
using CombineInternalFp8T = __hip_fp8_e4m3;
static constexpr float kCombineInternalFp8MaxFinite = 448.0f;
#else
static constexpr float kCombineInternalFp8MaxFinite = 0.0f;
#endif

__device__ __forceinline__ float WarpReduceMaxF32(float val) {
  for (int delta = (warpSize >> 1); delta > 0; delta >>= 1) {
    val = fmaxf(val, __shfl_down(val, delta));
  }
  return val;
}

__device__ __forceinline__ uint32_t WarpReduceMaxU32(uint32_t val) {
  for (int delta = (warpSize >> 1); delta > 0; delta >>= 1) {
    const int other = __shfl_down(static_cast<int>(val), delta);
    const int cur = static_cast<int>(val);
    val = static_cast<uint32_t>((cur > other) ? cur : other);
  }
  return val;
}

template <typename Fp8T>
__device__ __forceinline__ float2 CvtFp8x2ToFloat2(__hip_fp8x2_storage_t v) {
#if HIP_FP8_CVT_FAST_PATH
  auto f2 = __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<uint32_t>(static_cast<uint16_t>(v)), false);
  return float2{f2[0], f2[1]};
#else
  Fp8T lo;
  lo.__x = static_cast<__hip_fp8_storage_t>(static_cast<uint16_t>(v) & 0xFF);
  Fp8T hi;
  hi.__x = static_cast<__hip_fp8_storage_t>(static_cast<uint16_t>(v) >> 8);
  return float2{static_cast<float>(lo), static_cast<float>(hi)};
#endif
}

template <typename Fp8T>
__device__ __forceinline__ __hip_fp8x2_storage_t CvtFloat2ToFp8x2(float2 v) {
#if HIP_FP8_CVT_FAST_PATH
  if constexpr ((Fp8T::__default_interpret == __HIP_E4M3_FNUZ) ||
                (Fp8T::__default_interpret == __HIP_E4M3)) {
    const float fp8Max = kCombineInternalFp8MaxFinite;
    v.x = __builtin_amdgcn_fmed3f(v.x, fp8Max, -fp8Max);
    v.y = __builtin_amdgcn_fmed3f(v.y, fp8Max, -fp8Max);
    uint32_t packed = __builtin_amdgcn_cvt_pk_fp8_f32(v.x, v.y, 0, false);
    return static_cast<__hip_fp8x2_storage_t>(packed & 0xFFFF);
  }
#endif
  return __hip_cvt_float2_to_fp8x2(v, Fp8T::__default_saturation, Fp8T::__default_interpret);
}

template <typename OutT>
__device__ __forceinline__ void StoreOutPair(OutT* __restrict__ dst, int idx, float2 v) {
  if constexpr (sizeof(OutT) == 2) {
    union {
      uint32_t u32;
      OutT out[2];
    } tmp;
    tmp.out[0] = OutT(v.x);
    tmp.out[1] = OutT(v.y);
    store<4>(dst + idx, tmp.u32);
  } else {
    dst[idx] = OutT(v.x);
    dst[idx + 1] = OutT(v.y);
  }
}

template <typename Fp8T, typename InT>
__device__ __forceinline__ void WarpQuantizeToFp8Blockwise(Fp8T* __restrict__ dstToken,
                                                           float* __restrict__ dstScales,
                                                           const InT* __restrict__ srcToken,
                                                           int hiddenDim, int scaleDim) {
  const int laneId = threadIdx.x & (warpSize - 1);
  const int blockElems = (hiddenDim + scaleDim - 1) / scaleDim;

  const bool blockAligned2 = ((blockElems & 1) == 0);
  constexpr float fp8Max = kCombineInternalFp8MaxFinite;
  const float invFp8Max = 1.0f / fp8Max;

  auto* dstBytes = reinterpret_cast<__hip_fp8_storage_t*>(dstToken);

  constexpr int kMaxCachedVec2Iters = 4;
  constexpr int kMaxCachedElems = warpSize * 2 * kMaxCachedVec2Iters;
  const bool srcAligned4 = ((reinterpret_cast<uintptr_t>(srcToken) & 0x3) == 0);
  const bool canCacheVec2 = blockAligned2 && srcAligned4 && (blockElems <= kMaxCachedElems) &&
                            std::is_same_v<InT, hip_bfloat16>;

  auto bf16_abs_bits = [](uint16_t bits) -> uint16_t {
    // Match fmaxf(fabsf(x), ...) semantics for NaNs: ignore NaN payloads.
    bits = static_cast<uint16_t>(bits & 0x7FFF);
    const uint16_t exp = static_cast<uint16_t>(bits & 0x7F80);
    const uint16_t mant = static_cast<uint16_t>(bits & 0x007F);
    if ((exp == 0x7F80) && (mant != 0)) return 0;
    return bits;
  };

  for (int sb = 0; sb < scaleDim; ++sb) {
    const int start = sb * blockElems;
    const int end = std::min(start + blockElems, hiddenDim);

    float maxAbs = 0.0f;
    if (canCacheVec2) {
      uint32_t cachedPairs[kMaxCachedVec2Iters];
      int iters = 0;

      uint32_t localMaxBits = 0;
      const int base = start + (laneId << 1);
      int idx = base;
      for (; (idx + 1) < end; idx += (warpSize << 1)) {
        const uint32_t packed = load<4>(srcToken + idx);
        cachedPairs[iters] = packed;
        iters++;

        const uint16_t lo = bf16_abs_bits(static_cast<uint16_t>(packed & 0xFFFF));
        const uint16_t hi = bf16_abs_bits(static_cast<uint16_t>(packed >> 16));
        const uint32_t pairMax = static_cast<uint32_t>((lo > hi) ? lo : hi);
        localMaxBits = (localMaxBits > pairMax) ? localMaxBits : pairMax;
      }

      uint16_t tailBits = 0;
      bool hasTail = false;
      if (idx < end) {
        tailBits = reinterpret_cast<const hip_bfloat16*>(srcToken)[idx].data;
        const uint32_t tailAbs = static_cast<uint32_t>(bf16_abs_bits(tailBits));
        localMaxBits = (localMaxBits > tailAbs) ? localMaxBits : tailAbs;
        hasTail = true;
      }

      uint32_t maxBits = WarpReduceMaxU32(localMaxBits);
      maxBits = static_cast<uint32_t>(__shfl(static_cast<int>(maxBits), 0));
      hip_bfloat16 bf;
      bf.data = static_cast<__hip_uint16_t>(maxBits);
      maxAbs = static_cast<float>(bf);

      const float scale = (maxAbs == 0.0f) ? 1.0f : (maxAbs * invFp8Max);
      if (laneId == 0) dstScales[sb] = scale;
      const float invScale = (maxAbs == 0.0f) ? 1.0f : (fp8Max / maxAbs);

      for (int i = 0, storeIdx = base; i < iters; ++i, storeIdx += (warpSize << 1)) {
        const uint32_t packed = cachedPairs[i];
        hip_bfloat16 lo;
        lo.data = static_cast<__hip_uint16_t>(packed & 0xFFFF);
        hip_bfloat16 hi;
        hi.data = static_cast<__hip_uint16_t>(packed >> 16);
        float2 v;
        v.x = static_cast<float>(lo) * invScale;
        v.y = static_cast<float>(hi) * invScale;
        __hip_fp8x2_storage_t fp8 = CvtFloat2ToFp8x2<Fp8T>(v);
        store<2>(dstBytes + storeIdx, static_cast<uint16_t>(fp8));
      }

      if (hasTail) {
        hip_bfloat16 tail;
        tail.data = static_cast<__hip_uint16_t>(tailBits);
        dstToken[idx] = Fp8T(static_cast<float>(tail) * invScale);
      }
    } else {
      float localMaxAbs = 0.0f;
      if constexpr (std::is_same_v<InT, hip_bfloat16>) {
        uint32_t localMaxBits = 0;
        if (blockAligned2) {
          int idx = start + (laneId << 1);
          for (; (idx + 1) < end; idx += (warpSize << 1)) {
            const __hip_uint16_t b0 = reinterpret_cast<const hip_bfloat16*>(srcToken)[idx].data;
            const __hip_uint16_t b1 = reinterpret_cast<const hip_bfloat16*>(srcToken)[idx + 1].data;
            const uint16_t abs0 = bf16_abs_bits(static_cast<uint16_t>(b0));
            const uint16_t abs1 = bf16_abs_bits(static_cast<uint16_t>(b1));
            const uint32_t pairMax = static_cast<uint32_t>((abs0 > abs1) ? abs0 : abs1);
            localMaxBits = (localMaxBits > pairMax) ? localMaxBits : pairMax;
          }
          if (idx < end) {
            const __hip_uint16_t b0 = reinterpret_cast<const hip_bfloat16*>(srcToken)[idx].data;
            const uint32_t abs0 = static_cast<uint32_t>(bf16_abs_bits(b0));
            localMaxBits = (localMaxBits > abs0) ? localMaxBits : abs0;
          }
        } else {
          for (int idx = start + laneId; idx < end; idx += warpSize) {
            const __hip_uint16_t b0 = reinterpret_cast<const hip_bfloat16*>(srcToken)[idx].data;
            const uint32_t abs0 = static_cast<uint32_t>(bf16_abs_bits(b0));
            localMaxBits = (localMaxBits > abs0) ? localMaxBits : abs0;
          }
        }
        uint32_t maxBits = WarpReduceMaxU32(localMaxBits);
        maxBits = static_cast<uint32_t>(__shfl(static_cast<int>(maxBits), 0));
        hip_bfloat16 bf;
        bf.data = static_cast<__hip_uint16_t>(maxBits);
        maxAbs = static_cast<float>(bf);
      } else {
        if (blockAligned2) {
          int idx = start + (laneId << 1);
          for (; (idx + 1) < end; idx += (warpSize << 1)) {
            const float v0 = static_cast<float>(srcToken[idx]);
            const float v1 = static_cast<float>(srcToken[idx + 1]);
            localMaxAbs = fmaxf(localMaxAbs, fabsf(v0));
            localMaxAbs = fmaxf(localMaxAbs, fabsf(v1));
          }
          if (idx < end) {
            const float v0 = static_cast<float>(srcToken[idx]);
            localMaxAbs = fmaxf(localMaxAbs, fabsf(v0));
          }
        } else {
          for (int idx = start + laneId; idx < end; idx += warpSize) {
            localMaxAbs = fmaxf(localMaxAbs, fabsf(static_cast<float>(srcToken[idx])));
          }
        }
        maxAbs = WarpReduceMaxF32(localMaxAbs);
        maxAbs = __shfl(maxAbs, 0);
      }

      const float scale = (maxAbs == 0.0f) ? 1.0f : (maxAbs * invFp8Max);
      if (laneId == 0) dstScales[sb] = scale;
      const float invScale = (maxAbs == 0.0f) ? 1.0f : (fp8Max / maxAbs);

      if (blockAligned2) {
        int idx = start + (laneId << 1);
        for (; (idx + 1) < end; idx += (warpSize << 1)) {
          float2 v;
          v.x = static_cast<float>(srcToken[idx]) * invScale;
          v.y = static_cast<float>(srcToken[idx + 1]) * invScale;
          __hip_fp8x2_storage_t packed = CvtFloat2ToFp8x2<Fp8T>(v);
          store<2>(dstBytes + idx, static_cast<uint16_t>(packed));
        }
        if (idx < end) {
          dstToken[idx] = Fp8T(static_cast<float>(srcToken[idx]) * invScale);
        }
      } else {
        for (int idx = start + laneId; idx < end; idx += warpSize) {
          dstToken[idx] = Fp8T(static_cast<float>(srcToken[idx]) * invScale);
        }
      }
    }
  }
}

template <typename OutT, typename Fp8T, int AccumNum>
__device__ __forceinline__ void WarpAccumFp8DequantFullImpl(
    OutT* __restrict__ dstToken, const Fp8T* const* __restrict__ srcs,
    const float* const* __restrict__ srcScales, int hiddenDim, int scaleDim) {
  const int laneId = threadIdx.x & (warpSize - 1);
  const int blockElems = (hiddenDim + scaleDim - 1) / scaleDim;

  const Fp8T* cachedSrcs[AccumNum];
  const float* cachedScalePtrs[AccumNum];
  const __hip_fp8_storage_t* cachedSrcBytes[AccumNum];
#pragma unroll AccumNum
  for (int i = 0; i < AccumNum; ++i) {
    cachedSrcs[i] = srcs[i];
    cachedScalePtrs[i] =
        (cachedSrcs[i] != nullptr && srcScales[i] != nullptr) ? srcScales[i] : nullptr;
    cachedSrcBytes[i] = (cachedSrcs[i] != nullptr)
                            ? reinterpret_cast<const __hip_fp8_storage_t*>(cachedSrcs[i])
                            : nullptr;
  }

  const bool useVec2 =
      ((blockElems & 1) == 0) && ((reinterpret_cast<uintptr_t>(dstToken) & 0x3) == 0);
  for (int sb = 0; sb < scaleDim; ++sb) {
    const int start = sb * blockElems;
    const int end = std::min(start + blockElems, hiddenDim);

    float sbScales[AccumNum];
#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      sbScales[i] = cachedScalePtrs[i] != nullptr ? cachedScalePtrs[i][sb] : 1.0f;
    }

    if (useVec2) {
      int idx = start + (laneId << 1);
      for (; (idx + 1) < end; idx += (warpSize << 1)) {
        float2 acc2{0.0f, 0.0f};
#pragma unroll AccumNum
        for (int i = 0; i < AccumNum; ++i) {
          if (cachedSrcs[i] == nullptr) continue;
          __hip_fp8x2_storage_t packed =
              static_cast<__hip_fp8x2_storage_t>(load<2>(cachedSrcBytes[i] + idx));
          float2 v = CvtFp8x2ToFloat2<Fp8T>(packed);
          const float s = sbScales[i];
          acc2.x = fmaf(v.x, s, acc2.x);
          acc2.y = fmaf(v.y, s, acc2.y);
        }
        StoreOutPair(dstToken, idx, acc2);
      }
      if (idx < end) {
        float acc = 0.0f;
#pragma unroll AccumNum
        for (int i = 0; i < AccumNum; ++i) {
          if (cachedSrcs[i] == nullptr) continue;
          acc += static_cast<float>(cachedSrcs[i][idx]) * sbScales[i];
        }
        dstToken[idx] = OutT(acc);
      }
    } else {
      for (int idx = start + laneId; idx < end; idx += warpSize) {
        float acc = 0.0f;
#pragma unroll AccumNum
        for (int i = 0; i < AccumNum; ++i) {
          if (cachedSrcs[i] == nullptr) continue;
          acc += static_cast<float>(cachedSrcs[i][idx]) * sbScales[i];
        }
        dstToken[idx] = OutT(acc);
      }
    }
  }
}

template <typename OutT, typename Fp8T>
__device__ __forceinline__ void WarpAccumFp8DequantFull(OutT* __restrict__ dstToken,
                                                        const Fp8T* const* __restrict__ srcs,
                                                        const float* const* __restrict__ srcScales,
                                                        int accumNum, int hiddenDim, int scaleDim) {
  switch (accumNum) {
    case 1:
      WarpAccumFp8DequantFullImpl<OutT, Fp8T, 1>(dstToken, srcs, srcScales, hiddenDim, scaleDim);
      break;
    case 2:
      WarpAccumFp8DequantFullImpl<OutT, Fp8T, 2>(dstToken, srcs, srcScales, hiddenDim, scaleDim);
      break;
    case 4:
      WarpAccumFp8DequantFullImpl<OutT, Fp8T, 4>(dstToken, srcs, srcScales, hiddenDim, scaleDim);
      break;
    case 6:
      WarpAccumFp8DequantFullImpl<OutT, Fp8T, 6>(dstToken, srcs, srcScales, hiddenDim, scaleDim);
      break;
    case 8:
      WarpAccumFp8DequantFullImpl<OutT, Fp8T, 8>(dstToken, srcs, srcScales, hiddenDim, scaleDim);
      break;
    case 10:
      WarpAccumFp8DequantFullImpl<OutT, Fp8T, 10>(dstToken, srcs, srcScales, hiddenDim, scaleDim);
      break;
    default: {
      const int laneId = threadIdx.x & (warpSize - 1);
      const int blockElems = (hiddenDim + scaleDim - 1) / scaleDim;
      for (int sb = 0; sb < scaleDim; ++sb) {
        const int start = sb * blockElems;
        const int end = std::min(start + blockElems, hiddenDim);
        for (int idx = start + laneId; idx < end; idx += warpSize) {
          float acc = 0.0f;
          for (int i = 0; i < accumNum; ++i) {
            if (srcs[i] != nullptr && srcScales[i] != nullptr) {
              acc += static_cast<float>(srcs[i][idx]) * srcScales[i][sb];
            }
          }
          dstToken[idx] = OutT(acc);
        }
      }
      break;
    }
  }
}

template <typename OutT, typename Fp8T, int AccumNum>
__device__ __forceinline__ void WarpAccumFp8DequantSegmentImpl(
    OutT* __restrict__ dstToken, const Fp8T* const* __restrict__ srcs,
    const float* const* __restrict__ srcScales, int hiddenDimOffset, int hiddenDimSize,
    int hiddenDim, int scaleDim) {
  const int laneId = threadIdx.x & (warpSize - 1);
  const int blockElems = (hiddenDim + scaleDim - 1) / scaleDim;

  const Fp8T* cachedSrcs[AccumNum];
  const float* cachedScalePtrs[AccumNum];
  const __hip_fp8_storage_t* cachedSrcBytes[AccumNum];
#pragma unroll AccumNum
  for (int i = 0; i < AccumNum; ++i) {
    cachedSrcs[i] = srcs[i];
    cachedScalePtrs[i] =
        (cachedSrcs[i] != nullptr && srcScales[i] != nullptr) ? srcScales[i] : nullptr;
    cachedSrcBytes[i] = (cachedSrcs[i] != nullptr)
                            ? reinterpret_cast<const __hip_fp8_storage_t*>(cachedSrcs[i])
                            : nullptr;
  }

  const bool blockAligned2 = ((blockElems & 1) == 0);
  const bool offsetAligned2 = ((hiddenDimOffset & 1) == 0);
  const bool useVec2 =
      blockAligned2 && offsetAligned2 && ((reinterpret_cast<uintptr_t>(dstToken) & 0x3) == 0);

  const int globalStart = hiddenDimOffset;
  const int globalEnd = hiddenDimOffset + hiddenDimSize;
  const int sbStart = globalStart / blockElems;
  const int sbEnd = (globalEnd - 1) / blockElems;

  for (int sb = sbStart; sb <= sbEnd; ++sb) {
    const int blockStart = sb * blockElems;
    const int blockEnd = std::min(blockStart + blockElems, hiddenDim);
    int segStart = (globalStart > blockStart) ? globalStart : blockStart;
    int segEnd = (globalEnd < blockEnd) ? globalEnd : blockEnd;

    int localStart = segStart - hiddenDimOffset;
    int localEnd = segEnd - hiddenDimOffset;
    if (localStart >= localEnd) continue;

    float sbScales[AccumNum];
#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      sbScales[i] = cachedScalePtrs[i] != nullptr ? cachedScalePtrs[i][sb] : 1.0f;
    }

    if (useVec2) {
      if ((localStart & 1) != 0) {
        if (laneId == 0) {
          float acc = 0.0f;
#pragma unroll AccumNum
          for (int i = 0; i < AccumNum; ++i) {
            if (cachedSrcs[i] == nullptr) continue;
            acc += static_cast<float>(cachedSrcs[i][localStart]) * sbScales[i];
          }
          dstToken[localStart] = OutT(acc);
        }
        localStart += 1;
      }

      int idx = localStart + (laneId << 1);
      for (; (idx + 1) < localEnd; idx += (warpSize << 1)) {
        float2 acc2{0.0f, 0.0f};
#pragma unroll AccumNum
        for (int i = 0; i < AccumNum; ++i) {
          if (cachedSrcs[i] == nullptr) continue;
          __hip_fp8x2_storage_t packed =
              static_cast<__hip_fp8x2_storage_t>(load<2>(cachedSrcBytes[i] + idx));
          float2 v = CvtFp8x2ToFloat2<Fp8T>(packed);
          const float s = sbScales[i];
          acc2.x = fmaf(v.x, s, acc2.x);
          acc2.y = fmaf(v.y, s, acc2.y);
        }
        StoreOutPair(dstToken, idx, acc2);
      }

      if (idx < localEnd) {
        float acc = 0.0f;
#pragma unroll AccumNum
        for (int i = 0; i < AccumNum; ++i) {
          if (cachedSrcs[i] == nullptr) continue;
          acc += static_cast<float>(cachedSrcs[i][idx]) * sbScales[i];
        }
        dstToken[idx] = OutT(acc);
      }
    } else {
      for (int idx = localStart + laneId; idx < localEnd; idx += warpSize) {
        float acc = 0.0f;
#pragma unroll AccumNum
        for (int i = 0; i < AccumNum; ++i) {
          if (cachedSrcs[i] == nullptr) continue;
          acc += static_cast<float>(cachedSrcs[i][idx]) * sbScales[i];
        }
        dstToken[idx] = OutT(acc);
      }
    }
  }
}

template <typename OutT, typename Fp8T>
__device__ __forceinline__ void WarpAccumFp8DequantSegment(
    OutT* __restrict__ dstToken, const Fp8T* const* __restrict__ srcs,
    const float* const* __restrict__ srcScales, int accumNum, int hiddenDimOffset,
    int hiddenDimSize, int hiddenDim, int scaleDim) {
  switch (accumNum) {
    case 1:
      WarpAccumFp8DequantSegmentImpl<OutT, Fp8T, 1>(dstToken, srcs, srcScales, hiddenDimOffset,
                                                    hiddenDimSize, hiddenDim, scaleDim);
      break;
    case 2:
      WarpAccumFp8DequantSegmentImpl<OutT, Fp8T, 2>(dstToken, srcs, srcScales, hiddenDimOffset,
                                                    hiddenDimSize, hiddenDim, scaleDim);
      break;
    case 4:
      WarpAccumFp8DequantSegmentImpl<OutT, Fp8T, 4>(dstToken, srcs, srcScales, hiddenDimOffset,
                                                    hiddenDimSize, hiddenDim, scaleDim);
      break;
    case 6:
      WarpAccumFp8DequantSegmentImpl<OutT, Fp8T, 6>(dstToken, srcs, srcScales, hiddenDimOffset,
                                                    hiddenDimSize, hiddenDim, scaleDim);
      break;
    case 8:
      WarpAccumFp8DequantSegmentImpl<OutT, Fp8T, 8>(dstToken, srcs, srcScales, hiddenDimOffset,
                                                    hiddenDimSize, hiddenDim, scaleDim);
      break;
    case 10:
      WarpAccumFp8DequantSegmentImpl<OutT, Fp8T, 10>(dstToken, srcs, srcScales, hiddenDimOffset,
                                                     hiddenDimSize, hiddenDim, scaleDim);
      break;
    default: {
      const int laneId = threadIdx.x & (warpSize - 1);
      const int blockElems = (hiddenDim + scaleDim - 1) / scaleDim;
      for (int idx = laneId; idx < hiddenDimSize; idx += warpSize) {
        const int globalIdx = hiddenDimOffset + idx;
        const int sb = globalIdx / blockElems;
        float acc = 0.0f;
        for (int i = 0; i < accumNum; ++i) {
          if (srcs[i] != nullptr && srcScales[i] != nullptr) {
            acc += static_cast<float>(srcs[i][idx]) * srcScales[i][sb];
          }
        }
        dstToken[idx] = OutT(acc);
      }
      break;
    }
  }
}

}  // namespace core
}  // namespace mori
