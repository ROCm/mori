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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "mori/core/utils.hpp"
#include "mori/utils/data_types.hpp"

#ifndef HIP_FP8_CVT_FAST_PATH
#if defined(__AMDGCN__)
#define HIP_FP8_CVT_FAST_PATH 1
#else
#define HIP_FP8_CVT_FAST_PATH 0
#endif
#endif
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

#ifdef MORI_HAS_OCP_FP
template <>
struct VecTypeAdaptor<mori_fp4_e2m1, 2> {
  using dataType = mori_fp4x2_e2m1;
};

template <>
struct VecTypeAdaptor<mori_fp4_e2m1, 4> {
  using dataType = mori_fp4x4_e2m1;
};
#endif  // MORI_HAS_OCP_FP

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

template <typename T, int VecBytes, int Unroll>
inline __device__ void BlockCopyImpl(T* __restrict__ dst, const T* __restrict__ src, size_t& offset,
                                     size_t nelems) {
  static_assert((VecBytes <= 16) && (VecBytes >= 1) && IsPowerOf2(VecBytes));
  constexpr int vecSize = VecBytes / sizeof(T);
  if constexpr (vecSize <= 0) return;

  const int threadId = FlatBlockThreadId();
  const int blockSize = FlatBlockSize();
  using DataType = typename VecTypeSelector<VecBytes>::dataType;

  const size_t elemsPerBlock = static_cast<size_t>(Unroll) * blockSize * vecSize;
  const size_t numIters = (nelems - offset) / elemsPerBlock;
  for (size_t iter = 0; iter < numIters; iter++) {
    DataType vec[Unroll];
#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      vec[u] = load<VecBytes>(src + offset + (threadId + u * blockSize) * vecSize);
    }

#pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      store<VecBytes>(dst + offset + (threadId + u * blockSize) * vecSize, vec[u]);
    }

    offset += elemsPerBlock;
  }
}

template <typename T, int VecBytes, int Unroll = 1>
inline __device__ void BlockCopyVec(T* __restrict__ dst, const T* __restrict__ src, size_t& offset,
                                    size_t nelems) {
  if constexpr ((VecBytes % sizeof(T)) == 0) {
    BlockCopyImpl<T, VecBytes, Unroll>(dst, src, offset, nelems);
    if constexpr (Unroll > 1) {
      BlockCopyImpl<T, VecBytes, 1>(dst, src, offset, nelems);
    }
  }
}

template <typename T, int Unroll = 1>
inline __device__ void BlockCopy(T* __restrict__ dst, const T* __restrict__ src, size_t nelems) {
  const int threadId = FlatBlockThreadId();
  const int blockSize = FlatBlockSize();

  size_t offset = 0;
  BlockCopyVec<T, 16, Unroll>(dst, src, offset, nelems);
  BlockCopyVec<T, 8, Unroll>(dst, src, offset, nelems);
  BlockCopyVec<T, 4, Unroll>(dst, src, offset, nelems);
  BlockCopyVec<T, 2, Unroll>(dst, src, offset, nelems);
  BlockCopyVec<T, 1, Unroll>(dst, src, offset, nelems);

  offset += threadId;
  while (offset < nelems) {
    dst[offset] = src[offset];
    offset += blockSize;
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

#if defined(MORI_FP8_TYPE_OCP_ENABLED) || defined(MORI_FP8_TYPE_FNUZ_ENABLED)
#if defined(MORI_FP8_TYPE_OCP_ENABLED)
static constexpr float kCombineInternalFp8MaxFinite = 448.0f;
#else
static constexpr float kCombineInternalFp8MaxFinite = 240.0f;
#endif

__device__ __forceinline__ float WarpReduceMaxF32(float val) {
  for (int delta = (warpSize >> 1); delta > 0; delta >>= 1) {
    val = fmaxf(val, __shfl_down(val, delta));
  }
  return val;
}

template <int Width>
__device__ __forceinline__ uint32_t SubwarpReduceMaxU32(uint32_t val) {
  for (int delta = (Width >> 1); delta > 0; delta >>= 1) {
    const int other = __shfl_down(static_cast<int>(val), delta, Width);
    const int cur = static_cast<int>(val);
    val = static_cast<uint32_t>((cur > other) ? cur : other);
  }
  return val;
}

// Float32 (IEEE 754): 1 sign + 8 exp + 23 mantissa
// BFloat16: 1 sign + 8 exp + 7 mantissa
// Bf16BitsToF32 is faster than __bfloat162float
__device__ __forceinline__ float BitCastU32ToF32(uint32_t u) {
  union {
    uint32_t u32;
    float f32;
  } tmp;
  tmp.u32 = u;
  return tmp.f32;
}

__device__ __forceinline__ float Bf16BitsToF32(uint16_t bf16_bits) {
  return BitCastU32ToF32(static_cast<uint32_t>(bf16_bits) << 16);
}

template <typename Fp8T>
__device__ __forceinline__ float2 CvtFp8x2ToFloat2(__hip_fp8x2_storage_t v) {
#if HIP_FP8_CVT_FAST_PATH
  if constexpr ((Fp8T::__default_interpret == __HIP_E4M3_FNUZ) ||
                (Fp8T::__default_interpret == __HIP_E4M3)) {
    auto f2 =
        __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<uint32_t>(static_cast<uint16_t>(v)), false);
    return float2{f2[0], f2[1]};
  }
  if constexpr ((Fp8T::__default_interpret == __HIP_E5M2_FNUZ) ||
                (Fp8T::__default_interpret == __HIP_E5M2)) {
    auto f2 =
        __builtin_amdgcn_cvt_pk_f32_bf8(static_cast<uint32_t>(static_cast<uint16_t>(v)), false);
    return float2{f2[0], f2[1]};
  }
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
  if constexpr ((Fp8T::__default_interpret == __HIP_E5M2_FNUZ) ||
                (Fp8T::__default_interpret == __HIP_E5M2)) {
    const float fp8Max = kCombineInternalFp8MaxFinite;
    v.x = __builtin_amdgcn_fmed3f(v.x, fp8Max, -fp8Max);
    v.y = __builtin_amdgcn_fmed3f(v.y, fp8Max, -fp8Max);
    uint32_t packed = __builtin_amdgcn_cvt_pk_bf8_f32(v.x, v.y, 0, false);
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

namespace detail {

__device__ __forceinline__ uint16_t Bf16AbsBits(uint16_t bits) {
  // Match fmaxf(fabsf(x), ...) semantics for NaNs: ignore NaN payloads.
  bits = static_cast<uint16_t>(bits & 0x7FFF);
  const uint16_t exp = static_cast<uint16_t>(bits & 0x7F80);
  const uint16_t mant = static_cast<uint16_t>(bits & 0x007F);
  if ((exp == 0x7F80) && (mant != 0)) return 0;
  return bits;
}

template <int InVecBytes>
struct Bf16Vec;

template <>
struct Bf16Vec<4> {
  using LoadT = uint32_t;
  static constexpr int kElems = 2;

  __device__ __forceinline__ static uint32_t MaxAbsBits(LoadT packed) {
    const uint16_t lo = Bf16AbsBits(static_cast<uint16_t>(packed & 0xFFFF));
    const uint16_t hi = Bf16AbsBits(static_cast<uint16_t>(packed >> 16));
    return static_cast<uint32_t>((lo > hi) ? lo : hi);
  }

  template <typename Fp8T>
  __device__ __forceinline__ static void QuantizeStore(__hip_fp8_storage_t* __restrict__ dstBytes,
                                                       int idx, LoadT packed, float invScale) {
    float2 v;
    if (invScale == 1.0f) {
      v.x = Bf16BitsToF32(static_cast<uint16_t>(packed & 0xFFFF));
      v.y = Bf16BitsToF32(static_cast<uint16_t>(packed >> 16));
    } else {
      v.x = Bf16BitsToF32(static_cast<uint16_t>(packed & 0xFFFF)) * invScale;
      v.y = Bf16BitsToF32(static_cast<uint16_t>(packed >> 16)) * invScale;
    }
    __hip_fp8x2_storage_t fp8 = CvtFloat2ToFp8x2<Fp8T>(v);
    store<2>(dstBytes + idx, static_cast<uint16_t>(fp8));
  }
};

template <>
struct Bf16Vec<8> {
  using LoadT = uint64_t;
  static constexpr int kElems = 4;

  __device__ __forceinline__ static uint32_t MaxAbsBits(LoadT packed) {
    const uint16_t b0 = Bf16AbsBits(static_cast<uint16_t>(packed & 0xFFFF));
    const uint16_t b1 = Bf16AbsBits(static_cast<uint16_t>((packed >> 16) & 0xFFFF));
    const uint16_t b2 = Bf16AbsBits(static_cast<uint16_t>((packed >> 32) & 0xFFFF));
    const uint16_t b3 = Bf16AbsBits(static_cast<uint16_t>((packed >> 48) & 0xFFFF));
    const uint16_t m01 = (b0 > b1) ? b0 : b1;
    const uint16_t m23 = (b2 > b3) ? b2 : b3;
    return static_cast<uint32_t>((m01 > m23) ? m01 : m23);
  }

  template <typename Fp8T>
  __device__ __forceinline__ static void QuantizeStore(__hip_fp8_storage_t* __restrict__ dstBytes,
                                                       int idx, LoadT packed, float invScale) {
    const uint16_t b0 = static_cast<uint16_t>(packed & 0xFFFF);
    const uint16_t b1 = static_cast<uint16_t>((packed >> 16) & 0xFFFF);
    const uint16_t b2 = static_cast<uint16_t>((packed >> 32) & 0xFFFF);
    const uint16_t b3 = static_cast<uint16_t>((packed >> 48) & 0xFFFF);

    float2 v01;
    float2 v23;
    if (invScale == 1.0f) {
      v01.x = Bf16BitsToF32(b0);
      v01.y = Bf16BitsToF32(b1);
      v23.x = Bf16BitsToF32(b2);
      v23.y = Bf16BitsToF32(b3);
    } else {
      v01.x = Bf16BitsToF32(b0) * invScale;
      v01.y = Bf16BitsToF32(b1) * invScale;
      v23.x = Bf16BitsToF32(b2) * invScale;
      v23.y = Bf16BitsToF32(b3) * invScale;
    }

    const __hip_fp8x2_storage_t fp8_01 = CvtFloat2ToFp8x2<Fp8T>(v01);
    const __hip_fp8x2_storage_t fp8_23 = CvtFloat2ToFp8x2<Fp8T>(v23);
    const uint32_t packedFp8 = static_cast<uint32_t>(static_cast<uint16_t>(fp8_01)) |
                               (static_cast<uint32_t>(static_cast<uint16_t>(fp8_23)) << 16);
    store<4>(dstBytes + idx, packedFp8);
  }
};

template <>
struct Bf16Vec<16> {
  using LoadT = ulong2;
  static constexpr int kElems = 8;

  __device__ __forceinline__ static uint32_t MaxAbsBits(LoadT packed) {
    const uint64_t p0 = packed.x;
    const uint64_t p1 = packed.y;
    const uint16_t b0 = Bf16AbsBits(static_cast<uint16_t>(p0 & 0xFFFF));
    const uint16_t b1 = Bf16AbsBits(static_cast<uint16_t>((p0 >> 16) & 0xFFFF));
    const uint16_t b2 = Bf16AbsBits(static_cast<uint16_t>((p0 >> 32) & 0xFFFF));
    const uint16_t b3 = Bf16AbsBits(static_cast<uint16_t>((p0 >> 48) & 0xFFFF));
    const uint16_t b4 = Bf16AbsBits(static_cast<uint16_t>(p1 & 0xFFFF));
    const uint16_t b5 = Bf16AbsBits(static_cast<uint16_t>((p1 >> 16) & 0xFFFF));
    const uint16_t b6 = Bf16AbsBits(static_cast<uint16_t>((p1 >> 32) & 0xFFFF));
    const uint16_t b7 = Bf16AbsBits(static_cast<uint16_t>((p1 >> 48) & 0xFFFF));
    uint16_t m01 = (b0 > b1) ? b0 : b1;
    uint16_t m23 = (b2 > b3) ? b2 : b3;
    uint16_t m45 = (b4 > b5) ? b4 : b5;
    uint16_t m67 = (b6 > b7) ? b6 : b7;
    uint16_t m0123 = (m01 > m23) ? m01 : m23;
    uint16_t m4567 = (m45 > m67) ? m45 : m67;
    return static_cast<uint32_t>((m0123 > m4567) ? m0123 : m4567);
  }

  template <typename Fp8T>
  __device__ __forceinline__ static void QuantizeStore(__hip_fp8_storage_t* __restrict__ dstBytes,
                                                       int idx, LoadT packed, float invScale) {
    const uint64_t p0 = packed.x;
    const uint64_t p1 = packed.y;
    const uint16_t b0 = static_cast<uint16_t>(p0 & 0xFFFF);
    const uint16_t b1 = static_cast<uint16_t>((p0 >> 16) & 0xFFFF);
    const uint16_t b2 = static_cast<uint16_t>((p0 >> 32) & 0xFFFF);
    const uint16_t b3 = static_cast<uint16_t>((p0 >> 48) & 0xFFFF);
    const uint16_t b4 = static_cast<uint16_t>(p1 & 0xFFFF);
    const uint16_t b5 = static_cast<uint16_t>((p1 >> 16) & 0xFFFF);
    const uint16_t b6 = static_cast<uint16_t>((p1 >> 32) & 0xFFFF);
    const uint16_t b7 = static_cast<uint16_t>((p1 >> 48) & 0xFFFF);

    float2 v01;
    float2 v23;
    float2 v45;
    float2 v67;
    if (invScale == 1.0f) {
      v01.x = Bf16BitsToF32(b0);
      v01.y = Bf16BitsToF32(b1);
      v23.x = Bf16BitsToF32(b2);
      v23.y = Bf16BitsToF32(b3);
      v45.x = Bf16BitsToF32(b4);
      v45.y = Bf16BitsToF32(b5);
      v67.x = Bf16BitsToF32(b6);
      v67.y = Bf16BitsToF32(b7);
    } else {
      v01.x = Bf16BitsToF32(b0) * invScale;
      v01.y = Bf16BitsToF32(b1) * invScale;
      v23.x = Bf16BitsToF32(b2) * invScale;
      v23.y = Bf16BitsToF32(b3) * invScale;
      v45.x = Bf16BitsToF32(b4) * invScale;
      v45.y = Bf16BitsToF32(b5) * invScale;
      v67.x = Bf16BitsToF32(b6) * invScale;
      v67.y = Bf16BitsToF32(b7) * invScale;
    }

    const __hip_fp8x2_storage_t fp8_01 = CvtFloat2ToFp8x2<Fp8T>(v01);
    const __hip_fp8x2_storage_t fp8_23 = CvtFloat2ToFp8x2<Fp8T>(v23);
    const __hip_fp8x2_storage_t fp8_45 = CvtFloat2ToFp8x2<Fp8T>(v45);
    const __hip_fp8x2_storage_t fp8_67 = CvtFloat2ToFp8x2<Fp8T>(v67);
    const uint64_t packedFp8 = static_cast<uint64_t>(static_cast<uint16_t>(fp8_01)) |
                               (static_cast<uint64_t>(static_cast<uint16_t>(fp8_23)) << 16) |
                               (static_cast<uint64_t>(static_cast<uint16_t>(fp8_45)) << 32) |
                               (static_cast<uint64_t>(static_cast<uint16_t>(fp8_67)) << 48);
    store<8>(dstBytes + idx, packedFp8);
  }
};

template <int SubwarpSize, int InVecBytes, int MaxCacheIters, typename Fp8T>
__device__ __forceinline__ void WarpQuantizeBf16ToFp8BlockwiseVec(
    Fp8T* __restrict__ dstToken, float* __restrict__ dstScales,
    const hip_bfloat16* __restrict__ srcToken, int hiddenDim, int scaleDim) {
  static_assert((SubwarpSize & (SubwarpSize - 1)) == 0, "SubwarpSize must be a power of two");
  static_assert(SubwarpSize <= warpSize, "SubwarpSize must be <= warpSize");

  constexpr int kVecElems = Bf16Vec<InVecBytes>::kElems;
  constexpr int kStrideElems = SubwarpSize * kVecElems;
  constexpr float fp8Max = kCombineInternalFp8MaxFinite;
  const float invFp8Max = 1.0f / fp8Max;

  const int laneId = threadIdx.x & (warpSize - 1);
  const int subLaneId = laneId & (SubwarpSize - 1);
  const int subWarpId = laneId / SubwarpSize;
  constexpr int kSubwarpsPerWarp = warpSize / SubwarpSize;

  auto* dstBytes = reinterpret_cast<__hip_fp8_storage_t*>(dstToken);

  const int blockElems = (hiddenDim + scaleDim - 1) / scaleDim;
  const int maxIters = (blockElems + kStrideElems - 1) / kStrideElems;

  bool subwarpScaled = false;

  for (int sbBase = 0; sbBase < scaleDim; sbBase += kSubwarpsPerWarp) {
    const int sb = sbBase + subWarpId;
    if (sb >= scaleDim) continue;

    const int start = sb * blockElems;
    const int end = std::min(start + blockElems, hiddenDim);
    const int base = start + subLaneId * kVecElems;

    uint32_t localMaxBits = 0;
    float maxAbs = 0.0f;

    // Fast path for the common case where a subwarp covers the whole block in a single vector
    // iteration (e.g., blockElems == SubwarpSize * kVecElems). Avoids a fixed-size local array that
    // can increase register pressure.
    if (maxIters == 1) {
      typename Bf16Vec<InVecBytes>::LoadT cached0{};
      bool hasVec = false;
      int idx = base;
      if ((idx + kVecElems - 1) < end) {
        cached0 = load<InVecBytes>(srcToken + idx);
        hasVec = true;
        localMaxBits = Bf16Vec<InVecBytes>::MaxAbsBits(cached0);
        idx += kStrideElems;
      }
      for (int j = idx; j < end; ++j) {
        const uint16_t bits = srcToken[j].data;
        const uint32_t absBits = static_cast<uint32_t>(Bf16AbsBits(bits));
        localMaxBits = (localMaxBits > absBits) ? localMaxBits : absBits;
      }

      uint32_t maxBits = SubwarpReduceMaxU32<SubwarpSize>(localMaxBits);
      maxBits = static_cast<uint32_t>(__shfl(static_cast<int>(maxBits), 0, SubwarpSize));
      maxAbs = Bf16BitsToF32(static_cast<uint16_t>(maxBits));

      const bool sbScaled = (maxAbs > fp8Max);
      subwarpScaled = subwarpScaled || sbScaled;
      // Compute the inverse scale once per subwarp and broadcast to avoid paying a divide per lane.
      float invScale = 1.0f;
      if (subLaneId == 0) {
        dstScales[sb] = sbScaled ? (maxAbs * invFp8Max) : 1.0f;
        if (sbScaled) invScale = fp8Max / maxAbs;
      }
      invScale = __shfl(invScale, 0, SubwarpSize);

      idx = base;
      if (hasVec) {
        Bf16Vec<InVecBytes>::template QuantizeStore<Fp8T>(dstBytes, idx, cached0, invScale);
        idx += kStrideElems;
      }
      for (int j = idx; j < end; ++j) {
        const float v = Bf16BitsToF32(srcToken[j].data);
        dstToken[j] = (invScale == 1.0f) ? Fp8T(v) : Fp8T(v * invScale);
      }
    } else if (maxIters <= MaxCacheIters) {
      typename Bf16Vec<InVecBytes>::LoadT cached[MaxCacheIters];
      int iters = 0;
      int idx = base;
      for (; (idx + kVecElems - 1) < end; idx += kStrideElems) {
        const auto packed = load<InVecBytes>(srcToken + idx);
        cached[iters] = packed;
        iters++;
        const uint32_t vecMax = Bf16Vec<InVecBytes>::MaxAbsBits(packed);
        localMaxBits = (localMaxBits > vecMax) ? localMaxBits : vecMax;
      }

      for (int j = idx; j < end; ++j) {
        const uint16_t bits = srcToken[j].data;
        const uint32_t absBits = static_cast<uint32_t>(Bf16AbsBits(bits));
        localMaxBits = (localMaxBits > absBits) ? localMaxBits : absBits;
      }

      uint32_t maxBits = SubwarpReduceMaxU32<SubwarpSize>(localMaxBits);
      maxBits = static_cast<uint32_t>(__shfl(static_cast<int>(maxBits), 0, SubwarpSize));
      maxAbs = Bf16BitsToF32(static_cast<uint16_t>(maxBits));

      const bool sbScaled = (maxAbs > fp8Max);
      subwarpScaled = subwarpScaled || sbScaled;
      // Compute the inverse scale once per subwarp and broadcast to avoid paying a divide per lane.
      float invScale = 1.0f;
      if (subLaneId == 0) {
        dstScales[sb] = sbScaled ? (maxAbs * invFp8Max) : 1.0f;
        if (sbScaled) invScale = fp8Max / maxAbs;
      }
      invScale = __shfl(invScale, 0, SubwarpSize);

      idx = base;
      for (int i = 0; i < iters; ++i, idx += kStrideElems) {
        Bf16Vec<InVecBytes>::template QuantizeStore<Fp8T>(dstBytes, idx, cached[i], invScale);
      }
      for (int j = idx; j < end; ++j) {
        const float v = Bf16BitsToF32(srcToken[j].data);
        dstToken[j] = (invScale == 1.0f) ? Fp8T(v) : Fp8T(v * invScale);
      }
    } else {
      int idx = base;
      for (; (idx + kVecElems - 1) < end; idx += kStrideElems) {
        const auto packed = load<InVecBytes>(srcToken + idx);
        const uint32_t vecMax = Bf16Vec<InVecBytes>::MaxAbsBits(packed);
        localMaxBits = (localMaxBits > vecMax) ? localMaxBits : vecMax;
      }
      for (int j = idx; j < end; ++j) {
        const uint16_t bits = srcToken[j].data;
        const uint32_t absBits = static_cast<uint32_t>(Bf16AbsBits(bits));
        localMaxBits = (localMaxBits > absBits) ? localMaxBits : absBits;
      }

      uint32_t maxBits = SubwarpReduceMaxU32<SubwarpSize>(localMaxBits);
      maxBits = static_cast<uint32_t>(__shfl(static_cast<int>(maxBits), 0, SubwarpSize));
      maxAbs = Bf16BitsToF32(static_cast<uint16_t>(maxBits));

      const bool sbScaled = (maxAbs > fp8Max);
      subwarpScaled = subwarpScaled || sbScaled;
      // Compute the inverse scale once per subwarp and broadcast to avoid paying a divide per lane.
      float invScale = 1.0f;
      if (subLaneId == 0) {
        dstScales[sb] = sbScaled ? (maxAbs * invFp8Max) : 1.0f;
        if (sbScaled) invScale = fp8Max / maxAbs;
      }
      invScale = __shfl(invScale, 0, SubwarpSize);

      idx = base;
      for (; (idx + kVecElems - 1) < end; idx += kStrideElems) {
        const auto packed = load<InVecBytes>(srcToken + idx);
        Bf16Vec<InVecBytes>::template QuantizeStore<Fp8T>(dstBytes, idx, packed, invScale);
      }
      for (int j = idx; j < end; ++j) {
        const float v = Bf16BitsToF32(srcToken[j].data);
        dstToken[j] = (invScale == 1.0f) ? Fp8T(v) : Fp8T(v * invScale);
      }
    }
  }

  const int anyScaled = __any(static_cast<int>(subwarpScaled));
  if (laneId == 0 && anyScaled) dstScales[0] = -dstScales[0];
}

}  // namespace detail

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

  if constexpr (std::is_same_v<InT, hip_bfloat16>) {
    const uintptr_t srcPtr = reinterpret_cast<uintptr_t>(srcToken);
    const uintptr_t dstPtr = reinterpret_cast<uintptr_t>(dstBytes);
    const bool blockAligned8 = ((blockElems & 7) == 0);
    const bool blockAligned4 = ((blockElems & 3) == 0);
    const bool srcAligned8 = ((srcPtr & 0x7) == 0);
    const bool dstAligned8 = ((dstPtr & 0x7) == 0);
    const bool srcAligned4 = ((srcPtr & 0x3) == 0);
    const bool dstAligned4 = ((dstPtr & 0x3) == 0);

    if ((warpSize == 64) && (scaleDim > 1)) {
      if (blockAligned8 && srcAligned8 && dstAligned8) {
        detail::WarpQuantizeBf16ToFp8BlockwiseVec<16, 16, 4, Fp8T>(
            dstToken, dstScales, reinterpret_cast<const hip_bfloat16*>(srcToken), hiddenDim,
            scaleDim);
        return;
      }
      if (blockAligned4 && srcAligned8 && dstAligned4) {
        detail::WarpQuantizeBf16ToFp8BlockwiseVec<32, 8, 4, Fp8T>(
            dstToken, dstScales, reinterpret_cast<const hip_bfloat16*>(srcToken), hiddenDim,
            scaleDim);
        return;
      }
      if (blockAligned2 && srcAligned4 && ((dstPtr & 0x1) == 0)) {
        detail::WarpQuantizeBf16ToFp8BlockwiseVec<32, 4, 4, Fp8T>(
            dstToken, dstScales, reinterpret_cast<const hip_bfloat16*>(srcToken), hiddenDim,
            scaleDim);
        return;
      }
    }

    if (blockAligned8 && srcAligned8 && dstAligned8) {
      detail::WarpQuantizeBf16ToFp8BlockwiseVec<warpSize, 16, 4, Fp8T>(
          dstToken, dstScales, reinterpret_cast<const hip_bfloat16*>(srcToken), hiddenDim,
          scaleDim);
      return;
    }
    if (blockAligned4 && srcAligned8 && dstAligned4) {
      detail::WarpQuantizeBf16ToFp8BlockwiseVec<warpSize, 8, 4, Fp8T>(
          dstToken, dstScales, reinterpret_cast<const hip_bfloat16*>(srcToken), hiddenDim,
          scaleDim);
      return;
    }
    if (blockAligned2 && srcAligned4 && ((dstPtr & 0x1) == 0)) {
      detail::WarpQuantizeBf16ToFp8BlockwiseVec<warpSize, 4, 4, Fp8T>(
          dstToken, dstScales, reinterpret_cast<const hip_bfloat16*>(srcToken), hiddenDim,
          scaleDim);
      return;
    }
  }

  bool tokenScaled = false;
  for (int sb = 0; sb < scaleDim; ++sb) {
    const int start = sb * blockElems;
    const int end = std::min(start + blockElems, hiddenDim);

    float localMaxAbs = 0.0f;
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
    float maxAbs = WarpReduceMaxF32(localMaxAbs);
    maxAbs = __shfl(maxAbs, 0);

    const bool sbScaled = (maxAbs > fp8Max);
    tokenScaled = tokenScaled || sbScaled;
    const float scale = sbScaled ? (maxAbs * invFp8Max) : 1.0f;
    if (laneId == 0) dstScales[sb] = scale;
    const float invScale = sbScaled ? (fp8Max / maxAbs) : 1.0f;

    if (blockAligned2) {
      int idx = start + (laneId << 1);
      for (; (idx + 1) < end; idx += (warpSize << 1)) {
        float2 v;
        if (invScale == 1.0f) {
          v.x = static_cast<float>(srcToken[idx]);
          v.y = static_cast<float>(srcToken[idx + 1]);
        } else {
          v.x = static_cast<float>(srcToken[idx]) * invScale;
          v.y = static_cast<float>(srcToken[idx + 1]) * invScale;
        }
        __hip_fp8x2_storage_t packed = CvtFloat2ToFp8x2<Fp8T>(v);
        store<2>(dstBytes + idx, static_cast<uint16_t>(packed));
      }
      if (idx < end)
        dstToken[idx] = (invScale == 1.0f) ? Fp8T(static_cast<float>(srcToken[idx]))
                                           : Fp8T(static_cast<float>(srcToken[idx]) * invScale);
    } else {
      for (int idx = start + laneId; idx < end; idx += warpSize) {
        dstToken[idx] = (invScale == 1.0f) ? Fp8T(static_cast<float>(srcToken[idx]))
                                           : Fp8T(static_cast<float>(srcToken[idx]) * invScale);
      }
    }
  }

  if (laneId == 0 && tokenScaled) dstScales[0] = -dstScales[0];
}

namespace detail {

template <int VecBytes>
struct Fp8ByteVec;

template <>
struct Fp8ByteVec<4> {
  using LoadT = uint32_t;
  static constexpr int kSegs = 1;

  template <int Seg>
  __device__ __forceinline__ static uint32_t SegU32(LoadT v) {
    static_assert(Seg == 0);
    return v;
  }
};

template <>
struct Fp8ByteVec<8> {
  using LoadT = uint64_t;
  static constexpr int kSegs = 2;

  template <int Seg>
  __device__ __forceinline__ static uint32_t SegU32(LoadT v) {
    static_assert(Seg >= 0 && Seg < kSegs);
    if constexpr (Seg == 0) return static_cast<uint32_t>(v & 0xFFFFFFFFu);
    return static_cast<uint32_t>(v >> 32);
  }
};

template <>
struct Fp8ByteVec<16> {
  using LoadT = ulong2;
  static constexpr int kSegs = 4;

  template <int Seg>
  __device__ __forceinline__ static uint32_t SegU32(LoadT v) {
    static_assert(Seg >= 0 && Seg < kSegs);
    if constexpr (Seg == 0) return static_cast<uint32_t>(v.x & 0xFFFFFFFFull);
    if constexpr (Seg == 1) return static_cast<uint32_t>(v.x >> 32);
    if constexpr (Seg == 2) return static_cast<uint32_t>(v.y & 0xFFFFFFFFull);
    return static_cast<uint32_t>(v.y >> 32);
  }
};

template <typename Fp8T, bool Scaled>
__device__ __forceinline__ void AccumFp8Packed4(float2& acc01, float2& acc23, uint32_t packed4,
                                                float scale) {
  const __hip_fp8x2_storage_t p01 =
      static_cast<__hip_fp8x2_storage_t>(static_cast<uint16_t>(packed4 & 0xFFFF));
  const __hip_fp8x2_storage_t p23 =
      static_cast<__hip_fp8x2_storage_t>(static_cast<uint16_t>(packed4 >> 16));
  const float2 v01 = CvtFp8x2ToFloat2<Fp8T>(p01);
  const float2 v23 = CvtFp8x2ToFloat2<Fp8T>(p23);
  if constexpr (Scaled) {
    acc01.x = fmaf(v01.x, scale, acc01.x);
    acc01.y = fmaf(v01.y, scale, acc01.y);
    acc23.x = fmaf(v23.x, scale, acc23.x);
    acc23.y = fmaf(v23.y, scale, acc23.y);
  } else {
    (void)scale;
    acc01.x += v01.x;
    acc01.y += v01.y;
    acc23.x += v23.x;
    acc23.y += v23.y;
  }
}

template <int SubwarpSize, int VecBytes, typename OutT, typename Fp8T, int AccumNum, bool Scaled>
__device__ __forceinline__ void WarpAccumFp8DequantVecRange(OutT* __restrict__ dstToken,
                                                            const Fp8T* const* __restrict__ srcs,
                                                            const float* __restrict__ scales,
                                                            int start, int end, int subLaneId) {
  static_assert((SubwarpSize & (SubwarpSize - 1)) == 0, "SubwarpSize must be a power of two");
  static_assert((VecBytes & 3) == 0, "VecBytes must be a multiple of 4");

  using Vec = Fp8ByteVec<VecBytes>;
  using LoadT = typename Vec::LoadT;
  constexpr int kSegs = Vec::kSegs;

  const int vecEnd = start + ((end - start) / VecBytes) * VecBytes;

  int idx = start + subLaneId * VecBytes;
  for (; idx < vecEnd; idx += (SubwarpSize * VecBytes)) {
    float2 acc01[kSegs];
    float2 acc23[kSegs];
#pragma unroll
    for (int s = 0; s < kSegs; ++s) {
      acc01[s] = float2{0.0f, 0.0f};
      acc23[s] = float2{0.0f, 0.0f};
    }

#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      const Fp8T* src = srcs[i];
      if (src == nullptr) continue;
      const auto* srcB = reinterpret_cast<const __hip_fp8_storage_t*>(src);
      const float s = Scaled ? scales[i] : 1.0f;
      const LoadT packed = load<VecBytes>(srcB + idx);
#pragma unroll
      for (int seg = 0; seg < kSegs; ++seg) {
        const uint32_t packed4 = [&]() -> uint32_t {
          if constexpr (kSegs == 1) return Vec::template SegU32<0>(packed);
          if constexpr (kSegs == 2)
            return (seg == 0) ? Vec::template SegU32<0>(packed) : Vec::template SegU32<1>(packed);
          if constexpr (kSegs == 4) {
            if (seg == 0) return Vec::template SegU32<0>(packed);
            if (seg == 1) return Vec::template SegU32<1>(packed);
            if (seg == 2) return Vec::template SegU32<2>(packed);
            return Vec::template SegU32<3>(packed);
          }
          return 0;
        }();
        AccumFp8Packed4<Fp8T, Scaled>(acc01[seg], acc23[seg], packed4, s);
      }
    }

#pragma unroll
    for (int seg = 0; seg < kSegs; ++seg) {
      const int out = idx + (seg << 2);
      StoreOutPair(dstToken, out, acc01[seg]);
      StoreOutPair(dstToken, out + 2, acc23[seg]);
    }
  }

  for (int j = vecEnd + subLaneId; j < end; j += SubwarpSize) {
    float acc = 0.0f;
#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      const auto* src = srcs[i];
      if (src == nullptr) continue;
      float v = static_cast<float>(src[j]);
      if constexpr (Scaled) v *= scales[i];
      acc += v;
    }
    dstToken[j] = OutT(acc);
  }
}

template <int SubwarpSize, int VecBytes, typename OutT, typename Fp8T, int AccumNum>
__device__ __forceinline__ void WarpAccumFp8DequantSegmentBlockwiseVec(
    OutT* __restrict__ dstToken, const Fp8T* const* __restrict__ srcs,
    const float* const* __restrict__ srcScales, int hiddenDimOffset, int hiddenDimSize,
    int blockElems, int hiddenDim) {
  static_assert((SubwarpSize & (SubwarpSize - 1)) == 0, "SubwarpSize must be a power of two");

  const int laneId = threadIdx.x & (warpSize - 1);
  const int subLaneId = laneId & (SubwarpSize - 1);
  const int subWarpId = laneId / SubwarpSize;
  constexpr int kSubwarpsPerWarp = warpSize / SubwarpSize;

  const int globalStart = hiddenDimOffset;
  const int globalEnd = hiddenDimOffset + hiddenDimSize;
  const int sbStart = globalStart / blockElems;
  const int sbEnd = (globalEnd - 1) / blockElems;
  const int sbCount = sbEnd - sbStart + 1;

  for (int sbBase = 0; sbBase < sbCount; sbBase += kSubwarpsPerWarp) {
    const int sb = sbStart + sbBase + subWarpId;
    if (sb > sbEnd) continue;

    const int blockStart = sb * blockElems;
    const int blockEnd = std::min(blockStart + blockElems, hiddenDim);
    const int segStart = (globalStart > blockStart) ? globalStart : blockStart;
    const int segEnd = (globalEnd < blockEnd) ? globalEnd : blockEnd;

    const int localStart = segStart - hiddenDimOffset;
    const int localEnd = segEnd - hiddenDimOffset;
    if (localStart >= localEnd) continue;

    float sbScales[AccumNum];
#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      float s = 1.0f;
      if (subLaneId == 0) {
        if (srcs[i] != nullptr && srcScales != nullptr && srcScales[i] != nullptr) {
          s = srcScales[i][sb];
          if (sb == 0 && s < 0.0f) s = -s;
        }
      }
      sbScales[i] = __shfl(s, 0, SubwarpSize);
    }

    WarpAccumFp8DequantVecRange<SubwarpSize, VecBytes, OutT, Fp8T, AccumNum, true>(
        dstToken, srcs, sbScales, localStart, localEnd, subLaneId);
  }
}

template <int VecBytes, typename OutT, typename Fp8T, int AccumNum>
__device__ __forceinline__ void WarpAccumFp8DequantVecRangeBlockwiseScaleWave(
    OutT* __restrict__ dstToken, const Fp8T* const* __restrict__ srcs,
    const float* const* __restrict__ srcScales, int hiddenDimOffset, int start, int end,
    int blockElems) {
  static_assert((VecBytes & 3) == 0, "VecBytes must be a multiple of 4");

  using Vec = Fp8ByteVec<VecBytes>;
  using LoadT = typename Vec::LoadT;
  constexpr int kSegs = Vec::kSegs;

  const int laneId = threadIdx.x & (warpSize - 1);

  const bool blockPow2 = (blockElems & (blockElems - 1)) == 0;
  const int blockShift = blockPow2 ? (__ffs(blockElems) - 1) : 0;

  const int vecEnd = start + ((end - start) / VecBytes) * VecBytes;
  for (int idx = start + laneId * VecBytes; idx < vecEnd; idx += (warpSize * VecBytes)) {
    const int globalIdx = hiddenDimOffset + idx;
    const int sb = blockPow2 ? (globalIdx >> blockShift) : (globalIdx / blockElems);

    float2 acc01[kSegs];
    float2 acc23[kSegs];
#pragma unroll
    for (int s = 0; s < kSegs; ++s) {
      acc01[s] = float2{0.0f, 0.0f};
      acc23[s] = float2{0.0f, 0.0f};
    }

#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      const Fp8T* src = srcs[i];
      if (src == nullptr) continue;
      const auto* srcB = reinterpret_cast<const __hip_fp8_storage_t*>(src);
      const float* scalePtr = (srcScales != nullptr) ? srcScales[i] : nullptr;

      const LoadT packed = load<VecBytes>(srcB + idx);
      float s = 1.0f;
      if (scalePtr != nullptr) {
        s = scalePtr[sb];
        if (sb == 0 && s < 0.0f) s = -s;
      }
#pragma unroll
      for (int seg = 0; seg < kSegs; ++seg) {
        const uint32_t packed4 = [&]() -> uint32_t {
          if constexpr (kSegs == 1) return Vec::template SegU32<0>(packed);
          if constexpr (kSegs == 2)
            return (seg == 0) ? Vec::template SegU32<0>(packed) : Vec::template SegU32<1>(packed);
          if constexpr (kSegs == 4) {
            if (seg == 0) return Vec::template SegU32<0>(packed);
            if (seg == 1) return Vec::template SegU32<1>(packed);
            if (seg == 2) return Vec::template SegU32<2>(packed);
            return Vec::template SegU32<3>(packed);
          }
          return 0;
        }();
        if (scalePtr != nullptr) {
          AccumFp8Packed4<Fp8T, true>(acc01[seg], acc23[seg], packed4, s);
        } else {
          AccumFp8Packed4<Fp8T, false>(acc01[seg], acc23[seg], packed4, 1.0f);
        }
      }
    }

#pragma unroll
    for (int seg = 0; seg < kSegs; ++seg) {
      const int out = idx + (seg << 2);
      StoreOutPair(dstToken, out, acc01[seg]);
      StoreOutPair(dstToken, out + 2, acc23[seg]);
    }
  }

  for (int j = vecEnd + laneId; j < end; j += warpSize) {
    const int globalIdx = hiddenDimOffset + j;
    const int sb = blockPow2 ? (globalIdx >> blockShift) : (globalIdx / blockElems);

    float acc = 0.0f;
#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      const auto* src = srcs[i];
      if (src == nullptr) continue;
      float v = static_cast<float>(src[j]);
      const float* scalePtr = (srcScales != nullptr) ? srcScales[i] : nullptr;
      if (scalePtr != nullptr) {
        float s = scalePtr[sb];
        if (sb == 0 && s < 0.0f) s = -s;
        v *= s;
      }
      acc += v;
    }
    dstToken[j] = OutT(acc);
  }
}

}  // namespace detail

template <typename OutT, typename Fp8T, int AccumNum>
__device__ __forceinline__ void WarpAccumFp8DequantFullImpl(
    OutT* __restrict__ dstToken, const Fp8T* const* __restrict__ srcs,
    const float* const* __restrict__ srcScales, int hiddenDim, int scaleDim) {
  const int laneId = threadIdx.x & (warpSize - 1);

  const Fp8T* cachedSrcs[AccumNum];
#pragma unroll AccumNum
  for (int i = 0; i < AccumNum; ++i) {
    cachedSrcs[i] = srcs[i];
  }

  bool anyScale = false;
  if (srcScales != nullptr) {
#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      anyScale |= (cachedSrcs[i] != nullptr && srcScales[i] != nullptr);
    }
  }

  const bool dstAligned4 =
      ((sizeof(OutT) != 2) || ((reinterpret_cast<uintptr_t>(dstToken) & 0x3) == 0));
  const bool useVec2 = ((hiddenDim & 1) == 0) && dstAligned4;

  bool allSrcAligned4 = true;
  bool allSrcAligned8 = true;
#pragma unroll AccumNum
  for (int i = 0; i < AccumNum; ++i) {
    const Fp8T* src = cachedSrcs[i];
    if (src == nullptr) continue;
    const uintptr_t p = reinterpret_cast<uintptr_t>(src);
    allSrcAligned4 &= ((p & 0x3) == 0);
    allSrcAligned8 &= ((p & 0x7) == 0);
  }

  if (!anyScale) {
    const bool useVec16 = ((hiddenDim & 15) == 0) && dstAligned4 && allSrcAligned8;
    const bool useVec8 = ((hiddenDim & 7) == 0) && dstAligned4 && allSrcAligned8;
    const bool useVec4 = ((hiddenDim & 3) == 0) && dstAligned4 && allSrcAligned4;

    if (useVec16) {
      detail::WarpAccumFp8DequantVecRange<warpSize, 16, OutT, Fp8T, AccumNum, false>(
          dstToken, cachedSrcs, nullptr, 0, hiddenDim, laneId);
      return;
    }
    if (useVec8) {
      detail::WarpAccumFp8DequantVecRange<warpSize, 8, OutT, Fp8T, AccumNum, false>(
          dstToken, cachedSrcs, nullptr, 0, hiddenDim, laneId);
      return;
    }
    if (useVec4) {
      detail::WarpAccumFp8DequantVecRange<warpSize, 4, OutT, Fp8T, AccumNum, false>(
          dstToken, cachedSrcs, nullptr, 0, hiddenDim, laneId);
      return;
    }

    if (useVec2) {
      int idx = laneId << 1;
      for (; (idx + 1) < hiddenDim; idx += (warpSize << 1)) {
        float2 acc2{0.0f, 0.0f};
#pragma unroll AccumNum
        for (int i = 0; i < AccumNum; ++i) {
          const Fp8T* src = cachedSrcs[i];
          if (src == nullptr) continue;
          const auto* srcBytes = reinterpret_cast<const __hip_fp8_storage_t*>(src);
          __hip_fp8x2_storage_t packed =
              static_cast<__hip_fp8x2_storage_t>(load<2>(srcBytes + idx));
          float2 v = CvtFp8x2ToFloat2<Fp8T>(packed);
          acc2.x += v.x;
          acc2.y += v.y;
        }
        StoreOutPair(dstToken, idx, acc2);
      }
      if (idx < hiddenDim) {
        float acc = 0.0f;
#pragma unroll AccumNum
        for (int i = 0; i < AccumNum; ++i) {
          if (cachedSrcs[i] == nullptr) continue;
          acc += static_cast<float>(cachedSrcs[i][idx]);
        }
        dstToken[idx] = OutT(acc);
      }
    } else {
      for (int idx = laneId; idx < hiddenDim; idx += warpSize) {
        float acc = 0.0f;
#pragma unroll AccumNum
        for (int i = 0; i < AccumNum; ++i) {
          if (cachedSrcs[i] == nullptr) continue;
          acc += static_cast<float>(cachedSrcs[i][idx]);
        }
        dstToken[idx] = OutT(acc);
      }
    }
    return;
  }

  const int blockElems = (hiddenDim + scaleDim - 1) / scaleDim;

  const bool vec16Possible = ((blockElems & 15) == 0) && dstAligned4 && allSrcAligned8;
  const bool vec8Possible = ((blockElems & 7) == 0) && dstAligned4 && allSrcAligned8;
  const bool vec4Possible = ((blockElems & 3) == 0) && dstAligned4 && allSrcAligned4;

  if (vec16Possible && (scaleDim > 1)) {
    detail::WarpAccumFp8DequantVecRangeBlockwiseScaleWave<16, OutT, Fp8T, AccumNum>(
        dstToken, cachedSrcs, srcScales, /*hiddenDimOffset=*/0, /*start=*/0,
        /*end=*/hiddenDim, blockElems);
    return;
  }

  if (vec8Possible && (scaleDim > 1)) {
    detail::WarpAccumFp8DequantVecRangeBlockwiseScaleWave<8, OutT, Fp8T, AccumNum>(
        dstToken, cachedSrcs, srcScales, /*hiddenDimOffset=*/0, /*start=*/0,
        /*end=*/hiddenDim, blockElems);
    return;
  }

  if (vec4Possible && (scaleDim > 1)) {
    detail::WarpAccumFp8DequantVecRangeBlockwiseScaleWave<4, OutT, Fp8T, AccumNum>(
        dstToken, cachedSrcs, srcScales, /*hiddenDimOffset=*/0, /*start=*/0,
        /*end=*/hiddenDim, blockElems);
    return;
  }

  // Fallback: scalar dequant for unusual alignment/shape cases.
  for (int sb = 0; sb < scaleDim; ++sb) {
    const int start = sb * blockElems;
    const int end = std::min(start + blockElems, hiddenDim);

    float sbScales[AccumNum];
#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      float s = 1.0f;
      if (laneId == 0) {
        if (cachedSrcs[i] != nullptr && srcScales != nullptr && srcScales[i] != nullptr) {
          s = srcScales[i][sb];
          if (sb == 0 && s < 0.0f) s = -s;
        }
      }
      sbScales[i] = __shfl(s, 0);
    }

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

template <typename OutT, typename Fp8T, int BlockElems>
__device__ __forceinline__ void WarpAccumFp8DequantFullBlockVec8Top8(
    OutT* __restrict__ dstToken, const Fp8T* const* __restrict__ srcs,
    const float* const* __restrict__ srcScales, int hiddenDim) {
  static_assert(BlockElems > 0 && (BlockElems & (BlockElems - 1)) == 0,
                "BlockElems must be a positive power of two");
  constexpr int AccumNum = 8;

  const int laneId = threadIdx.x & (warpSize - 1);
  const Fp8T* cachedSrcs[AccumNum];
#pragma unroll AccumNum
  for (int i = 0; i < AccumNum; ++i) {
    cachedSrcs[i] = srcs[i];
  }

  bool anyScale = false;
  if (srcScales != nullptr) {
#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      anyScale |= (cachedSrcs[i] != nullptr && srcScales[i] != nullptr);
    }
  }

  if (!anyScale) {
    detail::WarpAccumFp8DequantVecRange<warpSize, 8, OutT, Fp8T, AccumNum, false>(
        dstToken, cachedSrcs, nullptr, 0, hiddenDim, laneId);
    return;
  }

  detail::WarpAccumFp8DequantVecRangeBlockwiseScaleWave<8, OutT, Fp8T, AccumNum>(
      dstToken, cachedSrcs, srcScales, /*hiddenDimOffset=*/0, /*start=*/0, /*end=*/hiddenDim,
      /*blockElems=*/BlockElems);
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
            if (srcs[i] == nullptr) continue;
            float s = 1.0f;
            if (srcScales != nullptr && srcScales[i] != nullptr) {
              s = srcScales[i][sb];
              if (sb == 0 && s < 0.0f) s = -s;
            }
            acc += static_cast<float>(srcs[i][idx]) * s;
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
  if (hiddenDimSize <= 0) return;

  const Fp8T* cachedSrcs[AccumNum];
#pragma unroll AccumNum
  for (int i = 0; i < AccumNum; ++i) {
    cachedSrcs[i] = srcs[i];
  }

  bool anyScale = false;
  if (srcScales != nullptr) {
#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      anyScale |= (cachedSrcs[i] != nullptr && srcScales[i] != nullptr);
    }
  }

  const bool dstAligned4 = ((reinterpret_cast<uintptr_t>(dstToken) & 0x3) == 0);
  bool allSrcAligned8 = true;
  bool allSrcAligned4 = true;
#pragma unroll AccumNum
  for (int i = 0; i < AccumNum; ++i) {
    const Fp8T* src = cachedSrcs[i];
    if (src == nullptr) continue;
    const uintptr_t srcPtr = reinterpret_cast<uintptr_t>(src);
    allSrcAligned8 &= ((srcPtr & 0x7) == 0);
    allSrcAligned4 &= ((srcPtr & 0x3) == 0);
  }

  if (!anyScale) {
    if (dstAligned4 && allSrcAligned8) {
      detail::WarpAccumFp8DequantVecRange<warpSize, 16, OutT, Fp8T, AccumNum, false>(
          dstToken, cachedSrcs, nullptr, 0, hiddenDimSize, laneId);
      return;
    }
    if (dstAligned4 && allSrcAligned4) {
      detail::WarpAccumFp8DequantVecRange<warpSize, 4, OutT, Fp8T, AccumNum, false>(
          dstToken, cachedSrcs, nullptr, 0, hiddenDimSize, laneId);
      return;
    }

    for (int idx = laneId; idx < hiddenDimSize; idx += warpSize) {
      float acc = 0.0f;
#pragma unroll AccumNum
      for (int i = 0; i < AccumNum; ++i) {
        if (cachedSrcs[i] == nullptr) continue;
        acc += static_cast<float>(cachedSrcs[i][idx]);
      }
      dstToken[idx] = OutT(acc);
    }
    return;
  }

  const int blockElems = (hiddenDim + scaleDim - 1) / scaleDim;
  const bool vec16Possible = ((blockElems & 15) == 0) && dstAligned4 && allSrcAligned8;
  const bool vec8Possible = ((blockElems & 7) == 0) && dstAligned4 && allSrcAligned8;
  const bool vec4Possible = ((blockElems & 3) == 0) && dstAligned4 && allSrcAligned4;

  const bool segmentAlignedToBlock = ((hiddenDimOffset % blockElems) == 0);

  if (segmentAlignedToBlock && vec16Possible) {
    detail::WarpAccumFp8DequantVecRangeBlockwiseScaleWave<16, OutT, Fp8T, AccumNum>(
        dstToken, cachedSrcs, srcScales, hiddenDimOffset, /*start=*/0,
        /*end=*/hiddenDimSize, blockElems);
    return;
  }

  if (vec16Possible) {
    const int lanes = blockElems >> 4;
    if (lanes <= 8) {
      detail::WarpAccumFp8DequantSegmentBlockwiseVec<8, 16, OutT, Fp8T, AccumNum>(
          dstToken, cachedSrcs, srcScales, hiddenDimOffset, hiddenDimSize, blockElems, hiddenDim);
    } else if (lanes <= 16) {
      detail::WarpAccumFp8DequantSegmentBlockwiseVec<16, 16, OutT, Fp8T, AccumNum>(
          dstToken, cachedSrcs, srcScales, hiddenDimOffset, hiddenDimSize, blockElems, hiddenDim);
    } else if (lanes <= 32) {
      detail::WarpAccumFp8DequantSegmentBlockwiseVec<32, 16, OutT, Fp8T, AccumNum>(
          dstToken, cachedSrcs, srcScales, hiddenDimOffset, hiddenDimSize, blockElems, hiddenDim);
    } else {
      detail::WarpAccumFp8DequantSegmentBlockwiseVec<warpSize, 16, OutT, Fp8T, AccumNum>(
          dstToken, cachedSrcs, srcScales, hiddenDimOffset, hiddenDimSize, blockElems, hiddenDim);
    }
    return;
  }

  if (segmentAlignedToBlock && vec8Possible) {
    detail::WarpAccumFp8DequantVecRangeBlockwiseScaleWave<8, OutT, Fp8T, AccumNum>(
        dstToken, cachedSrcs, srcScales, hiddenDimOffset, /*start=*/0,
        /*end=*/hiddenDimSize, blockElems);
    return;
  }

  if (vec8Possible) {
    const int lanes = blockElems >> 3;
    if (lanes <= 8) {
      detail::WarpAccumFp8DequantSegmentBlockwiseVec<8, 8, OutT, Fp8T, AccumNum>(
          dstToken, cachedSrcs, srcScales, hiddenDimOffset, hiddenDimSize, blockElems, hiddenDim);
    } else if (lanes <= 16) {
      detail::WarpAccumFp8DequantSegmentBlockwiseVec<16, 8, OutT, Fp8T, AccumNum>(
          dstToken, cachedSrcs, srcScales, hiddenDimOffset, hiddenDimSize, blockElems, hiddenDim);
    } else if (lanes <= 32) {
      detail::WarpAccumFp8DequantSegmentBlockwiseVec<32, 8, OutT, Fp8T, AccumNum>(
          dstToken, cachedSrcs, srcScales, hiddenDimOffset, hiddenDimSize, blockElems, hiddenDim);
    } else {
      detail::WarpAccumFp8DequantSegmentBlockwiseVec<warpSize, 8, OutT, Fp8T, AccumNum>(
          dstToken, cachedSrcs, srcScales, hiddenDimOffset, hiddenDimSize, blockElems, hiddenDim);
    }
    return;
  }

  if (segmentAlignedToBlock && vec4Possible) {
    detail::WarpAccumFp8DequantVecRangeBlockwiseScaleWave<4, OutT, Fp8T, AccumNum>(
        dstToken, cachedSrcs, srcScales, hiddenDimOffset, /*start=*/0,
        /*end=*/hiddenDimSize, blockElems);
    return;
  }

  if (vec4Possible) {
    const int lanes = blockElems >> 2;
    if (lanes <= 8) {
      detail::WarpAccumFp8DequantSegmentBlockwiseVec<8, 4, OutT, Fp8T, AccumNum>(
          dstToken, cachedSrcs, srcScales, hiddenDimOffset, hiddenDimSize, blockElems, hiddenDim);
    } else if (lanes <= 16) {
      detail::WarpAccumFp8DequantSegmentBlockwiseVec<16, 4, OutT, Fp8T, AccumNum>(
          dstToken, cachedSrcs, srcScales, hiddenDimOffset, hiddenDimSize, blockElems, hiddenDim);
    } else if (lanes <= 32) {
      detail::WarpAccumFp8DequantSegmentBlockwiseVec<32, 4, OutT, Fp8T, AccumNum>(
          dstToken, cachedSrcs, srcScales, hiddenDimOffset, hiddenDimSize, blockElems, hiddenDim);
    } else {
      detail::WarpAccumFp8DequantSegmentBlockwiseVec<warpSize, 4, OutT, Fp8T, AccumNum>(
          dstToken, cachedSrcs, srcScales, hiddenDimOffset, hiddenDimSize, blockElems, hiddenDim);
    }
    return;
  }

  for (int idx = laneId; idx < hiddenDimSize; idx += warpSize) {
    const int globalIdx = hiddenDimOffset + idx;
    const int sb = globalIdx / blockElems;
    float acc = 0.0f;
#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      if (cachedSrcs[i] == nullptr) continue;
      float s = 1.0f;
      if (srcScales != nullptr && srcScales[i] != nullptr) {
        s = srcScales[i][sb];
        if (sb == 0 && s < 0.0f) s = -s;
      }
      acc += static_cast<float>(cachedSrcs[i][idx]) * s;
    }
    dstToken[idx] = OutT(acc);
  }
}

// Caller must ensure hiddenDimOffset and hiddenDimSize are 8-byte aligned.
template <typename OutT, typename Fp8T, int BlockElems>
__device__ __forceinline__ void WarpAccumFp8DequantSegmentBlockVec8Top8(
    OutT* __restrict__ dstToken, const Fp8T* const* __restrict__ srcs,
    const float* const* __restrict__ srcScales, int hiddenDimOffset, int hiddenDimSize) {
  static_assert(BlockElems > 0 && (BlockElems & (BlockElems - 1)) == 0,
                "BlockElems must be a positive power of two");
  constexpr int AccumNum = 8;

  const int laneId = threadIdx.x & (warpSize - 1);
  if (hiddenDimSize <= 0) return;

  const Fp8T* cachedSrcs[AccumNum];
#pragma unroll AccumNum
  for (int i = 0; i < AccumNum; ++i) {
    cachedSrcs[i] = srcs[i];
  }

  bool anyScale = false;
  if (srcScales != nullptr) {
#pragma unroll AccumNum
    for (int i = 0; i < AccumNum; ++i) {
      anyScale |= (cachedSrcs[i] != nullptr && srcScales[i] != nullptr);
    }
  }

  if (!anyScale) {
    detail::WarpAccumFp8DequantVecRange<warpSize, 8, OutT, Fp8T, AccumNum, false>(
        dstToken, cachedSrcs, nullptr, 0, hiddenDimSize, laneId);
    return;
  }

  detail::WarpAccumFp8DequantVecRangeBlockwiseScaleWave<8, OutT, Fp8T, AccumNum>(
      dstToken, cachedSrcs, srcScales, hiddenDimOffset, /*start=*/0, /*end=*/hiddenDimSize,
      /*blockElems=*/BlockElems);
}

// Scalar fallback for the rare misaligned-segment case in the AccumNum=8 vec8 path. Kept tiny
// on purpose: delegating to WarpAccumFp8DequantSegmentImpl<...,8> would drag the generic
// vec16/vec8/vec4 dispatch tree into the caller and push num_vgpr from 79 back to 128.
template <typename OutT, typename Fp8T, int BlockElems>
__device__ __forceinline__ void WarpAccumFp8DequantSegmentScalarTop8(
    OutT* __restrict__ dstToken, const Fp8T* const* __restrict__ srcs,
    const float* const* __restrict__ srcScales, int hiddenDimOffset, int hiddenDimSize) {
  static_assert(BlockElems > 0 && (BlockElems & (BlockElems - 1)) == 0,
                "BlockElems must be a positive power of two");
  constexpr int AccumNum = 8;
  constexpr int kBlockShift = __builtin_ctz(BlockElems);

  const int laneId = threadIdx.x & (warpSize - 1);
  for (int j = laneId; j < hiddenDimSize; j += warpSize) {
    const int globalIdx = hiddenDimOffset + j;
    const int sb = globalIdx >> kBlockShift;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < AccumNum; ++i) {
      const Fp8T* src = srcs[i];
      if (src == nullptr) continue;
      float v = static_cast<float>(src[j]);
      const float* scalePtr = srcScales[i];
      if (scalePtr != nullptr) {
        float s = scalePtr[sb];
        if (sb == 0 && s < 0.0f) s = -s;
        v *= s;
      }
      acc += v;
    }
    dstToken[j] = OutT(acc);
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
          if (srcs[i] == nullptr) continue;
          float s = 1.0f;
          if (srcScales != nullptr && srcScales[i] != nullptr) {
            s = srcScales[i][sb];
            if (sb == 0 && s < 0.0f) s = -s;
          }
          acc += static_cast<float>(srcs[i][idx]) * s;
        }
        dstToken[idx] = OutT(acc);
      }
      break;
    }
  }
}
#endif

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
  static_assert(!sizeof(T*),
                "WarpCastBf16ToCombineInternalFp8 requires FP8 type support "
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
  static_assert(!sizeof(T*),
                "WarpAccumCombineInternalFp8ToBf16 requires FP8 type support "
                "(MORI_FP8_TYPE_OCP_ENABLED or MORI_FP8_TYPE_FNUZ_ENABLED)");
#endif
}

}  // namespace core
}  // namespace mori
