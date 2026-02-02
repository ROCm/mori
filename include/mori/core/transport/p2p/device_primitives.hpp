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
  // const T* cached_srcs[AccumNum];
#pragma unroll
  for (int i = 0; i < AccumNum; ++i) {
    scales[i] = (srcScales == nullptr) ? 1.0f : srcScales[i];
    // cached_srcs[i] = srcs[i];
  }

  for (size_t iter = 0; iter < numIters; ++iter) {
    float accumValFp32[vecSize] = {0};

    DataType srcVals[AccumNum];
#pragma unroll
    for (int i = 0; i < AccumNum; ++i) {
      if (srcs[i] != nullptr) srcVals[i] = load<VecBytes>(srcs[i] + offset + laneOffset);
    }

#pragma unroll
    for (int i = 0; i < AccumNum; ++i) {
      if (srcs[i] != nullptr) {
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

}  // namespace core
}  // namespace mori
