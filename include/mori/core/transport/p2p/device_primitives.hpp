#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>

namespace mori {
namespace core {

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

template <typename T>
inline __device__ void ThreadCopyAtomic(T* dst, T* src, size_t nelems) {
  int offset = 0;

  while (offset < nelems) {
    T val = AtomicLoadRelaxedSystem(src + offset);
    AtomicStoreRelaxedSystem(dst + offset, val);
    offset += 1;
  }
}

template <typename T>
inline __device__ void WarpCopy(T* dst, T* src, size_t nelems) {
  constexpr int vecSize = 16 / sizeof(T);
  int laneId = threadIdx.x & (warpSize - 1);
  int offset = laneId * vecSize;

  while ((offset + vecSize) <= nelems) {
    reinterpret_cast<uint4*>(dst + offset)[0] = reinterpret_cast<uint4*>(src + offset)[0];
    offset += warpSize * vecSize;
  }

  while (offset < nelems) {
    dst[offset] = src[offset];
    offset += 1;
  }
}

template <typename T>
inline __device__ void WarpCopyAtomic(T* dst, T* src, size_t nelems) {
  int laneId = threadIdx.x & (warpSize - 1);
  int offset = laneId;

  while (offset < nelems) {
    T val = AtomicLoadRelaxedSystem(src + offset);
    AtomicStoreRelaxedSystem(dst + offset, val);
    offset += warpSize;
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

// Accumulate multiple buffers
template <typename T>
inline __device__ void WarpAccum(T* dest, T** srcs, float* srcScales, size_t accumNum,
                                 size_t nelems) {
  constexpr int vecSize = 16 / sizeof(T);
  int laneId = threadIdx.x & (warpSize - 1);
  int offset = laneId * vecSize;

  while ((offset + vecSize) <= nelems) {
    float accumValFp32[vecSize] = {0};

#pragma unroll
    for (int i = 0; i < accumNum; i++) {
      T* srcPtr = srcs[i];
      if (srcPtr == nullptr) continue;

      uint4 srcVals = reinterpret_cast<uint4*>(srcPtr + offset)[0];
      float srcScale = (srcScales == nullptr) ? 1.0f : srcScales[i];

      if constexpr (vecSize > 0)
        accumValFp32[0] += float(reinterpret_cast<T*>(&srcVals)[0]) * srcScale;
      if constexpr (vecSize > 1)
        accumValFp32[1] += float(reinterpret_cast<T*>(&srcVals)[1]) * srcScale;
      if constexpr (vecSize > 2)
        accumValFp32[2] += float(reinterpret_cast<T*>(&srcVals)[2]) * srcScale;
      if constexpr (vecSize > 3)
        accumValFp32[3] += float(reinterpret_cast<T*>(&srcVals)[3]) * srcScale;
      if constexpr (vecSize > 4)
        accumValFp32[4] += float(reinterpret_cast<T*>(&srcVals)[4]) * srcScale;
      if constexpr (vecSize > 5)
        accumValFp32[5] += float(reinterpret_cast<T*>(&srcVals)[5]) * srcScale;
      if constexpr (vecSize > 6)
        accumValFp32[6] += float(reinterpret_cast<T*>(&srcVals)[6]) * srcScale;
      if constexpr (vecSize > 7)
        accumValFp32[7] += float(reinterpret_cast<T*>(&srcVals)[7]) * srcScale;
      if constexpr (vecSize > 8)
        accumValFp32[8] += float(reinterpret_cast<T*>(&srcVals)[8]) * srcScale;
      if constexpr (vecSize > 9)
        accumValFp32[9] += float(reinterpret_cast<T*>(&srcVals)[9]) * srcScale;
      if constexpr (vecSize > 10)
        accumValFp32[10] += float(reinterpret_cast<T*>(&srcVals)[10]) * srcScale;
      if constexpr (vecSize > 11)
        accumValFp32[11] += float(reinterpret_cast<T*>(&srcVals)[11]) * srcScale;
      if constexpr (vecSize > 12)
        accumValFp32[12] += float(reinterpret_cast<T*>(&srcVals)[12]) * srcScale;
      if constexpr (vecSize > 13)
        accumValFp32[13] += float(reinterpret_cast<T*>(&srcVals)[13]) * srcScale;
      if constexpr (vecSize > 14)
        accumValFp32[14] += float(reinterpret_cast<T*>(&srcVals)[14]) * srcScale;
      if constexpr (vecSize > 15)
        accumValFp32[15] += float(reinterpret_cast<T*>(&srcVals)[15]) * srcScale;
    }

    uint4 accumVals;
    if constexpr (vecSize > 0) reinterpret_cast<T*>(&accumVals)[0] = T(accumValFp32[0]);
    if constexpr (vecSize > 1) reinterpret_cast<T*>(&accumVals)[1] = T(accumValFp32[1]);
    if constexpr (vecSize > 2) reinterpret_cast<T*>(&accumVals)[2] = T(accumValFp32[2]);
    if constexpr (vecSize > 3) reinterpret_cast<T*>(&accumVals)[3] = T(accumValFp32[3]);
    if constexpr (vecSize > 4) reinterpret_cast<T*>(&accumVals)[4] = T(accumValFp32[4]);
    if constexpr (vecSize > 5) reinterpret_cast<T*>(&accumVals)[5] = T(accumValFp32[5]);
    if constexpr (vecSize > 6) reinterpret_cast<T*>(&accumVals)[6] = T(accumValFp32[6]);
    if constexpr (vecSize > 7) reinterpret_cast<T*>(&accumVals)[7] = T(accumValFp32[7]);
    if constexpr (vecSize > 8) reinterpret_cast<T*>(&accumVals)[8] = T(accumValFp32[8]);
    if constexpr (vecSize > 9) reinterpret_cast<T*>(&accumVals)[9] = T(accumValFp32[9]);
    if constexpr (vecSize > 10) reinterpret_cast<T*>(&accumVals)[10] = T(accumValFp32[10]);
    if constexpr (vecSize > 11) reinterpret_cast<T*>(&accumVals)[11] = T(accumValFp32[11]);
    if constexpr (vecSize > 12) reinterpret_cast<T*>(&accumVals)[12] = T(accumValFp32[12]);
    if constexpr (vecSize > 13) reinterpret_cast<T*>(&accumVals)[13] = T(accumValFp32[13]);
    if constexpr (vecSize > 14) reinterpret_cast<T*>(&accumVals)[14] = T(accumValFp32[14]);
    if constexpr (vecSize > 15) reinterpret_cast<T*>(&accumVals)[15] = T(accumValFp32[15]);

    reinterpret_cast<uint4*>(dest + offset)[0] = accumVals;

    offset += warpSize * vecSize;
  }

  while (offset < nelems) {
    float accumValFp32 = 0;

#pragma unroll
    for (int i = 0; i < accumNum; i++) {
      T* srcPtr = srcs[i];
      float srcScale = (srcScales == nullptr) ? 1.0f : srcScales[i];

      if (srcPtr == nullptr) continue;

      accumValFp32 += float(srcPtr[offset]) * srcScale;
    }
    dest[offset] = T(accumValFp32);
    offset += 1;
  }
}

}  // namespace core
}  // namespace mori