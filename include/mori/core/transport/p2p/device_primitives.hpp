#pragma once

namespace mori {
namespace core {

template <typename T>
__device__ void ThreadCopy(T* dst, T* src, size_t nelems) {
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
__device__ void ThreadCopyAtomic(T* dst, T* src, size_t nelems) {
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
__device__ void WarpCopyAtomic(T* dst, T* src, size_t nelems) {
  int laneId = threadIdx.x & (warpSize - 1);
  int offset = laneId;

  while (offset < nelems) {
    T val = AtomicLoadRelaxedSystem(src + offset);
    AtomicStoreRelaxedSystem(dst + offset, val);
    offset += warpSize;
  }
}

template <typename T>
__device__ T WarpReduceSum(T val) {
  int laneId = threadIdx.x & (warpSize - 1);
  for (int delta = (warpSize >> 1); delta > 0; delta = (delta >> 1)) {
    val += __shfl_down(val, delta);
  }
  return val;
}

template <typename T>
__device__ T WarpPrefixSum(T val, size_t laneNum) {
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
__device__ T BlockPrefixSum(T val, size_t thdNum) {
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
__device__ void WarpAccum(T* accum, T* src, size_t nelems) {
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
__device__ void WarpAccum(T* dest, T** srcs, float* srcScales, size_t accumNum, size_t nelems) {
  constexpr int vecSize = 16 / sizeof(T);
  int laneId = threadIdx.x & (warpSize - 1);
  int offset = laneId * vecSize;

  while ((offset + vecSize) <= nelems) {
    float accumValFp32[vecSize] = {0};

#pragma unroll
    for (int i = 0; i < accumNum; i++) {
      uint4 srcVals = reinterpret_cast<uint4*>(srcs[i] + offset)[0];
      float srcScale = srcScales[i];

#pragma unroll
      for (int j = 0; j < vecSize; j++) {
        float srcVal = float(reinterpret_cast<T*>(&srcVals)[j]);
        accumValFp32[j] += srcVal * srcScale;
      }
    }

    uint4 accumVals;
#pragma unroll
    for (int i = 0; i < vecSize; i++) {
      float accumVal = accumValFp32[i];
      reinterpret_cast<T*>(&accumVals)[i] = T(accumVal);
    }
    reinterpret_cast<uint4*>(dest + offset)[0] = accumVals;

    offset += warpSize * vecSize;
  }

  while (offset < nelems) {
    assert(false);
    float accumValFp32 = 0;
#pragma unroll
    for (int i = 0; i < accumNum; i++) {
      dest[offset] = T(float(srcs[i][offset]) * srcScales[i]);
    }
    offset += 1;
  }
}

}  // namespace core
}  // namespace mori