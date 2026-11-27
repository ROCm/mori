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
// DEVICE ONLY. The EP local kernels: they read the rows dispatch delivered to this
// rank, [0, totalRecv), and touch no peer. Semantics follow FlyDSL's
// make_local_expert_count / make_convert_dispatch_output / make_convert_combine_input.

#pragma once

#include <hip/hip_runtime.h>

#include "mori/core/transport/p2p/device_primitives.hpp"
#include "mori/ops/dispatch_combine_v2/ep_local_cfg.hpp"

namespace mori {
namespace ops {
namespace v2 {

// Rows this pass covers; the clamp keeps a bad count inside the arena rows.
template <EpLocalCfg kCfg>
__device__ __forceinline__ int EpLocalRows(const EpLocalArgs& a) {
  return min(*a.totalRecv, EpLocalMaxRecv(kCfg));
}

// expertCount[e] += #(row, k) whose id is this rank's expert e. Out-of-range and
// negative ids are skipped. One thread per (row, k), counted in a per-block LDS
// histogram so each block adds once per expert instead of once per id.
template <EpLocalCfg kCfg>
__device__ void EpLocalExpertCountBody(EpLocalArgs a) {
  constexpr unsigned kEpr = kCfg.numExpertPerRank;
  __shared__ int hist[kEpr];
  for (int e = threadIdx.x; e < static_cast<int>(kEpr); e += blockDim.x) hist[e] = 0;
  __syncthreads();
  const int n = EpLocalRows<kCfg>(a) * kCfg.numExpertPerToken;
  const int base = a.rank * kCfg.numExpertPerRank;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const unsigned local = static_cast<unsigned>(a.outIdx[i]) - static_cast<unsigned>(base);
    if (local < kEpr) atomicAdd(hist + local, 1);
  }
  __syncthreads();
  for (int e = threadIdx.x; e < static_cast<int>(kEpr); e += blockDim.x) {
    if (hist[e]) atomicAdd(a.expertCount + e, hist[e]);
  }
}

// StdMoE pack: each local (row, k) claims a row of its expert's bucket, records it
// in slotMap and the row's source in packedSrc, and copies the token there.
// One warp per (row, k); expertCount must start at zero.
template <EpLocalCfg kCfg, typename T>
__device__ void EpConvertDispatchBody(EpLocalArgs a) {
  constexpr int kTopk = kCfg.numExpertPerToken;
  constexpr unsigned kEpr = kCfg.numExpertPerRank;
  constexpr int kBucket = EpLocalBucket(kCfg);
  constexpr size_t kHidden = static_cast<size_t>(kCfg.hiddenDim);

  const int laneId = threadIdx.x & (kCfg.waveSize - 1);
  const int warpId = threadIdx.x / kCfg.waveSize;
  const int globalWarpId = blockIdx.x * kCfg.warpPerBlock + warpId;
  const int globalWarpNum = gridDim.x * kCfg.warpPerBlock;
  const int n = EpLocalRows<kCfg>(a) * kTopk;
  const int base = a.rank * kCfg.numExpertPerRank;
  const T* dispOut = reinterpret_cast<const T*>(a.dispOut);
  T* packedX = reinterpret_cast<T*>(a.packedX);

  for (int i = globalWarpId; i < n; i += globalWarpNum) {
    const unsigned local = static_cast<unsigned>(a.outIdx[i]) - static_cast<unsigned>(base);
    int slot = kBucket;
    if (local < kEpr && laneId == 0) {
      slot = atomicAdd(a.expertCount + local, 1);
      // A repeated id in one row can overrun the bucket: drop it, keep the count.
      if (slot >= kBucket) atomicSub(a.expertCount + local, 1);
    }
    slot = __shfl(slot, 0);
    if (slot >= kBucket) {
      if (laneId == 0) a.slotMap[i] = -1;
      continue;
    }
    const long long row = static_cast<long long>(local) * kBucket + slot;
    const int tok = i / kTopk;
    if (laneId == 0) {
      a.slotMap[i] = row;
      a.packedSrc[row] = a.recvToSrc[tok];
    }
    core::WarpCopy(packedX + row * kHidden, dispOut + tok * kHidden, kHidden);
  }
}

// StdMoE unpack: outTok[row] = sum_k outWts[row, k] * packedX[slotMap[row, k]] over
// the row's local slots, fp32 accumulate. Rows split across warps in 16 B chunks.
template <EpLocalCfg kCfg, typename T>
__device__ void EpConvertCombineBody(EpLocalArgs a) {
  constexpr int kTopk = kCfg.numExpertPerToken;
  constexpr int kVec = 16 / sizeof(T);
  constexpr int kUnits = kCfg.hiddenDim / kVec;
  constexpr size_t kHidden = static_cast<size_t>(kCfg.hiddenDim);

  const int laneId = threadIdx.x & (kCfg.waveSize - 1);
  const int warpId = threadIdx.x / kCfg.waveSize;
  const int globalWarpId = blockIdx.x * kCfg.warpPerBlock + warpId;
  const int globalWarpNum = gridDim.x * kCfg.warpPerBlock;
  const int rows = EpLocalRows<kCfg>(a);
  if (rows <= 0) return;

  // Per-warp source pointers; WarpAccum takes them by address.
  __shared__ T* srcPtrsAll[kCfg.warpPerBlock * kTopk];
  T** srcPtrs = srcPtrsAll + warpId * kTopk;
  T* packedX = reinterpret_cast<T*>(a.packedX);
  T* outTok = reinterpret_cast<T*>(a.outTok);

  const int warpsPerRow = max(1, (globalWarpNum + rows - 1) / rows);
  const int unitsPerWarp = (kUnits + warpsPerRow - 1) / warpsPerRow;
  for (int i = globalWarpId; i < rows * warpsPerRow; i += globalWarpNum) {
    const int tok = i / warpsPerRow;
    const int u0 = (i % warpsPerRow) * unitsPerWarp;
    if (u0 >= kUnits) continue;
    const size_t off = static_cast<size_t>(u0) * kVec;
    const size_t len = static_cast<size_t>(min(unitsPerWarp, kUnits - u0)) * kVec;
    if (laneId < kTopk) {
      const long long s = a.slotMap[tok * kTopk + laneId];
      srcPtrs[laneId] = s >= 0 ? packedX + s * kHidden + off : nullptr;
    }
    // This collapses to WarpAccumImpl<T, 16, kTopk, 1> after switch
    core::WarpAccumLF<T, 16, 1>(outTok + tok * kHidden + off, srcPtrs, a.outWts + tok * kTopk,
                                kTopk, len);
  }
}

}  // namespace v2
}  // namespace ops
}  // namespace mori
