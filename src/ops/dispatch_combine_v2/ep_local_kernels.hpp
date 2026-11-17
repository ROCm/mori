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
// make_local_expert_count.

#pragma once

#include <hip/hip_runtime.h>

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
    const unsigned local = static_cast<unsigned>(a.outIdx[i] - base);
    if (local < kEpr) atomicAdd(hist + local, 1);
  }
  __syncthreads();
  for (int e = threadIdx.x; e < static_cast<int>(kEpr); e += blockDim.x) {
    if (hist[e]) atomicAdd(a.expertCount + e, hist[e]);
  }
}

}  // namespace v2
}  // namespace ops
}  // namespace mori
