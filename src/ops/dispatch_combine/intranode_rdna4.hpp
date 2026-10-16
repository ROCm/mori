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
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include "src/ops/dispatch_combine/intranode.hpp"

namespace mori {
namespace moe {

// Dispatch keeps one routing slot per destination rank. Search every expert
// slot, in wave32-sized batches, so top-k may exceed the hardware wave width.
// Every lane must call this helper with the same row and destination rank.
template <int WorldSize>
inline __device__ index_t Rdna4FindRankToken(const index_t* slots, int topk, index_t stride,
                                             int rank) {
  const int lane = threadIdx.x & (warpSize - 1);
  const index_t invalid = WorldSize * stride;
  for (int base = 0; base < topk; base += warpSize) {
    const int k = base + lane;
    const index_t flat = k < topk ? slots[k] : invalid;
    const uint32_t mask = uint32_t(__ballot(flat >= rank * stride && flat < (rank + 1) * stride));
    if (mask) return __shfl(flat, __ffs(mask) - 1);
  }
  return invalid;
}

}  // namespace moe
}  // namespace mori
