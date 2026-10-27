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

#include "ep_config.hpp"

namespace mori::kiwi::ep {

// Inclusive prefix sum across one EP wave.  Lane i receives the sum of input
// values from lanes [0, i].  Callers can mask inactive lanes by passing zero.
__device__ inline int ep_wave_inclusive_prefix_sum(int value) {
  for (int offset = 1; offset < EP_WAVE_SIZE; offset <<= 1) {
    const int peer = __shfl_up(value, offset);
    const int lane = threadIdx.x % EP_WAVE_SIZE;
    if (lane >= offset) {
      value += peer;
    }
  }
  return value;
}

}  // namespace mori::kiwi::ep
