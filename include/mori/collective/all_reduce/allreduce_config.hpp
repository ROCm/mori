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

namespace mori {
namespace collective {

/**
 * Configuration parameters for All-Reduce framework
 */
struct AllReduceConfig {
  // Threshold for switching between 1D and 2D Ring
  // If data size < threshold, use 1D Ring; otherwise use 2D Ring
  size_t ring2dThresholdBytes = 1 * 1024 * 1024;
  // If rank size < threshold, use 1D Ring; otherwise use 2D Ring
  int ring2dThresholdRanks = 16;

  // Maximum number of blocks for kernel launch
  int maxBlocks = 80;

  // Threads per block
  int threadsPerBlock = 512;
};

}  // namespace collective
}  // namespace mori
