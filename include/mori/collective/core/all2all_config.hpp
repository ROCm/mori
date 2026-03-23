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
 * Enumeration of available All-Reduce algorithms
 */
enum class All2allAlgorithm {
  INVALID,
  INTRA_1_STAGE,  // Intra-node 1 stage
  INTRA_2_STAGE,  // Intra-node 2 stage
  INTER_RING_1D,  // Inter-node: Simple 1D Ring (Reduce-Scatter + AllGather)
  INTER_RING_2D,  // Inter-node: 2D Ring (hierarchical)
  INTER_ONE_SHOT  // Inter-node: One-shot
};

/**
 * Configuration parameters for All-Reduce framework
 */
struct All2allConfig {
  // Threshold for switching between One-shot and 1D Ring
  // If data size < threshold, use One-shot; otherwise use 1D Ring
  size_t ringThresholdBytes = 1 * 1024 * 1024;
  // If rank size < threshold, use One-shot; otherwise use 1D Ring
  int ringThresholdRanks = 16;

  // Maximum number of blocks for kernel launch
  int maxBlocks = 80;

  // Threads per block
  int threadsPerBlock = 512;

  All2allAlgorithm algorithm = All2allAlgorithm::INVALID;

  size_t dataSizeBytes = 0;
};

}  // namespace collective
}  // namespace mori
