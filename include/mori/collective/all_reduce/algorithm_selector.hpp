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

#include "mori/collective/all_reduce/allreduce_config.hpp"

namespace mori {
namespace collective {

/**
 * Enumeration of available All-Reduce algorithms
 */
enum class AllReduceAlgorithm {
  INVALID,
  INTRA_1_STAGE,  // Intra-node 1 stage
  INTRA_2_STAGE,  // Intra-node 2 stage
  INTER_RING_1D,  // Inter-node: Simple 1D Ring (Reduce-Scatter + AllGather)
  INTER_RING_2D   // Inter-node: 2D Ring (hierarchical)
};

/**
 * AlgorithmSelector: Dynamically selects optimal All-Reduce algorithm
 *
 * Selection criteria:
 * - Topology (intra-node vs inter-node)
 * - Data size (small -> 1D Ring, large -> 2D Ring)
 * - Number of ranks
 * - Network characteristics (if available)
 */
class AlgorithmSelector {
 public:
  /**
   * Select the optimal algorithm for given parameters
   *
   * @param data_size_bytes Total data size in bytes
   * @param num_ranks Number of participating ranks
   * @param is_intra_node Whether ranks are on same node
   * @param config Configuration parameters
   * @return Selected algorithm type
   */
  static AllReduceAlgorithm Select(size_t data_size_bytes, int num_ranks, bool is_intra_node,
                                   const AllReduceConfig& config = AllReduceConfig());
};

}  // namespace collective
}  // namespace mori
