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

#include "mori/application/utils/check.hpp"
#include "mori/collective/all_reduce/allreduce_config.hpp"
#include "mori/collective/all_reduce/allreduce_executor.hpp"
#include "mori/shmem/shmem.hpp"

namespace mori {
namespace collective {

/**
 * Ring1DAllReduceExecutor: Simple 1D Ring All-Reduce for inter-node
 *
 * Implements classic Ring AllReduce:
 * Phase 1: Reduce-Scatter (N-1 rounds)
 * Phase 2: AllGather (N-1 rounds)
 *
 * Optimal for small to medium data sizes
 */
class Ring1DAllReduceExecutor : public AllReduceExecutor {
 public:
  /**
   * Initialize 1D Ring executor
   *
   * @param num_ranks Total number of ranks
   * @param rank Current rank
   * @param config Configuration parameters
   */
  Ring1DAllReduceExecutor(int num_ranks, int rank,
                          const AllReduceConfig& config = AllReduceConfig());

  ~Ring1DAllReduceExecutor();

  int Execute(void* input, void* output, size_t count, size_t dtype_size, hipStream_t stream);

 private:
  int numRanks;
  int rank;
  AllReduceConfig config;
  bool initialized;

  // Phase 1: Reduce-Scatter
  int ReduceScatter(void* input, void* output_chunk, size_t total_count, size_t dtype_size,
                    hipStream_t stream);

  // Phase 2: AllGather
  int AllGather(void* input_chunk, void* output, size_t total_count, size_t dtype_size,
                hipStream_t stream);
};

}  // namespace collective
}  // namespace mori
