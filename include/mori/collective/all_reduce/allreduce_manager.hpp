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
#include "mori/collective/all_reduce/algorithm_selector.hpp"
#include "mori/collective/all_reduce/allreduce_config.hpp"
#include "mori/collective/all_reduce/allreduce_executor.hpp"
#include "mori/collective/all_reduce/one_shot_executor.hpp"
#include "mori/collective/all_reduce/ring_1d_executor.hpp"
#include "mori/collective/topology_detector.hpp"

namespace mori {
namespace collective {

/**
 * AllReduceManager: Main interface for unified All-Reduce operations
 *
 * Automatically handles:
 * - Topology detection
 * - Algorithm selection
 * - Executor creation and management
 * - Buffer registration
 */
template <typename T>
class AllReduceManager {
 public:
  /**
   * Initialize the unified All-Reduce manager
   *
   * @param config Configuration parameters
   * @return 0 on success, error code otherwise
   */
  int Initialize(const AllReduceConfig& config = AllReduceConfig());

  /**
   * Main All-Reduce function - unified API
   *
   * This is the primary interface users should call.
   * The framework automatically:
   * 1. Detects topology (intra-node vs inter-node)
   * 2. Selects optimal algorithm
   * 3. Executes the operation
   *
   * @param input Input data pointer (device memory)
   * @param output Output data pointer (device memory, can be same as input)
   * @param count Number of elements
   * @param dtype_size Size of each element in bytes
   * @param stream HIP stream for asynchronous execution
   * @return 0 on success, error code otherwise
   */
  int AllReduce(T* input, T* output, size_t count, hipStream_t stream = nullptr);

  void Finalize();

 private:
  AllReduceConfig config{};
  bool initialized = false;

  std::unique_ptr<AllReduceExecutor<T>> executor;

  // Current algorithm selection
  AllReduceAlgorithm currentAlgorithm = AllReduceAlgorithm::INVALID;
};

template <typename T>
int AllReduceManager<T>::Initialize(const AllReduceConfig& config) {
  if (initialized) {
    return 0;  // Already initialized
  }

  this->config = config;
  currentAlgorithm = config.algorithm;

  // Initialize the static TopologyDetector
  TopologyDetector::Initialize();

  AllReduceAlgorithm algorithm = AlgorithmSelector::Select(
      config.dataSizeBytes, TopologyDetector::GetNPes(), TopologyDetector::IsIntraNode(), config);
  if (algorithm == AllReduceAlgorithm::INVALID) {
    return -1;
  }
  currentAlgorithm = algorithm;
  if (currentAlgorithm == AllReduceAlgorithm::INTER_ONE_SHOT) {
    executor = std::make_unique<OneShotAllReduceExecutor<T>>(TopologyDetector::GetNPes(),
                                                             TopologyDetector::GetMyPe(), config);
  } else if (currentAlgorithm == AllReduceAlgorithm::INTER_RING_1D) {
    executor = std::make_unique<Ring1DAllReduceExecutor<T>>(TopologyDetector::GetNPes(),
                                                            TopologyDetector::GetMyPe(), config);
  } else {
    return -1;
  }

  initialized = true;
  return 0;
}

template <typename T>
void AllReduceManager<T>::Finalize() {
  if (!initialized) {
    return;
  }

  executor.reset();

  // Note: TopologyDetector::Finalize() is called separately
  // since it's a global singleton that might be shared across multiple managers
  TopologyDetector::Finalize();

  initialized = false;
}

template <typename T>
int AllReduceManager<T>::AllReduce(T* input, T* output, size_t count, hipStream_t stream) {
  bool needToCreateStream = false;
  if (stream == nullptr) {
    hipStream_t newStream;
    HIP_RUNTIME_CHECK(hipStreamCreate(&newStream));
    stream = newStream;
    needToCreateStream = true;
  }
  int status = executor->Execute(input, output, count, stream);
  if (status != 0) {
    if (needToCreateStream) {
      HIP_RUNTIME_CHECK(hipStreamDestroy(stream));
    }
    return status;
  }
  HIP_RUNTIME_CHECK(hipStreamSynchronize(stream));
  if (needToCreateStream) {
    HIP_RUNTIME_CHECK(hipStreamDestroy(stream));
  }
  return status;
}

}  // namespace collective
}  // namespace mori
