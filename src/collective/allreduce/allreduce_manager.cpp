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
#include "mori/collective/all_reduce/allreduce_manager.hpp"

namespace mori {
namespace collective {

int AllReduceManager::Initialize(const AllReduceConfig& config) {
  if (initialized) {
    return 0;  // Already initialized
  }

  this->config = config;
  currentAlgorithm = AllReduceAlgorithm::INTER_RING_1D;

  // Initialize the static TopologyDetector
  TopologyDetector::Initialize();

  // Create executor using topology information from TopologyDetector
  executor = std::make_unique<Ring1DAllReduceExecutor>(TopologyDetector::GetMyPe(),
                                                       TopologyDetector::GetNPes(), config);

  initialized = true;
  return 0;
}

void AllReduceManager::Finalize() {
  if (!initialized) {
    return;
  }

  executor.reset();

  // Note: TopologyDetector::Finalize() is called separately
  // since it's a global singleton that might be shared across multiple managers
  TopologyDetector::Finalize();

  initialized = false;
}

int AllReduceManager::AllReduce(void* input, void* output, size_t count, size_t dtype_size,
                                hipStream_t stream) {
  bool needToCreateStream = false;
  if (stream == nullptr) {
    hipStream_t newStream;
    HIP_RUNTIME_CHECK(hipStreamCreate(&newStream));
    stream = newStream;
    needToCreateStream = true;
  }
  int status = executor->Execute(input, output, count, dtype_size, stream);
  if (status != 0) {
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
