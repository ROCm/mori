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

#include <hip/hip_runtime.h>

#include "mori/collective/topology_detector.hpp"

namespace mori {
namespace collective {

/**
 * Base class for All-Reduce algorithm executors
 * Provides common interface for all algorithm implementations
 */
class AllReduceExecutor {
 public:
  virtual ~AllReduceExecutor() = default;

  /**
   * Execute All-Reduce operation
   *
   * @param input Input data pointer (device memory)
   * @param output Output data pointer (device memory, can be same as input)
   * @param count Number of elements
   * @param dtype_size Size of each element in bytes
   * @param stream HIP stream for asynchronous execution
   * @return 0 on success, error code otherwise
   */
  virtual int Execute(void* input, void* output, size_t count, size_t dtype_size,
                      hipStream_t stream) = 0;
};

}  // namespace collective
}  // namespace mori
