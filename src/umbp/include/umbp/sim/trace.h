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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "umbp/common/config.h"

namespace mori::umbp {

class SimTrace {
 public:
  explicit SimTrace(const UMBPSimConfig& config);
  void Write(uint64_t id, const char* api, size_t keys, uint64_t bytes, uint64_t start_ns,
             uint64_t elapsed_ns, int64_t result_count, bool exception);
  std::atomic<uint64_t> next_id{0};

 private:
  std::string profile_json_;
  std::ofstream output_;
  std::mutex mutex_;
};

// Inert outside sim mode. Logs after taking the end timestamp; never changes API outcomes.
class SimCall {
 public:
  template <typename Sizes>
  SimCall(SimTrace* trace, const char* api, size_t keys, const Sizes& sizes)
      : trace_(trace), api_(api), keys_(keys) {
    if (trace_) {
      start_ = std::chrono::steady_clock::now();
      id_ = trace_->next_id.fetch_add(1, std::memory_order_relaxed);
      exceptions_ = std::uncaught_exceptions();
      bytes_ = Bytes(sizes);
    }
  }
  ~SimCall();
  bool Result(bool result) {
    result_count_ = result;
    return result;
  }
  size_t Result(size_t result) {
    result_count_ = result;
    return result;
  }
  std::vector<bool> Result(std::vector<bool> result) {
    if (trace_) result_count_ = std::count(result.begin(), result.end(), true);
    return result;
  }

 private:
  static uint64_t Bytes(uint64_t size) { return size; }
  template <typename T>
  static uint64_t Bytes(const std::vector<T>& sizes) {
    uint64_t total = 0;
    for (const auto& size : sizes) total += Bytes(size);
    return total;
  }
  SimTrace* trace_;
  const char* api_;
  size_t keys_;
  uint64_t bytes_ = 0, id_ = 0;
  int64_t result_count_ = -1;  // void API or no result (exception)
  int exceptions_ = 0;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace mori::umbp
