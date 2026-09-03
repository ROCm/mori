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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "umbp/distributed/benchmark/workload_trace.h"

namespace mori::umbp::benchmark {

struct WorkloadTraceRecorderOptions {
  std::string path;
  uint32_t client_id = 0;
  uint64_t seed = 0;
  std::string node_id;
  std::string backend_policy;
};

// How the recorded system answered an operation. A miss is distinct from a
// failure because only one of them says something about the cache.
enum class WorkloadTraceOutcome {
  kSuccess,
  kMiss,
  kFailure,
};

class WorkloadTraceRecorder {
 public:
  using Options = WorkloadTraceRecorderOptions;

  explicit WorkloadTraceRecorder(Options options);
  ~WorkloadTraceRecorder();

  WorkloadTraceRecorder(const WorkloadTraceRecorder&) = delete;
  WorkloadTraceRecorder& operator=(const WorkloadTraceRecorder&) = delete;

  // Records every attempted operation, keys verbatim. A trace that keeps only
  // the operations that succeeded, or that rewrites keys to make each write
  // unique, is no longer the workload that ran: the reuse and overwrite
  // patterns that decide what a tier policy does are exactly what those
  // transformations remove.
  void RecordBatchPut(const std::vector<std::string>& keys, const std::vector<size_t>& sizes,
                      const std::vector<WorkloadTraceOutcome>& outcomes) {
    RecordBatch(::umbp::benchmark::WorkloadEvent::PUT, keys, sizes, outcomes);
  }
  void RecordBatchGet(const std::vector<std::string>& keys, const std::vector<size_t>& sizes,
                      const std::vector<WorkloadTraceOutcome>& outcomes) {
    RecordBatch(::umbp::benchmark::WorkloadEvent::GET, keys, sizes, outcomes);
  }
  void Close();
  uint64_t event_count() const;

 private:
  using Clock = std::chrono::steady_clock;

  void RecordBatch(::umbp::benchmark::WorkloadEvent::Operation operation,
                   const std::vector<std::string>& keys, const std::vector<size_t>& sizes,
                   const std::vector<WorkloadTraceOutcome>& outcomes);

  uint64_t RelativeTimestampNs() const;

  bool WriteLocked(const ::umbp::benchmark::WorkloadEvent& event);

  const uint32_t client_id_;
  mutable std::mutex mutex_;
  TraceWriter writer_;
  const Clock::time_point start_;
  uint64_t next_operation_id_ = 0;
  uint64_t next_batch_id_ = 0;
  uint64_t event_count_ = 0;
  bool failed_ = false;
  std::string error_;
};

}  // namespace mori::umbp::benchmark
