// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
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

class WorkloadTraceRecorder {
 public:
  using Options = WorkloadTraceRecorderOptions;

  explicit WorkloadTraceRecorder(Options options);
  ~WorkloadTraceRecorder();

  WorkloadTraceRecorder(const WorkloadTraceRecorder&) = delete;
  WorkloadTraceRecorder& operator=(const WorkloadTraceRecorder&) = delete;

  void RecordBatchPut(const std::vector<std::string>& keys,
                      const std::vector<size_t>& sizes,
                      const std::vector<bool>& successes);
  void RecordBatchGet(const std::vector<std::string>& keys,
                      const std::vector<size_t>& sizes,
                      const std::vector<bool>& successes);
  void Close();
  uint64_t event_count() const;

 private:
  using Clock = std::chrono::steady_clock;

  uint64_t RelativeTimestampNs() const;

  const uint32_t client_id_;
  mutable std::mutex mutex_;
  TraceWriter writer_;
  const Clock::time_point start_;
  std::unordered_map<std::string, uint64_t> last_put_ids_;
  std::unordered_map<std::string, std::string> recorded_keys_;
  uint64_t next_operation_id_ = 0;
  uint64_t next_batch_id_ = 0;
  uint64_t event_count_ = 0;
  bool failed_ = false;
  std::string error_;
};

}  // namespace mori::umbp::benchmark
