// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "umbp/distributed/benchmark/workload_source.h"

namespace mori::umbp::benchmark {

enum class ClientResult {
  kSuccess,
  kNotFound,
  kFailed,
};

// A transport-neutral interface suitable for a real UMBP adapter or a test fake.
// Implementations used with multiple clients must permit concurrent calls.
class WorkloadClient {
 public:
  virtual ~WorkloadClient() = default;
  virtual ClientResult Put(uint32_t client_id, const std::string& key,
                           const uint8_t* data, size_t size) = 0;
  virtual ClientResult Get(uint32_t client_id, const std::string& key, uint8_t* data,
                           size_t size) = 0;

  virtual std::vector<ClientResult> BatchPut(
      uint32_t client_id, const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values);
  virtual std::vector<ClientResult> BatchGet(uint32_t client_id,
                                             const std::vector<std::string>& keys,
                                             const std::vector<size_t>& sizes,
                                             std::vector<std::vector<uint8_t>>* values);
};

struct OperationMetrics {
  uint64_t attempted = 0;
  uint64_t succeeded = 0;
  uint64_t failed = 0;
  uint64_t attempted_bytes = 0;
  uint64_t succeeded_bytes = 0;
};

struct DistributionMetrics {
  std::vector<uint64_t> samples_ns;
  uint64_t p50_ns = 0;
  uint64_t p95_ns = 0;
  uint64_t p99_ns = 0;
  uint64_t max_ns = 0;
};

struct WorkloadMetrics {
  OperationMetrics total;
  OperationMetrics puts;
  OperationMetrics gets;
  uint64_t get_misses = 0;
  uint64_t get_validation_failures = 0;
  DistributionMetrics latency;
  DistributionMetrics schedule_lag;
  struct Window {
    uint64_t index = 0;
    uint64_t start_ns = 0;
    uint64_t end_ns = 0;
    OperationMetrics total;
  };
  std::vector<Window> windows;
};

struct WorkloadRunnerOptions {
  // Ignore trace timestamps and issue each client's operations immediately.
  bool max_throughput = false;
  // Trace intervals are multiplied by this value. Must be finite and positive.
  double time_scale = 1.0;
  bool validate_get_payloads = true;
  // Zero disables time-window aggregation.
  uint64_t window_ns = 0;
};

class WorkloadRunner {
 public:
  explicit WorkloadRunner(WorkloadClient* client, WorkloadRunnerOptions options = {});

  WorkloadMetrics Run(WorkloadSource* source);

 private:
  using Clock = std::chrono::steady_clock;

  WorkloadClient* client_;
  WorkloadRunnerOptions options_;
};

}  // namespace mori::umbp::benchmark
