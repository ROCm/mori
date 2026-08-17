// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "umbp/distributed/benchmark/workload_trace.h"

namespace mori::umbp::benchmark {

class WorkloadSource {
 public:
  virtual ~WorkloadSource() = default;
  virtual uint64_t seed() const = 0;
  virtual bool Next(::umbp::benchmark::WorkloadEvent* event) = 0;
};

class TraceWorkloadSource final : public WorkloadSource {
 public:
  explicit TraceWorkloadSource(const std::string& path, TraceLimits limits = {});

  uint64_t seed() const override { return reader_.header().seed(); }
  bool Next(::umbp::benchmark::WorkloadEvent* event) override;
  const ::umbp::benchmark::WorkloadTraceHeader& header() const { return reader_.header(); }

 private:
  TraceReader reader_;
};

enum class SyntheticProfile {
  kSequential,
  kUniform,
  kHotsetZipf,
  kReadAfterWrite,
  kMixed,
  kCapacityPressure,
};

enum class ValueSizeDistribution {
  kFixed,
  kUniform,
  kLogUniform,
};

struct SyntheticWorkloadConfig {
  SyntheticProfile profile = SyntheticProfile::kMixed;
  uint64_t seed = 1;
  uint64_t operation_count = 1000;
  uint64_t key_count = 100;
  size_t min_value_size = 4096;
  size_t max_value_size = 4096;
  ValueSizeDistribution value_size_distribution = ValueSizeDistribution::kFixed;
  double read_ratio = 0.5;
  uint32_t client_count = 1;
  uint32_t batch_size = 1;
  double qps = 0.0;
  double hotset_fraction = 0.1;
  double zipf_exponent = 1.1;
  std::string key_prefix = "umbp-bench-";
};

class SyntheticWorkloadSource final : public WorkloadSource {
 public:
  explicit SyntheticWorkloadSource(SyntheticWorkloadConfig config);

  uint64_t seed() const override { return config_.seed; }
  bool Next(::umbp::benchmark::WorkloadEvent* event) override;
  const SyntheticWorkloadConfig& config() const { return config_; }

 private:
  uint64_t SelectKeyIndex(uint64_t sequence);
  uint64_t SelectValueSize();
  bool SelectRead(uint64_t sequence);

  SyntheticWorkloadConfig config_;
  std::mt19937_64 random_;
  std::vector<double> zipf_cdf_;
  std::vector<uint64_t> key_versions_;
  std::vector<uint64_t> last_put_operation_;
  std::vector<uint64_t> last_put_size_;
  uint64_t generated_ = 0;
};

}  // namespace mori::umbp::benchmark
