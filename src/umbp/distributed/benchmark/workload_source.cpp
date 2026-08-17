// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include "umbp/distributed/benchmark/workload_source.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mori::umbp::benchmark {

TraceWorkloadSource::TraceWorkloadSource(const std::string& path, TraceLimits limits)
    : reader_(path, limits) {}

bool TraceWorkloadSource::Next(::umbp::benchmark::WorkloadEvent* event) {
  return reader_.ReadNext(event);
}

SyntheticWorkloadSource::SyntheticWorkloadSource(SyntheticWorkloadConfig config)
    : config_(std::move(config)), random_(config_.seed) {
  if (config_.key_count == 0) throw std::invalid_argument("key_count must be positive");
  if (config_.client_count == 0) throw std::invalid_argument("client_count must be positive");
  if (config_.batch_size == 0) throw std::invalid_argument("batch_size must be positive");
  if (config_.min_value_size > config_.max_value_size) {
    throw std::invalid_argument("min_value_size must not exceed max_value_size");
  }
  if (config_.read_ratio < 0.0 || config_.read_ratio > 1.0) {
    throw std::invalid_argument("read_ratio must be in [0, 1]");
  }
  if (config_.qps < 0.0 || !std::isfinite(config_.qps)) {
    throw std::invalid_argument("qps must be finite and non-negative");
  }
  if (config_.hotset_fraction <= 0.0 || config_.hotset_fraction > 1.0) {
    throw std::invalid_argument("hotset_fraction must be in (0, 1]");
  }
  if (config_.zipf_exponent <= 0.0 || !std::isfinite(config_.zipf_exponent)) {
    throw std::invalid_argument("zipf_exponent must be finite and positive");
  }

  key_versions_.assign(config_.key_count, 0);
  last_put_operation_.assign(config_.key_count, 0);
  last_put_size_.assign(config_.key_count, 0);
  if (config_.profile == SyntheticProfile::kHotsetZipf) {
    const uint64_t hot_keys =
        std::max<uint64_t>(1, static_cast<uint64_t>(std::ceil(
                                  static_cast<double>(config_.key_count) *
                                  config_.hotset_fraction)));
    zipf_cdf_.reserve(hot_keys);
    double sum = 0.0;
    for (uint64_t rank = 1; rank <= hot_keys; ++rank) {
      sum += 1.0 / std::pow(static_cast<double>(rank), config_.zipf_exponent);
      zipf_cdf_.push_back(sum);
    }
    for (double& value : zipf_cdf_) value /= sum;
  }
}

uint64_t SyntheticWorkloadSource::SelectKeyIndex(uint64_t sequence) {
  switch (config_.profile) {
    case SyntheticProfile::kSequential:
    case SyntheticProfile::kReadAfterWrite:
    case SyntheticProfile::kMixed:
      return (config_.profile == SyntheticProfile::kReadAfterWrite ? sequence / 2 : sequence) %
             config_.key_count;
    case SyntheticProfile::kCapacityPressure:
      // Pressure uses a non-repeating key stream so successful PUTs consume
      // fresh capacity instead of becoming backend dedup hits.
      return sequence;
    case SyntheticProfile::kHotsetZipf: {
      const double sample = std::generate_canonical<double, 53>(random_);
      return static_cast<uint64_t>(
          std::lower_bound(zipf_cdf_.begin(), zipf_cdf_.end(), sample) -
          zipf_cdf_.begin());
    }
    case SyntheticProfile::kUniform:
      return std::uniform_int_distribution<uint64_t>(0, config_.key_count - 1)(random_);
  }
  throw std::logic_error("unknown synthetic workload profile");
}

uint64_t SyntheticWorkloadSource::SelectValueSize() {
  if (config_.value_size_distribution == ValueSizeDistribution::kFixed ||
      config_.min_value_size == config_.max_value_size) {
    return config_.min_value_size;
  }
  if (config_.value_size_distribution == ValueSizeDistribution::kUniform) {
    return std::uniform_int_distribution<uint64_t>(config_.min_value_size,
                                                   config_.max_value_size)(random_);
  }
  const double low = std::log(static_cast<double>(std::max<size_t>(1, config_.min_value_size)));
  const double high = std::log(static_cast<double>(std::max<size_t>(1, config_.max_value_size)));
  const double sample = std::uniform_real_distribution<double>(low, high)(random_);
  return std::clamp<uint64_t>(static_cast<uint64_t>(std::llround(std::exp(sample))),
                              config_.min_value_size, config_.max_value_size);
}

bool SyntheticWorkloadSource::SelectRead(uint64_t sequence) {
  if (config_.profile == SyntheticProfile::kSequential ||
      config_.profile == SyntheticProfile::kCapacityPressure) {
    return false;
  }
  if (config_.profile == SyntheticProfile::kReadAfterWrite) return sequence % 2 == 1;
  return std::bernoulli_distribution(config_.read_ratio)(random_);
}

bool SyntheticWorkloadSource::Next(::umbp::benchmark::WorkloadEvent* event) {
  if (event == nullptr) throw std::invalid_argument("event must not be null");
  if (generated_ >= config_.operation_count) return false;

  const uint64_t sequence = generated_;
  const uint64_t key_index = SelectKeyIndex(sequence);
  bool is_read = SelectRead(sequence);
  const bool tracks_working_set = config_.profile != SyntheticProfile::kCapacityPressure;
  if (tracks_working_set && is_read && last_put_operation_[key_index] == 0) is_read = false;

  event->Clear();
  if (config_.qps > 0.0) {
    const uint64_t events_per_batch =
        static_cast<uint64_t>(config_.batch_size) * config_.client_count;
    const uint64_t batch_sequence = (sequence / events_per_batch) * events_per_batch;
    const long double timestamp =
        static_cast<long double>(batch_sequence) * 1000000000.0L / config_.qps;
    if (timestamp > static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
      throw std::overflow_error("synthetic timestamp exceeds uint64_t");
    }
    event->set_relative_timestamp_ns(static_cast<uint64_t>(timestamp));
  }
  // Keep every key's dependency chain on one logical client. The runner may
  // execute different clients concurrently, so assigning by event sequence
  // could otherwise let a GET overtake the PUT that established its payload.
  event->set_client_id(static_cast<uint32_t>(key_index % config_.client_count));
  if (!is_read && tracks_working_set) ++key_versions_[key_index];
  event->set_key(config_.key_prefix + std::to_string(key_index) + "-v" +
                 std::to_string(tracks_working_set ? key_versions_[key_index] : 1));
  event->set_batch_id(sequence /
                          (static_cast<uint64_t>(config_.batch_size) *
                           config_.client_count) +
                      1);

  const uint64_t new_operation_id = sequence + 1;
  if (is_read) {
    event->set_operation(::umbp::benchmark::WorkloadEvent::GET);
    event->set_operation_id(last_put_operation_[key_index]);
    event->set_value_size(last_put_size_[key_index]);
  } else {
    event->set_operation(::umbp::benchmark::WorkloadEvent::PUT);
    event->set_operation_id(new_operation_id);
    event->set_value_size(SelectValueSize());
    if (tracks_working_set) {
      last_put_operation_[key_index] = new_operation_id;
      last_put_size_[key_index] = event->value_size();
    }
  }
  ++generated_;
  return true;
}

}  // namespace mori::umbp::benchmark
