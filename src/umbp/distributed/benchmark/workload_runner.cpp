// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include "umbp/distributed/benchmark/workload_runner.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#include "umbp/distributed/benchmark/payload.h"

namespace mori::umbp::benchmark {
namespace {

using Event = ::umbp::benchmark::WorkloadEvent;

OperationMetrics& MetricsFor(Event::Operation operation, WorkloadMetrics* metrics) {
  return operation == Event::PUT ? metrics->puts : metrics->gets;
}

void AddAttempt(const Event& event, WorkloadMetrics* metrics) {
  OperationMetrics& operation = MetricsFor(event.operation(), metrics);
  ++operation.attempted;
  operation.attempted_bytes += event.value_size();
  ++metrics->total.attempted;
  metrics->total.attempted_bytes += event.value_size();
}

void AddResult(const Event& event, bool success, WorkloadMetrics* metrics) {
  OperationMetrics& operation = MetricsFor(event.operation(), metrics);
  if (success) {
    ++operation.succeeded;
    operation.succeeded_bytes += event.value_size();
    ++metrics->total.succeeded;
    metrics->total.succeeded_bytes += event.value_size();
  } else {
    ++operation.failed;
    ++metrics->total.failed;
  }
}

void MergeOperation(const OperationMetrics& source, OperationMetrics* destination) {
  destination->attempted += source.attempted;
  destination->succeeded += source.succeeded;
  destination->failed += source.failed;
  destination->attempted_bytes += source.attempted_bytes;
  destination->succeeded_bytes += source.succeeded_bytes;
}

void AddWindowResult(const Event& event, bool success, OperationMetrics* metrics) {
  ++metrics->attempted;
  metrics->attempted_bytes += event.value_size();
  if (success) {
    ++metrics->succeeded;
    metrics->succeeded_bytes += event.value_size();
  } else {
    ++metrics->failed;
  }
}

void MergeMetrics(const WorkloadMetrics& source, WorkloadMetrics* destination) {
  MergeOperation(source.total, &destination->total);
  MergeOperation(source.puts, &destination->puts);
  MergeOperation(source.gets, &destination->gets);
  destination->get_misses += source.get_misses;
  destination->get_validation_failures += source.get_validation_failures;
  destination->latency.samples_ns.insert(destination->latency.samples_ns.end(),
                                         source.latency.samples_ns.begin(),
                                         source.latency.samples_ns.end());
  destination->schedule_lag.samples_ns.insert(destination->schedule_lag.samples_ns.end(),
                                              source.schedule_lag.samples_ns.begin(),
                                              source.schedule_lag.samples_ns.end());
}

void FinalizeDistribution(DistributionMetrics* distribution) {
  if (distribution->samples_ns.empty()) return;
  std::vector<uint64_t> sorted = distribution->samples_ns;
  std::sort(sorted.begin(), sorted.end());
  auto percentile = [&sorted](double fraction) {
    const size_t rank =
        std::max<size_t>(1, static_cast<size_t>(std::ceil(fraction * sorted.size())));
    return sorted[rank - 1];
  };
  distribution->p50_ns = percentile(0.50);
  distribution->p95_ns = percentile(0.95);
  distribution->p99_ns = percentile(0.99);
  distribution->max_ns = sorted.back();
}

uint64_t Nanoseconds(std::chrono::steady_clock::duration duration) {
  if (duration <= std::chrono::steady_clock::duration::zero()) return 0;
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
}

}  // namespace

std::vector<ClientResult> WorkloadClient::BatchPut(
    uint32_t client_id, const std::vector<std::string>& keys,
    const std::vector<std::vector<uint8_t>>& values) {
  if (keys.size() != values.size()) {
    throw std::invalid_argument("BatchPut keys and values must have equal sizes");
  }
  std::vector<ClientResult> results;
  results.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    results.push_back(Put(client_id, keys[i], values[i].data(), values[i].size()));
  }
  return results;
}

std::vector<ClientResult> WorkloadClient::BatchGet(
    uint32_t client_id, const std::vector<std::string>& keys,
    const std::vector<size_t>& sizes, std::vector<std::vector<uint8_t>>* values) {
  if (keys.size() != sizes.size() || values == nullptr) {
    throw std::invalid_argument("BatchGet arguments are invalid");
  }
  values->clear();
  values->resize(keys.size());
  std::vector<ClientResult> results;
  results.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    (*values)[i].resize(sizes[i]);
    results.push_back(Get(client_id, keys[i], (*values)[i].data(), sizes[i]));
  }
  return results;
}

WorkloadRunner::WorkloadRunner(WorkloadClient* client, WorkloadRunnerOptions options)
    : client_(client), options_(options) {
  if (client_ == nullptr) throw std::invalid_argument("client must not be null");
  if (options_.time_scale <= 0.0 || !std::isfinite(options_.time_scale)) {
    throw std::invalid_argument("time_scale must be finite and positive");
  }
}

WorkloadMetrics WorkloadRunner::Run(WorkloadSource* source) {
  if (source == nullptr) throw std::invalid_argument("source must not be null");

  std::map<uint32_t, std::vector<Event>> client_events;
  std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> put_identity;
  Event event;
  while (source->Next(&event)) {
    if (event.operation() != Event::PUT && event.operation() != Event::GET) {
      throw std::invalid_argument("workload source produced an invalid operation");
    }
    if (event.key().empty()) {
      throw std::invalid_argument("workload source produced an empty key");
    }
    if (event.value_size() > std::numeric_limits<size_t>::max()) {
      throw std::length_error("workload value does not fit in size_t");
    }
    if (event.operation() == Event::PUT) {
      const auto [found, inserted] =
          put_identity.emplace(event.key(),
                               std::make_pair(event.operation_id(), event.value_size()));
      if (!inserted) {
        const bool conflicting =
            found->second.first != event.operation_id() ||
            found->second.second != event.value_size();
        throw std::invalid_argument(
            conflicting
                ? "workload contains conflicting PUT payload identities for an immutable key"
                : "workload contains a duplicate PUT for an immutable key");
      }
    }
    auto& events = client_events[event.client_id()];
    if (!events.empty() &&
        event.relative_timestamp_ns() < events.back().relative_timestamp_ns()) {
      throw std::invalid_argument("per-client workload timestamps must be nondecreasing");
    }
    events.push_back(event);
  }

  const auto start = Clock::now();
  const uint64_t payload_seed = source->seed();
  WorkloadMetrics metrics;
  std::mutex metrics_mutex;
  std::map<uint64_t, OperationMetrics> window_totals;
  std::exception_ptr worker_error;
  std::vector<std::thread> workers;
  workers.reserve(client_events.size());

  for (auto& entry : client_events) {
    const uint32_t client_id = entry.first;
    workers.emplace_back([&, client_id, events = std::move(entry.second)] {
      try {
        WorkloadMetrics local;
        std::map<uint64_t, OperationMetrics> local_windows;
        size_t begin = 0;
        while (begin < events.size()) {
          size_t end = begin + 1;
          if (events[begin].batch_id() != 0) {
            while (end < events.size() &&
                   events[end].batch_id() == events[begin].batch_id() &&
                   events[end].operation() == events[begin].operation()) {
              ++end;
            }
          }

          if (!options_.max_throughput) {
            const long double scaled =
                static_cast<long double>(events[end - 1].relative_timestamp_ns()) *
                options_.time_scale;
            if (scaled > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
              throw std::overflow_error("scaled workload timestamp is too large");
            }
            const auto target =
                start + std::chrono::nanoseconds(static_cast<int64_t>(scaled));
            std::this_thread::sleep_until(target);
          }

          const auto dispatch_time = Clock::now();
          for (size_t i = begin; i < end; ++i) {
            AddAttempt(events[i], &local);
            if (!options_.max_throughput) {
              const auto target =
                  start + std::chrono::nanoseconds(static_cast<int64_t>(
                              static_cast<long double>(events[i].relative_timestamp_ns()) *
                              options_.time_scale));
              local.schedule_lag.samples_ns.push_back(Nanoseconds(dispatch_time - target));
            }
          }

          std::vector<std::string> keys;
          std::vector<size_t> sizes;
          keys.reserve(end - begin);
          sizes.reserve(end - begin);
          for (size_t i = begin; i < end; ++i) {
            keys.push_back(events[i].key());
            sizes.push_back(static_cast<size_t>(events[i].value_size()));
          }

          std::vector<ClientResult> results;
          std::vector<std::vector<uint8_t>> values;
          const auto operation_start = Clock::now();
          if (events[begin].operation() == Event::PUT) {
            values.reserve(end - begin);
            for (size_t i = begin; i < end; ++i) {
              values.push_back(GenerateDeterministicPayload(
                  events[i].key(), events[i].operation_id(), payload_seed,
                  sizes[i - begin]));
            }
            results = client_->BatchPut(client_id, keys, values);
          } else {
            results = client_->BatchGet(client_id, keys, sizes, &values);
          }
          const uint64_t latency = Nanoseconds(Clock::now() - operation_start);
          const uint64_t completion_ns = Nanoseconds(Clock::now() - start);

          for (size_t i = begin; i < end; ++i) {
            const size_t result_index = i - begin;
            ClientResult result = result_index < results.size() ? results[result_index]
                                                                : ClientResult::kFailed;
            bool success = result == ClientResult::kSuccess;
            if (events[i].operation() == Event::GET) {
              if (result == ClientResult::kNotFound) {
                ++local.get_misses;
              } else if (success && options_.validate_get_payloads) {
                if (result_index >= values.size() ||
                    values[result_index].size() != sizes[result_index] ||
                    !ValidateDeterministicPayload(
                        events[i].key(), events[i].operation_id(), payload_seed,
                        values[result_index].data(), values[result_index].size())) {
                  ++local.get_validation_failures;
                  success = false;
                }
              }
            }
            AddResult(events[i], success, &local);
            local.latency.samples_ns.push_back(latency);
            if (options_.window_ns != 0) {
              AddWindowResult(events[i], success,
                              &local_windows[completion_ns / options_.window_ns]);
            }
          }
          begin = end;
        }
        std::lock_guard<std::mutex> lock(metrics_mutex);
        MergeMetrics(local, &metrics);
        for (const auto& [index, totals] : local_windows) {
          MergeOperation(totals, &window_totals[index]);
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(metrics_mutex);
        if (!worker_error) worker_error = std::current_exception();
      }
    });
  }

  for (std::thread& worker : workers) worker.join();
  if (worker_error) std::rethrow_exception(worker_error);
  FinalizeDistribution(&metrics.latency);
  FinalizeDistribution(&metrics.schedule_lag);
  metrics.windows.reserve(window_totals.size());
  for (const auto& [index, totals] : window_totals) {
    WorkloadMetrics::Window window;
    window.index = index;
    window.start_ns = index * options_.window_ns;
    window.end_ns = window.start_ns + options_.window_ns;
    window.total = totals;
    metrics.windows.push_back(window);
  }
  return metrics;
}

}  // namespace mori::umbp::benchmark
