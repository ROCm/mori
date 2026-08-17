// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/distributed/benchmark/workload_runner.h"

namespace mori::umbp::benchmark {
namespace {

using Event = ::umbp::benchmark::WorkloadEvent;

Event MakeEvent(uint32_t client, uint64_t id, Event::Operation operation,
                const std::string& key, uint64_t timestamp_ns = 0,
                uint64_t batch_id = 0) {
  Event event;
  event.set_client_id(client);
  event.set_operation_id(id);
  event.set_operation(operation);
  event.set_key(key);
  event.set_value_size(32);
  event.set_relative_timestamp_ns(timestamp_ns);
  event.set_batch_id(batch_id);
  return event;
}

WorkloadRunnerOptions MaxThroughputOptions() {
  WorkloadRunnerOptions options;
  options.max_throughput = true;
  return options;
}

class VectorSource final : public WorkloadSource {
 public:
  explicit VectorSource(std::vector<Event> events, uint64_t seed = 7)
      : events_(std::move(events)), seed_(seed) {}

  uint64_t seed() const override { return seed_; }
  bool Next(Event* event) override {
    if (next_ == events_.size()) return false;
    *event = events_[next_++];
    return true;
  }

 private:
  std::vector<Event> events_;
  uint64_t seed_;
  size_t next_ = 0;
};

struct ClientCall {
  uint32_t client_id;
  std::string key;
};

class FakeClient final : public WorkloadClient {
 public:
  ClientResult Put(uint32_t client_id, const std::string& key, const uint8_t* data,
                   size_t size) override {
    std::lock_guard<std::mutex> lock(mutex_);
    calls_.push_back({client_id, key});
    if (key == "put-fail") return ClientResult::kFailed;
    values_[client_id][key] = std::vector<uint8_t>(data, data + size);
    return ClientResult::kSuccess;
  }

  ClientResult Get(uint32_t client_id, const std::string& key, uint8_t* data,
                   size_t size) override {
    std::lock_guard<std::mutex> lock(mutex_);
    calls_.push_back({client_id, key});
    const auto client = values_.find(client_id);
    if (client == values_.end()) return ClientResult::kNotFound;
    const auto found = client->second.find(key);
    if (found == client->second.end()) return ClientResult::kNotFound;
    if (found->second.size() != size) return ClientResult::kFailed;
    std::copy(found->second.begin(), found->second.end(), data);
    if (key == "corrupt" && size != 0) data[0] ^= 1;
    return ClientResult::kSuccess;
  }

  std::vector<ClientResult> BatchPut(
      uint32_t client_id, const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values) override {
    ++batch_put_calls;
    return WorkloadClient::BatchPut(client_id, keys, values);
  }

  std::vector<ClientResult> BatchGet(
      uint32_t client_id, const std::vector<std::string>& keys,
      const std::vector<size_t>& sizes,
      std::vector<std::vector<uint8_t>>* values) override {
    ++batch_get_calls;
    return WorkloadClient::BatchGet(client_id, keys, sizes, values);
  }

  std::vector<ClientCall> Calls() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return calls_;
  }

  std::atomic<uint64_t> batch_put_calls{0};
  std::atomic<uint64_t> batch_get_calls{0};

 private:
  mutable std::mutex mutex_;
  std::unordered_map<uint32_t,
                     std::unordered_map<std::string, std::vector<uint8_t>>>
      values_;
  std::vector<ClientCall> calls_;
};

TEST(WorkloadRunnerTest, PreservesOrderingWithinEachClient) {
  std::vector<Event> events;
  for (uint64_t i = 0; i < 20; ++i) {
    events.push_back(MakeEvent(0, i + 1, Event::PUT, "a-" + std::to_string(i)));
    events.push_back(MakeEvent(1, i + 101, Event::PUT, "b-" + std::to_string(i)));
  }
  VectorSource source(std::move(events));
  FakeClient client;
  WorkloadRunner runner(&client, MaxThroughputOptions());
  const WorkloadMetrics metrics = runner.Run(&source);
  EXPECT_EQ(metrics.puts.succeeded, 40);

  uint64_t next_a = 0;
  uint64_t next_b = 0;
  for (const ClientCall& call : client.Calls()) {
    if (call.key[0] == 'a') {
      EXPECT_EQ(call.key, "a-" + std::to_string(next_a++));
    } else {
      EXPECT_EQ(call.key, "b-" + std::to_string(next_b++));
    }
  }
  EXPECT_EQ(next_a, 20);
  EXPECT_EQ(next_b, 20);
}

TEST(WorkloadRunnerTest, DeliversDistinctClientIdsInPerClientOrder) {
  VectorSource source({
      MakeEvent(11, 1, Event::PUT, "first-11"),
      MakeEvent(22, 2, Event::PUT, "first-22"),
      MakeEvent(11, 3, Event::PUT, "second-11"),
      MakeEvent(22, 4, Event::PUT, "second-22"),
  });
  FakeClient client;
  WorkloadRunner runner(&client, MaxThroughputOptions());
  EXPECT_EQ(runner.Run(&source).total.succeeded, 4);

  std::vector<std::string> client_11_keys;
  std::vector<std::string> client_22_keys;
  for (const ClientCall& call : client.Calls()) {
    if (call.client_id == 11) {
      client_11_keys.push_back(call.key);
    } else if (call.client_id == 22) {
      client_22_keys.push_back(call.key);
    } else {
      ADD_FAILURE() << "unexpected client id " << call.client_id;
    }
  }
  EXPECT_EQ(client_11_keys, (std::vector<std::string>{"first-11", "second-11"}));
  EXPECT_EQ(client_22_keys, (std::vector<std::string>{"first-22", "second-22"}));
}

TEST(WorkloadRunnerTest, HonorsSchedulingAndMaxThroughputModes) {
  constexpr uint64_t kDelayNs = 30'000'000;
  FakeClient scheduled_client;
  VectorSource scheduled_source({MakeEvent(0, 1, Event::PUT, "scheduled", kDelayNs)});
  const auto scheduled_start = std::chrono::steady_clock::now();
  WorkloadRunner scheduled_runner(&scheduled_client);
  WorkloadMetrics scheduled_metrics = scheduled_runner.Run(&scheduled_source);
  const auto scheduled_elapsed = std::chrono::steady_clock::now() - scheduled_start;
  EXPECT_GE(scheduled_elapsed, std::chrono::milliseconds(20));
  ASSERT_EQ(scheduled_metrics.schedule_lag.samples_ns.size(), 1);

  FakeClient fast_client;
  VectorSource fast_source({MakeEvent(0, 1, Event::PUT, "fast", kDelayNs)});
  const auto fast_start = std::chrono::steady_clock::now();
  WorkloadRunner fast_runner(&fast_client, MaxThroughputOptions());
  WorkloadMetrics fast_metrics = fast_runner.Run(&fast_source);
  const auto fast_elapsed = std::chrono::steady_clock::now() - fast_start;
  EXPECT_LT(fast_elapsed, scheduled_elapsed);
  EXPECT_TRUE(fast_metrics.schedule_lag.samples_ns.empty());
}

TEST(WorkloadRunnerTest, GroupsCompatibleBatchIds) {
  std::vector<Event> events;
  for (uint64_t i = 0; i < 4; ++i) {
    events.push_back(MakeEvent(0, i + 1, Event::PUT, "key-" + std::to_string(i), 0, 7));
  }
  for (uint64_t i = 0; i < 4; ++i) {
    events.push_back(MakeEvent(0, i + 1, Event::GET, "key-" + std::to_string(i), 0, 8));
  }
  VectorSource source(std::move(events));
  FakeClient client;
  WorkloadRunner runner(&client, MaxThroughputOptions());
  const WorkloadMetrics metrics = runner.Run(&source);
  EXPECT_EQ(client.batch_put_calls, 1);
  EXPECT_EQ(client.batch_get_calls, 1);
  EXPECT_EQ(metrics.total.succeeded, 8);
}

TEST(WorkloadRunnerTest, AccountsForFailuresAndComputesPercentiles) {
  std::vector<Event> events = {
      MakeEvent(0, 1, Event::PUT, "good"),
      MakeEvent(0, 1, Event::GET, "good"),
      MakeEvent(0, 2, Event::GET, "missing"),
      MakeEvent(0, 3, Event::PUT, "put-fail"),
      MakeEvent(0, 4, Event::PUT, "corrupt"),
      MakeEvent(0, 4, Event::GET, "corrupt"),
  };
  VectorSource source(std::move(events));
  FakeClient client;
  WorkloadRunner runner(&client, MaxThroughputOptions());
  const WorkloadMetrics metrics = runner.Run(&source);

  EXPECT_EQ(metrics.total.attempted, 6);
  EXPECT_EQ(metrics.total.succeeded, 3);
  EXPECT_EQ(metrics.total.failed, 3);
  EXPECT_EQ(metrics.get_misses, 1);
  EXPECT_EQ(metrics.get_validation_failures, 1);
  ASSERT_EQ(metrics.latency.samples_ns.size(), 6);
  EXPECT_LE(metrics.latency.p50_ns, metrics.latency.p95_ns);
  EXPECT_LE(metrics.latency.p95_ns, metrics.latency.p99_ns);
  EXPECT_LE(metrics.latency.p99_ns, metrics.latency.max_ns);
}

TEST(WorkloadRunnerTest, RejectsConflictingPutsForImmutableKeys) {
  VectorSource source({
      MakeEvent(0, 1, Event::PUT, "immutable"),
      MakeEvent(0, 2, Event::PUT, "immutable"),
  });
  FakeClient client;
  WorkloadRunner runner(&client, MaxThroughputOptions());
  EXPECT_THROW(runner.Run(&source), std::invalid_argument);
}

TEST(WorkloadRunnerTest, RejectsDuplicatePutsSoTransferredBytesStayExact) {
  VectorSource source({
      MakeEvent(0, 1, Event::PUT, "immutable"),
      MakeEvent(0, 1, Event::PUT, "immutable"),
  });
  FakeClient client;
  WorkloadRunner runner(&client, MaxThroughputOptions());
  EXPECT_THROW(runner.Run(&source), std::invalid_argument);
}

TEST(WorkloadRunnerTest, AggregatesOptionalTimeWindows) {
  VectorSource source({
      MakeEvent(0, 1, Event::PUT, "one"),
      MakeEvent(0, 2, Event::PUT, "two"),
  });
  FakeClient client;
  auto options = MaxThroughputOptions();
  options.window_ns = 1'000'000'000;
  WorkloadRunner runner(&client, options);
  const auto metrics = runner.Run(&source);
  ASSERT_EQ(metrics.windows.size(), 1u);
  EXPECT_EQ(metrics.windows[0].index, 0u);
  EXPECT_EQ(metrics.windows[0].total.attempted, 2u);
  EXPECT_EQ(metrics.windows[0].total.succeeded, 2u);
}

}  // namespace
}  // namespace mori::umbp::benchmark
