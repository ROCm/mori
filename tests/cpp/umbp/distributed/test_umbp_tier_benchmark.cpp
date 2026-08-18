// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "umbp/distributed/benchmark/payload.h"
#include "umbp/distributed/benchmark/workload_runner.h"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/distributed/routing/route_put_strategy.h"

namespace mori::umbp {
namespace {

namespace bench = benchmark;
using Event = ::umbp::benchmark::WorkloadEvent;

constexpr size_t kPageSize = 4096;
constexpr size_t kKeysPerClient = 12;
constexpr size_t kBackendCapacity = kPageSize * 64;

uint64_t AdvertisedDramTotal(MasterServer* master, const std::string& node_id) {
  if (master == nullptr) return 0;
  const auto record = master->GetClient(node_id);
  if (!record.has_value()) return 0;
  const auto found = record->tier_capacities.find(TierType::DRAM);
  return found == record->tier_capacities.end() ? 0 : found->second.total_bytes;
}

std::string AdvertisedDramDebug(MasterServer* master, const std::string& node_id) {
  if (master == nullptr) return "master=null";
  const auto record = master->GetClient(node_id);
  if (!record.has_value()) return "no ClientRecord";
  const auto found = record->tier_capacities.find(TierType::DRAM);
  if (found == record->tier_capacities.end()) return "no DRAM tier";
  return "total=" + std::to_string(found->second.total_bytes) +
         " available=" + std::to_string(found->second.available_bytes) +
         " max_alloc=" + std::to_string(found->second.max_allocatable_bytes);
}

bool WaitForAdvertisedDram(MasterServer* master, const std::string& node_id, uint64_t expected_total,
                           std::optional<uint64_t> expected_available = std::nullopt) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  do {
    const auto record = master == nullptr ? std::nullopt : master->GetClient(node_id);
    if (record.has_value()) {
      const auto found = record->tier_capacities.find(TierType::DRAM);
      if (found != record->tier_capacities.end() && found->second.total_bytes == expected_total &&
          (!expected_available.has_value() ||
           found->second.available_bytes == *expected_available)) {
        return true;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  } while (std::chrono::steady_clock::now() < deadline);
  ADD_FAILURE() << node_id << " " << AdvertisedDramDebug(master, node_id)
                << " expected_total=" << expected_total
                << (expected_available.has_value()
                        ? " expected_available=" + std::to_string(*expected_available)
                        : "");
  return false;
}

bool WaitForAdvertisedDramTotal(MasterServer* master, const std::string& node_id,
                                uint64_t expected) {
  return WaitForAdvertisedDram(master, node_id, expected);
}

class VectorSource final : public bench::WorkloadSource {
 public:
  explicit VectorSource(std::vector<Event> events, uint64_t seed)
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

Event MakeEvent(uint32_t client_id, uint64_t operation_id, Event::Operation operation,
                std::string key, uint64_t batch_id) {
  Event event;
  event.set_client_id(client_id);
  event.set_operation_id(operation_id);
  event.set_operation(operation);
  event.set_key(std::move(key));
  event.set_value_size(kPageSize);
  event.set_batch_id(batch_id);
  return event;
}

std::vector<Event> MakeReplay() {
  std::vector<Event> events;
  events.reserve(2 * 2 * kKeysPerClient);
  for (uint32_t client_id = 0; client_id < 2; ++client_id) {
    for (size_t key_index = 0; key_index < kKeysPerClient; ++key_index) {
      const uint64_t operation_id = client_id * kKeysPerClient + key_index + 1;
      events.push_back(MakeEvent(
          client_id, operation_id, Event::PUT,
          "tier-smoke-" + std::to_string(client_id) + "-" + std::to_string(key_index),
          100 + client_id));
    }
    for (size_t key_index = 0; key_index < kKeysPerClient; ++key_index) {
      const uint64_t operation_id = client_id * kKeysPerClient + key_index + 1;
      events.push_back(MakeEvent(
          client_id, operation_id, Event::GET,
          "tier-smoke-" + std::to_string(client_id) + "-" + std::to_string(key_index),
          200 + client_id));
    }
  }
  return events;
}

// WorkloadRunner preserves order within a client but runs different client IDs
// concurrently. This adapter maps each ID to its PoolClient and keeps the data
// path batched. A heartbeat flush after PUT plus bounded GET retries bridges the
// intentionally asynchronous peer-event -> master-index publication boundary.
class PoolWorkloadClient final : public bench::WorkloadClient {
 public:
  explicit PoolWorkloadClient(std::vector<std::unique_ptr<PoolClient>>* clients)
      : clients_(clients) {}

  bench::ClientResult Put(uint32_t client_id, const std::string& key, const uint8_t* data,
                          size_t size) override {
    PoolClient* client = Select(client_id);
    if (client == nullptr) return bench::ClientResult::kFailed;
    const bool success = client->Put(key, data, size);
    if (success) client->Master().FlushHeartbeat();
    return success ? bench::ClientResult::kSuccess : bench::ClientResult::kFailed;
  }

  bench::ClientResult Get(uint32_t client_id, const std::string& key, uint8_t* data,
                          size_t size) override {
    PoolClient* client = Select(client_id);
    if (client == nullptr) return bench::ClientResult::kFailed;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    do {
      if (client->Get(key, data, size)) return bench::ClientResult::kSuccess;
      std::this_thread::sleep_for(std::chrono::milliseconds(25));
    } while (std::chrono::steady_clock::now() < deadline);
    return client->Exists(key) ? bench::ClientResult::kFailed
                               : bench::ClientResult::kNotFound;
  }

  std::vector<bench::ClientResult> BatchPut(
      uint32_t client_id, const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values) override {
    PoolClient* client = Select(client_id);
    if (client == nullptr || keys.size() != values.size()) {
      return std::vector<bench::ClientResult>(keys.size(), bench::ClientResult::kFailed);
    }

    ++batch_put_calls_[client_id];
    std::vector<const void*> sources;
    std::vector<size_t> sizes;
    sources.reserve(values.size());
    sizes.reserve(values.size());
    for (const auto& value : values) {
      sources.push_back(value.data());
      sizes.push_back(value.size());
    }

    const auto results = client->BatchPut(keys, sources, sizes);
    std::vector<bench::ClientResult> converted(keys.size(), bench::ClientResult::kFailed);
    for (size_t i = 0; i < std::min(keys.size(), results.size()); ++i) {
      if (results[i]) converted[i] = bench::ClientResult::kSuccess;
    }
    client->Master().FlushHeartbeat();
    return converted;
  }

  std::vector<bench::ClientResult> BatchGet(
      uint32_t client_id, const std::vector<std::string>& keys,
      const std::vector<size_t>& sizes,
      std::vector<std::vector<uint8_t>>* values) override {
    PoolClient* client = Select(client_id);
    if (client == nullptr || keys.size() != sizes.size() || values == nullptr) {
      return std::vector<bench::ClientResult>(keys.size(), bench::ClientResult::kFailed);
    }

    ++batch_get_calls_[client_id];
    values->resize(keys.size());
    std::vector<void*> destinations(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      (*values)[i].assign(sizes[i], 0);
      destinations[i] = (*values)[i].data();
    }

    std::vector<bool> results;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    do {
      results = client->BatchGet(keys, destinations, sizes);
      if (results.size() == keys.size() &&
          std::all_of(results.begin(), results.end(), [](bool result) { return result; })) {
        return std::vector<bench::ClientResult>(keys.size(), bench::ClientResult::kSuccess);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(25));
    } while (std::chrono::steady_clock::now() < deadline);

    std::vector<bench::ClientResult> converted(keys.size(), bench::ClientResult::kFailed);
    const auto exists = client->BatchExists(keys);
    for (size_t i = 0; i < keys.size(); ++i) {
      if (i >= exists.size() || !exists[i]) converted[i] = bench::ClientResult::kNotFound;
    }
    return converted;
  }

  uint64_t BatchPutCalls(uint32_t client_id) const { return batch_put_calls_[client_id].load(); }
  uint64_t BatchGetCalls(uint32_t client_id) const { return batch_get_calls_[client_id].load(); }

 private:
  PoolClient* Select(uint32_t client_id) const {
    if (clients_ == nullptr || client_id >= clients_->size()) return nullptr;
    return (*clients_)[client_id].get();
  }

  std::vector<std::unique_ptr<PoolClient>>* clients_;
  std::atomic<uint64_t> batch_put_calls_[2]{{0}, {0}};
  std::atomic<uint64_t> batch_get_calls_[2]{{0}, {0}};
};

class TierBenchmarkSmokeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    MasterServerConfig config;
    config.listen_address = "127.0.0.1:0";
    config.metrics_port = 0;
    config.registry_config.default_dram_page_size = kPageSize;
    config.registry_config.heartbeat_ttl = std::chrono::seconds(1);
    config.put_strategy = std::make_unique<ConfigurableRoutePutStrategy>(
        ConfigurableRoutePutStrategy::SelectAlgo::kMostAvailable,
        ConfigurableRoutePutStrategy::NodeAffinity::kNone, 17);
    master_ = std::make_unique<MasterServer>(std::move(config));
    master_thread_ = std::thread([this] { master_->Run(); });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (master_->GetBoundPort() == 0 && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_NE(master_->GetBoundPort(), 0) << "in-process master failed to start";

    StartClient(0);
    StartClient(1);
    ASSERT_EQ(clients_.size(), 2u);

    for (const auto& client : clients_) client->Master().FlushHeartbeat();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  void TearDown() override {
    for (auto& client : clients_) {
      if (client) client->Clear();
    }
    for (auto it = clients_.rbegin(); it != clients_.rend(); ++it) {
      if (*it) (*it)->Shutdown();
    }
    clients_.clear();
    if (master_) master_->Shutdown();
    if (master_thread_.joinable()) master_thread_.join();
    master_.reset();
  }

  void StartClient(uint32_t client_id) {
    PoolClientConfig config;
    config.master_config.master_address =
        "127.0.0.1:" + std::to_string(master_->GetBoundPort());
    config.master_config.node_id = "tier-smoke-node-" + std::to_string(client_id);
    config.master_config.node_address = "127.0.0.1";
    config.master_config.auto_heartbeat = true;
    config.io_engine.host = "127.0.0.1";
    config.io_engine.port = 0;
    config.auto_peer_service_port = true;
    config.dram_page_size = kPageSize;
    config.cache_remote_fetches = false;
    config.placement_policy = PoolPlacementPolicy::WEIGHTED;

    for (uint32_t backend_id = 0; backend_id < 2; ++backend_id) {
      BackendInstanceConfig backend;
      backend.name = "dram-" + std::to_string(backend_id);
      backend.tier = TierType::DRAM;
      backend.dram.buffer_sizes = {kBackendCapacity};
      backend.placement_weight = backend_id + 1;
      config.backends.push_back(std::move(backend));
    }

    auto client = std::make_unique<PoolClient>(std::move(config));
    ASSERT_TRUE(client->Init()) << "failed to initialize PoolClient " << client_id;
    clients_.push_back(std::move(client));
  }

  std::unique_ptr<MasterServer> master_;
  std::thread master_thread_;
  std::vector<std::unique_ptr<PoolClient>> clients_;
};

TEST_F(TierBenchmarkSmokeTest, ReplaysValidatedBatchesAcrossWeightedDramBackends) {
  constexpr uint64_t kPayloadSeed = 0x5eed;
  VectorSource source(MakeReplay(), kPayloadSeed);
  PoolWorkloadClient adapter(&clients_);
  bench::WorkloadRunnerOptions options;
  options.max_throughput = true;
  options.validate_get_payloads = true;
  bench::WorkloadRunner runner(&adapter, options);

  const bench::WorkloadMetrics metrics = runner.Run(&source);

  EXPECT_EQ(metrics.total.attempted, 4 * kKeysPerClient);
  EXPECT_EQ(metrics.total.succeeded, metrics.total.attempted);
  EXPECT_EQ(metrics.total.failed, 0u);
  EXPECT_EQ(metrics.puts.succeeded, 2 * kKeysPerClient);
  EXPECT_EQ(metrics.gets.succeeded, 2 * kKeysPerClient);
  EXPECT_EQ(metrics.get_misses, 0u);
  EXPECT_EQ(metrics.get_validation_failures, 0u);
  for (uint32_t client_id = 0; client_id < 2; ++client_id) {
    EXPECT_EQ(adapter.BatchPutCalls(client_id), 1u);
    EXPECT_EQ(adapter.BatchGetCalls(client_id), 1u);
  }

  size_t populated_backends = 0;
  size_t owned_keys = 0;
  for (const auto& client : clients_) {
    ASSERT_EQ(client->Backends().All().size(), 2u);
    for (MediumBackend* backend : client->Backends().All()) {
      EXPECT_EQ(backend->Tier(), TierType::DRAM);
      const size_t backend_keys = backend->OwnedKeyCount();
      owned_keys += backend_keys;
      if (backend_keys != 0) ++populated_backends;
    }
  }
  EXPECT_EQ(owned_keys, 2 * kKeysPerClient);
  EXPECT_GE(populated_backends, 2u)
      << "the deterministic 24-key sample should exercise weighted placement";

  for (auto& client : clients_) ASSERT_TRUE(client->Clear());
  for (const auto& client : clients_) {
    for (MediumBackend* backend : client->Backends().All()) {
      EXPECT_EQ(backend->OwnedKeyCount(), 0u);
    }
  }
}

TEST_F(TierBenchmarkSmokeTest, ClientBReadsClientAKeysWithPayloadValidation) {
  constexpr size_t kRemoteKeys = 8;
  constexpr uint64_t kPayloadSeed = 0xA11CE;
  PoolClient* writer = clients_[0].get();
  PoolClient* reader = clients_[1].get();
  ASSERT_NE(writer, nullptr);
  ASSERT_NE(reader, nullptr);

  std::vector<std::string> keys;
  keys.reserve(kRemoteKeys);
  for (size_t i = 0; i < kRemoteKeys; ++i) {
    const std::string key = "tier-remote-" + std::to_string(i);
    auto payload = bench::GenerateDeterministicPayload(key, i + 1, kPayloadSeed, kPageSize);
    ASSERT_TRUE(writer->Put(key, payload.data(), payload.size())) << key;
    keys.push_back(key);
  }
  writer->Master().FlushHeartbeat();

  for (size_t i = 0; i < kRemoteKeys; ++i) {
    std::vector<uint8_t> got(kPageSize, 0);
    bool ok = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    do {
      if (reader->Get(keys[i], got.data(), got.size())) {
        ok = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(25));
    } while (std::chrono::steady_clock::now() < deadline);
    ASSERT_TRUE(ok) << "client B missed client A's key " << keys[i];
    EXPECT_TRUE(bench::ValidateDeterministicPayload(keys[i], i + 1, kPayloadSeed, got.data(),
                                                    got.size()));
  }
}

class TinyWeightedPoolTest : public ::testing::Test {
 protected:
  static constexpr size_t kTinyPages = 4;
  static constexpr size_t kTinyCapacity = kPageSize * kTinyPages;

  void SetUp() override {
    MasterServerConfig config;
    config.listen_address = "127.0.0.1:0";
    config.metrics_port = 0;
    config.registry_config.default_dram_page_size = kPageSize;
    config.registry_config.heartbeat_ttl = std::chrono::seconds(1);
    config.put_strategy = std::make_unique<ConfigurableRoutePutStrategy>(
        ConfigurableRoutePutStrategy::SelectAlgo::kMostAvailable,
        ConfigurableRoutePutStrategy::NodeAffinity::kLocal, 17);
    master_ = std::make_unique<MasterServer>(std::move(config));
    master_thread_ = std::thread([this] { master_->Run(); });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (master_->GetBoundPort() == 0 && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_NE(master_->GetBoundPort(), 0);

    PoolClientConfig client_config;
    client_config.master_config.master_address =
        "127.0.0.1:" + std::to_string(master_->GetBoundPort());
    client_config.master_config.node_id = "tier-tiny-node-0";
    client_config.master_config.node_address = "127.0.0.1";
    client_config.master_config.auto_heartbeat = true;
    client_config.io_engine.host = "127.0.0.1";
    client_config.io_engine.port = 0;
    client_config.auto_peer_service_port = true;
    client_config.dram_page_size = kPageSize;
    client_config.cache_remote_fetches = false;
    client_config.placement_policy = PoolPlacementPolicy::WEIGHTED;

    for (uint32_t backend_id = 0; backend_id < 2; ++backend_id) {
      BackendInstanceConfig backend;
      backend.name = "dram-" + std::to_string(backend_id);
      backend.tier = TierType::DRAM;
      backend.dram.buffer_sizes = {kTinyCapacity};
      backend.placement_weight = backend_id == 0 ? 100u : 1u;
      client_config.backends.push_back(std::move(backend));
    }

    client_ = std::make_unique<PoolClient>(std::move(client_config));
    ASSERT_TRUE(client_->Init());
    client_->Master().FlushHeartbeat();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  void TearDown() override {
    if (client_) {
      client_->Clear();
      client_->Shutdown();
    }
    client_.reset();
    if (master_) master_->Shutdown();
    if (master_thread_.joinable()) master_thread_.join();
    master_.reset();
  }

  std::unique_ptr<MasterServer> master_;
  std::thread master_thread_;
  std::unique_ptr<PoolClient> client_;
};

TEST_F(TinyWeightedPoolTest, FallsBackThenFailsWithoutLeakingSlots) {
  MediumBackend* primary = client_->Backends().Get("dram-0");
  MediumBackend* fallback = client_->Backends().Get("dram-1");
  ASSERT_NE(primary, nullptr);
  ASSERT_NE(fallback, nullptr);

  std::vector<uint8_t> value(kPageSize, 0x5A);
  size_t succeeded = 0;
  size_t failed = 0;
  constexpr size_t kAttempts = kTinyPages * 2 + 4;
  for (size_t i = 0; i < kAttempts; ++i) {
    const std::string key = "tiny-" + std::to_string(i);
    if (client_->Put(key, value.data(), value.size())) {
      ++succeeded;
    } else {
      ++failed;
    }
  }

  EXPECT_EQ(succeeded, 2 * kTinyPages);
  EXPECT_EQ(failed, 4u);
  EXPECT_EQ(primary->OwnedKeyCount(), kTinyPages)
      << "weight 100:1 should fill dram-0 first";
  EXPECT_EQ(fallback->OwnedKeyCount(), kTinyPages)
      << "no-space on the primary must fall back to dram-1";
  EXPECT_EQ(primary->OwnedKeyCount() + fallback->OwnedKeyCount(), succeeded);

  const auto primary_cap = primary->Capacity();
  const auto fallback_cap = fallback->Capacity();
  EXPECT_EQ(primary_cap.available_bytes, 0u);
  EXPECT_EQ(fallback_cap.available_bytes, 0u);
  EXPECT_EQ(primary_cap.max_allocatable_bytes, 0u);
  EXPECT_EQ(fallback_cap.max_allocatable_bytes, 0u);
  EXPECT_EQ(primary_cap.total_bytes, kTinyCapacity);
  EXPECT_EQ(fallback_cap.total_bytes, kTinyCapacity);
  EXPECT_EQ(primary_cap.available_bytes + primary->OwnedKeyCount() * kPageSize,
            primary_cap.total_bytes);
  EXPECT_EQ(fallback_cap.available_bytes + fallback->OwnedKeyCount() * kPageSize,
            fallback_cap.total_bytes);

  ASSERT_TRUE(client_->Clear());
  EXPECT_EQ(primary->OwnedKeyCount(), 0u);
  EXPECT_EQ(fallback->OwnedKeyCount(), 0u);
  EXPECT_EQ(primary->Capacity().available_bytes, kTinyCapacity);
  EXPECT_EQ(fallback->Capacity().available_bytes, kTinyCapacity);
}

class UnequalDramWeightedPoolTest : public ::testing::Test {
 protected:
  static constexpr size_t kPrimaryPages = 4;
  static constexpr size_t kSecondaryPages = 12;
  static constexpr size_t kPrimaryCapacity = kPageSize * kPrimaryPages;
  static constexpr size_t kSecondaryCapacity = kPageSize * kSecondaryPages;
  static constexpr size_t kPooledCapacity = kPrimaryCapacity + kSecondaryCapacity;
  static constexpr const char* kNodeId = "tier-cap-node-0";

  void StartCluster(bool advertise_max_allocatable_bytes) {
    MasterServerConfig config;
    config.listen_address = "127.0.0.1:0";
    config.metrics_port = 0;
    config.registry_config.default_dram_page_size = kPageSize;
    config.registry_config.heartbeat_ttl = std::chrono::seconds(1);
    config.registry_config.advertise_max_allocatable_bytes = advertise_max_allocatable_bytes;
    config.put_strategy = std::make_unique<ConfigurableRoutePutStrategy>(
        ConfigurableRoutePutStrategy::SelectAlgo::kMostAvailable,
        ConfigurableRoutePutStrategy::NodeAffinity::kLocal, 17);
    master_ = std::make_unique<MasterServer>(std::move(config));
    master_thread_ = std::thread([this] { master_->Run(); });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (master_->GetBoundPort() == 0 && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_NE(master_->GetBoundPort(), 0);

    PoolClientConfig client_config;
    client_config.master_config.master_address =
        "127.0.0.1:" + std::to_string(master_->GetBoundPort());
    client_config.master_config.node_id = kNodeId;
    client_config.master_config.node_address = "127.0.0.1";
    client_config.master_config.auto_heartbeat = true;
    client_config.io_engine.host = "127.0.0.1";
    client_config.io_engine.port = 0;
    client_config.auto_peer_service_port = true;
    client_config.dram_page_size = kPageSize;
    client_config.cache_remote_fetches = false;
    client_config.placement_policy = PoolPlacementPolicy::WEIGHTED;

    BackendInstanceConfig primary;
    primary.name = "dram-0";
    primary.tier = TierType::DRAM;
    primary.dram.buffer_sizes = {kPrimaryCapacity};
    primary.placement_weight = 1;
    client_config.backends.push_back(std::move(primary));

    BackendInstanceConfig secondary;
    secondary.name = "dram-1";
    secondary.tier = TierType::DRAM;
    secondary.dram.buffer_sizes = {kSecondaryCapacity};
    secondary.placement_weight = 1;
    client_config.backends.push_back(std::move(secondary));

    client_ = std::make_unique<PoolClient>(std::move(client_config));
    ASSERT_TRUE(client_->Init());
    client_->Master().FlushHeartbeat();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    client_->Master().FlushHeartbeat();
  }

  void TearDown() override {
    if (client_) {
      client_->Clear();
      client_->Shutdown();
    }
    client_.reset();
    if (master_) master_->Shutdown();
    if (master_thread_.joinable()) master_thread_.join();
    master_.reset();
  }

  size_t FillLocalPool(const std::string& key_prefix) {
    std::vector<uint8_t> value(kPageSize, 0x3C);
    size_t succeeded = 0;
    const size_t attempts = kPrimaryPages + kSecondaryPages;
    for (size_t i = 0; i < attempts; ++i) {
      if (client_->Put(key_prefix + std::to_string(i), value.data(), value.size())) ++succeeded;
    }
    return succeeded;
  }

  std::unique_ptr<MasterServer> master_;
  std::thread master_thread_;
  std::unique_ptr<PoolClient> client_;
};

class NewMasterWeightedPoolTest : public UnequalDramWeightedPoolTest {
 protected:
  void SetUp() override { StartCluster(/*advertise_max_allocatable_bytes=*/true); }
};

class LegacyMasterWeightedPoolTest : public UnequalDramWeightedPoolTest {
 protected:
  void SetUp() override { StartCluster(/*advertise_max_allocatable_bytes=*/false); }
};

TEST_F(NewMasterWeightedPoolTest, NegotiatesMaxAllocatableAndSumsSameTierCapacity) {
  EXPECT_TRUE(client_->Master().SupportsMaxAllocatableCapacity())
      << "current Master must advertise supports_max_allocatable_bytes so weighted "
         "peers can aggregate same-tier capacity";

  MediumBackend* primary = client_->Backends().Get("dram-0");
  MediumBackend* secondary = client_->Backends().Get("dram-1");
  ASSERT_NE(primary, nullptr);
  ASSERT_NE(secondary, nullptr);
  EXPECT_EQ(primary->Capacity().total_bytes + secondary->Capacity().total_bytes, kPooledCapacity);

  EXPECT_EQ(FillLocalPool("new-master-"), kPrimaryPages + kSecondaryPages);
  EXPECT_GT(primary->OwnedKeyCount(), 0u);
  EXPECT_GT(secondary->OwnedKeyCount(), 0u);

  client_->Master().FlushHeartbeat();
  EXPECT_TRUE(WaitForAdvertisedDram(master_.get(), kNodeId, kPooledCapacity, /*available=*/0))
      << "new Master must store the aggregated same-tier pool, not the first instance";
  EXPECT_NE(AdvertisedDramTotal(master_.get(), kNodeId), kPrimaryCapacity);
}

TEST_F(LegacyMasterWeightedPoolTest, KeepsFirstInstanceCapacityAndStillPlacesOnBothBackends) {
  EXPECT_FALSE(client_->Master().SupportsMaxAllocatableCapacity())
      << "omitting supports_max_allocatable_bytes must disable aggregate heartbeats";

  MediumBackend* primary = client_->Backends().Get("dram-0");
  MediumBackend* secondary = client_->Backends().Get("dram-1");
  ASSERT_NE(primary, nullptr);
  ASSERT_NE(secondary, nullptr);

  EXPECT_EQ(FillLocalPool("legacy-"), kPrimaryPages + kSecondaryPages);
  EXPECT_GT(primary->OwnedKeyCount(), 0u);
  EXPECT_GT(secondary->OwnedKeyCount(), 0u);
  EXPECT_EQ(primary->OwnedKeyCount() + secondary->OwnedKeyCount(), kPrimaryPages + kSecondaryPages)
      << "weighted placement remains enabled under a legacy Master; only advertisement shrinks";

  client_->Master().FlushHeartbeat();
  EXPECT_TRUE(WaitForAdvertisedDram(master_.get(), kNodeId, kPrimaryCapacity, /*available=*/0))
      << "heartbeats stay at the first same-tier instance so the old Master cannot "
         "over-admit a value larger than any single backend";
  EXPECT_EQ(AdvertisedDramTotal(master_.get(), kNodeId), kPrimaryCapacity);
  EXPECT_NE(AdvertisedDramTotal(master_.get(), kNodeId), kPooledCapacity);
}

}  // namespace
}  // namespace mori::umbp
