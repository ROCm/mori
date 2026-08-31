// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/benchmark/workload_runner.h"
#include "umbp/distributed/pool_client.h"

namespace mori::umbp::benchmark {

struct PoolWorkloadClientOptions {
  // A key becomes visible to other clients only once the writer's next
  // heartbeat reaches the master, so a GET that arrives first has to wait
  // rather than count as a miss. The window applies from the PUT this adapter
  // performed.
  std::chrono::milliseconds publication_timeout{5000};
  // Window for keys this adapter never wrote, e.g. keys a separate writer put
  // there. Zero means a missing key is reported immediately.
  std::chrono::milliseconds unpublished_timeout{0};
  std::chrono::milliseconds retry_interval{5};
};

// Bridges the transport-neutral WorkloadClient the runner drives onto real
// PoolClients, mapping each workload client id to its own PoolClient so a
// multi-client run exercises routing and remote fetch rather than one process
// talking to itself.
class PoolWorkloadClient final : public WorkloadClient {
 public:
  explicit PoolWorkloadClient(std::vector<std::unique_ptr<PoolClient>>* clients,
                              PoolWorkloadClientOptions options = {})
      : clients_(clients),
        options_(options),
        batch_put_calls_(clients == nullptr ? 0 : clients->size()),
        batch_get_calls_(clients == nullptr ? 0 : clients->size()) {}

  ClientResult Put(uint32_t id, const std::string& key, const uint8_t* data,
                   size_t size) override {
    PoolClient* client = Select(id);
    if (client == nullptr || !client->Put(key, data, size)) return ClientResult::kFailed;
    Published(key);
    client->Master().FlushHeartbeat();
    return ClientResult::kSuccess;
  }

  ClientResult Get(uint32_t id, const std::string& key, uint8_t* data, size_t size) override {
    PoolClient* client = Select(id);
    if (client == nullptr) return ClientResult::kFailed;
    const auto deadline = Deadline(key);
    do {
      if (client->Get(key, data, size)) {
        Clear(key);
        return ClientResult::kSuccess;
      }
      std::this_thread::sleep_for(options_.retry_interval);
    } while (std::chrono::steady_clock::now() < deadline);
    Clear(key);
    return client->Exists(key) ? ClientResult::kFailed : ClientResult::kNotFound;
  }

  std::vector<ClientResult> BatchPut(
      uint32_t id, const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values) override {
    PoolClient* client = Select(id);
    if (client == nullptr || keys.size() != values.size()) {
      return std::vector<ClientResult>(keys.size(), ClientResult::kFailed);
    }
    Count(&batch_put_calls_, id);
    std::vector<const void*> pointers;
    std::vector<size_t> sizes;
    pointers.reserve(values.size());
    sizes.reserve(values.size());
    for (const auto& value : values) {
      pointers.push_back(value.data());
      sizes.push_back(value.size());
    }
    const auto results = client->BatchPut(keys, pointers, sizes);
    std::vector<ClientResult> converted(keys.size(), ClientResult::kFailed);
    for (size_t i = 0; i < keys.size() && i < results.size(); ++i) {
      if (results[i]) {
        converted[i] = ClientResult::kSuccess;
        Published(keys[i]);
      }
    }
    client->Master().FlushHeartbeat();
    return converted;
  }

  std::vector<ClientResult> BatchGet(uint32_t id, const std::vector<std::string>& keys,
                                     const std::vector<size_t>& sizes,
                                     std::vector<std::vector<uint8_t>>* values) override {
    PoolClient* client = Select(id);
    if (client == nullptr || keys.size() != sizes.size() || values == nullptr) {
      return std::vector<ClientResult>(keys.size(), ClientResult::kFailed);
    }
    Count(&batch_get_calls_, id);
    values->resize(keys.size());
    std::vector<void*> pointers;
    pointers.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      (*values)[i].assign(sizes[i], 0);
      pointers.push_back((*values)[i].data());
    }
    const auto results = client->BatchGet(keys, pointers, sizes);
    std::vector<ClientResult> converted(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      if (i < results.size() && results[i]) {
        Clear(keys[i]);
        converted[i] = ClientResult::kSuccess;
      } else {
        // Retry the stragglers one key at a time so a single unpublished key
        // does not re-fetch the whole batch.
        converted[i] = Get(id, keys[i], (*values)[i].data(), sizes[i]);
      }
    }
    return converted;
  }

  uint64_t BatchPutCalls(uint32_t id) const { return Calls(batch_put_calls_, id); }
  uint64_t BatchGetCalls(uint32_t id) const { return Calls(batch_get_calls_, id); }

 private:
  PoolClient* Select(uint32_t id) const {
    return clients_ != nullptr && id < clients_->size() ? (*clients_)[id].get() : nullptr;
  }
  static void Count(std::vector<std::atomic<uint64_t>>* counters, uint32_t id) {
    if (id < counters->size()) ++(*counters)[id];
  }
  static uint64_t Calls(const std::vector<std::atomic<uint64_t>>& counters, uint32_t id) {
    return id < counters.size() ? counters[id].load() : 0;
  }
  void Published(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_[key] = std::chrono::steady_clock::now() + options_.publication_timeout;
  }
  std::chrono::steady_clock::time_point Deadline(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = pending_.find(key);
    if (found != pending_.end()) return found->second;
    return std::chrono::steady_clock::now() + options_.unpublished_timeout;
  }
  void Clear(const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.erase(key);
  }

  std::vector<std::unique_ptr<PoolClient>>* clients_;
  PoolWorkloadClientOptions options_;
  std::vector<std::atomic<uint64_t>> batch_put_calls_;
  std::vector<std::atomic<uint64_t>> batch_get_calls_;
  std::mutex mutex_;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point> pending_;
};

}  // namespace mori::umbp::benchmark
