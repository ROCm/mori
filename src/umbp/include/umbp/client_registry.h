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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "umbp/types.h"

namespace mori::umbp {

class BlockIndex;

struct AllocateResult {
  std::string peer_address;
  std::vector<uint8_t> engine_desc_bytes;
  std::vector<uint8_t> dram_memory_desc_bytes;
  uint64_t allocated_offset = 0;
  uint32_t buffer_index = 0;
};

struct ClientIOInfo {
  std::string peer_address;
  std::vector<uint8_t> engine_desc_bytes;
  std::vector<uint8_t> dram_memory_desc_bytes;
};

struct ClientRegistryConfig {
  std::chrono::seconds heartbeat_ttl{10};
  std::chrono::seconds reaper_interval{5};
  uint32_t max_missed_heartbeats = 3;
};

class ClientRegistry {
 public:
  explicit ClientRegistry(const ClientRegistryConfig& config);
  ClientRegistry(const ClientRegistryConfig& config, BlockIndex& index);
  ~ClientRegistry();

  ClientRegistry(const ClientRegistry&) = delete;
  ClientRegistry& operator=(const ClientRegistry&) = delete;

  void SetBlockIndex(BlockIndex* index);

  // --- Client lifecycle ---
  // Returns false when a live node with the same id already exists.
  // Returns true for new registrations or re-registration of expired nodes.
  bool RegisterClient(const std::string& node_id, const std::string& node_address,
                      const std::map<TierType, TierCapacity>& tier_capacities,
                      const std::string& peer_address = "",
                      const std::vector<uint8_t>& engine_desc_bytes = {},
                      const std::vector<std::vector<uint8_t>>& dram_memory_desc_bytes_list = {},
                      const std::vector<uint64_t>& dram_buffer_sizes = {},
                      const std::vector<uint64_t>& ssd_store_capacities = {});

  // Gracefully unregister. Returns number of block keys cleaned up.
  size_t UnregisterClient(const std::string& node_id);

  // Process heartbeat. Updates last_heartbeat and tier capacities.
  // Returns CLIENT_STATUS_UNKNOWN if node is not registered.
  // PA-3 fix: uses exclusive lock since it mutates record fields.
  ClientStatus Heartbeat(const std::string& node_id,
                         const std::map<TierType, TierCapacity>& tier_capacities);

  // --- Ownership tracking (called by BlockIndex) ---
  void TrackKey(const std::string& node_id, const std::string& key);
  void UntrackKey(const std::string& node_id, const std::string& key);

  // --- PoolClient allocation ---
  std::optional<AllocateResult> AllocateForPut(const std::string& node_id, TierType tier,
                                               uint64_t size);
  void DeallocateForUnregister(const std::string& node_id, TierType tier,
                               uint32_t buffer_index, uint64_t offset, uint64_t size);
  std::optional<ClientIOInfo> GetClientIOInfo(const std::string& node_id,
                                              uint32_t buffer_index = 0) const;

  // --- Queries ---
  bool IsClientAlive(const std::string& node_id) const;
  size_t ClientCount() const;

  // Returns all clients with status == ALIVE. Used by Router for RoutePut.
  std::vector<ClientRecord> GetAliveClients() const;

  // --- Reaper control ---
  void StartReaper();
  void StopReaper();

 private:
  ClientRegistryConfig config_;
  BlockIndex* index_ = nullptr;

  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, ClientRecord> clients_;
  std::unordered_map<std::string, std::set<std::string>> client_keys_;

  // Reaper thread
  std::thread reaper_thread_;
  std::atomic<bool> reaper_running_{false};
  std::mutex reaper_cv_mutex_;
  std::condition_variable reaper_cv_;

  void ReaperLoop();
  // PA-4 fix: uses iterator-safe erase pattern.
  void ReapExpiredClients();

  std::chrono::seconds ExpiryDuration() const {
    return config_.heartbeat_ttl * config_.max_missed_heartbeats;
  }
};

}  // namespace mori::umbp
