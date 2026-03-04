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
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "umbp/types.h"

namespace mori::umbp {

struct ClientRegistryConfig {
  std::chrono::seconds heartbeat_ttl{10};
  std::chrono::seconds reaper_interval{5};
  uint32_t max_missed_heartbeats = 3;
};

class ClientRegistry {
 public:
  explicit ClientRegistry(const ClientRegistryConfig& config);
  ~ClientRegistry();

  ClientRegistry(const ClientRegistry&) = delete;
  ClientRegistry& operator=(const ClientRegistry&) = delete;

  // --- Client lifecycle ---
  // Returns false when a live client with the same id already exists.
  // Returns true for new registrations or re-registration of expired clients.
  bool RegisterClient(const std::string& client_id, const std::string& node_address,
                      const std::map<TierType, TierCapacity>& tier_capacities);

  // Gracefully unregister. Returns number of block keys cleaned up.
  size_t UnregisterClient(const std::string& client_id);

  // Process heartbeat. Updates last_heartbeat and tier capacities.
  // Returns CLIENT_STATUS_UNKNOWN if client is not registered.
  // PA-3 fix: uses exclusive lock since it mutates record fields.
  ClientStatus Heartbeat(const std::string& client_id,
                         const std::map<TierType, TierCapacity>& tier_capacities);

  // --- Ownership tracking stubs (for future BlockIndex integration) ---
  void TrackKey(const std::string& client_id, const std::string& key);
  void UntrackKey(const std::string& client_id, const std::string& key);

  // --- Queries ---
  bool IsClientAlive(const std::string& client_id) const;
  size_t ClientCount() const;

  // Returns all clients with status == ALIVE. Used by Router for RoutePut.
  std::vector<ClientRecord> GetAliveClients() const;

  // --- Reaper control ---
  void StartReaper();
  void StopReaper();

 private:
  ClientRegistryConfig config_;

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
