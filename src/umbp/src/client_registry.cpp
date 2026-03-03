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
#include "umbp/client_registry.h"

#include <spdlog/spdlog.h>

namespace mori::umbp {

ClientRegistry::ClientRegistry(const ClientRegistryConfig& config) : config_(config) {}

ClientRegistry::~ClientRegistry() { StopReaper(); }

void ClientRegistry::RegisterClient(const std::string& client_id, const std::string& node_address,
                                    const std::map<TierType, TierCapacity>& tier_capacities) {
  std::unique_lock lock(mutex_);
  auto now = std::chrono::steady_clock::now();

  ClientRecord record;
  record.client_id = client_id;
  record.node_address = node_address;
  record.status = ClientStatus::ALIVE;
  record.last_heartbeat = now;
  record.registered_at = now;
  record.tier_capacities = tier_capacities;

  clients_[client_id] = std::move(record);
  client_keys_[client_id];  // ensure entry exists (empty set)

  spdlog::info("[Registry] Registered client: {} at {}", client_id, node_address);
}

size_t ClientRegistry::UnregisterClient(const std::string& client_id) {
  std::unique_lock lock(mutex_);
  auto it = clients_.find(client_id);
  if (it == clients_.end()) {
    return 0;
  }

  // Count tracked keys (will be used for BlockIndex cleanup later)
  size_t keys_removed = 0;
  auto keys_it = client_keys_.find(client_id);
  if (keys_it != client_keys_.end()) {
    keys_removed = keys_it->second.size();
    // Future: iterate keys and call BlockIndex::UnregisterByNode for each
    client_keys_.erase(keys_it);
  }

  clients_.erase(it);
  spdlog::info("[Registry] Unregistered client: {} (keys_removed={})", client_id, keys_removed);
  return keys_removed;
}

// PA-3 fix: exclusive lock because we mutate last_heartbeat and tier_capacities
ClientStatus ClientRegistry::Heartbeat(const std::string& client_id,
                                       const std::map<TierType, TierCapacity>& tier_capacities) {
  std::unique_lock lock(mutex_);
  auto it = clients_.find(client_id);
  if (it == clients_.end()) {
    spdlog::warn("[Registry] Heartbeat from unknown client: {}", client_id);
    return ClientStatus::UNKNOWN;
  }

  it->second.last_heartbeat = std::chrono::steady_clock::now();
  it->second.tier_capacities = tier_capacities;
  it->second.status = ClientStatus::ALIVE;

  return ClientStatus::ALIVE;
}

void ClientRegistry::TrackKey(const std::string& /*client_id*/, const std::string& /*key*/) {
  // Stub: will be used when BlockIndex is integrated
}

void ClientRegistry::UntrackKey(const std::string& /*client_id*/, const std::string& /*key*/) {
  // Stub: will be used when BlockIndex is integrated
}

bool ClientRegistry::IsClientAlive(const std::string& client_id) const {
  std::shared_lock lock(mutex_);
  auto it = clients_.find(client_id);
  return it != clients_.end() && it->second.status == ClientStatus::ALIVE;
}

size_t ClientRegistry::ClientCount() const {
  std::shared_lock lock(mutex_);
  return clients_.size();
}

std::vector<ClientRecord> ClientRegistry::GetAliveClients() const {
  std::shared_lock lock(mutex_);
  std::vector<ClientRecord> result;
  for (const auto& [id, record] : clients_) {
    if (record.status == ClientStatus::ALIVE) {
      result.push_back(record);
    }
  }
  return result;
}

void ClientRegistry::StartReaper() {
  reaper_running_ = true;
  reaper_thread_ = std::thread(&ClientRegistry::ReaperLoop, this);
  spdlog::info("[Reaper] Started (interval={}s, expiry={}s)", config_.reaper_interval.count(),
               ExpiryDuration().count());
}

void ClientRegistry::StopReaper() {
  if (reaper_running_) {
    reaper_running_ = false;
    reaper_cv_.notify_one();
    if (reaper_thread_.joinable()) {
      reaper_thread_.join();
    }
    spdlog::info("[Reaper] Stopped");
  }
}

void ClientRegistry::ReaperLoop() {
  while (reaper_running_) {
    {
      std::unique_lock cv_lock(reaper_cv_mutex_);
      reaper_cv_.wait_for(cv_lock, config_.reaper_interval,
                          [this] { return !reaper_running_.load(); });
    }
    if (!reaper_running_) {
      break;
    }
    ReapExpiredClients();
  }
}

// PA-4 fix: iterator-safe erase (never erase during range-for)
void ClientRegistry::ReapExpiredClients() {
  auto now = std::chrono::steady_clock::now();
  auto expiry = ExpiryDuration();

  std::unique_lock lock(mutex_);
  auto it = clients_.begin();
  while (it != clients_.end()) {
    if (now - it->second.last_heartbeat > expiry) {
      const std::string& dead_id = it->first;
      spdlog::warn("[Reaper] Reaping expired client: {}", dead_id);

      // Future: iterate client_keys_[dead_id] and call
      // BlockIndex::UnregisterByNode for each key
      client_keys_.erase(dead_id);
      it = clients_.erase(it);  // returns next valid iterator
    } else {
      ++it;
    }
  }
}

}  // namespace mori::umbp
