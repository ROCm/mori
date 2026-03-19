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

#include <algorithm>

#include "umbp/block_index/global_block_index.h"

namespace mori::umbp {

ClientRegistry::ClientRegistry(const ClientRegistryConfig& config) : config_(config) {}

ClientRegistry::ClientRegistry(const ClientRegistryConfig& config, GlobalBlockIndex& index)
    : config_(config), index_(&index) {}

ClientRegistry::~ClientRegistry() { StopReaper(); }

void ClientRegistry::SetBlockIndex(GlobalBlockIndex* index) {
  std::unique_lock lock(mutex_);
  index_ = index;
}

bool ClientRegistry::RegisterClient(
    const std::string& node_id, const std::string& node_address,
    const std::map<TierType, TierCapacity>& tier_capacities, const std::string& peer_address,
    const std::vector<uint8_t>& engine_desc_bytes,
    const std::vector<std::vector<uint8_t>>& dram_memory_desc_bytes_list,
    const std::vector<uint64_t>& dram_buffer_sizes,
    const std::vector<uint64_t>& ssd_store_capacities) {
  std::unique_lock lock(mutex_);
  auto now = std::chrono::steady_clock::now();

  auto it = clients_.find(node_id);
  if (it != clients_.end()) {
    const bool is_expired = (now - it->second.last_heartbeat > ExpiryDuration()) ||
                            (it->second.status == ClientStatus::EXPIRED);

    if (it->second.status == ClientStatus::ALIVE && !is_expired) {
      spdlog::warn("[Registry] Rejecting re-registration for alive node: {}", node_id);
      return false;
    }

    it->second.status = ClientStatus::EXPIRED;
    client_keys_.erase(node_id);
    spdlog::info("[Registry] Re-registering expired node: {}", node_id);
  }

  ClientRecord record;
  record.node_id = node_id;
  record.node_address = node_address;
  record.status = ClientStatus::ALIVE;
  record.last_heartbeat = now;
  record.registered_at = now;
  record.tier_capacities = tier_capacities;
  record.peer_address = peer_address;
  record.engine_desc_bytes = engine_desc_bytes;
  record.dram_memory_desc_bytes_list = dram_memory_desc_bytes_list;

  // Per-buffer DRAM allocators
  if (!dram_buffer_sizes.empty()) {
    for (size_t i = 0; i < dram_buffer_sizes.size(); ++i) {
      PoolAllocator alloc;
      alloc.total_size = dram_buffer_sizes[i];
      alloc.offset_tracker = PoolAllocator::OffsetTracker{};
      record.dram_allocators.push_back(std::move(alloc));
    }
  } else {
    // Backward compat: single allocator from tier_capacities (DRAM or HBM)
    for (auto check_tier : {TierType::HBM, TierType::DRAM}) {
      auto cap_it = tier_capacities.find(check_tier);
      if (cap_it != tier_capacities.end() && cap_it->second.total_bytes > 0) {
        PoolAllocator alloc;
        alloc.total_size = cap_it->second.total_bytes;
        alloc.offset_tracker = PoolAllocator::OffsetTracker{};
        record.dram_allocators.push_back(std::move(alloc));
      }
    }
  }

  // Per-store SSD allocators (capacity-only, no OffsetTracker)
  if (!ssd_store_capacities.empty()) {
    for (uint64_t cap : ssd_store_capacities) {
      PoolAllocator alloc;
      alloc.total_size = cap;
      record.ssd_allocators.push_back(std::move(alloc));
    }
  } else {
    // Backward compat: single allocator from tier_capacities
    auto ssd_it = tier_capacities.find(TierType::SSD);
    if (ssd_it != tier_capacities.end() && ssd_it->second.total_bytes > 0) {
      PoolAllocator alloc;
      alloc.total_size = ssd_it->second.total_bytes;
      record.ssd_allocators.push_back(std::move(alloc));
    }
  }

  clients_[node_id] = std::move(record);
  client_keys_[node_id];

  spdlog::info("[Registry] Registered node: {} at {} (dram_buffers={}, ssd_stores={})", node_id,
               node_address,
               dram_buffer_sizes.empty() ? (tier_capacities.count(TierType::DRAM) ? 1u : 0u)
                                         : static_cast<unsigned>(dram_buffer_sizes.size()),
               static_cast<unsigned>(ssd_store_capacities.empty()
                                         ? (tier_capacities.count(TierType::SSD) ? 1u : 0u)
                                         : ssd_store_capacities.size()));
  return true;
}

size_t ClientRegistry::UnregisterClient(const std::string& node_id) {
  size_t keys_removed = 0;
  std::vector<std::string> keys_to_cleanup;

  {
    std::unique_lock lock(mutex_);
    auto it = clients_.find(node_id);
    if (it == clients_.end()) {
      return 0;
    }

    auto keys_it = client_keys_.find(node_id);
    if (keys_it != client_keys_.end()) {
      keys_removed = keys_it->second.size();
      keys_to_cleanup.assign(keys_it->second.begin(), keys_it->second.end());
      client_keys_.erase(keys_it);
    }

    clients_.erase(it);
  }

  if (index_ != nullptr) {
    for (const auto& key : keys_to_cleanup) {
      index_->UnregisterByNode(key, node_id);
    }
  }

  spdlog::info("[Registry] Unregistered node: {} (keys_removed={})", node_id, keys_removed);
  return keys_removed;
}

// PA-3 fix: exclusive lock because we mutate last_heartbeat and tier_capacities
ClientStatus ClientRegistry::Heartbeat(const std::string& node_id,
                                       const std::map<TierType, TierCapacity>& tier_capacities) {
  std::unique_lock lock(mutex_);
  auto it = clients_.find(node_id);
  if (it == clients_.end()) {
    spdlog::warn("[Registry] Heartbeat from unknown node: {}", node_id);
    return ClientStatus::UNKNOWN;
  }

  it->second.last_heartbeat = std::chrono::steady_clock::now();
  it->second.tier_capacities = tier_capacities;
  it->second.status = ClientStatus::ALIVE;

  return ClientStatus::ALIVE;
}

void ClientRegistry::TrackKey(const std::string& node_id, const std::string& key) {
  std::unique_lock lock(mutex_);
  if (clients_.find(node_id) == clients_.end()) {
    return;
  }

  if (index_ != nullptr) {
    const auto locations = index_->Lookup(key);
    const bool owns_key =
        std::any_of(locations.begin(), locations.end(),
                    [&node_id](const Location& location) { return location.node_id == node_id; });
    if (!owns_key) {
      return;
    }
  }

  client_keys_[node_id].insert(key);
}

void ClientRegistry::UntrackKey(const std::string& node_id, const std::string& key) {
  std::unique_lock lock(mutex_);
  auto it = client_keys_.find(node_id);
  if (it == client_keys_.end()) {
    return;
  }

  if (index_ != nullptr) {
    const auto locations = index_->Lookup(key);
    const bool still_owns_key =
        std::any_of(locations.begin(), locations.end(),
                    [&node_id](const Location& location) { return location.node_id == node_id; });
    if (still_owns_key) {
      return;
    }
  }

  it->second.erase(key);
  if (it->second.empty()) {
    client_keys_.erase(it);
  }
}

bool ClientRegistry::IsClientAlive(const std::string& node_id) const {
  std::shared_lock lock(mutex_);
  auto it = clients_.find(node_id);
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

std::optional<AllocateResult> ClientRegistry::AllocateForPut(const std::string& node_id,
                                                             TierType tier, uint64_t size) {
  std::unique_lock lock(mutex_);
  auto it = clients_.find(node_id);
  if (it == clients_.end() || it->second.status != ClientStatus::ALIVE) {
    return std::nullopt;
  }

  auto& record = it->second;

  if (tier == TierType::DRAM || tier == TierType::HBM) {
    for (uint32_t i = 0; i < record.dram_allocators.size(); ++i) {
      auto offset = record.dram_allocators[i].Allocate(size);
      if (offset) {
        uint64_t total_avail = 0;
        for (auto& a : record.dram_allocators) total_avail += a.AvailableBytes();
        record.tier_capacities[tier].available_bytes = total_avail;

        AllocateResult result;
        result.peer_address = record.peer_address;
        result.engine_desc_bytes = record.engine_desc_bytes;
        if (i < record.dram_memory_desc_bytes_list.size())
          result.dram_memory_desc_bytes = record.dram_memory_desc_bytes_list[i];
        result.allocated_offset = *offset;
        result.buffer_index = i;
        return result;
      }
    }
    return std::nullopt;
  }

  if (tier == TierType::SSD) {
    for (uint32_t i = 0; i < record.ssd_allocators.size(); ++i) {
      auto offset = record.ssd_allocators[i].Allocate(size);
      if (offset) {
        uint64_t total_avail = 0;
        for (auto& a : record.ssd_allocators) total_avail += a.AvailableBytes();
        record.tier_capacities[tier].available_bytes = total_avail;

        AllocateResult result;
        result.peer_address = record.peer_address;
        result.engine_desc_bytes = record.engine_desc_bytes;
        if (!record.dram_memory_desc_bytes_list.empty())
          result.dram_memory_desc_bytes = record.dram_memory_desc_bytes_list[0];
        result.allocated_offset = 0;
        result.buffer_index = i;
        return result;
      }
    }
    return std::nullopt;
  }

  return std::nullopt;
}

void ClientRegistry::DeallocateForUnregister(const std::string& node_id, TierType tier,
                                             uint32_t buffer_index, uint64_t offset,
                                             uint64_t size) {
  std::unique_lock lock(mutex_);
  auto it = clients_.find(node_id);
  if (it == clients_.end()) {
    return;
  }

  auto& record = it->second;

  if (tier == TierType::DRAM || tier == TierType::HBM) {
    if (buffer_index < record.dram_allocators.size()) {
      record.dram_allocators[buffer_index].Deallocate(offset, size);
      uint64_t total_avail = 0;
      for (auto& a : record.dram_allocators) total_avail += a.AvailableBytes();
      record.tier_capacities[tier].available_bytes = total_avail;
    }
  } else if (tier == TierType::SSD) {
    if (buffer_index < record.ssd_allocators.size()) {
      record.ssd_allocators[buffer_index].Deallocate(offset, size);
      uint64_t total_avail = 0;
      for (auto& a : record.ssd_allocators) total_avail += a.AvailableBytes();
      record.tier_capacities[tier].available_bytes = total_avail;
    }
  }
}

std::optional<ClientIOInfo> ClientRegistry::GetClientIOInfo(const std::string& node_id,
                                                            uint32_t buffer_index) const {
  std::shared_lock lock(mutex_);
  auto it = clients_.find(node_id);
  if (it == clients_.end() || it->second.status != ClientStatus::ALIVE) {
    return std::nullopt;
  }

  ClientIOInfo info;
  info.peer_address = it->second.peer_address;
  info.engine_desc_bytes = it->second.engine_desc_bytes;
  if (buffer_index < it->second.dram_memory_desc_bytes_list.size()) {
    info.dram_memory_desc_bytes = it->second.dram_memory_desc_bytes_list[buffer_index];
  } else if (!it->second.dram_memory_desc_bytes_list.empty()) {
    info.dram_memory_desc_bytes = it->second.dram_memory_desc_bytes_list[0];
  }
  return info;
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
  std::vector<std::pair<std::string, std::vector<std::string>>> reap_cleanup;

  {
    std::unique_lock lock(mutex_);
    auto it = clients_.begin();
    while (it != clients_.end()) {
      if (now - it->second.last_heartbeat > expiry) {
        const std::string dead_id = it->first;
        spdlog::warn("[Reaper] Reaping expired client: {}", dead_id);

        std::vector<std::string> keys_to_cleanup;
        auto keys_it = client_keys_.find(dead_id);
        if (keys_it != client_keys_.end()) {
          keys_to_cleanup.assign(keys_it->second.begin(), keys_it->second.end());
          client_keys_.erase(keys_it);
        }

        reap_cleanup.emplace_back(dead_id, std::move(keys_to_cleanup));
        it = clients_.erase(it);  // returns next valid iterator
      } else {
        ++it;
      }
    }
  }

  if (index_ != nullptr) {
    for (const auto& [dead_id, keys_to_cleanup] : reap_cleanup) {
      for (const auto& key : keys_to_cleanup) {
        index_->UnregisterByNode(key, dead_id);
      }
    }
  }
}

}  // namespace mori::umbp
