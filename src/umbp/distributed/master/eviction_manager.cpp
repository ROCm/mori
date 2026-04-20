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
#include "umbp/distributed/master/eviction_manager.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/master/client_registry.h"
#include "umbp/distributed/master/global_block_index.h"

namespace mori::umbp {

EvictionManager::EvictionManager(GlobalBlockIndex& index, ClientRegistry& registry,
                                 const EvictionConfig& config)
    : index_(index), registry_(registry), config_(config) {}

EvictionManager::~EvictionManager() { Stop(); }

void EvictionManager::Start() {
  if (running_.load(std::memory_order_relaxed)) {
    return;
  }
  running_.store(true, std::memory_order_relaxed);
  thread_ = std::thread(&EvictionManager::EvictionLoop, this);
  MORI_UMBP_INFO("[EvictionManager] Started (interval={}s, high={}, low={})",
                 config_.check_interval.count(), config_.high_watermark, config_.low_watermark);
}

void EvictionManager::Stop() {
  if (!running_.load(std::memory_order_relaxed)) {
    return;
  }
  running_.store(false, std::memory_order_relaxed);
  cv_.notify_one();
  if (thread_.joinable()) {
    thread_.join();
  }
  MORI_UMBP_INFO("[EvictionManager] Stopped");
}

void EvictionManager::EvictionLoop() {
  while (running_.load(std::memory_order_relaxed)) {
    {
      std::unique_lock lock(cv_mutex_);
      cv_.wait_for(lock, config_.check_interval,
                   [this] { return !running_.load(std::memory_order_relaxed); });
    }
    if (!running_.load(std::memory_order_relaxed)) {
      break;
    }
    RunOnce();
  }
}

void EvictionManager::RunOnce() {
  // Step 1: determine overloaded nodes
  auto clients = registry_.GetAliveClients();

  using NodeTierKey = GlobalBlockIndex::NodeTierKey;
  struct NodeTierHash {
    size_t operator()(const NodeTierKey& k) const {
      return std::hash<std::string>{}(k.node_id) ^
             (std::hash<int>{}(static_cast<int>(k.tier)) << 16);
    }
  };
  std::set<NodeTierKey> overloaded_node_tiers;
  std::unordered_map<NodeTierKey, int64_t, NodeTierHash> bytes_to_free;

  for (const auto& client : clients) {
    for (const auto& [tier, cap] : client.tier_capacities) {
      if (cap.total_bytes == 0) {
        continue;
      }
      uint64_t used = cap.total_bytes - cap.available_bytes;
      double usage = static_cast<double>(used) / static_cast<double>(cap.total_bytes);
      if (usage >= config_.high_watermark) {
        auto target_used =
            static_cast<uint64_t>(static_cast<double>(cap.total_bytes) * config_.low_watermark);
        auto to_free = static_cast<int64_t>(used) - static_cast<int64_t>(target_used);
        if (to_free > 0) {
          overloaded_node_tiers.insert({client.node_id, tier});
          bytes_to_free[{client.node_id, tier}] += to_free;
        }
      }
    }
  }

  if (overloaded_node_tiers.empty()) {
    return;
  }

  MORI_UMBP_INFO("[EvictionManager] {} overloaded node-tiers detected",
                 overloaded_node_tiers.size());

  auto candidates = index_.FindEvictionCandidates(overloaded_node_tiers);

  if (candidates.empty()) {
    MORI_UMBP_DEBUG("[EvictionManager] No eviction candidates found");
    return;
  }

  // Step 2: sort and select victims (no lock)
  std::sort(candidates.begin(), candidates.end(),
            [](const EvictionCandidate& a, const EvictionCandidate& b) {
              if (a.last_accessed_at != b.last_accessed_at) {
                return a.last_accessed_at < b.last_accessed_at;
              }
              return a.depth > b.depth;
            });

  std::vector<EvictionCandidate> victims;
  for (const auto& c : candidates) {
    NodeTierKey key{c.location.node_id, c.location.tier};
    auto it = bytes_to_free.find(key);
    if (it == bytes_to_free.end() || it->second <= 0) {
      continue;
    }
    victims.push_back(c);
    it->second -= static_cast<int64_t>(c.size);

    bool all_satisfied = true;
    for (const auto& [_, remaining] : bytes_to_free) {
      if (remaining > 0) {
        all_satisfied = false;
        break;
      }
    }
    if (all_satisfied) {
      break;
    }
  }

  if (victims.empty()) {
    return;
  }

  MORI_UMBP_INFO("[EvictionManager] Selected {} victims for eviction", victims.size());

  // Step 3: evict in batches (write lock + yield between batches)
  size_t total_evicted = 0;
  for (size_t i = 0; i < victims.size(); i += config_.evict_batch_size) {
    if (!running_.load(std::memory_order_relaxed)) {
      break;
    }

    size_t end = std::min(i + config_.evict_batch_size, victims.size());
    std::vector<EvictionCandidate> batch(victims.begin() + static_cast<ptrdiff_t>(i),
                                         victims.begin() + static_cast<ptrdiff_t>(end));

    auto evicted = index_.EvictEntries(batch);
    total_evicted += evicted.size();

    for (const auto& entry : evicted) {
      // ClientRegistry::DeallocateForUnregister parses location_id internally:
      // DRAM/HBM uses the page-bitmap "0:p3,4;1:p0" format via
      // ParseDramLocationId; SSD uses the buffer-index-prefix format.
      registry_.DeallocateForUnregister(entry.location.node_id, entry.location);
    }

    if (end < victims.size()) {
      std::this_thread::yield();
    }
  }

  MORI_UMBP_INFO("[EvictionManager] Evicted {} entries", total_evicted);
}

}  // namespace mori::umbp
