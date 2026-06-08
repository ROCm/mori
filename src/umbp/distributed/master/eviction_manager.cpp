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
                                 const EvictionConfig& config, EvictKeyDispatcher* dispatcher)
    : index_(index), registry_(registry), config_(config), dispatcher_(dispatcher) {}

EvictionManager::~EvictionManager() { Stop(); }

void EvictionManager::Start() {
  if (running_.load(std::memory_order_relaxed)) return;
  running_.store(true, std::memory_order_relaxed);
  thread_ = std::thread(&EvictionManager::EvictionLoop, this);
  MORI_UMBP_INFO("[EvictionManager] Started (interval={}s, high={}, low={})",
                 config_.check_interval.count(), config_.high_watermark, config_.low_watermark);
}

void EvictionManager::Stop() {
  if (!running_.load(std::memory_order_relaxed)) return;
  running_.store(false, std::memory_order_relaxed);
  cv_.notify_one();
  if (thread_.joinable()) thread_.join();
  MORI_UMBP_INFO("[EvictionManager] Stopped");
}

void EvictionManager::EvictionLoop() {
  while (running_.load(std::memory_order_relaxed)) {
    {
      std::unique_lock lock(cv_mutex_);
      cv_.wait_for(lock, config_.check_interval,
                   [this] { return !running_.load(std::memory_order_relaxed); });
    }
    if (!running_.load(std::memory_order_relaxed)) break;
    RunOnce();
  }
}

// Master decides what to evict but the peer executes — master's view of the
// index only changes when the peer ships REMOVE events on the next heartbeat.
// This function picks victims and dispatches EvictKey to each peer via the
// dispatcher; master state itself is left untouched here.
void EvictionManager::RunOnce() {
  auto clients = registry_.GetAliveClients();

  using NodeTierKey = GlobalBlockIndex::NodeTierKey;
  std::set<NodeTierKey> overloaded_node_tiers;
  std::unordered_map<std::string, std::map<TierType, int64_t>> bytes_to_free;

  for (const auto& client : clients) {
    for (const auto& [tier, cap] : client.tier_capacities) {
      // SSD eviction is purely peer-local.  Master must NOT turn an SSD
      // overload into an EvictKey: EvictKey only acts on the peer's
      // PeerDramAllocator, so it would wrongly evict the DRAM copy of the same
      // key while leaving SSD untouched.
      if (tier == TierType::SSD) continue;
      if (cap.total_bytes == 0) continue;
      uint64_t used = cap.total_bytes - cap.available_bytes;
      double usage = static_cast<double>(used) / static_cast<double>(cap.total_bytes);
      if (usage >= config_.high_watermark) {
        auto target_used =
            static_cast<uint64_t>(static_cast<double>(cap.total_bytes) * config_.low_watermark);
        auto to_free = static_cast<int64_t>(used) - static_cast<int64_t>(target_used);
        if (to_free > 0) {
          overloaded_node_tiers.insert({client.node_id, tier});
          bytes_to_free[client.node_id][tier] += to_free;
        }
      }
    }
  }

  if (overloaded_node_tiers.empty()) return;
  MORI_UMBP_INFO("[EvictionManager] {} overloaded node-tiers detected",
                 overloaded_node_tiers.size());

  auto candidates = index_.FindEvictionCandidates(overloaded_node_tiers);
  if (candidates.empty()) {
    MORI_UMBP_DEBUG("[EvictionManager] No eviction candidates found");
    return;
  }

  // Sort by oldest-access first (LRU).  Depth-aware tiebreaking went away
  // along with master's per-key depth field — peers don't ship depth in
  // KvEvent — so a pure LRU sort is what we get.
  std::sort(candidates.begin(), candidates.end(),
            [](const EvictionCandidate& a, const EvictionCandidate& b) {
              return a.last_accessed_at < b.last_accessed_at;
            });

  // Group selected victims by node so the eventual EvictKey RPC takes a
  // single keys[] per peer instead of N round trips.
  std::unordered_map<std::string, std::vector<std::string>> per_node_keys;
  size_t selected = 0;
  for (const auto& c : candidates) {
    auto& tier_budget = bytes_to_free[c.location.node_id];
    auto it = tier_budget.find(c.location.tier);
    if (it == tier_budget.end() || it->second <= 0) continue;
    per_node_keys[c.location.node_id].push_back(c.key);
    it->second -= static_cast<int64_t>(c.size);
    ++selected;
  }

  if (selected == 0) return;

  MORI_UMBP_INFO("[EvictionManager] Selected {} victims across {} nodes", selected,
                 per_node_keys.size());

  // Look up peer addresses once per dispatch round.  ClientRegistry
  // owns the (node_id -> peer_address) mapping; we can't ship an
  // EvictKey to a node that has dropped out.
  std::unordered_map<std::string, std::string> node_to_peer;
  for (const auto& client : clients) node_to_peer[client.node_id] = client.peer_address;

  for (auto& [node_id, keys] : per_node_keys) {
    auto it = node_to_peer.find(node_id);
    if (it == node_to_peer.end() || it->second.empty()) {
      MORI_UMBP_WARN("[EvictionManager] no peer_address for node={} — skipping {} keys", node_id,
                     keys.size());
      continue;
    }
    if (dispatcher_ == nullptr) {
      MORI_UMBP_DEBUG("[EvictionManager] dispatcher unset; would EvictKey on node={} ({} keys)",
                      node_id, keys.size());
      continue;
    }
    // Master state is unchanged here — REMOVE events on the peer's
    // next heartbeat are what shrink the index.  Re-eviction next
    // round is safe because peer Evict is idempotent.
    dispatcher_->DispatchEvictKey(node_id, it->second, std::move(keys));
  }
}

}  // namespace mori::umbp
