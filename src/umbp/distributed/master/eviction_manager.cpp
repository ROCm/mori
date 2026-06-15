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

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/master/evict_strategy.h"
#include "umbp/distributed/master/master_metadata_store.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

EvictionManager::EvictionManager(IMasterMetadataStore& store, const EvictionConfig& config,
                                 EvictKeyDispatcher* dispatcher,
                                 std::unique_ptr<MasterEvictStrategy> strategy)
    : store_(store),
      config_(config),
      dispatcher_(dispatcher),
      strategy_(strategy ? std::move(strategy) : std::make_unique<LruMasterEvictStrategy>()) {}

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
  auto clients = store_.ListAliveClients();

  // Per-(node, tier) byte budget down to the LOW watermark.  The budget map's
  // keys double as the set of overloaded buckets to enumerate; the byte values
  // are handed to the strategy (never to the store) for the actual victim
  // decision.
  std::map<NodeTierKey, uint64_t> bytes_to_free;

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
        if (used > target_used) {
          bytes_to_free[{client.node_id, tier}] += used - target_used;
        }
      }
    }
  }

  if (bytes_to_free.empty()) return;
  MORI_UMBP_INFO("[EvictionManager] {} overloaded node-tiers detected", bytes_to_free.size());

  // Ask the store for eviction-eligible candidates in the overloaded buckets.
  // The store is policy-neutral: it returns rows ordered by the hint (LRU here,
  // so an indexed backend can push the ordering down) but makes no eviction
  // decision and never sees the byte budget.  max_per_bucket=0 → no cap, so the
  // strategy gets full candidate visibility.
  std::vector<NodeTierKey> buckets;
  buckets.reserve(bytes_to_free.size());
  for (const auto& [ntk, _bytes] : bytes_to_free) buckets.push_back(ntk);

  auto candidates_by_bucket = store_.EnumerateEvictionCandidates(
      buckets, EvictionOrder::kLeastRecentlyAccessed, /*max_per_bucket=*/0,
      std::chrono::system_clock::now());
  if (candidates_by_bucket.empty()) {
    MORI_UMBP_DEBUG("[EvictionManager] No eviction candidates found");
    return;
  }

  // Flatten the candidates and translate the per-(node, tier) byte budget into
  // the strategy's shape, then let the strategy pick victims (default LRU),
  // grouped by node so each peer gets a single EvictKey keys[].
  std::vector<EvictionCandidate> candidates;
  for (auto& [bucket, rows] : candidates_by_bucket) {
    for (auto& c : rows) candidates.push_back(std::move(c));
  }
  std::unordered_map<std::string, std::map<TierType, int64_t>> strategy_budget;
  for (const auto& [ntk, bytes] : bytes_to_free) {
    strategy_budget[ntk.node_id][ntk.tier] = static_cast<int64_t>(bytes);
  }

  auto per_node_keys =
      strategy_->SelectVictims(std::move(candidates), std::move(strategy_budget));

  size_t selected = 0;
  for (const auto& [node_id, keys] : per_node_keys) selected += keys.size();

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
