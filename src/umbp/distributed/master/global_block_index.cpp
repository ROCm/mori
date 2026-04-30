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
#include "umbp/distributed/master/global_block_index.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <set>
#include <unordered_set>

#include "umbp/distributed/master/client_registry.h"

namespace mori::umbp {

void GlobalBlockIndex::SetClientRegistry(ClientRegistry* registry) {
  std::unique_lock lock(mutex_);
  registry_ = registry;
}

void GlobalBlockIndex::Register(const std::string& node_id, const std::string& key,
                                const Location& location) {
  (void)BatchRegister(node_id, {{key, location}});
}

bool GlobalBlockIndex::Unregister(const std::string& node_id, const std::string& key,
                                  const Location& location) {
  return BatchUnregister(node_id, {{key, location}}) > 0;
}

size_t GlobalBlockIndex::UnregisterByNode(const std::string& key, const std::string& node_id) {
  size_t removed = 0;
  bool should_untrack = false;
  ClientRegistry* registry = nullptr;

  {
    std::unique_lock lock(mutex_);
    registry = registry_;

    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return 0;
    }

    auto& locs = it->second.locations;
    const size_t original_size = locs.size();
    locs.erase(std::remove_if(locs.begin(), locs.end(),
                              [&node_id](const Location& loc) { return loc.node_id == node_id; }),
               locs.end());
    removed = original_size - locs.size();
    should_untrack =
        removed > 0 && std::none_of(locs.begin(), locs.end(), [&node_id](const Location& loc) {
          return loc.node_id == node_id;
        });

    if (locs.empty()) {
      entries_.erase(it);
    }
  }

  if (removed > 0 && should_untrack && registry != nullptr) {
    registry->UntrackKey(node_id, key);
  }

  return removed;
}

size_t GlobalBlockIndex::BatchRegister(
    const std::string& node_id, const std::vector<std::pair<std::string, Location>>& entries) {
  if (entries.empty()) {
    return 0;
  }

  size_t inserted = 0;
  std::vector<std::string> keys_to_track;
  std::unordered_set<std::string> keys_seen;
  ClientRegistry* registry = nullptr;

  {
    std::unique_lock lock(mutex_);
    registry = registry_;

    for (const auto& [key, location] : entries) {
      auto& entry = entries_[key];
      const auto now = std::chrono::steady_clock::now();
      if (entry.locations.empty()) {
        entry.metrics.created_at = now;
        entry.metrics.last_accessed_at = now;
        entry.metrics.access_count = 0;
        entry.last_accessed_rep.store(now.time_since_epoch().count(), std::memory_order_release);
        entry.atomic_access_count.store(0, std::memory_order_relaxed);
      }

      auto it = std::find(entry.locations.begin(), entry.locations.end(), location);
      if (it != entry.locations.end()) {
        continue;
      }

      entry.locations.push_back(location);
      entry.metrics.last_accessed_at = now;
      ++entry.metrics.access_count;
      entry.last_accessed_rep.store(now.time_since_epoch().count(), std::memory_order_release);
      entry.atomic_access_count.fetch_add(1, std::memory_order_relaxed);
      ++inserted;

      if (keys_seen.insert(key).second) {
        keys_to_track.push_back(key);
      }
    }
  }

  if (registry != nullptr) {
    for (const auto& key : keys_to_track) {
      registry->TrackKey(node_id, key);
    }
  }

  return inserted;
}

size_t GlobalBlockIndex::BatchUnregister(
    const std::string& node_id, const std::vector<std::pair<std::string, Location>>& entries) {
  if (entries.empty()) {
    return 0;
  }

  size_t removed_count = 0;
  std::vector<std::string> keys_to_untrack;
  std::unordered_set<std::string> keys_seen;
  ClientRegistry* registry = nullptr;

  {
    std::unique_lock lock(mutex_);
    registry = registry_;

    for (const auto& [key, location] : entries) {
      auto it = entries_.find(key);
      if (it == entries_.end()) {
        continue;
      }

      auto& locs = it->second.locations;
      const size_t original_size = locs.size();
      locs.erase(std::remove(locs.begin(), locs.end(), location), locs.end());

      if (locs.size() == original_size) {
        continue;
      }

      ++removed_count;
      const bool has_remaining_for_client =
          std::any_of(locs.begin(), locs.end(),
                      [&node_id](const Location& loc) { return loc.node_id == node_id; });
      if (!has_remaining_for_client && keys_seen.insert(key).second) {
        keys_to_untrack.push_back(key);
      }

      if (locs.empty()) {
        entries_.erase(it);
      }
    }
  }

  if (registry != nullptr) {
    for (const auto& key : keys_to_untrack) {
      registry->UntrackKey(node_id, key);
    }
  }

  return removed_count;
}

void GlobalBlockIndex::RecordAccess(const std::string& key) {
  std::shared_lock lock(mutex_);

  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return;
  }

  it->second.RecordAccessAtomic();
}

std::vector<Location> GlobalBlockIndex::Lookup(const std::string& key) const {
  std::shared_lock lock(mutex_);

  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return {};
  }

  return it->second.locations;
}

std::vector<bool> GlobalBlockIndex::BatchLookupExists(const std::vector<std::string>& keys) const {
  std::vector<bool> results(keys.size(), false);
  if (keys.empty()) return results;

  std::shared_lock lock(mutex_);
  for (size_t i = 0; i < keys.size(); ++i) {
    auto it = entries_.find(keys[i]);
    results[i] = (it != entries_.end()) && !it->second.locations.empty();
  }
  return results;
}

std::optional<BlockMetrics> GlobalBlockIndex::GetMetrics(const std::string& key) const {
  std::shared_lock lock(mutex_);

  auto it = entries_.find(key);
  if (it == entries_.end()) {
    return std::nullopt;
  }

  BlockMetrics result = it->second.metrics;
  result.last_accessed_at = it->second.GetLastAccessed();
  result.access_count = it->second.atomic_access_count.load(std::memory_order_acquire);
  return result;
}

void GlobalBlockIndex::GrantLease(const std::string& key,
                                  std::chrono::steady_clock::duration duration) {
  std::shared_lock lock(mutex_);

  auto it = entries_.find(key);
  if (it != entries_.end()) {
    it->second.GrantLease(duration);
  }
}

void GlobalBlockIndex::SetDepth(const std::string& key, int32_t depth) {
  std::unique_lock lock(mutex_);
  auto it = entries_.find(key);
  if (it != entries_.end()) {
    it->second.depth = depth;
  }
}

std::vector<EvictionCandidate> GlobalBlockIndex::FindEvictionCandidates(
    const std::set<NodeTierKey>& overloaded_node_tiers) const {
  std::vector<EvictionCandidate> candidates;
  std::shared_lock lock(mutex_);

  for (const auto& [key, entry] : entries_) {
    if (entry.IsLeased()) {
      continue;
    }
    for (const auto& loc : entry.locations) {
      if (overloaded_node_tiers.count({loc.node_id, loc.tier})) {
        EvictionCandidate c;
        c.key = key;
        c.location = loc;
        c.last_accessed_at = entry.GetLastAccessed();
        c.depth = entry.depth;
        c.size = loc.size;
        candidates.push_back(std::move(c));
      }
    }
  }

  return candidates;
}

std::vector<EvictionCandidate> GlobalBlockIndex::EvictEntries(
    const std::vector<EvictionCandidate>& victims) {
  std::vector<EvictionCandidate> evicted;
  std::vector<std::pair<std::string, std::string>> keys_to_untrack;
  ClientRegistry* registry = nullptr;

  {
    std::unique_lock lock(mutex_);
    registry = registry_;

    for (const auto& victim : victims) {
      auto it = entries_.find(victim.key);
      if (it == entries_.end()) {
        continue;
      }

      auto& entry = it->second;
      if (entry.IsLeased()) {
        continue;
      }

      auto& locs = entry.locations;
      auto loc_it = std::find(locs.begin(), locs.end(), victim.location);
      if (loc_it == locs.end()) {
        continue;
      }

      locs.erase(loc_it);
      evicted.push_back(victim);

      const bool has_remaining_for_node = std::any_of(
          locs.begin(), locs.end(),
          [&victim](const Location& loc) { return loc.node_id == victim.location.node_id; });
      if (!has_remaining_for_node) {
        keys_to_untrack.emplace_back(victim.location.node_id, victim.key);
      }

      if (locs.empty()) {
        entries_.erase(it);
      }
    }
  }

  if (registry != nullptr) {
    for (const auto& [node_id, key] : keys_to_untrack) {
      registry->UntrackKey(node_id, key);
    }
  }

  return evicted;
}

}  // namespace mori::umbp
