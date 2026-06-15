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
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

namespace {

// Locate (or insert) the location for (node_id, tier) within an entry's
// location list.  Caller MUST hold the unique lock.  Returns a pointer
// into entry.locations that's stable until the next mutation.
std::pair<Location*, bool> FindOrInsertLocation(BlockEntry& entry, const std::string& node_id,
                                                TierType tier) {
  for (auto& loc : entry.locations) {
    if (loc.node_id == node_id && loc.tier == tier) return {&loc, false};
  }
  entry.locations.push_back(Location{node_id, /*size=*/0, tier});
  return {&entry.locations.back(), true};
}

bool HasLocationForNode(const BlockEntry& entry, const std::string& node_id) {
  return std::any_of(entry.locations.begin(), entry.locations.end(),
                     [&](const Location& loc) { return loc.node_id == node_id; });
}

size_t RemoveLocationsLocked(
    std::unordered_map<std::string, BlockEntry>& entries,
    std::unordered_map<std::string, std::unordered_set<std::string>>& node_to_keys,
    const std::string& node_id, std::optional<TierType> tier) {
  size_t removed = 0;
  for (auto it = entries.begin(); it != entries.end();) {
    auto& locs = it->second.locations;
    const size_t before = locs.size();
    locs.erase(std::remove_if(locs.begin(), locs.end(),
                              [&](const Location& l) {
                                if (l.node_id != node_id) return false;
                                if (tier.has_value() && l.tier != *tier) return false;
                                return true;
                              }),
               locs.end());
    const size_t removed_from_entry = before - locs.size();
    removed += removed_from_entry;
    if (removed_from_entry != 0 && !HasLocationForNode(it->second, node_id)) {
      auto rev_it = node_to_keys.find(node_id);
      if (rev_it != node_to_keys.end()) {
        rev_it->second.erase(it->first);
        if (rev_it->second.empty()) node_to_keys.erase(rev_it);
      }
    }
    if (locs.empty()) {
      it = entries.erase(it);
    } else {
      ++it;
    }
  }
  return removed;
}

}  // namespace

size_t GlobalBlockIndex::ApplyEvents(const std::string& node_id,
                                     const std::vector<KvEvent>& events) {
  if (events.empty()) return 0;
  std::unique_lock lock(mutex_);
  size_t mutated = 0;
  const auto now = std::chrono::system_clock::now();

  for (const auto& ev : events) {
    if (ev.kind == KvEvent::Kind::CLEAR_AT_TIER) {
      mutated += RemoveLocationsLocked(entries_, node_to_keys_, node_id, ev.tier);
    } else if (ev.kind == KvEvent::Kind::ADD) {
      auto& entry = entries_[ev.key];
      if (entry.locations.empty()) {
        entry.metrics.created_at = now;
        entry.metrics.last_accessed_at = now;
        entry.metrics.access_count = 0;
        entry.last_accessed_rep.store(now.time_since_epoch().count(), std::memory_order_release);
        entry.atomic_access_count.store(0, std::memory_order_relaxed);
      }
      auto [loc, inserted] = FindOrInsertLocation(entry, node_id, ev.tier);
      // Idempotent; must run on duplicate ADDs too.
      node_to_keys_[node_id].insert(ev.key);
      if (!inserted) {
        MORI_UMBP_WARN(
            "[GlobalBlockIndex] duplicate ADD for key='{}' node={} tier={} old_size={} "
            "new_size={}; keeping existing location",
            ev.key, node_id, TierTypeName(ev.tier), loc->size, ev.size);
      } else {
        loc->size = ev.size;
        ++mutated;
      }
    } else {  // REMOVE
      auto it = entries_.find(ev.key);
      if (it == entries_.end()) continue;
      auto& locs = it->second.locations;
      const size_t before = locs.size();
      locs.erase(std::remove_if(
                     locs.begin(), locs.end(),
                     [&](const Location& l) { return l.node_id == node_id && l.tier == ev.tier; }),
                 locs.end());
      if (locs.size() != before) {
        ++mutated;
        // find(), not operator[]: don't grow an empty bucket for strangers.
        if (!HasLocationForNode(it->second, node_id)) {
          auto rev_it = node_to_keys_.find(node_id);
          if (rev_it != node_to_keys_.end()) {
            rev_it->second.erase(ev.key);
            if (rev_it->second.empty()) node_to_keys_.erase(rev_it);
          }
        }
        if (locs.empty()) entries_.erase(it);
      }
    }
  }
  return mutated;
}

void GlobalBlockIndex::ReplaceNodeLocations(const std::string& node_id,
                                            const std::vector<KvEvent>& adds) {
  std::unique_lock lock(mutex_);
  const auto now = std::chrono::system_clock::now();

  // O(N_node + |adds|) via the reverse index.
  auto rev_it = node_to_keys_.find(node_id);
  if (rev_it != node_to_keys_.end()) {
    auto old_keys = std::move(rev_it->second);
    node_to_keys_.erase(rev_it);
    for (const auto& key : old_keys) {
      auto eit = entries_.find(key);
      if (eit == entries_.end()) continue;
      auto& locs = eit->second.locations;
      locs.erase(std::remove_if(locs.begin(), locs.end(),
                                [&](const Location& l) { return l.node_id == node_id; }),
                 locs.end());
      if (locs.empty()) {
        entries_.erase(eit);
      }
    }
  }

  for (const auto& ev : adds) {
    if (ev.kind != KvEvent::Kind::ADD) continue;
    auto& entry = entries_[ev.key];
    if (entry.locations.empty()) {
      entry.metrics.created_at = now;
      entry.metrics.last_accessed_at = now;
      entry.metrics.access_count = 0;
      entry.last_accessed_rep.store(now.time_since_epoch().count(), std::memory_order_release);
      entry.atomic_access_count.store(0, std::memory_order_relaxed);
    }
    auto [loc, inserted] = FindOrInsertLocation(entry, node_id, ev.tier);
    (void)inserted;
    loc->size = ev.size;
    node_to_keys_[node_id].insert(ev.key);
  }
}

void GlobalBlockIndex::RemoveByNode(const std::string& node_id) {
  std::unique_lock lock(mutex_);
  RemoveLocationsLocked(entries_, node_to_keys_, node_id, std::nullopt);
}

void GlobalBlockIndex::RecordAccess(const std::string& key) {
  std::shared_lock lock(mutex_);
  auto it = entries_.find(key);
  if (it == entries_.end()) return;
  it->second.RecordAccessAtomic();
}

void GlobalBlockIndex::GrantLease(const std::string& key,
                                  std::chrono::system_clock::duration duration) {
  std::shared_lock lock(mutex_);
  auto it = entries_.find(key);
  if (it != entries_.end()) it->second.GrantLease(duration);
}

std::vector<Location> GlobalBlockIndex::Lookup(const std::string& key) const {
  std::shared_lock lock(mutex_);
  auto it = entries_.find(key);
  if (it == entries_.end()) return {};
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
  if (it == entries_.end()) return std::nullopt;
  BlockMetrics result = it->second.metrics;
  result.last_accessed_at = it->second.GetLastAccessed();
  result.access_count = it->second.atomic_access_count.load(std::memory_order_acquire);
  return result;
}

std::vector<std::vector<Location>> GlobalBlockIndex::BatchLookupForRouteGet(
    const std::vector<std::string>& keys, const std::unordered_set<std::string>& exclude_nodes,
    std::chrono::system_clock::duration lease_duration) {
  std::vector<std::vector<Location>> out(keys.size());
  if (keys.empty()) return out;
  std::shared_lock lock(mutex_);
  for (size_t i = 0; i < keys.size(); ++i) {
    auto it = entries_.find(keys[i]);
    if (it == entries_.end()) continue;
    auto& locs = out[i];
    for (const auto& loc : it->second.locations) {
      if (!exclude_nodes.empty() && exclude_nodes.count(loc.node_id)) continue;
      locs.push_back(loc);
    }
    if (locs.empty()) continue;
    it->second.RecordAccessAtomic();
    it->second.GrantLease(lease_duration);
  }
  return out;
}

std::vector<EvictionCandidate> GlobalBlockIndex::FindEvictionCandidates(
    const std::set<NodeTierKey>& overloaded_node_tiers) const {
  std::vector<EvictionCandidate> candidates;
  std::shared_lock lock(mutex_);
  for (const auto& [key, entry] : entries_) {
    if (entry.IsLeased()) continue;
    for (const auto& loc : entry.locations) {
      if (overloaded_node_tiers.count({loc.node_id, loc.tier})) {
        EvictionCandidate c;
        c.key = key;
        c.location = loc;
        c.last_accessed_at = entry.GetLastAccessed();
        c.size = loc.size;
        candidates.push_back(std::move(c));
      }
    }
  }
  return candidates;
}

}  // namespace mori::umbp
