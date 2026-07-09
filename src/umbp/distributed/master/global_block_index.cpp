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
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

namespace {

// Default shard count.  Picked so a typical 128-key heartbeat fans out to a
// handful of events per shard while keeping per-shard maps cache-friendly.
constexpr size_t kDefaultIndexShards = 32;

// Upper bound so a fat-fingered env value can't try to allocate a runaway
// number of shards (each shard is a shared_mutex + two maps) and OOM at start.
constexpr size_t kMaxIndexShards = 4096;

size_t IndexShardCount() {
  if (const char* e = std::getenv("UMBP_MASTER_INDEX_SHARDS")) {
    char* end = nullptr;
    const long v = std::strtol(e, &end, 10);
    if (end != e && v >= 1) {
      const size_t shards = static_cast<size_t>(v);
      if (shards > kMaxIndexShards) {
        MORI_UMBP_WARN("[GlobalBlockIndex] clamping UMBP_MASTER_INDEX_SHARDS={} to max {}", shards,
                       kMaxIndexShards);
        return kMaxIndexShards;
      }
      return shards;
    }
    MORI_UMBP_WARN("[GlobalBlockIndex] ignoring invalid UMBP_MASTER_INDEX_SHARDS='{}'", e);
  }
  return kDefaultIndexShards;
}

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

using EntryMap = std::unordered_map<std::string, BlockEntry>;
using NodeToKeys = std::unordered_map<std::string, std::unordered_set<std::string>>;

// Drop every location owned by `node_id` within one shard, driven by the
// shard-local reverse index so the cost is O(node's keys in this shard) rather
// than a full O(shard entries) scan.  Caller holds the shard's unique lock.
void RemoveNodeLocationsLocked(EntryMap& entries, NodeToKeys& node_to_keys,
                               const std::string& node_id) {
  auto rev_it = node_to_keys.find(node_id);
  if (rev_it == node_to_keys.end()) return;
  auto keys = std::move(rev_it->second);
  node_to_keys.erase(rev_it);
  for (const auto& key : keys) {
    auto eit = entries.find(key);
    if (eit == entries.end()) continue;
    auto& locs = eit->second.locations;
    locs.erase(std::remove_if(locs.begin(), locs.end(),
                              [&](const Location& l) { return l.node_id == node_id; }),
               locs.end());
    if (locs.empty()) entries.erase(eit);
  }
}

// Apply a single ADD/REMOVE event to one shard's maps (caller holds the
// shard's unique lock).  Returns 1 if it mutated a location, else 0.
size_t ApplyAddOrRemoveLocked(EntryMap& entries, NodeToKeys& node_to_keys,
                              const std::string& node_id, const KvEvent& ev,
                              std::chrono::steady_clock::time_point now) {
  if (ev.kind == KvEvent::Kind::ADD) {
    auto& entry = entries[ev.key];
    if (entry.locations.empty()) {
      entry.metrics.created_at = now;
      entry.metrics.last_accessed_at = now;
      entry.metrics.access_count = 0;
      entry.last_accessed_rep.store(now.time_since_epoch().count(), std::memory_order_release);
      entry.atomic_access_count.store(0, std::memory_order_relaxed);
    }
    auto [loc, inserted] = FindOrInsertLocation(entry, node_id, ev.tier);
    // Idempotent; must run on duplicate ADDs too.
    node_to_keys[node_id].insert(ev.key);
    if (!inserted) {
      MORI_UMBP_WARN(
          "[GlobalBlockIndex] duplicate ADD for key='{}' node={} tier={} old_size={} "
          "new_size={}; keeping existing location",
          ev.key, node_id, TierTypeName(ev.tier), loc->size, ev.size);
      return 0;
    }
    loc->size = ev.size;
    loc->encoding = ev.encoding;
    if (loc->encoding.stored_bytes == 0) loc->encoding.stored_bytes = ev.size;
    if (loc->encoding.logical_bytes == 0) loc->encoding.logical_bytes = loc->encoding.stored_bytes;
    return 1;
  }
  // REMOVE
  auto it = entries.find(ev.key);
  if (it == entries.end()) return 0;
  auto& locs = it->second.locations;
  const size_t before = locs.size();
  locs.erase(
      std::remove_if(locs.begin(), locs.end(),
                     [&](const Location& l) { return l.node_id == node_id && l.tier == ev.tier; }),
      locs.end());
  if (locs.size() == before) return 0;
  // find(), not operator[]: don't grow an empty bucket for strangers.
  if (!HasLocationForNode(it->second, node_id)) {
    auto rev_it = node_to_keys.find(node_id);
    if (rev_it != node_to_keys.end()) {
      rev_it->second.erase(ev.key);
      if (rev_it->second.empty()) node_to_keys.erase(rev_it);
    }
  }
  if (locs.empty()) entries.erase(it);
  return 1;
}

}  // namespace

GlobalBlockIndex::GlobalBlockIndex() : num_shards_(IndexShardCount()) {
  shards_.reserve(num_shards_);
  for (size_t i = 0; i < num_shards_; ++i) shards_.push_back(std::make_unique<Shard>());
}

size_t GlobalBlockIndex::ApplyEvents(const std::string& node_id,
                                     const std::vector<KvEvent>& events) {
  if (events.empty()) return 0;
  const auto now = std::chrono::steady_clock::now();

  // Sort one (shard, original index) array into per-shard runs; the index
  // tie-break keeps same-key events in order.  Each shard is locked once for
  // its contiguous run, leaving other shards readable concurrently.
  std::vector<std::pair<size_t, size_t>> shard_order(events.size());
  for (size_t i = 0; i < events.size(); ++i) shard_order[i] = {shard_index(events[i].key), i};
  std::sort(shard_order.begin(), shard_order.end());

  size_t mutated = 0;
  for (size_t run_begin = 0; run_begin < shard_order.size();) {
    const size_t si = shard_order[run_begin].first;
    auto& sh = shard_at(si);
    std::unique_lock lock(sh.mutex);
    size_t run_end = run_begin;
    for (; run_end < shard_order.size() && shard_order[run_end].first == si; ++run_end) {
      const KvEvent& ev = events[shard_order[run_end].second];
      mutated += ApplyAddOrRemoveLocked(sh.entries, sh.node_to_keys, node_id, ev, now);
    }
    run_begin = run_end;
  }
  return mutated;
}

void GlobalBlockIndex::ReplaceNodeLocations(const std::string& node_id,
                                            const std::vector<KvEvent>& adds) {
  const auto now = std::chrono::steady_clock::now();

  // Group ADDs by destination shard up front so each shard is locked once.
  std::vector<std::vector<const KvEvent*>> adds_by_shard(num_shards_);
  for (const auto& ev : adds) {
    if (ev.kind != KvEvent::Kind::ADD) continue;
    adds_by_shard[shard_index(ev.key)].push_back(&ev);
  }

  // Each shard owns exactly the node's keys that hash to it, so clearing the
  // node + replaying its ADDs can be done shard-local with one lock per shard.
  // This replaces the old single giant exclusive critical section with
  // num_shards_ small ones — the win for full-sync tail latency.
  for (size_t si = 0; si < num_shards_; ++si) {
    auto& sh = shard_at(si);
    std::unique_lock lock(sh.mutex);

    // O(N_node_in_shard) clear via the shard-local reverse index.
    RemoveNodeLocationsLocked(sh.entries, sh.node_to_keys, node_id);

    for (const KvEvent* ev : adds_by_shard[si]) {
      auto& entry = sh.entries[ev->key];
      if (entry.locations.empty()) {
        entry.metrics.created_at = now;
        entry.metrics.last_accessed_at = now;
        entry.metrics.access_count = 0;
        entry.last_accessed_rep.store(now.time_since_epoch().count(), std::memory_order_release);
        entry.atomic_access_count.store(0, std::memory_order_relaxed);
      }
      auto [loc, inserted] = FindOrInsertLocation(entry, node_id, ev->tier);
      (void)inserted;
      loc->size = ev->size;
      loc->encoding = ev->encoding;
      if (loc->encoding.stored_bytes == 0) loc->encoding.stored_bytes = ev->size;
      if (loc->encoding.logical_bytes == 0) loc->encoding.logical_bytes = loc->encoding.stored_bytes;
      sh.node_to_keys[node_id].insert(ev->key);
    }
  }
}

void GlobalBlockIndex::RemoveByNode(const std::string& node_id) {
  // O(N_node_in_shard) per shard via the reverse index (not a full scan).
  for (size_t si = 0; si < num_shards_; ++si) {
    auto& sh = shard_at(si);
    std::unique_lock lock(sh.mutex);
    RemoveNodeLocationsLocked(sh.entries, sh.node_to_keys, node_id);
  }
}

void GlobalBlockIndex::RecordAccess(const std::string& key) {
  auto& sh = shard_for(key);
  std::shared_lock lock(sh.mutex);
  auto it = sh.entries.find(key);
  if (it == sh.entries.end()) return;
  it->second.RecordAccessAtomic();
}

void GlobalBlockIndex::GrantLease(const std::string& key,
                                  std::chrono::steady_clock::duration duration) {
  auto& sh = shard_for(key);
  std::shared_lock lock(sh.mutex);
  auto it = sh.entries.find(key);
  if (it != sh.entries.end()) it->second.GrantLease(duration);
}

std::vector<Location> GlobalBlockIndex::Lookup(const std::string& key) const {
  auto& sh = shard_for(key);
  std::shared_lock lock(sh.mutex);
  auto it = sh.entries.find(key);
  if (it == sh.entries.end()) return {};
  return it->second.locations;
}

std::vector<bool> GlobalBlockIndex::BatchLookupExists(const std::vector<std::string>& keys) const {
  std::vector<bool> results(keys.size(), false);
  if (keys.empty()) return results;
  // Single-key fast path (RoutePut/RouteGet route through size-1 batches):
  // skip the per-shard grouping vectors and lock just the one shard.
  if (keys.size() == 1) {
    auto& sh = shard_for(keys[0]);
    std::shared_lock lock(sh.mutex);
    auto it = sh.entries.find(keys[0]);
    results[0] = (it != sh.entries.end()) && !it->second.locations.empty();
    return results;
  }
  // Group key indices by shard so each shard's shared lock is taken once.
  std::vector<std::vector<size_t>> idx_by_shard(num_shards_);
  for (size_t i = 0; i < keys.size(); ++i) idx_by_shard[shard_index(keys[i])].push_back(i);
  for (size_t si = 0; si < num_shards_; ++si) {
    if (idx_by_shard[si].empty()) continue;
    auto& sh = shard_at(si);
    std::shared_lock lock(sh.mutex);
    for (size_t i : idx_by_shard[si]) {
      auto it = sh.entries.find(keys[i]);
      results[i] = (it != sh.entries.end()) && !it->second.locations.empty();
    }
  }
  return results;
}

std::optional<BlockMetrics> GlobalBlockIndex::GetMetrics(const std::string& key) const {
  auto& sh = shard_for(key);
  std::shared_lock lock(sh.mutex);
  auto it = sh.entries.find(key);
  if (it == sh.entries.end()) return std::nullopt;
  BlockMetrics result = it->second.metrics;
  result.last_accessed_at = it->second.GetLastAccessed();
  result.access_count = it->second.atomic_access_count.load(std::memory_order_acquire);
  return result;
}

std::vector<std::vector<Location>> GlobalBlockIndex::BatchLookupForRouteGet(
    const std::vector<std::string>& keys, const std::unordered_set<std::string>& exclude_nodes,
    std::chrono::steady_clock::duration lease_duration) {
  std::vector<std::vector<Location>> out(keys.size());
  if (keys.empty()) return out;
  const bool has_exclude = !exclude_nodes.empty();
  // Read the clock once for the whole batch: the per-hit access/lease bump
  // would otherwise call steady_clock::now() twice per key. All keys in a
  // batch are stamped within the same ~ms, which is well within lease/LRU
  // granularity.
  const auto now = std::chrono::steady_clock::now();

  // Per-key body, shared across the single-key fast path and the grouped path.
  auto resolve_one = [&](Shard& sh, size_t i) {
    auto it = sh.entries.find(keys[i]);
    if (it == sh.entries.end()) return;
    const auto& src = it->second.locations;
    auto& locs = out[i];
    if (!has_exclude) {
      // Common case: copy the whole location vector in one allocation
      // (no per-element vector growth).
      locs = src;
    } else {
      locs.reserve(src.size());
      for (const auto& loc : src) {
        if (exclude_nodes.count(loc.node_id)) continue;
        locs.push_back(loc);
      }
    }
    if (locs.empty()) return;
    it->second.RecordAccessAtomic(now);
    it->second.GrantLease(now, lease_duration);
  };

  // Single-key fast path (RoutePut/RouteGet route through size-1 batches):
  // skip the per-shard grouping vectors and lock just the one shard.
  if (keys.size() == 1) {
    auto& sh = shard_for(keys[0]);
    std::shared_lock lock(sh.mutex);
    resolve_one(sh, 0);
    return out;
  }

  std::vector<std::vector<size_t>> idx_by_shard(num_shards_);
  for (size_t i = 0; i < keys.size(); ++i) idx_by_shard[shard_index(keys[i])].push_back(i);
  for (size_t si = 0; si < num_shards_; ++si) {
    if (idx_by_shard[si].empty()) continue;
    auto& sh = shard_at(si);
    std::shared_lock lock(sh.mutex);
    for (size_t i : idx_by_shard[si]) resolve_one(sh, i);
  }
  return out;
}

std::vector<EvictionCandidate> GlobalBlockIndex::FindEvictionCandidates(
    const std::set<NodeTierKey>& overloaded_node_tiers) const {
  // Best-effort advisory scan: lock one shard at a time (not a global atomic
  // snapshot).  Candidates are re-validated downstream, so a concurrent apply
  // on an already-scanned shard is harmless.
  std::vector<EvictionCandidate> candidates;
  for (size_t si = 0; si < num_shards_; ++si) {
    auto& sh = shard_at(si);
    std::shared_lock lock(sh.mutex);
    for (const auto& [key, entry] : sh.entries) {
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
  }
  return candidates;
}

}  // namespace mori::umbp
