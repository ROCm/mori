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
#include "umbp/distributed/master/in_memory_master_metadata_store.h"

#include <algorithm>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <unordered_set>
#include <utility>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

namespace {

// Locate (or insert) the location for (node_id, tier) within a location list.
// Caller MUST hold the unique lock. Returns a pointer into `locations` that's
// stable until the next mutation.
std::pair<Location*, bool> FindOrInsertLocation(std::vector<Location>& locations,
                                                const std::string& node_id, TierType tier) {
  for (auto& loc : locations) {
    if (loc.node_id == node_id && loc.tier == tier) return {&loc, false};
  }
  locations.push_back(Location{node_id, /*size=*/0, tier});
  return {&locations.back(), true};
}

bool HasLocationForNode(const std::vector<Location>& locations, const std::string& node_id) {
  return std::any_of(locations.begin(), locations.end(),
                     [&](const Location& loc) { return loc.node_id == node_id; });
}

}  // namespace

// =====================================================================
// Locked helpers
// =====================================================================

size_t InMemoryMasterMetadataStore::ApplyEventsLocked(const std::string& node_id,
                                                      const std::vector<KvEvent>& events,
                                                      std::chrono::system_clock::time_point now) {
  size_t mutated = 0;
  for (const auto& ev : events) {
    if (ev.kind == KvEvent::Kind::ADD) {
      auto& entry = entries_[ev.key];
      if (entry.locations.empty()) {
        entry.metrics.created_at = now;
        entry.metrics.last_accessed_at = now;
        entry.metrics.access_count = 0;
        entry.last_accessed_rep.store(now.time_since_epoch().count(), std::memory_order_release);
        entry.atomic_access_count.store(0, std::memory_order_relaxed);
      }
      auto [loc, inserted] = FindOrInsertLocation(entry.locations, node_id, ev.tier);
      // Idempotent; must run on duplicate ADDs too.
      node_to_keys_[node_id].insert(ev.key);
      if (!inserted) {
        MORI_UMBP_WARN(
            "[MetadataStore] duplicate ADD for key='{}' node={} tier={} old_size={} "
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
        if (!HasLocationForNode(it->second.locations, node_id)) {
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

void InMemoryMasterMetadataStore::ReplaceNodeLocationsLocked(
    const std::string& node_id, const std::vector<KvEvent>& adds,
    std::chrono::system_clock::time_point now) {
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
      if (locs.empty()) entries_.erase(eit);
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
    auto [loc, inserted] = FindOrInsertLocation(entry.locations, node_id, ev.tier);
    (void)inserted;
    loc->size = ev.size;
    node_to_keys_[node_id].insert(ev.key);
  }
}

void InMemoryMasterMetadataStore::RemoveBlocksByNodeLocked(const std::string& node_id) {
  // Drive the removal off the reverse index so the cost is O(node's keys)
  // rather than a full O(entries_) scan — UnregisterClient/ExpireStaleClients
  // hit this per dead node.
  auto rev_it = node_to_keys_.find(node_id);
  if (rev_it == node_to_keys_.end()) return;
  auto keys = std::move(rev_it->second);
  node_to_keys_.erase(rev_it);
  for (const auto& key : keys) {
    auto eit = entries_.find(key);
    if (eit == entries_.end()) continue;
    auto& locs = eit->second.locations;
    locs.erase(std::remove_if(locs.begin(), locs.end(),
                              [&](const Location& l) { return l.node_id == node_id; }),
               locs.end());
    if (locs.empty()) entries_.erase(eit);
  }
}

void InMemoryMasterMetadataStore::RemoveExternalKvByNodeLocked(const std::string& node_id) {
  auto it = external_kv_entries_.begin();
  while (it != external_kv_entries_.end()) {
    it->second.erase(node_id);
    if (it->second.empty()) {
      it = external_kv_entries_.erase(it);
    } else {
      ++it;
    }
  }
}

bool InMemoryMasterMetadataStore::IsClientAliveLocked(const std::string& node_id) const {
  auto it = clients_.find(node_id);
  return it != clients_.end() && it->second.status == ClientStatus::ALIVE;
}

// =====================================================================
// Cross-store writes
// =====================================================================

bool InMemoryMasterMetadataStore::RegisterClient(const ClientRegistration& registration,
                                                 std::chrono::system_clock::time_point now,
                                                 std::chrono::system_clock::duration stale_after) {
  std::unique_lock lock(mutex_);

  auto it = clients_.find(registration.node_id);
  if (it != clients_.end()) {
    const bool is_stale = (now - it->second.last_heartbeat > stale_after) ||
                          (it->second.status == ClientStatus::EXPIRED);
    if (it->second.status == ClientStatus::ALIVE && !is_stale) {
      MORI_UMBP_WARN("[MetadataStore] Rejecting re-registration for alive node: {}",
                     registration.node_id);
      return false;
    }
    MORI_UMBP_INFO("[MetadataStore] Re-registering stale/expired node: {}", registration.node_id);
  }

  ClientRecord record;
  record.node_id = registration.node_id;
  record.node_address = registration.node_address;
  record.status = ClientStatus::ALIVE;
  record.last_heartbeat = now;
  record.registered_at = now;
  record.tier_capacities = registration.tier_capacities;
  record.peer_address = registration.peer_address;
  record.engine_desc_bytes = registration.engine_desc_bytes;
  record.last_applied_seq = 0;
  record.tags = registration.tags;

  clients_[registration.node_id] = std::move(record);

  std::string tags_str;
  for (const auto& t : registration.tags) {
    if (!tags_str.empty()) tags_str += ',';
    tags_str += t;
  }
  MORI_UMBP_INFO("[MetadataStore] Registered node: {} at {} (peer={}) tags=[{}]",
                 registration.node_id, registration.node_address, registration.peer_address,
                 tags_str);
  return true;
}

void InMemoryMasterMetadataStore::UnregisterClient(const std::string& node_id) {
  std::unique_lock lock(mutex_);
  auto it = clients_.find(node_id);
  if (it == clients_.end()) return;
  clients_.erase(it);
  RemoveBlocksByNodeLocked(node_id);
  RemoveExternalKvByNodeLocked(node_id);
  MORI_UMBP_INFO("[MetadataStore] Unregistered node: {}", node_id);
}

HeartbeatResult InMemoryMasterMetadataStore::ApplyHeartbeat(
    const std::string& node_id, uint64_t seq, std::chrono::system_clock::time_point now,
    const std::map<TierType, TierCapacity>& caps, const std::vector<KvEvent>& events,
    bool is_full_sync) {
  std::unique_lock lock(mutex_);

  auto it = clients_.find(node_id);
  if (it == clients_.end()) {
    MORI_UMBP_WARN("[MetadataStore] Heartbeat from unknown node: {}", node_id);
    return HeartbeatResult{HeartbeatResult::UNKNOWN, 0};
  }
  auto& record = it->second;

  // Gap check (CAS) on the delta path only — full_sync replaces wholesale and
  // re-baselines last_applied_seq.
  if (!is_full_sync && seq != record.last_applied_seq + 1) {
    // SEQ_GAP: keep the node alive (it IS heartbeating, just mid-recovery) but
    // do NOT advance caps or last_applied_seq. See hazard #1.
    MORI_UMBP_WARN(
        "[MetadataStore] Heartbeat seq gap from {}: got {}, expected {} — requesting full sync",
        node_id, seq, record.last_applied_seq + 1);
    record.last_heartbeat = now;
    record.status = ClientStatus::ALIVE;
    return HeartbeatResult{HeartbeatResult::SEQ_GAP, record.last_applied_seq};
  }

  record.last_heartbeat = now;
  record.status = ClientStatus::ALIVE;
  record.tier_capacities = caps;
  record.last_applied_seq = seq;

  if (is_full_sync) {
    ReplaceNodeLocationsLocked(node_id, events, now);
  } else {
    ApplyEventsLocked(node_id, events, now);
  }
  return HeartbeatResult{HeartbeatResult::APPLIED, seq};
}

std::vector<std::string> InMemoryMasterMetadataStore::ExpireStaleClients(
    std::chrono::system_clock::time_point cutoff) {
  std::unique_lock lock(mutex_);
  std::vector<std::string> dead_nodes;

  for (auto& [node_id, record] : clients_) {
    // Only ALIVE rows can transition to EXPIRED; an already-EXPIRED row is left
    // alone so re-ticking the reaper is idempotent (its locations are already
    // gone). EXPIRED rows are KEPT, not erased — see hazard #3.
    if (record.status == ClientStatus::ALIVE && record.last_heartbeat < cutoff) {
      MORI_UMBP_WARN("[MetadataStore] Expiring stale client: {}", node_id);
      record.status = ClientStatus::EXPIRED;
      dead_nodes.push_back(node_id);
    }
  }

  for (const auto& dead_id : dead_nodes) {
    RemoveBlocksByNodeLocked(dead_id);
    RemoveExternalKvByNodeLocked(dead_id);
  }
  return dead_nodes;
}

// =====================================================================
// External-KV writes
// =====================================================================

bool InMemoryMasterMetadataStore::RegisterExternalKvIfAlive(const std::string& node_id,
                                                            const std::vector<std::string>& hashes,
                                                            TierType tier) {
  std::unique_lock lock(mutex_);
  if (!IsClientAliveLocked(node_id)) return false;
  for (const auto& hash : hashes) {
    external_kv_entries_[hash][node_id].insert(tier);
  }
  return true;
}

void InMemoryMasterMetadataStore::UnregisterExternalKv(const std::string& node_id,
                                                       const std::vector<std::string>& hashes,
                                                       TierType tier) {
  std::unique_lock lock(mutex_);
  for (const auto& hash : hashes) {
    auto it = external_kv_entries_.find(hash);
    if (it == external_kv_entries_.end()) continue;
    auto node_it = it->second.find(node_id);
    if (node_it == it->second.end()) continue;
    node_it->second.erase(tier);
    if (node_it->second.empty()) it->second.erase(node_it);
    if (it->second.empty()) external_kv_entries_.erase(it);
  }
}

void InMemoryMasterMetadataStore::UnregisterExternalKvByTier(const std::string& node_id,
                                                             TierType tier) {
  std::unique_lock lock(mutex_);
  auto it = external_kv_entries_.begin();
  while (it != external_kv_entries_.end()) {
    auto node_it = it->second.find(node_id);
    if (node_it != it->second.end()) {
      node_it->second.erase(tier);
      if (node_it->second.empty()) it->second.erase(node_it);
    }
    if (it->second.empty()) {
      it = external_kv_entries_.erase(it);
    } else {
      ++it;
    }
  }
}

void InMemoryMasterMetadataStore::UnregisterExternalKvByNode(const std::string& node_id) {
  std::unique_lock lock(mutex_);
  RemoveExternalKvByNodeLocked(node_id);
}

std::size_t InMemoryMasterMetadataStore::GarbageCollectHits(
    std::chrono::system_clock::time_point cutoff) {
  std::unique_lock lock(mutex_);
  std::size_t dropped = 0;
  auto it = external_kv_hits_.begin();
  while (it != external_kv_hits_.end()) {
    if (it->second.last_seen < cutoff) {
      it = external_kv_hits_.erase(it);
      ++dropped;
    } else {
      ++it;
    }
  }
  return dropped;
}

// =====================================================================
// Block reads
// =====================================================================

std::vector<Location> InMemoryMasterMetadataStore::LookupBlock(const std::string& key) const {
  std::shared_lock lock(mutex_);
  auto it = entries_.find(key);
  if (it == entries_.end()) return {};
  return it->second.locations;
}

std::vector<Location> InMemoryMasterMetadataStore::LookupBlockForRouteGet(
    const std::string& key, const std::unordered_set<std::string>& exclude_nodes,
    std::chrono::system_clock::time_point now, std::chrono::system_clock::duration lease_duration) {
  std::shared_lock lock(mutex_);
  auto it = entries_.find(key);
  if (it == entries_.end()) return {};

  std::vector<Location> out;
  for (const auto& loc : it->second.locations) {
    if (!exclude_nodes.empty() && exclude_nodes.count(loc.node_id)) continue;
    out.push_back(loc);
  }
  if (out.empty()) return out;
  it->second.RecordAccessAtomic(now);
  it->second.GrantLease(now, lease_duration);
  return out;
}

std::vector<std::vector<Location>> InMemoryMasterMetadataStore::BatchLookupBlockForRouteGet(
    const std::vector<std::string>& keys, const std::unordered_set<std::string>& exclude_nodes,
    std::chrono::system_clock::time_point now, std::chrono::system_clock::duration lease_duration) {
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
    it->second.RecordAccessAtomic(now);
    it->second.GrantLease(now, lease_duration);
  }
  return out;
}

std::vector<bool> InMemoryMasterMetadataStore::BatchExistsBlock(
    const std::vector<std::string>& keys) const {
  std::vector<bool> results(keys.size(), false);
  if (keys.empty()) return results;
  std::shared_lock lock(mutex_);
  for (size_t i = 0; i < keys.size(); ++i) {
    auto it = entries_.find(keys[i]);
    results[i] = (it != entries_.end()) && !it->second.locations.empty();
  }
  return results;
}

std::map<NodeTierKey, std::vector<EvictionCandidate>>
InMemoryMasterMetadataStore::EnumerateEvictionCandidates(
    const std::vector<NodeTierKey>& buckets, EvictionOrder order, size_t max_per_bucket,
    std::chrono::system_clock::time_point now) const {
  std::map<NodeTierKey, std::vector<EvictionCandidate>> result;
  if (buckets.empty()) return result;

  // Requested buckets as a set for O(log n) membership during the scan.
  const std::set<NodeTierKey> wanted(buckets.begin(), buckets.end());

  std::shared_lock lock(mutex_);

  // 1. Full scan: collect non-leased candidates whose (node, tier) is a
  //    requested bucket. No maintained LRU index — the scan reads entries_
  //    directly, so it is always consistent and tie-timestamp candidates are
  //    never dropped (§2d, Option A). This is policy-neutral: every eligible
  //    row is collected; the byte budget and victim choice live in the
  //    strategy, not here.
  for (const auto& [key, entry] : entries_) {
    if (entry.IsLeased(now)) continue;
    const auto last_accessed = entry.GetLastAccessed();
    for (const auto& loc : entry.locations) {
      NodeTierKey ntk{loc.node_id, loc.tier};
      if (wanted.find(ntk) == wanted.end()) continue;
      EvictionCandidate c;
      c.key = key;
      c.location = loc;
      c.last_accessed_at = last_accessed;
      c.size = loc.size;
      result[ntk].push_back(std::move(c));
    }
  }

  // 2. Honor the ordering hint and the per-bucket cap. For LRU, partial_sort
  //    is enough when a cap is set (an eviction tick is seconds, not a hot
  //    path). max_per_bucket == 0 means "no cap".
  const auto older_first = [](const EvictionCandidate& a, const EvictionCandidate& b) {
    return a.last_accessed_at < b.last_accessed_at;
  };
  for (auto& [ntk, candidates] : result) {
    const bool cap = max_per_bucket > 0 && candidates.size() > max_per_bucket;
    if (order == EvictionOrder::kLeastRecentlyAccessed) {
      if (cap) {
        std::partial_sort(candidates.begin(), candidates.begin() + max_per_bucket,
                          candidates.end(), older_first);
        candidates.resize(max_per_bucket);
      } else {
        std::sort(candidates.begin(), candidates.end(), older_first);
      }
    } else if (cap) {
      candidates.resize(max_per_bucket);
    }
  }
  return result;
}

// =====================================================================
// Client reads
// =====================================================================

std::optional<ClientRecord> InMemoryMasterMetadataStore::GetClient(
    const std::string& node_id) const {
  std::shared_lock lock(mutex_);
  auto it = clients_.find(node_id);
  if (it == clients_.end()) return std::nullopt;
  return it->second;
}

bool InMemoryMasterMetadataStore::IsClientAlive(const std::string& node_id) const {
  std::shared_lock lock(mutex_);
  return IsClientAliveLocked(node_id);
}

std::optional<std::string> InMemoryMasterMetadataStore::GetPeerAddress(
    const std::string& node_id) const {
  std::shared_lock lock(mutex_);
  auto it = clients_.find(node_id);
  if (it == clients_.end()) return std::nullopt;
  return it->second.peer_address;
}

std::vector<ClientRecord> InMemoryMasterMetadataStore::ListAliveClients() const {
  std::shared_lock lock(mutex_);
  std::vector<ClientRecord> result;
  for (const auto& [id, record] : clients_) {
    if (record.status == ClientStatus::ALIVE) result.push_back(record);
  }
  return result;
}

std::unordered_map<std::string, std::string> InMemoryMasterMetadataStore::GetAlivePeerView()
    const {
  std::shared_lock lock(mutex_);
  std::unordered_map<std::string, std::string> view;
  view.reserve(clients_.size());
  for (const auto& [id, record] : clients_) {
    if (record.status != ClientStatus::ALIVE) continue;
    view.emplace(record.node_id, record.peer_address);
  }
  return view;
}

std::size_t InMemoryMasterMetadataStore::AliveClientCount() const {
  std::shared_lock lock(mutex_);
  std::size_t count = 0;
  for (const auto& [id, record] : clients_) {
    if (record.status == ClientStatus::ALIVE) ++count;
  }
  return count;
}

std::vector<std::string> InMemoryMasterMetadataStore::GetClientTags(
    const std::string& node_id) const {
  std::shared_lock lock(mutex_);
  auto it = clients_.find(node_id);
  if (it == clients_.end()) return {};
  return it->second.tags;
}

// =====================================================================
// External-KV reads
// =====================================================================

std::vector<NodeMatch> InMemoryMasterMetadataStore::MatchExternalKv(
    const std::vector<std::string>& hashes, bool count_as_hit,
    std::chrono::system_clock::time_point now) {
  // count_as_hit mutates external_kv_hits_, so take the exclusive lock in that
  // case; a pure read stays shared. This is the one formerly-shared path that
  // becomes exclusive under the single mutex (§2a), but it's one acquisition
  // per RPC, not per hash.
  std::unordered_map<std::string, std::map<TierType, std::vector<std::string>>> acc;

  auto match_into = [&]() {
    for (const auto& hash : hashes) {
      auto it = external_kv_entries_.find(hash);
      if (it == external_kv_entries_.end()) continue;
      for (const auto& [node_id, tiers] : it->second) {
        auto& by_tier = acc[node_id];
        for (TierType tier : tiers) by_tier[tier].push_back(hash);
      }
    }
  };

  if (count_as_hit) {
    std::unique_lock lock(mutex_);
    match_into();
    // Increment each unique matched hash once and stamp last_seen = now.
    std::unordered_set<std::string> matched;
    for (const auto& [node_id, by_tier] : acc) {
      for (const auto& [tier, hs] : by_tier) {
        for (const auto& h : hs) matched.insert(h);
      }
    }
    for (const auto& h : matched) {
      auto& entry = external_kv_hits_[h];
      ++entry.count;
      if (entry.last_seen < now) entry.last_seen = now;
    }
  } else {
    std::shared_lock lock(mutex_);
    match_into();
  }

  std::vector<NodeMatch> result;
  result.reserve(acc.size());
  for (auto& [node_id, by_tier] : acc) {
    NodeMatch m;
    m.node_id = node_id;
    m.hashes_by_tier = std::move(by_tier);
    result.push_back(std::move(m));
  }
  return result;
}

std::vector<ExternalKvHitCountEntry> InMemoryMasterMetadataStore::GetExternalKvHitCounts(
    const std::vector<std::string>& hashes) const {
  std::shared_lock lock(mutex_);
  std::vector<ExternalKvHitCountEntry> out;
  std::unordered_set<std::string> seen;
  seen.reserve(hashes.size());
  for (const auto& hash : hashes) {
    if (!seen.insert(hash).second) continue;
    auto it = external_kv_hits_.find(hash);
    if (it == external_kv_hits_.end()) continue;
    out.push_back(ExternalKvHitCountEntry{hash, it->second.count});
  }
  return out;
}

std::size_t InMemoryMasterMetadataStore::GetExternalKvCount(const std::string& node_id) const {
  std::shared_lock lock(mutex_);
  std::size_t count = 0;
  for (const auto& [hash, nodes] : external_kv_entries_) {
    (void)hash;
    if (nodes.count(node_id)) ++count;
  }
  return count;
}

}  // namespace mori::umbp
