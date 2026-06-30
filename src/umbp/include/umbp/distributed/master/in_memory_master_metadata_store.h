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

// In-process implementation of IMasterMetadataStore.
//
// This is the single-master / unit-test backend: it folds the four former
// state holders (GlobalBlockIndex, ClientRegistry, ExternalKvBlockIndex,
// ExternalKvHitIndex) into one class behind one std::shared_mutex. The logic
// is a near-verbatim lift of those classes; what changes is the locking
// (four independent lock domains collapse to one) and that every timestamp
// crossing the interface boundary is now caller-supplied system_clock — see
// the hazards in master_metadata_store.h.
//
// Per-hash hit counts live in process memory and are lost on restart, exactly
// as the old ExternalKvHitIndex did; crash-durability is a Redis-backend
// concern only.
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "umbp/distributed/master/master_metadata_store.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

class InMemoryMasterMetadataStore : public IMasterMetadataStore {
 public:
  InMemoryMasterMetadataStore() = default;
  ~InMemoryMasterMetadataStore() override = default;

  InMemoryMasterMetadataStore(const InMemoryMasterMetadataStore&) = delete;
  InMemoryMasterMetadataStore& operator=(const InMemoryMasterMetadataStore&) = delete;

  // --- Cross-store writes ---
  bool RegisterClient(const ClientRegistration& registration,
                      std::chrono::system_clock::time_point now,
                      std::chrono::system_clock::duration stale_after) override;
  void UnregisterClient(const std::string& node_id) override;
  HeartbeatResult ApplyHeartbeat(const std::string& node_id, uint64_t seq,
                                 std::chrono::system_clock::time_point now,
                                 const std::map<TierType, TierCapacity>& caps,
                                 const std::vector<KvEvent>& events, bool is_full_sync) override;
  std::vector<std::string> ExpireStaleClients(
      std::chrono::system_clock::time_point cutoff) override;

  // --- External-KV writes ---
  bool RegisterExternalKvIfAlive(const std::string& node_id, const std::vector<std::string>& hashes,
                                 TierType tier) override;
  void UnregisterExternalKv(const std::string& node_id, const std::vector<std::string>& hashes,
                            TierType tier) override;
  void UnregisterExternalKvByTier(const std::string& node_id, TierType tier) override;
  void UnregisterExternalKvByNode(const std::string& node_id) override;
  std::size_t GarbageCollectHits(std::chrono::system_clock::time_point cutoff) override;

  // --- Block reads ---
  std::vector<Location> LookupBlock(const std::string& key) const override;
  std::vector<Location> LookupBlockForRouteGet(
      const std::string& key, const std::unordered_set<std::string>& exclude_nodes,
      std::chrono::system_clock::time_point now,
      std::chrono::system_clock::duration lease_duration) override;
  std::vector<std::vector<Location>> BatchLookupBlockForRouteGet(
      const std::vector<std::string>& keys, const std::unordered_set<std::string>& exclude_nodes,
      std::chrono::system_clock::time_point now,
      std::chrono::system_clock::duration lease_duration) override;
  std::vector<bool> BatchExistsBlock(const std::vector<std::string>& keys) const override;
  std::map<NodeTierKey, std::vector<EvictionCandidate>> EnumerateEvictionCandidates(
      const std::vector<NodeTierKey>& buckets, EvictionOrder order, size_t max_per_bucket,
      std::chrono::system_clock::time_point now) const override;

  // --- Client reads ---
  std::optional<ClientRecord> GetClient(const std::string& node_id) const override;
  bool IsClientAlive(const std::string& node_id) const override;
  std::optional<std::string> GetPeerAddress(const std::string& node_id) const override;
  std::vector<ClientRecord> ListAliveClients() const override;
  std::unordered_map<std::string, std::string> GetAlivePeerView() const override;
  std::size_t AliveClientCount() const override;
  std::vector<std::string> GetClientTags(const std::string& node_id) const override;

  // --- External-KV reads ---
  std::vector<NodeMatch> MatchExternalKv(const std::vector<std::string>& hashes, bool count_as_hit,
                                         std::chrono::system_clock::time_point now) override;
  std::vector<ExternalKvHitCountEntry> GetExternalKvHitCounts(
      const std::vector<std::string>& hashes) const override;
  std::size_t GetExternalKvCount(const std::string& node_id) const override;

 private:
  // One block's locations + LRU/lease metadata. Lifted from
  // GlobalBlockIndex::BlockEntry, but the lease/access mutators now take a
  // caller-supplied `now` (system_clock) instead of reading the clock
  // internally — the value crosses the store boundary (hazard #7). The
  // lease/access state stays in atomics so the RouteGet path can mutate it
  // under a shared lock, exactly as today (§2a).
  struct BlockEntry {
    std::vector<Location> locations;
    BlockMetrics metrics;

    std::atomic<int64_t> lease_expiry_rep{0};
    std::atomic<int64_t> last_accessed_rep{0};
    std::atomic<uint64_t> atomic_access_count{0};

    void GrantLease(std::chrono::system_clock::time_point now,
                    std::chrono::system_clock::duration duration) {
      auto expiry = now + duration;
      lease_expiry_rep.store(expiry.time_since_epoch().count(), std::memory_order_release);
    }

    bool IsLeased(std::chrono::system_clock::time_point now) const {
      return lease_expiry_rep.load(std::memory_order_acquire) > now.time_since_epoch().count();
    }

    void RecordAccessAtomic(std::chrono::system_clock::time_point now) {
      last_accessed_rep.store(now.time_since_epoch().count(), std::memory_order_release);
      atomic_access_count.fetch_add(1, std::memory_order_relaxed);
    }

    std::chrono::system_clock::time_point GetLastAccessed() const {
      auto rep = last_accessed_rep.load(std::memory_order_acquire);
      return std::chrono::system_clock::time_point(std::chrono::system_clock::duration(rep));
    }
  };

  // Per-hash cumulative hit counter (lifted from ExternalKvHitIndex, collapsed
  // from 256 atomic shards to a single map under mutex_). last_seen is
  // system_clock now that it crosses the boundary and feeds GarbageCollectHits.
  struct HitEntry {
    uint64_t count = 0;
    std::chrono::system_clock::time_point last_seen;
  };

  // --- Locked helpers (caller MUST hold the unique lock) ---
  size_t ApplyEventsLocked(const std::string& node_id, const std::vector<KvEvent>& events,
                           std::chrono::system_clock::time_point now);
  void ReplaceNodeLocationsLocked(const std::string& node_id, const std::vector<KvEvent>& adds,
                                  std::chrono::system_clock::time_point now);
  void RemoveBlocksByNodeLocked(const std::string& node_id);
  void RemoveExternalKvByNodeLocked(const std::string& node_id);
  bool IsClientAliveLocked(const std::string& node_id) const;

  mutable std::shared_mutex mutex_;

  // Block locations (from GlobalBlockIndex).
  std::unordered_map<std::string, BlockEntry> entries_;
  // Reverse index node_id -> keys, so node-scoped removal skips a full scan.
  std::unordered_map<std::string, std::unordered_set<std::string>> node_to_keys_;

  // Client records (from ClientRegistry).
  std::unordered_map<std::string, ClientRecord> clients_;

  // External-KV locations (from ExternalKvBlockIndex): hash -> node -> tier-set.
  // Keyed hash-first so MatchExternalKv (the hot RPC path) stays O(1) per hash.
  std::unordered_map<std::string, std::unordered_map<std::string, std::set<TierType>>>
      external_kv_entries_;

  // Per-hash hit counts (from ExternalKvHitIndex).
  std::unordered_map<std::string, HitEntry> external_kv_hits_;
};

}  // namespace mori::umbp
