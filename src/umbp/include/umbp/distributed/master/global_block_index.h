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
#pragma once

#include <atomic>
#include <chrono>
#include <optional>
#include <set>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

struct BlockEntry {
  std::vector<Location> locations;
  BlockMetrics metrics;

  std::atomic<int64_t> lease_expiry_rep{0};
  std::atomic<int64_t> last_accessed_rep{0};
  std::atomic<uint64_t> atomic_access_count{0};

  void GrantLease(std::chrono::system_clock::duration duration) {
    auto expiry = std::chrono::system_clock::now() + duration;
    lease_expiry_rep.store(expiry.time_since_epoch().count(), std::memory_order_release);
  }

  bool IsLeased() const {
    auto now_rep = std::chrono::system_clock::now().time_since_epoch().count();
    return lease_expiry_rep.load(std::memory_order_acquire) > now_rep;
  }

  void RecordAccessAtomic() {
    last_accessed_rep.store(std::chrono::system_clock::now().time_since_epoch().count(),
                            std::memory_order_release);
    atomic_access_count.fetch_add(1, std::memory_order_relaxed);
  }

  std::chrono::system_clock::time_point GetLastAccessed() const {
    auto rep = last_accessed_rep.load(std::memory_order_acquire);
    return std::chrono::system_clock::time_point(std::chrono::system_clock::duration(rep));
  }
};

struct EvictionCandidate {
  std::string key;
  Location location;
  std::chrono::system_clock::time_point last_accessed_at;
  uint64_t size;
};

// Master-side projection of every peer's owned-key set.  In the
// master-as-advisor design this index is *only* mutated through the
// event-shipping heartbeat — there are no per-Put or per-Eviction
// master RPCs.  Routing and eviction read from here.
class GlobalBlockIndex {
 public:
  GlobalBlockIndex() = default;
  ~GlobalBlockIndex() = default;

  GlobalBlockIndex(const GlobalBlockIndex&) = delete;
  GlobalBlockIndex& operator=(const GlobalBlockIndex&) = delete;

  // --- Mutators (event-driven only) ---

  // Apply one peer's heartbeat-shipped event batch.  Returns the count
  // of location mutations.  ADD with a (node_id, tier) that already exists
  // for the key is an idempotent no-op on the location's size.
  // REMOVE for an unknown (key, node_id, tier) is a silent no-op.
  // CLEAR_AT_TIER drops every key for (node_id, tier) and returns
  // the number of locations removed.
  size_t ApplyEvents(const std::string& node_id, const std::vector<KvEvent>& events);

  // Replace this node's full set of locations with the ADDs carried in `adds`.
  void ReplaceNodeLocations(const std::string& node_id, const std::vector<KvEvent>& adds);

  void RemoveByNode(const std::string& node_id);

  // Bump last_accessed_at and access_count.  Lock-free under the shared lock.
  void RecordAccess(const std::string& key);

  // Grant a time-limited lease to protect a key from eviction.
  void GrantLease(const std::string& key, std::chrono::system_clock::duration duration);

  // Batched Lookup + filter + (on non-empty result) RecordAccess + GrantLease,
  // under a single shared_lock.
  std::vector<std::vector<Location>> BatchLookupForRouteGet(
      const std::vector<std::string>& keys, const std::unordered_set<std::string>& exclude_nodes,
      std::chrono::system_clock::duration lease_duration);

  // --- Queries ---

  std::vector<Location> Lookup(const std::string& key) const;

  // Batched existence check — single shared_lock acquisition for the
  // whole batch.  Read-only, no access-count or lease side-effects.
  // Returns a vector parallel to `keys` where entry i is true iff the
  // key has at least one registered Location.
  std::vector<bool> BatchLookupExists(const std::vector<std::string>& keys) const;

  std::optional<BlockMetrics> GetMetrics(const std::string& key) const;

  // --- Eviction ---

  // Hoisted to umbp/distributed/types.h; alias kept temporarily so existing
  // callers (e.g. GlobalBlockIndex::NodeTierKey) compile.  Removed in Phase 5.
  using NodeTierKey = mori::umbp::NodeTierKey;

  std::vector<EvictionCandidate> FindEvictionCandidates(
      const std::set<NodeTierKey>& overloaded_node_tiers) const;

 private:
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, BlockEntry> entries_;
  // Reverse index: lets ReplaceNodeLocations skip a full entries_ scan.
  std::unordered_map<std::string, std::unordered_set<std::string>> node_to_keys_;
};

}  // namespace mori::umbp
