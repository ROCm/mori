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
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

class ClientRegistry;

struct BlockEntry {
  std::vector<Location> locations;
  BlockMetrics metrics;

  // Atomic lease: steady_clock duration rep, lock-free
  std::atomic<int64_t> lease_expiry_rep{0};

  // Atomic access tracking, lock-free
  std::atomic<int64_t> last_accessed_rep{0};
  std::atomic<uint64_t> atomic_access_count{0};

  // Prefix-aware: depth from BatchPutWithDepth (-1 = unknown)
  int32_t depth = -1;

  void GrantLease(std::chrono::steady_clock::duration duration) {
    auto expiry = std::chrono::steady_clock::now() + duration;
    lease_expiry_rep.store(expiry.time_since_epoch().count(), std::memory_order_release);
  }

  bool IsLeased() const {
    auto now_rep = std::chrono::steady_clock::now().time_since_epoch().count();
    return lease_expiry_rep.load(std::memory_order_acquire) > now_rep;
  }

  void RecordAccessAtomic() {
    last_accessed_rep.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                            std::memory_order_release);
    atomic_access_count.fetch_add(1, std::memory_order_relaxed);
  }

  std::chrono::steady_clock::time_point GetLastAccessed() const {
    auto rep = last_accessed_rep.load(std::memory_order_acquire);
    return std::chrono::steady_clock::time_point(std::chrono::steady_clock::duration(rep));
  }
};

struct EvictionCandidate {
  std::string key;
  Location location;
  std::chrono::steady_clock::time_point last_accessed_at;
  int32_t depth;
  uint64_t size;
};

class GlobalBlockIndex {
 public:
  GlobalBlockIndex() = default;
  ~GlobalBlockIndex() = default;

  GlobalBlockIndex(const GlobalBlockIndex&) = delete;
  GlobalBlockIndex& operator=(const GlobalBlockIndex&) = delete;

  void SetClientRegistry(ClientRegistry* registry);

  // --- Mutators ---
  void Register(const std::string& node_id, const std::string& key, const Location& location);

  bool Unregister(const std::string& node_id, const std::string& key, const Location& location);

  size_t UnregisterByNode(const std::string& key, const std::string& node_id);

  // Batch variants — single lock acquisition for the entire batch.
  size_t BatchRegister(const std::string& node_id,
                       const std::vector<std::pair<std::string, Location>>& entries);
  size_t BatchUnregister(const std::string& node_id,
                         const std::vector<std::pair<std::string, Location>>& entries);

  // Bump last_accessed_at and access_count. Called by Router on RouteGet.
  void RecordAccess(const std::string& key);

  // Grant a time-limited lease to protect a key from eviction.
  void GrantLease(const std::string& key, std::chrono::steady_clock::duration duration);

  void SetDepth(const std::string& key, int32_t depth);

  // --- Queries ---
  std::vector<Location> Lookup(const std::string& key) const;

  // Returns metrics for a key, or nullopt if the key doesn't exist.
  std::optional<BlockMetrics> GetMetrics(const std::string& key) const;

  // --- Eviction ---

  struct NodeTierKey {
    std::string node_id;
    TierType tier;
    bool operator<(const NodeTierKey& o) const {
      if (node_id != o.node_id) return node_id < o.node_id;
      return tier < o.tier;
    }
    bool operator==(const NodeTierKey& o) const { return node_id == o.node_id && tier == o.tier; }
  };

  std::vector<EvictionCandidate> FindEvictionCandidates(
      const std::set<NodeTierKey>& overloaded_node_tiers) const;

  // Double-check and remove entries (write lock). Returns actually evicted candidates.
  std::vector<EvictionCandidate> EvictEntries(const std::vector<EvictionCandidate>& victims);

 private:
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, BlockEntry> entries_;
  ClientRegistry* registry_ = nullptr;
};

}  // namespace mori::umbp
