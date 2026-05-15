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

#include <map>
#include <set>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

/// Lightweight index for externally-managed KV blocks (e.g. sglang L1/L2/L3
/// cache).  Each (node, hash) pair tracks the *set* of tiers the node has
/// reported the block on — the same block can simultaneously live on HBM
/// (GPU), DRAM (CPU mirror) and SSD (storage backup), and the index keeps
/// every tier so the cost-aware scheduler can see the full physical layout.
class ExternalKvBlockIndex {
 public:
  ExternalKvBlockIndex() = default;
  ~ExternalKvBlockIndex() = default;

  ExternalKvBlockIndex(const ExternalKvBlockIndex&) = delete;
  ExternalKvBlockIndex& operator=(const ExternalKvBlockIndex&) = delete;

  // Add `tier` to the tier set of every (node_id, hash) pair.  Idempotent:
  // re-registering at the same tier is a no-op; registering at a *new* tier
  // adds a bucket without touching existing tiers.
  void Register(const std::string& node_id, const std::vector<std::string>& hashes, TierType tier);

  // Remove `tier` from the tier set of every (node_id, hash) pair.  Other
  // tiers for the same hash are untouched.  When a hash's tier set becomes
  // empty, the (node, hash) entry is dropped.
  void Unregister(const std::string& node_id, const std::vector<std::string>& hashes,
                  TierType tier);

  // Remove `tier` from every hash currently registered by `node_id`.  Used
  // when a node clears or detaches a whole tier (e.g. storage backend wipe).
  void UnregisterByNodeAtTier(const std::string& node_id, TierType tier);

  // Remove all tiers for all hashes registered by `node_id` (bulk, called
  // on node expiry / unregister).
  void UnregisterByNode(const std::string& node_id);

  struct NodeMatch {
    std::string node_id;
    // Matched hashes grouped by tier.  A single hash may appear in MORE
    // THAN ONE tier bucket when the node holds multiple physical copies
    // (e.g. GPU + CPU mirror).  std::map iterates in sorted TierType
    // order, so the first non-empty bucket is the fastest available tier.
    std::map<TierType, std::vector<std::string>> hashes_by_tier;

    // Number of *distinct* matched hashes (size of the union across tiers).
    // NOT the sum of bucket sizes — a hash on HBM+DRAM still counts once.
    size_t MatchedHashCount() const {
      std::unordered_set<std::string_view> seen;
      for (const auto& [tier, hashes] : hashes_by_tier) {
        for (const auto& h : hashes) seen.insert(h);
      }
      return seen.size();
    }
  };

  // Return per-node matches across all queried hashes.
  std::vector<NodeMatch> Match(const std::vector<std::string>& hashes) const;

  // Return the number of distinct hashes (across all tiers) registered for
  // a node.  Used by master metrics.
  size_t GetKvCount(const std::string& node_id) const;

 private:
  // hash -> (node_id -> set<TierType>).  std::set chosen for deterministic
  // iteration in tier order; the cardinality is small (≤ 4 tiers).
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, std::unordered_map<std::string, std::set<TierType>>> entries_;
};

}  // namespace mori::umbp
