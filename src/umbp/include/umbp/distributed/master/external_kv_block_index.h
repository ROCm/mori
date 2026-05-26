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

/// Lightweight index for externally-managed KV blocks (e.g. sglang HiCache).
/// Each (node, hash) pair tracks the set of tiers the node has reported.
class ExternalKvBlockIndex {
 public:
  ExternalKvBlockIndex() = default;
  ~ExternalKvBlockIndex() = default;

  ExternalKvBlockIndex(const ExternalKvBlockIndex&) = delete;
  ExternalKvBlockIndex& operator=(const ExternalKvBlockIndex&) = delete;

  // Mutators return the count of actually changed (hash, node, tier) tuples.
  size_t Register(const std::string& node_id, const std::vector<std::string>& hashes,
                  TierType tier);
  size_t Unregister(const std::string& node_id, const std::vector<std::string>& hashes,
                    TierType tier);
  size_t UnregisterByNodeAtTier(const std::string& node_id, TierType tier);
  size_t UnregisterByNode(const std::string& node_id);

  struct NodeMatch {
    std::string node_id;
    std::map<TierType, std::vector<std::string>> hashes_by_tier;

    size_t MatchedHashCount() const {
      std::unordered_set<std::string_view> seen;
      for (const auto& [tier, hashes] : hashes_by_tier) {
        for (const auto& h : hashes) seen.insert(h);
      }
      return seen.size();
    }
  };

  std::vector<NodeMatch> Match(const std::vector<std::string>& hashes) const;
  size_t GetKvCount(const std::string& node_id) const;

 private:
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, std::unordered_map<std::string, std::set<TierType>>> entries_;
};

}  // namespace mori::umbp
