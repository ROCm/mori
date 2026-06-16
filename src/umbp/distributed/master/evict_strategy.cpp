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
#include "umbp/distributed/master/evict_strategy.h"

#include <algorithm>
#include <cstdint>

namespace mori::umbp {

std::unordered_map<std::string, std::vector<std::string>> LruMasterEvictStrategy::SelectVictims(
    std::vector<EvictionCandidate> candidates,
    std::unordered_map<std::string, std::map<TierType, int64_t>> bytes_to_free) {
  // Oldest-access first (LRU).  Depth-aware tiebreaking is intentionally absent
  // — peers don't ship depth in KvEvent, so a pure LRU sort is what we get.
  std::sort(candidates.begin(), candidates.end(),
            [](const EvictionCandidate& a, const EvictionCandidate& b) {
              return a.last_accessed_at < b.last_accessed_at;
            });

  // Greedy: walk oldest-first and take a candidate whenever its (node, tier)
  // still has budget left, decrementing as we go.  Victims are grouped by node
  // so the eventual EvictKey RPC carries a single keys[] per peer.
  std::unordered_map<std::string, std::vector<std::string>> per_node_keys;
  for (const auto& c : candidates) {
    auto& tier_budget = bytes_to_free[c.location.node_id];
    auto it = tier_budget.find(c.location.tier);
    if (it == tier_budget.end() || it->second <= 0) continue;
    per_node_keys[c.location.node_id].push_back(c.key);
    it->second -= static_cast<int64_t>(c.size);
  }
  return per_node_keys;
}

}  // namespace mori::umbp
