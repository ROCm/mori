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
#include "umbp/distributed/routing/route_get_strategy.h"

#include <algorithm>
#include <random>
#include <sstream>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

namespace {

std::string SummarizeLocations(const std::vector<Location>& locations) {
  if (locations.empty()) return "<empty>";
  std::ostringstream oss;
  bool first = true;
  for (const auto& loc : locations) {
    if (!first) oss << ", ";
    first = false;
    oss << loc.node_id << ':' << TierTypeName(loc.tier) << '/' << loc.size;
  }
  return oss.str();
}

// Lower rank = higher read priority.  SSD is the slow cold tier, so it ranks
// last among the real tiers; UNKNOWN sorts after everything so a malformed
// location never wins over a usable one.
int TierReadRank(TierType tier) {
  switch (tier) {
    case TierType::HBM:
      return 0;
    case TierType::DRAM:
      return 1;
    case TierType::SSD:
      return 2;
    default:
      return 3;
  }
}

// Pick a uniformly random element of a non-empty index list using the shared
// thread_local RNG.
size_t PickRandomIndex(const std::vector<size_t>& indices) {
  thread_local std::mt19937 rng{std::random_device{}()};
  std::uniform_int_distribution<size_t> dist(0, indices.size() - 1);
  return indices[dist(rng)];
}

}  // namespace

Location RandomRouteGetStrategy::Select(const std::vector<Location>& locations,
                                        const std::string& /*node_id*/) {
  if (locations.empty()) {
    MORI_UMBP_WARN("[RouteGetStrategy] received empty location set; returning default Location");
    return {};
  }

  if (locations.size() == 1) {
    const auto& single = locations[0];
    MORI_UMBP_DEBUG("[RouteGetStrategy] single candidate selected node={} tier={} size={}",
                    single.node_id, TierTypeName(single.tier), single.size);
    return single;
  }

  thread_local std::mt19937 rng{std::random_device{}()};
  std::uniform_int_distribution<size_t> dist(0, locations.size() - 1);
  size_t choice = dist(rng);
  const auto& selected = locations[choice];
  MORI_UMBP_DEBUG(
      "[RouteGetStrategy] {} candidates -> choice={} node={} tier={} size={}, candidates=[{}]",
      locations.size(), choice, selected.node_id, TierTypeName(selected.tier), selected.size,
      SummarizeLocations(locations));
  return selected;
}

Location TierPriorityRouteGetStrategy::Select(const std::vector<Location>& locations,
                                              const std::string& node_id) {
  if (locations.empty()) {
    MORI_UMBP_WARN(
        "[TierPriorityRouteGetStrategy] received empty location set; returning default Location");
    return {};
  }

  // Find the best (lowest-rank) tier present, then collect every replica on it
  // and choose one at random so load still spreads within a tier.
  int best_rank = TierReadRank(locations[0].tier);
  for (const auto& loc : locations) {
    best_rank = std::min(best_rank, TierReadRank(loc.tier));
  }
  std::vector<size_t> best_tier_indices;
  for (size_t i = 0; i < locations.size(); ++i) {
    if (TierReadRank(locations[i].tier) == best_rank) best_tier_indices.push_back(i);
  }

  // Requester-local preference: if the asking node itself holds a replica in the
  // best tier, serve it locally instead of a random peer. This is what makes
  // cache_remote_fetches pay off — a node that re-cached a remotely-fetched block
  // reads its own copy on the next Get (no RDMA), matching the local-first
  // behavior of the pre-dual-scheme UMBPClient. Non-replica requesters still
  // spread randomly. Empty node_id (unknown caller) falls through to random.
  if (!node_id.empty()) {
    for (size_t i : best_tier_indices) {
      if (locations[i].node_id == node_id) {
        MORI_UMBP_DEBUG(
            "[TierPriorityRouteGetStrategy] requester-local hit node={} tier={} size={}",
            locations[i].node_id, TierTypeName(locations[i].tier), locations[i].size);
        return locations[i];
      }
    }
  }

  size_t choice = PickRandomIndex(best_tier_indices);
  const auto& selected = locations[choice];
  MORI_UMBP_DEBUG(
      "[TierPriorityRouteGetStrategy] {} candidates -> best_tier={} ({} replicas) choice node={} "
      "tier={} size={}, candidates=[{}]",
      locations.size(), TierTypeName(selected.tier), best_tier_indices.size(), selected.node_id,
      TierTypeName(selected.tier), selected.size, SummarizeLocations(locations));
  return selected;
}

}  // namespace mori::umbp
