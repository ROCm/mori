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

[[maybe_unused]] std::string SummarizeLocations(const std::vector<Location>& locations) {
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

// Core selection cores shared by the single-key Select and the batched
// BatchSelect.  Precondition: `locations` is non-empty (callers guarantee it).
Location PickRandomReplica(const std::vector<Location>& locations) {
  if (locations.size() == 1) {
    const auto& single = locations[0];
    // NOTE: temporarily disabled — the macro args are evaluated even when the
    // log level suppresses output; SummarizeLocations() dominated the hot path.
    // MORI_UMBP_DEBUG("[RouteGetStrategy] single candidate selected node={} tier={} size={}",
    //                 single.node_id, TierTypeName(single.tier), single.size);
    return single;
  }

  thread_local std::mt19937 rng{std::random_device{}()};
  std::uniform_int_distribution<size_t> dist(0, locations.size() - 1);
  size_t choice = dist(rng);
  const auto& selected = locations[choice];
  // NOTE: temporarily disabled — see above; SummarizeLocations() ran on every call.
  // MORI_UMBP_DEBUG(
  //     "[RouteGetStrategy] {} candidates -> choice={} node={} tier={} size={}, candidates=[{}]",
  //     locations.size(), choice, selected.node_id, TierTypeName(selected.tier), selected.size,
  //     SummarizeLocations(locations));
  return selected;
}

Location PickLocalPreferringReplica(const std::vector<Location>& locations,
                                    const std::string& node_id) {
  // No tier ranking: every medium is equivalent, so any replica is as good as
  // any other and the choice is made on locality and load alone (backend-
  // agnostic refactor Phase 4 — the HBM > DRAM > SSD order is deleted, not
  // replaced).  Re-introducing a preference means advertising it from the
  // backend, not hardcoding a tier list here.
  //
  // Requester-local preference: if the asking node itself holds a replica, serve
  // it locally instead of a random peer. This is what makes cache_remote_fetches
  // pay off — a node that re-cached a remotely-fetched block reads its own copy
  // on the next Get (no RDMA), matching the local-first behavior of the
  // pre-dual-scheme UMBPClient. Non-replica requesters still spread randomly.
  // Empty node_id (unknown caller) falls through to random.
  if (!node_id.empty()) {
    for (const auto& loc : locations) {
      if (loc.node_id == node_id) {
        MORI_UMBP_DEBUG(
            "[LocalPreferringRouteGetStrategy] requester-local hit node={} tier={} size={}",
            loc.node_id, TierTypeName(loc.tier), loc.size);
        return loc;
      }
    }
  }

  return PickRandomReplica(locations);
}

}  // namespace

// Base default: one virtual dispatch per key.  Kept so custom strategies that
// only override Select() get a working BatchSelect for free.  An empty
// candidate list is left as a default Location (the caller treats it as "not
// routed") and Select() is not invoked — mirroring the router's pre-batch
// skip and avoiding a spurious empty-set WARN.
std::vector<Location> RouteGetStrategy::BatchSelect(
    const std::vector<std::vector<Location>>& per_key_locations, const std::string& node_id) {
  std::vector<Location> out(per_key_locations.size());
  for (size_t i = 0; i < per_key_locations.size(); ++i) {
    if (per_key_locations[i].empty()) continue;
    out[i] = Select(per_key_locations[i], node_id);
  }
  return out;
}

Location RandomRouteGetStrategy::Select(const std::vector<Location>& locations,
                                        const std::string& /*node_id*/) {
  if (locations.empty()) {
    MORI_UMBP_WARN("[RouteGetStrategy] received empty location set; returning default Location");
    return {};
  }
  return PickRandomReplica(locations);
}

std::vector<Location> RandomRouteGetStrategy::BatchSelect(
    const std::vector<std::vector<Location>>& per_key_locations, const std::string& /*node_id*/) {
  std::vector<Location> out(per_key_locations.size());
  for (size_t i = 0; i < per_key_locations.size(); ++i) {
    if (per_key_locations[i].empty()) continue;  // not routed; leave default
    out[i] = PickRandomReplica(per_key_locations[i]);
  }
  return out;
}

Location LocalPreferringRouteGetStrategy::Select(const std::vector<Location>& locations,
                                                 const std::string& node_id) {
  if (locations.empty()) {
    MORI_UMBP_WARN(
        "[LocalPreferringRouteGetStrategy] received empty location set; returning default "
        "Location");
    return {};
  }
  return PickLocalPreferringReplica(locations, node_id);
}

std::vector<Location> LocalPreferringRouteGetStrategy::BatchSelect(
    const std::vector<std::vector<Location>>& per_key_locations, const std::string& node_id) {
  std::vector<Location> out(per_key_locations.size());
  for (size_t i = 0; i < per_key_locations.size(); ++i) {
    if (per_key_locations[i].empty()) continue;  // not routed; leave default
    out[i] = PickLocalPreferringReplica(per_key_locations[i], node_id);
  }
  return out;
}

}  // namespace mori::umbp
