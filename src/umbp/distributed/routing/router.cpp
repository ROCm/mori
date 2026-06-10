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
#include "umbp/distributed/routing/router.h"

#include <algorithm>
#include <unordered_map>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

Router::Router(GlobalBlockIndex& index, ClientRegistry& registry,
               std::unique_ptr<RouteGetStrategy> get_strategy,
               std::unique_ptr<RoutePutStrategy> put_strategy)
    : index_(index), registry_(registry) {
  // Default to tier-priority (HBM > DRAM > SSD): with the SSD cold tier live, a
  // random pick could route a key that also has a DRAM/HBM copy to the slow
  // SSD.  Callers can still inject RandomRouteGetStrategy (or any other) via
  // config_.get_strategy.
  get_strategy_ =
      get_strategy ? std::move(get_strategy) : std::make_unique<TierPriorityRouteGetStrategy>();
  put_strategy_ =
      put_strategy ? std::move(put_strategy) : std::make_unique<TierAwareMostAvailableStrategy>();
}

std::optional<RouteGetResolution> Router::RouteGet(
    const std::string& key, const std::string& node_id,
    const std::unordered_set<std::string>& exclude_nodes) {
  auto results = BatchRouteGet({key}, node_id, exclude_nodes);
  return std::move(results.front());
}

std::optional<RoutePutResult> Router::RoutePut(
    const std::string& key, const std::string& node_id, uint64_t block_size,
    const std::unordered_set<std::string>& exclude_nodes) {
  // Master-side dedup lives only in BatchRoutePut (single RoutePut proto carries
  // no already_exists; PoolClient::Put wraps BatchPut).  Route through
  // SelectBatch(size=1) so node-affinity logic has a single home; this is the
  // base most-available/none default's per-key path, so behavior is unchanged.
  (void)key;
  auto candidates = registry_.GetAliveClients();
  if (candidates.empty()) {
    MORI_UMBP_DEBUG("[Router] RoutePut from={}: no alive clients", node_id);
    return std::nullopt;
  }
  auto results = put_strategy_->SelectBatch(node_id, {block_size}, {false}, std::move(candidates),
                                            exclude_nodes);
  if (!results.front()) {
    MORI_UMBP_DEBUG("[Router] RoutePut from={}: no node with sufficient capacity", node_id);
  }
  return std::move(results.front());
}

std::vector<std::optional<RoutePutResult>> Router::BatchRoutePut(
    const std::vector<std::string>& keys, const std::string& node_id,
    const std::vector<uint64_t>& block_sizes,
    const std::unordered_set<std::string>& exclude_nodes) {
  // SelectBatch applies dedup + projected capacity on this batch-local snapshot;
  // the peer allocator stays the final ENOSPC arbiter. A keys/block_sizes length
  // mismatch surfaces as a SelectBatch throw (no silent coercion).
  auto exists_mask = index_.BatchLookupExists(keys);
  auto candidates = registry_.GetAliveClients();
  return put_strategy_->SelectBatch(node_id, block_sizes, exists_mask, std::move(candidates),
                                    exclude_nodes);
}

std::vector<std::optional<RouteGetResolution>> Router::BatchRouteGet(
    const std::vector<std::string>& keys, const std::string& node_id,
    const std::unordered_set<std::string>& exclude_nodes) {
  std::vector<std::optional<RouteGetResolution>> results(keys.size());

  // Snapshot peer addresses once for the whole batch.  Master assumes
  // the snapshot is stable for the duration of one BatchRouteGet.
  std::unordered_map<std::string, std::string> node_to_peer;
  for (const auto& client : registry_.GetAliveClients()) {
    node_to_peer[client.node_id] = client.peer_address;
  }

  auto all_locs = index_.BatchLookupForRouteGet(keys, exclude_nodes, lease_duration_);
  for (size_t i = 0; i < keys.size(); ++i) {
    auto& locations = all_locs[i];
    if (locations.empty()) {
      MORI_UMBP_DEBUG(
          "[Router] BatchRouteGet key='{}': not routed (missing or every replica excluded)",
          keys[i]);
      continue;
    }
    Location selected = get_strategy_->Select(locations, node_id);

    RouteGetResolution out;
    out.location = selected;
    auto it = node_to_peer.find(selected.node_id);
    if (it != node_to_peer.end()) out.peer_address = it->second;
    MORI_UMBP_DEBUG("[Router] BatchRouteGet key='{}': selected node={}, tier={}, size={}", keys[i],
                    selected.node_id, TierTypeName(selected.tier), selected.size);
    results[i] = std::move(out);
  }
  return results;
}

}  // namespace mori::umbp
