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
  get_strategy_ =
      get_strategy ? std::move(get_strategy) : std::make_unique<RandomRouteGetStrategy>();
  put_strategy_ =
      put_strategy ? std::move(put_strategy) : std::make_unique<TierAwareMostAvailableStrategy>();
}

std::optional<RouteGetResolution> Router::RouteGet(
    const std::string& key, const std::string& node_id,
    const std::unordered_set<std::string>& exclude_nodes) {
  auto locations = index_.Lookup(key);
  if (locations.empty()) {
    MORI_UMBP_DEBUG("[Router] RouteGet key='{}': not found", key);
    return std::nullopt;
  }

  if (!exclude_nodes.empty()) {
    locations.erase(
        std::remove_if(locations.begin(), locations.end(),
                       [&](const Location& l) { return exclude_nodes.count(l.node_id); }),
        locations.end());
    if (locations.empty()) {
      MORI_UMBP_DEBUG("[Router] RouteGet key='{}': every replica excluded", key);
      return std::nullopt;
    }
  }

  Location selected = get_strategy_->Select(locations, node_id);
  index_.RecordAccess(key);
  index_.GrantLease(key, lease_duration_);

  // Resolve the peer address for the chosen replica from the registry.
  // Routing-time read; the reader doesn't need to round-trip master to
  // discover where to RDMA from.
  std::string peer_address;
  for (const auto& client : registry_.GetAliveClients()) {
    if (client.node_id == selected.node_id) {
      peer_address = client.peer_address;
      break;
    }
  }

  RouteGetResolution out;
  out.location = selected;
  out.peer_address = std::move(peer_address);
  MORI_UMBP_DEBUG("[Router] RouteGet key='{}': selected node={}, tier={}, size={}", key,
                  selected.node_id, TierTypeName(selected.tier), selected.size);
  return out;
}

std::optional<RoutePutResult> Router::RoutePut(
    const std::string& key, const std::string& node_id, uint64_t block_size,
    const std::unordered_set<std::string>& exclude_nodes) {
  // Master-side dedup lives only in BatchRoutePut (single RoutePut
  // proto carries no already_exists; PoolClient::Put wraps BatchPut).
  (void)key;
  auto candidates = registry_.GetAliveClients();
  if (candidates.empty()) {
    MORI_UMBP_DEBUG("[Router] RoutePut from={}: no alive clients", node_id);
    return std::nullopt;
  }
  auto selected = put_strategy_->Select(candidates, block_size, exclude_nodes);
  if (!selected) {
    MORI_UMBP_DEBUG("[Router] RoutePut from={}: no node with sufficient capacity", node_id);
    return std::nullopt;
  }
  return selected;
}

std::vector<std::optional<RoutePutResult>> Router::BatchRoutePut(
    const std::vector<std::string>& keys, const std::string& node_id,
    const std::vector<uint64_t>& block_sizes,
    const std::unordered_set<std::string>& exclude_nodes) {
  (void)node_id;
  std::vector<std::optional<RoutePutResult>> results(keys.size());
  // Single shared_lock for the whole batch: dedup mask + alive snapshot.
  // Two entries picking the same (node, tier) is fine — peer will sort
  // out ENOSPC at AllocateSlot.
  auto exists_mask = index_.BatchLookupExists(keys);
  auto candidates = registry_.GetAliveClients();
  for (size_t i = 0; i < keys.size(); ++i) {
    if (i < exists_mask.size() && exists_mask[i]) {
      results[i] = RoutePutResult{.outcome = RoutePutOutcome::kAlreadyExists};
      continue;
    }
    if (candidates.empty()) continue;
    results[i] = put_strategy_->Select(candidates, block_sizes[i], exclude_nodes);
  }
  return results;
}

std::vector<std::optional<RouteGetResolution>> Router::BatchRouteGet(
    const std::vector<std::string>& keys, const std::string& node_id,
    const std::unordered_set<std::string>& exclude_nodes) {
  std::vector<std::optional<RouteGetResolution>> results(keys.size());

  // Snapshot peer addresses once for the whole batch instead of pulling
  // them per entry.  Master assumes the snapshot is stable for the
  // duration of one BatchRouteGet.
  std::unordered_map<std::string, std::string> node_to_peer;
  for (const auto& client : registry_.GetAliveClients()) {
    node_to_peer[client.node_id] = client.peer_address;
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    auto locations = index_.Lookup(keys[i]);
    if (locations.empty()) continue;
    if (!exclude_nodes.empty()) {
      locations.erase(
          std::remove_if(locations.begin(), locations.end(),
                         [&](const Location& l) { return exclude_nodes.count(l.node_id); }),
          locations.end());
      if (locations.empty()) continue;
    }
    Location selected = get_strategy_->Select(locations, node_id);
    index_.RecordAccess(keys[i]);
    index_.GrantLease(keys[i], lease_duration_);

    RouteGetResolution out;
    out.location = selected;
    auto it = node_to_peer.find(selected.node_id);
    if (it != node_to_peer.end()) out.peer_address = it->second;
    results[i] = std::move(out);
  }
  return results;
}

}  // namespace mori::umbp
