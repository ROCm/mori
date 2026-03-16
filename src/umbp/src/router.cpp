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
#include "umbp/router.h"

#include <spdlog/spdlog.h>

namespace mori::umbp {

Router::Router(BlockIndex& index, ClientRegistry& registry,
               std::unique_ptr<RouteGetStrategy> get_strategy,
               std::unique_ptr<RoutePutStrategy> put_strategy)
    : index_(index), registry_(registry) {
  get_strategy_ =
      get_strategy ? std::move(get_strategy) : std::make_unique<RandomRouteGetStrategy>();
  put_strategy_ =
      put_strategy ? std::move(put_strategy) : std::make_unique<TierAwareMostAvailableStrategy>();
}

std::optional<Location> Router::RouteGet(const std::string& key, const std::string& node_id) {
  auto locations = index_.Lookup(key);

  if (locations.empty()) {
    spdlog::debug("[Router] RouteGet key='{}': not found", key);
    return std::nullopt;
  }

  Location selected = get_strategy_->Select(locations, node_id);
  index_.RecordAccess(key);

  spdlog::debug("[Router] RouteGet key='{}': selected node={}, location={}", key,
                selected.node_id, selected.location_id);
  return selected;
}

std::optional<RoutePutResult> Router::RoutePut(const std::string& key, const std::string& node_id,
                                               uint64_t block_size) {
  auto candidates = registry_.GetAliveClients();

  if (candidates.empty()) {
    spdlog::debug("[Router] RoutePut key='{}' from={}: no alive clients", key, node_id);
    return std::nullopt;
  }

  for (;;) {
    auto result = put_strategy_->Select(candidates, block_size);
    if (!result) {
      spdlog::debug("[Router] RoutePut key='{}' from={}: no node with sufficient capacity", key,
                    node_id);
      return std::nullopt;
    }

    auto alloc = registry_.AllocateForPut(result->node_id, result->tier, block_size);
    if (alloc) {
      result->peer_address = std::move(alloc->peer_address);
      result->engine_desc_bytes = std::move(alloc->engine_desc_bytes);
      result->dram_memory_desc_bytes = std::move(alloc->dram_memory_desc_bytes);
      result->allocated_offset = alloc->allocated_offset;
      result->buffer_index = alloc->buffer_index;
      spdlog::debug("[Router] RoutePut key='{}' from={}: selected node={}, tier={}, offset={}",
                    key, node_id, result->node_id, TierTypeName(result->tier),
                    result->allocated_offset);
      return result;
    }

    spdlog::debug("[Router] RoutePut key='{}': allocation failed on node={} tier={}, retrying",
                  key, result->node_id, TierTypeName(result->tier));
    for (auto& c : candidates) {
      if (c.node_id == result->node_id) {
        c.tier_capacities.erase(result->tier);
      }
    }
  }
}

}  // namespace mori::umbp
