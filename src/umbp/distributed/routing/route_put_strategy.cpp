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
#include "umbp/distributed/routing/route_put_strategy.h"

#include <algorithm>
#include <array>
#include <sstream>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

namespace {

std::string JoinStrings(const std::vector<std::string>& items) {
  if (items.empty()) return "";
  std::ostringstream oss;
  bool first = true;
  for (const auto& item : items) {
    if (!first) oss << ", ";
    first = false;
    oss << item;
  }
  return oss.str();
}

std::string FormatExcludeSet(const std::unordered_set<std::string>& exclude_nodes) {
  if (exclude_nodes.empty()) return "<none>";
  std::vector<std::string> nodes;
  nodes.reserve(exclude_nodes.size());
  for (const auto& node : exclude_nodes) nodes.push_back(node);
  std::sort(nodes.begin(), nodes.end());
  return JoinStrings(nodes);
}

std::string SummarizeClientTiers(const std::vector<ClientRecord>& alive_clients) {
  if (alive_clients.empty()) return "<no-alive-clients>";
  std::vector<std::string> summaries;
  summaries.reserve(alive_clients.size());
  for (const auto& client : alive_clients) {
    std::ostringstream tiers;
    bool first = true;
    for (const auto& kv : client.tier_capacities) {
      if (!first) tiers << ", ";
      first = false;
      tiers << TierTypeName(kv.first) << '=' << kv.second.available_bytes;
    }
    if (first) tiers << "<no-tiers>";
    summaries.push_back(client.node_id + "[" + tiers.str() + "]");
  }
  std::sort(summaries.begin(), summaries.end());
  return JoinStrings(summaries);
}

}  // namespace

std::optional<RoutePutResult> TierAwareMostAvailableStrategy::Select(
    const std::vector<ClientRecord>& alive_clients, uint64_t block_size,
    const std::unordered_set<std::string>& exclude_nodes) {
  static constexpr std::array<TierType, 3> kTierOrder = {TierType::HBM, TierType::DRAM,
                                                         TierType::SSD};

  const std::string exclude_snapshot = FormatExcludeSet(exclude_nodes);

  for (TierType tier : kTierOrder) {
    const ClientRecord* best = nullptr;
    uint64_t best_available = 0;
    uint64_t candidates_considered = 0;

    for (const auto& client : alive_clients) {
      if (exclude_nodes.count(client.node_id)) continue;
      auto it = client.tier_capacities.find(tier);
      if (it == client.tier_capacities.end()) continue;
      if (it->second.available_bytes < block_size) continue;
      ++candidates_considered;
      if (best == nullptr || it->second.available_bytes > best_available) {
        best = &client;
        best_available = it->second.available_bytes;
      }
    }

    if (best != nullptr) {
      MORI_UMBP_INFO(
          "[RoutePutStrategy] block_size={} tier={} selected node={} available_bytes={} "
          "candidates={} excludes=[{}]",
          block_size, TierTypeName(tier), best->node_id, best_available, candidates_considered,
          exclude_snapshot);
      return RoutePutResult{
          .outcome = RoutePutOutcome::kRouted,
          .node_id = best->node_id,
          .peer_address = best->peer_address,
          .tier = tier,
      };
    }

    MORI_UMBP_DEBUG(
        "[RoutePutStrategy] block_size={} tier={} no eligible node (candidates_checked={} "
        "excludes=[{}])",
        block_size, TierTypeName(tier), candidates_considered, exclude_snapshot);
  }

  MORI_UMBP_WARN(
      "[RoutePutStrategy] block_size={} no suitable target. excludes=[{}] capacity_snapshot=[{}]",
      block_size, exclude_snapshot, SummarizeClientTiers(alive_clients));
  return std::nullopt;
}

}  // namespace mori::umbp
