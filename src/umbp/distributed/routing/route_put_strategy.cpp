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
#include <random>
#include <sstream>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

// ---------------------------------------------------------------------------
//  Internal helpers (logging, tier order, projected-capacity deduction)
// ---------------------------------------------------------------------------
namespace {

// SSD is intentionally excluded: there is no direct SSD put — the SSD copy is
// filled asynchronously by copy-on-commit.  RoutePut must never steer a put at
// a tier with no direct-put semantics, even though SSD capacity is reported via
// heartbeat.
constexpr std::array<TierType, 2> kPutTierOrder = {TierType::HBM, TierType::DRAM};

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

// Deduct a routed pick's block_size from the batch-local candidates copy so
// later entries in the same batch see the reservation.  A routed result always
// names a node/tier that exists here with enough room (Select / TrySelectOnNode
// only return such picks, and candidates is a single-thread local copy); a
// violation means the selector's contract is broken.  This is a best-effort
// system: on a violation we log a MORI ERROR and return false so the caller
// drops the route (treats the key as unroutable) instead of crashing.
bool ApplyProjectedDeduction(std::vector<ClientRecord>& candidates, const RoutePutResult& result,
                             uint64_t block_size) {
  auto client_it = std::find_if(candidates.begin(), candidates.end(),
                                [&](const ClientRecord& c) { return c.node_id == result.node_id; });
  if (client_it == candidates.end()) {
    MORI_UMBP_ERROR("[RoutePutStrategy] projected-deduction: selected node not in candidates: {}",
                    result.node_id);
    return false;
  }
  auto tier_it = client_it->tier_capacities.find(result.tier);
  if (tier_it == client_it->tier_capacities.end()) {
    MORI_UMBP_ERROR("[RoutePutStrategy] projected-deduction: selected tier absent on node {}",
                    result.node_id);
    return false;
  }
  if (tier_it->second.available_bytes < block_size) {
    MORI_UMBP_ERROR("[RoutePutStrategy] projected-deduction: capacity underflow on node {}",
                    result.node_id);
    return false;
  }
  tier_it->second.available_bytes -= block_size;
  return true;
}

// Indices of candidates that can fit block_size on a single @p tier.
std::vector<size_t> CollectEligibleOnTier(const std::vector<ClientRecord>& candidates,
                                          TierType tier, uint64_t block_size,
                                          const std::unordered_set<std::string>& exclude_nodes) {
  std::vector<size_t> indices;
  for (size_t i = 0; i < candidates.size(); ++i) {
    const auto& client = candidates[i];
    if (exclude_nodes.count(client.node_id)) continue;
    auto it = client.tier_capacities.find(tier);
    if (it == client.tier_capacities.end()) continue;
    if (it->second.available_bytes < block_size) continue;
    indices.push_back(i);
  }
  return indices;
}

RoutePutResult MakeRouted(const ClientRecord& client, TierType tier) {
  return RoutePutResult{
      .outcome = RoutePutOutcome::kRouted,
      .node_id = client.node_id,
      .peer_address = client.peer_address,
      .tier = tier,
  };
}

}  // namespace

// ---------------------------------------------------------------------------
//  RoutePutStrategy (base): default batch planner over the virtual Select()
// ---------------------------------------------------------------------------
std::vector<std::optional<RoutePutResult>> RoutePutStrategy::SelectBatch(
    const std::string& /*requester_node_id*/, const std::vector<uint64_t>& block_sizes,
    const std::vector<bool>& already_exists, std::vector<ClientRecord> candidates,
    const std::unordered_set<std::string>& exclude_nodes) {
  if (already_exists.size() != block_sizes.size()) {
    MORI_UMBP_ERROR(
        "[RoutePutStrategy] SelectBatch: already_exists length ({}) must match block_sizes ({}); "
        "treating every key as unroutable",
        already_exists.size(), block_sizes.size());
    return std::vector<std::optional<RoutePutResult>>(block_sizes.size());
  }

  std::vector<std::optional<RoutePutResult>> results;
  results.reserve(block_sizes.size());

  for (size_t i = 0; i < block_sizes.size(); ++i) {
    if (already_exists[i]) {
      results.push_back(RoutePutResult{.outcome = RoutePutOutcome::kAlreadyExists});
      continue;
    }
    if (candidates.empty()) {
      results.push_back(std::nullopt);
      continue;
    }
    auto selected = Select(candidates, block_sizes[i], exclude_nodes);
    if (selected && selected->outcome == RoutePutOutcome::kRouted &&
        !ApplyProjectedDeduction(candidates, *selected, block_sizes[i])) {
      // Selector broke its own contract: drop the route (best-effort failure).
      selected = std::nullopt;
    }
    results.push_back(std::move(selected));
  }

  return results;
}

// ---------------------------------------------------------------------------
//  TierAwareMostAvailableStrategy
// ---------------------------------------------------------------------------
std::optional<RoutePutResult> TierAwareMostAvailableStrategy::Select(
    const std::vector<ClientRecord>& alive_clients, uint64_t block_size,
    const std::unordered_set<std::string>& exclude_nodes) {
  const std::string exclude_snapshot = FormatExcludeSet(exclude_nodes);

  for (TierType tier : kPutTierOrder) {
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

// ---------------------------------------------------------------------------
//  ConfigurableRoutePutStrategy
// ---------------------------------------------------------------------------
ConfigurableRoutePutStrategy::ConfigurableRoutePutStrategy(SelectAlgo algo, NodeAffinity affinity)
    : algo_(algo), affinity_(affinity) {}

ConfigurableRoutePutStrategy::ConfigurableRoutePutStrategy(SelectAlgo algo, NodeAffinity affinity,
                                                           uint64_t rng_seed)
    : algo_(algo), affinity_(affinity), seeded_(true), rng_(rng_seed) {}

std::string ConfigurableRoutePutStrategy::Describe() const {
  std::string out = (algo_ == SelectAlgo::kRandom) ? "random" : "most_available";
  out += '/';
  switch (affinity_) {
    case NodeAffinity::kSame:
      out += "same";
      break;
    case NodeAffinity::kLocal:
      out += "local";
      break;
    case NodeAffinity::kNone:
    default:
      out += "none";
      break;
  }
  return out;
}

size_t ConfigurableRoutePutStrategy::PickWeighted(const std::vector<uint64_t>& weights) {
  std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
  if (seeded_) {
    std::lock_guard<std::mutex> lock(rng_mutex_);
    return dist(rng_);
  }
  thread_local std::mt19937 rng{std::random_device{}()};
  return dist(rng);
}

std::optional<RoutePutResult> ConfigurableRoutePutStrategy::TrySelectOnNodeTier(
    const std::vector<ClientRecord>& candidates, const std::string& node_id, TierType tier,
    uint64_t block_size, const std::unordered_set<std::string>& exclude_nodes) const {
  if (node_id.empty() || exclude_nodes.count(node_id)) return std::nullopt;
  auto it = std::find_if(candidates.begin(), candidates.end(),
                         [&](const ClientRecord& c) { return c.node_id == node_id; });
  if (it == candidates.end()) return std::nullopt;
  auto cap = it->tier_capacities.find(tier);
  if (cap == it->tier_capacities.end() || cap->second.available_bytes < block_size) {
    return std::nullopt;
  }
  return MakeRouted(*it, tier);
}

std::optional<RoutePutResult> ConfigurableRoutePutStrategy::TrySelectOnNode(
    const std::vector<ClientRecord>& candidates, const std::string& node_id, uint64_t block_size,
    const std::unordered_set<std::string>& exclude_nodes) const {
  for (TierType tier : kPutTierOrder) {
    if (auto r = TrySelectOnNodeTier(candidates, node_id, tier, block_size, exclude_nodes)) {
      return r;
    }
  }
  return std::nullopt;
}

std::optional<RoutePutResult> ConfigurableRoutePutStrategy::SelectByAlgo(
    const std::vector<ClientRecord>& candidates, uint64_t block_size,
    const std::unordered_set<std::string>& exclude_nodes,
    const std::optional<std::string>& preferred_node) {
  for (TierType tier : kPutTierOrder) {
    // Node preference applies only within this tier, so a preferred node that is
    // full on the faster tier never preempts a remote node that still has room
    // there: tier priority is preserved.
    if (preferred_node) {
      if (auto r =
              TrySelectOnNodeTier(candidates, *preferred_node, tier, block_size, exclude_nodes)) {
        return r;
      }
    }
    std::vector<size_t> eligible =
        CollectEligibleOnTier(candidates, tier, block_size, exclude_nodes);
    if (eligible.empty()) continue;

    auto available = [&](size_t idx) {
      return candidates[idx].tier_capacities.at(tier).available_bytes;
    };
    size_t chosen = eligible.front();
    if (algo_ == SelectAlgo::kRandom) {
      std::vector<uint64_t> weights;
      weights.reserve(eligible.size());
      for (size_t idx : eligible) weights.push_back(available(idx));
      chosen = eligible[PickWeighted(weights)];
    } else {
      for (size_t idx : eligible) {
        if (available(idx) > available(chosen)) chosen = idx;
      }
    }
    return MakeRouted(candidates[chosen], tier);
  }
  return std::nullopt;
}

std::optional<RoutePutResult> ConfigurableRoutePutStrategy::Select(
    const std::vector<ClientRecord>& alive_clients, uint64_t block_size,
    const std::unordered_set<std::string>& exclude_nodes) {
  // Single-key default ignores affinity (no batch context, no requester here);
  // batch affinity lives in SelectBatch.
  return SelectByAlgo(alive_clients, block_size, exclude_nodes);
}

std::vector<std::optional<RoutePutResult>> ConfigurableRoutePutStrategy::SelectBatch(
    const std::string& requester_node_id, const std::vector<uint64_t>& block_sizes,
    const std::vector<bool>& already_exists, std::vector<ClientRecord> candidates,
    const std::unordered_set<std::string>& exclude_nodes) {
  if (already_exists.size() != block_sizes.size()) {
    MORI_UMBP_ERROR(
        "[ConfigurableRoutePutStrategy] SelectBatch: already_exists length ({}) must match "
        "block_sizes ({}); treating every key as unroutable",
        already_exists.size(), block_sizes.size());
    return std::vector<std::optional<RoutePutResult>>(block_sizes.size());
  }

  // Affinity anchor: the node (and optionally tier) we try first for each key
  // before the explicit SelectByAlgo fallback.  Affinity biases node choice and
  // never makes a key fail that SelectByAlgo could route.
  //   - kNone:  no anchor; every key goes straight to SelectByAlgo.
  //   - kLocal: anchor fixed to the requester node, tried node-first across its
  //             own tiers (local HBM then local DRAM) before the global
  //             fallback; never re-anchored.  This intentionally prefers a local
  //             DRAM placement over a remote HBM one (locality for later gets).
  //   - kSame:  if one node/tier fits the whole non-dedup total, anchor is
  //             pinned to that exact node AND tier so the batch lands together.
  //             Otherwise each key is placed tier-first with the sticky node
  //             only preferred within a tier (so a spill never beats a remote
  //             HBM with the anchor's DRAM); the anchor re-points to the latest
  //             pick as nodes fill.
  std::optional<std::string> anchor_node;
  std::optional<TierType> anchor_tier;  // pinned only for the kSame whole-batch hit
  if (affinity_ == NodeAffinity::kLocal) {
    if (!requester_node_id.empty()) anchor_node = requester_node_id;
  } else if (affinity_ == NodeAffinity::kSame) {
    uint64_t total = 0;
    for (size_t i = 0; i < block_sizes.size(); ++i) {
      if (!already_exists[i]) total += block_sizes[i];
    }
    if (total > 0) {
      // SelectByAlgo honors HBM-before-DRAM, so a hit here is the fastest tier
      // on which the whole batch fits — pin both node and tier to it.
      if (auto whole = SelectByAlgo(candidates, total, exclude_nodes)) {
        anchor_node = whole->node_id;
        anchor_tier = whole->tier;
      }
    }
  }

  std::vector<std::optional<RoutePutResult>> results;
  results.reserve(block_sizes.size());

  for (size_t i = 0; i < block_sizes.size(); ++i) {
    if (already_exists[i]) {
      results.push_back(RoutePutResult{.outcome = RoutePutOutcome::kAlreadyExists});
      continue;
    }
    if (candidates.empty()) {
      results.push_back(std::nullopt);
      continue;
    }
    const uint64_t block_size = block_sizes[i];

    std::optional<RoutePutResult> selected;
    if (affinity_ == NodeAffinity::kLocal) {
      // Node-first: local HBM -> local DRAM, then global HBM -> DRAM fallback.
      if (anchor_node) {
        selected = TrySelectOnNode(candidates, *anchor_node, block_size, exclude_nodes);
      }
      if (!selected) selected = SelectByAlgo(candidates, block_size, exclude_nodes);
    } else if (affinity_ == NodeAffinity::kSame) {
      // Whole-batch hit: keep every key on the pinned node AND tier.
      if (anchor_tier) {
        selected =
            TrySelectOnNodeTier(candidates, *anchor_node, *anchor_tier, block_size, exclude_nodes);
      }
      if (!selected) {
        // Tier-first with the sticky node only preferred within each tier, so a
        // spill never prefers the anchor's DRAM over a remote node's HBM.  Drop
        // the tier pin and re-anchor to wherever this key actually landed.
        selected = SelectByAlgo(candidates, block_size, exclude_nodes, anchor_node);
        anchor_tier = std::nullopt;
        if (selected && selected->outcome == RoutePutOutcome::kRouted) {
          anchor_node = selected->node_id;
        }
      }
    } else {
      selected = SelectByAlgo(candidates, block_size, exclude_nodes);
    }

    if (selected && selected->outcome == RoutePutOutcome::kRouted &&
        !ApplyProjectedDeduction(candidates, *selected, block_size)) {
      // Selector broke its own contract: drop the route (best-effort failure).
      selected = std::nullopt;
    }
    results.push_back(std::move(selected));
  }

  return results;
}

}  // namespace mori::umbp
