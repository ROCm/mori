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
#include <map>
#include <random>
#include <sstream>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/env_time.h"
#include "umbp/distributed/batch_dist_log.h"

namespace mori::umbp {

// ---------------------------------------------------------------------------
//  Internal helpers (logging, tier order, projected-capacity deduction)
// ---------------------------------------------------------------------------
namespace {

// Tier priority, fastest first.  SSD is included only when the strategy's
// SsdPutMode allows it (see ConfigurableRoutePutStrategy::TierOrder) — with the
// direct-SSD put path there IS now a put semantics for SSD, but a mixed node
// must not silently demote a put to its cold tier.
std::vector<TierType> BuildTierOrder(SsdPutMode mode) {
  std::vector<TierType> order = {TierType::HBM, TierType::DRAM};
  if (mode != SsdPutMode::kNever) order.push_back(TierType::SSD);
  return order;
}

// Total (not available) DRAM+HBM bytes a node reports.  Zero means the node has
// no host-memory tier at all — a pure-SSD node — as opposed to a node whose
// DRAM merely happens to be full right now.
uint64_t HostMemoryTotalBytes(const ClientRecord& client) {
  uint64_t total = 0;
  for (TierType t : {TierType::HBM, TierType::DRAM}) {
    auto it = client.tier_capacities.find(t);
    if (it != client.tier_capacities.end()) total += it->second.total_bytes;
  }
  return total;
}

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

// Indices of candidates that can fit block_size on a single @p tier and pass
// @p tier_eligible (the SSD-put gate; always true for HBM/DRAM).
template <typename EligibleFn>
std::vector<size_t> CollectEligibleOnTier(const std::vector<ClientRecord>& candidates,
                                          TierType tier, uint64_t block_size,
                                          const std::unordered_set<std::string>& exclude_nodes,
                                          const EligibleFn& tier_eligible) {
  std::vector<size_t> indices;
  for (size_t i = 0; i < candidates.size(); ++i) {
    const auto& client = candidates[i];
    if (exclude_nodes.count(client.node_id)) continue;
    if (!tier_eligible(client, tier)) continue;
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

// Available bytes for @p node_id on @p tier in the (projected) candidates copy;
// 0 when the node or tier is absent.  Used only to report the pre-deduction
// figure in the routed INFO log.
uint64_t LookupAvailableBytes(const std::vector<ClientRecord>& candidates,
                              const std::string& node_id, TierType tier) {
  auto client_it = std::find_if(candidates.begin(), candidates.end(),
                                [&](const ClientRecord& c) { return c.node_id == node_id; });
  if (client_it == candidates.end()) return 0;
  auto tier_it = client_it->tier_capacities.find(tier);
  if (tier_it == client_it->tier_capacities.end()) return 0;
  return tier_it->second.available_bytes;
}

// Deduct a routed pick's @p block_size from the batch-local @p candidates copy
// so later entries in the same batch see the reservation.  A routed result
// always names a node/tier that exists here with enough room; a violation means
// the selector's contract is broken.  Best-effort: on a violation log a MORI
// ERROR and return false (caller drops the route, treats the key as unroutable)
// instead of crashing.  Returns true after a successful deduction.
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

// Log one batch-distribution line every N batches; 0 disables the line
// entirely.  Read once (the value cannot change under a running master).
uint32_t PutDistLogEvery() {
  static const uint32_t v = GetEnvUint32("UMBP_PUT_DIST_LOG", 1);
  return v;
}

}  // namespace

// ---------------------------------------------------------------------------
//  ConfigurableRoutePutStrategy
// ---------------------------------------------------------------------------
ConfigurableRoutePutStrategy::ConfigurableRoutePutStrategy(SelectAlgo algo, NodeAffinity affinity,
                                                           SsdPutMode ssd_mode)
    : algo_(algo),
      affinity_(affinity),
      ssd_mode_(ssd_mode),
      tier_order_(BuildTierOrder(ssd_mode)) {}

ConfigurableRoutePutStrategy::ConfigurableRoutePutStrategy(SelectAlgo algo, NodeAffinity affinity,
                                                           uint64_t rng_seed, SsdPutMode ssd_mode)
    : algo_(algo),
      affinity_(affinity),
      ssd_mode_(ssd_mode),
      tier_order_(BuildTierOrder(ssd_mode)),
      seeded_(true),
      rng_(rng_seed) {}

bool ConfigurableRoutePutStrategy::SsdEligibleOnNode(const ClientRecord& client) const {
  switch (ssd_mode_) {
    case SsdPutMode::kAlways:
      return true;
    case SsdPutMode::kNever:
      return false;
    case SsdPutMode::kAuto:
    default:
      // Pure-SSD node only.  Keyed on TOTAL host-memory bytes, not available:
      // a node whose DRAM is merely full still wants the put to go to a peer
      // with free DRAM rather than to its own cold tier.
      return HostMemoryTotalBytes(client) == 0;
  }
}

bool ConfigurableRoutePutStrategy::TierEligibleOnNode(const ClientRecord& client,
                                                      TierType tier) const {
  if (tier != TierType::SSD) return true;
  return SsdEligibleOnNode(client);
}

std::string ConfigurableRoutePutStrategy::Describe() const {
  std::string out;
  switch (algo_) {
    case SelectAlgo::kRandom:
      out = "random";
      break;
    case SelectAlgo::kRoundRobin:
      out = "round_robin";
      break;
    case SelectAlgo::kMostAvailable:
    default:
      out = "most_available";
      break;
  }
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
  out += '/';
  switch (ssd_mode_) {
    case SsdPutMode::kAlways:
      out += "ssd:always";
      break;
    case SsdPutMode::kNever:
      out += "ssd:never";
      break;
    case SsdPutMode::kAuto:
    default:
      out += "ssd:auto";
      break;
  }
  return out;
}

size_t ConfigurableRoutePutStrategy::NextRoundRobin(size_t n) {
  if (n == 0) return 0;
  // relaxed: the cursor only has to be unique per call, not ordered against
  // any other memory.  Two concurrent puts may land in either order; what
  // matters is that they take different slots.
  return static_cast<size_t>(rr_cursor_.fetch_add(1, std::memory_order_relaxed) % n);
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
  if (!TierEligibleOnNode(*it, tier)) return std::nullopt;
  auto cap = it->tier_capacities.find(tier);
  if (cap == it->tier_capacities.end() || cap->second.available_bytes < block_size) {
    return std::nullopt;
  }
  return MakeRouted(*it, tier);
}

std::optional<RoutePutResult> ConfigurableRoutePutStrategy::TrySelectOnNode(
    const std::vector<ClientRecord>& candidates, const std::string& node_id, uint64_t block_size,
    const std::unordered_set<std::string>& exclude_nodes) const {
  for (TierType tier : TierOrder()) {
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
  const auto tier_eligible = [this](const ClientRecord& c, TierType t) {
    return TierEligibleOnNode(c, t);
  };
  for (TierType tier : TierOrder()) {
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
        CollectEligibleOnTier(candidates, tier, block_size, exclude_nodes, tier_eligible);
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
    } else if (algo_ == SelectAlgo::kRoundRobin) {
      // Rotate over a node_id-sorted view, not over `eligible` as returned.
      // `candidates` comes from the registry and its order is not guaranteed
      // stable across calls; rotating over an unstable order would revisit the
      // same node while skipping others — exactly the clumping being removed.
      // Sorting by node_id makes the rotation reproducible and node-count small
      // enough (one entry per peer) that the sort is free.
      std::vector<size_t> ordered = eligible;
      std::sort(ordered.begin(), ordered.end(),
                [&](size_t a, size_t b) { return candidates[a].node_id < candidates[b].node_id; });
      chosen = ordered[NextRoundRobin(ordered.size())];
    } else {
      for (size_t idx : eligible) {
        if (available(idx) > available(chosen)) chosen = idx;
      }
    }
    return MakeRouted(candidates[chosen], tier);
  }
  return std::nullopt;
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

  // exclude_nodes is constant for the whole batch, so format it once for logs.
  const std::string exclude_snapshot = FormatExcludeSet(exclude_nodes);
  const std::string algo_desc = Describe();

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

  // One unroutable log path for every dropped key (empty candidates, no fit, or
  // a deduction-contract violation), so each carries the projected capacity
  // snapshot — which reads "<no-alive-clients>" when candidates is empty.
  auto log_unroutable = [&](uint64_t bs) {
    MORI_UMBP_WARN(
        "[RoutePutStrategy] block_size={} no suitable target. algo=[{}] excludes=[{}] "
        "capacity_snapshot=[{}]",
        bs, algo_desc, exclude_snapshot, SummarizeClientTiers(candidates));
  };

  for (size_t i = 0; i < block_sizes.size(); ++i) {
    if (already_exists[i]) {
      // Master-side dedup hit: not a routing failure, so no WARN.
      MORI_UMBP_DEBUG("[RoutePutStrategy] block_size={} already-exists (dedup hit)",
                      block_sizes[i]);
      results.push_back(RoutePutResult{.outcome = RoutePutOutcome::kAlreadyExists});
      continue;
    }
    const uint64_t block_size = block_sizes[i];
    if (candidates.empty()) {
      log_unroutable(block_size);
      results.push_back(std::nullopt);
      continue;
    }

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

    // Apply projected capacity, then log against the FINAL returned value: a
    // routed pick whose deduction fails is dropped to nullopt and reported as a
    // failure, never as a route (capture available_bytes before the deduction
    // so the INFO reflects the snapshot the decision was made on).
    if (selected && selected->outcome == RoutePutOutcome::kRouted) {
      const uint64_t available =
          LookupAvailableBytes(candidates, selected->node_id, selected->tier);
      if (ApplyProjectedDeduction(candidates, *selected, block_size)) {
        MORI_UMBP_INFO(
            "[RoutePutStrategy] block_size={} tier={} selected node={} available_bytes={} "
            "algo=[{}] excludes=[{}]",
            block_size, TierTypeName(selected->tier), selected->node_id, available, algo_desc,
            exclude_snapshot);
      } else {
        selected = std::nullopt;  // selector broke its contract (best-effort drop)
        log_unroutable(block_size);
      }
    } else {
      log_unroutable(block_size);
    }
    results.push_back(std::move(selected));
  }

  LogBatchDistribution(requester_node_id, results, block_sizes, algo_desc);
  return results;
}

void ConfigurableRoutePutStrategy::LogBatchDistribution(
    const std::string& requester_node_id, const std::vector<std::optional<RoutePutResult>>& results,
    const std::vector<uint64_t>& block_sizes, const std::string& algo_desc) {
  const uint32_t every = PutDistLogEvery();
  if (every == 0) return;

  BatchDistMap batch;
  uint64_t routed = 0, dedup = 0, unroutable = 0, routed_bytes = 0;
  for (size_t i = 0; i < results.size(); ++i) {
    if (!results[i].has_value()) {
      ++unroutable;
      continue;
    }
    if (results[i]->outcome == RoutePutOutcome::kAlreadyExists) {
      ++dedup;
      continue;
    }
    const uint64_t bytes = i < block_sizes.size() ? block_sizes[i] : 0;
    auto& entry = batch[results[i]->node_id + "/" + TierTypeName(results[i]->tier)];
    entry.first += 1;
    entry.second += bytes;
    ++routed;
    routed_bytes += bytes;
  }

  // Fold into the cumulative view and decide whether this batch prints, both
  // under one lock so a sampled line still reflects every preceding batch.
  std::string cumulative_str;
  uint64_t cumulative_keys = 0;
  double cumulative_max_share = 0.0;
  uint64_t batch_index = 0;
  {
    std::lock_guard<std::mutex> lock(dist_mutex_);
    AccumulateBatchDist(batch, &dist_cumulative_);
    batch_index = ++dist_batches_;
    if (batch_index % every != 0) return;
    cumulative_keys = BatchDistTotalKeys(dist_cumulative_);
    cumulative_str = FormatBatchDist(dist_cumulative_, cumulative_keys);
    cumulative_max_share = BatchDistMaxShare(dist_cumulative_, cumulative_keys);
  }

  // Muted: this fires once per BatchRoutePut on the master, which sees every
  // client's traffic, so it is the noisiest of the three.  The cumulative
  // accounting above still runs; uncomment to see where the router is actually
  // placing keys (see doc/pure-ssd-mode.md, "Observability").
  // MORI_UMBP_INFO(
  //     "[RoutePutStrategy] batch_dist #{} requester={} keys={} routed={} dedup={} unroutable={} "
  //     "routed_bytes={} targets={} algo=[{}] batch=[{}] batch_max_share={:.1f}% "
  //     "cumulative_keys={} cumulative=[{}] cumulative_max_share={:.1f}%",
  //     batch_index, requester_node_id.empty() ? "<unknown>" : requester_node_id, results.size(),
  //     routed, dedup, unroutable, routed_bytes, batch.size(), algo_desc,
  //     FormatBatchDist(batch, routed), BatchDistMaxShare(batch, routed), cumulative_keys,
  //     cumulative_str, cumulative_max_share);
  (void)requester_node_id;
  (void)algo_desc;
  (void)routed;
  (void)dedup;
  (void)unroutable;
  (void)routed_bytes;
  (void)cumulative_str;
  (void)cumulative_max_share;
}

}  // namespace mori::umbp
