// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License

#include "umbp/distributed/pool/tier_graph.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <utility>

namespace mori::umbp {
namespace {

LogicalTierGraph::CompileResult CompileError(std::string error) {
  return {nullptr, std::move(error)};
}

uint64_t SaturatingAdd(uint64_t lhs, uint64_t rhs) {
  return rhs > std::numeric_limits<uint64_t>::max() - lhs
             ? std::numeric_limits<uint64_t>::max()
             : lhs + rhs;
}

}  // namespace

uint64_t StablePlacementHash(std::string_view key) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char byte : key) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::vector<uint32_t> WeightedBackendOrder(
    const std::vector<LogicalTierBackendConfig>& members, const BackendRegistry& backends,
    std::string_view key) {
  struct Candidate {
    uint32_t backend_id;
    uint32_t weight;
  };

  std::vector<Candidate> candidates;
  candidates.reserve(members.size());
  uint64_t total_weight = 0;
  for (const auto& member : members) {
    const auto* entry = backends.GetEntry(member.backend_name);
    if (entry == nullptr || member.weight == 0) continue;
    candidates.push_back({entry->backend_id, member.weight});
    total_weight += member.weight;
  }
  if (candidates.empty()) return {};

  uint64_t bucket = StablePlacementHash(key) % total_weight;
  size_t primary = 0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (bucket < candidates[i].weight) {
      primary = i;
      break;
    }
    bucket -= candidates[i].weight;
  }

  std::vector<uint32_t> order;
  order.reserve(candidates.size());
  for (size_t offset = 0; offset < candidates.size(); ++offset) {
    order.push_back(candidates[(primary + offset) % candidates.size()].backend_id);
  }
  return order;
}

LogicalTierGraph::LogicalTierGraph(
    const BackendRegistry& backends, std::vector<Node> nodes,
    std::vector<std::vector<TierIndex>> upstream,
    std::unordered_map<uint32_t, TierIndex> tier_by_backend_id, TierIndex entry_tier)
    : backends_(&backends),
      nodes_(std::move(nodes)),
      upstream_(std::move(upstream)),
      tier_by_backend_id_(std::move(tier_by_backend_id)),
      entry_tier_(entry_tier) {}

LogicalTierGraph::CompileResult LogicalTierGraph::Compile(
    const std::vector<LogicalTierConfig>& tiers, const BackendRegistry& backends) {
  LogicalTierIndex index;
  if (std::string error = IndexLogicalTiers(tiers, &index); !error.empty()) {
    return CompileError(std::move(error));
  }

  // Everything the tier list can say about itself has been checked; what is
  // left is binding member names to the backends this peer actually registered.
  std::vector<Node> nodes;
  nodes.reserve(tiers.size());
  std::unordered_map<uint32_t, TierIndex> tier_by_backend_id;
  std::vector<std::vector<TierIndex>> upstream(tiers.size());

  for (TierIndex i = 0; i < tiers.size(); ++i) {
    Node node;
    node.name = index.names[i];
    node.members = tiers[i].members;
    node.offload_to = index.offload_to[i];
    node.trigger = tiers[i].trigger;
    node.high_watermark = tiers[i].high_watermark;
    node.low_watermark = tiers[i].low_watermark;
    node.promote_on_read = tiers[i].promote_on_read;
    node.promotion_mode = tiers[i].promotion_mode;
    for (const auto& member : node.members) {
      const auto* entry = backends.GetEntry(member.backend_name);
      if (entry == nullptr) {
        return CompileError("logical tier '" + node.name + "' references unknown backend '" +
                            member.backend_name + "'");
      }
      tier_by_backend_id.emplace(entry->backend_id, i);
    }
    for (TierIndex target : node.offload_to) upstream[target].push_back(i);
    nodes.push_back(std::move(node));
  }

  auto* graph = new LogicalTierGraph(backends, std::move(nodes), std::move(upstream),
                                     std::move(tier_by_backend_id), index.entry_tier);
  return {std::shared_ptr<const LogicalTierGraph>(graph), {}};
}

std::optional<LogicalTierGraph::TierIndex> LogicalTierGraph::TierIndexForBackendId(
    uint32_t backend_id) const {
  const auto it = tier_by_backend_id_.find(backend_id);
  return it == tier_by_backend_id_.end() ? std::nullopt
                                         : std::optional<TierIndex>{it->second};
}

std::string LogicalTierGraph::NameForBackend(uint32_t backend_id) const {
  auto tier = TierIndexForBackendId(backend_id);
  return tier.has_value() ? NodeAt(*tier).name : std::string{};
}

std::map<std::string, LogicalTierCapacity> LogicalTierGraph::CapacitySnapshot() const {
  std::map<std::string, LogicalTierCapacity> snapshot;
  for (TierIndex tier = 0; tier < nodes_.size(); ++tier) {
    LogicalTierCapacity logical;
    logical.put_eligible = tier == entry_tier_;
    for (const auto& member : NodeAt(tier).members) {
      const auto* backend = backends_->Get(member.backend_name);
      if (backend == nullptr) continue;
      const auto cap = backend->Capacity();
      if (logical.representative_tier == TierType::UNKNOWN) {
        logical.representative_tier = backend->Tier();
      }
      logical.capacity.total_bytes =
          SaturatingAdd(logical.capacity.total_bytes, cap.total_bytes);
      logical.capacity.available_bytes =
          SaturatingAdd(logical.capacity.available_bytes, cap.available_bytes);
      const uint64_t allocatable =
          cap.max_allocatable_bytes == 0 ? cap.available_bytes
                                         : std::min(cap.available_bytes,
                                                    cap.max_allocatable_bytes);
      logical.capacity.max_allocatable_bytes =
          std::max(logical.capacity.max_allocatable_bytes, allocatable);
    }
    snapshot.emplace(NodeAt(tier).name, logical);
  }
  TierCapacity entry_pool;
  for (TierIndex tier : ReachableTierOrder(entry_tier_, false, true)) {
    const auto& capacity = snapshot.at(NodeAt(tier).name).capacity;
    entry_pool.total_bytes =
        SaturatingAdd(entry_pool.total_bytes, capacity.total_bytes);
    entry_pool.available_bytes =
        SaturatingAdd(entry_pool.available_bytes, capacity.available_bytes);
    entry_pool.max_allocatable_bytes =
        std::max(entry_pool.max_allocatable_bytes, capacity.max_allocatable_bytes);
  }
  snapshot.at(NodeAt(entry_tier_).name).capacity = entry_pool;
  return snapshot;
}

namespace {

double UtilizationOf(const TierCapacity& capacity) {
  if (capacity.total_bytes == 0) return 0.0;
  const uint64_t used =
      capacity.total_bytes - std::min(capacity.total_bytes, capacity.available_bytes);
  return static_cast<double>(used) / static_cast<double>(capacity.total_bytes);
}

}  // namespace

double LogicalTierGraph::MemberUtilization(uint32_t backend_id) const {
  const auto* backend = backends_->Get(backend_id);
  return backend == nullptr ? 0.0 : UtilizationOf(backend->Capacity());
}

double LogicalTierGraph::PeakMemberUtilization(TierIndex tier) const {
  double peak = 0.0;
  for (const auto& member : NodeAt(tier).members) {
    const auto* backend = backends_->Get(member.backend_name);
    if (backend == nullptr) continue;
    peak = std::max(peak, UtilizationOf(backend->Capacity()));
  }
  return peak;
}

bool LogicalTierGraph::AtOrAbove(TierIndex tier, double watermark) const {
  return PeakMemberUtilization(tier) >= watermark;
}

bool LogicalTierGraph::MemberAtOrAbove(uint32_t backend_id, double watermark) const {
  return MemberUtilization(backend_id) >= watermark;
}

std::vector<uint32_t> LogicalTierGraph::WeightedMemberOrder(TierIndex tier,
                                                            std::string_view key) const {
  return WeightedBackendOrder(NodeAt(tier).members, *backends_, key);
}

std::vector<LogicalTierGraph::TierIndex> LogicalTierGraph::ReachableTierOrder(
    TierIndex source, bool upstream, bool include_source) const {
  if (source >= nodes_.size()) return {};
  std::vector<TierIndex> order;
  std::vector<bool> visited(nodes_.size(), false);
  std::function<void(TierIndex)> visit = [&](TierIndex tier) {
    if (visited[tier]) return;
    visited[tier] = true;
    order.push_back(tier);
    const auto& edges = upstream ? upstream_[tier] : nodes_[tier].offload_to;
    for (TierIndex target : edges) visit(target);
  };
  visit(source);
  if (!include_source) order.erase(order.begin());
  if (upstream) std::sort(order.begin(), order.end());
  return order;
}

std::vector<uint32_t> LogicalTierGraph::BackendOrder(const std::vector<TierIndex>& tiers,
                                                     std::string_view key) const {
  std::vector<uint32_t> order;
  for (TierIndex tier : tiers) {
    auto members = WeightedMemberOrder(tier, key);
    order.insert(order.end(), members.begin(), members.end());
  }
  return order;
}

std::vector<uint32_t> LogicalTierGraph::PutOrder(std::string_view key) const {
  return BackendOrder(ReachableTierOrder(entry_tier_, false, true), key);
}

std::vector<uint32_t> LogicalTierGraph::PutOrderFromTier(std::string_view name,
                                                         std::string_view key) const {
  for (TierIndex tier = 0; tier < nodes_.size(); ++tier) {
    if (NodeAt(tier).name == name) {
      return BackendOrder(ReachableTierOrder(tier, false, true), key);
    }
  }
  return {};
}

std::vector<uint32_t> LogicalTierGraph::TransitionTargetOrder(
    TierIndex source, std::string_view key) const {
  return BackendOrder(ReachableTierOrder(source, false, false), key);
}

std::vector<uint32_t> LogicalTierGraph::PromoteTargetOrder(
    TierIndex source, std::string_view key) const {
  return BackendOrder(ReachableTierOrder(source, true, false), key);
}

std::vector<uint32_t> LogicalTierGraph::ReadOrder() const {
  std::vector<uint32_t> order;
  order.reserve(tier_by_backend_id_.size());
  for (const auto& node : nodes_) {
    for (const auto& member : node.members) {
      const auto* entry = backends_->GetEntry(member.backend_name);
      if (entry != nullptr) order.push_back(entry->backend_id);
    }
  }
  return order;
}

}  // namespace mori::umbp
