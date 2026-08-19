// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License

#include "umbp/distributed/pool/tier_graph.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <unordered_set>
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
    std::unordered_map<std::string, TierIndex> tier_by_backend_name,
    std::unordered_map<uint32_t, TierIndex> tier_by_backend_id, TierIndex entry_tier)
    : backends_(&backends),
      nodes_(std::move(nodes)),
      upstream_(std::move(upstream)),
      tier_by_backend_name_(std::move(tier_by_backend_name)),
      tier_by_backend_id_(std::move(tier_by_backend_id)),
      entry_tier_(entry_tier) {}

LogicalTierGraph::CompileResult LogicalTierGraph::Compile(
    const std::vector<LogicalTierConfig>& tiers, const BackendRegistry& backends) {
  if (tiers.empty()) return CompileError("logical tier graph must not be empty");

  std::vector<Node> nodes;
  nodes.reserve(tiers.size());
  std::unordered_map<std::string, TierIndex> tier_names;
  std::unordered_map<std::string, TierIndex> tier_by_backend_name;
  std::unordered_map<uint32_t, TierIndex> tier_by_backend_id;
  TierIndex entry_tier = kEntryTierIndex;
  size_t entry_count = 0;

  for (TierIndex i = 0; i < tiers.size(); ++i) {
    const auto& tier = tiers[i];
    Node node;
    node.name = tier.name.empty() ? "tier_" + std::to_string(i) : tier.name;
    if (node.name.find('|') != std::string::npos) {
      return CompileError("logical tier name '" + node.name + "' must not contain '|'");
    }
    if (!tier_names.emplace(node.name, i).second) {
      return CompileError("duplicate logical tier name '" + node.name + "'");
    }
    if (tier.entry) {
      entry_tier = i;
      ++entry_count;
    }
    if (tier.members.empty()) {
      return CompileError("logical tier '" + node.name + "' must not be empty");
    }
    if (!std::isfinite(tier.low_watermark) || !std::isfinite(tier.high_watermark) ||
        !(tier.low_watermark > 0.0 && tier.low_watermark < tier.high_watermark &&
          tier.high_watermark <= 1.0)) {
      return CompileError("logical tier '" + node.name +
                          "' watermarks must satisfy 0 < low < high <= 1");
    }

    node.members = tier.members;
    node.trigger = tier.trigger;
    node.high_watermark = tier.high_watermark;
    node.low_watermark = tier.low_watermark;
    node.candidate_policy = tier.candidate_policy;
    node.promote_on_read = tier.promote_on_read;
    node.promotion_mode = tier.promotion_mode;
    for (const auto& member : node.members) {
      if (member.backend_name.empty()) {
        return CompileError("logical tier '" + node.name + "' has an empty backend name");
      }
      if (member.weight == 0) {
        return CompileError("backend '" + member.backend_name + "' has zero placement weight");
      }
      const auto* entry = backends.GetEntry(member.backend_name);
      if (entry == nullptr) {
        return CompileError("logical tier '" + node.name + "' references unknown backend '" +
                            member.backend_name + "'");
      }
      if (!tier_by_backend_name.emplace(member.backend_name, i).second) {
        return CompileError("backend '" + member.backend_name +
                            "' belongs to more than one logical tier");
      }
      tier_by_backend_id.emplace(entry->backend_id, i);
    }
    nodes.push_back(std::move(node));
  }
  if (entry_count > 1) return CompileError("logical tier graph has multiple entry tiers");

  std::vector<std::vector<TierIndex>> upstream(tiers.size());
  for (TierIndex i = 0; i < tiers.size(); ++i) {
    std::unordered_set<TierIndex> seen;
    for (const auto& target_name : tiers[i].offload_to) {
      const auto owner = tier_by_backend_name.find(target_name);
      const auto named_tier = tier_names.find(target_name);
      if (owner == tier_by_backend_name.end() && named_tier == tier_names.end()) {
        return CompileError("logical tier '" + nodes[i].name +
                            "' has unknown offload target '" + target_name + "'");
      }
      const TierIndex target =
          owner != tier_by_backend_name.end() ? owner->second : named_tier->second;
      if (target <= i) {
        return CompileError("logical tier '" + nodes[i].name + "' offload target '" +
                            target_name + "' must belong to a strictly later tier");
      }
      if (seen.insert(target).second) {
        nodes[i].offload_to.push_back(target);
        upstream[target].push_back(i);
      }
    }
  }

  auto* graph = new LogicalTierGraph(
      backends, std::move(nodes), std::move(upstream), std::move(tier_by_backend_name),
      std::move(tier_by_backend_id), entry_tier);
  return {std::shared_ptr<const LogicalTierGraph>(graph), {}};
}

std::optional<LogicalTierGraph::TierIndex> LogicalTierGraph::TierIndexForBackendName(
    std::string_view name) const {
  const auto it = tier_by_backend_name_.find(std::string(name));
  return it == tier_by_backend_name_.end() ? std::nullopt
                                           : std::optional<TierIndex>{it->second};
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

double LogicalTierGraph::Utilization(TierIndex tier) const {
  long double total = 0;
  long double used = 0;
  for (const auto& member : NodeAt(tier).members) {
    const auto* backend = backends_->Get(member.backend_name);
    if (backend == nullptr) continue;
    const TierCapacity capacity = backend->Capacity();
    total += capacity.total_bytes;
    used += capacity.total_bytes -
            std::min(capacity.total_bytes, capacity.available_bytes);
  }
  return total == 0 ? 0.0 : static_cast<double>(used / total);
}

bool LogicalTierGraph::AtOrAbove(TierIndex tier, double watermark) const {
  return Utilization(tier) >= watermark;
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
