// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/pool/policy_config.h"

namespace mori::umbp {

uint64_t StablePlacementHash(std::string_view key);

std::vector<uint32_t> WeightedBackendOrder(
    const std::vector<LogicalTierBackendConfig>& members, const BackendRegistry& backends,
    std::string_view key);

class LogicalTierGraph {
 public:
  using TierIndex = size_t;
  static constexpr TierIndex kEntryTierIndex = 0;

  struct Node {
    std::string name;
    std::vector<LogicalTierBackendConfig> members;
    std::vector<TierIndex> offload_to;
    PoolOffloadTrigger trigger = PoolOffloadTrigger::kOnEvict;
    double high_watermark = 0.9;
    double low_watermark = 0.7;
    PoolPromoteTrigger promote_trigger = PoolPromoteTrigger::kNever;
    uint32_t promote_hits = 0;
    PoolTransitionMode promotion_mode = PoolTransitionMode::kCopy;
  };

  struct CompileResult;

  static CompileResult Compile(const std::vector<LogicalTierConfig>& tiers,
                               const BackendRegistry& backends);

  size_t TierCount() const { return nodes_.size(); }
  TierIndex EntryTierIndex() const { return entry_tier_; }
  const Node& NodeAt(TierIndex tier) const { return nodes_.at(tier); }

  std::optional<TierIndex> TierIndexForBackendId(uint32_t backend_id) const;
  std::string NameForBackend(uint32_t backend_id) const;
  std::map<std::string, LogicalTierCapacity> CapacitySnapshot() const;

  double MemberUtilization(uint32_t backend_id) const;
  // Watermarks describe pressure, and pressure does not average: a tier that
  // mixes a 512 GiB DRAM member with a 1 TiB SSD member would report only ~33%
  // aggregate use with DRAM completely full, so nothing would ever offload and
  // the DRAM-weighted share of the traffic would silently spill to SSD. The
  // fullest member therefore speaks for the tier.
  double PeakMemberUtilization(TierIndex tier) const;
  bool AtOrAbove(TierIndex tier, double watermark) const;
  bool MemberAtOrAbove(uint32_t backend_id, double watermark) const;

  std::vector<uint32_t> WeightedMemberOrder(TierIndex tier, std::string_view key) const;
  std::vector<uint32_t> PutOrder(std::string_view key) const;
  std::vector<uint32_t> PutOrderFromTier(std::string_view name,
                                         std::string_view key) const;
  std::vector<uint32_t> TransitionTargetOrder(TierIndex source, std::string_view key) const;
  std::vector<uint32_t> PromoteTargetOrder(TierIndex source, std::string_view key) const;
  std::vector<uint32_t> ReadOrder() const;

 private:
  LogicalTierGraph(const BackendRegistry& backends, std::vector<Node> nodes,
                   std::vector<std::vector<TierIndex>> upstream,
                   std::unordered_map<uint32_t, TierIndex> tier_by_backend_id,
                   TierIndex entry_tier);

  std::vector<TierIndex> ReachableTierOrder(TierIndex source, bool upstream,
                                            bool include_source) const;
  std::vector<uint32_t> BackendOrder(const std::vector<TierIndex>& tiers,
                                     std::string_view key) const;

  const BackendRegistry* backends_;
  std::vector<Node> nodes_;
  std::vector<std::vector<TierIndex>> upstream_;
  std::unordered_map<uint32_t, TierIndex> tier_by_backend_id_;
  TierIndex entry_tier_ = kEntryTierIndex;
};

struct LogicalTierGraph::CompileResult {
  std::shared_ptr<const LogicalTierGraph> graph;
  std::string error;

  bool ok() const { return graph != nullptr; }
};

}  // namespace mori::umbp
