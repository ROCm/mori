// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

enum class PoolOffloadTrigger {
  kOnEvict,
  kWatermark,
};

// Mirrors PoolOffloadTrigger so a tier states when to promote the same way it
// states when to offload, and a new rule is a new enumerator rather than
// another boolean beside the last one.
enum class PoolPromoteTrigger {
  kNever,
  kOnRead,
  kOnHits,
};

enum class PoolTransitionMode {
  kMove,
  kCopy,
};

struct PolicyBackendSpec {
  std::string name;
  TierType tier = TierType::UNKNOWN;
  uint64_t capacity_bytes = 0;
  std::vector<int> devices;
  int numa_node = -1;
  std::string path;
};

struct LogicalTierBackendConfig {
  std::string backend_name;
  uint32_t weight = 0;
};

struct LogicalTierConfig {
  std::vector<LogicalTierBackendConfig> members;
  std::vector<std::string> offload_to;
  PoolOffloadTrigger trigger = PoolOffloadTrigger::kOnEvict;
  double high_watermark = 0.9;
  double low_watermark = 0.7;
  std::string name;
  bool entry = false;
  PoolPromoteTrigger promote_trigger = PoolPromoteTrigger::kNever;
  // Reads on this tier before a key is promoted. Only kOnHits reads it, and a
  // threshold of 1 is kOnRead spelled differently, so the parser requires >= 2.
  uint32_t promote_hits = 0;
  PoolTransitionMode promotion_mode = PoolTransitionMode::kCopy;
};

struct BackendPolicyConfig {
  std::vector<PolicyBackendSpec> backends;
  std::vector<LogicalTierConfig> logical_tiers;
  uint32_t schema_version = 1;
  std::string entry_tier;
};

// Everything a caller can learn about a tier list by looking at the list
// alone: resolved names, which tier owns each backend name, the entry tier,
// and offload edges resolved to tier indices with duplicates dropped.
struct LogicalTierIndex {
  std::vector<std::string> names;
  std::unordered_map<std::string, size_t> by_name;
  std::unordered_map<std::string, size_t> by_backend;
  std::vector<std::vector<size_t>> offload_to;
  size_t entry_tier = 0;
  bool entry_flagged = false;
};

// The single statement of what makes a tier list well formed, shared by the
// JSON front end, ApplyBackendPolicy, and LogicalTierGraph::Compile so a rule
// is written once and every entry point holds configs to the same bar. Returns
// an empty string and fills the index when the list is well formed, otherwise
// the reason it is not and an index no caller may read.
std::string IndexLogicalTiers(const std::vector<LogicalTierConfig>& tiers,
                              LogicalTierIndex* index);

struct BackendPolicyLoadResult {
  std::optional<BackendPolicyConfig> config;
  std::string error;

  bool ok() const { return config.has_value(); }
};

BackendPolicyLoadResult LoadBackendPolicyJson(std::string_view json);
BackendPolicyLoadResult LoadBackendPolicyFile(std::string_view path);

struct PoolClientConfig;

bool ApplyBackendPolicy(const BackendPolicyConfig& policy, PoolClientConfig* config,
                        std::string* error, std::string storage_path_suffix = {});

}  // namespace mori::umbp
