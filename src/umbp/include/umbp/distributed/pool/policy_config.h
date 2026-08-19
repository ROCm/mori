// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

enum class PoolOffloadTrigger {
  kOnEvict,
  kWatermark,
};

enum class PoolTransitionMode {
  kMove,
  kCopy,
};

enum class TierCandidatePolicy {
  kLru,
  kKeyOrder,
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
  TierCandidatePolicy candidate_policy = TierCandidatePolicy::kLru;
  bool promote_on_read = false;
  PoolTransitionMode promotion_mode = PoolTransitionMode::kCopy;
};

struct BackendPolicyConfig {
  std::vector<PolicyBackendSpec> backends;
  std::vector<LogicalTierConfig> logical_tiers;
  uint32_t schema_version = 1;
  std::string entry_tier;
};

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
