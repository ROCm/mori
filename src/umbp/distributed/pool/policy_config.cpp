// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License

#include "umbp/distributed/pool/policy_config.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>

#include "umbp/distributed/config.h"

namespace mori::umbp {
namespace {

using google::protobuf::ListValue;
using google::protobuf::Struct;
using google::protobuf::Value;

[[noreturn]] void Invalid(const std::string& message) {
  throw std::runtime_error(message);
}

const char* ValueTypeName(const Value& value) {
  switch (value.kind_case()) {
    case Value::kNullValue:
      return "null";
    case Value::kNumberValue:
      return "number";
    case Value::kStringValue:
      return "string";
    case Value::kBoolValue:
      return "boolean";
    case Value::kStructValue:
      return "object";
    case Value::kListValue:
      return "array";
    case Value::KIND_NOT_SET:
      return "unset";
  }
  return "unknown";
}

const Value& RequiredField(const Struct& object, const std::string& field,
                           const std::string& context) {
  const auto it = object.fields().find(field);
  if (it == object.fields().end()) Invalid(context + ": missing required field '" + field + "'");
  return it->second;
}

void RejectUnknownFields(const Struct& object,
                         std::initializer_list<std::string_view> allowed,
                         const std::string& context) {
  for (const auto& [name, value] : object.fields()) {
    (void)value;
    const bool known =
        std::any_of(allowed.begin(), allowed.end(), [&](std::string_view candidate) {
          return candidate == name;
        });
    if (!known) Invalid(context + ": unknown field '" + name + "'");
  }
}

const Struct& AsObject(const Value& value, const std::string& context) {
  if (value.kind_case() != Value::kStructValue) {
    Invalid(context + ": expected object, got " + ValueTypeName(value));
  }
  return value.struct_value();
}

const ListValue& AsArray(const Value& value, const std::string& context) {
  if (value.kind_case() != Value::kListValue) {
    Invalid(context + ": expected array, got " + ValueTypeName(value));
  }
  return value.list_value();
}

std::string AsString(const Value& value, const std::string& context) {
  if (value.kind_case() != Value::kStringValue) {
    Invalid(context + ": expected string, got " + ValueTypeName(value));
  }
  return value.string_value();
}

double AsNumber(const Value& value, const std::string& context) {
  if (value.kind_case() != Value::kNumberValue || !std::isfinite(value.number_value())) {
    Invalid(context + ": expected finite number");
  }
  return value.number_value();
}

bool AsBool(const Value& value, const std::string& context) {
  if (value.kind_case() != Value::kBoolValue) {
    Invalid(context + ": expected boolean, got " + ValueTypeName(value));
  }
  return value.bool_value();
}

uint32_t AsPositiveUint32(const Value& value, const std::string& context) {
  const double number = AsNumber(value, context);
  if (number <= 0.0 || std::trunc(number) != number ||
      number > static_cast<double>(std::numeric_limits<uint32_t>::max())) {
    Invalid(context + ": expected positive integer no greater than uint32 max");
  }
  return static_cast<uint32_t>(number);
}

int AsInt(const Value& value, int minimum, const std::string& context) {
  const double number = AsNumber(value, context);
  if (std::trunc(number) != number || number < static_cast<double>(minimum) ||
      number > static_cast<double>(std::numeric_limits<int>::max())) {
    Invalid(context + ": expected integer >= " + std::to_string(minimum));
  }
  return static_cast<int>(number);
}

uint64_t ParseCapacity(std::string_view text, const std::string& context) {
  if (text.empty()) Invalid(context + ": capacity must not be empty");

  size_t split = 0;
  uint64_t amount = 0;
  while (split < text.size() && text[split] >= '0' && text[split] <= '9') {
    const uint64_t digit = static_cast<uint64_t>(text[split] - '0');
    if (amount > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
      Invalid(context + ": capacity overflows uint64");
    }
    amount = amount * 10 + digit;
    ++split;
  }
  if (split == 0 || amount == 0) {
    Invalid(context + ": capacity must be a positive integer followed by B, KiB, MiB, GiB, or TiB");
  }

  const std::string_view unit = text.substr(split);
  uint64_t multiplier = 0;
  if (unit == "B") {
    multiplier = 1;
  } else if (unit == "KiB") {
    multiplier = 1ULL << 10;
  } else if (unit == "MiB") {
    multiplier = 1ULL << 20;
  } else if (unit == "GiB") {
    multiplier = 1ULL << 30;
  } else if (unit == "TiB") {
    multiplier = 1ULL << 40;
  } else {
    Invalid(context + ": capacity unit must be B, KiB, MiB, GiB, or TiB");
  }
  if (amount > std::numeric_limits<uint64_t>::max() / multiplier) {
    Invalid(context + ": capacity overflows uint64");
  }
  return amount * multiplier;
}

PolicyBackendSpec ParseBackend(const std::string& name, const Value& value) {
  const std::string context = "backend '" + name + "'";
  if (name.empty()) Invalid("backend names must not be empty");
  const Struct& object = AsObject(value, context);

  const std::string type = AsString(RequiredField(object, "type", context), context + ".type");
  PolicyBackendSpec backend;
  backend.name = name;
  backend.capacity_bytes =
      ParseCapacity(AsString(RequiredField(object, "capacity", context), context + ".capacity"),
                    context + ".capacity");

  if (type == "hbm") {
    RejectUnknownFields(object, {"type", "capacity", "devices"}, context);
    backend.tier = TierType::HBM;
    const ListValue& devices =
        AsArray(RequiredField(object, "devices", context), context + ".devices");
    if (devices.values().empty()) Invalid(context + ".devices: must not be empty");
    std::unordered_set<int> seen;
    for (int i = 0; i < devices.values_size(); ++i) {
      const int device =
          AsInt(devices.values(i), 0, context + ".devices[" + std::to_string(i) + "]");
      if (!seen.insert(device).second) {
        Invalid(context + ".devices: duplicate device " + std::to_string(device));
      }
      backend.devices.push_back(device);
    }
  } else if (type == "dram") {
    RejectUnknownFields(object, {"type", "capacity", "numa_node"}, context);
    backend.tier = TierType::DRAM;
    const auto numa = object.fields().find("numa_node");
    if (numa != object.fields().end()) {
      backend.numa_node = AsInt(numa->second, -1, context + ".numa_node");
    }
  } else if (type == "ssd") {
    RejectUnknownFields(object, {"type", "capacity", "path"}, context);
    backend.tier = TierType::SSD;
    backend.path = AsString(RequiredField(object, "path", context), context + ".path");
    if (backend.path.empty()) Invalid(context + ".path: must not be empty");
  } else {
    Invalid(context + ".type: expected 'dram', 'hbm', or 'ssd'");
  }
  return backend;
}

LogicalTierConfig ParseLogicalTier(const Value& value, size_t index) {
  const std::string context = "tiers[" + std::to_string(index) + "]";
  const Struct& object = AsObject(value, context);
  RejectUnknownFields(object,
                      {"name", "backends", "placement_policy", "offload_to",
                       "offload_trigger", "high_watermark", "low_watermark",
                       "candidate_policy", "promote_on_read", "promotion_mode"},
                      context);

  LogicalTierConfig tier;
  const auto name = object.fields().find("name");
  tier.name = name == object.fields().end()
                  ? "tier_" + std::to_string(index)
                  : AsString(name->second, context + ".name");
  if (tier.name.empty()) Invalid(context + ".name: must not be empty");
  if (tier.name.find('|') != std::string::npos) {
    Invalid(context + ".name: must not contain '|'");
  }

  const auto placement = object.fields().find("placement_policy");
  if (placement != object.fields().end() &&
      AsString(placement->second, context + ".placement_policy") != "weighted") {
    Invalid(context + ".placement_policy: expected 'weighted'");
  }

  const Struct& members =
      AsObject(RequiredField(object, "backends", context), context + ".backends");
  if (members.fields().empty()) Invalid(context + ".backends: must not be empty");
  for (const auto& [backend_name, weight] : members.fields()) {
    if (backend_name.empty()) Invalid(context + ".backends: backend name must not be empty");
    tier.members.push_back(
        {backend_name, AsPositiveUint32(weight, context + ".backends." + backend_name)});
  }
  std::sort(tier.members.begin(), tier.members.end(),
            [](const auto& left, const auto& right) {
              return left.backend_name < right.backend_name;
            });

  const auto targets_field = object.fields().find("offload_to");
  if (targets_field != object.fields().end()) {
    const ListValue& targets = AsArray(targets_field->second, context + ".offload_to");
    for (int i = 0; i < targets.values_size(); ++i) {
      std::string target =
          AsString(targets.values(i), context + ".offload_to[" + std::to_string(i) + "]");
      if (target.empty()) Invalid(context + ".offload_to: backend name must not be empty");
      tier.offload_to.push_back(std::move(target));
    }
  }

  const auto trigger_field = object.fields().find("offload_trigger");
  if (trigger_field != object.fields().end()) {
    const std::string trigger = AsString(trigger_field->second, context + ".offload_trigger");
    if (trigger == "on_evict") {
      tier.trigger = PoolOffloadTrigger::kOnEvict;
    } else if (trigger == "watermark") {
      tier.trigger = PoolOffloadTrigger::kWatermark;
    } else {
      Invalid(context + ".offload_trigger: expected 'on_evict' or 'watermark'");
    }
  } else if (!tier.offload_to.empty()) {
    Invalid(context + ": offload_trigger is required when offload_to is not empty");
  }

  const auto high = object.fields().find("high_watermark");
  if (high != object.fields().end()) {
    tier.high_watermark = AsNumber(high->second, context + ".high_watermark");
  }
  const auto low = object.fields().find("low_watermark");
  if (low != object.fields().end()) {
    tier.low_watermark = AsNumber(low->second, context + ".low_watermark");
  }
  if (!(tier.low_watermark > 0.0 && tier.low_watermark < tier.high_watermark &&
        tier.high_watermark <= 1.0)) {
    Invalid(context + ": watermarks must satisfy 0 < low_watermark < high_watermark <= 1");
  }

  const auto candidate = object.fields().find("candidate_policy");
  if (candidate != object.fields().end()) {
    const std::string policy = AsString(candidate->second, context + ".candidate_policy");
    if (policy == "lru") {
      tier.candidate_policy = TierCandidatePolicy::kLru;
    } else if (policy == "key_order") {
      tier.candidate_policy = TierCandidatePolicy::kKeyOrder;
    } else {
      Invalid(context + ".candidate_policy: expected 'lru' or 'key_order'");
    }
  }

  const auto promote = object.fields().find("promote_on_read");
  if (promote != object.fields().end()) {
    tier.promote_on_read = AsBool(promote->second, context + ".promote_on_read");
  }
  const auto promotion_mode = object.fields().find("promotion_mode");
  if (promotion_mode != object.fields().end()) {
    const std::string mode =
        AsString(promotion_mode->second, context + ".promotion_mode");
    if (mode == "copy") {
      tier.promotion_mode = PoolTransitionMode::kCopy;
    } else if (mode == "move") {
      tier.promotion_mode = PoolTransitionMode::kMove;
    } else {
      Invalid(context + ".promotion_mode: expected 'copy' or 'move'");
    }
  }
  return tier;
}

BackendPolicyConfig ParsePolicy(const Struct& root) {
  RejectUnknownFields(root, {"schema_version", "entry_tier", "backends", "tiers"}, "policy");

  BackendPolicyConfig policy;
  const auto schema = root.fields().find("schema_version");
  if (schema != root.fields().end()) {
    policy.schema_version =
        static_cast<uint32_t>(AsInt(schema->second, 1, "policy.schema_version"));
  }
  if (policy.schema_version != 1) {
    Invalid("policy.schema_version: only version 1 is supported");
  }
  const auto entry = root.fields().find("entry_tier");
  if (entry != root.fields().end()) {
    policy.entry_tier = AsString(entry->second, "policy.entry_tier");
    if (policy.entry_tier.empty()) Invalid("policy.entry_tier: must not be empty");
  }
  const Struct& backends =
      AsObject(RequiredField(root, "backends", "policy"), "policy.backends");
  if (backends.fields().empty()) Invalid("policy.backends: must not be empty");
  policy.backends.reserve(backends.fields_size());
  std::unordered_set<std::string> defined;
  for (const auto& [name, value] : backends.fields()) {
    policy.backends.push_back(ParseBackend(name, value));
    defined.insert(name);
  }

  const ListValue& tiers = AsArray(RequiredField(root, "tiers", "policy"), "policy.tiers");
  if (tiers.values().empty()) Invalid("policy.tiers: must not be empty");
  policy.logical_tiers.reserve(tiers.values_size());
  for (int i = 0; i < tiers.values_size(); ++i) {
    policy.logical_tiers.push_back(ParseLogicalTier(tiers.values(i), static_cast<size_t>(i)));
  }

  std::unordered_map<std::string, size_t> membership;
  std::unordered_map<std::string, size_t> tier_names;
  for (size_t i = 0; i < policy.logical_tiers.size(); ++i) {
    if (!tier_names.emplace(policy.logical_tiers[i].name, i).second) {
      Invalid("duplicate logical tier name '" + policy.logical_tiers[i].name + "'");
    }
    for (const auto& member : policy.logical_tiers[i].members) {
      if (defined.count(member.backend_name) == 0) {
        Invalid("tiers[" + std::to_string(i) + "]: undefined backend '" + member.backend_name +
                "'");
      }
      if (!membership.emplace(member.backend_name, i).second) {
        Invalid("backend '" + member.backend_name + "' is a member of more than one tier");
      }
    }
  }
  for (const auto& backend : policy.backends) {
    if (membership.count(backend.name) == 0) {
      Invalid("backend '" + backend.name + "' is not a member of any tier");
    }
  }
  const size_t entry_index = policy.entry_tier.empty()
                                 ? 0
                                 : tier_names.count(policy.entry_tier)
                                       ? tier_names.at(policy.entry_tier)
                                       : policy.logical_tiers.size();
  if (entry_index == policy.logical_tiers.size()) {
    Invalid("policy.entry_tier: undefined logical tier '" + policy.entry_tier + "'");
  }
  policy.logical_tiers[entry_index].entry = true;

  for (size_t i = 0; i < policy.logical_tiers.size(); ++i) {
    for (const auto& target : policy.logical_tiers[i].offload_to) {
      const auto member = membership.find(target);
      const auto named_tier = tier_names.find(target);
      if (member == membership.end() && named_tier == tier_names.end()) {
        Invalid("tiers[" + std::to_string(i) + "].offload_to: undefined backend or tier '" +
                target + "'");
      }
      const size_t target_index =
          member != membership.end() ? member->second : named_tier->second;
      if (target_index <= i) {
        Invalid("tiers[" + std::to_string(i) + "].offload_to: target '" + target +
                "' must be a strictly later tier");
      }
    }
  }
  return policy;
}

BackendPolicyLoadResult ErrorResult(std::string error) {
  BackendPolicyLoadResult result;
  result.error = std::move(error);
  return result;
}

bool SetApplyError(std::string* error, std::string message) {
  if (error != nullptr) *error = std::move(message);
  return false;
}

}  // namespace

BackendPolicyLoadResult LoadBackendPolicyJson(std::string_view json) {
  try {
    Struct root;
    google::protobuf::util::JsonParseOptions options;
    options.ignore_unknown_fields = false;
    const auto status =
        google::protobuf::util::JsonStringToMessage(std::string(json), &root, options);
    if (!status.ok()) return ErrorResult("invalid backend policy JSON: " + status.ToString());

    BackendPolicyLoadResult result;
    result.config = ParsePolicy(root);
    return result;
  } catch (const std::exception& exception) {
    return ErrorResult(exception.what());
  } catch (...) {
    return ErrorResult("unknown error while loading backend policy JSON");
  }
}

BackendPolicyLoadResult LoadBackendPolicyFile(std::string_view path) {
  try {
    const std::string file_path(path);
    std::ifstream input(file_path, std::ios::in | std::ios::binary);
    if (!input) return ErrorResult("failed to open backend policy file '" + file_path + "'");
    std::ostringstream contents;
    contents << input.rdbuf();
    if (input.bad()) return ErrorResult("failed to read backend policy file '" + file_path + "'");
    return LoadBackendPolicyJson(contents.str());
  } catch (const std::exception& exception) {
    return ErrorResult(exception.what());
  } catch (...) {
    return ErrorResult("unknown error while reading backend policy file");
  }
}

bool ApplyBackendPolicy(const BackendPolicyConfig& policy, PoolClientConfig* config,
                        std::string* error, std::string storage_path_suffix) {
  if (config == nullptr) return SetApplyError(error, "PoolClientConfig output must not be null");

  try {
    if (policy.schema_version != 1) {
      Invalid("only backend policy schema version 1 is supported");
    }
    if (policy.backends.empty()) Invalid("backend policy must define at least one backend");
    if (policy.logical_tiers.empty()) {
      Invalid("backend policy must define at least one logical tier");
    }
    std::vector<BackendInstanceConfig> concrete_backends;
    std::unordered_map<std::string, std::vector<std::string>> expansions;
    std::unordered_set<std::string> aliases;

    for (const auto& backend : policy.backends) {
      if (backend.name.empty()) Invalid("backend name must not be empty");
      if (!aliases.insert(backend.name).second) Invalid("duplicate backend '" + backend.name + "'");
      if (backend.capacity_bytes == 0) {
        Invalid("backend '" + backend.name + "': capacity must be positive");
      }

      std::vector<std::string> concrete_names;
      if (backend.tier == TierType::HBM) {
        if (backend.devices.empty()) Invalid("backend '" + backend.name + "': devices is empty");
        if (backend.capacity_bytes < backend.devices.size()) {
          Invalid("backend '" + backend.name + "': per-device capacity would be zero");
        }
        std::unordered_set<int> devices;
        const uint64_t base = backend.capacity_bytes / backend.devices.size();
        const uint64_t remainder = backend.capacity_bytes % backend.devices.size();
        for (size_t i = 0; i < backend.devices.size(); ++i) {
          const int device = backend.devices[i];
          if (device < 0 || !devices.insert(device).second) {
            Invalid("backend '" + backend.name + "': devices must be unique and nonnegative");
          }
          BackendInstanceConfig instance;
          instance.name = backend.devices.size() == 1
                              ? backend.name
                              : backend.name + "@" + std::to_string(device);
          instance.tier = TierType::HBM;
          instance.hbm.device = device;
          instance.hbm.buffer_sizes = {base + (i < remainder ? 1 : 0)};
          concrete_names.push_back(instance.name);
          concrete_backends.push_back(std::move(instance));
        }
      } else {
        BackendInstanceConfig instance;
        instance.name = backend.name;
        instance.tier = backend.tier;
        if (backend.tier == TierType::DRAM) {
          if (backend.numa_node < -1) {
            Invalid("backend '" + backend.name + "': numa_node must be >= -1");
          }
          instance.dram.buffer_sizes = {backend.capacity_bytes};
          instance.dram.numa_node = backend.numa_node;
        } else if (backend.tier == TierType::SSD) {
          if (backend.path.empty()) Invalid("backend '" + backend.name + "': path is empty");
          if (backend.capacity_bytes > std::numeric_limits<size_t>::max()) {
            Invalid("backend '" + backend.name + "': capacity does not fit size_t");
          }
          instance.ssd.enabled = true;
          instance.ssd.ssd.enabled = true;
          instance.ssd.ssd.capacity_bytes = static_cast<size_t>(backend.capacity_bytes);
          instance.ssd.ssd.storage_dir = backend.path + storage_path_suffix;
          instance.ssd.ssd.ssd_backend = "file";
        } else {
          Invalid("backend '" + backend.name + "': unsupported tier");
        }
        concrete_names.push_back(instance.name);
        concrete_backends.push_back(std::move(instance));
      }
      expansions.emplace(backend.name, std::move(concrete_names));
    }

    std::unordered_set<std::string> concrete_names;
    for (const auto& backend : concrete_backends) {
      if (!concrete_names.insert(backend.name).second) {
        Invalid("backend expansion produces duplicate concrete name '" + backend.name + "'");
      }
    }

    std::unordered_map<std::string, size_t> membership;
    std::unordered_map<std::string, size_t> tier_names;
    size_t entry_count = 0;
    for (size_t tier_index = 0; tier_index < policy.logical_tiers.size(); ++tier_index) {
      const auto& tier = policy.logical_tiers[tier_index];
      const std::string tier_name =
          tier.name.empty() ? "tier_" + std::to_string(tier_index) : tier.name;
      if (!tier_names.emplace(tier_name, tier_index).second) {
        Invalid("duplicate logical tier name '" + tier_name + "'");
      }
      entry_count += tier.entry ? 1 : 0;
      if (!(tier.low_watermark > 0.0 && tier.low_watermark < tier.high_watermark &&
            tier.high_watermark <= 1.0)) {
        Invalid("logical tier " + std::to_string(tier_index) +
                ": watermarks must satisfy 0 < low < high <= 1");
      }
      if (tier.trigger != PoolOffloadTrigger::kOnEvict &&
          tier.trigger != PoolOffloadTrigger::kWatermark) {
        Invalid("logical tier " + std::to_string(tier_index) + ": unsupported offload trigger");
      }
      for (const auto& member : tier.members) {
        if (expansions.count(member.backend_name) == 0) {
          Invalid("logical tier " + std::to_string(tier_index) + ": undefined backend '" +
                  member.backend_name + "'");
        }
        if (!membership.emplace(member.backend_name, tier_index).second) {
          Invalid("backend '" + member.backend_name + "' is a member of more than one tier");
        }
      }
    }
    for (const auto& backend : policy.backends) {
      if (membership.count(backend.name) == 0) {
        Invalid("backend '" + backend.name + "' is not a member of any logical tier");
      }
    }
    if (entry_count > 1) Invalid("exactly one logical tier may be the entry tier");
    if (!policy.entry_tier.empty() && tier_names.count(policy.entry_tier) == 0) {
      Invalid("undefined entry tier '" + policy.entry_tier + "'");
    }
    for (size_t tier_index = 0; tier_index < policy.logical_tiers.size(); ++tier_index) {
      for (const auto& target : policy.logical_tiers[tier_index].offload_to) {
        const auto target_tier = membership.find(target);
        const auto target_name = tier_names.find(target);
        if (target_tier == membership.end() && target_name == tier_names.end()) {
          Invalid("logical tier " + std::to_string(tier_index) +
                  ": undefined offload target '" + target + "'");
        }
        const size_t target_index =
            target_tier != membership.end() ? target_tier->second : target_name->second;
        if (target_index <= tier_index) {
          Invalid("logical tier " + std::to_string(tier_index) + ": offload target '" + target +
                  "' must belong to a strictly later tier");
        }
      }
    }

    std::vector<LogicalTierConfig> concrete_tiers;
    concrete_tiers.reserve(policy.logical_tiers.size());
    for (size_t tier_index = 0; tier_index < policy.logical_tiers.size(); ++tier_index) {
      const auto& source = policy.logical_tiers[tier_index];
      if (source.members.empty()) {
        Invalid("logical tier " + std::to_string(tier_index) + ": members is empty");
      }

      uint64_t expansion_lcm = 1;
      for (const auto& member : source.members) {
        if (member.weight == 0) {
          Invalid("logical tier " + std::to_string(tier_index) + ": weight must be positive");
        }
        const auto expansion = expansions.find(member.backend_name);
        if (expansion == expansions.end()) {
          Invalid("logical tier " + std::to_string(tier_index) + ": undefined backend '" +
                  member.backend_name + "'");
        }
        const uint64_t count = expansion->second.size();
        const uint64_t divisor = std::gcd(expansion_lcm, count);
        if (expansion_lcm > std::numeric_limits<uint64_t>::max() / (count / divisor)) {
          Invalid("logical tier " + std::to_string(tier_index) + ": expansion LCM overflows");
        }
        expansion_lcm *= count / divisor;
      }

      LogicalTierConfig lowered;
      lowered.trigger = source.trigger;
      lowered.high_watermark = source.high_watermark;
      lowered.low_watermark = source.low_watermark;
      lowered.name =
          source.name.empty() ? "tier_" + std::to_string(tier_index) : source.name;
      lowered.entry = source.entry;
      if (entry_count == 0) {
        lowered.entry =
            policy.entry_tier.empty() ? tier_index == 0 : lowered.name == policy.entry_tier;
      }
      lowered.candidate_policy = source.candidate_policy;
      lowered.promote_on_read = source.promote_on_read;
      lowered.promotion_mode = source.promotion_mode;
      for (const auto& member : source.members) {
        const auto& names = expansions.at(member.backend_name);
        const uint64_t factor = expansion_lcm / names.size();
        if (factor > std::numeric_limits<uint32_t>::max() / member.weight) {
          Invalid("logical tier " + std::to_string(tier_index) +
                  ": expanded backend weight overflows uint32");
        }
        const uint32_t concrete_weight = static_cast<uint32_t>(factor * member.weight);
        for (const auto& name : names) {
          lowered.members.push_back({name, concrete_weight});
        }
      }
      for (const auto& target : source.offload_to) {
        const auto expansion = expansions.find(target);
        if (expansion != expansions.end()) {
          lowered.offload_to.insert(lowered.offload_to.end(), expansion->second.begin(),
                                    expansion->second.end());
          continue;
        }
        const size_t target_tier = tier_names.at(target);
        for (const auto& member : policy.logical_tiers[target_tier].members) {
          const auto& concrete = expansions.at(member.backend_name);
          lowered.offload_to.insert(lowered.offload_to.end(), concrete.begin(), concrete.end());
        }
      }
      concrete_tiers.push_back(std::move(lowered));
    }

    if (concrete_backends.size() > kMaxBackendsPerPeer) {
      Invalid("backend policy expands beyond the per-peer backend limit of " +
              std::to_string(kMaxBackendsPerPeer));
    }
    const uint64_t page_size =
        config->dram_page_size == 0 ? 2ULL * 1024 * 1024 : config->dram_page_size;
    for (const auto& backend : concrete_backends) {
      uint64_t capacity = 0;
      if (backend.tier == TierType::DRAM && !backend.dram.buffer_sizes.empty()) {
        capacity = backend.dram.buffer_sizes.front();
      } else if (backend.tier == TierType::HBM && !backend.hbm.buffer_sizes.empty()) {
        capacity = backend.hbm.buffer_sizes.front();
      } else if (backend.tier == TierType::SSD) {
        capacity = backend.ssd.ssd.capacity_bytes;
      }
      if (capacity < page_size) {
        Invalid("backend '" + backend.name + "' capacity is smaller than page size");
      }
    }

    config->backends.swap(concrete_backends);
    config->logical_tiers.swap(concrete_tiers);
    config->placement_policy = PoolPlacementPolicy::TIERED;
    if (error != nullptr) error->clear();
    return true;
  } catch (const std::exception& exception) {
    return SetApplyError(error, exception.what());
  } catch (...) {
    return SetApplyError(error, "unknown error while applying backend policy");
  }
}

}  // namespace mori::umbp
