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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License

#include "umbp/distributed/pool/policy_config.h"

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>

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

#include "umbp/distributed/config.h"

namespace mori::umbp {
namespace {

using google::protobuf::ListValue;
using google::protobuf::Struct;
using google::protobuf::Value;

[[noreturn]] void Invalid(const std::string& message) { throw std::runtime_error(message); }

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

const Value* OptionalField(const Struct& object, const std::string& field) {
  const auto it = object.fields().find(field);
  return it == object.fields().end() ? nullptr : &it->second;
}

void RejectUnknownFields(const Struct& object, std::initializer_list<std::string_view> allowed,
                         const std::string& context) {
  for (const auto& [name, value] : object.fields()) {
    (void)value;
    const bool known = std::any_of(allowed.begin(), allowed.end(),
                                   [&](std::string_view candidate) { return candidate == name; });
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

template <typename Enum>
Enum AsEnum(const Value& value, const std::string& context,
            std::initializer_list<std::pair<std::string_view, Enum>> allowed) {
  const std::string text = AsString(value, context);
  std::string expected;
  for (const auto& [name, mapped] : allowed) {
    if (text == name) return mapped;
    if (!expected.empty()) expected += " or ";
    expected += "'" + std::string(name) + "'";
  }
  Invalid(context + ": expected " + expected);
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

  static constexpr std::pair<std::string_view, uint64_t> kUnits[] = {
      {"B", 1}, {"KiB", 1ULL << 10}, {"MiB", 1ULL << 20}, {"GiB", 1ULL << 30}, {"TiB", 1ULL << 40}};
  const std::string_view unit = text.substr(split);
  uint64_t multiplier = 0;
  for (const auto& [name, scale] : kUnits) {
    if (unit == name) {
      multiplier = scale;
      break;
    }
  }
  if (multiplier == 0) {
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
    if (const Value* numa = OptionalField(object, "numa_node")) {
      backend.numa_node = AsInt(*numa, -1, context + ".numa_node");
    }
  } else if (type == "ssd") {
    RejectUnknownFields(object, {"type", "capacity", "path", "staging_slots"}, context);
    backend.tier = TierType::SSD;
    backend.path = AsString(RequiredField(object, "path", context), context + ".path");
    if (backend.path.empty()) Invalid(context + ".path: must not be empty");
    // Minimum 1, not 0: 0 is how a config assembled in code says "leave the
    // default", which is not a thing a written-out field can mean.
    if (const Value* slots = OptionalField(object, "staging_slots")) {
      backend.staging_slots = AsInt(*slots, 1, context + ".staging_slots");
    }
  } else {
    Invalid(context + ".type: expected 'dram', 'hbm', or 'ssd'");
  }
  return backend;
}

LogicalTierConfig ParseLogicalTier(const Value& value, size_t index) {
  const std::string context = "tiers[" + std::to_string(index) + "]";
  const Struct& object = AsObject(value, context);
  RejectUnknownFields(
      object,
      {"name", "backends", "placement_policy", "offload_to", "offload_trigger", "high_watermark",
       "low_watermark", "promote_trigger", "promote_hits", "promote_mode"},
      context);

  LogicalTierConfig tier;
  const Value* name = OptionalField(object, "name");
  tier.name =
      name == nullptr ? "tier_" + std::to_string(index) : AsString(*name, context + ".name");
  if (tier.name.empty()) Invalid(context + ".name: must not be empty");

  // Placement within a tier is weighted by construction; the field exists so a
  // policy can say so, not so it can ask for something else.
  if (const Value* placement = OptionalField(object, "placement_policy")) {
    if (AsString(*placement, context + ".placement_policy") != "weighted") {
      Invalid(context + ".placement_policy: expected 'weighted'");
    }
  }

  const Struct& members =
      AsObject(RequiredField(object, "backends", context), context + ".backends");
  if (members.fields().empty()) Invalid(context + ".backends: must not be empty");
  for (const auto& [backend_name, weight] : members.fields()) {
    if (backend_name.empty()) Invalid(context + ".backends: backend name must not be empty");
    tier.members.push_back(
        {backend_name, AsPositiveUint32(weight, context + ".backends." + backend_name)});
  }
  std::sort(tier.members.begin(), tier.members.end(), [](const auto& left, const auto& right) {
    return left.backend_name < right.backend_name;
  });

  if (const Value* targets_field = OptionalField(object, "offload_to")) {
    const ListValue& targets = AsArray(*targets_field, context + ".offload_to");
    for (int i = 0; i < targets.values_size(); ++i) {
      tier.offload_to.push_back(
          AsString(targets.values(i), context + ".offload_to[" + std::to_string(i) + "]"));
    }
  }

  if (const Value* trigger = OptionalField(object, "offload_trigger")) {
    tier.trigger = AsEnum<PoolOffloadTrigger>(*trigger, context + ".offload_trigger",
                                              {{"on_evict", PoolOffloadTrigger::kOnEvict},
                                               {"watermark", PoolOffloadTrigger::kWatermark}});
  } else if (!tier.offload_to.empty()) {
    Invalid(context + ": offload_trigger is required when offload_to is not empty");
  }

  if (const Value* high = OptionalField(object, "high_watermark")) {
    tier.high_watermark = AsNumber(*high, context + ".high_watermark");
  }
  if (const Value* low = OptionalField(object, "low_watermark")) {
    tier.low_watermark = AsNumber(*low, context + ".low_watermark");
  }
  if (const Value* promote = OptionalField(object, "promote_trigger")) {
    tier.promote_trigger = AsEnum<PoolPromoteTrigger>(*promote, context + ".promote_trigger",
                                                      {{"never", PoolPromoteTrigger::kNever},
                                                       {"on_read", PoolPromoteTrigger::kOnRead},
                                                       {"on_hits", PoolPromoteTrigger::kOnHits}});
  }
  // A hit threshold has no meaningful default, so accepting it where nothing
  // reads it would hide the misconfiguration instead of reporting it. The
  // watermark fields above are lenient because they do have defaults.
  if (const Value* hits = OptionalField(object, "promote_hits")) {
    if (tier.promote_trigger != PoolPromoteTrigger::kOnHits) {
      Invalid(context + ".promote_hits: only valid when promote_trigger is 'on_hits'");
    }
    tier.promote_hits = static_cast<uint32_t>(AsInt(*hits, 2, context + ".promote_hits"));
  } else if (tier.promote_trigger == PoolPromoteTrigger::kOnHits) {
    Invalid(context + ": promote_hits is required when promote_trigger is 'on_hits'");
  }
  if (const Value* mode = OptionalField(object, "promote_mode")) {
    tier.promote_mode = AsEnum<PoolTransitionMode>(
        *mode, context + ".promote_mode",
        {{"copy", PoolTransitionMode::kCopy}, {"move", PoolTransitionMode::kMove}});
  }
  return tier;
}

// Promotion moves a key upstream and the entry tier has no upstream, so a
// trigger declared there can never fire: the read path tests the same condition
// and skips it. Checked wherever the entry tier is finally known, because
// naming it in `entry_tier` can move it after the per-tier pass.
std::string RejectEntryTierPromotion(const std::vector<LogicalTierConfig>& tiers, size_t entry_tier,
                                     const std::string& name) {
  if (entry_tier >= tiers.size() ||
      tiers[entry_tier].promote_trigger == PoolPromoteTrigger::kNever) {
    return {};
  }
  return "logical tier '" + name + "': the entry tier has no upstream tier to promote into";
}

// The single definition of a well formed policy. The JSON front end and
// ApplyBackendPolicy both run it: a config assembled in code reaches the same
// bar as one that came from a file, and a rule only has to be stated once.
LogicalTierIndex ValidateBackendPolicy(const BackendPolicyConfig& policy) {
  if (policy.schema_version != 1) {
    Invalid("policy: only backend policy schema version 1 is supported");
  }
  if (policy.backends.empty()) Invalid("policy: must define at least one backend");
  if (policy.logical_tiers.empty()) {
    Invalid("policy: must define at least one logical tier");
  }

  std::unordered_set<std::string> defined;
  for (const auto& backend : policy.backends) {
    if (backend.name.empty()) Invalid("policy.backends: names must not be empty");
    const std::string context = "backend '" + backend.name + "'";
    if (!defined.insert(backend.name).second) Invalid("duplicate " + context);
    if (backend.capacity_bytes == 0) Invalid(context + ": capacity must be positive");
    if (backend.tier == TierType::HBM) {
      if (backend.devices.empty()) Invalid(context + ": devices must not be empty");
      if (backend.capacity_bytes < backend.devices.size()) {
        Invalid(context + ": per-device capacity would be zero");
      }
      std::unordered_set<int> devices;
      for (int device : backend.devices) {
        if (device < 0 || !devices.insert(device).second) {
          Invalid(context + ": devices must be unique and nonnegative");
        }
      }
    } else if (backend.tier == TierType::DRAM) {
      if (backend.numa_node < -1) Invalid(context + ": numa_node must be >= -1");
    } else if (backend.tier == TierType::SSD) {
      if (backend.path.empty()) Invalid(context + ": path must not be empty");
      if (backend.staging_slots < 0) Invalid(context + ": staging_slots must be >= 0");
    } else {
      Invalid(context + ": unsupported tier");
    }
  }

  LogicalTierIndex index;
  if (const std::string error = IndexLogicalTiers(policy.logical_tiers, &index); !error.empty()) {
    Invalid(error);
  }
  for (size_t i = 0; i < policy.logical_tiers.size(); ++i) {
    for (const auto& member : policy.logical_tiers[i].members) {
      if (defined.count(member.backend_name) == 0) {
        Invalid("logical tier '" + index.names[i] + "': undefined backend '" + member.backend_name +
                "'");
      }
    }
  }
  for (const auto& backend : policy.backends) {
    if (index.by_backend.count(backend.name) == 0) {
      Invalid("backend '" + backend.name + "' is not a member of any logical tier");
    }
  }

  // A policy may name its entry tier instead of flagging it; the two must agree
  // when both are present.
  if (!policy.entry_tier.empty()) {
    const auto named = index.by_name.find(policy.entry_tier);
    if (named == index.by_name.end()) {
      Invalid("policy.entry_tier: undefined logical tier '" + policy.entry_tier + "'");
    }
    if (index.entry_flagged && index.entry_tier != named->second) {
      Invalid("policy.entry_tier names '" + policy.entry_tier +
              "' but a different logical tier is flagged as the entry tier");
    }
    index.entry_tier = named->second;
    if (const std::string error = RejectEntryTierPromotion(policy.logical_tiers, index.entry_tier,
                                                           index.names[index.entry_tier]);
        !error.empty()) {
      Invalid(error);
    }
  }
  return index;
}

BackendPolicyConfig ParsePolicy(const Struct& root) {
  RejectUnknownFields(root, {"schema_version", "entry_tier", "backends", "tiers"}, "policy");

  BackendPolicyConfig policy;
  if (const Value* schema = OptionalField(root, "schema_version")) {
    policy.schema_version = static_cast<uint32_t>(AsInt(*schema, 1, "policy.schema_version"));
  }
  if (const Value* entry = OptionalField(root, "entry_tier")) {
    policy.entry_tier = AsString(*entry, "policy.entry_tier");
    if (policy.entry_tier.empty()) Invalid("policy.entry_tier: must not be empty");
  }
  const Struct& backends = AsObject(RequiredField(root, "backends", "policy"), "policy.backends");
  if (backends.fields().empty()) Invalid("policy.backends: must not be empty");
  policy.backends.reserve(backends.fields_size());
  for (const auto& [name, value] : backends.fields()) {
    policy.backends.push_back(ParseBackend(name, value));
  }

  const ListValue& tiers = AsArray(RequiredField(root, "tiers", "policy"), "policy.tiers");
  if (tiers.values().empty()) Invalid("policy.tiers: must not be empty");
  policy.logical_tiers.reserve(tiers.values_size());
  for (int i = 0; i < tiers.values_size(); ++i) {
    policy.logical_tiers.push_back(ParseLogicalTier(tiers.values(i), static_cast<size_t>(i)));
  }

  policy.logical_tiers[ValidateBackendPolicy(policy).entry_tier].entry = true;
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

std::string IndexLogicalTiers(const std::vector<LogicalTierConfig>& tiers,
                              LogicalTierIndex* index) {
  if (tiers.empty()) return "logical tier graph must not be empty";

  *index = {};
  index->names.reserve(tiers.size());
  index->offload_to.resize(tiers.size());

  for (size_t i = 0; i < tiers.size(); ++i) {
    const auto& tier = tiers[i];
    std::string name = tier.name.empty() ? "tier_" + std::to_string(i) : tier.name;
    const std::string context = "logical tier '" + name + "'";
    // Slot tokens join a tier name with a backend name, so '|' cannot appear in
    // either half.
    if (name.find('|') != std::string::npos) return context + ": name must not contain '|'";
    if (!index->by_name.emplace(name, i).second) {
      return "duplicate logical tier name '" + name + "'";
    }
    index->names.push_back(std::move(name));

    if (tier.entry) {
      if (index->entry_flagged) return "exactly one logical tier may be the entry tier";
      index->entry_flagged = true;
      index->entry_tier = i;
    }
    if (!std::isfinite(tier.low_watermark) || !std::isfinite(tier.high_watermark) ||
        !(tier.low_watermark > 0.0 && tier.low_watermark < tier.high_watermark &&
          tier.high_watermark <= 1.0)) {
      return context + ": watermarks must satisfy 0 < low_watermark < high_watermark <= 1";
    }
    if (tier.trigger != PoolOffloadTrigger::kOnEvict &&
        tier.trigger != PoolOffloadTrigger::kWatermark) {
      return context + ": unsupported offload trigger";
    }
    switch (tier.promote_trigger) {
      case PoolPromoteTrigger::kNever:
      case PoolPromoteTrigger::kOnRead:
        if (tier.promote_hits != 0) {
          return context + ": promote_hits is only meaningful for the 'on_hits' trigger";
        }
        break;
      case PoolPromoteTrigger::kOnHits:
        // 1 would be the on_read trigger under another name, and one policy
        // meaning two things is how the two drift apart.
        if (tier.promote_hits < 2) {
          return context + ": promote_hits must be at least 2";
        }
        break;
      default:
        return context + ": unsupported promote trigger";
    }
    if (tier.members.empty()) return context + ": must have at least one backend";
    for (const auto& member : tier.members) {
      if (member.backend_name.empty()) return context + ": backend names must not be empty";
      if (member.weight == 0) {
        return context + ": weight for '" + member.backend_name + "' must be positive";
      }
      if (!index->by_backend.emplace(member.backend_name, i).second) {
        return "backend '" + member.backend_name + "' is a member of more than one tier";
      }
    }
  }

  for (size_t i = 0; i < tiers.size(); ++i) {
    const std::string context = "logical tier '" + index->names[i] + "'";
    std::unordered_set<size_t> seen;
    for (const auto& target : tiers[i].offload_to) {
      if (target.empty()) return context + ": offload target must not be empty";
      const auto by_backend = index->by_backend.find(target);
      const auto by_name = index->by_name.find(target);
      if (by_backend == index->by_backend.end() && by_name == index->by_name.end()) {
        return context + ": undefined offload target '" + target + "'";
      }
      const size_t resolved =
          by_backend != index->by_backend.end() ? by_backend->second : by_name->second;
      // Offload only ever runs downstream; an edge back up would let a key
      // cycle between two tiers forever.
      if (resolved <= i) {
        return context + ": offload target '" + target + "' must belong to a strictly later tier";
      }
      if (seen.insert(resolved).second) index->offload_to[i].push_back(resolved);
    }
  }
  return RejectEntryTierPromotion(tiers, index->entry_tier, index->names[index->entry_tier]);
}

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
    const LogicalTierIndex index = ValidateBackendPolicy(policy);

    std::vector<BackendInstanceConfig> concrete_backends;
    std::unordered_map<std::string, std::vector<std::string>> expansions;

    for (const auto& backend : policy.backends) {
      std::vector<std::string> concrete_names;
      if (backend.tier == TierType::HBM) {
        const uint64_t base = backend.capacity_bytes / backend.devices.size();
        const uint64_t remainder = backend.capacity_bytes % backend.devices.size();
        for (size_t i = 0; i < backend.devices.size(); ++i) {
          const int device = backend.devices[i];
          BackendInstanceConfig instance;
          instance.name = backend.devices.size() == 1 ? backend.name
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
          instance.dram.buffer_sizes = {backend.capacity_bytes};
          instance.dram.numa_node = backend.numa_node;
        } else {
          if (backend.capacity_bytes > std::numeric_limits<size_t>::max()) {
            Invalid("backend '" + backend.name + "': capacity does not fit size_t");
          }
          instance.ssd.enabled = true;
          instance.ssd.ssd.enabled = true;
          instance.ssd.ssd.capacity_bytes = static_cast<size_t>(backend.capacity_bytes);
          instance.ssd.ssd.storage_dir = backend.path + storage_path_suffix;
          instance.ssd.ssd.ssd_backend = "file";
          if (backend.staging_slots > 0) {
            instance.ssd_staging_buffer_slots = backend.staging_slots;
          }
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

    std::vector<LogicalTierConfig> concrete_tiers;
    concrete_tiers.reserve(policy.logical_tiers.size());
    for (size_t tier_index = 0; tier_index < policy.logical_tiers.size(); ++tier_index) {
      const auto& source = policy.logical_tiers[tier_index];

      // Members are weighted per named backend, but placement happens on the
      // concrete instances a name expands to. Scaling every weight by the LCM
      // of the expansion sizes keeps the configured ratio exact in integers.
      uint64_t expansion_lcm = 1;
      for (const auto& member : source.members) {
        const uint64_t count = expansions.at(member.backend_name).size();
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
      lowered.name = index.names[tier_index];
      lowered.entry = tier_index == index.entry_tier;
      lowered.promote_trigger = source.promote_trigger;
      lowered.promote_hits = source.promote_hits;
      lowered.promote_mode = source.promote_mode;
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
        const size_t target_tier = index.by_name.at(target);
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
