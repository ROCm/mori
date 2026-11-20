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
#include "umbp/distributed/pool/pool_rules.h"

#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <utility>

namespace mori::umbp {
namespace {

using google::protobuf::Struct;
using google::protobuf::Value;

int64_t SteadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

const char* PromoteName(PoolRulePromote promote) {
  switch (promote) {
    case PoolRulePromote::kNever:
      return "never";
    case PoolRulePromote::kOnRead:
      return "on_read";
    case PoolRulePromote::kTier:
      break;
  }
  return "tier";
}

bool Fail(std::string* error, std::string message) {
  if (error != nullptr) *error = std::move(message);
  return false;
}

bool ParseStruct(std::string_view json, Struct* out, std::string* error) {
  const auto status = google::protobuf::util::JsonStringToMessage(std::string(json), out);
  if (!status.ok()) return Fail(error, "not valid JSON: " + std::string(status.message()));
  return true;
}

// `rule` starts as the inherited default; only the fields present overwrite it.
bool ParseRule(const Value& value, const std::string& where, PoolRule* rule, std::string* error) {
  if (value.kind_case() != Value::kStructValue) return Fail(error, where + " must be an object");
  for (const auto& [name, field] : value.struct_value().fields()) {
    if (name == "admit" || name == "demote") {
      if (field.kind_case() != Value::kBoolValue) {
        return Fail(error, where + "." + name + " must be true or false");
      }
      (name == "admit" ? rule->admit : rule->demote) = field.bool_value();
    } else if (name == "weight") {
      if (field.kind_case() != Value::kNumberValue || !(field.number_value() > 0.0) ||
          !std::isfinite(field.number_value())) {
        return Fail(error, where + ".weight must be a number above 0");
      }
      rule->weight = field.number_value();
    } else if (name == "retain_s") {
      if (field.kind_case() != Value::kNumberValue || field.number_value() < 0.0 ||
          field.number_value() > 4294967295.0) {
        return Fail(error, where + ".retain_s must be a number of seconds, 0 or more");
      }
      rule->retain_s = static_cast<uint32_t>(field.number_value());
    } else if (name == "promote") {
      const std::string text =
          field.kind_case() == Value::kStringValue ? field.string_value() : std::string();
      if (text == "tier") {
        rule->promote = PoolRulePromote::kTier;
      } else if (text == "never") {
        rule->promote = PoolRulePromote::kNever;
      } else if (text == "on_read") {
        rule->promote = PoolRulePromote::kOnRead;
      } else {
        return Fail(error, where + ".promote must be \"tier\", \"never\" or \"on_read\"");
      }
    } else {
      return Fail(error, where + " has an unknown field \"" + name + "\"");
    }
  }
  return true;
}

bool ParseTable(const Struct& root, PoolRuleTable* out, std::string* error) {
  PoolRuleTable table;
  for (const auto& [name, field] : root.fields()) {
    if (name != "policy" && name != "default" && name != "rules") {
      return Fail(error, "unknown field \"" + name + "\"");
    }
  }
  auto policy = root.fields().find("policy");
  if (policy != root.fields().end()) {
    if (policy->second.kind_case() != Value::kStringValue) {
      return Fail(error, "policy must be a string");
    }
    table.policy = policy->second.string_value();
  }
  auto fallback = root.fields().find("default");
  if (fallback != root.fields().end() &&
      !ParseRule(fallback->second, "default", &table.fallback, error)) {
    return false;
  }
  auto rules = root.fields().find("rules");
  if (rules != root.fields().end()) {
    if (rules->second.kind_case() != Value::kStructValue) {
      return Fail(error, "rules must be an object keyed by tag");
    }
    for (const auto& [tag_text, value] : rules->second.struct_value().fields()) {
      size_t used = 0;
      unsigned long long tag = 0;
      try {
        tag = std::stoull(tag_text, &used);
      } catch (...) {
        used = 0;
      }
      if (used != tag_text.size() || tag_text.empty() || tag > 4294967295ULL) {
        return Fail(error, "rules key \"" + tag_text + "\" is not a tag number");
      }
      PoolRule rule = table.fallback;
      if (!ParseRule(value, "rules." + tag_text, &rule, error)) return false;
      table.by_tag[static_cast<uint32_t>(tag)] = rule;
    }
  }
  *out = std::move(table);
  return true;
}

void RuleToStruct(const PoolRule& rule, Struct* out) {
  auto& fields = *out->mutable_fields();
  fields["admit"].set_bool_value(rule.admit);
  fields["weight"].set_number_value(rule.weight);
  fields["demote"].set_bool_value(rule.demote);
  fields["retain_s"].set_number_value(rule.retain_s);
  fields["promote"].set_string_value(PromoteName(rule.promote));
}

void TableToStruct(const PoolRuleTable& table, Struct* out) {
  auto& fields = *out->mutable_fields();
  fields["policy"].set_string_value(table.policy);
  fields["version"].set_number_value(static_cast<double>(table.version));
  RuleToStruct(table.fallback, fields["default"].mutable_struct_value());
  auto* rules = fields["rules"].mutable_struct_value();
  for (const auto& [tag, rule] : table.by_tag) {
    RuleToStruct(rule, (*rules->mutable_fields())[std::to_string(tag)].mutable_struct_value());
  }
}

std::string StructJson(const Struct& message) {
  std::string json;
  google::protobuf::util::MessageToJsonString(message, &json);
  return json;
}

}  // namespace

bool ParsePoolRuleTable(std::string_view json, PoolRuleTable* out, std::string* error) {
  Struct root;
  return ParseStruct(json, &root, error) && ParseTable(root, out, error);
}

std::string PoolRuleTableJson(const PoolRuleTable& table) {
  Struct root;
  TableToStruct(table, &root);
  return StructJson(root);
}

PoolRules::PoolRules() : clock_(SteadyNowNs) {}

void PoolRules::SetClockForTest(Clock clock) {
  std::lock_guard<std::mutex> lock(mutex_);
  clock_ = std::move(clock);
}

size_t PoolRules::Bucket(double idle_s) {
  if (idle_s < 1.0) return 0;
  const auto bucket = static_cast<size_t>(1 + std::floor(std::log2(idle_s)));
  return std::min(bucket, kAgeBuckets - 1);
}

uint32_t PoolRules::PoolIndexLocked(const std::string& pool) {
  auto it = pool_index_.find(pool);
  if (it != pool_index_.end()) return it->second;
  const auto index = static_cast<uint32_t>(pools_.size());
  pools_.push_back(Pool{pool, PoolRuleTable{}, {}});
  pool_index_.emplace(pool, index);
  active_.store(true, std::memory_order_relaxed);
  return index;
}

PoolRules::Class& PoolRules::ClassOfLocked(const Record& record) {
  return pools_[record.pool].classes[record.tag];
}

const PoolRule& PoolRules::RuleOfLocked(const Record& record) const {
  return pools_[record.pool].table.For(record.tag);
}

void PoolRules::RefreshFlagsLocked() {
  bool weighted = false;
  for (const auto& pool : pools_) {
    weighted = weighted || pool.table.fallback.weight != 1.0;
    for (const auto& [tag, rule] : pool.table.by_tag) weighted = weighted || rule.weight != 1.0;
  }
  weighted_.store(weighted, std::memory_order_relaxed);
}

bool PoolRules::SetTable(const std::string& pool, PoolRuleTable table, std::string* error) {
  if (pool.empty()) return Fail(error, "a pool needs a name");
  const auto valid = [](const PoolRule& rule) {
    return rule.weight > 0.0 && std::isfinite(rule.weight);
  };
  if (!valid(table.fallback)) return Fail(error, "default.weight must be a number above 0");
  for (const auto& [tag, rule] : table.by_tag) {
    if (!valid(rule)) {
      return Fail(error, "rules." + std::to_string(tag) + ".weight must be a number above 0");
    }
  }
  std::lock_guard<std::mutex> lock(mutex_);
  Pool& target = pools_[PoolIndexLocked(pool)];
  table.version = target.table.version + 1;
  target.table = std::move(table);
  RefreshFlagsLocked();
  return true;
}

bool PoolRules::SetTableJson(const std::string& pool, std::string_view json, std::string* error) {
  PoolRuleTable table;
  if (!ParsePoolRuleTable(json, &table, error)) return false;
  return SetTable(pool, std::move(table), error);
}

bool PoolRules::LoadJson(std::string_view json, std::string* error) {
  Struct root;
  if (!ParseStruct(json, &root, error)) return false;
  auto pools = root.fields().find("pools");
  if (pools == root.fields().end() || pools->second.kind_case() != Value::kStructValue) {
    return Fail(error, "expected {\"pools\": {\"<pool>\": {...}}}");
  }
  // Parse everything before applying anything: a file is taken whole or not at all.
  std::vector<std::pair<std::string, PoolRuleTable>> parsed;
  for (const auto& [name, value] : pools->second.struct_value().fields()) {
    if (value.kind_case() != Value::kStructValue) {
      return Fail(error, "pools." + name + " must be an object");
    }
    PoolRuleTable table;
    std::string reason;
    if (!ParseTable(value.struct_value(), &table, &reason)) {
      return Fail(error, "pools." + name + ": " + reason);
    }
    parsed.emplace_back(name, std::move(table));
  }
  for (auto& [name, table] : parsed) {
    if (!SetTable(name, std::move(table), error)) return false;
  }
  return true;
}

bool PoolRules::LoadFile(const std::string& path, std::string* error) {
  std::ifstream in(path);
  if (!in) return Fail(error, "cannot read " + path);
  std::ostringstream text;
  text << in.rdbuf();
  return LoadJson(text.str(), error);
}

size_t PoolRules::SetKeyClass(const std::string& pool, const std::vector<std::string>& keys,
                              const std::vector<uint32_t>& tags) {
  if (pool.empty() || keys.empty()) return 0;
  if (tags.size() != keys.size() && tags.size() != 1) return 0;
  std::lock_guard<std::mutex> lock(mutex_);
  const uint32_t pool_index = PoolIndexLocked(pool);
  const int64_t now = clock_();
  size_t recorded = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    const uint32_t tag = tags.size() == 1 ? tags[0] : tags[i];
    auto [it, fresh] = keys_.try_emplace(keys[i]);
    Record& record = it->second;
    if (!fresh && record.pool == pool_index && record.tag == tag) {
      ++recorded;
      continue;
    }
    if (!fresh && record.stored) {
      // Change class in place: leave the old class's books, join the new one's
      // with the recency it already had.
      Class& old_class = ClassOfLocked(record);
      auto& live = old_class.stats.live[TierTypeName(record.tier)];
      if (live.first > 0) --live.first;
      live.second -= std::min(live.second, record.bytes);
      old_class.lru.erase(record.lru);
      record.pool = pool_index;
      record.tag = tag;
      Class& new_class = ClassOfLocked(record);
      // Keep the list ordered by last use: insert before the first older key.
      // Walk from the recent end, then from the cold end; a list too long to
      // settle within the bound takes the key at the cold-end walk's stop, which
      // can only make it look older than it is.
      const auto older = [this, &record](const std::string& other_key) {
        auto other = keys_.find(other_key);
        return other == keys_.end() || other->second.last_ns <= record.last_ns;
      };
      auto position = new_class.lru.begin();
      size_t steps = 0;
      while (position != new_class.lru.end() && steps < kMaxPlacementWalk && !older(*position)) {
        ++position;
        ++steps;
      }
      if (position != new_class.lru.end() && !older(*position)) {
        position = new_class.lru.end();
        for (steps = 0; steps < kMaxPlacementWalk && position != new_class.lru.begin(); ++steps) {
          auto previous = std::prev(position);
          if (!older(*previous)) break;
          position = previous;
        }
      }
      record.lru = new_class.lru.insert(position, keys[i]);
      auto& joined = new_class.stats.live[TierTypeName(record.tier)];
      ++joined.first;
      joined.second += record.bytes;
      ++new_class.stats.declared;
    } else {
      record.pool = pool_index;
      record.tag = tag;
      record.last_ns = now;
      ++ClassOfLocked(record).stats.declared;
      if (fresh) {
        declared_fifo_.push_back(keys[i]);
        // The queue only has to remember keys still waiting for their put:
        // drop the ones at its head that were stored or forgotten since, and
        // past the bound the oldest one still waiting.
        while (!declared_fifo_.empty()) {
          auto oldest = keys_.find(declared_fifo_.front());
          const bool waiting = oldest != keys_.end() && !oldest->second.stored;
          if (waiting && declared_fifo_.size() <= kMaxDeclared) break;
          if (waiting) keys_.erase(oldest);
          declared_fifo_.pop_front();
        }
      }
    }
    ++recorded;
  }
  return recorded;
}

bool PoolRules::Admit(const std::string& key, uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = keys_.find(key);
  if (it == keys_.end()) return true;
  const Record& record = it->second;
  if (RuleOfLocked(record).admit) return true;
  ClassStats& stats = ClassOfLocked(record).stats;
  ++stats.declined;
  stats.declined_bytes += bytes;
  // The declaration stays: a caller that retries the put without declaring
  // again must get the same answer, not an unclassified object.
  return false;
}

bool PoolRules::EntryTierOnly(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = keys_.find(key);
  return it != keys_.end() && !RuleOfLocked(it->second).demote;
}

void PoolRules::OnStored(const std::string& key, uint32_t backend_id, TierType tier,
                         uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = keys_.find(key);
  if (it == keys_.end()) return;
  Record& record = it->second;
  Class& cls = ClassOfLocked(record);
  if (record.stored) UnstoreLocked(&record, &cls);
  const int64_t now = clock_();
  record.stored = true;
  record.backend_id = backend_id;
  record.tier = tier;
  record.bytes = bytes;
  record.last_ns = record.born_ns = now;
  record.reads = 0;
  cls.lru.push_front(key);
  record.lru = cls.lru.begin();
  ++cls.stats.admitted;
  cls.stats.admitted_bytes += bytes;
  auto& live = cls.stats.live[TierTypeName(tier)];
  ++live.first;
  live.second += bytes;
}

void PoolRules::UnstoreLocked(Record* record, Class* cls) {
  auto& live = cls->stats.live[TierTypeName(record->tier)];
  if (live.first > 0) --live.first;
  live.second -= std::min(live.second, record->bytes);
  cls->lru.erase(record->lru);
  record->stored = false;
}

void PoolRules::OnAccess(const std::string& key, bool read) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = keys_.find(key);
  if (it == keys_.end() || !it->second.stored) return;
  Record& record = it->second;
  Class& cls = ClassOfLocked(record);
  const int64_t now = clock_();
  const double idle_s = static_cast<double>(now - record.last_ns) / 1e9;
  ++cls.stats.reuse_by_idle[Bucket(idle_s)];
  if (read) {
    ++cls.stats.reads;
    ++record.reads;
  } else {
    ++cls.stats.touches;
  }
  record.last_ns = now;
  cls.lru.splice(cls.lru.begin(), cls.lru, record.lru);
}

void PoolRules::OnMoved(const std::string& key, uint32_t backend_id, TierType tier, bool promotion,
                        uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = keys_.find(key);
  if (it == keys_.end() || !it->second.stored) return;
  Record& record = it->second;
  ClassStats& stats = ClassOfLocked(record).stats;
  auto& from = stats.live[TierTypeName(record.tier)];
  if (from.first > 0) --from.first;
  from.second -= std::min(from.second, record.bytes);
  record.backend_id = backend_id;
  record.tier = tier;
  if (bytes != 0) record.bytes = bytes;  // the target medium may round differently
  auto& to = stats.live[TierTypeName(tier)];
  ++to.first;
  to.second += record.bytes;
  if (promotion) {
    ++stats.promoted;
    stats.promoted_bytes += bytes;
  } else {
    ++stats.demoted;
    stats.demoted_bytes += bytes;
  }
}

void PoolRules::OnDropped(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = keys_.find(key);
  if (it == keys_.end() || !it->second.stored) return;
  Record& record = it->second;
  Class& cls = ClassOfLocked(record);
  ++cls.stats.dropped;
  cls.stats.dropped_bytes += record.bytes;
  if (record.reads == 0) ++cls.stats.dropped_unread;
  cls.stats.residency_s += static_cast<double>(clock_() - record.born_ns) / 1e9;
  UnstoreLocked(&record, &cls);
  keys_.erase(it);
}

bool PoolRules::ShouldDemote(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = keys_.find(key);
  if (it == keys_.end()) return true;
  const PoolRule& rule = RuleOfLocked(it->second);
  if (!rule.demote) return false;
  if (rule.retain_s == 0) return true;
  const double idle_s = static_cast<double>(clock_() - it->second.last_ns) / 1e9;
  return idle_s < static_cast<double>(rule.retain_s);
}

PoolRulePromote PoolRules::Promote(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = keys_.find(key);
  return it == keys_.end() ? PoolRulePromote::kTier : RuleOfLocked(it->second).promote;
}

double PoolRules::Weight(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = keys_.find(key);
  return it == keys_.end() ? 1.0 : RuleOfLocked(it->second).weight;
}

void PoolRules::ApplyWeights(uint32_t backend_id, bool complete,
                             std::vector<EvictionOffer>* offers) {
  if (offers == nullptr || offers->empty()) return;
  for (const auto& offer : *offers) {
    if (offer.idle_ms < 0) return;  // this medium keeps no idle times: its order stands
  }
  std::lock_guard<std::mutex> lock(mutex_);
  // Everything the medium did not offer is at most this idle, so an unweighted
  // key outside the window cannot outrank a candidate scoring above it.
  const double cut = complete ? 0.0 : static_cast<double>(offers->back().idle_ms);
  std::vector<std::pair<double, EvictionOffer>> scored;
  scored.reserve(offers->size());
  for (auto& offer : *offers) {
    auto it = keys_.find(offer.key);
    const double weight = it == keys_.end() ? 1.0 : RuleOfLocked(it->second).weight;
    // Fast-ageing classes are collected from their own lists below.
    if (weight > 1.0 && it->second.stored) continue;
    const double score = static_cast<double>(offer.idle_ms) * weight;
    if (score >= cut) scored.emplace_back(score, std::move(offer));
  }
  const int64_t now = clock_();
  const size_t limit = std::max<size_t>(offers->size(), 64);
  for (auto& pool : pools_) {
    for (auto& [tag, cls] : pool.classes) {
      const double weight = pool.table.For(tag).weight;
      if (weight <= 1.0) continue;
      size_t taken = 0;
      for (auto key = cls.lru.rbegin(); key != cls.lru.rend() && taken < limit; ++key) {
        auto it = keys_.find(*key);
        if (it == keys_.end()) continue;
        const Record& record = it->second;
        const double idle_ms = static_cast<double>(now - record.last_ns) / 1e6;
        const double score = idle_ms * weight;
        if (score < cut) break;  // the list is coldest first: the rest score lower still
        if (record.backend_id != backend_id) continue;
        scored.emplace_back(score,
                            EvictionOffer{*key, record.bytes, static_cast<int64_t>(idle_ms)});
        ++taken;
      }
    }
  }
  std::stable_sort(scored.begin(), scored.end(),
                   [](const auto& a, const auto& b) { return a.first > b.first; });
  offers->clear();
  offers->reserve(scored.size());
  for (auto& entry : scored) offers->push_back(std::move(entry.second));
}

void PoolRules::ClearKeys() {
  std::lock_guard<std::mutex> lock(mutex_);
  keys_.clear();
  declared_fifo_.clear();
  for (auto& pool : pools_) {
    for (auto& [tag, cls] : pool.classes) {
      cls.lru.clear();
      cls.stats.live.clear();
    }
  }
}

std::string PoolRules::StateJson() const {
  std::lock_guard<std::mutex> lock(mutex_);
  Struct root;
  auto& top = *root.mutable_fields();
  auto* edges = top["idle_bucket_upper_s"].mutable_list_value();
  for (size_t bucket = 0; bucket + 1 < kAgeBuckets; ++bucket) {
    edges->add_values()->set_number_value(static_cast<double>(uint64_t{1} << bucket));
  }
  auto* pools = top["pools"].mutable_struct_value();
  for (const auto& pool : pools_) {
    auto& entry = *(*pools->mutable_fields())[pool.name].mutable_struct_value()->mutable_fields();
    TableToStruct(pool.table, entry["table"].mutable_struct_value());
    auto* classes = entry["classes"].mutable_struct_value();
    for (const auto& [tag, cls] : pool.classes) {
      const ClassStats& stats = cls.stats;
      auto& fields = *(*classes->mutable_fields())[std::to_string(tag)]
                          .mutable_struct_value()
                          ->mutable_fields();
      const auto number = [&fields](const char* name, double value) {
        fields[name].set_number_value(value);
      };
      number("declared", static_cast<double>(stats.declared));
      number("admitted", static_cast<double>(stats.admitted));
      number("admitted_bytes", static_cast<double>(stats.admitted_bytes));
      number("declined", static_cast<double>(stats.declined));
      number("declined_bytes", static_cast<double>(stats.declined_bytes));
      number("reads", static_cast<double>(stats.reads));
      number("touches", static_cast<double>(stats.touches));
      number("demoted", static_cast<double>(stats.demoted));
      number("demoted_bytes", static_cast<double>(stats.demoted_bytes));
      number("promoted", static_cast<double>(stats.promoted));
      number("promoted_bytes", static_cast<double>(stats.promoted_bytes));
      number("dropped", static_cast<double>(stats.dropped));
      number("dropped_bytes", static_cast<double>(stats.dropped_bytes));
      number("dropped_unread", static_cast<double>(stats.dropped_unread));
      number("mean_residency_s",
             stats.dropped == 0 ? 0.0 : stats.residency_s / static_cast<double>(stats.dropped));
      auto* reuse = fields["reuse_by_idle"].mutable_list_value();
      for (uint64_t count : stats.reuse_by_idle) {
        reuse->add_values()->set_number_value(static_cast<double>(count));
      }
      auto* live = fields["live"].mutable_struct_value();
      for (const auto& [tier, counts] : stats.live) {
        auto& tier_fields =
            *(*live->mutable_fields())[tier].mutable_struct_value()->mutable_fields();
        tier_fields["keys"].set_number_value(static_cast<double>(counts.first));
        tier_fields["bytes"].set_number_value(static_cast<double>(counts.second));
      }
    }
  }
  return StructJson(root);
}

}  // namespace mori::umbp
