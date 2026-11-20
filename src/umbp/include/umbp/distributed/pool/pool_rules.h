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
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

// Pool rules: a pool is a name the caller chose for a group of keys, and each
// key in it carries a tag, an opaque class label.  UMBP never interprets either.
// Each pool has one table with a rule per tag, and the rule answers the four
// management decisions for the objects of that class:
//
//   put    admit              store the object, or decline it
//   evict  weight             the object with the largest idle time x weight leaves first
//   move   demote, retain_s   on leaving a tier: go down to the next one, or be dropped
//   get    promote            on a read from a lower tier: move up, or stay
//
// A table is plain data.  It can be loaded at start-up or replaced at any time
// while the pool runs (SetTable); stored bytes are untouched and the next
// decision uses the new table.  Keys that were never given a class behave
// exactly as before.

enum class PoolRulePromote {
  kTier,    // follow the logical tier's promote_trigger (the behaviour without rules)
  kNever,   // stay where it is
  kOnRead,  // move up on the first read
};

struct PoolRule {
  bool admit = true;
  // > 1 ages faster than an unclassified key, < 1 slower.  Must be > 0.
  double weight = 1.0;
  bool demote = true;
  // Idle seconds after which the object is no longer worth a write to a lower
  // tier and is dropped instead.  0 means no limit.
  uint32_t retain_s = 0;
  PoolRulePromote promote = PoolRulePromote::kTier;
};

struct PoolRuleTable {
  std::string policy = "default";  // free text: who or what produced the table
  uint64_t version = 0;            // assigned by PoolRules::SetTable
  PoolRule fallback;               // tags without a rule of their own
  std::map<uint32_t, PoolRule> by_tag;

  const PoolRule& For(uint32_t tag) const {
    auto it = by_tag.find(tag);
    return it == by_tag.end() ? fallback : it->second;
  }
};

// JSON form of one table:
//   {"policy": "name",
//    "default": {"admit": true, "weight": 1, "demote": true, "retain_s": 0, "promote": "tier"},
//    "rules": {"3": {"weight": 8, "demote": false}, "7": {"admit": false}}}
// Every field is optional; a rule inherits what it omits from "default".
bool ParsePoolRuleTable(std::string_view json, PoolRuleTable* out, std::string* error);
std::string PoolRuleTableJson(const PoolRuleTable& table);

class PoolRules {
 public:
  // Idle-time buckets of the reuse histogram: < 1 s, 1-2 s, 2-4 s, ... >= 4096 s.
  static constexpr size_t kAgeBuckets = 14;
  // Classes declared for keys that are not stored yet are remembered up to here.
  static constexpr size_t kMaxDeclared = size_t{1} << 18;
  // Steps a re-tagged key's new list is walked from either end to find its
  // place by recency; past that the place is approximate.
  static constexpr size_t kMaxPlacementWalk = 1024;

  using Clock = std::function<int64_t()>;  // nanoseconds, monotonic

  PoolRules();

  // ---- configuration (any thread, any time) ----

  // Replace `pool`'s table.  False, with *error set, when the table is malformed.
  bool SetTable(const std::string& pool, PoolRuleTable table, std::string* error = nullptr);
  bool SetTableJson(const std::string& pool, std::string_view json, std::string* error = nullptr);
  // {"pools": {"<pool>": <table>, ...}}
  bool LoadJson(std::string_view json, std::string* error = nullptr);
  bool LoadFile(const std::string& path, std::string* error = nullptr);

  // Give keys their class.  A key that is already stored changes class at once;
  // otherwise the class is remembered and applied when the key is put.
  // `tags` is one per key, or a single tag for all of them.  Returns the number
  // of keys recorded.
  size_t SetKeyClass(const std::string& pool, const std::vector<std::string>& keys,
                     const std::vector<uint32_t>& tags);

  // Tables and per-class counters as JSON (see doc/pool-rules.md).
  std::string StateJson() const;

  // ---- decisions and events (called by PeerPool) ----

  // True once any key class or table exists; lets the hot path skip the lock.
  bool Active() const { return active_.load(std::memory_order_relaxed); }
  // True while some rule in force has a weight other than 1.
  bool Weighted() const { return weighted_.load(std::memory_order_relaxed); }

  bool Admit(const std::string& key, uint64_t bytes);
  void OnStored(const std::string& key, uint32_t backend_id, TierType tier, uint64_t bytes);
  void OnAccess(const std::string& key, bool read);
  void OnMoved(const std::string& key, uint32_t backend_id, TierType tier, bool promotion,
               uint64_t bytes);
  void OnDropped(const std::string& key);
  bool ShouldDemote(const std::string& key);
  // True for a key whose rule keeps it off lower tiers (`demote: false`): the
  // put is then not placed on one either when its entry tier is full.
  bool EntryTierOnly(const std::string& key);
  PoolRulePromote Promote(const std::string& key);
  double Weight(const std::string& key);

  // Reorders one medium's eviction offers (coldest first, each with idle_ms)
  // by idle time x weight, adding keys of fast-ageing classes that sit deeper
  // than the offered window.  `complete` says the offers are everything the
  // medium has.  Offers without idle times are returned unchanged.
  void ApplyWeights(uint32_t backend_id, bool complete, std::vector<EvictionOffer>* offers);

  // Forget every key (ClearLocal); tables and totals stay.
  void ClearKeys();

  void SetClockForTest(Clock clock);

 private:
  struct ClassStats {
    uint64_t declared = 0;
    uint64_t admitted = 0, admitted_bytes = 0;
    uint64_t declined = 0, declined_bytes = 0;
    uint64_t reads = 0, touches = 0;
    std::array<uint64_t, kAgeBuckets> reuse_by_idle{};
    uint64_t demoted = 0, demoted_bytes = 0;
    uint64_t promoted = 0, promoted_bytes = 0;
    uint64_t dropped = 0, dropped_bytes = 0, dropped_unread = 0;
    double residency_s = 0.0;
    std::map<std::string, std::pair<uint64_t, uint64_t>> live;  // tier name -> (keys, bytes)
  };
  struct Class {
    ClassStats stats;
    std::list<std::string> lru;  // stored keys, most recently used first
  };
  struct Pool {
    std::string name;
    PoolRuleTable table;
    std::map<uint32_t, Class> classes;
  };
  struct Record {
    uint32_t pool = 0;
    uint32_t tag = 0;
    bool stored = false;
    uint32_t backend_id = 0;
    TierType tier = TierType::UNKNOWN;
    uint64_t bytes = 0;
    int64_t last_ns = 0;
    int64_t born_ns = 0;
    uint32_t reads = 0;
    std::list<std::string>::iterator lru{};
  };

  uint32_t PoolIndexLocked(const std::string& pool);
  Class& ClassOfLocked(const Record& record);
  const PoolRule& RuleOfLocked(const Record& record) const;
  void UnstoreLocked(Record* record, Class* cls);
  void RefreshFlagsLocked();
  static size_t Bucket(double idle_s);

  mutable std::mutex mutex_;
  std::vector<Pool> pools_;
  std::unordered_map<std::string, uint32_t> pool_index_;
  std::unordered_map<std::string, Record> keys_;
  std::deque<std::string> declared_fifo_;
  std::atomic<bool> active_{false};
  std::atomic<bool> weighted_{false};
  Clock clock_;
};

}  // namespace mori::umbp
