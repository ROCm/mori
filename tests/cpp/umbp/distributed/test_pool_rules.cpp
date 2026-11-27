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
#include <google/protobuf/struct.pb.h>
#include <google/protobuf/util/json_util.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "umbp/distributed/pool/pool_rules.h"

namespace mori::umbp {
namespace {

constexpr int64_t kSecond = 1000000000;

// Reads one number out of PoolRules::StateJson() by path, e.g.
// {"pools", "state", "classes", "7", "declined"}.  -1 when the path is absent.
double StateNumber(const std::string& json, const std::vector<std::string>& path) {
  google::protobuf::Struct root;
  if (!google::protobuf::util::JsonStringToMessage(json, &root).ok()) return -1;
  const google::protobuf::Struct* node = &root;
  for (size_t i = 0; i < path.size(); ++i) {
    auto it = node->fields().find(path[i]);
    if (it == node->fields().end()) return -1;
    if (i + 1 == path.size()) {
      return it->second.kind_case() == google::protobuf::Value::kNumberValue
                 ? it->second.number_value()
                 : -1;
    }
    if (it->second.kind_case() != google::protobuf::Value::kStructValue) return -1;
    node = &it->second.struct_value();
  }
  return -1;
}

double ClassNumber(const std::string& json, const std::string& pool, const std::string& tag,
                   const std::vector<std::string>& field) {
  std::vector<std::string> path = {"pools", pool, "classes", tag};
  path.insert(path.end(), field.begin(), field.end());
  return StateNumber(json, path);
}

std::vector<std::string> Keys(const std::vector<EvictionOffer>& offers) {
  std::vector<std::string> keys;
  for (const auto& offer : offers) keys.push_back(offer.key);
  return keys;
}

// A PoolRules on a clock the test moves by hand.
struct Fixture {
  Fixture() {
    rules.SetClockForTest([this] { return now; });
  }
  void Store(const std::string& pool, const std::string& key, uint32_t tag, uint32_t backend = 0,
             TierType tier = TierType::DRAM, uint64_t bytes = 100) {
    ASSERT_EQ(rules.SetKeyClass(pool, {key}, {tag}), 1u);
    ASSERT_TRUE(rules.Admit(key, bytes));
    rules.OnStored(key, backend, tier, bytes);
  }
  int64_t now = 1000 * kSecond;
  PoolRules rules;
};

TEST(PoolRuleTable, RulesInheritWhatTheyOmitFromTheDefault) {
  PoolRuleTable table;
  std::string error;
  ASSERT_TRUE(ParsePoolRuleTable(
      R"({"policy": "hand", "default": {"weight": 2, "promote": "never"},
          "rules": {"3": {"demote": false}, "7": {"admit": false, "weight": 8, "retain_s": 60}}})",
      &table, &error))
      << error;
  EXPECT_EQ(table.policy, "hand");
  EXPECT_DOUBLE_EQ(table.For(3).weight, 2.0);
  EXPECT_FALSE(table.For(3).demote);
  EXPECT_EQ(table.For(3).promote, PoolRulePromote::kNever);
  EXPECT_FALSE(table.For(7).admit);
  EXPECT_DOUBLE_EQ(table.For(7).weight, 8.0);
  EXPECT_EQ(table.For(7).retain_s, 60u);
  EXPECT_TRUE(table.For(99).admit) << "a tag without a rule takes the default";
  EXPECT_DOUBLE_EQ(table.For(99).weight, 2.0);
}

TEST(PoolRuleTable, MalformedTablesAreRejectedWithTheReason) {
  PoolRuleTable table;
  std::string error;
  EXPECT_FALSE(ParsePoolRuleTable("{", &table, &error));
  EXPECT_FALSE(ParsePoolRuleTable(R"({"rules": {"x": {}}})", &table, &error));
  EXPECT_NE(error.find("not a tag number"), std::string::npos) << error;
  EXPECT_FALSE(ParsePoolRuleTable(R"({"default": {"weight": 0}})", &table, &error));
  EXPECT_NE(error.find("weight"), std::string::npos) << error;
  EXPECT_FALSE(ParsePoolRuleTable(R"({"default": {"promote": "sometimes"}})", &table, &error));
  EXPECT_FALSE(ParsePoolRuleTable(R"({"default": {"admitt": true}})", &table, &error));
  EXPECT_NE(error.find("unknown field"), std::string::npos) << error;
  EXPECT_FALSE(ParsePoolRuleTable(R"({"tables": {}})", &table, &error));
}

TEST(PoolRuleTable, PrintsEveryFieldOfEveryRule) {
  PoolRuleTable table;
  table.policy = "t";
  table.fallback.weight = 0.5;
  table.by_tag[4] = PoolRule{false, 3.0, false, 30, PoolRulePromote::kOnRead};
  PoolRuleTable parsed;
  std::string error;
  // The printed form carries the version PoolRules assigned; a table to load does not.
  EXPECT_FALSE(ParsePoolRuleTable(PoolRuleTableJson(table), &parsed, &error));
  EXPECT_NE(error.find("version"), std::string::npos) << error;
  const std::string json = PoolRuleTableJson(table);
  EXPECT_EQ(StateNumber(json, {"rules", "4", "weight"}), 3) << json;
  EXPECT_EQ(StateNumber(json, {"rules", "4", "retain_s"}), 30) << json;
  EXPECT_EQ(StateNumber(json, {"default", "weight"}), 0.5) << json;
}

TEST(PoolRules, UnclassifiedKeysAreUntouchedByAnyTable) {
  Fixture f;
  ASSERT_TRUE(f.rules.SetTableJson("state", R"({"default": {"admit": false, "weight": 9}})"));
  EXPECT_TRUE(f.rules.Admit("never-declared", 10));
  EXPECT_TRUE(f.rules.ShouldDemote("never-declared"));
  EXPECT_EQ(f.rules.Promote("never-declared"), PoolRulePromote::kTier);
  EXPECT_DOUBLE_EQ(f.rules.Weight("never-declared"), 1.0);
}

TEST(PoolRules, InactiveUntilAPoolOrKeyClassExists) {
  PoolRules rules;
  EXPECT_FALSE(rules.Active());
  EXPECT_FALSE(rules.Weighted());
  ASSERT_TRUE(rules.SetTableJson("p", R"({"rules": {"1": {"weight": 4}}})"));
  EXPECT_TRUE(rules.Active());
  EXPECT_TRUE(rules.Weighted());
  ASSERT_TRUE(rules.SetTableJson("p", "{}"));
  EXPECT_FALSE(rules.Weighted()) << "the replaced table had the only weight";
}

TEST(PoolRules, PutDecisionFollowsTheTableInForce) {
  Fixture f;
  ASSERT_TRUE(f.rules.SetTableJson("state", R"({"rules": {"7": {"admit": false}}})"));
  ASSERT_EQ(f.rules.SetKeyClass("state", {"a", "b"}, {7, 1}), 2u);
  EXPECT_FALSE(f.rules.Admit("a", 64));
  EXPECT_TRUE(f.rules.Admit("b", 64));
  // Replaced while running: the same class is admitted from the next put on.
  ASSERT_TRUE(f.rules.SetTableJson("state", "{}"));
  ASSERT_EQ(f.rules.SetKeyClass("state", {"a"}, {7}), 1u);
  EXPECT_TRUE(f.rules.Admit("a", 64));
  const std::string state = f.rules.StateJson();
  EXPECT_EQ(ClassNumber(state, "state", "7", {"declined"}), 1) << state;
  EXPECT_EQ(ClassNumber(state, "state", "7", {"declined_bytes"}), 64) << state;
  EXPECT_EQ(StateNumber(state, {"pools", "state", "table", "version"}), 2) << state;
}

TEST(PoolRules, OneTagForAllKeysOrOnePerKey) {
  Fixture f;
  EXPECT_EQ(f.rules.SetKeyClass("p", {"a", "b", "c"}, {5}), 3u);
  EXPECT_EQ(f.rules.SetKeyClass("p", {"a", "b"}, {1, 2}), 2u);
  EXPECT_EQ(f.rules.SetKeyClass("p", {"a", "b"}, {1, 2, 3}), 0u) << "tag count matches nothing";
  EXPECT_EQ(f.rules.SetKeyClass("", {"a"}, {1}), 0u);
}

TEST(PoolRules, MoveDecisionDropsInsteadOfDemoting) {
  Fixture f;
  ASSERT_TRUE(f.rules.SetTableJson(
      "state", R"({"rules": {"1": {"demote": false}, "2": {"retain_s": 30}}})"));
  f.Store("state", "never-down", 1);
  f.Store("state", "down-while-fresh", 2);
  f.Store("state", "plain", 3);
  EXPECT_FALSE(f.rules.ShouldDemote("never-down"));
  EXPECT_TRUE(f.rules.ShouldDemote("down-while-fresh"));
  EXPECT_TRUE(f.rules.ShouldDemote("plain"));
  f.now += 31 * kSecond;
  EXPECT_FALSE(f.rules.ShouldDemote("down-while-fresh")) << "idle past retain_s";
  f.rules.OnAccess("down-while-fresh", /*read=*/true);
  EXPECT_TRUE(f.rules.ShouldDemote("down-while-fresh")) << "a use restarts the clock";
}

TEST(PoolRules, GetDecisionOverridesTheTierOnlyWhereARuleSaysSo) {
  Fixture f;
  ASSERT_TRUE(f.rules.SetTableJson(
      "state", R"({"rules": {"1": {"promote": "never"}, "2": {"promote": "on_read"}}})"));
  f.Store("state", "a", 1);
  f.Store("state", "b", 2);
  f.Store("state", "c", 3);
  EXPECT_EQ(f.rules.Promote("a"), PoolRulePromote::kNever);
  EXPECT_EQ(f.rules.Promote("b"), PoolRulePromote::kOnRead);
  EXPECT_EQ(f.rules.Promote("c"), PoolRulePromote::kTier);
}

TEST(PoolRules, EvictionOrderIsIdleTimeTimesWeight) {
  Fixture f;
  ASSERT_TRUE(f.rules.SetTableJson("state", R"({"rules": {"1": {"weight": 10}}})"));
  // Unclassified keys idle 100 s and 60 s; a fast-ageing key idle 20 s (scores 200).
  f.Store("state", "fast", 1);
  f.now += 20 * kSecond;
  std::vector<EvictionOffer> offers = {
      {"old", 10, 100000}, {"mid", 10, 60000}, {"fast", 100, 20000}};
  f.rules.ApplyWeights(0, /*complete=*/true, &offers);
  EXPECT_EQ(Keys(offers), (std::vector<std::string>{"fast", "old", "mid"}));
}

TEST(PoolRules, FastAgeingKeysAreFoundBeyondTheOfferedWindow) {
  Fixture f;
  ASSERT_TRUE(f.rules.SetTableJson("state", R"({"rules": {"1": {"weight": 10}}})"));
  f.Store("state", "deep", 1, /*backend=*/3);
  f.Store("state", "elsewhere", 1, /*backend=*/4);
  f.now += 20 * kSecond;
  // The medium offered only its two coldest keys; "deep" (idle 20 s) was not among them.
  std::vector<EvictionOffer> offers = {{"old", 10, 100000}, {"mid", 10, 60000}};
  f.rules.ApplyWeights(3, /*complete=*/false, &offers);
  EXPECT_EQ(Keys(offers), (std::vector<std::string>{"deep", "old", "mid"}))
      << "a key on another medium must not be offered";
}

TEST(PoolRules, SlowAgeingKeysYieldToYoungerOnes) {
  Fixture f;
  ASSERT_TRUE(f.rules.SetTableJson("kv", R"({"rules": {"1": {"weight": 0.1}}})"));
  f.Store("kv", "kept", 1);
  // Window complete: everything is ranked, the protected key last.
  std::vector<EvictionOffer> offers = {{"kept", 10, 100000}, {"plain", 10, 60000}};
  f.rules.ApplyWeights(0, /*complete=*/true, &offers);
  EXPECT_EQ(Keys(offers), (std::vector<std::string>{"plain", "kept"}));
  // Window incomplete: a key scoring below what was not offered is held back.
  offers = {{"kept", 10, 100000}, {"plain", 10, 60000}};
  f.rules.ApplyWeights(0, /*complete=*/false, &offers);
  EXPECT_EQ(Keys(offers), (std::vector<std::string>{"plain"}));
}

TEST(PoolRules, AMediumWithoutIdleTimesKeepsItsOwnOrder) {
  Fixture f;
  ASSERT_TRUE(f.rules.SetTableJson("state", R"({"rules": {"1": {"weight": 10}}})"));
  f.Store("state", "fast", 1);
  std::vector<EvictionOffer> offers = {{"a", 10}, {"fast", 10}, {"b", 10}};
  f.rules.ApplyWeights(0, /*complete=*/true, &offers);
  EXPECT_EQ(Keys(offers), (std::vector<std::string>{"a", "fast", "b"}));
}

TEST(PoolRules, CountersFollowAKeyThroughItsLife) {
  Fixture f;
  f.Store("state", "k", 9, /*backend=*/0, TierType::DRAM, 4096);
  f.now += 3 * kSecond;
  f.rules.OnAccess("k", /*read=*/true);  // idle 3 s: bucket 2 (2-4 s)
  f.now += 100 * kSecond;
  f.rules.OnAccess("k", /*read=*/false);  // idle 100 s: bucket 7 (64-128 s)
  f.rules.OnMoved("k", 1, TierType::SSD, /*promotion=*/false, 4096);
  std::string state = f.rules.StateJson();
  const auto number = [&state](const std::vector<std::string>& field) {
    return ClassNumber(state, "state", "9", field);
  };
  EXPECT_EQ(number({"admitted"}), 1) << state;
  EXPECT_EQ(number({"admitted_bytes"}), 4096);
  EXPECT_EQ(number({"reads"}), 1);
  EXPECT_EQ(number({"touches"}), 1);
  EXPECT_NE(state.find("\"reuse_by_idle\":[0,0,1,0,0,0,0,1,0,0,0,0,0,0]"), std::string::npos)
      << state;
  EXPECT_EQ(number({"demoted_bytes"}), 4096);
  EXPECT_EQ(number({"live", "SSD", "keys"}), 1);
  EXPECT_EQ(number({"live", "SSD", "bytes"}), 4096);
  EXPECT_EQ(number({"live", "DRAM", "keys"}), 0);

  f.now += 17 * kSecond;
  f.rules.OnDropped("k");
  state = f.rules.StateJson();
  EXPECT_EQ(number({"dropped"}), 1) << state;
  EXPECT_EQ(number({"dropped_unread"}), 0);
  EXPECT_EQ(number({"mean_residency_s"}), 120);
  EXPECT_EQ(number({"live", "SSD", "keys"}), 0);
  EXPECT_TRUE(f.rules.Admit("k", 1)) << "a dropped key is unclassified again";
}

TEST(PoolRules, AStoredKeyCanChangeClassAndKeepsItsRecency) {
  Fixture f;
  ASSERT_TRUE(f.rules.SetTableJson("state", R"({"rules": {"2": {"weight": 10}}})"));
  f.Store("state", "older", 2);
  f.now += 50 * kSecond;
  f.Store("state", "moved", 1);
  f.now += 10 * kSecond;
  f.Store("state", "newer", 2);
  ASSERT_EQ(f.rules.SetKeyClass("state", {"moved"}, {2}), 1u);  // joins the fast-ageing class
  f.now += 10 * kSecond;
  std::vector<EvictionOffer> offers = {{"plain", 10, 1}};
  f.rules.ApplyWeights(0, /*complete=*/true, &offers);
  EXPECT_EQ(Keys(offers), (std::vector<std::string>{"older", "moved", "newer", "plain"}));
  const std::string state = f.rules.StateJson();
  EXPECT_EQ(ClassNumber(state, "state", "1", {"live", "DRAM", "keys"}), 0) << state;
  EXPECT_EQ(ClassNumber(state, "state", "2", {"live", "DRAM", "keys"}), 3) << state;
}

TEST(PoolRules, PendingClassesAreBoundedAndTheOldestIsForgottenFirst) {
  Fixture f;
  ASSERT_TRUE(f.rules.SetTableJson("state", R"({"default": {"admit": false}})"));
  std::vector<std::string> keys;
  keys.reserve(PoolRules::kMaxDeclared + 2);
  for (size_t i = 0; i < PoolRules::kMaxDeclared + 2; ++i) keys.push_back("k" + std::to_string(i));
  ASSERT_EQ(f.rules.SetKeyClass("state", keys, {1}), keys.size());
  EXPECT_TRUE(f.rules.Admit(keys.front(), 1)) << "the oldest pending class was forgotten";
  EXPECT_FALSE(f.rules.Admit(keys.back(), 1)) << "the newest is still remembered";
}

TEST(PoolRules, AFileIsTakenWholeOrNotAtAll) {
  PoolRules rules;
  std::string error;
  EXPECT_FALSE(rules.LoadJson(
      R"({"pools": {"good": {"default": {"weight": 2}}, "bad": {"default": {"weight": -1}}}})",
      &error));
  EXPECT_NE(error.find("pools.bad"), std::string::npos) << error;
  EXPECT_FALSE(rules.Weighted()) << "the good table must not have been applied";
  EXPECT_TRUE(rules.LoadJson(
      R"({"pools": {"kv": {}, "state": {"rules": {"1": {"weight": 8, "demote": false}}}}})",
      &error))
      << error;
  EXPECT_TRUE(rules.Weighted());
  const std::string state = rules.StateJson();
  EXPECT_EQ(StateNumber(state, {"pools", "kv", "table", "version"}), 1) << state;
  EXPECT_EQ(StateNumber(state, {"pools", "state", "table", "rules", "1", "weight"}), 8) << state;
}

TEST(PoolRules, ClearForgetsKeysButKeepsTables) {
  Fixture f;
  ASSERT_TRUE(f.rules.SetTableJson("state", R"({"rules": {"1": {"admit": false}}})"));
  f.Store("state", "k", 2);
  f.rules.ClearKeys();
  EXPECT_DOUBLE_EQ(f.rules.Weight("k"), 1.0);
  ASSERT_EQ(f.rules.SetKeyClass("state", {"again"}, {1}), 1u);
  EXPECT_FALSE(f.rules.Admit("again", 1)) << "the table survived the clear";
}

}  // namespace
}  // namespace mori::umbp
