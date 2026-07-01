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
//
// Master-side eviction policy plug-in.  Covers the default
// LruMasterEvictStrategy behavior: oldest-first within each per-(node,tier)
// byte budget.
#include <gtest/gtest.h>

#include <chrono>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/master/evict_strategy.h"
#include "umbp/distributed/master/global_block_index.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {
namespace {

using Clock = std::chrono::steady_clock;

EvictionCandidate MakeCandidate(const std::string& key, const std::string& node, TierType tier,
                                uint64_t size, Clock::time_point accessed) {
  EvictionCandidate c;
  c.key = key;
  c.location = Location{node, size, tier};
  c.last_accessed_at = accessed;
  c.size = size;
  return c;
}

TEST(LruMasterEvictStrategy, PicksOldestFirstUntilBudgetMet) {
  LruMasterEvictStrategy strategy;
  auto now = Clock::now();
  // newest -> oldest: c (now), a (now-2s), b (now-1s).  Oldest is `a`.
  std::vector<EvictionCandidate> candidates = {
      MakeCandidate("c", "n1", TierType::DRAM, 100, now),
      MakeCandidate("a", "n1", TierType::DRAM, 100, now - std::chrono::seconds(2)),
      MakeCandidate("b", "n1", TierType::DRAM, 100, now - std::chrono::seconds(1)),
  };
  std::unordered_map<std::string, std::map<TierType, int64_t>> budget;
  budget["n1"][TierType::DRAM] = 150;  // needs 2 victims of 100 each

  auto victims = strategy.SelectVictims(candidates, budget);
  ASSERT_EQ(victims.count("n1"), 1u);
  // Oldest-first: a, then b; c (newest) is spared once the 150-byte budget met.
  ASSERT_EQ(victims["n1"].size(), 2u);
  EXPECT_EQ(victims["n1"][0], "a");
  EXPECT_EQ(victims["n1"][1], "b");
}

TEST(LruMasterEvictStrategy, HonoursPerNodeTierBudgetIndependently) {
  LruMasterEvictStrategy strategy;
  auto now = Clock::now();
  std::vector<EvictionCandidate> candidates = {
      MakeCandidate("n1-old", "n1", TierType::DRAM, 100, now - std::chrono::seconds(5)),
      MakeCandidate("n1-new", "n1", TierType::DRAM, 100, now),
      MakeCandidate("n2-old", "n2", TierType::HBM, 100, now - std::chrono::seconds(5)),
  };
  std::unordered_map<std::string, std::map<TierType, int64_t>> budget;
  budget["n1"][TierType::DRAM] = 50;  // one victim
  budget["n2"][TierType::HBM] = 50;   // one victim

  auto victims = strategy.SelectVictims(candidates, budget);
  ASSERT_EQ(victims["n1"].size(), 1u);
  EXPECT_EQ(victims["n1"][0], "n1-old");
  ASSERT_EQ(victims["n2"].size(), 1u);
  EXPECT_EQ(victims["n2"][0], "n2-old");
}

TEST(LruMasterEvictStrategy, SkipsTiersWithNoBudget) {
  LruMasterEvictStrategy strategy;
  auto now = Clock::now();
  std::vector<EvictionCandidate> candidates = {
      MakeCandidate("k", "n1", TierType::DRAM, 100, now),
  };
  std::unordered_map<std::string, std::map<TierType, int64_t>> budget;  // empty: nothing to free
  auto victims = strategy.SelectVictims(candidates, budget);
  EXPECT_TRUE(victims.empty());
}

}  // namespace
}  // namespace mori::umbp
