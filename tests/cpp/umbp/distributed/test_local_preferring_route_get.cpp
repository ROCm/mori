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

// LocalPreferringRouteGetStrategy (backend-agnostic refactor Phase 4).
//
// Replaces test_tier_priority_route_get.cpp.  That file asserted the
// HBM > DRAM > SSD read order, which Phase 4 DELETED rather than re-expressed
// as an advertised backend property — every medium is currently equivalent, so
// there is nothing left to rank.  The tests below assert what remains, and the
// two that invert an old expectation are called out where they appear.

#include <gtest/gtest.h>

#include <set>
#include <string>
#include <vector>

#include "umbp/distributed/routing/route_get_strategy.h"

namespace mori::umbp {
namespace {

Location MakeLoc(const std::string& node_id, TierType tier) {
  Location loc;
  loc.node_id = node_id;
  loc.size = 4096;
  loc.tier = tier;
  return loc;
}

// INVERTS test_tier_priority_route_get's RequesterLocalLowerTierDoesNotBeatBestTier.
// With no tier ranking, a local replica is simply the cheapest read there is —
// no RDMA — so it wins whatever medium it happens to sit on.
TEST(LocalPreferringRouteGetStrategyTest, LocalReplicaWinsWhateverItsTier) {
  LocalPreferringRouteGetStrategy strategy;
  std::vector<Location> locations = {
      MakeLoc("dram-a", TierType::DRAM),
      MakeLoc("requester", TierType::SSD),
  };

  for (int i = 0; i < 100; ++i) {
    auto selected = strategy.Select(locations, "requester");
    EXPECT_EQ(selected.node_id, "requester");
    EXPECT_EQ(selected.tier, TierType::SSD);
  }
}

TEST(LocalPreferringRouteGetStrategyTest, LocalReplicaBeatsRemotePeersOnTheSameTier) {
  LocalPreferringRouteGetStrategy strategy;
  std::vector<Location> locations = {
      MakeLoc("dram-a", TierType::DRAM),
      MakeLoc("requester", TierType::DRAM),
      MakeLoc("dram-c", TierType::DRAM),
  };

  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(strategy.Select(locations, "requester").node_id, "requester");
  }
}

// INVERTS RandomWithinBestTierOnly.  Selection no longer confines itself to the
// "best" tier: with every medium equivalent, every replica is reachable.
TEST(LocalPreferringRouteGetStrategyTest, NonLocalRequesterSpreadsAcrossEveryTier) {
  LocalPreferringRouteGetStrategy strategy;
  std::vector<Location> locations = {
      MakeLoc("dram-a", TierType::DRAM),
      MakeLoc("dram-b", TierType::DRAM),
      MakeLoc("hbm-x", TierType::HBM),
      MakeLoc("ssd-y", TierType::SSD),
  };

  std::set<std::string> seen;
  for (int i = 0; i < 2000; ++i) {
    seen.insert(strategy.Select(locations, "requester").node_id);
  }
  EXPECT_EQ(seen.size(), 4u) << "every replica should be reachable, regardless of tier";
}

TEST(LocalPreferringRouteGetStrategyTest, EmptyRequesterFallsBackToRandom) {
  LocalPreferringRouteGetStrategy strategy;
  std::vector<Location> locations = {
      MakeLoc("node-a", TierType::DRAM),
      MakeLoc("node-b", TierType::DRAM),
  };

  std::set<std::string> seen;
  for (int i = 0; i < 500; ++i) seen.insert(strategy.Select(locations, "").node_id);
  EXPECT_EQ(seen.size(), 2u);
}

TEST(LocalPreferringRouteGetStrategyTest, SingleCandidateIsReturned) {
  LocalPreferringRouteGetStrategy strategy;
  std::vector<Location> locations = {MakeLoc("only", TierType::SSD)};
  auto selected = strategy.Select(locations, "requester");
  EXPECT_EQ(selected.node_id, "only");
  EXPECT_EQ(selected.tier, TierType::SSD);
}

TEST(LocalPreferringRouteGetStrategyTest, EmptyReturnsDefault) {
  LocalPreferringRouteGetStrategy strategy;
  std::vector<Location> locations;
  auto selected = strategy.Select(locations, "requester");
  EXPECT_EQ(selected.tier, TierType::UNKNOWN);
  EXPECT_TRUE(selected.node_id.empty());
}

TEST(LocalPreferringRouteGetStrategyTest, BatchSelectAppliesPerKeyAndLeavesEmptyKeysDefault) {
  LocalPreferringRouteGetStrategy strategy;
  std::vector<std::vector<Location>> per_key = {
      {MakeLoc("dram-a", TierType::DRAM), MakeLoc("requester", TierType::SSD)},
      {},  // not routed
      {MakeLoc("only", TierType::HBM)},
  };

  auto out = strategy.BatchSelect(per_key, "requester");
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0].node_id, "requester");  // local wins, tier irrelevant
  EXPECT_TRUE(out[1].node_id.empty());     // left default
  EXPECT_EQ(out[2].node_id, "only");
}

}  // namespace
}  // namespace mori::umbp
