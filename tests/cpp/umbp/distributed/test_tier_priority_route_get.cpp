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

// With a DRAM (or HBM) replica present alongside an SSD one, the strategy must
// never route to the slow SSD tier.
TEST(TierPriorityRouteGetStrategyTest, PrefersDramOverSsd) {
  TierPriorityRouteGetStrategy strategy;
  std::vector<Location> locations = {
      MakeLoc("ssd-node", TierType::SSD),
      MakeLoc("dram-node", TierType::DRAM),
  };
  for (int i = 0; i < 100; ++i) {
    auto selected = strategy.Select(locations, "requester");
    EXPECT_EQ(selected.tier, TierType::DRAM);
    EXPECT_EQ(selected.node_id, "dram-node");
  }
}

// HBM beats both DRAM and SSD.
TEST(TierPriorityRouteGetStrategyTest, PrefersHbmOverDramAndSsd) {
  TierPriorityRouteGetStrategy strategy;
  std::vector<Location> locations = {
      MakeLoc("ssd-node", TierType::SSD),
      MakeLoc("dram-node", TierType::DRAM),
      MakeLoc("hbm-node", TierType::HBM),
  };
  for (int i = 0; i < 100; ++i) {
    auto selected = strategy.Select(locations, "requester");
    EXPECT_EQ(selected.tier, TierType::HBM);
  }
}

// When SSD is the only tier present it is selected (read-from-SSD is valid).
TEST(TierPriorityRouteGetStrategyTest, FallsBackToSsdWhenOnlyTier) {
  TierPriorityRouteGetStrategy strategy;
  std::vector<Location> locations = {
      MakeLoc("ssd-a", TierType::SSD),
      MakeLoc("ssd-b", TierType::SSD),
  };
  for (int i = 0; i < 50; ++i) {
    auto selected = strategy.Select(locations, "requester");
    EXPECT_EQ(selected.tier, TierType::SSD);
  }
}

// Within the winning tier, selection spreads across all replicas on that tier
// and never leaks to a lower tier.
TEST(TierPriorityRouteGetStrategyTest, RandomWithinBestTierOnly) {
  TierPriorityRouteGetStrategy strategy;
  std::vector<Location> locations = {
      MakeLoc("dram-a", TierType::DRAM),
      MakeLoc("dram-b", TierType::DRAM),
      MakeLoc("dram-c", TierType::DRAM),
      MakeLoc("ssd-x", TierType::SSD),
  };
  std::set<std::string> seen;
  for (int i = 0; i < 2000; ++i) {
    auto selected = strategy.Select(locations, "requester");
    ASSERT_EQ(selected.tier, TierType::DRAM) << "must never pick the SSD replica";
    seen.insert(selected.node_id);
  }
  EXPECT_EQ(seen.size(), 3u) << "all three DRAM replicas should be reachable";
  EXPECT_EQ(seen.count("ssd-x"), 0u);
}

TEST(TierPriorityRouteGetStrategyTest, PrefersRequesterLocalReplicaWithinBestTier) {
  TierPriorityRouteGetStrategy strategy;
  std::vector<Location> locations = {
      MakeLoc("dram-a", TierType::DRAM),
      MakeLoc("requester", TierType::DRAM),
      MakeLoc("dram-c", TierType::DRAM),
      MakeLoc("ssd-x", TierType::SSD),
  };

  for (int i = 0; i < 100; ++i) {
    auto selected = strategy.Select(locations, "requester");
    EXPECT_EQ(selected.tier, TierType::DRAM);
    EXPECT_EQ(selected.node_id, "requester");
  }
}

TEST(TierPriorityRouteGetStrategyTest, RequesterLocalLowerTierDoesNotBeatBestTier) {
  TierPriorityRouteGetStrategy strategy;
  std::vector<Location> locations = {
      MakeLoc("dram-a", TierType::DRAM),
      MakeLoc("requester", TierType::SSD),
  };

  for (int i = 0; i < 100; ++i) {
    auto selected = strategy.Select(locations, "requester");
    EXPECT_EQ(selected.tier, TierType::DRAM);
    EXPECT_EQ(selected.node_id, "dram-a");
  }
}

TEST(TierPriorityRouteGetStrategyTest, EmptyReturnsDefault) {
  TierPriorityRouteGetStrategy strategy;
  std::vector<Location> locations;
  auto selected = strategy.Select(locations, "requester");
  EXPECT_EQ(selected.tier, TierType::UNKNOWN);
  EXPECT_TRUE(selected.node_id.empty());
}

}  // namespace
}  // namespace mori::umbp
