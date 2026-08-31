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

#include <map>
#include <set>
#include <string>
#include <vector>

#include "umbp/distributed/routing/route_get_strategy.h"

namespace mori::umbp {
namespace {

Location MakeLoc(const std::string& node_id, const std::string& loc_id) {
  (void)loc_id;  // location_id is no longer part of Location; kept for call-site readability
  Location loc;
  loc.node_id = node_id;
  loc.size = 4096;
  loc.tier = TierType::HBM;
  return loc;
}

// ---- RandomRouteGetStrategy tests ----

TEST(RandomRouteGetStrategyTest, SingleReplicaReturnedDirectly) {
  RandomRouteGetStrategy strategy;
  std::vector<Location> locations = {MakeLoc("node-a", "loc-1")};

  auto selected = strategy.Select(locations, "requester");
  EXPECT_EQ(selected.node_id, "node-a");
  EXPECT_EQ(selected.size, 4096u);
}

TEST(RandomRouteGetStrategyTest, MultipleReplicasReturnValidOne) {
  RandomRouteGetStrategy strategy;
  std::vector<Location> locations = {
      MakeLoc("node-a", "loc-1"),
      MakeLoc("node-b", "loc-2"),
      MakeLoc("node-c", "loc-3"),
  };

  for (int i = 0; i < 100; ++i) {
    auto selected = strategy.Select(locations, "requester");
    bool found = false;
    for (const auto& loc : locations) {
      if (selected == loc) {
        found = true;
        break;
      }
    }
    ASSERT_TRUE(found) << "Selected location not in original list";
  }
}

TEST(RandomRouteGetStrategyTest, RoughlyUniformDistribution) {
  RandomRouteGetStrategy strategy;
  std::vector<Location> locations = {
      MakeLoc("node-a", "loc-1"),
      MakeLoc("node-b", "loc-2"),
      MakeLoc("node-c", "loc-3"),
  };

  constexpr int kIterations = 9000;
  std::map<std::string, int> counts;

  for (int i = 0; i < kIterations; ++i) {
    auto selected = strategy.Select(locations, "requester");
    counts[selected.node_id]++;
  }

  ASSERT_EQ(counts.size(), 3u);
  for (const auto& [node_id, count] : counts) {
    double ratio = static_cast<double>(count) / kIterations;
    EXPECT_GT(ratio, 0.2) << node_id << " selected too rarely: " << count;
    EXPECT_LT(ratio, 0.47) << node_id << " selected too often: " << count;
  }
}

// ---- Custom strategy test ----

class LocalityAwareGetStrategy : public RouteGetStrategy {
 public:
  Location Select(const std::vector<Location>& locations, const std::string& node_id) override {
    for (const auto& loc : locations) {
      if (loc.node_id == node_id) return loc;
    }
    return locations[0];
  }
};

// ---- BatchSelect tests ----

TEST(RouteGetBatchSelectTest, ReturnsOneResultPerKeyInOrder) {
  LocalPreferringRouteGetStrategy strategy;
  std::vector<std::vector<Location>> per_key = {
      {MakeLoc("node-a", "loc-1")},
      {MakeLoc("node-b", "loc-2")},
      {MakeLoc("node-c", "loc-3")},
  };

  auto out = strategy.BatchSelect(per_key, "requester");
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0].node_id, "node-a");
  EXPECT_EQ(out[1].node_id, "node-b");
  EXPECT_EQ(out[2].node_id, "node-c");
}

TEST(RouteGetBatchSelectTest, EmptyCandidateListLeftAsDefault) {
  RandomRouteGetStrategy strategy;
  std::vector<std::vector<Location>> per_key = {
      {MakeLoc("node-a", "loc-1")},
      {},  // not routed
      {MakeLoc("node-c", "loc-3")},
  };

  auto out = strategy.BatchSelect(per_key, "requester");
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0].node_id, "node-a");
  EXPECT_TRUE(out[1].node_id.empty());  // default Location => caller skips
  EXPECT_EQ(out[2].node_id, "node-c");
}

// Was TierPriorityPicksFastestTierPerKey.  Phase 4 deleted the tier order, so
// what BatchSelect applies per key is the locality rule: the requester's own
// replica wins even though it sits on the "slower" medium.
TEST(RouteGetBatchSelectTest, AppliesLocalPreferencePerKey) {
  LocalPreferringRouteGetStrategy strategy;
  Location remote_hbm = MakeLoc("node-a", "loc-1");  // HBM by default in MakeLoc
  Location local_ssd = MakeLoc("requester", "loc-2");
  local_ssd.tier = TierType::SSD;

  std::vector<std::vector<Location>> per_key = {{remote_hbm, local_ssd}};
  auto out = strategy.BatchSelect(per_key, "requester");
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].node_id, "requester");
  EXPECT_EQ(out[0].tier, TierType::SSD);
}

TEST(RouteGetBatchSelectTest, CustomStrategyUsesBaseDefaultLoop) {
  // LocalityAwareGetStrategy overrides only Select(); BatchSelect must still
  // work via the base-class default that loops Select().
  LocalityAwareGetStrategy strategy;
  std::vector<std::vector<Location>> per_key = {
      {MakeLoc("node-a", "loc-1"), MakeLoc("node-b", "loc-2")},
      {},
      {MakeLoc("node-a", "loc-3"), MakeLoc("node-c", "loc-4")},
  };

  auto out = strategy.BatchSelect(per_key, "node-c");
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0].node_id, "node-a");  // no local replica -> first
  EXPECT_TRUE(out[1].node_id.empty());  // empty -> Select not invoked
  EXPECT_EQ(out[2].node_id, "node-c");  // local replica preferred
}

// ---- LocalPreferringRouteGetStrategy tests ----

Location MakeLoc(const std::string& node_id, TierType tier) {
  Location loc;
  loc.node_id = node_id;
  loc.size = 4096;
  loc.tier = tier;
  return loc;
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

// With every medium equivalent, a non-local request can reach every replica
// rather than being confined to a preferred tier.
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

TEST(LocalPreferringRouteGetStrategyTest, EmptyReturnsDefault) {
  LocalPreferringRouteGetStrategy strategy;
  auto selected = strategy.Select({}, "requester");
  EXPECT_EQ(selected.tier, TierType::UNKNOWN);
  EXPECT_TRUE(selected.node_id.empty());
}

}  // namespace
}  // namespace mori::umbp
