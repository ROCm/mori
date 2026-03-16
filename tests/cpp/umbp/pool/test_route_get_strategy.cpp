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
#include <string>
#include <vector>

#include "umbp/route_get_strategy.h"

namespace mori::umbp {
namespace {

Location MakeLoc(const std::string& node_id, const std::string& loc_id) {
  Location loc;
  loc.node_id = node_id;
  loc.location_id = loc_id;
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
  EXPECT_EQ(selected.location_id, "loc-1");
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

TEST(CustomRouteGetStrategyTest, PrefersLocalReplica) {
  LocalityAwareGetStrategy strategy;
  std::vector<Location> locations = {
      MakeLoc("node-a", "loc-1"),
      MakeLoc("node-b", "loc-2"),
      MakeLoc("node-c", "loc-3"),
  };

  auto selected = strategy.Select(locations, "node-b");
  EXPECT_EQ(selected.node_id, "node-b");
}

TEST(CustomRouteGetStrategyTest, FallsBackToFirstWhenNoLocalReplica) {
  LocalityAwareGetStrategy strategy;
  std::vector<Location> locations = {
      MakeLoc("node-a", "loc-1"),
      MakeLoc("node-b", "loc-2"),
  };

  auto selected = strategy.Select(locations, "node-x");
  EXPECT_EQ(selected.node_id, "node-a");
}

}  // namespace
}  // namespace mori::umbp
