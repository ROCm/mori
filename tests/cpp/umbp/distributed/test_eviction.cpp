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

#include <algorithm>
#include <chrono>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/master/client_registry.h"
#include "umbp/distributed/master/global_block_index.h"

namespace mori::umbp {
namespace {

Location MakeLocation(const std::string& node_id, const std::string& location_id, uint64_t size,
                      TierType tier) {
  Location loc;
  loc.node_id = node_id;
  loc.location_id = location_id;
  loc.size = size;
  loc.tier = tier;
  return loc;
}

std::map<TierType, TierCapacity> MakeTierCapacities(TierType tier, uint64_t total_bytes,
                                                    uint64_t available_bytes) {
  return {{tier, TierCapacity{total_bytes, available_bytes}}};
}

std::map<TierType, TierCapacity> MakeMultiTierCapacities(uint64_t dram_total, uint64_t dram_avail,
                                                         uint64_t ssd_total, uint64_t ssd_avail) {
  return {
      {TierType::DRAM, TierCapacity{dram_total, dram_avail}},
      {TierType::SSD, TierCapacity{ssd_total, ssd_avail}},
  };
}

class EvictionTest : public ::testing::Test {
 protected:
  void SetUp() override { index_.SetClientRegistry(&registry_); }

  GlobalBlockIndex index_;
  ClientRegistry registry_{ClientRegistryConfig{}, index_};
};

TEST_F(EvictionTest, Basic) {
  registry_.RegisterClient("node-a", "addr-a", MakeTierCapacities(TierType::HBM, 1000, 100));

  index_.Register("node-a", "key-1", MakeLocation("node-a", "0:0", 200, TierType::HBM));
  index_.Register("node-a", "key-2", MakeLocation("node-a", "0:200", 300, TierType::HBM));
  index_.Register("node-a", "key-3", MakeLocation("node-a", "0:500", 400, TierType::HBM));

  std::set<GlobalBlockIndex::NodeTierKey> overloaded;
  overloaded.insert({"node-a", TierType::HBM});

  auto candidates = index_.FindEvictionCandidates(overloaded);
  EXPECT_GE(candidates.size(), 1u);
  EXPECT_LE(candidates.size(), 3u);

  for (const auto& c : candidates) {
    EXPECT_EQ(c.location.node_id, "node-a");
    EXPECT_EQ(c.location.tier, TierType::HBM);
  }
}

TEST_F(EvictionTest, LeasedSkipped) {
  registry_.RegisterClient("node-a", "addr-a", MakeTierCapacities(TierType::HBM, 1000, 100));

  index_.Register("node-a", "key-leased", MakeLocation("node-a", "0:0", 200, TierType::HBM));
  index_.Register("node-a", "key-free", MakeLocation("node-a", "0:200", 300, TierType::HBM));

  index_.GrantLease("key-leased", std::chrono::seconds(60));

  std::set<GlobalBlockIndex::NodeTierKey> overloaded;
  overloaded.insert({"node-a", TierType::HBM});

  auto candidates = index_.FindEvictionCandidates(overloaded);
  ASSERT_EQ(candidates.size(), 1u);
  EXPECT_EQ(candidates[0].key, "key-free");
}

TEST_F(EvictionTest, DepthTieBreaker) {
  auto now = std::chrono::steady_clock::now();

  EvictionCandidate shallow;
  shallow.key = "key-shallow";
  shallow.location = MakeLocation("node-a", "0:0", 100, TierType::HBM);
  shallow.last_accessed_at = now;
  shallow.depth = 2;
  shallow.size = 100;

  EvictionCandidate deep;
  deep.key = "key-deep";
  deep.location = MakeLocation("node-a", "0:100", 100, TierType::HBM);
  deep.last_accessed_at = now;
  deep.depth = 10;
  deep.size = 100;

  std::vector<EvictionCandidate> candidates = {shallow, deep};

  std::sort(candidates.begin(), candidates.end(),
            [](const EvictionCandidate& a, const EvictionCandidate& b) {
              if (a.last_accessed_at != b.last_accessed_at) {
                return a.last_accessed_at < b.last_accessed_at;
              }
              return a.depth > b.depth;
            });

  EXPECT_EQ(candidates[0].key, "key-deep");
  EXPECT_EQ(candidates[0].depth, 10);
  EXPECT_EQ(candidates[1].key, "key-shallow");
  EXPECT_EQ(candidates[1].depth, 2);
}

TEST_F(EvictionTest, TierAware) {
  registry_.RegisterClient("node-a", "addr-a", MakeMultiTierCapacities(1000, 50, 10000, 8000));

  index_.Register("node-a", "key-dram", MakeLocation("node-a", "0:0", 200, TierType::DRAM));
  index_.Register("node-a", "key-ssd", MakeLocation("node-a", "0:0", 500, TierType::SSD));

  std::set<GlobalBlockIndex::NodeTierKey> overloaded;
  overloaded.insert({"node-a", TierType::DRAM});

  auto candidates = index_.FindEvictionCandidates(overloaded);
  ASSERT_EQ(candidates.size(), 1u);
  EXPECT_EQ(candidates[0].key, "key-dram");
  EXPECT_EQ(candidates[0].location.tier, TierType::DRAM);
}

TEST_F(EvictionTest, EvictEntries_DoubleCheck) {
  registry_.RegisterClient("node-a", "addr-a", MakeTierCapacities(TierType::HBM, 1000, 100));

  Location loc1 = MakeLocation("node-a", "0:0", 200, TierType::HBM);
  Location loc2 = MakeLocation("node-a", "0:200", 300, TierType::HBM);
  Location loc_leased = MakeLocation("node-a", "0:500", 400, TierType::HBM);

  index_.Register("node-a", "key-1", loc1);
  index_.Register("node-a", "key-2", loc2);
  index_.Register("node-a", "key-leased", loc_leased);

  index_.GrantLease("key-leased", std::chrono::seconds(60));

  std::set<GlobalBlockIndex::NodeTierKey> overloaded;
  overloaded.insert({"node-a", TierType::HBM});
  auto candidates = index_.FindEvictionCandidates(overloaded);

  EvictionCandidate leased_victim;
  leased_victim.key = "key-leased";
  leased_victim.location = loc_leased;
  leased_victim.size = 400;
  leased_victim.depth = -1;
  leased_victim.last_accessed_at = std::chrono::steady_clock::now();
  candidates.push_back(leased_victim);

  auto evicted = index_.EvictEntries(candidates);

  bool leased_evicted = false;
  for (const auto& e : evicted) {
    EXPECT_NE(e.key, "key-leased");
    if (e.key == "key-leased") {
      leased_evicted = true;
    }
  }
  EXPECT_FALSE(leased_evicted);

  EXPECT_TRUE(index_.Lookup("key-1").empty());
  EXPECT_TRUE(index_.Lookup("key-2").empty());
  EXPECT_FALSE(index_.Lookup("key-leased").empty());
}

}  // namespace
}  // namespace mori::umbp
