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
#include <memory>
#include <string>
#include <vector>

#include "umbp/router.h"

namespace mori::umbp {
namespace {

constexpr uint64_t GB = 1024ULL * 1024 * 1024;

Location MakeLoc(const std::string& node_id, const std::string& loc_id, uint64_t size = 4096,
                 TierType tier = TierType::HBM) {
  Location loc;
  loc.node_id = node_id;
  loc.location_id = loc_id;
  loc.size = size;
  loc.tier = tier;
  return loc;
}

std::map<TierType, TierCapacity> MakeCaps(uint64_t hbm_total, uint64_t hbm_avail,
                                          uint64_t dram_total = 0, uint64_t dram_avail = 0) {
  std::map<TierType, TierCapacity> caps;
  caps[TierType::HBM] = {hbm_total, hbm_avail};
  if (dram_total > 0) {
    caps[TierType::DRAM] = {dram_total, dram_avail};
  }
  return caps;
}

// ---- Fixture that sets up BlockIndex + ClientRegistry + Router ----

class RouterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    index_.SetClientRegistry(&registry_);
    router_ = std::make_unique<Router>(index_, registry_);
  }

  BlockIndex index_;
  ClientRegistry registry_{ClientRegistryConfig{}, index_};
  std::unique_ptr<Router> router_;
};

// ---- RouteGet tests ----

TEST_F(RouterTest, RouteGetUnknownKeyReturnsNullopt) {
  auto result = router_->RouteGet("nonexistent-key", "requester");
  EXPECT_FALSE(result.has_value());
}

TEST_F(RouterTest, RouteGetReturnsSingleReplica) {
  index_.Register("node-a", "key-1", MakeLoc("node-a", "loc-1"));

  auto result = router_->RouteGet("key-1", "requester");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->node_id, "node-a");
  EXPECT_EQ(result->location_id, "loc-1");
}

TEST_F(RouterTest, RouteGetReturnsValidReplicaFromMultiple) {
  index_.Register("node-a", "key-1", MakeLoc("node-a", "loc-1"));
  index_.Register("node-b", "key-1", MakeLoc("node-b", "loc-2"));
  index_.Register("node-c", "key-1", MakeLoc("node-c", "loc-3"));

  for (int i = 0; i < 50; ++i) {
    auto result = router_->RouteGet("key-1", "requester");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->node_id == "node-a" || result->node_id == "node-b" ||
                result->node_id == "node-c");
  }
}

TEST_F(RouterTest, RouteGetBumpsAccessMetrics) {
  index_.Register("node-a", "key-1", MakeLoc("node-a", "loc-1"));

  auto before = index_.GetMetrics("key-1");
  ASSERT_TRUE(before.has_value());
  uint64_t count_before = before->access_count;

  router_->RouteGet("key-1", "requester");
  router_->RouteGet("key-1", "requester");

  auto after = index_.GetMetrics("key-1");
  ASSERT_TRUE(after.has_value());
  EXPECT_EQ(after->access_count, count_before + 2);
}

// ---- RoutePut tests ----

TEST_F(RouterTest, RoutePutNoAliveClientsReturnsNullopt) {
  auto result = router_->RoutePut("key-1", "requester", 4096);
  EXPECT_FALSE(result.has_value());
}

TEST_F(RouterTest, RoutePutSelectsAliveNodeWithCapacity) {
  registry_.RegisterClient("node-a", "addr-a", MakeCaps(80 * GB, 40 * GB));

  auto result = router_->RoutePut("key-1", "requester", 4096);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->node_id, "node-a");
  EXPECT_EQ(result->node_address, "addr-a");
  EXPECT_EQ(result->tier, TierType::HBM);
}

TEST_F(RouterTest, RoutePutPicksMostAvailableNode) {
  registry_.RegisterClient("node-a", "addr-a", MakeCaps(80 * GB, 10 * GB));
  registry_.RegisterClient("node-b", "addr-b", MakeCaps(80 * GB, 60 * GB));

  auto result = router_->RoutePut("key-1", "requester", 4096);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->node_id, "node-b");
}

TEST_F(RouterTest, RoutePutFallsThroughTiers) {
  registry_.RegisterClient("node-a", "addr-a",
                           MakeCaps(80 * GB, 0, 512 * GB, 200 * GB));

  auto result = router_->RoutePut("key-1", "requester", 4096);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->tier, TierType::DRAM);
}

TEST_F(RouterTest, RoutePutReturnsNulloptWhenAllFull) {
  registry_.RegisterClient("node-a", "addr-a", MakeCaps(80 * GB, 0));

  auto result = router_->RoutePut("key-1", "requester", 4096);
  EXPECT_FALSE(result.has_value());
}

// ---- Custom strategy injection ----

class AlwaysFirstGetStrategy : public RouteGetStrategy {
 public:
  Location Select(const std::vector<Location>& locations,
                  const std::string& /*node_id*/) override {
    return locations[0];
  }
};

class AlwaysNulloptPutStrategy : public RoutePutStrategy {
 public:
  std::optional<RoutePutResult> Select(const std::vector<ClientRecord>& /*alive_clients*/,
                                       uint64_t /*block_size*/) override {
    return std::nullopt;
  }
};

TEST(RouterCustomStrategyTest, UsesInjectedGetStrategy) {
  BlockIndex index;
  ClientRegistry registry(ClientRegistryConfig{}, index);
  index.SetClientRegistry(&registry);

  auto custom_get = std::make_unique<AlwaysFirstGetStrategy>();
  Router router(index, registry, std::move(custom_get), nullptr);

  index.Register("node-a", "key-1", MakeLoc("node-a", "loc-1"));
  index.Register("node-b", "key-1", MakeLoc("node-b", "loc-2"));

  for (int i = 0; i < 20; ++i) {
    auto result = router.RouteGet("key-1", "requester");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->node_id, "node-a");
  }
}

TEST(RouterCustomStrategyTest, UsesInjectedPutStrategy) {
  BlockIndex index;
  ClientRegistry registry(ClientRegistryConfig{}, index);
  index.SetClientRegistry(&registry);

  auto custom_put = std::make_unique<AlwaysNulloptPutStrategy>();
  Router router(index, registry, nullptr, std::move(custom_put));

  registry.RegisterClient("node-a", "addr-a", MakeCaps(80 * GB, 40 * GB));

  auto result = router.RoutePut("key-1", "requester", 4096);
  EXPECT_FALSE(result.has_value());
}

// ---- Default strategy creation ----

TEST(RouterDefaultStrategyTest, NullStrategiesCreateDefaults) {
  BlockIndex index;
  ClientRegistry registry(ClientRegistryConfig{}, index);
  index.SetClientRegistry(&registry);

  Router router(index, registry, nullptr, nullptr);

  index.Register("node-a", "key-1", MakeLoc("node-a", "loc-1"));
  auto get_result = router.RouteGet("key-1", "requester");
  EXPECT_TRUE(get_result.has_value());

  registry.RegisterClient("node-a", "addr-a", MakeCaps(80 * GB, 40 * GB));
  auto put_result = router.RoutePut("key-2", "requester", 4096);
  EXPECT_TRUE(put_result.has_value());
}

}  // namespace
}  // namespace mori::umbp
