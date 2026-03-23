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

#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include "umbp/distributed/routing/route_put_strategy.h"

namespace mori::umbp {
namespace {

ClientRecord MakeClient(const std::string& node_id, const std::string& addr,
                        std::map<TierType, TierCapacity> caps) {
  ClientRecord rec;
  rec.node_id = node_id;
  rec.node_address = addr;
  rec.status = ClientStatus::ALIVE;
  rec.last_heartbeat = std::chrono::steady_clock::now();
  rec.registered_at = std::chrono::steady_clock::now();
  rec.tier_capacities = std::move(caps);
  return rec;
}

constexpr uint64_t GB = 1024ULL * 1024 * 1024;

// ---- TierAwareMostAvailableStrategy tests ----

TEST(TierAwareMostAvailableTest, PrefersHBMOverDRAM) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a",
                 {{TierType::HBM, {80 * GB, 10 * GB}}, {TierType::DRAM, {512 * GB, 400 * GB}}}),
  };

  auto result = strategy.Select(clients, 4096);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->node_id, "node-a");
  EXPECT_EQ(result->tier, TierType::HBM);
}

TEST(TierAwareMostAvailableTest, FallsThroughToDRAMWhenHBMFull) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a",
                 {{TierType::HBM, {80 * GB, 0}}, {TierType::DRAM, {512 * GB, 200 * GB}}}),
  };

  auto result = strategy.Select(clients, 4096);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->tier, TierType::DRAM);
}

TEST(TierAwareMostAvailableTest, FallsThroughToSSDWhenHBMAndDRAMFull) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a",
                 {{TierType::HBM, {80 * GB, 0}},
                  {TierType::DRAM, {512 * GB, 0}},
                  {TierType::SSD, {4096 * GB, 3000 * GB}}}),
  };

  auto result = strategy.Select(clients, 4096);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->tier, TierType::SSD);
}

TEST(TierAwareMostAvailableTest, ReturnsNulloptWhenAllFull) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a",
                 {{TierType::HBM, {80 * GB, 0}},
                  {TierType::DRAM, {512 * GB, 0}},
                  {TierType::SSD, {4096 * GB, 0}}}),
  };

  auto result = strategy.Select(clients, 4096);
  EXPECT_FALSE(result.has_value());
}

TEST(TierAwareMostAvailableTest, ReturnsNulloptWhenBlockTooLarge) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 10 * GB}}}),
  };

  auto result = strategy.Select(clients, 100 * GB);
  EXPECT_FALSE(result.has_value());
}

TEST(TierAwareMostAvailableTest, PicksMostAvailableOnSameTier) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 10 * GB}}}),
      MakeClient("node-b", "addr-b", {{TierType::HBM, {80 * GB, 50 * GB}}}),
      MakeClient("node-c", "addr-c", {{TierType::HBM, {80 * GB, 30 * GB}}}),
  };

  auto result = strategy.Select(clients, 4096);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->node_id, "node-b");
  EXPECT_EQ(result->node_address, "addr-b");
  EXPECT_EQ(result->tier, TierType::HBM);
}

TEST(TierAwareMostAvailableTest, HBMPreferredEvenIfDRAMHasMoreSpace) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a",
                 {{TierType::HBM, {80 * GB, 5 * GB}}, {TierType::DRAM, {512 * GB, 400 * GB}}}),
  };

  auto result = strategy.Select(clients, 4096);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->tier, TierType::HBM);
}

TEST(TierAwareMostAvailableTest, EmptyClientListReturnsNullopt) {
  TierAwareMostAvailableStrategy strategy;
  std::vector<ClientRecord> empty;

  auto result = strategy.Select(empty, 4096);
  EXPECT_FALSE(result.has_value());
}

TEST(TierAwareMostAvailableTest, ClientWithNoTierCapacitiesSkipped) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {}),
  };

  auto result = strategy.Select(clients, 4096);
  EXPECT_FALSE(result.has_value());
}

}  // namespace
}  // namespace mori::umbp
