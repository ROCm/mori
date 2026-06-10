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
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "umbp/distributed/routing/route_put_strategy.h"

namespace mori::umbp {
namespace {

ClientRecord MakeClient(const std::string& node_id, const std::string& addr,
                        std::map<TierType, TierCapacity> caps) {
  ClientRecord rec;
  rec.node_id = node_id;
  rec.node_address = addr;
  rec.peer_address = addr;
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

  auto result = strategy.Select(clients, 4096, /*exclude=*/{});
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

  auto result = strategy.Select(clients, 4096, /*exclude=*/{});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->tier, TierType::DRAM);
}

TEST(TierAwareMostAvailableTest, SsdIsNotADirectPutTarget) {
  // SSD capacity is reported via heartbeat but RoutePut never steers a put at
  // SSD: the SSD copy is filled asynchronously by copy-on-commit.  With HBM and
  // DRAM full, Select must return nullopt even when SSD has ample space.
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a",
                 {{TierType::HBM, {80 * GB, 0}},
                  {TierType::DRAM, {512 * GB, 0}},
                  {TierType::SSD, {4096 * GB, 3000 * GB}}}),
  };

  auto result = strategy.Select(clients, 4096, /*exclude=*/{});
  EXPECT_FALSE(result.has_value());
}

TEST(TierAwareMostAvailableTest, ReturnsNulloptWhenAllFull) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a",
                 {{TierType::HBM, {80 * GB, 0}},
                  {TierType::DRAM, {512 * GB, 0}},
                  {TierType::SSD, {4096 * GB, 0}}}),
  };

  auto result = strategy.Select(clients, 4096, /*exclude=*/{});
  EXPECT_FALSE(result.has_value());
}

TEST(TierAwareMostAvailableTest, ReturnsNulloptWhenBlockTooLarge) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 10 * GB}}}),
  };

  auto result = strategy.Select(clients, 100 * GB, /*exclude=*/{});
  EXPECT_FALSE(result.has_value());
}

TEST(TierAwareMostAvailableTest, PicksMostAvailableOnSameTier) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 10 * GB}}}),
      MakeClient("node-b", "addr-b", {{TierType::HBM, {80 * GB, 50 * GB}}}),
      MakeClient("node-c", "addr-c", {{TierType::HBM, {80 * GB, 30 * GB}}}),
  };

  auto result = strategy.Select(clients, 4096, /*exclude=*/{});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->node_id, "node-b");
  EXPECT_EQ(result->peer_address, "addr-b");
  EXPECT_EQ(result->tier, TierType::HBM);
}

TEST(TierAwareMostAvailableTest, HBMPreferredEvenIfDRAMHasMoreSpace) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a",
                 {{TierType::HBM, {80 * GB, 5 * GB}}, {TierType::DRAM, {512 * GB, 400 * GB}}}),
  };

  auto result = strategy.Select(clients, 4096, /*exclude=*/{});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->tier, TierType::HBM);
}

TEST(TierAwareMostAvailableTest, EmptyClientListReturnsNullopt) {
  TierAwareMostAvailableStrategy strategy;
  std::vector<ClientRecord> empty;

  auto result = strategy.Select(empty, 4096, /*exclude=*/{});
  EXPECT_FALSE(result.has_value());
}

TEST(TierAwareMostAvailableTest, ClientWithNoTierCapacitiesSkipped) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {}),
  };

  auto result = strategy.Select(clients, 4096, /*exclude=*/{});
  EXPECT_FALSE(result.has_value());
}

// ---- SelectBatch projected-capacity tests ----

// Two 6GB blocks against node-a (10GB) and node-b (8GB) on the same tier.
// Without projected capacity both pick node-a (most available in the
// snapshot).  With per-batch deduction, block 1 lands on node-a (10GB -> 4GB),
// and block 2 is forced to node-b because node-a no longer fits 6GB.
TEST(SelectBatchTest, ProjectedCapacitySpreadsAcrossNodes) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 10 * GB}}}),
      MakeClient("node-b", "addr-b", {{TierType::HBM, {80 * GB, 8 * GB}}}),
  };

  auto results = strategy.SelectBatch({6 * GB, 6 * GB}, {false, false}, clients, /*exclude=*/{});
  ASSERT_EQ(results.size(), 2u);

  ASSERT_TRUE(results[0].has_value());
  EXPECT_EQ(results[0]->outcome, RoutePutOutcome::kRouted);
  EXPECT_EQ(results[0]->node_id, "node-a");

  ASSERT_TRUE(results[1].has_value());
  EXPECT_EQ(results[1]->outcome, RoutePutOutcome::kRouted);
  EXPECT_EQ(results[1]->node_id, "node-b");
}

// A dedup hit returns kAlreadyExists and consumes no projected capacity:
// node-a fits exactly one 6GB block, block 0 is a dedup hit, so block 1 must
// still route to node-a.
TEST(SelectBatchTest, DedupHitConsumesNoCapacity) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 6 * GB}}}),
  };

  auto results = strategy.SelectBatch({6 * GB, 6 * GB}, {true, false}, clients, /*exclude=*/{});
  ASSERT_EQ(results.size(), 2u);

  ASSERT_TRUE(results[0].has_value());
  EXPECT_EQ(results[0]->outcome, RoutePutOutcome::kAlreadyExists);

  ASSERT_TRUE(results[1].has_value());
  EXPECT_EQ(results[1]->outcome, RoutePutOutcome::kRouted);
  EXPECT_EQ(results[1]->node_id, "node-a");
}

TEST(SelectBatchTest, ThrowsOnAlreadyExistsLengthMismatch) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 10 * GB}}}),
  };

  EXPECT_THROW(strategy.SelectBatch({4096, 4096}, {false}, clients, /*exclude=*/{}),
               std::runtime_error);
}

// The by-value candidates copy is never mutated for the caller: passing the
// same snapshot again yields the same placement (no projected state leaks out).
TEST(SelectBatchTest, DoesNotMutateCallerCandidates) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 10 * GB}}}),
      MakeClient("node-b", "addr-b", {{TierType::HBM, {80 * GB, 8 * GB}}}),
  };

  strategy.SelectBatch({6 * GB, 6 * GB}, {false, false}, clients, /*exclude=*/{});

  EXPECT_EQ(clients[0].tier_capacities.at(TierType::HBM).available_bytes, 10 * GB);
  EXPECT_EQ(clients[1].tier_capacities.at(TierType::HBM).available_bytes, 8 * GB);
}

// SelectBatch with one request must match a direct Select() call: the default
// batch path does not alter single-key placement semantics.
TEST(SelectBatchTest, SizeOneMatchesSelect) {
  TierAwareMostAvailableStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 10 * GB}}}),
      MakeClient("node-b", "addr-b", {{TierType::HBM, {80 * GB, 50 * GB}}}),
  };

  auto single = strategy.Select(clients, 4096, /*exclude=*/{});
  auto batch = strategy.SelectBatch({4096}, {false}, clients, /*exclude=*/{});

  ASSERT_EQ(batch.size(), 1u);
  ASSERT_TRUE(single.has_value());
  ASSERT_TRUE(batch[0].has_value());
  EXPECT_EQ(batch[0]->node_id, single->node_id);
  EXPECT_EQ(batch[0]->tier, single->tier);
}

// A strategy whose Select() breaks its own contract (routes to a node/tier
// without enough room) must trip the projected-capacity invariant and throw,
// never silently clamp the deduction.
class BrokenContractStrategy : public RoutePutStrategy {
 public:
  std::optional<RoutePutResult> Select(
      const std::vector<ClientRecord>& alive_clients, uint64_t /*block_size*/,
      const std::unordered_set<std::string>& /*exclude_nodes*/) override {
    return RoutePutResult{
        .outcome = RoutePutOutcome::kRouted,
        .node_id = alive_clients.front().node_id,
        .peer_address = alive_clients.front().peer_address,
        .tier = TierType::HBM,
    };
  }
};

TEST(SelectBatchTest, ThrowsOnProjectedCapacityUnderflow) {
  BrokenContractStrategy strategy;

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 1 * GB}}}),
  };

  EXPECT_THROW(strategy.SelectBatch({4 * GB}, {false}, clients, /*exclude=*/{}),
               std::runtime_error);
}

}  // namespace
}  // namespace mori::umbp
