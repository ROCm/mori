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
#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "umbp/common/env_time.h"
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

using Algo = ConfigurableRoutePutStrategy::SelectAlgo;
using Affinity = ConfigurableRoutePutStrategy::NodeAffinity;

// Single-key convenience over the batch interface (the Router routes single-key
// RoutePut as a size-1 batch).  Returns the per-key result, or nullopt when the
// key is unroutable.
std::optional<RoutePutResult> SelectOne(ConfigurableRoutePutStrategy& strat,
                                        const std::vector<ClientRecord>& clients,
                                        uint64_t block_size,
                                        const std::unordered_set<std::string>& exclude) {
  auto out = strat.SelectBatch(/*requester=*/"req", {block_size}, {false}, clients, exclude);
  if (out.empty()) return std::nullopt;
  return out.front();
}

// ---- most_available / none: single-key placement (migrated from the old
//      TierAwareMostAvailableStrategy::Select coverage) ----

TEST(MostAvailableNoneTest, PrefersHBMOverDRAM) {
  ConfigurableRoutePutStrategy strategy(Algo::kMostAvailable, Affinity::kNone);

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a",
                 {{TierType::HBM, {80 * GB, 10 * GB}}, {TierType::DRAM, {512 * GB, 400 * GB}}}),
  };

  auto result = SelectOne(strategy, clients, 4096, /*exclude=*/{});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->node_id, "node-a");
  EXPECT_EQ(result->tier, TierType::HBM);
}

TEST(MostAvailableNoneTest, FallsThroughToDRAMWhenHBMFull) {
  ConfigurableRoutePutStrategy strategy(Algo::kMostAvailable, Affinity::kNone);

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a",
                 {{TierType::HBM, {80 * GB, 0}}, {TierType::DRAM, {512 * GB, 200 * GB}}}),
  };

  auto result = SelectOne(strategy, clients, 4096, /*exclude=*/{});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->tier, TierType::DRAM);
}

TEST(MostAvailableNoneTest, SsdIsNotADirectPutTarget) {
  // SSD capacity is reported via heartbeat but RoutePut never steers a put at
  // SSD: the SSD copy is filled asynchronously by copy-on-commit.  With HBM and
  // DRAM full, placement must return nullopt even when SSD has ample space.
  ConfigurableRoutePutStrategy strategy(Algo::kMostAvailable, Affinity::kNone);

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a",
                 {{TierType::HBM, {80 * GB, 0}},
                  {TierType::DRAM, {512 * GB, 0}},
                  {TierType::SSD, {4096 * GB, 3000 * GB}}}),
  };

  auto result = SelectOne(strategy, clients, 4096, /*exclude=*/{});
  EXPECT_FALSE(result.has_value());
}

TEST(MostAvailableNoneTest, ReturnsNulloptWhenAllFull) {
  ConfigurableRoutePutStrategy strategy(Algo::kMostAvailable, Affinity::kNone);

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a",
                 {{TierType::HBM, {80 * GB, 0}},
                  {TierType::DRAM, {512 * GB, 0}},
                  {TierType::SSD, {4096 * GB, 0}}}),
  };

  auto result = SelectOne(strategy, clients, 4096, /*exclude=*/{});
  EXPECT_FALSE(result.has_value());
}

TEST(MostAvailableNoneTest, ReturnsNulloptWhenBlockTooLarge) {
  ConfigurableRoutePutStrategy strategy(Algo::kMostAvailable, Affinity::kNone);

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 10 * GB}}}),
  };

  auto result = SelectOne(strategy, clients, 100 * GB, /*exclude=*/{});
  EXPECT_FALSE(result.has_value());
}

TEST(MostAvailableNoneTest, PicksMostAvailableOnSameTier) {
  ConfigurableRoutePutStrategy strategy(Algo::kMostAvailable, Affinity::kNone);

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 10 * GB}}}),
      MakeClient("node-b", "addr-b", {{TierType::HBM, {80 * GB, 50 * GB}}}),
      MakeClient("node-c", "addr-c", {{TierType::HBM, {80 * GB, 30 * GB}}}),
  };

  auto result = SelectOne(strategy, clients, 4096, /*exclude=*/{});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->node_id, "node-b");
  EXPECT_EQ(result->peer_address, "addr-b");
  EXPECT_EQ(result->tier, TierType::HBM);
}

TEST(MostAvailableNoneTest, HBMPreferredEvenIfDRAMHasMoreSpace) {
  ConfigurableRoutePutStrategy strategy(Algo::kMostAvailable, Affinity::kNone);

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a",
                 {{TierType::HBM, {80 * GB, 5 * GB}}, {TierType::DRAM, {512 * GB, 400 * GB}}}),
  };

  auto result = SelectOne(strategy, clients, 4096, /*exclude=*/{});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->tier, TierType::HBM);
}

TEST(MostAvailableNoneTest, EmptyClientListReturnsNullopt) {
  ConfigurableRoutePutStrategy strategy(Algo::kMostAvailable, Affinity::kNone);
  std::vector<ClientRecord> empty;

  auto result = SelectOne(strategy, empty, 4096, /*exclude=*/{});
  EXPECT_FALSE(result.has_value());
}

TEST(MostAvailableNoneTest, ClientWithNoTierCapacitiesSkipped) {
  ConfigurableRoutePutStrategy strategy(Algo::kMostAvailable, Affinity::kNone);

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {}),
  };

  auto result = SelectOne(strategy, clients, 4096, /*exclude=*/{});
  EXPECT_FALSE(result.has_value());
}

TEST(MostAvailableNoneTest, RespectsExcludeNodes) {
  ConfigurableRoutePutStrategy strategy(Algo::kMostAvailable, Affinity::kNone);

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 50 * GB}}}),
      MakeClient("node-b", "addr-b", {{TierType::HBM, {80 * GB, 10 * GB}}}),
  };

  // node-a is the most-available pick, but excluded: must fall to node-b.
  auto result = SelectOne(strategy, clients, 4096, /*exclude=*/{"node-a"});
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->node_id, "node-b");
}

// ---- SelectBatch projected-capacity tests (most_available / none) ----

// Two 6GB blocks against node-a (10GB) and node-b (8GB) on the same tier.
// Without projected capacity both pick node-a (most available in the
// snapshot).  With per-batch deduction, block 1 lands on node-a (10GB -> 4GB),
// and block 2 is forced to node-b because node-a no longer fits 6GB.
TEST(SelectBatchTest, ProjectedCapacitySpreadsAcrossNodes) {
  ConfigurableRoutePutStrategy strategy(Algo::kMostAvailable, Affinity::kNone);

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 10 * GB}}}),
      MakeClient("node-b", "addr-b", {{TierType::HBM, {80 * GB, 8 * GB}}}),
  };

  auto results =
      strategy.SelectBatch(/*requester=*/"req", {6 * GB, 6 * GB}, {false, false}, clients,
                           /*exclude=*/{});
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
  ConfigurableRoutePutStrategy strategy(Algo::kMostAvailable, Affinity::kNone);

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 6 * GB}}}),
  };

  auto results = strategy.SelectBatch(/*requester=*/"req", {6 * GB, 6 * GB}, {true, false}, clients,
                                      /*exclude=*/{});
  ASSERT_EQ(results.size(), 2u);

  ASSERT_TRUE(results[0].has_value());
  EXPECT_EQ(results[0]->outcome, RoutePutOutcome::kAlreadyExists);

  ASSERT_TRUE(results[1].has_value());
  EXPECT_EQ(results[1]->outcome, RoutePutOutcome::kRouted);
  EXPECT_EQ(results[1]->node_id, "node-a");
}

// Best-effort: an already_exists/block_sizes length mismatch is not fatal — it
// logs a MORI ERROR and returns an all-nullopt result sized to block_sizes.
TEST(SelectBatchTest, AlreadyExistsLengthMismatchYieldsAllNullopt) {
  ConfigurableRoutePutStrategy strategy(Algo::kMostAvailable, Affinity::kNone);

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 10 * GB}}}),
  };

  auto results = strategy.SelectBatch(/*requester=*/"req", {4096, 4096}, {false}, clients,
                                      /*exclude=*/{});
  ASSERT_EQ(results.size(), 2u);
  EXPECT_FALSE(results[0].has_value());
  EXPECT_FALSE(results[1].has_value());
}

// The by-value candidates copy is never mutated for the caller: passing the
// same snapshot again yields the same placement (no projected state leaks out).
TEST(SelectBatchTest, DoesNotMutateCallerCandidates) {
  ConfigurableRoutePutStrategy strategy(Algo::kMostAvailable, Affinity::kNone);

  std::vector<ClientRecord> clients = {
      MakeClient("node-a", "addr-a", {{TierType::HBM, {80 * GB, 10 * GB}}}),
      MakeClient("node-b", "addr-b", {{TierType::HBM, {80 * GB, 8 * GB}}}),
  };

  strategy.SelectBatch(/*requester=*/"req", {6 * GB, 6 * GB}, {false, false}, clients,
                       /*exclude=*/{});

  EXPECT_EQ(clients[0].tier_capacities.at(TierType::HBM).available_bytes, 10 * GB);
  EXPECT_EQ(clients[1].tier_capacities.at(TierType::HBM).available_bytes, 8 * GB);
}

// ---- ConfigurableRoutePutStrategy tests ----
//
// Deterministic, master-free: build capacity snapshots directly, call
// SelectBatch(), and assert on the node_id / tier distribution of the routes.

namespace {

// One DRAM tier with available == total (the strategy only reads available).
ClientRecord DramClient(const std::string& node_id, uint64_t avail) {
  return MakeClient(node_id, node_id + "-addr", {{TierType::DRAM, {avail, avail}}});
}

std::vector<bool> NoDedup(size_t n) { return std::vector<bool>(n, false); }

}  // namespace

TEST(ConfigurableRoutePut, MostAvailablePicksLargestFreeNode) {
  ConfigurableRoutePutStrategy strat(Algo::kMostAvailable, Affinity::kNone);
  std::vector<ClientRecord> clients{DramClient("a", 10 * GB), DramClient("b", 20 * GB)};
  auto out = strat.SelectBatch("req", {1 * GB}, NoDedup(1), clients, {});
  ASSERT_EQ(out.size(), 1u);
  ASSERT_TRUE(out[0].has_value());
  EXPECT_EQ(out[0]->node_id, "b");
  EXPECT_EQ(out[0]->tier, TierType::DRAM);
}

// Projected capacity de-clusters within a batch: once a's free space drops
// below the next block, the planner moves to b.
TEST(ConfigurableRoutePut, ProjectedCapacityDeclustersBatch) {
  ConfigurableRoutePutStrategy strat(Algo::kMostAvailable, Affinity::kNone);
  std::vector<ClientRecord> clients{DramClient("a", 10 * GB), DramClient("b", 6 * GB)};
  auto out = strat.SelectBatch("req", {6 * GB, 6 * GB}, NoDedup(2), clients, {});
  ASSERT_EQ(out.size(), 2u);
  ASSERT_TRUE(out[0].has_value());
  ASSERT_TRUE(out[1].has_value());
  EXPECT_EQ(out[0]->node_id, "a");
  EXPECT_EQ(out[1]->node_id, "b");
}

TEST(ConfigurableRoutePut, OrderAndDedupPreserved) {
  ConfigurableRoutePutStrategy strat(Algo::kMostAvailable, Affinity::kNone);
  std::vector<ClientRecord> clients{
      MakeClient("a", "a-addr", {{TierType::HBM, {80 * GB, 6 * GB}}})};
  // Index 0 + 2 are dedup hits; a fits exactly one 6G block, so index 1 routes
  // only if dedup consumed no capacity.
  auto out = strat.SelectBatch("req", {6 * GB, 6 * GB, 6 * GB}, {true, false, true}, clients, {});
  ASSERT_EQ(out.size(), 3u);
  ASSERT_TRUE(out[0].has_value());
  EXPECT_EQ(out[0]->outcome, RoutePutOutcome::kAlreadyExists);
  EXPECT_TRUE(out[0]->node_id.empty());
  ASSERT_TRUE(out[1].has_value());
  EXPECT_EQ(out[1]->outcome, RoutePutOutcome::kRouted);
  EXPECT_EQ(out[1]->node_id, "a");
  EXPECT_EQ(out[1]->tier, TierType::HBM);
  ASSERT_TRUE(out[2].has_value());
  EXPECT_EQ(out[2]->outcome, RoutePutOutcome::kAlreadyExists);
}

TEST(ConfigurableRoutePut, NeverRoutesToSsd) {
  ConfigurableRoutePutStrategy strat(Algo::kMostAvailable, Affinity::kNone);
  std::vector<ClientRecord> only_ssd{
      MakeClient("a", "a-addr", {{TierType::SSD, {100 * GB, 100 * GB}}})};
  auto out = strat.SelectBatch("req", {1 * GB}, NoDedup(1), only_ssd, {});
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FALSE(out[0].has_value());

  std::vector<ClientRecord> ssd_and_dram{MakeClient(
      "a", "a-addr", {{TierType::SSD, {100 * GB, 100 * GB}}, {TierType::DRAM, {2 * GB, 2 * GB}}})};
  auto out2 = strat.SelectBatch("req", {1 * GB}, NoDedup(1), ssd_and_dram, {});
  ASSERT_TRUE(out2[0].has_value());
  EXPECT_EQ(out2[0]->tier, TierType::DRAM);
}

// Strong, seed-independent: a node that cannot fit the block is never selected.
TEST(ConfigurableRoutePut, RandomNeverSelectsInsufficientNode) {
  ConfigurableRoutePutStrategy strat(Algo::kRandom, Affinity::kNone, /*rng_seed=*/7);
  std::vector<ClientRecord> clients{DramClient("big", 10000 * GB), DramClient("tiny", 1024)};
  std::vector<uint64_t> sizes(200, 1 * GB);
  auto out = strat.SelectBatch("req", sizes, NoDedup(sizes.size()), clients, {});
  for (const auto& r : out) {
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->node_id, "big");
  }
}

// Weak, fixed-seed smoke: weight roughly proportional to free space.  Capacities
// dwarf the total drawn so projected deduction does not skew the ratio.
TEST(ConfigurableRoutePut, RandomWeightedDistributionSmoke) {
  ConfigurableRoutePutStrategy strat(Algo::kRandom, Affinity::kNone, /*rng_seed=*/12345);
  std::vector<ClientRecord> clients{DramClient("a", 80000 * GB), DramClient("b", 20000 * GB)};
  const size_t n = 2000;
  std::vector<uint64_t> sizes(n, 1 * GB);
  auto out = strat.SelectBatch("req", sizes, NoDedup(n), clients, {});
  size_t a = 0, b = 0;
  for (const auto& r : out) {
    ASSERT_TRUE(r.has_value());
    if (r->node_id == "a") ++a;
    if (r->node_id == "b") ++b;
  }
  EXPECT_EQ(a + b, n);
  EXPECT_GT(b, 0u);
  const double frac_a = static_cast<double>(a) / n;
  EXPECT_GT(frac_a, 0.65);
  EXPECT_LT(frac_a, 0.92);
}

// NODE_AFFINITY = same: a node fits the whole non-dedup total -> all keys land
// on it.
TEST(ConfigurableRoutePut, SameWholeBatchOnOneNode) {
  ConfigurableRoutePutStrategy strat(Algo::kMostAvailable, Affinity::kSame);
  std::vector<ClientRecord> clients{DramClient("a", 100 * GB), DramClient("b", 100 * GB)};
  std::vector<uint64_t> sizes{1 * GB, 1 * GB, 1 * GB, 1 * GB};
  auto out = strat.SelectBatch("req", sizes, NoDedup(sizes.size()), clients, {});
  ASSERT_EQ(out.size(), 4u);
  for (const auto& r : out) ASSERT_TRUE(r.has_value());
  const std::string node = out[0]->node_id;
  for (const auto& r : out) EXPECT_EQ(r->node_id, node);
}

// No node fits the whole batch: per-key sticky reuse, re-anchoring only when the
// current anchor can no longer fit.
TEST(ConfigurableRoutePut, SamePerKeyStickyFallback) {
  ConfigurableRoutePutStrategy strat(Algo::kMostAvailable, Affinity::kSame);
  std::vector<ClientRecord> clients{DramClient("a", 5 * GB), DramClient("b", 5 * GB)};
  std::vector<uint64_t> sizes{2 * GB, 2 * GB, 2 * GB, 2 * GB};  // total 8G fits no node alone
  auto out = strat.SelectBatch("req", sizes, NoDedup(sizes.size()), clients, {});
  ASSERT_EQ(out.size(), 4u);
  for (const auto& r : out) ASSERT_TRUE(r.has_value());
  EXPECT_EQ(out[0]->node_id, out[1]->node_id);
  EXPECT_EQ(out[2]->node_id, out[3]->node_id);
  EXPECT_NE(out[1]->node_id, out[2]->node_id);
}

// Affinity must never make a key fail that the base algorithm could route: no
// single node fits the whole batch, but the keys still route by spilling across
// nodes via the explicit fallback.
TEST(ConfigurableRoutePut, SameNeverDropsRoutableKey) {
  ConfigurableRoutePutStrategy strat(Algo::kMostAvailable, Affinity::kSame);
  std::vector<ClientRecord> clients{DramClient("a", 3 * GB), DramClient("b", 3 * GB)};
  // total 4G fits no single node (each 3G); both keys route only by spilling.
  std::vector<uint64_t> sizes{2 * GB, 2 * GB};
  auto out = strat.SelectBatch("req", sizes, NoDedup(sizes.size()), clients, {});
  ASSERT_EQ(out.size(), 2u);
  for (const auto& r : out) ASSERT_TRUE(r.has_value());  // none dropped
  EXPECT_NE(out[0]->node_id, out[1]->node_id);           // second key spilled to the other node
}

// same whole-batch pins node AND tier: the batch fits node a's DRAM but not its
// HBM, so every key lands on a's DRAM even though HBM has room for early keys.
TEST(ConfigurableRoutePut, SameWholeBatchPinsNodeAndTier) {
  ConfigurableRoutePutStrategy strat(Algo::kMostAvailable, Affinity::kSame);
  std::vector<ClientRecord> clients{MakeClient(
      "a", "a-addr", {{TierType::HBM, {3 * GB, 3 * GB}}, {TierType::DRAM, {100 * GB, 100 * GB}}})};
  std::vector<uint64_t> sizes{2 * GB, 2 * GB, 2 * GB, 2 * GB};  // total 8G: HBM(3G) no, DRAM yes
  auto out = strat.SelectBatch("req", sizes, NoDedup(sizes.size()), clients, {});
  ASSERT_EQ(out.size(), 4u);
  for (const auto& r : out) {
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->node_id, "a");
    EXPECT_EQ(r->tier, TierType::DRAM);  // pinned tier, HBM left untouched
  }
}

// same per-key fallback must not break tier priority: when the sticky anchor's
// HBM is full, a remote node's HBM beats the anchor's own DRAM.
TEST(ConfigurableRoutePut, SameFallbackPrefersRemoteHbmOverAnchorDram) {
  ConfigurableRoutePutStrategy strat(Algo::kMostAvailable, Affinity::kSame);
  std::vector<ClientRecord> clients{
      MakeClient("a", "a-addr", {{TierType::HBM, {2 * GB, 2 * GB}}}),
      MakeClient("b", "b-addr",
                 {{TierType::HBM, {3 * GB, 3 * GB}}, {TierType::DRAM, {3 * GB, 3 * GB}}})};
  // total 4G exceeds every single node/tier (max 3G) -> no whole-batch pin.
  std::vector<uint64_t> sizes{2 * GB, 2 * GB};
  auto out = strat.SelectBatch("req", sizes, NoDedup(sizes.size()), clients, {});
  ASSERT_EQ(out.size(), 2u);
  ASSERT_TRUE(out[0].has_value());
  ASSERT_TRUE(out[1].has_value());
  // key 0: most-available HBM -> b (3G > a 2G); anchor becomes b, b HBM -> 1G.
  EXPECT_EQ(out[0]->node_id, "b");
  EXPECT_EQ(out[0]->tier, TierType::HBM);
  // key 1: anchor b's HBM (1G) cannot fit; instead of falling to b's DRAM, tier
  // priority routes to a's HBM.
  EXPECT_EQ(out[1]->node_id, "a");
  EXPECT_EQ(out[1]->tier, TierType::HBM);
}

// dedup keys consume no projected capacity and do not count toward the same-node
// whole-batch total.
TEST(ConfigurableRoutePut, SameDedupExcludedFromBatchTotal) {
  ConfigurableRoutePutStrategy strat(Algo::kMostAvailable, Affinity::kSame);
  std::vector<ClientRecord> clients{DramClient("a", 4 * GB)};
  // Non-dedup total is only 2G (index 1+3); the 100G dedup blocks must not push
  // the total past a's 4G or it would refuse the whole-batch placement.
  auto out = strat.SelectBatch("req", {100 * GB, 1 * GB, 100 * GB, 1 * GB},
                               {true, false, true, false}, clients, {});
  ASSERT_EQ(out.size(), 4u);
  EXPECT_EQ(out[0]->outcome, RoutePutOutcome::kAlreadyExists);
  EXPECT_EQ(out[2]->outcome, RoutePutOutcome::kAlreadyExists);
  ASSERT_TRUE(out[1].has_value());
  ASSERT_TRUE(out[3].has_value());
  EXPECT_EQ(out[1]->node_id, "a");
  EXPECT_EQ(out[3]->node_id, "a");
}

// NODE_AFFINITY = local: requester node preferred over a node with more space.
TEST(ConfigurableRoutePut, LocalPrefersRequesterNode) {
  ConfigurableRoutePutStrategy strat(Algo::kMostAvailable, Affinity::kLocal);
  std::vector<ClientRecord> clients{DramClient("a", 100 * GB), DramClient("b", 10 * GB)};
  auto out = strat.SelectBatch(/*requester=*/"b", {1 * GB, 1 * GB}, NoDedup(2), clients, {});
  ASSERT_EQ(out.size(), 2u);
  for (const auto& r : out) {
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->node_id, "b");
  }
}

// Per-key local-first: spill to the base algorithm once local is full.
TEST(ConfigurableRoutePut, LocalFallsBackWhenLocalFull) {
  ConfigurableRoutePutStrategy strat(Algo::kMostAvailable, Affinity::kLocal);
  std::vector<ClientRecord> clients{DramClient("a", 100 * GB), DramClient("b", 2 * GB)};
  auto out =
      strat.SelectBatch(/*requester=*/"b", {1 * GB, 1 * GB, 1 * GB}, NoDedup(3), clients, {});
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0]->node_id, "b");
  EXPECT_EQ(out[1]->node_id, "b");
  EXPECT_EQ(out[2]->node_id, "a");  // local exhausted -> base algo
}

// Requester not in the candidate set: every key falls back to the base
// algorithm, none dropped.
TEST(ConfigurableRoutePut, LocalUnknownRequesterFallsBack) {
  ConfigurableRoutePutStrategy strat(Algo::kMostAvailable, Affinity::kLocal);
  std::vector<ClientRecord> clients{DramClient("a", 100 * GB)};
  auto out = strat.SelectBatch("not-a-member", {1 * GB, 1 * GB}, NoDedup(2), clients, {});
  ASSERT_EQ(out.size(), 2u);
  for (const auto& r : out) {
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->node_id, "a");
  }
}

// ---- GetEnvEnum tests ----

TEST(GetEnvEnum, ResolvesValidEmptyAndUnknown) {
  const char* kName = "UMBP_TEST_ROUTE_PUT_ENUM";
  ResetEnvWarnStateForTesting();

  ::unsetenv(kName);
  EXPECT_EQ(GetEnvEnum(kName, "none", {"none", "same", "local"}), "none");

  ::setenv(kName, "local", 1);
  EXPECT_EQ(GetEnvEnum(kName, "none", {"none", "same", "local"}), "local");

  ::setenv(kName, "  same  ", 1);  // surrounding whitespace trimmed
  EXPECT_EQ(GetEnvEnum(kName, "none", {"none", "same", "local"}), "same");

  ::setenv(kName, "", 1);  // empty -> default, silent
  EXPECT_EQ(GetEnvEnum(kName, "none", {"none", "same", "local"}), "none");

  ::setenv(kName, "bogus", 1);  // unknown -> default + WARN once
  EXPECT_EQ(GetEnvEnum(kName, "none", {"none", "same", "local"}), "none");

  ::unsetenv(kName);
}

}  // namespace
}  // namespace mori::umbp
