// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/peer/backend/mock_backend.h"
#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/pool/peer_pool.h"
#include "umbp/distributed/transfer/local_copy_engine.h"

namespace mori::umbp {
namespace {

PoolPlacementRequest PutRequest(std::string key, std::string backend_name = {}) {
  PoolPlacementRequest request;
  request.key = std::move(key);
  request.size = 64;
  request.tier = TierType::DRAM;
  request.backend_name = std::move(backend_name);
  return request;
}

class NoSpaceBackend final : public MockBackend {
 public:
  explicit NoSpaceBackend(TierType tier) : MockBackend(tier) {}

  std::vector<AllocateResult> BatchAllocate(
      const std::vector<AllocateRequest>& entries) override {
    std::vector<AllocateResult> results(entries.size());
    for (auto& result : results) result.outcome = AllocateOutcome::kFailedNoSpace;
    return results;
  }
};

TEST(PeerPool, DefaultPolicyPreservesTierOnlySelection) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(
      registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());

  auto allocated = pool.BatchAllocate({PutRequest("k")}).front();
  ASSERT_EQ(allocated.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_EQ(allocated.backend_id, registry.BackendId(registry.Get("dram_a")));
}

TEST(PeerPool, NamedBackendCommitBuildsLogicalPlacement) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(
      registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());

  auto allocated = pool.BatchAllocate({PutRequest("k", "dram_b")}).front();
  ASSERT_EQ(allocated.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  const uint32_t second_id = registry.BackendId(registry.Get("dram_b"));
  ASSERT_EQ(allocated.backend_id, second_id);

  PoolCommitRequest commit;
  commit.slot = PoolSlotRef{allocated.backend_id, allocated.allocation.slot_id};
  commit.key = "k";
  ASSERT_TRUE(pool.BatchCommit({commit}).front().commit.success);
  EXPECT_EQ(pool.PlacementBackend("k"), std::optional<uint32_t>{second_id});
  EXPECT_EQ(pool.PlacementCount(), 1u);

  auto resolved = pool.BatchResolve({"k"}, /*include_descs=*/false).front();
  ASSERT_TRUE(resolved.resolved.found);
  EXPECT_EQ(resolved.backend_id, second_id);

  // Logical dedup happens above the physical backends: asking another instance
  // for the same committed key must not create a second copy.
  auto duplicate = pool.BatchAllocate({PutRequest("k", "dram_a")}).front();
  EXPECT_EQ(duplicate.allocation.outcome, AllocateOutcome::kSuccessAlreadyExists);
  EXPECT_EQ(duplicate.backend_id, second_id);
  EXPECT_EQ(registry.Get("dram_a")->OwnedKeyCount(), 0u);

  auto evicted = pool.Evict({"k"});
  ASSERT_EQ(evicted.size(), 1u);
  EXPECT_EQ(evicted[0].bytes_freed, 64u);
  EXPECT_EQ(pool.PlacementCount(), 0u);
}

TEST(PeerPool, ResolveFallbackRepairsPlacementIndex) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(
      registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());

  auto* second = registry.Get("dram_b");
  auto allocated = second->BatchAllocate({AllocateRequest{"recovered", 32}}).front();
  ASSERT_TRUE(second->BatchCommit({CommitRequest{allocated.slot_id, "recovered"}}).front().success);

  auto resolved = pool.BatchResolve({"recovered"}, /*include_descs=*/false).front();
  ASSERT_TRUE(resolved.resolved.found);
  const uint32_t second_id = registry.BackendId(second);
  EXPECT_EQ(resolved.backend_id, second_id);
  EXPECT_EQ(pool.PlacementBackend("recovered"), std::optional<uint32_t>{second_id});
}

TEST(PeerPool, ColdIndexDoesNotDuplicatePersistentKey) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(
      registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  auto* second = registry.Get("dram_b");
  auto physical = second->BatchAllocate({AllocateRequest{"persistent", 32}}).front();
  ASSERT_TRUE(second->BatchCommit({CommitRequest{physical.slot_id, "persistent"}}).front().success);

  PeerPool pool(&registry, MakeSingleBackendPolicy());
  auto result = pool.BatchAllocate({PutRequest("persistent", "dram_a")}).front();
  EXPECT_EQ(result.allocation.outcome, AllocateOutcome::kSuccessAlreadyExists);
  EXPECT_EQ(result.backend_id, registry.BackendId(second));
  EXPECT_EQ(registry.Get("dram_a")->OwnedKeyCount(), 0u);
}

TEST(PeerPool, ClearDropsPhysicalAndLogicalState) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("dram", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());

  auto allocated = pool.BatchAllocate({PutRequest("k")}).front();
  PoolCommitRequest commit;
  commit.slot = PoolSlotRef{allocated.backend_id, allocated.allocation.slot_id};
  commit.key = "k";
  ASSERT_TRUE(pool.BatchCommit({commit}).front().commit.success);
  ASSERT_EQ(pool.PlacementCount(), 1u);

  pool.ClearLocal();
  EXPECT_EQ(pool.PlacementCount(), 0u);
  EXPECT_EQ(registry.Get("dram")->OwnedKeyCount(), 0u);
}

TEST(PeerPool, BatchResultsRemainPositionalAcrossBackends) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(
      registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());

  auto results = pool.BatchAllocate(
      {PutRequest("a", "dram_b"), PutRequest("bad", "missing"), PutRequest("b", "dram_a")});
  ASSERT_EQ(results.size(), 3u);
  EXPECT_EQ(results[0].backend_id, registry.BackendId(registry.Get("dram_b")));
  EXPECT_EQ(results[0].allocation.outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_EQ(results[1].allocation.outcome, AllocateOutcome::kFailed);
  EXPECT_EQ(results[2].backend_id, registry.BackendId(registry.Get("dram_a")));
  EXPECT_EQ(results[2].allocation.outcome, AllocateOutcome::kSuccessAllocated);
}

TEST(PeerPool, ReservesDuplicateKeysUntilCommitOrAbort) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(
      registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());

  auto results =
      pool.BatchAllocate({PutRequest("same", "dram_a"), PutRequest("same", "dram_b")});
  ASSERT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].allocation.outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_EQ(results[1].allocation.outcome, AllocateOutcome::kFailed);

  ASSERT_TRUE(pool
                  .BatchAbort(
                      {PoolSlotRef{results[0].backend_id, results[0].allocation.slot_id}})
                  .front());
  auto retried = pool.BatchAllocate({PutRequest("same", "dram_b")}).front();
  EXPECT_EQ(retried.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_EQ(retried.backend_id, registry.BackendId(registry.Get("dram_b")));
}

TEST(PeerPool, CommitMustMatchReservedSlotAndKey) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("dram", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());

  auto allocated = pool.BatchAllocate({PutRequest("reserved")}).front();
  PoolCommitRequest wrong;
  wrong.slot = PoolSlotRef{allocated.backend_id, allocated.allocation.slot_id};
  wrong.key = "different";
  EXPECT_FALSE(pool.BatchCommit({wrong}).front().commit.success);

  PoolCommitRequest correct = wrong;
  correct.key = "reserved";
  EXPECT_TRUE(pool.BatchCommit({correct}).front().commit.success);
  EXPECT_EQ(pool.PlacementBackend("reserved"),
            std::optional<uint32_t>{allocated.backend_id});
  EXPECT_FALSE(pool.PlacementBackend("different").has_value());
}

TEST(PeerPool, StalePlacementDoesNotSuppressNewAllocation) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(
      registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());

  auto allocated = pool.BatchAllocate({PutRequest("stale", "dram_a")}).front();
  PoolCommitRequest commit;
  commit.slot = PoolSlotRef{allocated.backend_id, allocated.allocation.slot_id};
  commit.key = "stale";
  ASSERT_TRUE(pool.BatchCommit({commit}).front().commit.success);

  // Simulate physical state being cleared/recovered outside this process-local
  // index. Allocate validates the indexed backend before claiming a duplicate.
  registry.Get("dram_a")->ClearLocal();
  auto replacement = pool.BatchAllocate({PutRequest("stale", "dram_b")}).front();
  EXPECT_EQ(replacement.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_EQ(replacement.backend_id, registry.BackendId(registry.Get("dram_b")));
}

TEST(WeightedPlacementPolicy, DistributesDeterministicallyWithinRequestedTier) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(
      registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(
      registry.Register("ssd_a", std::make_unique<MockBackend>(TierType::SSD)));
  auto policy = MakeWeightedPlacementPolicy(
      {{"dram_a", 3}, {"dram_b", 7}, {"ssd_a", 100}});

  const uint32_t dram_a = registry.BackendId(registry.Get("dram_a"));
  const uint32_t dram_b = registry.BackendId(registry.Get("dram_b"));
  size_t on_a = 0;
  size_t on_b = 0;
  for (size_t i = 0; i < 10000; ++i) {
    auto request = PutRequest("weighted-" + std::to_string(i * 2654435761ULL));
    auto selected = policy->SelectPutBackend(registry, request);
    ASSERT_TRUE(selected.has_value());
    if (*selected == dram_a) ++on_a;
    if (*selected == dram_b) ++on_b;
  }

  EXPECT_EQ(on_a + on_b, 10000u);
  EXPECT_NEAR(static_cast<double>(on_a) / 10000.0, 0.3, 0.03);

  auto repeated = PutRequest("stable-key");
  auto first = policy->SelectPutBackend(registry, repeated);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(policy->SelectPutBackend(registry, repeated), first);
  }
}

TEST(WeightedPlacementPolicy, ExplicitBackendNameOverridesWeightedSelection) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(
      registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  auto policy = MakeWeightedPlacementPolicy({{"dram_a", 1}, {"dram_b", 100}});

  auto selected =
      policy->SelectPutBackend(registry, PutRequest("forced", "dram_a"));
  ASSERT_TRUE(selected.has_value());
  EXPECT_EQ(*selected, registry.BackendId(registry.Get("dram_a")));
}

TEST(PeerPool, WeightedPolicyPlacesNewKeysOnMultipleBackends) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(
      registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(
      &registry, MakeWeightedPlacementPolicy({{"dram_a", 1}, {"dram_b", 1}}));

  std::vector<PoolPlacementRequest> requests;
  for (size_t i = 0; i < 128; ++i) {
    requests.push_back(PutRequest("pool-weighted-" + std::to_string(i)));
  }
  auto results = pool.BatchAllocate(requests);

  const uint32_t dram_a = registry.BackendId(registry.Get("dram_a"));
  const uint32_t dram_b = registry.BackendId(registry.Get("dram_b"));
  size_t on_a = 0;
  size_t on_b = 0;
  std::vector<PoolSlotRef> slots;
  for (const auto& result : results) {
    ASSERT_EQ(result.allocation.outcome, AllocateOutcome::kSuccessAllocated);
    if (result.backend_id == dram_a) ++on_a;
    if (result.backend_id == dram_b) ++on_b;
    slots.push_back(PoolSlotRef{result.backend_id, result.allocation.slot_id});
  }
  EXPECT_GT(on_a, 0u);
  EXPECT_GT(on_b, 0u);
  pool.BatchAbort(slots);
}

TEST(PeerPool, WeightedPolicyFallsBackWhenPrimaryHasNoSpace) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("dram_a", std::make_unique<NoSpaceBackend>(TierType::DRAM)));
  ASSERT_TRUE(
      registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  auto policy = MakeWeightedPlacementPolicy({{"dram_a", 1}, {"dram_b", 1}});

  PoolPlacementRequest request;
  bool found_primary = false;
  for (size_t i = 0; i < 100; ++i) {
    request = PutRequest("fallback-" + std::to_string(i));
    auto order = policy->PutOrder(registry, request);
    if (!order.empty() && order.front() == registry.BackendId(registry.Get("dram_a"))) {
      found_primary = true;
      break;
    }
  }
  ASSERT_TRUE(found_primary);
  ASSERT_EQ(policy->PutOrder(registry, request).front(),
            registry.BackendId(registry.Get("dram_a")));

  PeerPool pool(&registry, std::move(policy));
  auto result = pool.BatchAllocate({request}).front();
  EXPECT_EQ(result.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_EQ(result.backend_id, registry.BackendId(registry.Get("dram_b")));
}

TEST(TieredPlacementPolicy, IgnoresPhysicalRouteAndSpillsToConfiguredTarget) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("hot", std::make_unique<NoSpaceBackend>(TierType::HBM)));
  ASSERT_TRUE(
      registry.Register("cold", std::make_unique<MockBackend>(TierType::SSD)));
  std::vector<LogicalTierConfig> tiers = {
      LogicalTierConfig{{{"hot", 100}}, {"cold"}, PoolOffloadTrigger::kOnEvict},
      LogicalTierConfig{{{"cold", 100}}, {}, PoolOffloadTrigger::kOnEvict},
  };
  auto compiled = LogicalTierGraph::Compile(tiers, registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph));

  auto request = PutRequest("spill");
  request.tier = TierType::DRAM;
  auto result = pool.BatchAllocate({request}).front();
  EXPECT_EQ(result.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_EQ(result.backend_id, registry.BackendId(registry.Get("cold")));
}

TEST(TieredPlacementPolicy, HonorsExplicitLogicalTierAndTagsEvents) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("hot_backend", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(
      registry.Register("cold_backend", std::make_unique<MockBackend>(TierType::DRAM)));
  LogicalTierConfig hot;
  hot.name = "hot";
  hot.entry = true;
  hot.members = {{"hot_backend", 1}};
  hot.offload_to = {"cold"};
  LogicalTierConfig cold;
  cold.name = "cold";
  cold.members = {{"cold_backend", 1}};
  auto compiled = LogicalTierGraph::Compile({hot, cold}, registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph));

  auto request = PutRequest("named");
  request.logical_tier = "cold";
  auto allocation = pool.BatchAllocate({request}).front();
  ASSERT_EQ(allocation.backend_id,
            registry.BackendId(registry.Get("cold_backend")));
  PoolCommitRequest commit{{allocation.backend_id,
                            allocation.allocation.slot_id},
                           "named"};
  ASSERT_TRUE(pool.BatchCommit({commit}).front().commit.success);

  auto events = pool.DrainPendingEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_EQ(events.front().logical_tier, "cold");
  auto capacities = pool.LogicalTierCapacities();
  EXPECT_TRUE(capacities.count("hot"));
  EXPECT_TRUE(capacities.count("cold"));
}

TEST(PeerPool, OnEvictMigratesBytesToConfiguredBackend) {
  constexpr uint64_t kPageSize = 64;
  LocalCopyEngine engine;
  BackendRegistry registry;
  PageBackend::OwnershipConfig ownership;
  ownership.buffer_sizes = {2 * kPageSize};
  auto hot = MakePageBackend(TierType::DRAM, kPageSize, ownership,
                             std::chrono::milliseconds(30000),
                             std::chrono::milliseconds(30000));
  auto cold = MakePageBackend(TierType::SSD, kPageSize, ownership,
                              std::chrono::milliseconds(30000),
                              std::chrono::milliseconds(30000));
  ASSERT_TRUE(hot->Init(&engine));
  ASSERT_TRUE(cold->Init(&engine));
  ASSERT_TRUE(registry.Register("hot", std::move(hot)));
  ASSERT_TRUE(registry.Register("cold", std::move(cold)));

  std::vector<LogicalTierConfig> tiers = {
      LogicalTierConfig{{{"hot", 100}}, {"cold"}, PoolOffloadTrigger::kOnEvict},
      LogicalTierConfig{{{"cold", 100}}, {}, PoolOffloadTrigger::kOnEvict},
  };
  auto compiled = LogicalTierGraph::Compile(tiers, registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  auto allocation = pool.BatchAllocate({PutRequest("move")}).front();
  ASSERT_EQ(allocation.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  auto source_ref = registry.Get("hot")->BufferRef(allocation.allocation.pages[0].buffer_index);
  ASSERT_TRUE(source_ref.HasHostPtr());
  const std::string payload(64, 'x');
  std::memcpy(static_cast<char*>(source_ref.host_ptr) +
                  allocation.allocation.pages[0].page_index * kPageSize,
              payload.data(), payload.size());
  ASSERT_TRUE(pool.BatchCommit({PoolCommitRequest{
                                   {allocation.backend_id, allocation.allocation.slot_id},
                                   "move"}})
                  .front()
                  .commit.success);

  auto evicted = pool.Evict({"move"});
  ASSERT_EQ(evicted.size(), 1u);
  EXPECT_EQ(evicted.front().bytes_freed, payload.size());
  EXPECT_FALSE(registry.Get("hot")->Contains("move"));
  EXPECT_TRUE(registry.Get("cold")->Contains("move"));
  EXPECT_EQ(pool.PlacementBackend("move"),
            std::optional<uint32_t>{registry.BackendId(registry.Get("cold"))});

  auto resolved = pool.BatchResolve({"move"}, false).front();
  ASSERT_TRUE(resolved.resolved.found);
  auto target_ref =
      registry.Get("cold")->BufferRef(resolved.resolved.pages[0].buffer_index);
  ASSERT_TRUE(target_ref.HasHostPtr());
  EXPECT_EQ(std::string(static_cast<char*>(target_ref.host_ptr) +
                            resolved.resolved.pages[0].page_index * kPageSize,
                        payload.size()),
            payload);
}

TEST(PeerPool, EvictionRetryDrainsLeasedSourceWithoutDeletingTarget) {
  constexpr uint64_t kPageSize = 64;
  LocalCopyEngine engine;
  BackendRegistry registry;
  PageBackend::OwnershipConfig ownership;
  ownership.buffer_sizes = {2 * kPageSize};
  auto hot = MakePageBackend(TierType::DRAM, kPageSize, ownership,
                             std::chrono::milliseconds(30000),
                             std::chrono::milliseconds(1));
  auto cold = MakePageBackend(TierType::SSD, kPageSize, ownership,
                              std::chrono::milliseconds(30000),
                              std::chrono::milliseconds(1));
  ASSERT_TRUE(hot->Init(&engine));
  ASSERT_TRUE(cold->Init(&engine));
  ASSERT_TRUE(registry.Register("hot", std::move(hot)));
  ASSERT_TRUE(registry.Register("cold", std::move(cold)));
  std::vector<LogicalTierConfig> tiers = {
      LogicalTierConfig{{{"hot", 100}}, {"cold"}, PoolOffloadTrigger::kOnEvict},
      LogicalTierConfig{{{"cold", 100}}, {}, PoolOffloadTrigger::kOnEvict},
  };
  auto compiled = LogicalTierGraph::Compile(tiers, registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  auto allocation = pool.BatchAllocate({PutRequest("leased")}).front();
  ASSERT_TRUE(pool.BatchCommit({PoolCommitRequest{
                                   {allocation.backend_id, allocation.allocation.slot_id},
                                   "leased"}})
                  .front()
                  .commit.success);
  ASSERT_TRUE(registry.Get("hot")->BatchResolve({"leased"}, false).front().found);

  auto first = pool.Evict({"leased"});
  ASSERT_EQ(first.front().bytes_freed, 0u);
  EXPECT_TRUE(registry.Get("hot")->Contains("leased"));
  EXPECT_TRUE(registry.Get("cold")->Contains("leased"));

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  auto retry = pool.Evict({"leased"});
  EXPECT_EQ(retry.front().bytes_freed, 64u);
  EXPECT_FALSE(registry.Get("hot")->Contains("leased"));
  EXPECT_TRUE(registry.Get("cold")->Contains("leased"));
}

TEST(PeerPool, WatermarkMigratesCommittedKey) {
  constexpr uint64_t kPageSize = 64;
  LocalCopyEngine engine;
  BackendRegistry registry;
  PageBackend::OwnershipConfig ownership;
  ownership.buffer_sizes = {2 * kPageSize};
  auto hot = MakePageBackend(TierType::DRAM, kPageSize, ownership,
                             std::chrono::milliseconds(30000),
                             std::chrono::milliseconds(30000));
  auto cold = MakePageBackend(TierType::SSD, kPageSize, ownership,
                              std::chrono::milliseconds(30000),
                              std::chrono::milliseconds(30000));
  ASSERT_TRUE(hot->Init(&engine));
  ASSERT_TRUE(cold->Init(&engine));
  ASSERT_TRUE(registry.Register("hot", std::move(hot)));
  ASSERT_TRUE(registry.Register("cold", std::move(cold)));

  LogicalTierConfig hot_tier;
  hot_tier.members = {{"hot", 100}};
  hot_tier.offload_to = {"cold"};
  hot_tier.trigger = PoolOffloadTrigger::kWatermark;
  hot_tier.high_watermark = 0.4;
  hot_tier.low_watermark = 0.2;
  std::vector<LogicalTierConfig> tiers = {
      hot_tier,
      LogicalTierConfig{{{"cold", 100}}, {}, PoolOffloadTrigger::kOnEvict},
  };
  auto compiled = LogicalTierGraph::Compile(tiers, registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  auto allocation = pool.BatchAllocate({PutRequest("watermark")}).front();
  ASSERT_EQ(allocation.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_TRUE(pool.BatchCommit({PoolCommitRequest{
                                   {allocation.backend_id, allocation.allocation.slot_id},
                                   "watermark"}})
                  .front()
                  .commit.success);
  for (int attempt = 0; attempt < 100 && !registry.Get("cold")->Contains("watermark");
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  EXPECT_FALSE(registry.Get("hot")->Contains("watermark"));
  EXPECT_TRUE(registry.Get("cold")->Contains("watermark"));
}

TEST(PeerPool, ReadPromotesColdKeyWithoutDeletingSource) {
  constexpr uint64_t kPageSize = 64;
  LocalCopyEngine engine;
  BackendRegistry registry;
  PageBackend::OwnershipConfig ownership;
  ownership.buffer_sizes = {2 * kPageSize};
  auto hot = MakePageBackend(TierType::DRAM, kPageSize, ownership,
                             std::chrono::milliseconds(30000),
                             std::chrono::milliseconds(30000));
  auto cold = MakePageBackend(TierType::SSD, kPageSize, ownership,
                              std::chrono::milliseconds(30000),
                              std::chrono::milliseconds(30000));
  ASSERT_TRUE(hot->Init(&engine));
  ASSERT_TRUE(cold->Init(&engine));
  ASSERT_TRUE(registry.Register("hot", std::move(hot)));
  ASSERT_TRUE(registry.Register("cold", std::move(cold)));

  LogicalTierConfig hot_tier;
  hot_tier.name = "hot";
  hot_tier.entry = true;
  hot_tier.members = {{"hot", 1}};
  hot_tier.offload_to = {"cold"};
  LogicalTierConfig cold_tier;
  cold_tier.name = "cold";
  cold_tier.members = {{"cold", 1}};
  cold_tier.promote_on_read = true;
  auto compiled = LogicalTierGraph::Compile({hot_tier, cold_tier}, registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  auto request = PutRequest("promote");
  request.logical_tier = "cold";
  auto allocation = pool.BatchAllocate({request}).front();
  ASSERT_EQ(allocation.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  PoolCommitRequest commit{
      {allocation.backend_id, allocation.allocation.slot_id}, "promote"};
  ASSERT_TRUE(pool.BatchCommit({commit})
                  .front()
                  .commit.success);
  ASSERT_TRUE(pool.BatchResolve({"promote"}, false).front().resolved.found);

  for (int attempt = 0; attempt < 100 && !registry.Get("hot")->Contains("promote");
       ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  EXPECT_TRUE(registry.Get("hot")->Contains("promote"));
  EXPECT_TRUE(registry.Get("cold")->Contains("promote"));
}

}  // namespace
}  // namespace mori::umbp
