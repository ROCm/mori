// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
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

constexpr auto kNoExpiry = std::chrono::milliseconds(30000);

// A page backend the tier tests can move real bytes through.
void RegisterPageBackend(BackendRegistry* registry, TransferEngine* engine,
                         const std::string& name, TierType tier, uint64_t page_size,
                         uint64_t pages,
                         std::chrono::milliseconds read_lease_ttl = kNoExpiry) {
  PageBackend::OwnershipConfig ownership;
  ownership.buffer_sizes = {pages * page_size};
  auto backend = MakePageBackend(tier, page_size, ownership, kNoExpiry, read_lease_ttl);
  ASSERT_TRUE(backend->Init(engine));
  ASSERT_TRUE(registry->Register(name, std::move(backend)));
}

// The topology most tier tests need: a 'hot' tier that offloads to a 'cold'
// one. Watermarks are only consulted for the watermark trigger, so the same
// pair of values suits both triggers.
std::vector<LogicalTierConfig> HotColdTiers(PoolOffloadTrigger trigger) {
  LogicalTierConfig hot;
  hot.name = "hot";
  hot.entry = true;
  hot.members = {{"hot", 100}};
  hot.offload_to = {"cold"};
  hot.trigger = trigger;
  hot.high_watermark = 0.4;
  hot.low_watermark = 0.2;
  LogicalTierConfig cold;
  cold.name = "cold";
  cold.members = {{"cold", 100}};
  return {hot, cold};
}

// Puts a key through the pool and returns where it landed.
PoolAllocateResult CommitKey(PeerPool* pool, const std::string& key,
                             const std::string& backend_name = {},
                             const std::string& logical_tier = {}) {
  auto request = PutRequest(key, backend_name);
  request.logical_tier = logical_tier;
  auto allocation = pool->BatchAllocate({request}).front();
  EXPECT_EQ(allocation.allocation.outcome, AllocateOutcome::kSuccessAllocated) << key;
  EXPECT_TRUE(pool->BatchCommit({PoolCommitRequest{
                                     {allocation.backend_id, allocation.allocation.slot_id},
                                     key}})
                  .front()
                  .commit.success)
      << key;
  return allocation;
}

// Transitions run on a worker thread, so arrival is observed rather than
// awaited.
bool WaitForKey(BackendRegistry* registry, const std::string& backend,
                const std::string& key) {
  for (int attempt = 0; attempt < 100 && !registry->Get(backend)->Contains(key); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return registry->Get(backend)->Contains(key);
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

// Stands in for a slow migration target: parks inside the allocation a
// transition performs, so a test can observe the pool while bytes are moving.
class BlockingAllocateBackend final : public MockBackend {
 public:
  BlockingAllocateBackend(TierType tier, uint64_t page_size)
      : MockBackend(tier, page_size) {}

  std::vector<AllocateResult> BatchAllocate(
      const std::vector<AllocateRequest>& entries) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      entered_ = true;
    }
    cv_.notify_all();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait_for(lock, std::chrono::seconds(5), [this] { return released_; });
    }
    std::vector<AllocateResult> results(entries.size());
    for (auto& result : results) result.outcome = AllocateOutcome::kFailedNoSpace;
    return results;
  }

  bool WaitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds(5), [this] { return entered_; });
  }

  void Release() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      released_ = true;
    }
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool entered_ = false;
  bool released_ = false;
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

  auto evicted = pool.Evict({"k"}, PoolEvictMode::kReclaim);
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
    auto order = policy->PutOrder(registry, request);
    ASSERT_FALSE(order.empty());
    if (order.front() == dram_a) ++on_a;
    if (order.front() == dram_b) ++on_b;
  }

  EXPECT_EQ(on_a + on_b, 10000u);
  EXPECT_NEAR(static_cast<double>(on_a) / 10000.0, 0.3, 0.03);

  auto repeated = PutRequest("stable-key");
  auto first = policy->PutOrder(registry, repeated);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(policy->PutOrder(registry, repeated), first);
  }
}

TEST(WeightedPlacementPolicy, ExplicitBackendNameOverridesWeightedSelection) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(
      registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  auto policy = MakeWeightedPlacementPolicy({{"dram_a", 1}, {"dram_b", 100}});

  auto order = policy->PutOrder(registry, PutRequest("forced", "dram_a"));
  ASSERT_FALSE(order.empty());
  EXPECT_EQ(order.front(), registry.BackendId(registry.Get("dram_a")));
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

TEST(TieredPlacementPolicy, RejectsTierGraphWithoutTransferEngine) {
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("only", std::make_unique<MockBackend>(TierType::DRAM)));
  auto compiled = LogicalTierGraph::Compile({LogicalTierConfig{{{"only", 1}}}}, registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  EXPECT_THROW(PeerPool(&registry, MakeTieredPlacementPolicy(compiled.graph)),
               std::invalid_argument);
}

TEST(TieredPlacementPolicy, IgnoresPhysicalRouteAndSpillsToConfiguredTarget) {
  LocalCopyEngine engine;
  BackendRegistry registry;
  ASSERT_TRUE(
      registry.Register("hot", std::make_unique<NoSpaceBackend>(TierType::HBM)));
  ASSERT_TRUE(
      registry.Register("cold", std::make_unique<MockBackend>(TierType::SSD)));
  auto compiled =
      LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kOnEvict), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  auto request = PutRequest("spill");
  request.tier = TierType::DRAM;
  auto result = pool.BatchAllocate({request}).front();
  EXPECT_EQ(result.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_EQ(result.backend_id, registry.BackendId(registry.Get("cold")));
}

TEST(TieredPlacementPolicy, HonorsExplicitLogicalTierAndTagsEvents) {
  LocalCopyEngine engine;
  BackendRegistry registry;
  ASSERT_TRUE(registry.Register("hot", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(registry.Register("cold", std::make_unique<MockBackend>(TierType::DRAM)));
  auto compiled =
      LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kOnEvict), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  auto allocation = CommitKey(&pool, "named", {}, "cold");
  ASSERT_EQ(allocation.backend_id, registry.BackendId(registry.Get("cold")));

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
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 2);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 2);
  auto compiled =
      LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kOnEvict), registry);
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

  auto evicted = pool.Evict({"move"}, PoolEvictMode::kReclaim);
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
  constexpr auto kShortLease = std::chrono::milliseconds(1);
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 2, kShortLease);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 2, kShortLease);
  auto compiled =
      LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kOnEvict), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  CommitKey(&pool, "leased");
  ASSERT_TRUE(registry.Get("hot")->BatchResolve({"leased"}, false).front().found);

  auto first = pool.Evict({"leased"}, PoolEvictMode::kReclaim);
  ASSERT_EQ(first.front().bytes_freed, 0u);
  EXPECT_TRUE(registry.Get("hot")->Contains("leased"));
  EXPECT_TRUE(registry.Get("cold")->Contains("leased"));

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  auto retry = pool.Evict({"leased"}, PoolEvictMode::kReclaim);
  EXPECT_EQ(retry.front().bytes_freed, 64u);
  EXPECT_FALSE(registry.Get("hot")->Contains("leased"));
  EXPECT_TRUE(registry.Get("cold")->Contains("leased"));
}

TEST(PeerPool, DiscardDeletesInsteadOfDemoting) {
  constexpr uint64_t kPageSize = 64;
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 2);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 2);
  // The same on_evict topology that demotes under kReclaim.
  auto compiled =
      LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kOnEvict), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  CommitKey(&pool, "gone");

  auto discarded = pool.Evict({"gone"}, PoolEvictMode::kDiscard);
  ASSERT_EQ(discarded.size(), 1u);
  EXPECT_EQ(discarded.front().bytes_freed, 64u);
  EXPECT_FALSE(registry.Get("hot")->Contains("gone"));
  EXPECT_FALSE(registry.Get("cold")->Contains("gone"));
  EXPECT_EQ(pool.PlacementCount(), 0u);
}

TEST(PeerPool, WatermarkMigratesCommittedKey) {
  // Unlike the other tier tests, this one asserts on utilization, so the owned
  // buffer must be a whole number of host allocation granules: a smaller
  // request is rounded up by the memory source, and PageBackend adopts the
  // size it actually got, which would put utilization far below the watermark.
  constexpr uint64_t kPageSize = 4096;
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 2);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 2);
  // Fails loudly if the granularity assumption above ever stops holding.
  ASSERT_EQ(registry.Get("hot")->Capacity().total_bytes, 2 * kPageSize);

  auto compiled =
      LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kWatermark), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  CommitKey(&pool, "watermark");
  EXPECT_TRUE(WaitForKey(&registry, "cold", "watermark"));
  EXPECT_FALSE(registry.Get("hot")->Contains("watermark"));
}

TEST(PeerPool, WatermarkFollowsTheFullestMemberOfAMixedTier) {
  constexpr uint64_t kPageSize = 4096;
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "small", TierType::DRAM, kPageSize, 2);
  RegisterPageBackend(&registry, &engine, "large", TierType::DRAM, kPageSize, 64);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 64);

  LogicalTierConfig parallel;
  parallel.members = {{"small", 1}, {"large", 1}};
  parallel.offload_to = {"cold"};
  parallel.trigger = PoolOffloadTrigger::kWatermark;
  parallel.high_watermark = 0.4;
  parallel.low_watermark = 0.2;
  auto compiled = LogicalTierGraph::Compile(
      {parallel, LogicalTierConfig{{{"cold", 1}}, {}, PoolOffloadTrigger::kOnEvict}},
      registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  // One page on each member. Aggregated over the tier that is ~3% used, so an
  // averaged watermark would never fire, yet "small" is half full.
  CommitKey(&pool, "on_large", "large");
  CommitKey(&pool, "on_small", "small");

  EXPECT_TRUE(WaitForKey(&registry, "cold", "on_small"));
  EXPECT_FALSE(registry.Get("small")->Contains("on_small"));
  // The roomy member is not under pressure, so its key must stay put.
  EXPECT_TRUE(registry.Get("large")->Contains("on_large"));
  EXPECT_FALSE(registry.Get("cold")->Contains("on_large"));
}

TEST(PeerPool, MigrationDoesNotBlockConcurrentPoolOperations) {
  constexpr uint64_t kPageSize = 4096;
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 2);
  auto cold = std::make_unique<BlockingAllocateBackend>(TierType::SSD, kPageSize);
  auto* blocking = cold.get();
  ASSERT_TRUE(registry.Register("cold", std::move(cold)));

  auto compiled =
      LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kWatermark), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  CommitKey(&pool, "blocked");
  ASSERT_TRUE(blocking->WaitUntilEntered());

  // The transition is parked mid-copy. An unrelated resolve must not wait for
  // it, which is only true while the copy runs without the pool lock.
  auto concurrent = std::async(std::launch::async, [&] {
    return pool.BatchResolve({"unrelated"}, false).size();
  });
  EXPECT_EQ(concurrent.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  blocking->Release();
  EXPECT_EQ(concurrent.get(), 1u);
}

TEST(PeerPool, ReadPromotesColdKeyWithoutDeletingSource) {
  constexpr uint64_t kPageSize = 64;
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 2);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 2);

  auto tiers = HotColdTiers(PoolOffloadTrigger::kOnEvict);
  tiers.back().promote_trigger = PoolPromoteTrigger::kOnRead;
  auto compiled = LogicalTierGraph::Compile(tiers, registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  CommitKey(&pool, "promote", {}, "cold");
  ASSERT_TRUE(pool.BatchResolve({"promote"}, false).front().resolved.found);

  EXPECT_TRUE(WaitForKey(&registry, "hot", "promote"));
  EXPECT_TRUE(registry.Get("cold")->Contains("promote"));
}

// The on_hits trigger, which is what made promote_trigger a key-value policy
// rather than a boolean. The threshold has to be counted per key, so one key
// crossing it must not carry another one along, and the count must not survive
// the promotion it caused.
TEST(PeerPool, ReadPromotesOnlyAfterConfiguredHitCount) {
  constexpr uint64_t kPageSize = 64;
  constexpr uint32_t kHits = 3;
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 4);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 4);

  auto tiers = HotColdTiers(PoolOffloadTrigger::kOnEvict);
  tiers.back().promote_trigger = PoolPromoteTrigger::kOnHits;
  tiers.back().promote_hits = kHits;
  auto compiled = LogicalTierGraph::Compile(tiers, registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  CommitKey(&pool, "warm", {}, "cold");
  CommitKey(&pool, "bystander", {}, "cold");

  // Below the threshold the reads still have to be served, from the cold tier,
  // while the count builds up.
  for (uint32_t read = 1; read < kHits; ++read) {
    ASSERT_TRUE(pool.BatchResolve({"warm"}, false).front().resolved.found) << read;
  }
  // Settled first, because promotion is asynchronous: checking straight after
  // the read would also pass while a wrongly queued promotion was still in
  // flight, which is the bug this is meant to catch.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(registry.Get("hot")->Contains("warm"))
      << "promoted after " << kHits - 1 << " reads, below the threshold of " << kHits;

  ASSERT_TRUE(pool.BatchResolve({"warm"}, false).front().resolved.found);
  EXPECT_TRUE(WaitForKey(&registry, "hot", "warm"));

  // The other cold key was read zero times and must not have been dragged along
  // by a count kept per tier instead of per key.
  EXPECT_FALSE(registry.Get("hot")->Contains("bystander"));
  EXPECT_TRUE(registry.Get("cold")->Contains("bystander"));
}

// The move half of promotion_mode, which nothing else covers. Copy leaves the
// cold copy behind, so a lost promotion costs nothing; move deletes it, and the
// promoted copy becomes the only one. That makes two things load-bearing that
// copy mode never has to get right: the source has to go away (or the tier keeps
// paying for a key it no longer serves), and the bytes have to survive the hop
// (or the delete has destroyed the only copy).
//
// The read that triggers the promotion is still holding a lease on the source
// page while the promotion runs, so the delete cannot land inside the transition
// itself; the source is parked as draining and the next evict of that key
// finishes it. Hence the short lease and the explicit evict below.
TEST(PeerPool, ReadPromotesWithMoveDrainsSourceAndKeepsBytes) {
  constexpr uint64_t kPageSize = 64;
  constexpr auto kShortLease = std::chrono::milliseconds(1);
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 2, kShortLease);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 2, kShortLease);

  auto tiers = HotColdTiers(PoolOffloadTrigger::kOnEvict);
  tiers.back().promote_trigger = PoolPromoteTrigger::kOnRead;
  tiers.back().promotion_mode = PoolTransitionMode::kMove;
  auto compiled = LogicalTierGraph::Compile(tiers, registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  // Placed straight onto cold and written through its own page, so the payload
  // the promotion has to carry is known rather than incidental.
  auto request = PutRequest("moved");
  request.logical_tier = "cold";
  auto allocation = pool.BatchAllocate({request}).front();
  ASSERT_EQ(allocation.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_EQ(allocation.backend_id, registry.BackendId(registry.Get("cold")));
  auto source_ref = registry.Get("cold")->BufferRef(allocation.allocation.pages[0].buffer_index);
  ASSERT_TRUE(source_ref.HasHostPtr());
  const std::string payload(kPageSize, 'm');
  std::memcpy(static_cast<char*>(source_ref.host_ptr) +
                  allocation.allocation.pages[0].page_index * kPageSize,
              payload.data(), payload.size());
  ASSERT_TRUE(pool.BatchCommit({PoolCommitRequest{
                                   {allocation.backend_id, allocation.allocation.slot_id},
                                   "moved"}})
                  .front()
                  .commit.success);

  ASSERT_TRUE(pool.BatchResolve({"moved"}, false).front().resolved.found);
  EXPECT_TRUE(WaitForKey(&registry, "hot", "moved"));
  EXPECT_EQ(pool.PlacementBackend("moved"),
            std::optional<uint32_t>{registry.BackendId(registry.Get("hot"))});

  // Past the lease, so the parked source can actually be freed. The evict has to
  // take the draining source and leave the promoted copy alone, which is the one
  // way move can lose data outright.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  pool.Evict({"moved"}, PoolEvictMode::kReclaim);
  EXPECT_FALSE(registry.Get("cold")->Contains("moved"));
  EXPECT_EQ(registry.Get("cold")->OwnedKeyCount(), 0u);
  EXPECT_TRUE(registry.Get("hot")->Contains("moved"));
  EXPECT_EQ(registry.Get("hot")->OwnedKeyCount(), 1u);
  EXPECT_EQ(pool.PlacementBackend("moved"),
            std::optional<uint32_t>{registry.BackendId(registry.Get("hot"))});

  auto resolved = pool.BatchResolve({"moved"}, false).front();
  ASSERT_TRUE(resolved.resolved.found);
  auto target_ref = registry.Get("hot")->BufferRef(resolved.resolved.pages[0].buffer_index);
  ASSERT_TRUE(target_ref.HasHostPtr());
  EXPECT_EQ(std::string(static_cast<char*>(target_ref.host_ptr) +
                            resolved.resolved.pages[0].page_index * kPageSize,
                        payload.size()),
            payload);
}

}  // namespace
}  // namespace mori::umbp
