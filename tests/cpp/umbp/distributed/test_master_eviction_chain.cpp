// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// The master decides what to evict and the peer executes it, and until now the
// two halves were only tested apart: test_master_evict_strategy covers victim
// selection given candidates, and test_peer_pool covers PeerPool::Evict given
// keys. Nothing joined them, so a break anywhere between "this tier is over its
// watermark" and "the key moved to the tier below" was invisible.
//
// That gap is not hypothetical. offload_trigger=on_evict and promotion_mode=move
// both do their work inside PeerPool::Evict, whose only production caller is the
// EvictKey handler the master drives. Measuring them under umbp_tier_bench --
// which starts no EvictionManager -- reports zero offloads and no reclamation,
// which reads exactly like the features being unimplemented.
#include <gtest/gtest.h>

#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// eviction_manager.h only forward-declares MasterEvictStrategy, and the
// constructor's defaulted unique_ptr needs the complete type to destroy it.
#include "umbp/distributed/master/evict_strategy.h"
#include "umbp/distributed/master/eviction_manager.h"
#include "umbp/distributed/master/in_memory_master_metadata_store.h"
#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/pool/peer_pool.h"
#include "umbp/distributed/transfer/local_copy_engine.h"

namespace mori::umbp {
namespace {

using Clock = std::chrono::system_clock;

constexpr uint64_t kPageSize = 64;
constexpr auto kNoExpiry = std::chrono::milliseconds(30000);
constexpr char kNodeId[] = "node-a";

void RegisterPageBackend(BackendRegistry* registry, TransferEngine* engine,
                         const std::string& name, TierType tier, uint64_t pages) {
  PageBackend::OwnershipConfig ownership;
  ownership.buffer_sizes = {pages * kPageSize};
  auto backend = MakePageBackend(tier, kPageSize, ownership, kNoExpiry, kNoExpiry);
  ASSERT_TRUE(backend->Init(engine));
  ASSERT_TRUE(registry->Register(name, std::move(backend)));
}

// hot offloads to cold on evict, which is the trigger whose only driver is the
// master. Cold is a distinct medium so the two tiers land in different master
// buckets; a two-DRAM-tier policy would collapse into one, since the master
// tracks capacity per TierType and knows nothing of logical tiers.
std::vector<LogicalTierConfig> HotColdOnEvict() {
  LogicalTierConfig hot;
  hot.name = "hot";
  hot.entry = true;
  hot.members = {{"hot", 100}};
  hot.offload_to = {"cold"};
  hot.trigger = PoolOffloadTrigger::kOnEvict;
  hot.high_watermark = 0.9;
  hot.low_watermark = 0.7;
  LogicalTierConfig cold;
  cold.name = "cold";
  cold.members = {{"cold", 100}};
  return {hot, cold};
}

void CommitKey(PeerPool* pool, const std::string& key) {
  PoolPlacementRequest request;
  request.key = key;
  request.size = kPageSize;
  request.tier = TierType::DRAM;
  auto allocation = pool->BatchAllocate({request}).front();
  ASSERT_EQ(allocation.allocation.outcome, AllocateOutcome::kSuccessAllocated) << key;
  ASSERT_TRUE(pool->BatchCommit({PoolCommitRequest{
                                    {allocation.backend_id, allocation.allocation.slot_id}, key}})
                  .front()
                  .commit.success)
      << key;
}

// Stands in for MasterPeerStubPool, minus gRPC: the handler in peer_service.cpp
// does exactly this one call, so routing the dispatch straight into the pool
// exercises the same decision-to-execution path a deployed master takes.
class PoolDispatcher final : public EvictKeyDispatcher {
 public:
  explicit PoolDispatcher(PeerPool* pool) : pool_(pool) {}

  void DispatchEvictKey(const std::string& node_id, const std::string& /*peer_address*/,
                        std::vector<std::string> keys) override {
    auto results = pool_->Evict(keys, PoolEvictMode::kReclaim);
    std::lock_guard<std::mutex> lock(mutex_);
    ++rounds_;
    for (size_t i = 0; i < keys.size(); ++i) {
      dispatched_.push_back(keys[i]);
      if (i < results.size()) freed_bytes_ += results[i].bytes_freed;
    }
    node_ids_.push_back(node_id);
  }

  std::vector<std::string> Dispatched() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return dispatched_;
  }
  size_t Rounds() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return rounds_;
  }
  uint64_t FreedBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return freed_bytes_;
  }
  std::vector<std::string> NodeIds() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return node_ids_;
  }

 private:
  PeerPool* pool_;
  mutable std::mutex mutex_;
  std::vector<std::string> dispatched_;
  std::vector<std::string> node_ids_;
  size_t rounds_ = 0;
  uint64_t freed_bytes_ = 0;
};

// Reports the node as holding `used` of `total` DRAM bytes, so RunOnce sees a
// bucket over its high watermark. The keys are registered with staggered access
// times because the default strategy evicts least-recently-accessed first, and a
// test that cannot tell the order apart cannot tell LRU from arbitrary.
void PublishNode(InMemoryMasterMetadataStore* store, const std::vector<std::string>& keys,
                 uint64_t total_bytes, uint64_t used_bytes) {
  const auto now = Clock::now();

  ClientRegistration registration;
  registration.node_id = kNodeId;
  registration.node_address = "127.0.0.1:1";
  registration.peer_address = "127.0.0.1:2";
  registration.tier_capacities[TierType::DRAM] = TierCapacity{total_bytes, total_bytes, kPageSize};
  ASSERT_TRUE(store->RegisterClient(registration, now, std::chrono::seconds(60)));

  std::vector<KvEvent> events;
  events.reserve(keys.size());
  for (const auto& key : keys) {
    KvEvent event;
    event.kind = KvEvent::Kind::ADD;
    event.key = key;
    event.tier = TierType::DRAM;
    event.size = kPageSize;
    event.logical_tier = "hot";
    events.push_back(std::move(event));
  }

  std::map<TierType, TierCapacity> caps;
  caps[TierType::DRAM] = TierCapacity{total_bytes, total_bytes - used_bytes, kPageSize};
  store->ApplyHeartbeat(kNodeId, /*seq=*/1, now, caps, events, /*is_full_sync=*/true);
}

EvictionConfig OneSecondRounds() {
  EvictionConfig config;
  config.high_watermark = 0.9;
  config.low_watermark = 0.7;
  // The loop sleeps before its first pass, and one second is the floor the
  // env override enforces, so this is the shortest a test can wait for a
  // round it did not trigger itself.
  config.check_interval = std::chrono::seconds(1);
  return config;
}

bool WaitFor(const std::function<bool()>& done, std::chrono::milliseconds budget) {
  const auto deadline = std::chrono::steady_clock::now() + budget;
  while (std::chrono::steady_clock::now() < deadline) {
    if (done()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return done();
}

// The whole chain: a tier over its high watermark, through victim selection and
// the dispatch the master performs, into the pool that owns the bytes. on_evict
// has to demote here -- dropping the key instead would free the same bytes and
// look identical to the master, which is why the assertion is on where the key
// ended up and not on how much was freed.
TEST(MasterEvictionChain, OverWatermarkTierDemotesIntoTheColdTier) {
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, /*pages=*/4);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, /*pages=*/8);

  auto compiled = LogicalTierGraph::Compile(HotColdOnEvict(), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  const std::vector<std::string> keys = {"k0", "k1", "k2", "k3"};
  for (const auto& key : keys) CommitKey(&pool, key);
  ASSERT_EQ(registry.Get("hot")->OwnedKeyCount(), keys.size());

  InMemoryMasterMetadataStore store;
  // 4 of 4 pages used is over the 0.9 watermark, and the drain target of 0.7
  // leaves a budget big enough for more than one key.
  PublishNode(&store, keys, /*total_bytes=*/4 * kPageSize, /*used_bytes=*/4 * kPageSize);

  PoolDispatcher dispatcher(&pool);
  EvictionManager manager(store, OneSecondRounds(), &dispatcher);
  manager.Start();
  const bool dispatched =
      WaitFor([&] { return !dispatcher.Dispatched().empty(); }, std::chrono::seconds(8));
  manager.Stop();

  ASSERT_TRUE(dispatched) << "master never dispatched an EvictKey for an over-watermark tier";
  EXPECT_EQ(dispatcher.NodeIds().front(), kNodeId);

  // Every key the master named must have moved down a tier rather than vanished.
  const auto evicted = dispatcher.Dispatched();
  ASSERT_FALSE(evicted.empty());
  for (const auto& key : evicted) {
    EXPECT_FALSE(registry.Get("hot")->Contains(key)) << key << " still in hot";
    EXPECT_TRUE(registry.Get("cold")->Contains(key)) << key << " was dropped, not demoted";
  }
  EXPECT_GT(dispatcher.FreedBytes(), 0u);

  // Keys the master did not name are untouched: eviction is bounded by the
  // budget, so a round must not drain the whole tier.
  EXPECT_EQ(registry.Get("hot")->OwnedKeyCount() + evicted.size(), keys.size());
}

// A tier under its watermark must produce no eviction at all. Without this the
// test above would still pass if RunOnce evicted unconditionally, which is the
// same observation -- keys moving to cold -- for an entirely different reason.
TEST(MasterEvictionChain, TierBelowWatermarkIsLeftAlone) {
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, /*pages=*/8);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, /*pages=*/8);

  auto compiled = LogicalTierGraph::Compile(HotColdOnEvict(), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  const std::vector<std::string> keys = {"k0", "k1"};
  for (const auto& key : keys) CommitKey(&pool, key);

  InMemoryMasterMetadataStore store;
  // 2 of 8 pages is a quarter full, far below the 0.9 watermark.
  PublishNode(&store, keys, /*total_bytes=*/8 * kPageSize, /*used_bytes=*/2 * kPageSize);

  PoolDispatcher dispatcher(&pool);
  EvictionManager manager(store, OneSecondRounds(), &dispatcher);
  manager.Start();
  // Long enough for two rounds at the configured interval.
  std::this_thread::sleep_for(std::chrono::milliseconds(2500));
  manager.Stop();

  EXPECT_EQ(dispatcher.Rounds(), 0u) << "evicted from a tier that was under its watermark";
  for (const auto& key : keys) {
    EXPECT_TRUE(registry.Get("hot")->Contains(key)) << key;
    EXPECT_FALSE(registry.Get("cold")->Contains(key)) << key;
  }
}

// promotion_mode=move deletes the cold copy once a key is promoted, but the
// delete races the read lease the triggering read still holds, so it defers: the
// source parks in draining_sources_ and the promotion reports success. Nothing
// requeues it, and the one place that drains it is PeerPool::Evict -- so whether
// `move` ever reclaims anything at all depends entirely on this chain, and a
// benchmark with no eviction driver reports move and copy as byte-identical.
//
// This pins the reclamation down: a master round that names the key frees the
// parked source and leaves the promoted copy alone. Losing the target instead
// would be silent data loss, which is the one way move can be worse than copy.
TEST(MasterEvictionChain, MasterRoundReclaimsTheParkedSourceOfAMovePromotion) {
  // Short enough that the lease which defers the source delete has expired by
  // the time the master's round arrives a second later.
  constexpr auto kShortLease = std::chrono::milliseconds(50);
  LocalCopyEngine engine;
  BackendRegistry registry;
  PageBackend::OwnershipConfig hot_ownership;
  hot_ownership.buffer_sizes = {4 * kPageSize};
  auto hot = MakePageBackend(TierType::DRAM, kPageSize, hot_ownership, kNoExpiry, kShortLease);
  ASSERT_TRUE(hot->Init(&engine));
  ASSERT_TRUE(registry.Register("hot", std::move(hot)));
  PageBackend::OwnershipConfig cold_ownership;
  cold_ownership.buffer_sizes = {4 * kPageSize};
  auto cold = MakePageBackend(TierType::SSD, kPageSize, cold_ownership, kNoExpiry, kShortLease);
  ASSERT_TRUE(cold->Init(&engine));
  ASSERT_TRUE(registry.Register("cold", std::move(cold)));

  auto tiers = HotColdOnEvict();
  tiers.back().promote_trigger = PoolPromoteTrigger::kOnRead;
  tiers.back().promotion_mode = PoolTransitionMode::kMove;
  auto compiled = LogicalTierGraph::Compile(tiers, registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  // Put the key in cold directly, then read it so the promotion fires.
  PoolPlacementRequest request;
  request.key = "moved";
  request.size = kPageSize;
  request.tier = TierType::DRAM;
  request.logical_tier = "cold";
  auto allocation = pool.BatchAllocate({request}).front();
  ASSERT_EQ(allocation.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_TRUE(pool.BatchCommit({PoolCommitRequest{
                                   {allocation.backend_id, allocation.allocation.slot_id},
                                   "moved"}})
                  .front()
                  .commit.success);
  ASSERT_TRUE(registry.Get("cold")->Contains("moved"));

  ASSERT_TRUE(pool.BatchResolve({"moved"}, false).front().resolved.found);
  ASSERT_TRUE(WaitFor([&] { return registry.Get("hot")->Contains("moved"); },
                      std::chrono::seconds(2)))
      << "promotion never reached the hot tier";
  // The source is still there: this is the deferral, not a failure.
  EXPECT_TRUE(registry.Get("cold")->Contains("moved"))
      << "source was deleted during the promotion, so there is nothing to reclaim";

  InMemoryMasterMetadataStore store;
  PublishNode(&store, {"moved"}, /*total_bytes=*/4 * kPageSize, /*used_bytes=*/4 * kPageSize);

  PoolDispatcher dispatcher(&pool);
  EvictionManager manager(store, OneSecondRounds(), &dispatcher);
  manager.Start();
  const bool reclaimed = WaitFor([&] { return !registry.Get("cold")->Contains("moved"); },
                                 std::chrono::seconds(8));
  manager.Stop();

  EXPECT_TRUE(reclaimed) << "the parked source outlived a master eviction round that named it";
  // The promoted copy is the only one left, so the key is still servable.
  EXPECT_TRUE(registry.Get("hot")->Contains("moved"))
      << "the evict drained the promoted target instead of the parked source";
}

}  // namespace
}  // namespace mori::umbp
