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
// That gap is not hypothetical. offload_trigger=on_evict and promote_mode=move
// both do their work inside PeerPool::Evict, whose only production caller is the
// EvictKey handler the master drives. Measuring them under umbp_tier_bench --
// which starts no EvictionManager -- reports zero offloads and no reclamation,
// which reads exactly like the features being unimplemented.
#include <gtest/gtest.h>

#include <algorithm>
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

void RegisterNode(InMemoryMasterMetadataStore* store, Clock::time_point now) {
  ClientRegistration registration;
  registration.node_id = kNodeId;
  registration.node_address = "127.0.0.1:1";
  registration.peer_address = "127.0.0.1:2";
  ASSERT_TRUE(store->RegisterClient(registration, now, std::chrono::seconds(60)));
}

// One heartbeat: `caps` is what makes a bucket look over or under its watermark,
// and `now` becomes the access time of any key this heartbeat introduces --
// the store stamps it on first insertion only, so staggering heartbeats is how a
// test gets a victim order it can predict.
void ReportHeartbeat(InMemoryMasterMetadataStore* store, uint64_t seq, Clock::time_point now,
                     const std::map<TierType, TierCapacity>& caps, TierType key_tier,
                     const std::string& logical_tier, const std::vector<std::string>& keys,
                     bool full_sync) {
  std::vector<KvEvent> events;
  events.reserve(keys.size());
  for (const auto& key : keys) {
    KvEvent event;
    event.kind = KvEvent::Kind::ADD;
    event.key = key;
    event.tier = key_tier;
    event.size = kPageSize;
    event.logical_tier = logical_tier;
    events.push_back(std::move(event));
  }
  store->ApplyHeartbeat(kNodeId, seq, now, caps, events, full_sync);
}

// Reports the node as holding `used` of `total` DRAM bytes, so RunOnce sees a
// bucket over its high watermark.
void PublishNode(InMemoryMasterMetadataStore* store, const std::vector<std::string>& keys,
                 uint64_t total_bytes, uint64_t used_bytes) {
  const auto now = Clock::now();
  RegisterNode(store, now);
  std::map<TierType, TierCapacity> caps;
  caps[TierType::DRAM] = TierCapacity{total_bytes, total_bytes - used_bytes, kPageSize};
  ReportHeartbeat(store, /*seq=*/1, now, caps, TierType::DRAM, "hot", keys, /*full_sync=*/true);
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

// promote_mode=move deletes the cold copy once a key is promoted, but the
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
  tiers.back().promote_mode = PoolTransitionMode::kMove;
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

// Master-side eviction reads one capacity per TierType, and a client with a
// weighted multi-backend policy sums its backends into that one figure
// (MasterClient::SnapshotAndCacheTierCapacities). So a hot tier sharing DRAM with
// a larger cold tier is reported against their combined size, and a hot tier that
// is completely full reads as a medium that is nearly empty.
//
// The pool side already rejects this reasoning for its own watermarks:
// PeakMemberUtilization exists because "pressure does not average", and lets the
// fullest member speak for the tier. RunOnce averages anyway, being one level up
// and seeing only the sum.
//
// Both halves below hold the hot tier at 4 of 4 pages. The only difference is
// whether a sibling logical tier shares its medium, which is not a property of
// the hot tier at all -- and it decides whether the master acts.
TEST(MasterEvictionChain, SiblingTierInTheSameMediumHidesTheOverload) {
  const std::vector<std::string> keys = {"k0", "k1", "k2", "k3"};

  const auto run_round = [&keys](uint64_t reported_total_pages) {
    LocalCopyEngine engine;
    BackendRegistry registry;
    RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, /*pages=*/4);
    RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, /*pages=*/8);
    auto compiled = LogicalTierGraph::Compile(HotColdOnEvict(), registry);
    PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);
    for (const auto& key : keys) CommitKey(&pool, key);

    InMemoryMasterMetadataStore store;
    PublishNode(&store, keys, reported_total_pages * kPageSize, /*used_bytes=*/4 * kPageSize);
    PoolDispatcher dispatcher(&pool);
    EvictionManager manager(store, OneSecondRounds(), &dispatcher);
    manager.Start();
    WaitFor([&] { return !dispatcher.Dispatched().empty(); }, std::chrono::seconds(4));
    manager.Stop();
    return dispatcher.Dispatched().size();
  };

  // Hot alone in its medium: 4 of 4 pages, over the watermark, master acts.
  EXPECT_GT(run_round(/*reported_total_pages=*/4), 0u)
      << "a full tier alone in its medium was not evicted from";

  // Same full hot tier, now summed with a 32-page DRAM sibling: 4 of 36 pages
  // reads as 11% and the master sees nothing to do. This is the bug, and the
  // assertion is deliberately on the current behaviour -- when the decision moves
  // to per-tier utilisation this expectation is what should fail.
  EXPECT_EQ(run_round(/*reported_total_pages=*/36), 0u)
      << "master evicted despite the aggregate being under the watermark -- if "
         "this now fails, the decision has moved off the per-medium sum and this "
         "test should be inverted";
}

// The timing half of the finding above, which the single-key test cannot reach:
// with one candidate the master has no ordering decision to make. Victims are
// named least-recently-accessed first, and a key is promoted precisely because
// it was just read, so its parked source should be the last thing its tier gives
// up. That is what makes `move` reclaim late rather than not at all -- a
// distinction worth a test, because the two look identical over any window
// shorter than the tier's turnover.
TEST(MasterEvictionChain, ParkedMoveSourceIsNamedLastAmongColderKeys) {
  constexpr auto kShortLease = std::chrono::milliseconds(50);
  LocalCopyEngine engine;
  BackendRegistry registry;
  PageBackend::OwnershipConfig hot_ownership;
  hot_ownership.buffer_sizes = {4 * kPageSize};
  auto hot = MakePageBackend(TierType::DRAM, kPageSize, hot_ownership, kNoExpiry, kShortLease);
  ASSERT_TRUE(hot->Init(&engine));
  ASSERT_TRUE(registry.Register("hot", std::move(hot)));
  PageBackend::OwnershipConfig cold_ownership;
  cold_ownership.buffer_sizes = {10 * kPageSize};
  auto cold = MakePageBackend(TierType::SSD, kPageSize, cold_ownership, kNoExpiry, kShortLease);
  ASSERT_TRUE(cold->Init(&engine));
  ASSERT_TRUE(registry.Register("cold", std::move(cold)));

  auto tiers = HotColdOnEvict();
  tiers.back().promote_trigger = PoolPromoteTrigger::kOnRead;
  tiers.back().promote_mode = PoolTransitionMode::kMove;
  auto compiled = LogicalTierGraph::Compile(tiers, registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  const std::vector<std::string> colder = {"cold0", "cold1", "cold2"};
  const auto place_in_cold = [&](const std::string& key) {
    PoolPlacementRequest request;
    request.key = key;
    request.size = kPageSize;
    request.tier = TierType::DRAM;
    request.logical_tier = "cold";
    auto allocation = pool.BatchAllocate({request}).front();
    ASSERT_EQ(allocation.allocation.outcome, AllocateOutcome::kSuccessAllocated) << key;
    ASSERT_TRUE(pool.BatchCommit({PoolCommitRequest{
                                     {allocation.backend_id, allocation.allocation.slot_id}, key}})
                    .front()
                    .commit.success)
        << key;
  };
  for (const auto& key : colder) place_in_cold(key);
  place_in_cold("moved");

  // Read "moved" so it promotes; move defers the source delete behind the read
  // lease, leaving the source parked in cold alongside the three colder keys.
  ASSERT_TRUE(pool.BatchResolve({"moved"}, false).front().resolved.found);
  ASSERT_TRUE(
      WaitFor([&] { return registry.Get("hot")->Contains("moved"); }, std::chrono::seconds(2)));
  ASSERT_TRUE(registry.Get("cold")->Contains("moved"));

  InMemoryMasterMetadataStore store;
  const auto now = Clock::now();
  RegisterNode(&store, now);
  // 10 of 10 SSD pages used, draining to 0.7, so the budget covers three keys of
  // the four present. DRAM is left comfortably under its watermark so the hot
  // tier contributes no bucket of its own.
  std::map<TierType, TierCapacity> caps;
  caps[TierType::SSD] = TierCapacity{10 * kPageSize, 0, kPageSize};
  caps[TierType::DRAM] = TierCapacity{10 * kPageSize, 9 * kPageSize, kPageSize};
  // The three colder keys enter ten seconds earlier, so their access stamps are
  // unambiguously older than the promoted key's.
  ReportHeartbeat(&store, /*seq=*/1, now - std::chrono::seconds(10), caps, TierType::SSD, "cold",
                  colder, /*full_sync=*/true);
  ReportHeartbeat(&store, /*seq=*/2, now, caps, TierType::SSD, "cold", {"moved"},
                  /*full_sync=*/false);

  PoolDispatcher dispatcher(&pool);
  EvictionManager manager(store, OneSecondRounds(), &dispatcher);
  manager.Start();
  const bool dispatched =
      WaitFor([&] { return dispatcher.Dispatched().size() >= colder.size(); },
              std::chrono::seconds(8));
  manager.Stop();

  ASSERT_TRUE(dispatched) << "master never named the colder keys";
  const auto named = dispatcher.Dispatched();

  // The parked source is the newest entry in its tier, so a budget that covers
  // three of four keys must spend it on the other three.
  EXPECT_EQ(std::count(named.begin(), named.end(), "moved"), 0)
      << "the just-promoted key was named ahead of colder entries";
  for (const auto& key : colder) {
    EXPECT_EQ(std::count(named.begin(), named.end(), key), 1) << key << " was not named";
  }
  // And it really is still occupying the tier: this is the lag, stated as an
  // observation rather than inferred from the ordering.
  EXPECT_TRUE(registry.Get("cold")->Contains("moved"))
      << "parked source was reclaimed after all, so the ordering concern is moot";
  EXPECT_TRUE(registry.Get("hot")->Contains("moved"));
}

}  // namespace
}  // namespace mori::umbp
