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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "umbp/distributed/peer/backend/instrumented_backend.h"
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
void RegisterPageBackend(BackendRegistry* registry, TransferEngine* engine, const std::string& name,
                         TierType tier, uint64_t page_size, uint64_t pages,
                         std::chrono::milliseconds read_lease_ttl = kNoExpiry) {
  PageBackend::OwnershipConfig ownership;
  ownership.buffer_sizes = {pages * page_size};
  auto backend = MakePageBackend(tier, page_size, ownership, kNoExpiry, read_lease_ttl);
  backend = MakeInstrumentedBackend(std::move(backend));
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
                                    {allocation.backend_id, allocation.allocation.slot_id}, key}})
                  .front()
                  .commit.success)
      << key;
  return allocation;
}

// Transitions run on a worker thread, so arrival is observed rather than
// awaited.
bool WaitForKey(BackendRegistry* registry, const std::string& backend, const std::string& key) {
  for (int attempt = 0; attempt < 100 && !registry->Get(backend)->Contains(key); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return registry->Get(backend)->Contains(key);
}

// Eviction demotions run on the transition worker, so wait for the source.
bool WaitForKeyGone(BackendRegistry* registry, const std::string& backend, const std::string& key) {
  for (int attempt = 0; attempt < 500 && registry->Get(backend)->Contains(key); ++attempt) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return !registry->Get(backend)->Contains(key);
}

// Holds every copy until Release(), or 10 s so a failed test can't hang.
class GatedEngine final : public TransferEngine {
 public:
  const char* Name() const override { return "gated"; }
  TransferRef RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                             int device) override {
    return inner_.RegisterMemory(base, size, loc, device);
  }
  void Deregister(const TransferRef& ref) override { inner_.Deregister(ref); }
  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override {
    return inner_.CanHandle(src, dst);
  }
  std::unique_ptr<TransferPlanBuilder> CreatePlanBuilder(size_t expected_items) const override {
    return inner_.CreatePlanBuilder(expected_items);
  }
  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan> plans) override {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      ++held_;
      cv_.notify_all();
      cv_.wait_for(lock, std::chrono::seconds(10), [&] { return open_; });
    }
    return inner_.Submit(std::move(plans));
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    open_ = true;
    cv_.notify_all();
  }
  // True once `n` copies are being held.
  bool WaitForHeld(int n) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds(5), [&] { return held_ >= n; });
  }

 private:
  LocalCopyEngine inner_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool open_ = false;
  int held_ = 0;
};

class NoSpaceBackend final : public MockBackend {
 public:
  explicit NoSpaceBackend(TierType tier) : MockBackend(tier) {}

  std::vector<AllocateResult> BatchAllocate(const std::vector<AllocateRequest>& entries) override {
    std::vector<AllocateResult> results(entries.size());
    for (auto& result : results) result.outcome = AllocateOutcome::kFailedNoSpace;
    return results;
  }
};

// LocalCopyEngine whose next `failures` copies fail.
class FailingEngine final : public TransferEngine {
 public:
  explicit FailingEngine(int failures) : failures_(failures) {}
  void FailNext(int n) { failures_.store(n); }

  const char* Name() const override { return "failing"; }
  TransferRef RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                             int device) override {
    return inner_.RegisterMemory(base, size, loc, device);
  }
  void Deregister(const TransferRef& ref) override { inner_.Deregister(ref); }
  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override {
    return inner_.CanHandle(src, dst);
  }
  std::unique_ptr<TransferPlanBuilder> CreatePlanBuilder(size_t expected_items) const override {
    return inner_.CreatePlanBuilder(expected_items);
  }
  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan> plans) override {
    if (failures_.load() > 0) {
      failures_.fetch_sub(1);
      return nullptr;  // the copy never happened
    }
    return inner_.Submit(std::move(plans));
  }

 private:
  LocalCopyEngine inner_;
  std::atomic<int> failures_;
};

class BusyResolveBackend final : public MockBackend {
 public:
  explicit BusyResolveBackend(TierType tier) : MockBackend(tier) {}

  std::vector<ResolvedEntry> BatchResolve(
      const std::vector<std::string>& keys, bool include_descs, bool allow_file_refs = false,
      ResolveReader reader = ResolveReader::kRemote,
      const std::vector<ObjectExtent>* extents = nullptr) override {
    if (!busy_) {
      return MockBackend::BatchResolve(keys, include_descs, allow_file_refs, reader, extents);
    }
    std::vector<ResolvedEntry> results(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      if (Contains(keys[i])) results[i].outcome = ResolveOutcome::kBusy;
    }
    return results;
  }

  void SetBusy(bool busy) { busy_ = busy; }

 private:
  bool busy_ = true;
};

// Stands in for a slow migration target: parks inside the allocation a
// transition performs, so a test can observe the pool while bytes are moving.
class BlockingAllocateBackend final : public MockBackend {
 public:
  BlockingAllocateBackend(TierType tier, uint64_t page_size) : MockBackend(tier, page_size) {}

  std::vector<AllocateResult> BatchAllocate(const std::vector<AllocateRequest>& entries) override {
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

// Parks one backend resolve so the test can prove PeerPool does not retain its
// peer-wide metadata lock while a medium performs slow IO.
class BlockingResolveBackend final : public MockBackend {
 public:
  explicit BlockingResolveBackend(TierType tier) : MockBackend(tier) {}

  std::vector<ResolvedEntry> BatchResolve(
      const std::vector<std::string>& keys, bool include_descs, bool allow_file_refs = false,
      ResolveReader reader = ResolveReader::kRemote,
      const std::vector<ObjectExtent>* extents = nullptr) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      entered_ = true;
    }
    cv_.notify_all();
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait_for(lock, std::chrono::seconds(5), [this] { return released_; });
    }
    return MockBackend::BatchResolve(keys, include_descs, allow_file_refs, reader, extents);
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

// Parks one backend commit, the step where an SSD medium writes the device.
class BlockingCommitBackend final : public MockBackend {
 public:
  explicit BlockingCommitBackend(TierType tier) : MockBackend(tier) {}

  std::vector<CommitResult> BatchCommit(const std::vector<CommitRequest>& entries) override {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      entered_ = true;
      cv_.notify_all();
      cv_.wait_for(lock, std::chrono::seconds(5), [this] { return released_; });
    }
    return MockBackend::BatchCommit(entries);
  }

  bool WaitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds(5), [this] { return entered_; });
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
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
  ASSERT_TRUE(registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());

  auto allocated = pool.BatchAllocate({PutRequest("k")}).front();
  ASSERT_EQ(allocated.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_EQ(allocated.backend_id, registry.BackendId(registry.Get("dram_a")));
}

TEST(PeerPool, PreservesBusyOwnerInsteadOfTurningItIntoAMiss) {
  BackendRegistry registry;
  auto backend = std::make_unique<BusyResolveBackend>(TierType::DRAM);
  auto* busy = backend.get();
  ASSERT_TRUE(registry.Register(std::move(backend)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());
  CommitKey(&pool, "busy-key");

  auto first = pool.BatchResolve({"busy-key"}, false);
  ASSERT_EQ(first.size(), 1u);
  EXPECT_FALSE(first[0].resolved.found);
  EXPECT_EQ(first[0].resolved.outcome, ResolveOutcome::kBusy);

  busy->SetBusy(false);
  auto second = pool.BatchResolve({"busy-key"}, false);
  ASSERT_TRUE(second[0].resolved.found);
  EXPECT_EQ(EffectiveResolveOutcome(second[0].resolved), ResolveOutcome::kFound);
}

TEST(PeerPool, NamedBackendCommitBuildsLogicalPlacement) {
  BackendRegistry registry;
  ASSERT_TRUE(registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
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
  ASSERT_TRUE(registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
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
  ASSERT_TRUE(registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
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
  ASSERT_TRUE(registry.Register("dram", std::make_unique<MockBackend>(TierType::DRAM)));
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
  ASSERT_TRUE(registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());

  auto results = pool.BatchAllocate(
      {PutRequest("a", "dram_b"), PutRequest("bad", "missing"), PutRequest("b", "dram_a")});
  ASSERT_EQ(results.size(), 3u);
  EXPECT_EQ(results[0].backend_id, registry.BackendId(registry.Get("dram_b")));
  EXPECT_EQ(results[0].allocation.outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_EQ(results[1].allocation.outcome, AllocateOutcome::kFailed);
  EXPECT_FALSE(results[1].in_flight);
  EXPECT_EQ(results[2].backend_id, registry.BackendId(registry.Get("dram_a")));
  EXPECT_EQ(results[2].allocation.outcome, AllocateOutcome::kSuccessAllocated);
}

TEST(PeerPool, ReservesDuplicateKeysUntilCommitOrAbort) {
  BackendRegistry registry;
  ASSERT_TRUE(registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());

  auto results = pool.BatchAllocate({PutRequest("same", "dram_a"), PutRequest("same", "dram_b")});
  ASSERT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].allocation.outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_EQ(results[1].allocation.outcome, AllocateOutcome::kFailed);
  EXPECT_TRUE(results[1].in_flight);

  ASSERT_TRUE(
      pool.BatchAbort({PoolSlotRef{results[0].backend_id, results[0].allocation.slot_id}}).front());
  auto retried = pool.BatchAllocate({PutRequest("same", "dram_b")}).front();
  EXPECT_EQ(retried.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_EQ(retried.backend_id, registry.BackendId(registry.Get("dram_b")));
}

TEST(PeerPool, CommitMustMatchReservedSlotAndKey) {
  BackendRegistry registry;
  ASSERT_TRUE(registry.Register("dram", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());

  auto allocated = pool.BatchAllocate({PutRequest("reserved")}).front();
  PoolCommitRequest wrong;
  wrong.slot = PoolSlotRef{allocated.backend_id, allocated.allocation.slot_id};
  wrong.key = "different";
  EXPECT_FALSE(pool.BatchCommit({wrong}).front().commit.success);

  PoolCommitRequest correct = wrong;
  correct.key = "reserved";
  EXPECT_TRUE(pool.BatchCommit({correct}).front().commit.success);
  EXPECT_EQ(pool.PlacementBackend("reserved"), std::optional<uint32_t>{allocated.backend_id});
  EXPECT_FALSE(pool.PlacementBackend("different").has_value());
}

TEST(PeerPool, StalePlacementDoesNotSuppressNewAllocation) {
  BackendRegistry registry;
  ASSERT_TRUE(registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
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
  ASSERT_TRUE(registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(registry.Register("ssd_a", std::make_unique<MockBackend>(TierType::SSD)));
  auto policy = MakeWeightedPlacementPolicy({{"dram_a", 3}, {"dram_b", 7}, {"ssd_a", 100}});

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
  ASSERT_TRUE(registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  auto policy = MakeWeightedPlacementPolicy({{"dram_a", 1}, {"dram_b", 100}});

  auto order = policy->PutOrder(registry, PutRequest("forced", "dram_a"));
  ASSERT_FALSE(order.empty());
  EXPECT_EQ(order.front(), registry.BackendId(registry.Get("dram_a")));
}

TEST(PeerPool, WeightedPolicyPlacesNewKeysOnMultipleBackends) {
  BackendRegistry registry;
  ASSERT_TRUE(registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeWeightedPlacementPolicy({{"dram_a", 1}, {"dram_b", 1}}));

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
  ASSERT_TRUE(registry.Register("dram_a", std::make_unique<NoSpaceBackend>(TierType::DRAM)));
  ASSERT_TRUE(registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
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
  ASSERT_TRUE(registry.Register("only", std::make_unique<MockBackend>(TierType::DRAM)));
  auto compiled = LogicalTierGraph::Compile({LogicalTierConfig{{{"only", 1}}}}, registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  EXPECT_THROW(PeerPool(&registry, MakeTieredPlacementPolicy(compiled.graph)),
               std::invalid_argument);
}

TEST(TieredPlacementPolicy, IgnoresPhysicalRouteAndSpillsToConfiguredTarget) {
  LocalCopyEngine engine;
  BackendRegistry registry;
  ASSERT_TRUE(registry.Register("hot", std::make_unique<NoSpaceBackend>(TierType::HBM)));
  ASSERT_TRUE(registry.Register("cold", std::make_unique<MockBackend>(TierType::SSD)));
  auto compiled = LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kOnEvict), registry);
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
  auto compiled = LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kOnEvict), registry);
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
  GatedEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 2);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 2);
  auto compiled = LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kOnEvict), registry);
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
                                   {allocation.backend_id, allocation.allocation.slot_id}, "move"}})
                  .front()
                  .commit.success);

  // Evict returns while the copy is held, and does not queue it twice.
  auto evicted = pool.Evict({"move"}, PoolEvictMode::kReclaim);
  ASSERT_EQ(evicted.size(), 1u);
  EXPECT_EQ(evicted.front().outcome, PoolEvictOutcome::kDemotionQueued);
  EXPECT_EQ(evicted.front().bytes_freed, 0u);
  ASSERT_TRUE(engine.WaitForHeld(1)) << "the demotion never reached the copy";
  EXPECT_TRUE(registry.Get("hot")->Contains("move"));
  EXPECT_EQ(pool.Evict({"move"}, PoolEvictMode::kReclaim).front().outcome,
            PoolEvictOutcome::kInFlight);

  engine.Release();
  ASSERT_TRUE(WaitForKeyGone(&registry, "hot", "move"));
  EXPECT_TRUE(registry.Get("cold")->Contains("move"));
  EXPECT_EQ(pool.PlacementBackend("move"),
            std::optional<uint32_t>{registry.BackendId(registry.Get("cold"))});

  auto resolved = pool.BatchResolve({"move"}, false).front();
  ASSERT_TRUE(resolved.resolved.found);
  auto target_ref = registry.Get("cold")->BufferRef(resolved.resolved.pages[0].buffer_index);
  ASSERT_TRUE(target_ref.HasHostPtr());
  EXPECT_EQ(std::string(static_cast<char*>(target_ref.host_ptr) +
                            resolved.resolved.pages[0].page_index * kPageSize,
                        payload.size()),
            payload);
}

TEST(PeerPool, EvictionRetryDrainsLeasedSourceWithoutDeletingTarget) {
  constexpr uint64_t kPageSize = 64;
  // Outlives the queued demotion, so its source delete is refused.
  constexpr auto kLease = std::chrono::milliseconds(300);
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 2, kLease);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 2, kLease);
  auto compiled = LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kOnEvict), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  CommitKey(&pool, "leased");
  ASSERT_TRUE(registry.Get("hot")->BatchResolve({"leased"}, false).front().found);

  auto first = pool.Evict({"leased"}, PoolEvictMode::kReclaim);
  ASSERT_EQ(first.front().bytes_freed, 0u);
  ASSERT_TRUE(WaitForKey(&registry, "cold", "leased"));
  EXPECT_TRUE(registry.Get("hot")->Contains("leased")) << "the lease did not hold the source";

  std::this_thread::sleep_for(kLease + std::chrono::milliseconds(50));
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
  auto compiled = LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kOnEvict), registry);
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

// Copy-promotes `key`: both media hold it, the placement is the hot copy, and
// the read's lease on the hot copy has lapsed.
void PromoteByCopy(PeerPool* pool, BackendRegistry* registry, const std::string& key) {
  auto request = PutRequest(key);
  request.logical_tier = "cold";
  auto allocation = pool->BatchAllocate({request}).front();
  ASSERT_EQ(allocation.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_TRUE(pool->BatchCommit({PoolCommitRequest{
                                    {allocation.backend_id, allocation.allocation.slot_id}, key}})
                  .front()
                  .commit.success);
  ASSERT_TRUE(pool->BatchResolve({key}, false).front().resolved.found);
  ASSERT_TRUE(registry->Get("hot")->Contains(key));
  ASSERT_TRUE(registry->Get("cold")->Contains(key));
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

std::shared_ptr<const LogicalTierGraph> CopyPromotingHotCold(BackendRegistry* registry) {
  auto tiers = HotColdTiers(PoolOffloadTrigger::kOnEvict);
  tiers.back().promote_trigger = PoolPromoteTrigger::kOnRead;
  tiers.back().promote_mode = PoolTransitionMode::kCopy;
  auto compiled = LogicalTierGraph::Compile(tiers, *registry);
  EXPECT_TRUE(compiled.ok()) << compiled.error;
  return compiled.graph;
}

TEST(PeerPool, TierScopedEvictFreesOnlyThatMediumsCopy) {
  constexpr uint64_t kPageSize = 64;
  constexpr auto kShortLease = std::chrono::milliseconds(1);
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 4, kShortLease);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 4, kShortLease);
  PeerPool pool(&registry, MakeTieredPlacementPolicy(CopyPromotingHotCold(&registry)), &engine);
  const uint32_t hot = registry.BackendId(registry.Get("hot"));
  const uint32_t cold = registry.BackendId(registry.Get("cold"));
  PromoteByCopy(&pool, &registry, "a");
  PromoteByCopy(&pool, &registry, "b");

  // SSD scope: only the SSD copy goes; the promoted DRAM copy keeps serving.
  auto evicted = pool.Evict({PoolEvictRequest{"a", TierType::SSD}}, PoolEvictMode::kReclaim);
  EXPECT_EQ(evicted.front().bytes_freed, 64u);
  EXPECT_FALSE(registry.Get("cold")->Contains("a"));
  EXPECT_TRUE(registry.Get("hot")->Contains("a"));
  EXPECT_EQ(pool.PlacementBackend("a"), std::optional<uint32_t>{hot});

  // DRAM scope: cold already holds the key, so DRAM frees without a copy.
  evicted = pool.Evict({PoolEvictRequest{"b", TierType::DRAM}}, PoolEvictMode::kReclaim);
  EXPECT_EQ(evicted.front().outcome, PoolEvictOutcome::kDemotionQueued);
  ASSERT_TRUE(WaitForKeyGone(&registry, "hot", "b"));
  EXPECT_EQ(pool.PlacementBackend("b"), std::optional<uint32_t>{cold});
  EXPECT_TRUE(pool.BatchResolve({"b"}, false).front().resolved.found);
}

TEST(PeerPool, ScopedEvictFreesOnlyTheNamedCopy) {
  // The same key on two DRAM instances.
  BackendRegistry registry;
  ASSERT_TRUE(registry.Register("dram_a", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(registry.Register("dram_b", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());
  for (const char* name : {"dram_a", "dram_b"}) {
    auto* backend = registry.Get(name);
    auto allocated = backend->BatchAllocate({AllocateRequest{"k", 32}}).front();
    ASSERT_TRUE(backend->BatchCommit({CommitRequest{allocated.slot_id, "k"}}).front().success);
  }

  // A tier with no copy (stale master metadata) frees nothing.
  auto evicted = pool.Evict({PoolEvictRequest{"k", TierType::SSD}}, PoolEvictMode::kReclaim);
  EXPECT_EQ(evicted.front().bytes_freed, 0u);

  // A backend scope frees only that instance.
  PoolEvictRequest request;
  request.key = "k";
  request.backend_id = registry.BackendId(registry.Get("dram_b"));
  evicted = pool.Evict({request}, PoolEvictMode::kReclaim);
  EXPECT_EQ(evicted.front().bytes_freed, 32u);
  EXPECT_TRUE(registry.Get("dram_a")->Contains("k"));
  EXPECT_FALSE(registry.Get("dram_b")->Contains("k"));
}

TEST(PeerPool, ReclaimDemotesOnAWatermarkTierToo) {
  // Reclaim demotes whenever the tier has a downstream, whatever its trigger.
  constexpr uint64_t kPageSize = 64;
  LocalCopyEngine engine;
  BackendRegistry registry;
  // One key stays far under the watermark, so only the explicit evict moves it.
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 16);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 16);
  auto compiled = LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kWatermark), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);
  CommitKey(&pool, "k");

  auto evicted = pool.Evict({PoolEvictRequest{"k", TierType::DRAM}}, PoolEvictMode::kReclaim);
  EXPECT_EQ(evicted.front().outcome, PoolEvictOutcome::kDemotionQueued);
  // Runs although the tier is far under its own watermark.
  ASSERT_TRUE(WaitForKeyGone(&registry, "hot", "k"));
  EXPECT_TRUE(registry.Get("cold")->Contains("k"))
      << "a watermark tier deleted instead of demoting";
}

TEST(PeerPool, DemotionWithNoRoomDownstreamDropsTheCopy) {
  // Reclaim still frees the tier when the tier below is full (cold: one page).
  constexpr uint64_t kPageSize = 4096;
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 4);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 1);
  auto compiled = LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kOnEvict), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);
  pool.SetEvictDemotionRetryBudgetForTest(std::chrono::milliseconds{50});
  // Fill cold's only page directly.
  auto* cold = registry.Get("cold");
  auto filler = cold->BatchAllocate({AllocateRequest{"filler", kPageSize}}).front();
  ASSERT_EQ(filler.outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_TRUE(cold->BatchCommit({CommitRequest{filler.slot_id, "filler"}}).front().success);
  CommitKey(&pool, "k");

  // The worker finds no room downstream and drops the copy.
  auto evicted = pool.Evict({PoolEvictRequest{"k", TierType::DRAM}}, PoolEvictMode::kReclaim);
  EXPECT_EQ(evicted.front().outcome, PoolEvictOutcome::kDemotionQueued);
  ASSERT_TRUE(WaitForKeyGone(&registry, "hot", "k")) << "a failed demotion left the copy in place";
  EXPECT_FALSE(cold->Contains("k"));
  EXPECT_TRUE(cold->Contains("filler"));
  EXPECT_FALSE(pool.PlacementBackend("k").has_value());
  EXPECT_EQ(pool.TransitionMetrics().dropped, 1u);
}

// A demotion copy that fails is retried, since the evicted copy is the only
// one; it is dropped only once the retry budget runs out, so the tier frees.
TEST(PeerPool, AFailedDemotionIsRetriedUntilItsBudgetRunsOut) {
  constexpr uint64_t kPageSize = 4096;
  FailingEngine engine(/*failures=*/0);
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 4);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 4);
  auto compiled = LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kOnEvict), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);
  CommitKey(&pool, "a");
  CommitKey(&pool, "b");

  engine.FailNext(2);  // the first two copies fail; the third lands
  auto evicted = pool.Evict({PoolEvictRequest{"a", TierType::DRAM}}, PoolEvictMode::kReclaim);
  EXPECT_EQ(evicted.front().outcome, PoolEvictOutcome::kDemotionQueued);
  ASSERT_TRUE(WaitForKeyGone(&registry, "hot", "a"));
  EXPECT_TRUE(registry.Get("cold")->Contains("a")) << "dropped, not demoted";
  EXPECT_EQ(pool.TransitionMetrics().dropped, 0u);
  EXPECT_GE(pool.TransitionMetrics().failed, 2u) << "the failed attempts still count";
  EXPECT_GE(pool.TransitionMetrics().retried, 2u);

  pool.SetEvictDemotionRetryBudgetForTest(std::chrono::milliseconds{50});
  engine.FailNext(1000000);  // every copy fails
  pool.Evict({PoolEvictRequest{"b", TierType::DRAM}}, PoolEvictMode::kReclaim);
  ASSERT_TRUE(WaitForKeyGone(&registry, "hot", "b"));
  EXPECT_FALSE(registry.Get("cold")->Contains("b"));
  EXPECT_FALSE(pool.PlacementBackend("b").has_value());
  EXPECT_EQ(pool.TransitionMetrics().dropped, 1u);
}

// No space at the target (an SSD whose staging is full of reads) is waited
// out, not a reason to drop the only copy.
TEST(PeerPool, ADemotionWaitsOutATargetWithNoRoomForNow) {
  constexpr uint64_t kPageSize = 4096;
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 4);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 1);
  auto compiled = LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kOnEvict), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);
  auto* cold = registry.Get("cold");
  auto filler = cold->BatchAllocate({AllocateRequest{"filler", kPageSize}}).front();
  ASSERT_EQ(filler.outcome, AllocateOutcome::kSuccessAllocated);  // cold's only page, pending
  CommitKey(&pool, "k");

  pool.Evict({PoolEvictRequest{"k", TierType::DRAM}}, PoolEvictMode::kReclaim);
  std::this_thread::sleep_for(std::chrono::milliseconds{100});
  EXPECT_TRUE(registry.Get("hot")->Contains("k")) << "dropped while the target was only busy";

  cold->BatchAbort({filler.slot_id});  // the pressure clears
  ASSERT_TRUE(WaitForKeyGone(&registry, "hot", "k"));
  EXPECT_TRUE(cold->Contains("k"));
  EXPECT_EQ(pool.TransitionMetrics().dropped, 0u);
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

  auto compiled = LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kWatermark), registry);
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
      {parallel, LogicalTierConfig{{{"cold", 1}}, {}, PoolOffloadTrigger::kOnEvict}}, registry);
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

  auto compiled = LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kWatermark), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);

  CommitKey(&pool, "blocked");
  ASSERT_TRUE(blocking->WaitUntilEntered());

  // The transition is parked mid-copy. An unrelated resolve must not wait for
  // it, which is only true while the copy runs without the pool lock.
  auto concurrent = std::async(std::launch::async,
                               [&] { return pool.BatchResolve({"unrelated"}, false).size(); });
  EXPECT_EQ(concurrent.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  blocking->Release();
  EXPECT_EQ(concurrent.get(), 1u);
}

TEST(PeerPool, SlowResolveDoesNotBlockUnrelatedPreferredBackend) {
  BackendRegistry registry;
  auto slow = std::make_unique<BlockingResolveBackend>(TierType::SSD);
  auto* blocking = slow.get();
  ASSERT_TRUE(registry.Register("slow", std::move(slow)));
  ASSERT_TRUE(registry.Register("fast", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());
  CommitKey(&pool, "fast-key", "fast");

  auto slow_resolve =
      std::async(std::launch::async, [&] { return pool.BatchResolve({"missing"}, false); });
  ASSERT_TRUE(blocking->WaitUntilEntered());

  // The preferred placement resolves entirely on the fast backend. It must
  // not queue behind unrelated IO already running on another medium.
  auto fast_resolve =
      std::async(std::launch::async, [&] { return pool.BatchResolve({"fast-key"}, false); });
  const auto status = fast_resolve.wait_for(std::chrono::seconds(2));
  blocking->Release();

  EXPECT_EQ(status, std::future_status::ready);
  auto fast_result = fast_resolve.get();
  ASSERT_EQ(fast_result.size(), 1u);
  EXPECT_TRUE(fast_result.front().resolved.found);
  EXPECT_EQ(slow_resolve.get().size(), 1u);
}

TEST(PeerPool, SlowCommitDoesNotBlockLookupsOrReads) {
  BackendRegistry registry;
  auto slow = std::make_unique<BlockingCommitBackend>(TierType::DRAM);
  auto* blocking = slow.get();
  ASSERT_TRUE(registry.Register("slow", std::move(slow)));
  ASSERT_TRUE(registry.Register("fast", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());
  CommitKey(&pool, "fast-key", "fast");

  auto allocation = pool.BatchAllocate({PutRequest("slow-key", "slow")}).front();
  ASSERT_EQ(allocation.allocation.outcome, AllocateOutcome::kSuccessAllocated);
  auto commit = std::async(std::launch::async, [&] {
    return pool.BatchCommit(
        {PoolCommitRequest{{allocation.backend_id, allocation.allocation.slot_id}, "slow-key"}});
  });
  ASSERT_TRUE(blocking->WaitUntilEntered());

  auto lookups = std::async(std::launch::async, [&] {
    return std::make_tuple(pool.BatchContains({"fast-key"}), pool.BatchResolve({"fast-key"}, false),
                           pool.BatchAllocate({PutRequest("slow-key", "slow")}).front());
  });
  const auto status = lookups.wait_for(std::chrono::seconds(2));
  blocking->Release();

  EXPECT_EQ(status, std::future_status::ready);
  const auto [contains, resolved, second_put] = lookups.get();
  EXPECT_TRUE(contains.front());
  EXPECT_TRUE(resolved.front().resolved.found);
  EXPECT_TRUE(second_put.in_flight);  // the committing key stays reserved
  EXPECT_TRUE(commit.get().front().commit.success);
  EXPECT_EQ(pool.PlacementBackend("slow-key"),
            std::optional<uint32_t>{registry.BackendId(registry.Get("slow"))});
}

TEST(PeerPool, ResolveMissDoesNotEraseConcurrentCommit) {
  BackendRegistry registry;
  auto slow = std::make_unique<BlockingResolveBackend>(TierType::SSD);
  auto* blocking = slow.get();
  ASSERT_TRUE(registry.Register("slow", std::move(slow)));
  ASSERT_TRUE(registry.Register("fast", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());

  auto stale_resolve =
      std::async(std::launch::async, [&] { return pool.BatchResolve({"raced"}, false); });
  ASSERT_TRUE(blocking->WaitUntilEntered());

  auto concurrent_commit = std::async(std::launch::async, [&] {
    auto allocation = pool.BatchAllocate({PutRequest("raced", "fast")}).front();
    bool committed = false;
    if (allocation.allocation.outcome == AllocateOutcome::kSuccessAllocated) {
      committed =
          pool.BatchCommit({PoolCommitRequest{
                               {allocation.backend_id, allocation.allocation.slot_id}, "raced"}})
              .front()
              .commit.success;
    }
    return std::make_pair(allocation, committed);
  });
  const auto status = concurrent_commit.wait_for(std::chrono::seconds(2));
  blocking->Release();

  EXPECT_EQ(status, std::future_status::ready);
  const auto [allocation, committed] = concurrent_commit.get();
  EXPECT_TRUE(committed);
  EXPECT_EQ(allocation.backend_id, registry.BackendId(registry.Get("fast")));
  EXPECT_EQ(stale_resolve.get().size(), 1u);
  EXPECT_EQ(pool.PlacementBackend("raced"),
            std::optional<uint32_t>{registry.BackendId(registry.Get("fast"))});
}

TEST(PeerPool, ClearWaitsForInFlightResolveAndLeavesNoStalePlacement) {
  BackendRegistry registry;
  auto backend = std::make_unique<BlockingResolveBackend>(TierType::DRAM);
  auto* blocking = backend.get();
  ASSERT_TRUE(registry.Register("dram", std::move(backend)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());
  CommitKey(&pool, "clear-race", "dram");

  auto resolve =
      std::async(std::launch::async, [&] { return pool.BatchResolve({"clear-race"}, false); });
  ASSERT_TRUE(blocking->WaitUntilEntered());

  auto clear = std::async(std::launch::async, [&] { pool.ClearLocal(); });
  EXPECT_EQ(clear.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);
  blocking->Release();

  ASSERT_TRUE(resolve.get().front().resolved.found);
  clear.get();
  EXPECT_EQ(pool.PlacementBackend("clear-race"), std::nullopt);
  EXPECT_FALSE(registry.Get("dram")->Contains("clear-race"));
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
  pool.ClearLocal();  // drains the in-flight copy
  // Only the source is charged the read, not the target's post-write pin.
  const TierTransitionMetrics moves = pool.TransitionMetrics();
  EXPECT_EQ(moves.read_bytes, (std::map<std::string, uint64_t>{{"SSD", 64}}));
  EXPECT_EQ(moves.write_bytes, (std::map<std::string, uint64_t>{{"DRAM", 64}}));
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
  EXPECT_FALSE(registry.Get("hot")->Contains("warm"))
      << "promoted after " << kHits - 1 << " reads, below the threshold of " << kHits;

  ASSERT_TRUE(pool.BatchResolve({"warm"}, false).front().resolved.found);
  EXPECT_TRUE(WaitForKey(&registry, "hot", "warm"));

  // The other cold key was read zero times and must not have been dragged along
  // by a count kept per tier instead of per key.
  EXPECT_FALSE(registry.Get("hot")->Contains("bystander"));
  EXPECT_TRUE(registry.Get("cold")->Contains("bystander"));
}

// The move half of promote_mode: the source goes away and the bytes survive
// the hop, since the promoted copy is then the only one.
TEST(PeerPool, ReadPromotesWithMoveDropsSourceAndKeepsBytes) {
  constexpr uint64_t kPageSize = 64;
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 2);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 2);

  auto tiers = HotColdTiers(PoolOffloadTrigger::kOnEvict);
  tiers.back().promote_trigger = PoolPromoteTrigger::kOnRead;
  tiers.back().promote_mode = PoolTransitionMode::kMove;
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
  ASSERT_TRUE(
      pool.BatchCommit(
              {PoolCommitRequest{{allocation.backend_id, allocation.allocation.slot_id}, "moved"}})
          .front()
          .commit.success);

  auto resolved = pool.BatchResolve({"moved"}, false).front();
  ASSERT_TRUE(resolved.resolved.found);
  EXPECT_EQ(resolved.backend_id, registry.BackendId(registry.Get("hot")));
  EXPECT_FALSE(registry.Get("cold")->Contains("moved"));
  EXPECT_EQ(pool.PlacementBackend("moved"),
            std::optional<uint32_t>{registry.BackendId(registry.Get("hot"))});
  auto target_ref = registry.Get("hot")->BufferRef(resolved.resolved.pages[0].buffer_index);
  ASSERT_TRUE(target_ref.HasHostPtr());
  EXPECT_EQ(std::string(static_cast<char*>(target_ref.host_ptr) +
                            resolved.resolved.pages[0].page_index * kPageSize,
                        payload.size()),
            payload);
}

// A promoting read lands the key in the hot tier and is served from there, even
// with the hot tier above its high watermark: that headroom is for this.
TEST(PeerPool, ReadPromotesAboveHighWatermarkAndServesFromHotTier) {
  constexpr uint64_t kPageSize = 64;
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kPageSize, 4);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kPageSize, 4);
  PeerPool pool(&registry, MakeTieredPlacementPolicy(CopyPromotingHotCold(&registry)), &engine);

  CommitKey(&pool, "a");
  CommitKey(&pool, "b");  // hot is half full, above its 0.4 high watermark
  CommitKey(&pool, "promote", {}, "cold");

  const auto read = pool.BatchResolve({"promote"}, false).front();
  ASSERT_TRUE(read.resolved.found);
  EXPECT_EQ(read.backend_id, registry.BackendId(registry.Get("hot")));
  EXPECT_TRUE(registry.Get("cold")->Contains("promote"));  // copy mode keeps the source
}

// ---- Local watermark eviction (frees through Evict(), as above) ----

bool WaitUntil(const std::function<bool()>& done,
               std::chrono::milliseconds budget = std::chrono::seconds(5)) {
  const auto deadline = std::chrono::steady_clock::now() + budget;
  while (std::chrono::steady_clock::now() < deadline) {
    if (done()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return done();
}

// Host-page-sized pages, so a tier is exactly N pages; every key is one page.
constexpr uint64_t kHostPage = 4096;

TEST(PeerPoolLocalEviction, DrainsAFullBackendColdestFirstToItsLowWatermark) {
  LocalCopyEngine engine;
  BackendRegistry registry;
  // 1 ms lease: lapsed by the time the worker looks.
  RegisterPageBackend(&registry, &engine, "dram", TierType::DRAM, kHostPage, 10,
                      std::chrono::milliseconds(1));
  PeerPool pool(&registry, MakeSingleBackendPolicy(), &engine);
  // Enabled before any commit, as PoolClient does.
  ASSERT_TRUE(pool.EnableLocalEviction(registry.BackendId(registry.Get("dram")), 0.9, 0.5));

  for (int i = 0; i < 8; ++i) CommitKey(&pool, "k" + std::to_string(i));
  // Reading the oldest key renews its recency.
  ASSERT_TRUE(pool.BatchResolve({"k0"}, false).front().resolved.found);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  CommitKey(&pool, "k8");

  // 9 of 10 pages hits the high watermark; the worker frees exactly down to 5.
  auto* dram = registry.Get("dram");
  ASSERT_TRUE(WaitUntil([&] { return dram->OwnedKeyCount() <= 5; }));
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(dram->OwnedKeyCount(), 5u);
  EXPECT_TRUE(dram->Contains("k0")) << "a just-read key was evicted ahead of colder ones";
  for (int i = 1; i <= 4; ++i) EXPECT_FALSE(dram->Contains("k" + std::to_string(i))) << i;
  for (int i = 5; i <= 8; ++i) EXPECT_TRUE(dram->Contains("k" + std::to_string(i))) << i;
  EXPECT_EQ(pool.PlacementCount(), 5u);
  EXPECT_GE(pool.LocalEvictionStats().rounds, 1u);
  EXPECT_EQ(pool.LocalEvictionStats().keys, 4u);
}

TEST(PeerPoolLocalEviction, RankedCandidatesGoFirstAndTheMediumCoversTheRest) {
  // The source (the master, in peer-driven mode) ranks differently from the
  // medium's LRU, names a key this backend no longer holds, and reports value
  // sizes rather than the whole pages the budget counts.
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "dram", TierType::DRAM, kHostPage, 10,
                      std::chrono::milliseconds(1));
  PeerPool pool(&registry, MakeSingleBackendPolicy(), &engine);
  std::atomic<int> queries{0};
  std::atomic<uint64_t> asked_bytes{0};
  pool.SetEvictionCandidateSource([&](TierType tier, const std::string&, uint64_t bytes) {
    ++queries;
    asked_bytes = tier == TierType::DRAM ? bytes : 0;
    return std::optional<std::vector<EvictionOffer>>({{"k8", 64}, {"gone", 64}, {"k7", 64}});
  });
  ASSERT_TRUE(pool.EnableLocalEviction(registry.BackendId(registry.Get("dram")), 0.9, 0.5));

  for (int i = 0; i < 9; ++i) CommitKey(&pool, "k" + std::to_string(i));
  auto* dram = registry.Get("dram");
  ASSERT_TRUE(WaitUntil([&] { return dram->OwnedKeyCount() <= 5; }));
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(dram->OwnedKeyCount(), 5u) << "evicted past the overage";
  EXPECT_EQ(queries.load(), 1);
  EXPECT_EQ(asked_bytes.load(), 4 * kHostPage);  // 9 pages down to 5
  // Ranked k8, k7 first; then the medium's coldest, k0 and k1.
  for (const char* gone : {"k8", "k7", "k0", "k1"}) EXPECT_FALSE(dram->Contains(gone)) << gone;
  for (int i = 2; i <= 6; ++i) EXPECT_TRUE(dram->Contains("k" + std::to_string(i))) << i;
  EXPECT_EQ(pool.LocalEvictionStats().master_keys, 2u);

  // A source slower than a pass waits for (a master with a large index): the
  // medium's order covers the overage meanwhile.
  std::atomic<bool> answered{false};
  pool.SetEvictionCandidateSource([&](TierType, const std::string&, uint64_t) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    answered = true;
    return std::optional<std::vector<EvictionOffer>>({{"k2", 64}});
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(250));  // past the query interval
  for (int i = 9; i < 13; ++i) CommitKey(&pool, "k" + std::to_string(i));
  ASSERT_TRUE(WaitUntil([&] { return dram->OwnedKeyCount() <= 5; }));
  EXPECT_FALSE(answered.load()) << "the pass waited on the source";
  EXPECT_EQ(pool.LocalEvictionStats().master_keys, 2u);
}

TEST(PeerPoolLocalEviction, KeysUnderAReadLeaseAreNeverOffered) {
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "dram", TierType::DRAM, kHostPage, 10);
  PeerPool pool(&registry, MakeSingleBackendPolicy(), &engine);
  std::vector<std::string> keys;
  for (int i = 0; i < 9; ++i) {
    keys.push_back("k" + std::to_string(i));
    CommitKey(&pool, keys.back());
  }
  // Every key is being read (the default lease outlives the test).
  for (const auto& entry : pool.BatchResolve(keys, false)) ASSERT_TRUE(entry.resolved.found);

  ASSERT_TRUE(pool.EnableLocalEviction(registry.BackendId(registry.Get("dram")), 0.9, 0.5));
  EXPECT_EQ(pool.RunLocalEvictionOnceForTest(), 0u);
  EXPECT_EQ(registry.Get("dram")->OwnedKeyCount(), 9u);
  EXPECT_GE(pool.LocalEvictionStats().no_candidate, 1u);
}

TEST(PeerPoolLocalEviction, ATierWithADownstreamDemotesJustTheOverage) {
  // A tier with a downstream demotes, and passes that run while copies are
  // held count them rather than queueing the whole tier.
  GatedEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kHostPage, 10);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kHostPage, 20);
  auto compiled = LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kOnEvict), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);
  ASSERT_TRUE(pool.EnableLocalEviction(registry.BackendId(registry.Get("hot")), 0.9, 0.5));

  for (int i = 0; i < 9; ++i) CommitKey(&pool, "k" + std::to_string(i));
  auto* hot = registry.Get("hot");
  auto* cold = registry.Get("cold");
  ASSERT_TRUE(engine.WaitForHeld(1)) << "no demotion was queued";
  for (int i = 0; i < 3; ++i) pool.RunLocalEvictionOnceForTest();
  std::this_thread::sleep_for(std::chrono::milliseconds(250));  // let polls run too
  EXPECT_EQ(pool.LocalEvictionStats().keys, 4u) << "queued past the overage";
  EXPECT_EQ(hot->OwnedKeyCount(), 9u) << "a copy completed while held";

  engine.Release();
  ASSERT_TRUE(WaitUntil([&] { return hot->OwnedKeyCount() <= 5; }));
  for (int i = 0; i < 4; ++i) {
    const std::string key = "k" + std::to_string(i);
    EXPECT_FALSE(hot->Contains(key)) << key;
    EXPECT_TRUE(cold->Contains(key)) << key << " was dropped, not demoted";
    EXPECT_TRUE(pool.BatchResolve({key}, false).front().resolved.found) << key;
  }
}

class CountingPageBackend final : public PageBackend {
 public:
  using PageBackend::PageBackend;
  std::vector<EvictionOffer> EvictionCandidates(size_t max_count) override {
    ++calls;
    return PageBackend::EvictionCandidates(max_count);
  }
  std::atomic<int> calls{0};
};

TEST(PeerPoolLocalEviction, ARoundDoesNotRescanItsQueuedDemotionsPerBatch) {
  // Held demotions stay offered; refetching them for every 64-key batch made a round quadratic.
  GatedEngine engine;
  BackendRegistry registry;
  PageBackend::OwnershipConfig ownership;
  ownership.buffer_sizes = {8192 * kHostPage};
  auto counting = std::make_unique<CountingPageBackend>(TierType::DRAM, kHostPage, ownership,
                                                        kNoExpiry, kNoExpiry);
  auto* hot = counting.get();
  ASSERT_TRUE(hot->Init(&engine));
  ASSERT_TRUE(registry.Register("hot", std::move(counting)));
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kHostPage, 8192);
  auto compiled = LogicalTierGraph::Compile(HotColdTiers(PoolOffloadTrigger::kOnEvict), registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);
  hot->TrackEvictionOrder();
  for (int i = 0; i < 8000; ++i) CommitKey(&pool, "k" + std::to_string(i));

  ASSERT_TRUE(pool.EnableLocalEviction(registry.BackendId(hot), 0.9, 0.5));
  const int before = hot->calls;
  pool.RunLocalEvictionOnceForTest();
  EXPECT_EQ(pool.LocalEvictionStats().keys, 8000u - 4096u);
  EXPECT_LE(hot->calls - before, 40);  // ~13 now; ~67 when it refetched per batch
  engine.Release();
}

TEST(PeerPoolLocalEviction, NeverDeletesWhatAWatermarkOffloadWouldMove) {
  // Regression: the old in-commit round deleted keys offload was about to move.
  LocalCopyEngine engine;
  BackendRegistry registry;
  RegisterPageBackend(&registry, &engine, "hot", TierType::DRAM, kHostPage, 10);
  RegisterPageBackend(&registry, &engine, "cold", TierType::SSD, kHostPage, 40);
  auto tiers = HotColdTiers(PoolOffloadTrigger::kWatermark);
  tiers.front().high_watermark = 0.9;
  tiers.front().low_watermark = 0.5;
  auto compiled = LogicalTierGraph::Compile(tiers, registry);
  ASSERT_TRUE(compiled.ok()) << compiled.error;
  PeerPool pool(&registry, MakeTieredPlacementPolicy(compiled.graph), &engine);
  // The same watermarks as the offload: the case that used to lose keys.
  ASSERT_TRUE(pool.EnableLocalEviction(registry.BackendId(registry.Get("hot")), 0.9, 0.5));

  std::vector<std::string> keys;
  for (int i = 0; i < 20; ++i) {
    keys.push_back("k" + std::to_string(i));
    CommitKey(&pool, keys.back());
    // Let the workers keep up, as a real put rate would.
    ASSERT_TRUE(WaitUntil([&] { return registry.Get("hot")->OwnedKeyCount() < 9; }));
  }
  for (const auto& key : keys) {
    EXPECT_TRUE(registry.Get("hot")->Contains(key) || registry.Get("cold")->Contains(key))
        << key << " was deleted although the cold tier had room for it";
  }
}

TEST(PeerPoolLocalEviction, RejectsBadConfigAndLeavesAnUnorderedMediumAlone) {
  BackendRegistry registry;
  ASSERT_TRUE(registry.Register("dram", std::make_unique<MockBackend>(TierType::DRAM)));
  PeerPool pool(&registry, MakeSingleBackendPolicy());
  const uint32_t id = registry.BackendId(registry.Get("dram"));
  EXPECT_FALSE(pool.EnableLocalEviction(id, 0.5, 0.7));  // low above high
  EXPECT_FALSE(pool.EnableLocalEviction(id, 1.5, 0.7));  // high above 1
  EXPECT_FALSE(pool.EnableLocalEviction(id, 0.9, 0.0));  // low not positive
  EXPECT_FALSE(pool.EnableLocalEviction(BackendRegistry::kMaxBackends - 1, 0.9, 0.7));

  // MockBackend offers no candidates, so it must be left alone.
  CommitKey(&pool, "k");
  ASSERT_TRUE(pool.EnableLocalEviction(id, 0.9, 0.5));
  pool.RunLocalEvictionOnceForTest();
  EXPECT_TRUE(registry.Get("dram")->Contains("k"));
}

}  // namespace
}  // namespace mori::umbp
