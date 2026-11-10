// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include <gtest/gtest.h>

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "umbp/distributed/peer/backend/engine_hbm_backend.h"
#include "umbp/distributed/peer/backend/instrumented_backend.h"
#include "umbp/distributed/pool/peer_pool.h"

namespace mori::umbp {
namespace {

engine::OperationId Op(uint64_t seq) {
  engine::OperationId id;
  id.session_id[0] = 4;
  id.seq = seq;
  return id;
}

class EngineFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    a_.resize(5 * 4);
    b_.resize(5 * 6);
    engine::PageGeometry geometry;
    geometry.page_tokens = 256;
    geometry.num_pages = 4;
    geometry.object_bytes = 10;
    geometry.regions = {
        {"a", TransferRef::HostBytes(a_.data(), a_.size()), 2, 2, 0, 0, "kv"},
        {"b", TransferRef::HostBytes(b_.data(), b_.size()), 3, 2, 4, 1, "rope"},
    };
    geometry.layout_id = engine::ComputeLayoutId(geometry);
    engine::PoolEpoch epoch;
    epoch.boot_id[0] = 1;
    backend_ = std::make_unique<EngineHbmBackend>(
        EngineRegistration{"pool", epoch, 2, 7, std::move(geometry)});
    ASSERT_TRUE(backend_->Init(nullptr));
  }

  engine::AllocationHandle Publish(const std::string& key, uint64_t generation = 1) {
    auto delegated = backend_->Delegate(7, 1, Op(1));
    EXPECT_TRUE(delegated.success);
    auto handle = delegated.handles[0];
    handle.generation = generation;
    auto published = backend_->PublishResident({{key, {}, handle}}, Op(2));
    EXPECT_TRUE(published.success);
    return handle;
  }

  std::vector<uint8_t> a_, b_;
  std::unique_ptr<EngineHbmBackend> backend_;
};

class StartBarrier {
 public:
  explicit StartBarrier(size_t participants) : participants_(participants) {}
  void ArriveAndWait() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (++arrived_ == participants_) {
      released_ = true;
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [&] { return released_; });
  }

 private:
  const size_t participants_;
  size_t arrived_ = 0;
  bool released_ = false;
  std::mutex mutex_;
  std::condition_variable cv_;
};

TEST_F(EngineFixture, PageZeroIsNeverDelegatedAndGeometryProducesAllSpans) {
  auto delegated = backend_->Delegate(7, 4, Op(1));
  ASSERT_TRUE(delegated.success);
  ASSERT_EQ(delegated.handles.size(), 4u);
  EXPECT_EQ(delegated.handles.front().page_id, 1u);
  EXPECT_EQ(delegated.handles.back().page_id, 4u);
  const auto view = backend_->View(delegated.handles.front());
  ASSERT_TRUE(ValidateObjectView(view));
  ASSERT_EQ(view.spans.size(), 2u);
  EXPECT_EQ(view.spans[0].offset, 4u);
  EXPECT_EQ(view.spans[1].offset, 6u);
  EXPECT_EQ(view.layout_id.substr(0, 15), "engine-page-v1-");
}

TEST_F(EngineFixture, LayoutDigestDoesNotEncodePoolCapacityOrAddresses) {
  engine::PageGeometry one;
  one.page_tokens = 256;
  one.num_pages = 4;
  one.object_bytes = 10;
  one.regions = {
      {"a", TransferRef::HostBytes(a_.data(), a_.size()), 2, 2, 0, 0, "kv"},
      {"b", TransferRef::HostBytes(b_.data(), b_.size()), 3, 2, 4, 1, "rope"},
  };
  auto other = one;
  other.num_pages = 99;
  other.regions[0].buffer = TransferRef::HostBytes(reinterpret_cast<void*>(0x1000), 1000);
  other.regions[1].buffer = TransferRef::HostBytes(reinterpret_cast<void*>(0x2000), 1000);
  EXPECT_EQ(engine::ComputeLayoutId(one), engine::ComputeLayoutId(other));
}

TEST_F(EngineFixture, PublishAcquireRetireAndReleasedAreAuthoritative) {
  auto handle = Publish("key");
  auto grant = backend_->Acquire({"reader", "request"}, {handle}, Op(3),
                                 GrantTable::Clock::now() + std::chrono::seconds(1));
  ASSERT_TRUE(grant[0].granted);
  auto ticket = backend_->Retire({handle}, Op(4));
  EXPECT_EQ(ticket.state, engine::RetireState::kWaitingHolds);
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kRetiring);
  EXPECT_FALSE(backend_
                   ->Acquire({"late", "request"}, {handle}, Op(5),
                             GrantTable::Clock::now() + std::chrono::seconds(1))[0]
                   .granted);
  EXPECT_TRUE(backend_->PollReadyRetires().empty());

  backend_->Release({"reader", "request"}, {grant[0].id}, engine::CompletionProof::kCompleted,
                    Op(6));
  auto ready = backend_->PollReadyRetires();
  ASSERT_EQ(ready.size(), 1u);
  EXPECT_EQ(ready[0].state, engine::RetireState::kReady);
  EXPECT_TRUE(backend_->PollReadyRetires().empty());
  EXPECT_TRUE(backend_->AcknowledgeReleased(Op(4)));
  EXPECT_TRUE(backend_->AcknowledgeReleased(Op(4)));  // duplicate ACK is inert
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kDelegated);
  EXPECT_FALSE(backend_->Contains("key"));

  auto events = backend_->DrainPendingEvents();
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].kind, KvEvent::Kind::ADD);
  EXPECT_EQ(events[1].kind, KvEvent::Kind::REMOVE);
}

TEST_F(EngineFixture, GenerationMaySkipButOldAndDuplicateKeysAreRejectedAtomically) {
  auto delegated = backend_->Delegate(7, 2, Op(1));
  ASSERT_TRUE(delegated.success);
  auto first = delegated.handles[0];
  auto second = delegated.handles[1];
  first.generation = 9;
  second.generation = 3;
  auto duplicate = backend_->PublishResident({{"same", {}, first}, {"same", {}, second}}, Op(2));
  EXPECT_FALSE(duplicate.success);
  EXPECT_EQ(backend_->State(first.page_id), engine::SlotState::kDelegated);
  EXPECT_EQ(backend_->State(second.page_id), engine::SlotState::kDelegated);

  EXPECT_TRUE(backend_->PublishResident({{"first", {}, first}}, Op(3)).success);
  auto stale = second;
  stale.generation = 0;
  EXPECT_FALSE(backend_->PublishResident({{"second", {}, stale}}, Op(4)).success);
  EXPECT_EQ(backend_->State(second.page_id), engine::SlotState::kDelegated);
}

TEST_F(EngineFixture, DuplicateDelegateAndPublishOperationsReplaySameResult) {
  auto delegated = backend_->Delegate(7, 1, Op(10));
  auto replay = backend_->Delegate(7, 1, Op(10));
  ASSERT_TRUE(delegated.success);
  ASSERT_TRUE(replay.success);
  EXPECT_EQ(replay.handles, delegated.handles);
  auto handle = delegated.handles[0];
  handle.generation = 5;
  auto published = backend_->PublishResident({{"key", {}, handle}}, Op(11));
  auto published_replay = backend_->PublishResident({{"key", {}, handle}}, Op(11));
  EXPECT_TRUE(published.success);
  EXPECT_TRUE(published_replay.success);
  EXPECT_EQ(backend_->OwnedKeyCount(), 1u);
  EXPECT_EQ(backend_->DrainPendingEvents().size(), 1u);
}

TEST_F(EngineFixture, AcquireReplayUsesOriginalShapeAndBecomesTerminalAfterRelease) {
  auto handle = Publish("replay");
  const auto deadline = GrantTable::Clock::now() + std::chrono::seconds(1);
  auto first = backend_->Acquire({"reader", "req"}, {handle}, Op(3), deadline);
  auto before_retire = backend_->Acquire({"reader", "req"}, {handle}, Op(3), deadline);
  ASSERT_TRUE(first[0].granted);
  ASSERT_TRUE(before_retire[0].granted);
  EXPECT_EQ(before_retire[0].id, first[0].id);

  EXPECT_EQ(backend_->Retire({handle}, Op(4)).state, engine::RetireState::kWaitingHolds);
  auto after_retire = backend_->Acquire({"reader", "req"}, {handle}, Op(3), deadline);
  ASSERT_EQ(after_retire.size(), 1u);
  EXPECT_TRUE(after_retire[0].granted);
  EXPECT_EQ(after_retire[0].id, first[0].id);

  backend_->Release({"reader", "req"}, {first[0].id}, engine::CompletionProof::kCompleted, Op(5));
  auto after_release = backend_->Acquire({"reader", "req"}, {handle}, Op(3), deadline);
  ASSERT_EQ(after_release.size(), 1u);
  EXPECT_FALSE(after_release[0].granted);
  EXPECT_EQ(backend_->PollReadyRetires().size(), 1u);
}

TEST_F(EngineFixture, AcquireRetireAndLastReleaseShareOneAcceptanceDomain) {
  auto handle = Publish("interleaved");
  const auto deadline = GrantTable::Clock::now() + std::chrono::seconds(1);
  auto old_grant = backend_->Acquire({"old", "req"}, {handle}, Op(3), deadline);
  ASSERT_TRUE(old_grant[0].granted);

  StartBarrier barrier(3);
  std::vector<GrantResult> new_grant;
  engine::RetireTicket ticket;
  std::thread acquire([&] {
    barrier.ArriveAndWait();
    new_grant = backend_->Acquire({"new", "req"}, {handle}, Op(4), deadline);
  });
  std::thread retire([&] {
    barrier.ArriveAndWait();
    ticket = backend_->Retire({handle}, Op(5));
  });
  std::thread release([&] {
    barrier.ArriveAndWait();
    backend_->Release({"old", "req"}, {old_grant[0].id}, engine::CompletionProof::kCompleted,
                      Op(6));
  });
  acquire.join();
  retire.join();
  release.join();

  ASSERT_EQ(new_grant.size(), 1u);
  if (new_grant[0].granted) {
    EXPECT_EQ(ticket.state, engine::RetireState::kWaitingHolds);
    EXPECT_TRUE(backend_->PollReadyRetires().empty());
    backend_->Release({"new", "req"}, {new_grant[0].id}, engine::CompletionProof::kCompleted,
                      Op(7));
  }
  auto ready = backend_->PollReadyRetires();
  ASSERT_EQ(ready.size(), 1u);
  EXPECT_TRUE(backend_->AcknowledgeReleased(Op(5)));
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kDelegated);
}

TEST_F(EngineFixture, PublishAndRetireRejectDuplicatePhysicalSlotsAtomically) {
  auto delegated = backend_->Delegate(7, 1, Op(1));
  ASSERT_TRUE(delegated.success);
  auto handle = delegated.handles[0];
  handle.generation = 1;
  auto duplicate_slot =
      backend_->PublishResident({{"first", {}, handle}, {"second", {}, handle}}, Op(2));
  EXPECT_FALSE(duplicate_slot.success);
  EXPECT_FALSE(backend_->Contains("first"));
  EXPECT_FALSE(backend_->Contains("second"));
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kDelegated);

  ASSERT_TRUE(backend_->PublishResident({{"first", {}, handle}}, Op(3)).success);
  auto duplicate_retire = backend_->Retire({handle, handle}, Op(4));
  EXPECT_EQ(duplicate_retire.state, engine::RetireState::kQuarantined);
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kPublished);
  EXPECT_TRUE(backend_->Contains("first"));
}

TEST_F(EngineFixture, StaleQuarantineCannotAffectReusedGeneration) {
  auto old = Publish("old", 1);
  ASSERT_EQ(backend_->Retire({old}, Op(3)).state, engine::RetireState::kReady);
  ASSERT_EQ(backend_->PollReadyRetires().size(), 1u);
  ASSERT_TRUE(backend_->AcknowledgeReleased(Op(3)));
  auto current = old;
  current.generation = 2;
  ASSERT_TRUE(backend_->PublishResident({{"current", {}, current}}, Op(4)).success);

  EXPECT_FALSE(backend_->Quarantine({old}));
  EXPECT_EQ(backend_->State(current.page_id), engine::SlotState::kPublished);
  EXPECT_TRUE(backend_->Contains("current"));
  EXPECT_EQ(backend_->HandleForKey("current"), current);
}

TEST_F(EngineFixture, DelegatedQuarantineRequiresRanksStrictlyNewerGeneration) {
  auto delegated = backend_->Delegate(7, 1, Op(1));
  ASSERT_TRUE(delegated.success);
  auto stale = delegated.handles[0];
  EXPECT_FALSE(backend_->Quarantine({stale}));
  auto in_use = stale;
  ++in_use.generation;
  EXPECT_TRUE(backend_->Quarantine({in_use}));
  EXPECT_EQ(backend_->State(in_use.page_id), engine::SlotState::kQuarantined);
}

TEST_F(EngineFixture, UnknownGrantCompletionQuarantinesAndLogicalClearDoesNotRevive) {
  auto handle = Publish("unsafe");
  auto grant = backend_->Acquire({"reader", "req"}, {handle}, Op(3),
                                 GrantTable::Clock::now() + std::chrono::seconds(1));
  backend_->Release({"reader", "req"}, {grant[0].id}, engine::CompletionProof::kUnknown, Op(4));
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kQuarantined);
  backend_->ClearLocal();
  backend_->ClearFullSyncAcked();
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kQuarantined);
  EXPECT_FALSE(backend_->Delegate(7, 4, Op(5)).success);
}

TEST_F(EngineFixture, ExpiredGrantQuarantinesInsteadOfAuthorizingReuse) {
  auto handle = Publish("expired");
  const auto now = GrantTable::Clock::now();
  auto grant = backend_->Acquire({"reader", "req"}, {handle}, Op(3), now);
  ASSERT_TRUE(grant[0].granted);
  auto expired = backend_->ExpireGrants(now);
  ASSERT_EQ(expired.size(), 1u);
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kQuarantined);
  EXPECT_TRUE(backend_->PollReadyRetires().empty());
}

TEST_F(EngineFixture, ShutdownIsTerminalAndPreservesQuarantineAndGeneration) {
  auto handle = Publish("shutdown", 7);
  auto grant = backend_->Acquire({"reader", "req"}, {handle}, Op(3),
                                 GrantTable::Clock::now() + std::chrono::seconds(1));
  ASSERT_TRUE(grant[0].granted);
  backend_->Release({"reader", "req"}, {grant[0].id}, engine::CompletionProof::kUnknown, Op(4));
  ASSERT_EQ(backend_->State(handle.page_id), engine::SlotState::kQuarantined);

  backend_->Shutdown();
  EXPECT_FALSE(backend_->Init(nullptr));
  EXPECT_FALSE(backend_->Delegate(7, 1, Op(5)).success);
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kQuarantined);
  EXPECT_EQ(backend_->HistoryStats().grants.quarantined_sources, 1u);
  EXPECT_EQ(backend_->Capacity().total_bytes, 0u);
}

TEST_F(EngineFixture, AckWatermarkKeepsOperationHistoryAndReadyChecksBounded) {
  auto delegated = backend_->Delegate(7, 1, Op(1));
  ASSERT_TRUE(delegated.success);
  auto handle = delegated.handles[0];
  constexpr uint64_t kCycles = 256;
  uint64_t seq = 2;
  for (uint64_t cycle = 0; cycle < kCycles; ++cycle) {
    ++handle.generation;
    ASSERT_TRUE(backend_->PublishResident({{"key-" + std::to_string(cycle), {}, handle}}, Op(seq++))
                    .success);
    const std::string request = "cycle-" + std::to_string(cycle);
    auto grant = backend_->Acquire({"reader", request}, {handle}, Op(seq++),
                                   GrantTable::Clock::now() + std::chrono::seconds(1));
    ASSERT_TRUE(grant[0].granted);
    const auto retire_op = Op(seq++);
    ASSERT_EQ(backend_->Retire({handle}, retire_op).state, engine::RetireState::kWaitingHolds);
    const auto release_op = Op(seq++);
    auto released = backend_->Release({"reader", request}, {grant[0].id},
                                      engine::CompletionProof::kCompleted, release_op);
    ASSERT_TRUE(released[0].released);
    ASSERT_EQ(backend_->PollReadyRetires().size(), 1u);
    ASSERT_TRUE(backend_->AcknowledgeReleased(retire_op));
    backend_->AdvanceAckWatermark(release_op);
  }
  const auto stats = backend_->HistoryStats();
  EXPECT_EQ(stats.delegate_records, 0u);
  EXPECT_EQ(stats.publish_records, 0u);
  EXPECT_EQ(stats.retire_records, 0u);
  EXPECT_EQ(stats.ready_queue, 0u);
  EXPECT_EQ(stats.grants.acquire_records, 0u);
  EXPECT_EQ(stats.grants.release_records, 0u);
  EXPECT_EQ(stats.grants.closed_sources, 0u);
  EXPECT_EQ(stats.retire_handle_checks, 2 * kCycles);

  auto too_old = handle;
  ++too_old.generation;
  EXPECT_FALSE(backend_->PublishResident({{"too-old", {}, too_old}}, Op(seq - 1)).success);
}

TEST_F(EngineFixture, AckWatermarkDoesNotDropAnUnconsumedReadyRetire) {
  auto handle = Publish("ready-before-watermark");
  ASSERT_EQ(backend_->Retire({handle}, Op(3)).state, engine::RetireState::kReady);
  backend_->AdvanceAckWatermark(Op(10));
  auto ready = backend_->PollReadyRetires();
  ASSERT_EQ(ready.size(), 1u);
  EXPECT_EQ(ready[0].id, Op(3));
  EXPECT_TRUE(backend_->AcknowledgeReleased(Op(3)));
  backend_->AdvanceAckWatermark(Op(11));
  EXPECT_EQ(backend_->HistoryStats().retire_records, 0u);
}

TEST_F(EngineFixture, LegacyDataPlaneCannotBypassNegotiatedEngineProtocol) {
  EXPECT_EQ(backend_->BatchAllocate({{"x", 10}})[0].outcome, AllocateOutcome::kFailed);
  EXPECT_EQ(backend_->BatchResolve({"x"}, true)[0].outcome, ResolveOutcome::kFailed);
  EXPECT_EQ(backend_->Evict({"x"})[0].bytes_freed, 0u);
}

TEST_F(EngineFixture, PeerPoolOwnsPublishedResidencyUntilReleasedAck) {
  BackendRegistry registry;
  auto* engine_backend = backend_.get();
  ASSERT_TRUE(registry.Register(
      "engine", MakeInstrumentedBackend(std::unique_ptr<MediumBackend>(backend_.release()))));
  const uint32_t backend_id = registry.BackendId(registry.Get("engine"));
  PeerPool pool(&registry, MakeSingleBackendPolicy());

  auto delegated = engine_backend->Delegate(7, 1, Op(20));
  ASSERT_TRUE(delegated.success);
  auto handle = delegated.handles[0];
  handle.generation = 1;
  ASSERT_TRUE(pool.PublishResident(backend_id, {{"pool-key", {}, handle}}, Op(21)).success);
  EXPECT_EQ(pool.ResidencyBackends("pool-key"), std::vector<uint32_t>{backend_id});

  auto grant = pool.AcquireEngineGrants(backend_id, {"reader", "request"}, {handle}, Op(22),
                                        GrantTable::Clock::now() + std::chrono::seconds(1));
  ASSERT_TRUE(grant[0].granted);
  auto ticket = pool.RetireEnginePages(backend_id, {handle}, Op(23));
  EXPECT_EQ(ticket.state, engine::RetireState::kWaitingHolds);
  auto released = pool.ReleaseEngineGrants(backend_id, {"reader", "request"}, {grant[0].id},
                                           engine::CompletionProof::kCompleted, Op(24));
  ASSERT_TRUE(released[0].released);
  ASSERT_EQ(pool.PollReadyEngineRetires(backend_id).size(), 1u);
  EXPECT_TRUE(pool.AcknowledgeEngineReleased(backend_id, Op(23)));
  EXPECT_TRUE(pool.AcknowledgeEngineReleased(backend_id, Op(23)));
  EXPECT_TRUE(pool.ResidencyBackends("pool-key").empty());
  EXPECT_EQ(engine_backend->State(handle.page_id), engine::SlotState::kDelegated);
}

}  // namespace
}  // namespace mori::umbp
