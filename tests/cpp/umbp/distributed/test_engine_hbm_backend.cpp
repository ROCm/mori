// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include <gtest/gtest.h>

#include <condition_variable>
#include <future>
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
  ASSERT_TRUE(backend_->CompleteRetireTransition(Op(4), true, false));
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

TEST_F(EngineFixture, ZeroAutoFlushThresholdDisablesCallback) {
  int callbacks = 0;
  backend_->SetAutoFlushHook(0, [&] { ++callbacks; });
  auto handle = Publish("no-auto-flush");
  EXPECT_TRUE(backend_->Contains("no-auto-flush"));
  EXPECT_EQ(callbacks, 0);
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
  ASSERT_TRUE(backend_->CompleteRetireTransition(Op(4), true, false));
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
  ASSERT_TRUE(backend_->CompleteRetireTransition(Op(5), true, false));
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
  ASSERT_EQ(backend_->Retire({old}, Op(3)).state, engine::RetireState::kTransitionPending);
  ASSERT_EQ(backend_->PollTransitionRetires().size(), 1u);
  ASSERT_TRUE(backend_->CompleteRetireTransition(Op(3), true, false));
  ASSERT_TRUE(backend_->ReopenRetireTransition(Op(3)));
  auto reopened = backend_->PollTransitionRetires();
  ASSERT_EQ(reopened.size(), 1u);
  EXPECT_EQ(reopened[0].state, engine::RetireState::kTransitionPending);
  ASSERT_TRUE(backend_->CompleteRetireTransition(Op(3), true, false));
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

TEST_F(EngineFixture, DelegatedQuarantineAcceptsTheReleasedGeneration) {
  auto delegated = backend_->Delegate(7, 1, Op(1));
  ASSERT_TRUE(delegated.success);
  EXPECT_TRUE(backend_->Quarantine({delegated.handles[0]}))
      << "a late guard for the generation just Released remains rank-authoritative";
  EXPECT_EQ(backend_->State(delegated.handles[0].page_id), engine::SlotState::kQuarantined);
}

TEST_F(EngineFixture, RetireQuarantinesOnlyTheInvalidMemberOfAMixedBatch) {
  auto delegated = backend_->Delegate(7, 2, Op(1));
  ASSERT_TRUE(delegated.success);
  auto first = delegated.handles[0];
  auto second = delegated.handles[1];
  ++first.generation;
  ++second.generation;
  ASSERT_TRUE(backend_
                  ->PublishResident(
                      {{"mixed-expired-a", {}, first}, {"mixed-expired-b", {}, second}}, Op(2))
                  .success);
  const auto now = GrantTable::Clock::now();
  auto grant = backend_->Acquire({"reader", "mixed"}, {first}, Op(3), now);
  ASSERT_TRUE(grant[0].granted);
  ASSERT_EQ(backend_->ExpireGrants(now + std::chrono::milliseconds(1)),
            std::vector<engine::AllocationHandle>({first}));
  ASSERT_EQ(backend_->State(first.page_id), engine::SlotState::kQuarantined);

  auto ticket = backend_->Retire({first, second}, Op(4));
  ASSERT_EQ(ticket.handles, std::vector<engine::AllocationHandle>({second}));
  EXPECT_EQ(ticket.state, engine::RetireState::kTransitionPending);
  auto partial = backend_->PollReadyRetires();
  ASSERT_EQ(partial.size(), 1u);
  EXPECT_EQ(partial[0].handles, std::vector<engine::AllocationHandle>({first}));
  EXPECT_EQ(partial[0].state, engine::RetireState::kQuarantined);
  ASSERT_TRUE(backend_->CompleteRetireTransition(Op(4), true, false));
  auto ready = backend_->PollReadyRetires();
  ASSERT_EQ(ready.size(), 1u);
  EXPECT_EQ(ready[0].handles, std::vector<engine::AllocationHandle>({second}));
  EXPECT_EQ(backend_->State(second.page_id), engine::SlotState::kRetiring);
}

TEST_F(EngineFixture, ReportedReadyRetireCannotBeReopenedUnderTheRankAck) {
  auto handle = Publish("reported-ready");
  ASSERT_EQ(backend_->Retire({handle}, Op(3)).state, engine::RetireState::kTransitionPending);
  ASSERT_TRUE(backend_->CompleteRetireTransition(Op(3), true, false));
  ASSERT_EQ(backend_->PollReadyRetires().size(), 1u);
  EXPECT_FALSE(backend_->ReopenRetireTransition(Op(3)));
  EXPECT_TRUE(backend_->AcknowledgeReleased(Op(3)));
}

TEST_F(EngineFixture, ReportedReadyTicketQuarantinesAsOneImmutableTerminalBatch) {
  auto delegated = backend_->Delegate(7, 2, Op(1));
  ASSERT_TRUE(delegated.success);
  auto first = delegated.handles[0];
  auto second = delegated.handles[1];
  ++first.generation;
  ++second.generation;
  ASSERT_TRUE(
      backend_->PublishResident({{"reported-a", {}, first}, {"reported-b", {}, second}}, Op(2))
          .success);
  ASSERT_EQ(backend_->Retire({first, second}, Op(3)).state,
            engine::RetireState::kTransitionPending);
  ASSERT_TRUE(backend_->CompleteRetireTransition(Op(3), true, false));
  auto reported = backend_->PollReadyRetires();
  ASSERT_EQ(reported.size(), 1u);
  ASSERT_EQ(reported[0].handles.size(), 2u);

  ASSERT_TRUE(backend_->Quarantine({first}));
  auto terminal = backend_->PollReadyRetires();
  ASSERT_EQ(terminal.size(), 1u);
  EXPECT_EQ(terminal[0].state, engine::RetireState::kQuarantined);
  EXPECT_EQ(terminal[0].handles, reported[0].handles)
      << "a ticket already reported to the rank must never shrink under its ACK";
  EXPECT_EQ(backend_->State(first.page_id), engine::SlotState::kQuarantined);
  EXPECT_EQ(backend_->State(second.page_id), engine::SlotState::kQuarantined);
  EXPECT_FALSE(backend_->AcknowledgeReleased(Op(3)));
}

TEST_F(EngineFixture, MarkReportedReturnsCurrentTicketAfterPartialQuarantineRace) {
  auto delegated = backend_->Delegate(7, 2, Op(1));
  ASSERT_TRUE(delegated.success);
  auto first = delegated.handles[0];
  auto second = delegated.handles[1];
  ++first.generation;
  ++second.generation;
  ASSERT_TRUE(
      backend_->PublishResident({{"race-a", {}, first}, {"race-b", {}, second}}, Op(2)).success);
  ASSERT_EQ(backend_->Retire({first, second}, Op(3)).state,
            engine::RetireState::kTransitionPending);
  ASSERT_TRUE(backend_->CompleteRetireTransition(Op(3), true, false));

  std::mutex mutex;
  std::condition_variable cv;
  bool popped = false;
  bool allow_mark = false;
  auto reporter = std::async(std::launch::async, [&] {
    auto stale = backend_->PollReadyRetires(/*mark_reported=*/false);
    EXPECT_EQ(stale.front().handles.size(), 2u);
    {
      std::unique_lock<std::mutex> lock(mutex);
      popped = true;
      cv.notify_all();
      cv.wait(lock, [&] { return allow_mark; });
    }
    return backend_->MarkRetireReported(Op(3));
  });
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(cv.wait_for(lock, std::chrono::seconds(1), [&] { return popped; }));
  }
  ASSERT_TRUE(backend_->Quarantine({first}));
  {
    std::lock_guard<std::mutex> lock(mutex);
    allow_mark = true;
  }
  cv.notify_all();
  auto current = reporter.get();
  ASSERT_TRUE(current.has_value());
  EXPECT_EQ(current->handles, std::vector<engine::AllocationHandle>{second});
  auto partial = backend_->PollReadyRetires();
  ASSERT_EQ(partial.size(), 1u);
  EXPECT_EQ(partial[0].handles, std::vector<engine::AllocationHandle>{first});
  EXPECT_TRUE(backend_->AcknowledgeReleased(Op(3)));
  EXPECT_EQ(backend_->State(second.page_id), engine::SlotState::kDelegated);
}

TEST_F(EngineFixture, ReportedReadyAckRetriesAfterInternalReadWithoutLeakingTheSlot) {
  auto handle = Publish("reported-ready-read");
  ASSERT_EQ(backend_->Retire({handle}, Op(3)).state, engine::RetireState::kTransitionPending);
  ASSERT_EQ(backend_->PollTransitionRetires().size(), 1u);
  ASSERT_TRUE(backend_->CompleteRetireTransition(Op(3), true, false));
  ASSERT_EQ(backend_->PollReadyRetires().size(), 1u);
  ResolvedEntry held;
  ASSERT_TRUE(backend_->AcquireRetireMigrationRead("reported-ready-read", Op(3), &held));
  EXPECT_FALSE(backend_->AcknowledgeReleased(Op(3)))
      << "rank retries a transient ACK rejection; it does not fail-stop";
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kRetiring);
  backend_->ReleaseMigrationRead("reported-ready-read");
  EXPECT_TRUE(backend_->AcknowledgeReleased(Op(3)));
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kDelegated);
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

TEST_F(EngineFixture, ExpiringSeveralReadersDeduplicatesTheQuarantineHandle) {
  auto handle = Publish("expired-many");
  const auto now = GrantTable::Clock::now();
  ASSERT_TRUE(backend_->Acquire({"reader-a", "req"}, {handle}, Op(3), now)[0].granted);
  ASSERT_TRUE(backend_->Acquire({"reader-b", "req"}, {handle}, Op(4), now)[0].granted);
  auto expired = backend_->ExpireGrants(now);
  ASSERT_EQ(expired, std::vector<engine::AllocationHandle>{handle});
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kQuarantined);
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
    ASSERT_TRUE(backend_->CompleteRetireTransition(retire_op, true, false));
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
  ASSERT_EQ(backend_->Retire({handle}, Op(3)).state, engine::RetireState::kTransitionPending);
  ASSERT_TRUE(backend_->CompleteRetireTransition(Op(3), true, false));
  backend_->AdvanceAckWatermark(Op(10));
  auto ready = backend_->PollReadyRetires();
  ASSERT_EQ(ready.size(), 1u);
  EXPECT_EQ(ready[0].id, Op(3));
  EXPECT_TRUE(backend_->AcknowledgeReleased(Op(3)));
  backend_->AdvanceAckWatermark(Op(11));
  EXPECT_EQ(backend_->HistoryStats().retire_records, 0u);
}

TEST_F(EngineFixture, F3SafeRefusalMakesRetireReadyWithoutQuarantine) {
  auto handle = Publish("safe-f3-refusal");
  ASSERT_EQ(backend_->Retire({handle}, Op(3)).state, engine::RetireState::kTransitionPending);
  ASSERT_TRUE(backend_->CompleteRetireTransition(Op(3), /*success=*/false,
                                                 /*unknown=*/false));
  auto ready = backend_->PollReadyRetires();
  ASSERT_EQ(ready.size(), 1u);
  EXPECT_EQ(ready[0].state, engine::RetireState::kReady);
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kRetiring);
  EXPECT_TRUE(backend_->AcknowledgeReleased(Op(3)));
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kDelegated);
}

TEST_F(EngineFixture, ResetIsBusyWithGrantThenSealsAndCommitsOneRank) {
  auto handle = Publish("reset-key");
  auto grant = backend_->Acquire({"reader", "req"}, {handle}, Op(3),
                                 GrantTable::Clock::now() + std::chrono::seconds(1));
  ASSERT_TRUE(grant[0].granted);

  auto busy = backend_->BeginReset(Op(4));
  EXPECT_FALSE(busy.ready);
  auto after_busy = backend_->Acquire({"reader-2", "req"}, {handle}, Op(5),
                                      GrantTable::Clock::now() + std::chrono::seconds(1));
  ASSERT_TRUE(after_busy[0].granted) << "Busy must cancel the seal before replying";
  backend_->Release({"reader", "req"}, {grant[0].id}, engine::CompletionProof::kCompleted, Op(6));
  backend_->Release({"reader-2", "req"}, {after_busy[0].id}, engine::CompletionProof::kCompleted,
                    Op(7));

  auto ready = backend_->BeginReset(Op(8));
  ASSERT_TRUE(ready.ready);
  EXPECT_EQ(ready.new_epoch.reset_seq, ready.old_epoch.reset_seq + 1);
  EXPECT_FALSE(backend_->Delegate(7, 1, Op(9)).success);
  engine::PoolEpoch committed;
  std::vector<std::string> removed;
  ASSERT_TRUE(backend_->FinishReset(ready.id, true, &committed, &removed));
  EXPECT_EQ(committed, ready.new_epoch);
  EXPECT_EQ(removed, std::vector<std::string>{"reset-key"});
  EXPECT_FALSE(backend_->Contains("reset-key"));
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kDelegated);
}

TEST_F(EngineFixture, ReadyResetPublishesAStableApprovedSnapshotThatOnlyShrinks) {
  auto delegated = backend_->Delegate(7, 2, Op(1));
  ASSERT_TRUE(delegated.success);
  auto published = delegated.handles[0];
  ++published.generation;
  ASSERT_TRUE(backend_->PublishResident({{"reset-approved", {}, published}}, Op(2)).success);

  auto lease = backend_->BeginReset(Op(3));
  ASSERT_TRUE(lease.ready);
  ASSERT_EQ(lease.approved.size(), 2u);
  EXPECT_NE(std::find_if(lease.approved.begin(), lease.approved.end(),
                         [&](const auto& handle) {
                           return handle.page_id == published.page_id &&
                                  handle.generation == published.generation;
                         }),
            lease.approved.end());

  ASSERT_TRUE(backend_->Quarantine({published}));
  auto replay = backend_->BeginReset(Op(3));
  ASSERT_TRUE(replay.ready);
  ASSERT_EQ(replay.approved.size(), 1u);
  EXPECT_NE(replay.approved[0].page_id, published.page_id);
  auto notices = backend_->DrainQuarantineNotices();
  ASSERT_EQ(notices.size(), 1u);
  EXPECT_EQ(notices[0].page_id, published.page_id)
      << "the owning rank must be told when the Ready snapshot shrinks";

  engine::PoolEpoch epoch;
  std::vector<std::string> removed;
  std::vector<engine::AllocationHandle> approved;
  ASSERT_TRUE(backend_->FinishReset(lease.id, true, &epoch, &removed, &approved));
  ASSERT_EQ(approved.size(), 1u);
  EXPECT_EQ(approved[0].page_id, replay.approved[0].page_id);
  EXPECT_EQ(approved[0].epoch, epoch);
}

TEST_F(EngineFixture, ResetSealRejectsNewMigrationAndCommitRechecksHolds) {
  auto handle = Publish("migration-reset");
  ResolvedEntry held;
  ASSERT_TRUE(backend_->AcquireMigrationRead("migration-reset", &held, true));
  EXPECT_FALSE(backend_->BeginReset(Op(3)).ready);
  backend_->ReleaseMigrationRead("migration-reset");
  auto lease = backend_->BeginReset(Op(4));
  ASSERT_TRUE(lease.ready);
  ResolvedEntry blocked;
  EXPECT_FALSE(backend_->AcquireMigrationRead("migration-reset", &blocked, true));
  engine::PoolEpoch epoch;
  std::vector<std::string> removed;
  std::vector<engine::AllocationHandle> approved;
  EXPECT_TRUE(backend_->FinishReset(lease.id, true, &epoch, &removed, &approved));
  EXPECT_EQ(approved.size(), 1u);
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kDelegated);
}

TEST_F(EngineFixture, RetireWaitsForIoHoldAndUnknownCompletionQuarantines) {
  auto handle = Publish("io-key");
  ASSERT_TRUE(backend_->HoldIO({handle}, Op(3)));
  auto ticket = backend_->Retire({handle}, Op(4));
  EXPECT_EQ(ticket.state, engine::RetireState::kWaitingHolds);
  EXPECT_TRUE(backend_->PollReadyRetires().empty());
  ASSERT_TRUE(backend_->CompleteIO(Op(3), engine::CompletionProof::kUnknown));
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kQuarantined);
  auto quarantined = backend_->PollReadyRetires();
  ASSERT_EQ(quarantined.size(), 1u);
  EXPECT_EQ(quarantined[0].state, engine::RetireState::kQuarantined);
  auto reset = backend_->BeginReset(Op(5));
  ASSERT_TRUE(reset.ready) << "holds retained only by quarantined pages must not block reset";
  engine::PoolEpoch epoch;
  std::vector<std::string> removed;
  std::vector<engine::AllocationHandle> approved;
  ASSERT_TRUE(backend_->FinishReset(reset.id, true, &epoch, &removed, &approved));
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kQuarantined);
  EXPECT_TRUE(std::none_of(approved.begin(), approved.end(),
                           [&](const auto& item) { return item.page_id == handle.page_id; }));
}

TEST_F(EngineFixture, QuarantiningOnePageDoesNotStrandTheRestOfARetireBatch) {
  auto delegated = backend_->Delegate(7, 2, Op(1));
  ASSERT_TRUE(delegated.success);
  auto first = delegated.handles[0];
  auto second = delegated.handles[1];
  ++first.generation;
  ++second.generation;
  ASSERT_TRUE(
      backend_->PublishResident({{"multi-a", {}, first}, {"multi-b", {}, second}}, Op(2)).success);
  ASSERT_EQ(backend_->Retire({first, second}, Op(3)).state,
            engine::RetireState::kTransitionPending);
  ASSERT_TRUE(backend_->Quarantine({first}));
  auto partial = backend_->PollReadyRetires();
  ASSERT_EQ(partial.size(), 1u);
  EXPECT_EQ(partial[0].state, engine::RetireState::kQuarantined);
  EXPECT_EQ(partial[0].handles, std::vector<engine::AllocationHandle>{first});
  ASSERT_TRUE(backend_->CompleteRetireTransition(Op(3), true, false));
  auto ready = backend_->PollReadyRetires();
  ASSERT_EQ(ready.size(), 1u);
  EXPECT_EQ(ready[0].handles, std::vector<engine::AllocationHandle>{second});
  ASSERT_TRUE(backend_->AcknowledgeReleased(Op(3)));
  EXPECT_EQ(backend_->State(first.page_id), engine::SlotState::kQuarantined);
  EXPECT_EQ(backend_->State(second.page_id), engine::SlotState::kDelegated);
}

TEST_F(EngineFixture, UnknownMigrationHoldIsScopedToTheOldPageGeneration) {
  auto old = Publish("reused-key");
  ASSERT_EQ(backend_->Retire({old}, Op(3)).state, engine::RetireState::kTransitionPending);
  ResolvedEntry old_read;
  ASSERT_TRUE(backend_->AcquireRetireMigrationRead("reused-key", Op(3), &old_read));
  ASSERT_TRUE(backend_->Quarantine({old}));
  (void)backend_->PollReadyRetires();

  auto delegated = backend_->Delegate(7, 1, Op(4));
  ASSERT_TRUE(delegated.success);
  auto current = delegated.handles[0];
  ++current.generation;
  ASSERT_TRUE(backend_->PublishResident({{"reused-key", {}, current}}, Op(5)).success);
  EXPECT_EQ(backend_->Retire({current}, Op(6)).state, engine::RetireState::kTransitionPending);
}

TEST_F(EngineFixture, RetireDuringResetSealReturnsBusyWithoutChangingTheSlot) {
  auto handle = Publish("sealed-retire");
  auto reset = backend_->BeginReset(Op(3));
  ASSERT_TRUE(reset.ready);
  auto retire = backend_->Retire({handle}, Op(4));
  EXPECT_EQ(retire.state, engine::RetireState::kBusy);
  EXPECT_EQ(backend_->State(handle.page_id), engine::SlotState::kPublished);
  engine::PoolEpoch epoch;
  std::vector<std::string> removed;
  EXPECT_TRUE(backend_->FinishReset(reset.id, false, &epoch, &removed));
}

TEST_F(EngineFixture, LegacyDataPlaneCannotBypassNegotiatedEngineProtocol) {
  EXPECT_EQ(backend_->BatchAllocate({{"x", 10}})[0].outcome, AllocateOutcome::kFailed);
  EXPECT_EQ(backend_->BatchResolve({"x"}, true)[0].outcome, ResolveOutcome::kMissing);
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
  auto ready = pool.PollReadyEngineRetires(backend_id);
  ASSERT_EQ(ready.size(), 1u);
  EXPECT_EQ(ready.front().state, engine::RetireState::kReady);
  EXPECT_TRUE(pool.AcknowledgeEngineReleased(backend_id, Op(23)));
  EXPECT_TRUE(pool.AcknowledgeEngineReleased(backend_id, Op(23)));
  EXPECT_TRUE(pool.ResidencyBackends("pool-key").empty());
  EXPECT_EQ(engine_backend->State(handle.page_id), engine::SlotState::kDelegated);
}

TEST_F(EngineFixture, PeerPoolMaintenanceExpiresGrantAndRefreshesResidency) {
  BackendRegistry registry;
  auto* engine_backend = backend_.get();
  ASSERT_TRUE(registry.Register(
      "engine", MakeInstrumentedBackend(std::unique_ptr<MediumBackend>(backend_.release()))));
  const uint32_t backend_id = registry.BackendId(registry.Get("engine"));
  PeerPool pool(&registry, MakeSingleBackendPolicy());
  auto delegated = engine_backend->Delegate(7, 1, Op(30));
  ASSERT_TRUE(delegated.success);
  auto handle = delegated.handles[0];
  handle.generation = 1;
  ASSERT_TRUE(pool.PublishResident(backend_id, {{"ttl-key", {}, handle}}, Op(31)).success);
  auto grant = pool.AcquireEngineGrants(backend_id, {"reader", "ttl"}, {handle}, Op(32),
                                        GrantTable::Clock::now());
  ASSERT_TRUE(grant[0].granted);
  pool.RunEngineMaintenanceForTest();
  EXPECT_EQ(engine_backend->State(handle.page_id), engine::SlotState::kQuarantined);
  EXPECT_TRUE(pool.ResidencyBackends("ttl-key").empty());
}

TEST_F(EngineFixture, PeerPoolRejectsLegacyClearWhileEngineBackendIsActive) {
  BackendRegistry registry;
  auto* engine_backend = backend_.get();
  ASSERT_TRUE(registry.Register(
      "engine", MakeInstrumentedBackend(std::unique_ptr<MediumBackend>(backend_.release()))));
  const uint32_t backend_id = registry.BackendId(registry.Get("engine"));
  PeerPool pool(&registry, MakeSingleBackendPolicy());
  auto delegated = engine_backend->Delegate(7, 1, Op(40));
  ASSERT_TRUE(delegated.success);
  auto handle = delegated.handles[0];
  ++handle.generation;
  ASSERT_TRUE(pool.PublishResident(backend_id, {{"clear-rejected", {}, handle}}, Op(41)).success);
  const auto started = std::chrono::steady_clock::now();
  EXPECT_FALSE(pool.ClearLocal());
  EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::milliseconds(100));
  EXPECT_TRUE(engine_backend->Contains("clear-rejected"));
  EXPECT_EQ(engine_backend->State(handle.page_id), engine::SlotState::kPublished);
}

}  // namespace
}  // namespace mori::umbp
