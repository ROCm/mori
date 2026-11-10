// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include <gtest/gtest.h>

#include "umbp/distributed/pool/grant_table.h"

namespace mori::umbp {
namespace {

engine::AllocationHandle Handle(uint32_t page, uint64_t generation = 1) {
  engine::PoolEpoch epoch;
  epoch.boot_id[0] = 7;
  return {"pool", epoch, 3, page, generation};
}

engine::OperationId Op(uint64_t seq) {
  engine::OperationId id;
  id.session_id[0] = 9;
  id.seq = seq;
  return id;
}

TEST(GrantTableTest, ReadersReleaseIndependentlyAndClosedSourceRejectsNewGrant) {
  GrantTable table;
  const auto source = Handle(1);
  const auto deadline = GrantTable::Clock::now() + std::chrono::seconds(1);
  auto a = table.Acquire({"session-a", "req"}, {source}, Op(1), deadline);
  auto b = table.Acquire({"session-b", "req"}, {source}, Op(2), deadline);
  ASSERT_TRUE(a[0].granted);
  ASSERT_TRUE(b[0].granted);
  EXPECT_EQ(table.HoldCount(source), 2u);

  table.StopNewGrants({source});
  EXPECT_FALSE(table.Acquire({"c", "r"}, {source}, Op(3), deadline)[0].granted);
  auto released =
      table.Release({"session-a", "req"}, {a[0].id}, engine::CompletionProof::kCompleted, Op(4));
  EXPECT_TRUE(released[0].released);
  EXPECT_EQ(table.HoldCount(source), 1u);
  auto duplicate =
      table.Release({"session-a", "req"}, {a[0].id}, engine::CompletionProof::kCompleted, Op(5));
  EXPECT_FALSE(duplicate[0].known);
  table.Release({"session-b", "req"}, {b[0].id}, engine::CompletionProof::kCompleted, Op(6));
  EXPECT_FALSE(table.Held(source));
}

TEST(GrantTableTest, DuplicateOperationIdsReplayWithoutAddingOrReleasingTwice) {
  GrantTable table;
  const auto source = Handle(4);
  const auto deadline = GrantTable::Clock::now() + std::chrono::seconds(1);
  auto first = table.Acquire({"reader", "req"}, {source}, Op(10), deadline);
  auto replay = table.Acquire({"reader", "req"}, {source}, Op(10), deadline);
  ASSERT_TRUE(first[0].granted);
  EXPECT_EQ(replay[0].id.value, first[0].id.value);
  EXPECT_EQ(table.HoldCount(source), 1u);
  auto released =
      table.Release({"reader", "req"}, {first[0].id}, engine::CompletionProof::kCompleted, Op(11));
  auto released_replay =
      table.Release({"reader", "req"}, {first[0].id}, engine::CompletionProof::kCompleted, Op(11));
  EXPECT_TRUE(released[0].released);
  EXPECT_TRUE(released_replay[0].released);
  EXPECT_FALSE(table.Held(source));
}

TEST(GrantTableTest, WrongReaderCannotReleaseAnotherReadersGrant) {
  GrantTable table;
  auto grant = table.Acquire({"owner", "req"}, {Handle(2)}, Op(1),
                             GrantTable::Clock::now() + std::chrono::seconds(1));
  auto wrong =
      table.Release({"other", "req"}, {grant[0].id}, engine::CompletionProof::kCompleted, Op(2));
  EXPECT_FALSE(wrong[0].known);
  EXPECT_TRUE(table.Held(Handle(2)));
}

TEST(GrantTableTest, ExpiryAndUnknownCompletionRetainHoldForQuarantine) {
  GrantTable table;
  const auto source = Handle(3);
  const auto now = GrantTable::Clock::now();
  auto grant = table.Acquire({"reader", "req"}, {source}, Op(1), now);
  ASSERT_TRUE(grant[0].granted);
  auto expired = table.Expire(now);
  ASSERT_EQ(expired.size(), 1u);
  EXPECT_TRUE(table.Held(source));
  auto release =
      table.Release({"reader", "req"}, {grant[0].id}, engine::CompletionProof::kUnknown, Op(2));
  EXPECT_TRUE(release[0].quarantined);
  EXPECT_TRUE(table.Held(source));
  auto quarantine = table.DrainQuarantineCandidates();
  ASSERT_EQ(quarantine.size(), 1u);
  EXPECT_EQ(quarantine[0], source);
  EXPECT_TRUE(table.DrainQuarantineCandidates().empty());
  EXPECT_FALSE(table.Acquire({"new", "req"}, {source}, Op(3), now)[0].granted);
}

TEST(GrantTableTest, TerminalAcquireReplayAndAckWatermarkCannotReauthorize) {
  GrantTable table;
  const auto source = Handle(5);
  const auto deadline = GrantTable::Clock::now() + std::chrono::seconds(1);
  auto grant = table.Acquire({"reader", "req"}, {source}, Op(10), deadline);
  ASSERT_TRUE(grant[0].granted);
  auto released =
      table.Release({"reader", "req"}, {grant[0].id}, engine::CompletionProof::kCompleted, Op(11));
  ASSERT_TRUE(released[0].released);
  auto terminal = table.Acquire({"reader", "req"}, {source}, Op(10), deadline);
  ASSERT_EQ(terminal.size(), 1u);
  EXPECT_FALSE(terminal[0].granted);

  table.AdvanceAckWatermark(Op(11));
  auto too_old = table.Acquire({"reader", "req"}, {source}, Op(10), deadline);
  ASSERT_EQ(too_old.size(), 1u);
  EXPECT_FALSE(too_old[0].granted);
  const auto stats = table.Stats();
  EXPECT_EQ(stats.acquire_records, 0u);
  EXPECT_EQ(stats.release_records, 0u);
}

}  // namespace
}  // namespace mori::umbp
