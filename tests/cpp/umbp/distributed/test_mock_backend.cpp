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

// Proves BackendRegistry dispatch across two distinct MediumBackend
// implementations (design doc §5 Phase 2, default option (b)): registering a
// MockBackend alongside a real one and driving both purely through the
// registry / MediumBackend interface, with no concrete type named at the call
// site.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "umbp/distributed/peer/medium_backend.h"
#include "umbp/distributed/peer/mock_backend.h"

namespace mori::umbp {

TEST(MockBackend, ZeroCapacityByDesign) {
  MockBackend b(TierType::HBM);
  EXPECT_EQ(b.Capacity().total_bytes, 0u);
  EXPECT_EQ(b.Capacity().available_bytes, 0u);
}

TEST(MockBackend, AllocateCommitResolveEvictRoundTrip) {
  MockBackend b(TierType::HBM);
  ASSERT_TRUE(b.Init(nullptr));

  auto allocated = b.BatchAllocate({{"k", 128}});
  ASSERT_EQ(allocated.size(), 1u);
  ASSERT_EQ(allocated[0].outcome, AllocateOutcome::kSuccessAllocated);

  auto committed = b.BatchCommit({{allocated[0].slot_id, "k"}});
  ASSERT_EQ(committed.size(), 1u);
  EXPECT_TRUE(committed[0].success);
  EXPECT_EQ(committed[0].bytes_committed, 128u);
  EXPECT_EQ(b.OwnedKeyCount(), 1u);

  auto resolved = b.BatchResolve({"k", "missing"}, /*include_descs=*/true);
  ASSERT_EQ(resolved.size(), 2u);
  EXPECT_TRUE(resolved[0].found);
  EXPECT_EQ(resolved[0].size, 128u);
  EXPECT_FALSE(resolved[1].found);

  auto evicted = b.Evict({"k"});
  ASSERT_EQ(evicted.size(), 1u);
  EXPECT_EQ(evicted[0].bytes_freed, 128u);
  EXPECT_EQ(b.OwnedKeyCount(), 0u);
}

TEST(MockBackend, DrainAndSnapshotEvents) {
  MockBackend b(TierType::HBM);
  auto allocated = b.BatchAllocate({{"a", 8}, {"b", 8}});
  b.BatchCommit({{allocated[0].slot_id, "a"}, {allocated[1].slot_id, "b"}});

  auto drained = b.DrainPendingEvents();
  EXPECT_EQ(drained.size(), 2u);
  EXPECT_TRUE(b.DrainPendingEvents().empty());  // outbox cleared

  auto snap = b.SnapshotOwnedKeys();
  EXPECT_EQ(snap.size(), 2u);
}

// The registry dispatches to whichever concrete backend is registered for a
// tier, without the caller naming MockBackend anywhere below this line — this
// is the acceptance property the design doc calls out for Phase 2.
TEST(BackendRegistryDispatch, RoutesByTierThroughTheInterfaceOnly) {
  BackendRegistry registry;
  registry.Register(std::make_unique<MockBackend>(TierType::HBM));

  MediumBackend* hbm = registry.Get(TierType::HBM);
  ASSERT_NE(hbm, nullptr);
  EXPECT_EQ(hbm->Tier(), TierType::HBM);
  EXPECT_EQ(registry.Get(TierType::DRAM), nullptr);

  auto allocated = hbm->BatchAllocate({{"key", 64}});
  ASSERT_EQ(allocated.size(), 1u);
  ASSERT_EQ(allocated[0].outcome, AllocateOutcome::kSuccessAllocated);
  hbm->BatchCommit({{allocated[0].slot_id, "key"}});
  EXPECT_EQ(hbm->OwnedKeyCount(), 1u);

  ASSERT_EQ(registry.All().size(), 1u);
  EXPECT_EQ(registry.All()[0], hbm);
}

// ---- Heartbeat event aggregation -------------------------------------------
// These replace the OwnedLocationSourceAgg tests that lived in
// test_peer_ssd_manager.cpp: OwnedLocationSource is gone and MediumBackend
// carries the event contract (design doc §3, backend-agnostic refactor
// Phase 3).  What matters is unchanged — every medium's events concat into ONE
// list, in order, so the shipper can wrap them in a single bundle under one
// monotonic seq.

namespace {

// Commit `key` on `b` so exactly one ADD lands in its outbox.
void CommitOne(MediumBackend* b, const std::string& key, uint64_t size) {
  auto allocated = b->BatchAllocate({{key, size}});
  b->BatchCommit({{allocated[0].slot_id, key}});
}

}  // namespace

TEST(BackendEventAgg, DrainAndSnapshotConcatAcrossBackendsInOrder) {
  MockBackend hbm(TierType::HBM);
  MockBackend ssd(TierType::SSD);
  CommitOne(&hbm, "h1", 10);
  CommitOne(&ssd, "s1", 30);

  std::vector<MediumBackend*> backends = {&hbm, &ssd};

  auto drained = DrainAllBackends(backends);
  ASSERT_EQ(drained.size(), 2u);
  EXPECT_EQ(drained[0].key, "h1");
  EXPECT_EQ(drained[0].tier, TierType::HBM);
  EXPECT_EQ(drained[1].key, "s1");
  EXPECT_EQ(drained[1].tier, TierType::SSD);
  // Drain cleared both outboxes.
  EXPECT_TRUE(DrainAllBackends(backends).empty());

  // The full-sync snapshot reports owned keys regardless of the drained outbox.
  auto snap = SnapshotAllBackendsForFullSync(backends);
  ASSERT_EQ(snap.size(), 2u);
  EXPECT_EQ(snap[1].tier, TierType::SSD);
}

TEST(BackendEventAgg, FullSyncSnapshotAlsoDropsTheOutbox) {
  MockBackend b(TierType::HBM);
  CommitOne(&b, "x", 1);

  // Snapshot is authoritative, so the still-queued ADD must not be re-shipped
  // as a redundant delta afterwards.
  auto snap = SnapshotAllBackendsForFullSync({&b});
  ASSERT_EQ(snap.size(), 1u);
  EXPECT_TRUE(DrainAllBackends({&b}).empty());
}

TEST(BackendEventAgg, NullBackendsAreSkipped) {
  MockBackend only(TierType::SSD);
  CommitOne(&only, "x", 1);
  std::vector<MediumBackend*> backends = {nullptr, &only, nullptr};
  auto drained = DrainAllBackends(backends);
  ASSERT_EQ(drained.size(), 1u);
  EXPECT_EQ(drained[0].key, "x");
  EXPECT_TRUE(SnapshotAllBackendsForFullSync({nullptr}).empty());
}

TEST(BackendEventAgg, AutoFlushHookFiresAtThreshold) {
  MockBackend b(TierType::HBM);
  int fires = 0;
  b.SetAutoFlushHook(2, [&] { ++fires; });
  CommitOne(&b, "a", 1);
  EXPECT_EQ(fires, 0);
  CommitOne(&b, "b", 1);
  EXPECT_EQ(fires, 1);
}

}  // namespace mori::umbp
