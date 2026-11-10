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

// BackendRegistry dispatch and cross-backend event aggregation. MockBackend is
// only the test double; its own behavior is exercised transitively here and by
// the PeerPool / PeerService suites.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "umbp/distributed/peer/backend/engine_hbm_backend.h"
#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/peer/backend/mock_backend.h"

namespace mori::umbp {

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

TEST(BackendRegistryDispatch, NamedSameTierInstancesKeepDenseStableIds) {
  BackendRegistry registry;
  ASSERT_TRUE(registry.Register("dram-primary", std::make_unique<MockBackend>(TierType::DRAM)));
  ASSERT_TRUE(registry.Register("dram-secondary", std::make_unique<MockBackend>(TierType::DRAM)));

  ASSERT_EQ(registry.Size(), 2u);
  EXPECT_EQ(registry.Get(TierType::DRAM), registry.Get("dram-primary"));
  EXPECT_EQ(registry.Get(0u), registry.Get("dram-primary"));
  EXPECT_EQ(registry.Get(1u), registry.Get("dram-secondary"));
  EXPECT_EQ(registry.BackendId(registry.Get("dram-secondary")), 1u);

  auto allocated = registry.Get("dram-primary")->BatchAllocate({{"old-state", 8}});
  registry.Get("dram-primary")->BatchCommit({{allocated.front().slot_id, "old-state"}});
  ASSERT_EQ(registry.Get("dram-primary")->OwnedKeyCount(), 1u);
  ASSERT_TRUE(registry.Register("dram-primary", std::make_unique<MockBackend>(TierType::DRAM)));
  EXPECT_EQ(registry.Get("dram-primary")->OwnedKeyCount(), 0u);
  EXPECT_EQ(registry.Size(), 2u);
  EXPECT_EQ(registry.BackendId(registry.Get("dram-primary")), 0u);
  ASSERT_EQ(registry.All().size(), 2u);
  EXPECT_EQ(registry.All()[0], registry.Get("dram-primary"));
  EXPECT_EQ(registry.All()[1], registry.Get("dram-secondary"));
}

TEST(BackendRegistryDispatch, ResultGeometryAllowsDifferentBackendPageSizes) {
  BackendRegistry registry;
  ASSERT_TRUE(registry.Register("dram", std::make_unique<MockBackend>(TierType::DRAM, 4096)));
  engine::PageGeometry geometry;
  geometry.page_tokens = 256;
  geometry.num_pages = 1;
  geometry.object_bytes = 1399040;
  geometry.regions.push_back(
      {"region", TransferRef::HostBytes(reinterpret_cast<void*>(1), 2 * geometry.object_bytes),
       geometry.object_bytes, 1, 0, 0, "kv"});
  engine::PoolEpoch epoch;
  ASSERT_TRUE(registry.Register("engine-hbm",
                                MakeEngineHbmBackend({"pool", epoch, 1, 0, std::move(geometry)})));
  EXPECT_EQ(registry.Get("dram")->PageSize(), 4096u);
  EXPECT_EQ(registry.Get("engine-hbm")->PageSize(), 1399040u);
  // Kept only for the legacy uniform PageBackend handshake.
  EXPECT_EQ(registry.PageSize(), 4096u);
}

TEST(BackendRegistryDispatch, LegacyBackendsStillRequireOneWirePageSize) {
  BackendRegistry registry;
  ASSERT_TRUE(registry.Register("a", std::make_unique<MockBackend>(TierType::DRAM, 4096)));
  EXPECT_FALSE(registry.Register("b", std::make_unique<MockBackend>(TierType::SSD, 8192)));
  EXPECT_EQ(registry.Size(), 1u);
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

TEST(BackendEventAgg, SameTierRelocationKeepsMasterLocation) {
  MockBackend source(TierType::SSD);
  MockBackend target(TierType::SSD);
  CommitOne(&source, "moved", 8);
  ASSERT_EQ(DrainAllBackends({&source, &target}).size(), 1u);

  CommitOne(&target, "moved", 8);
  source.Evict({"moved"});
  auto relocated = DrainAllBackends({&source, &target});
  ASSERT_EQ(relocated.size(), 1u);
  EXPECT_EQ(relocated.front().kind, KvEvent::Kind::ADD);
  EXPECT_EQ(relocated.front().tier, TierType::SSD);

  target.Evict({"moved"});
  auto removed = DrainAllBackends({&source, &target});
  ASSERT_EQ(removed.size(), 1u);
  EXPECT_EQ(removed.front().kind, KvEvent::Kind::REMOVE);
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
