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
//
// Cross-component reliability tests: combinations no single-component test
// covers.  All deterministic, no real disk / RDMA / master RPC:
//   * the unified owned-location event source merges DRAM + SSD into one
//     snapshot/delta (so a heartbeat full-sync ships SSD owned keys too);
//   * a local SSD eviction's REMOVE SSD event converges the master
//     metadata store while leaving the DRAM bucket intact;
//   * tier-priority RouteGet over the real index picks DRAM, then SSD once the
//     DRAM replica is removed;
//   * crash-restart leftover is discarded at startup;
//   * the SSD observability counters increment at the right events.
//
// (copy-pin vs DRAM evict is covered by test_ssd_copy_pipeline's
// EvictBlockedWhilePinnedThenAllowedAfterRelease; seq-gap -> full-sync by
// test_in_memory_master_metadata_store's heartbeat SeqGap cases.)
#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "umbp/distributed/master/in_memory_master_metadata_store.h"
#include "umbp/distributed/peer/owned_location_source.h"
#include "umbp/distributed/peer/peer_ssd_manager.h"
#include "umbp/distributed/routing/route_get_strategy.h"
#include "umbp/distributed/types.h"
#include "umbp/local/tiers/tier_backend.h"

namespace mori::umbp {
namespace {

// Minimal in-memory TierBackend (mirrors the one in test_peer_ssd_eviction) so
// PeerSsdManager runs without real disk IO.  Exposes a forced evict failure for
// the backend-failure counter and lets the test pre-seed bytes (crash leftover).
class FakeBackend : public TierBackend {
 public:
  explicit FakeBackend(size_t capacity)
      : TierBackend(StorageTier::LOCAL_SSD), capacity_(capacity) {}

  bool Write(const std::string& key, const void* data, size_t size) override {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = store_.find(key);
    size_t prev = (it == store_.end()) ? 0 : it->second.size();
    if (used_ - prev + size > capacity_) return false;
    store_[key].assign(static_cast<const char*>(data), static_cast<const char*>(data) + size);
    used_ = used_ - prev + size;
    return true;
  }
  bool ReadIntoPtr(const std::string& key, uintptr_t dst, size_t size) override {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = store_.find(key);
    if (it == store_.end() || it->second.size() != size) return false;
    std::memcpy(reinterpret_cast<void*>(dst), it->second.data(), size);
    return true;
  }
  bool Exists(const std::string& key) const override {
    std::lock_guard<std::mutex> lk(mu_);
    return store_.count(key) != 0;
  }
  bool Evict(const std::string& key) override {
    std::lock_guard<std::mutex> lk(mu_);
    if (fail_evict_) return false;
    auto it = store_.find(key);
    if (it == store_.end()) return false;
    used_ -= it->second.size();
    store_.erase(it);
    return true;
  }
  std::pair<size_t, size_t> Capacity() const override {
    std::lock_guard<std::mutex> lk(mu_);
    return {used_, capacity_};
  }
  void Clear() override {
    std::lock_guard<std::mutex> lk(mu_);
    ++clear_calls_;
    store_.clear();
    used_ = 0;
  }
  void SetFailEvict(bool f) {
    std::lock_guard<std::mutex> lk(mu_);
    fail_evict_ = f;
  }
  int clear_calls() const {
    std::lock_guard<std::mutex> lk(mu_);
    return clear_calls_;
  }

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, std::vector<char>> store_;
  size_t used_ = 0;
  size_t capacity_;
  bool fail_evict_ = false;
  int clear_calls_ = 0;
};

std::vector<std::pair<const void*, size_t>> OneSeg(const std::string& s) {
  return {{s.data(), s.size()}};
}

bool HasLoc(const std::vector<Location>& locs, const std::string& node, TierType tier) {
  for (const auto& l : locs) {
    if (l.node_id == node && l.tier == tier) return true;
  }
  return false;
}

int CountTier(const std::vector<KvEvent>& events, KvEvent::Kind kind, TierType tier) {
  int n = 0;
  for (const auto& e : events) {
    if (e.kind == kind && e.tier == tier) ++n;
  }
  return n;
}

// Under the merged store a block location can only be created through an
// ApplyHeartbeat from a registered (alive) node — locations no longer exist
// independently of a client record the way the old GlobalBlockIndex allowed.
// These helpers register a node once and apply heartbeat deltas with an
// ascending seq, standing in for the old GlobalBlockIndex::ApplyEvents.
constexpr uint64_t kGB = 1024ULL * 1024 * 1024;

std::map<TierType, TierCapacity> MakeCaps() {
  std::map<TierType, TierCapacity> caps;
  caps[TierType::DRAM] = {8 * kGB, 8 * kGB};
  caps[TierType::SSD] = {8 * kGB, 8 * kGB};
  return caps;
}

ClientRegistration MakeReg(const std::string& node_id) {
  ClientRegistration reg;
  reg.node_id = node_id;
  reg.node_address = node_id + ":1";
  reg.peer_address = node_id + ":peer";
  reg.tier_capacities = MakeCaps();
  return reg;
}

// A canned owned-location source standing in for PeerDramAllocator so the
// aggregation can be tested without standing up a DRAM allocator.
class FakeOwnedSource : public OwnedLocationSource {
 public:
  std::vector<KvEvent> delta;
  std::vector<KvEvent> snapshot;
  std::vector<KvEvent> DrainPendingEvents() override {
    std::vector<KvEvent> out;
    out.swap(delta);
    return out;
  }
  std::vector<KvEvent> SnapshotOwnedKeys() const override { return snapshot; }
};

// ---------------------------------------------------------------------------
//  Unified owned-location source: DRAM + SSD merge into one bundle.
// ---------------------------------------------------------------------------

// A heartbeat full-sync snapshots ALL sources; SSD owned keys must be present
// alongside DRAM in the merged snapshot (otherwise master would drop the SSD
// tier on a seq-gap recovery).
TEST(SsdReliability, FullSyncSnapshotMergesDramAndSsdOwnedKeys) {
  FakeOwnedSource dram;
  dram.snapshot = {KvEvent{KvEvent::Kind::ADD, "d-key", TierType::DRAM, 10}};

  auto be = std::make_unique<FakeBackend>(1'000'000);
  PeerSsdManager ssd(std::move(be), 0.9, 0.7);
  ASSERT_TRUE(ssd.Write("s-key", OneSeg("ssddata"), 7));
  ssd.DrainPendingEvents();  // the ADD SSD delta; snapshot is independent

  std::vector<OwnedLocationSource*> sources = {&dram, &ssd};
  auto snap = SnapshotAllSources(sources);

  EXPECT_EQ(CountTier(snap, KvEvent::Kind::ADD, TierType::DRAM), 1);
  EXPECT_EQ(CountTier(snap, KvEvent::Kind::ADD, TierType::SSD), 1);
}

// A delta heartbeat drains ALL sources and concatenates into one list.
TEST(SsdReliability, DeltaDrainMergesDramAndSsdEvents) {
  FakeOwnedSource dram;
  dram.delta = {KvEvent{KvEvent::Kind::REMOVE, "d-key", TierType::DRAM, 0}};

  auto be = std::make_unique<FakeBackend>(1'000'000);
  PeerSsdManager ssd(std::move(be), 0.9, 0.7);
  ASSERT_TRUE(ssd.Write("s-key", OneSeg("ssddata"), 7));  // queues ADD SSD delta

  std::vector<OwnedLocationSource*> sources = {&dram, &ssd};
  auto merged = DrainAllSources(sources);

  EXPECT_EQ(CountTier(merged, KvEvent::Kind::REMOVE, TierType::DRAM), 1);
  EXPECT_EQ(CountTier(merged, KvEvent::Kind::ADD, TierType::SSD), 1);
  // Draining again yields nothing (outbox cleared on both sources).
  EXPECT_TRUE(DrainAllSources(sources).empty());
}

// ---------------------------------------------------------------------------
//  SSD local eviction -> REMOVE SSD -> master metadata store converges.
// ---------------------------------------------------------------------------

// A key mirrored on DRAM + SSD of one owner: a local SSD eviction emits
// REMOVE SSD, and applying that to the master store drops only the SSD bucket
// (the DRAM replica, owned independently, stays routable).
TEST(SsdReliability, LocalSsdEvictionRemoveConvergesMasterIndex) {
  InMemoryMasterMetadataStore store;
  const auto now = std::chrono::system_clock::now();
  ASSERT_TRUE(store.RegisterClient(MakeReg("owner"), now, std::chrono::seconds{30}));

  auto be = std::make_unique<FakeBackend>(1'000'000);
  PeerSsdManager ssd(std::move(be), 0.9, 0.7);

  // DRAM replica added independently (a DRAM owner would emit this).
  ASSERT_EQ(store
                .ApplyHeartbeat("owner", /*seq=*/1, now, MakeCaps(),
                                {KvEvent{KvEvent::Kind::ADD, "k", TierType::DRAM, 100}},
                                /*is_full_sync=*/false)
                .status,
            HeartbeatResult::APPLIED);
  // SSD copy lands -> ADD SSD drained into the store.
  ASSERT_TRUE(ssd.Write("k", OneSeg(std::string(100, 'x')), 100));
  ASSERT_EQ(store
                .ApplyHeartbeat("owner", /*seq=*/2, now, MakeCaps(), ssd.DrainPendingEvents(),
                                /*is_full_sync=*/false)
                .status,
            HeartbeatResult::APPLIED);

  auto both = store.LookupBlock("k");
  ASSERT_TRUE(HasLoc(both, "owner", TierType::DRAM));
  ASSERT_TRUE(HasLoc(both, "owner", TierType::SSD));

  // Local SSD eviction -> REMOVE SSD -> store drops only the SSD bucket.
  ASSERT_TRUE(ssd.Evict("k"));
  auto ssd_events = ssd.DrainPendingEvents();
  EXPECT_EQ(CountTier(ssd_events, KvEvent::Kind::REMOVE, TierType::SSD), 1);
  ASSERT_EQ(store
                .ApplyHeartbeat("owner", /*seq=*/3, now, MakeCaps(), ssd_events,
                                /*is_full_sync=*/false)
                .status,
            HeartbeatResult::APPLIED);

  auto after = store.LookupBlock("k");
  EXPECT_TRUE(HasLoc(after, "owner", TierType::DRAM));  // DRAM replica still routable
  EXPECT_FALSE(HasLoc(after, "owner", TierType::SSD));  // SSD bucket converged away
}

// ---------------------------------------------------------------------------
//  Tier-priority RouteGet over the real store: DRAM first, SSD after evict.
// ---------------------------------------------------------------------------

TEST(SsdReliability, TierPriorityRoutesDramThenSsdAfterDramRemoved) {
  InMemoryMasterMetadataStore store;
  const auto now = std::chrono::system_clock::now();
  const std::unordered_set<std::string> kNoExclude;
  ASSERT_TRUE(store.RegisterClient(MakeReg("owner"), now, std::chrono::seconds{30}));
  ASSERT_EQ(store
                .ApplyHeartbeat("owner", /*seq=*/1, now, MakeCaps(),
                                {KvEvent{KvEvent::Kind::ADD, "k", TierType::DRAM, 100},
                                 KvEvent{KvEvent::Kind::ADD, "k", TierType::SSD, 100}},
                                /*is_full_sync=*/false)
                .status,
            HeartbeatResult::APPLIED);

  TierPriorityRouteGetStrategy strategy;

  auto locs = store.BatchLookupBlockForRouteGet({"k"}, kNoExclude, now, std::chrono::seconds{10});
  ASSERT_EQ(locs.size(), 1u);
  auto dram_pick = strategy.Select(locs[0], "reader");
  EXPECT_EQ(dram_pick.tier, TierType::DRAM) << "prefers the fast DRAM replica";

  // DRAM evicted -> only the SSD bucket remains -> RouteGet must serve from SSD.
  ASSERT_EQ(store
                .ApplyHeartbeat("owner", /*seq=*/2, now, MakeCaps(),
                                {KvEvent{KvEvent::Kind::REMOVE, "k", TierType::DRAM, 0}},
                                /*is_full_sync=*/false)
                .status,
            HeartbeatResult::APPLIED);
  auto locs2 = store.BatchLookupBlockForRouteGet({"k"}, kNoExclude, now, std::chrono::seconds{10});
  ASSERT_EQ(locs2.size(), 1u);
  auto ssd_pick = strategy.Select(locs2[0], "reader");
  EXPECT_EQ(ssd_pick.tier, TierType::SSD) << "falls back to the surviving SSD replica";
  EXPECT_EQ(ssd_pick.node_id, "owner");
}

// ---------------------------------------------------------------------------
//  Crash-restart leftover handling (discard).
// ---------------------------------------------------------------------------

// After a crash owned_ is empty but the backend still holds bytes from the
// previous run.  DiscardLeftoverOnStartup wipes them so used capacity starts
// at 0 (no divergence between the empty owned_ map and the physical device).
TEST(SsdReliability, StartupDiscardWipesLeftoverBytes) {
  auto be = std::make_unique<FakeBackend>(1'000'000);
  FakeBackend* raw = be.get();
  // Simulate a previous process's bytes left on the device.
  ASSERT_TRUE(raw->Write("orphan-1", "leftover-a", 10));
  ASSERT_TRUE(raw->Write("orphan-2", "leftover-b", 10));
  ASSERT_GT(raw->Capacity().first, 0u);

  // Fresh manager: owned_ is empty, but the backend reports used > 0.
  PeerSsdManager ssd(std::move(be), 0.9, 0.7);
  EXPECT_TRUE(ssd.SnapshotOwnedKeys().empty());
  ASSERT_GT(ssd.Capacity().first, 0u);

  ssd.DiscardLeftoverOnStartup();

  EXPECT_EQ(raw->clear_calls(), 1);
  EXPECT_EQ(ssd.Capacity().first, 0u);  // leftover gone -> consistent with empty owned_
}

TEST(SsdReliability, StartupDiscardOnCleanTierIsNoop) {
  auto be = std::make_unique<FakeBackend>(1'000'000);
  FakeBackend* raw = be.get();
  PeerSsdManager ssd(std::move(be), 0.9, 0.7);

  ssd.DiscardLeftoverOnStartup();  // used == 0 -> skip the wipe entirely
  EXPECT_EQ(raw->clear_calls(), 0);
}

// ---------------------------------------------------------------------------
//  Observability counters increment at the right events.
// ---------------------------------------------------------------------------

TEST(SsdReliability, ReadCountersTrackOutcomes) {
  auto be = std::make_unique<FakeBackend>(1'000'000);
  PeerSsdManager ssd(std::move(be), 0.9, 0.7);
  ASSERT_TRUE(ssd.Write("k", OneSeg("0123456789"), 10));

  std::vector<char> buf(10);
  EXPECT_EQ(ssd.PrepareRead("k", buf.data(), buf.size()).status, SsdReadStatus::kOk);
  EXPECT_EQ(ssd.PrepareRead("absent", buf.data(), buf.size()).status, SsdReadStatus::kNotFound);
  EXPECT_EQ(ssd.PrepareRead("k", buf.data(), /*cap=*/1).status, SsdReadStatus::kSizeTooLarge);

  EXPECT_EQ(ssd.ReadOk(), 1u);
  EXPECT_EQ(ssd.ReadNotFound(), 1u);
  EXPECT_EQ(ssd.ReadSizeTooLarge(), 1u);
  EXPECT_EQ(ssd.ReadError(), 0u);
}

TEST(SsdReliability, EvictionCountersTrackVictimsBytesAndBackendFailures) {
  auto be = std::make_unique<FakeBackend>(1'000'000);
  FakeBackend* raw = be.get();
  PeerSsdManager ssd(std::move(be), 0.9, 0.7);
  ASSERT_TRUE(ssd.Write("a", OneSeg(std::string(40, 'a')), 40));
  ASSERT_TRUE(ssd.Write("b", OneSeg(std::string(60, 'b')), 60));

  ASSERT_TRUE(ssd.Evict("a"));
  EXPECT_EQ(ssd.EvictionVictims(), 1u);
  EXPECT_EQ(ssd.EvictionBytesFreed(), 40u);
  EXPECT_EQ(ssd.EvictionBackendFailures(), 0u);

  // Backend refuses the next evict -> the failure is counted, the key kept.
  raw->SetFailEvict(true);
  EXPECT_FALSE(ssd.Evict("b"));
  EXPECT_EQ(ssd.EvictionBackendFailures(), 1u);
  EXPECT_EQ(ssd.EvictionVictims(), 1u);  // unchanged
  EXPECT_TRUE(ssd.Exists("b"));
}

TEST(SsdReliability, WatermarkEvictionCountsARound) {
  // capacity 1000, high 0.9 (=>900), low 0.5 (=>500); 100-byte values.  After
  // the 9th write used hits 900 -> one eviction round runs.
  auto be = std::make_unique<FakeBackend>(1000);
  PeerSsdManager ssd(std::move(be), 0.9, 0.5);
  std::string val(100, 'x');
  for (int i = 1; i <= 9; ++i) {
    ASSERT_TRUE(ssd.Write("k" + std::to_string(i), OneSeg(val), val.size()));
  }
  EXPECT_GE(ssd.EvictionRounds(), 1u);
  EXPECT_GE(ssd.EvictionVictims(), 1u);
  EXPECT_LE(ssd.Capacity().first, 500u);
}

}  // namespace
}  // namespace mori::umbp
