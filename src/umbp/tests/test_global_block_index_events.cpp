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
#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "umbp/distributed/master/client_registry.h"
#include "umbp/distributed/master/global_block_index.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

namespace {

KvEvent Add(std::string key, TierType tier, uint64_t size) {
  return KvEvent{KvEvent::Kind::ADD, std::move(key), tier, size};
}

KvEvent AddExternal(std::string key, TierType tier) {
  return KvEvent{KvEvent::Kind::ADD, std::move(key), tier, 0, LocationOwner::EXTERNAL_HICACHE};
}

KvEvent Remove(std::string key, TierType tier) {
  return KvEvent{KvEvent::Kind::REMOVE, std::move(key), tier, 0};
}

KvEvent RemoveExternal(std::string key, TierType tier) {
  return KvEvent{KvEvent::Kind::REMOVE, std::move(key), tier, 0, LocationOwner::EXTERNAL_HICACHE};
}

KvEvent ClearExternal(TierType tier) {
  return KvEvent{KvEvent::Kind::CLEAR_AT_TIER, "", tier, 0, LocationOwner::EXTERNAL_HICACHE};
}

EventBundle Bundle(uint64_t seq, std::vector<KvEvent> events) {
  return EventBundle{seq, std::move(events)};
}

bool HasLocation(const std::vector<Location>& locs, const std::string& node, TierType tier,
                 uint64_t size, LocationOwner owner = LocationOwner::UMBP_OWNED) {
  for (const auto& l : locs) {
    if (l.node_id == node && l.tier == tier && l.size == size && l.owner == owner) return true;
  }
  return false;
}

}  // namespace

// ---- ApplyEvents: ADD/REMOVE round-trip ------------------------------------

TEST(GlobalBlockIndexEvents, ApplyAddInsertsLocation) {
  GlobalBlockIndex idx;
  ASSERT_EQ(idx.ApplyEvents("node-A", {Add("k1", TierType::DRAM, 1024)}), 1u);
  auto locs = idx.Lookup("k1");
  ASSERT_EQ(locs.size(), 1u);
  EXPECT_EQ(locs[0].node_id, "node-A");
  EXPECT_EQ(locs[0].tier, TierType::DRAM);
  EXPECT_EQ(locs[0].size, 1024u);
}

TEST(GlobalBlockIndexEvents, ApplyAddSameNodeTierUpdatesSize) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k", TierType::DRAM, 1024)});
  // ADD again with a different size on the same (node, tier) — the
  // dedup key is (node, tier), so the existing location's size is
  // overwritten rather than duplicated.
  idx.ApplyEvents("node-A", {Add("k", TierType::DRAM, 2048)});
  auto locs = idx.Lookup("k");
  ASSERT_EQ(locs.size(), 1u);
  EXPECT_EQ(locs[0].size, 2048u);
}

TEST(GlobalBlockIndexEvents, MultipleNodesCoexistForSameKey) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k", TierType::DRAM, 100)});
  idx.ApplyEvents("node-B", {Add("k", TierType::DRAM, 200)});
  idx.ApplyEvents("node-A", {Add("k", TierType::HBM, 300)});  // different tier on A
  auto locs = idx.Lookup("k");
  EXPECT_EQ(locs.size(), 3u);
  EXPECT_TRUE(HasLocation(locs, "node-A", TierType::DRAM, 100));
  EXPECT_TRUE(HasLocation(locs, "node-B", TierType::DRAM, 200));
  EXPECT_TRUE(HasLocation(locs, "node-A", TierType::HBM, 300));
}

TEST(GlobalBlockIndexEvents, RemoveErasesMatchingLocationOnly) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k", TierType::DRAM, 100)});
  idx.ApplyEvents("node-B", {Add("k", TierType::DRAM, 200)});

  idx.ApplyEvents("node-A", {Remove("k", TierType::DRAM)});
  auto locs = idx.Lookup("k");
  ASSERT_EQ(locs.size(), 1u);
  EXPECT_EQ(locs[0].node_id, "node-B");
}

TEST(GlobalBlockIndexEvents, RemoveLastLocationErasesEntry) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k", TierType::DRAM, 100)});
  idx.ApplyEvents("node-A", {Remove("k", TierType::DRAM)});
  EXPECT_TRUE(idx.Lookup("k").empty());
  EXPECT_FALSE(idx.GetMetrics("k").has_value());
}

TEST(GlobalBlockIndexEvents, RemoveUnknownIsNoop) {
  GlobalBlockIndex idx;
  EXPECT_EQ(idx.ApplyEvents("ghost", {Remove("ghost-key", TierType::DRAM)}), 0u);
}

TEST(GlobalBlockIndexEvents, MixedOwnersCoexistAndRemoveByOwner) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k", TierType::DRAM, 4096)});
  idx.ApplyEvents("node-A", {AddExternal("k", TierType::DRAM)});

  auto locs = idx.Lookup("k");
  ASSERT_EQ(locs.size(), 2u);
  EXPECT_TRUE(HasLocation(locs, "node-A", TierType::DRAM, 4096, LocationOwner::UMBP_OWNED));
  EXPECT_TRUE(HasLocation(locs, "node-A", TierType::DRAM, 0, LocationOwner::EXTERNAL_HICACHE));

  idx.ApplyEvents("node-A", {RemoveExternal("k", TierType::DRAM)});
  locs = idx.Lookup("k");
  ASSERT_EQ(locs.size(), 1u);
  EXPECT_TRUE(HasLocation(locs, "node-A", TierType::DRAM, 4096, LocationOwner::UMBP_OWNED));
}

TEST(GlobalBlockIndexEvents, ClearAtTierClearsOnlyTargetOwnerTier) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k1", TierType::DRAM, 1), AddExternal("k1", TierType::DRAM),
                             AddExternal("k2", TierType::SSD), AddExternal("k3", TierType::DRAM)});

  EXPECT_EQ(idx.ApplyEvents("node-A", {ClearExternal(TierType::DRAM)}), 2u);

  EXPECT_TRUE(
      HasLocation(idx.Lookup("k1"), "node-A", TierType::DRAM, 1, LocationOwner::UMBP_OWNED));
  EXPECT_FALSE(
      HasLocation(idx.Lookup("k1"), "node-A", TierType::DRAM, 0, LocationOwner::EXTERNAL_HICACHE));
  EXPECT_TRUE(
      HasLocation(idx.Lookup("k2"), "node-A", TierType::SSD, 0, LocationOwner::EXTERNAL_HICACHE));
  EXPECT_TRUE(idx.Lookup("k3").empty());
}

TEST(GlobalBlockIndexEvents, ServableLookupAndExternalMatchFilterOwners) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {AddExternal("h1", TierType::HBM), AddExternal("h1", TierType::DRAM)});
  idx.ApplyEvents("node-B", {Add("h1", TierType::DRAM, 4096)});

  auto servable = idx.LookupServable("h1");
  ASSERT_EQ(servable.size(), 1u);
  EXPECT_EQ(servable[0].node_id, "node-B");
  EXPECT_EQ(idx.BatchLookupExistsServable({"h1", "missing"}), std::vector<bool>({true, false}));

  auto matches = idx.MatchExternal({"h1"});
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].node_id, "node-A");
  EXPECT_EQ(matches[0].MatchedHashCount(), 1u);
  EXPECT_EQ(matches[0].hashes_by_tier[TierType::HBM], std::vector<std::string>({"h1"}));
  EXPECT_EQ(matches[0].hashes_by_tier[TierType::DRAM], std::vector<std::string>({"h1"}));
}

// ---- ReplaceNodeLocations: full-sync recovery ------------------------------

TEST(GlobalBlockIndexEvents, ReplaceNodeLocationsClearsThenInserts) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k1", TierType::DRAM, 100), Add("k2", TierType::DRAM, 200)});
  idx.ApplyEvents("node-B", {Add("k1", TierType::DRAM, 999)});  // shared key, different node

  // Full-sync from node-A: k1 stays (different size), k2 is gone, new k3 appears.
  idx.ReplaceNodeLocations("node-A",
                           {Add("k1", TierType::DRAM, 150), Add("k3", TierType::DRAM, 300)});

  auto k1 = idx.Lookup("k1");
  EXPECT_TRUE(HasLocation(k1, "node-A", TierType::DRAM, 150));
  EXPECT_TRUE(HasLocation(k1, "node-B", TierType::DRAM, 999));  // node-B untouched

  EXPECT_TRUE(idx.Lookup("k2").empty());  // dropped — node-A's full-sync didn't include it

  auto k3 = idx.Lookup("k3");
  EXPECT_TRUE(HasLocation(k3, "node-A", TierType::DRAM, 300));
}

TEST(GlobalBlockIndexEvents, ReplaceNodeLocationsEmptyClearsAllForNode) {
  // Used by ClientRegistry::UnregisterClient and the reaper to drop a
  // dead node's index entries.
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k1", TierType::DRAM, 1), Add("k2", TierType::HBM, 2)});
  idx.ApplyEvents("node-B", {Add("k1", TierType::DRAM, 3)});

  idx.ReplaceNodeLocations("node-A", {});
  EXPECT_EQ(idx.Lookup("k1").size(), 1u);  // node-B still owns k1
  EXPECT_TRUE(idx.Lookup("k2").empty());   // node-A's only HBM location is gone
}

TEST(GlobalBlockIndexEvents, ReplaceNodeLocationsIgnoresRemoveEntries) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k1", TierType::DRAM, 100)});
  // Snapshot full-sync conventionally carries only ADDs; sneaking
  // a REMOVE in is silently skipped (the snapshot is the truth).
  idx.ReplaceNodeLocations("node-A",
                           {Add("k2", TierType::DRAM, 200), Remove("k3", TierType::DRAM)});
  EXPECT_TRUE(idx.Lookup("k1").empty());
  EXPECT_FALSE(idx.Lookup("k2").empty());
  EXPECT_TRUE(idx.Lookup("k3").empty());
}

// ---- ClientRegistry::Heartbeat applies events end-to-end --------------------

TEST(ClientRegistryHeartbeat, AppliesEventsAndAdvancesSeq) {
  GlobalBlockIndex idx;
  ClientRegistryConfig cfg;
  ClientRegistry reg(cfg, idx);

  ASSERT_TRUE(reg.RegisterClient("node-A", "10.0.0.1:1", /*caps=*/{}, "10.0.0.1:2", {}));

  uint64_t acked = 0;
  uint32_t need_full = 0;
  auto status = reg.Heartbeat("node-A", /*caps=*/{}, {Bundle(1, {Add("k", TierType::DRAM, 42)})},
                              FullSyncScope::NONE, /*delta_seq_baseline=*/0, &acked, &need_full);
  EXPECT_EQ(status, ClientStatus::ALIVE);
  EXPECT_EQ(acked, 1u);
  EXPECT_EQ(need_full, 0u);
  EXPECT_FALSE(idx.Lookup("k").empty());
}

TEST(ClientRegistryHeartbeat, SeqGapTriggersFullSyncRequest) {
  GlobalBlockIndex idx;
  ClientRegistryConfig cfg;
  ClientRegistry reg(cfg, idx);
  reg.RegisterClient("node-A", "10.0.0.1:1", {}, "10.0.0.1:2", {});

  uint64_t acked = 0;
  uint32_t need_full = 0;
  // First heartbeat seq=1 — applied normally.
  reg.Heartbeat("node-A", {}, {Bundle(1, {Add("k1", TierType::DRAM, 1)})}, FullSyncScope::NONE, 0,
                &acked, &need_full);
  ASSERT_EQ(need_full, 0u);
  ASSERT_EQ(acked, 1u);

  // Second heartbeat skips seq=2: master detects the gap.
  reg.Heartbeat("node-A", {}, {Bundle(3, {Add("k2", TierType::DRAM, 2)})}, FullSyncScope::NONE, 0,
                &acked, &need_full);
  EXPECT_EQ(need_full, kLocationOwnerUmbpOwnedBit | kLocationOwnerExternalHiCacheBit);
  EXPECT_EQ(acked, 1u);  // unchanged — no events applied from this batch

  // k2 is NOT in the index because the gap-batch was rejected.
  EXPECT_TRUE(idx.Lookup("k2").empty());
  EXPECT_FALSE(idx.Lookup("k1").empty());
}

TEST(ClientRegistryHeartbeat, FullSyncReplacesNodeLocations) {
  GlobalBlockIndex idx;
  ClientRegistryConfig cfg;
  ClientRegistry reg(cfg, idx);
  reg.RegisterClient("node-A", "10.0.0.1:1", {}, "10.0.0.1:2", {});

  uint64_t acked = 0;
  uint32_t need_full = 0;
  reg.Heartbeat("node-A", {},
                {Bundle(1, {Add("k1", TierType::DRAM, 1), Add("k2", TierType::DRAM, 2)})},
                FullSyncScope::NONE, 0, &acked, &need_full);
  ASSERT_FALSE(idx.Lookup("k1").empty());
  ASSERT_FALSE(idx.Lookup("k2").empty());

  // Full-sync: only k1 + k3 should remain for node-A.
  reg.Heartbeat("node-A", {},
                {Bundle(2, {Add("k1", TierType::DRAM, 10), Add("k3", TierType::DRAM, 30)})},
                FullSyncScope::UMBP_OWNED, /*delta_seq_baseline=*/2, &acked, &need_full);
  EXPECT_EQ(acked, 2u);
  EXPECT_EQ(need_full, 0u);

  auto k1 = idx.Lookup("k1");
  ASSERT_EQ(k1.size(), 1u);
  EXPECT_EQ(k1[0].size, 10u);  // updated via full-sync
  EXPECT_TRUE(idx.Lookup("k2").empty());
  EXPECT_FALSE(idx.Lookup("k3").empty());
}

TEST(ClientRegistryHeartbeat, UnregisterClearsNodeFromIndex) {
  GlobalBlockIndex idx;
  ClientRegistryConfig cfg;
  ClientRegistry reg(cfg, idx);
  reg.RegisterClient("node-A", "10.0.0.1:1", {}, "10.0.0.1:2", {});

  uint64_t acked = 0;
  uint32_t need_full = 0;
  reg.Heartbeat("node-A", {}, {Bundle(1, {Add("k1", TierType::DRAM, 1)})}, FullSyncScope::NONE, 0,
                &acked, &need_full);
  ASSERT_FALSE(idx.Lookup("k1").empty());

  reg.UnregisterClient("node-A");
  EXPECT_TRUE(idx.Lookup("k1").empty());
  EXPECT_FALSE(reg.IsClientAlive("node-A"));
}

// ---- FindEvictionCandidates --------------------------------------------------

TEST(GlobalBlockIndexEvents, FindEvictionCandidatesFiltersByOverloadedNodeTier) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k1", TierType::DRAM, 100), Add("k2", TierType::HBM, 200)});
  idx.ApplyEvents("node-B", {Add("k1", TierType::DRAM, 100)});

  std::set<GlobalBlockIndex::NodeTierKey> overloaded = {
      {"node-A", TierType::DRAM},
  };
  auto candidates = idx.FindEvictionCandidates(overloaded);
  // Only node-A's DRAM location of k1 is a candidate.
  ASSERT_EQ(candidates.size(), 1u);
  EXPECT_EQ(candidates[0].key, "k1");
  EXPECT_EQ(candidates[0].location.node_id, "node-A");
  EXPECT_EQ(candidates[0].location.tier, TierType::DRAM);
}

}  // namespace mori::umbp
