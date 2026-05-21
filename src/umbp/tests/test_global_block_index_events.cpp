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

KvEvent Remove(std::string key, TierType tier) {
  return KvEvent{KvEvent::Kind::REMOVE, std::move(key), tier, 0};
}

bool HasLocation(const std::vector<Location>& locs, const std::string& node, TierType tier,
                 uint64_t size) {
  for (const auto& l : locs) {
    if (l.node_id == node && l.tier == tier && l.size == size) return true;
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

// Duplicate ADD keeps the first observed size; only REMOVE retires it.
TEST(GlobalBlockIndexEvents, ApplyAddSameNodeTierKeepsExistingSize) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k", TierType::DRAM, 1024)});
  idx.ApplyEvents("node-A", {Add("k", TierType::DRAM, 2048)});
  auto locs = idx.Lookup("k");
  ASSERT_EQ(locs.size(), 1u);
  EXPECT_EQ(locs[0].size, 1024u);
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

// ---- Reverse-index (node_to_keys_) invariants ------------------------------

TEST(GlobalBlockIndexEvents, ReplaceNodeLocationsAfterMultiTierRemoveKeepsKeyClean) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k", TierType::DRAM, 100), Add("k", TierType::HBM, 200)});
  idx.ApplyEvents("node-B", {Add("k", TierType::DRAM, 300)});

  // A still owns (k, HBM): reverse index must keep k.
  idx.ApplyEvents("node-A", {Remove("k", TierType::DRAM)});
  auto mid = idx.Lookup("k");
  ASSERT_EQ(mid.size(), 2u);
  EXPECT_TRUE(HasLocation(mid, "node-A", TierType::HBM, 200));
  EXPECT_TRUE(HasLocation(mid, "node-B", TierType::DRAM, 300));

  idx.ReplaceNodeLocations("node-A", {});
  auto after = idx.Lookup("k");
  ASSERT_EQ(after.size(), 1u);
  EXPECT_EQ(after[0].node_id, "node-B");
  EXPECT_EQ(after[0].tier, TierType::DRAM);
  EXPECT_EQ(after[0].size, 300u);
}

TEST(GlobalBlockIndexEvents, ReplaceNodeLocationsLeavesOtherNodesIntact) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k1", TierType::DRAM, 1), Add("k2", TierType::DRAM, 2)});
  idx.ApplyEvents("node-B", {Add("k1", TierType::DRAM, 10), Add("k3", TierType::HBM, 30)});
  idx.ApplyEvents("node-C", {Add("k2", TierType::HBM, 200), Add("k4", TierType::DRAM, 400)});

  idx.ReplaceNodeLocations("node-A", {Add("k_new", TierType::DRAM, 999)});

  auto k1 = idx.Lookup("k1");
  ASSERT_EQ(k1.size(), 1u);
  EXPECT_EQ(k1[0].node_id, "node-B");
  EXPECT_EQ(k1[0].size, 10u);

  auto k2 = idx.Lookup("k2");
  ASSERT_EQ(k2.size(), 1u);
  EXPECT_EQ(k2[0].node_id, "node-C");
  EXPECT_EQ(k2[0].tier, TierType::HBM);

  EXPECT_TRUE(HasLocation(idx.Lookup("k_new"), "node-A", TierType::DRAM, 999));

  auto k3 = idx.Lookup("k3");
  ASSERT_EQ(k3.size(), 1u);
  EXPECT_EQ(k3[0].node_id, "node-B");
  EXPECT_EQ(k3[0].size, 30u);

  auto k4 = idx.Lookup("k4");
  ASSERT_EQ(k4.size(), 1u);
  EXPECT_EQ(k4[0].node_id, "node-C");
  EXPECT_EQ(k4[0].size, 400u);
}

// 2nd sync must see reverse index repopulated by 1st sync's replay.
TEST(GlobalBlockIndexEvents, ReplaceNodeLocationsTwiceRotatesKeys) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k_old", TierType::DRAM, 1)});

  idx.ReplaceNodeLocations("node-A",
                           {Add("k_mid_a", TierType::DRAM, 2), Add("k_mid_b", TierType::HBM, 3)});
  EXPECT_TRUE(idx.Lookup("k_old").empty());
  EXPECT_FALSE(idx.Lookup("k_mid_a").empty());
  EXPECT_FALSE(idx.Lookup("k_mid_b").empty());

  idx.ReplaceNodeLocations("node-A", {Add("k_final", TierType::DRAM, 4)});
  EXPECT_TRUE(idx.Lookup("k_mid_a").empty());
  EXPECT_TRUE(idx.Lookup("k_mid_b").empty());
  auto final_locs = idx.Lookup("k_final");
  ASSERT_EQ(final_locs.size(), 1u);
  EXPECT_EQ(final_locs[0].node_id, "node-A");
  EXPECT_EQ(final_locs[0].size, 4u);
}

// Reverse-index insert must run even when inserted==false.
TEST(GlobalBlockIndexEvents, DuplicateAddKeepsReverseConsistent) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("dup", TierType::DRAM, 1024)});
  idx.ApplyEvents("node-A", {Add("dup", TierType::DRAM, 2048)});
  idx.ApplyEvents("node-A", {Add("dup", TierType::DRAM, 4096)});

  auto locs = idx.Lookup("dup");
  ASSERT_EQ(locs.size(), 1u);

  idx.ReplaceNodeLocations("node-A", {});
  EXPECT_TRUE(idx.Lookup("dup").empty());
}

// No-op REMOVE must leave node_to_keys_ untouched on both sides.
TEST(GlobalBlockIndexEvents, RemoveNonMatchingTierLeavesReverseUntouched) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k", TierType::DRAM, 100)});

  idx.ApplyEvents("node-A", {Remove("k", TierType::HBM)});
  auto mid = idx.Lookup("k");
  ASSERT_EQ(mid.size(), 1u);
  EXPECT_EQ(mid[0].tier, TierType::DRAM);

  idx.ReplaceNodeLocations("node-A", {});
  EXPECT_TRUE(idx.Lookup("k").empty());

  idx.ApplyEvents("node-A", {Add("k", TierType::DRAM, 100)});
  idx.ApplyEvents("node-B", {Remove("k", TierType::DRAM)});
  idx.ReplaceNodeLocations("node-B", {});
  auto after = idx.Lookup("k");
  ASSERT_EQ(after.size(), 1u);
  EXPECT_EQ(after[0].node_id, "node-A");
}

// ---- ClientRegistry::Heartbeat applies events end-to-end --------------------

TEST(ClientRegistryHeartbeat, AppliesEventsAndAdvancesSeq) {
  GlobalBlockIndex idx;
  ClientRegistryConfig cfg;
  ClientRegistry reg(cfg, idx);

  ASSERT_TRUE(reg.RegisterClient("node-A", "10.0.0.1:1", /*caps=*/{}, "10.0.0.1:2", {}));

  uint64_t acked = 0;
  bool need_full = false;
  auto status =
      reg.Heartbeat("node-A", /*seq=*/1, /*last_acked=*/0, /*caps=*/{},
                    {Add("k", TierType::DRAM, 42)}, /*is_full_sync=*/false, &acked, &need_full);
  EXPECT_EQ(status, ClientStatus::ALIVE);
  EXPECT_EQ(acked, 1u);
  EXPECT_FALSE(need_full);
  EXPECT_FALSE(idx.Lookup("k").empty());
}

TEST(ClientRegistryHeartbeat, SeqGapTriggersFullSyncRequest) {
  GlobalBlockIndex idx;
  ClientRegistryConfig cfg;
  ClientRegistry reg(cfg, idx);
  reg.RegisterClient("node-A", "10.0.0.1:1", {}, "10.0.0.1:2", {});

  uint64_t acked = 0;
  bool need_full = false;
  // First heartbeat seq=1 — applied normally.
  reg.Heartbeat("node-A", 1, 0, {}, {Add("k1", TierType::DRAM, 1)}, false, &acked, &need_full);
  ASSERT_FALSE(need_full);
  ASSERT_EQ(acked, 1u);

  // Second heartbeat skips seq=2: master detects the gap.
  reg.Heartbeat("node-A", 3, 1, {}, {Add("k2", TierType::DRAM, 2)}, false, &acked, &need_full);
  EXPECT_TRUE(need_full);
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
  bool need_full = false;
  reg.Heartbeat("node-A", 1, 0, {}, {Add("k1", TierType::DRAM, 1), Add("k2", TierType::DRAM, 2)},
                false, &acked, &need_full);
  ASSERT_FALSE(idx.Lookup("k1").empty());
  ASSERT_FALSE(idx.Lookup("k2").empty());

  // Full-sync: only k1 + k3 should remain for node-A.
  reg.Heartbeat("node-A", 2, 1, {}, {Add("k1", TierType::DRAM, 10), Add("k3", TierType::DRAM, 30)},
                /*is_full_sync=*/true, &acked, &need_full);
  EXPECT_EQ(acked, 2u);
  EXPECT_FALSE(need_full);

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
  bool need_full = false;
  reg.Heartbeat("node-A", 1, 0, {}, {Add("k1", TierType::DRAM, 1)}, false, &acked, &need_full);
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

// ---- BatchLookupForRouteGet ------------------------------------------------

TEST(GlobalBlockIndexEvents, BatchLookupForRouteGetEmptyInputReturnsEmpty) {
  GlobalBlockIndex idx;
  EXPECT_TRUE(idx.BatchLookupForRouteGet({}, {}, std::chrono::seconds{1}).empty());
}

TEST(GlobalBlockIndexEvents, BatchLookupForRouteGetMixedHitsAndMisses) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k1", TierType::DRAM, 100)});
  idx.ApplyEvents("node-B", {Add("k1", TierType::DRAM, 200), Add("k2", TierType::HBM, 300)});

  auto ref_k1 = idx.Lookup("k1");
  auto ref_k2 = idx.Lookup("k2");
  auto before_k1 = idx.GetMetrics("k1");
  auto before_k2 = idx.GetMetrics("k2");
  ASSERT_TRUE(before_k1.has_value());
  ASSERT_TRUE(before_k2.has_value());

  auto results = idx.BatchLookupForRouteGet({"k1", "ghost", "k2"}, {}, std::chrono::seconds{10});
  ASSERT_EQ(results.size(), 3u);
  EXPECT_EQ(results[0], ref_k1);
  EXPECT_TRUE(results[1].empty());
  EXPECT_EQ(results[2], ref_k2);

  auto after_k1 = idx.GetMetrics("k1");
  auto after_k2 = idx.GetMetrics("k2");
  ASSERT_TRUE(after_k1.has_value());
  ASSERT_TRUE(after_k2.has_value());
  EXPECT_EQ(after_k1->access_count, before_k1->access_count + 1);
  EXPECT_EQ(after_k2->access_count, before_k2->access_count + 1);
  EXPECT_FALSE(idx.GetMetrics("ghost").has_value());
}

TEST(GlobalBlockIndexEvents, BatchLookupForRouteGetGrantsLeaseForHitsOnly) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("hit", TierType::DRAM, 100), Add("other", TierType::DRAM, 200)});

  std::set<GlobalBlockIndex::NodeTierKey> overloaded{{"node-A", TierType::DRAM}};
  ASSERT_EQ(idx.FindEvictionCandidates(overloaded).size(), 2u);

  idx.BatchLookupForRouteGet({"hit", "ghost"}, {}, std::chrono::seconds{10});

  auto candidates = idx.FindEvictionCandidates(overloaded);
  ASSERT_EQ(candidates.size(), 1u);
  EXPECT_EQ(candidates[0].key, "other");
}

// All replicas excluded -> slot empty, access_count NOT bumped,
// lease NOT granted.  A key whose every replica is unreachable must
// not pollute LRU or block eviction.
TEST(GlobalBlockIndexEvents, BatchLookupForRouteGetSkipsSideEffectsWhenAllReplicasExcluded) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k", TierType::DRAM, 100)});
  idx.ApplyEvents("node-B", {Add("k", TierType::DRAM, 200)});

  auto before = idx.GetMetrics("k");
  ASSERT_TRUE(before.has_value());
  std::set<GlobalBlockIndex::NodeTierKey> overloaded{{"node-A", TierType::DRAM},
                                                     {"node-B", TierType::DRAM}};
  ASSERT_EQ(idx.FindEvictionCandidates(overloaded).size(), 2u);

  std::unordered_set<std::string> excludes{"node-A", "node-B"};
  auto results = idx.BatchLookupForRouteGet({"k"}, excludes, std::chrono::seconds{10});
  ASSERT_EQ(results.size(), 1u);
  EXPECT_TRUE(results[0].empty());

  auto after = idx.GetMetrics("k");
  ASSERT_TRUE(after.has_value());
  EXPECT_EQ(after->access_count, before->access_count);
  EXPECT_EQ(idx.FindEvictionCandidates(overloaded).size(), 2u);
}

// Some replicas excluded but not all -> returned slot has only the
// survivors, access_count IS bumped, lease IS granted.
TEST(GlobalBlockIndexEvents, BatchLookupForRouteGetFiltersAndLeasesWhenSomeReplicasSurvive) {
  GlobalBlockIndex idx;
  idx.ApplyEvents("node-A", {Add("k", TierType::DRAM, 100)});
  idx.ApplyEvents("node-B", {Add("k", TierType::DRAM, 200)});

  auto before = idx.GetMetrics("k");
  ASSERT_TRUE(before.has_value());

  std::unordered_set<std::string> excludes{"node-A"};
  auto results = idx.BatchLookupForRouteGet({"k"}, excludes, std::chrono::seconds{10});
  ASSERT_EQ(results.size(), 1u);
  ASSERT_EQ(results[0].size(), 1u);
  EXPECT_EQ(results[0][0].node_id, "node-B");

  auto after = idx.GetMetrics("k");
  ASSERT_TRUE(after.has_value());
  EXPECT_EQ(after->access_count, before->access_count + 1);
}

}  // namespace mori::umbp
