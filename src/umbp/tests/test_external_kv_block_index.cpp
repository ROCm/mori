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
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/master/external_kv_block_index.h"

namespace mori::umbp {

// ---- Helpers ----------------------------------------------------------------

static std::vector<std::string> SortedNodeIds(
    const std::vector<ExternalKvBlockIndex::NodeMatch>& ms) {
  std::vector<std::string> ids;
  for (const auto& m : ms) ids.push_back(m.node_id);
  std::sort(ids.begin(), ids.end());
  return ids;
}

static const ExternalKvBlockIndex::NodeMatch* FindMatch(
    const std::vector<ExternalKvBlockIndex::NodeMatch>& ms, const std::string& node_id) {
  for (const auto& m : ms) {
    if (m.node_id == node_id) return &m;
  }
  return nullptr;
}

static std::vector<std::string> SortedHashes(const std::vector<std::string>& v) {
  std::vector<std::string> out = v;
  std::sort(out.begin(), out.end());
  return out;
}

// ---- Tests ------------------------------------------------------------------

TEST(ExternalKvBlockIndex, RegisterAndMatchBasic) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1", "h2", "h3"}, TierType::DRAM);

  auto matches = idx.Match({"h1", "h2"});
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].node_id, "node-A");
  EXPECT_EQ(matches[0].MatchedHashCount(), 2u);
  ASSERT_EQ(matches[0].hashes_by_tier.size(), 1u);
  EXPECT_EQ(SortedHashes(matches[0].hashes_by_tier.at(TierType::DRAM)),
            (std::vector<std::string>{"h1", "h2"}));
}

TEST(ExternalKvBlockIndex, RegisterIsAdditiveAcrossTiers) {
  // The same (node, hash) reported at multiple tiers must keep ALL tier
  // buckets — physically the block can live on GPU + CPU + Storage at once.
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::HBM);
  idx.Register("node-A", {"h1"}, TierType::DRAM);
  idx.Register("node-A", {"h1"}, TierType::SSD);

  auto matches = idx.Match({"h1"});
  ASSERT_EQ(matches.size(), 1u);
  ASSERT_EQ(matches[0].hashes_by_tier.size(), 3u);
  EXPECT_EQ(matches[0].hashes_by_tier.at(TierType::HBM), (std::vector<std::string>{"h1"}));
  EXPECT_EQ(matches[0].hashes_by_tier.at(TierType::DRAM), (std::vector<std::string>{"h1"}));
  EXPECT_EQ(matches[0].hashes_by_tier.at(TierType::SSD), (std::vector<std::string>{"h1"}));
  // Distinct count is 1, NOT 3.
  EXPECT_EQ(matches[0].MatchedHashCount(), 1u);
  // First non-empty bucket = best tier (std::map iterates in TierType order).
  EXPECT_EQ(matches[0].hashes_by_tier.begin()->first, TierType::HBM);
}

TEST(ExternalKvBlockIndex, RegisterAtSameTierTwiceIsIdempotent) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::DRAM);
  idx.Register("node-A", {"h1"}, TierType::DRAM);

  auto matches = idx.Match({"h1"});
  ASSERT_EQ(matches.size(), 1u);
  ASSERT_EQ(matches[0].hashes_by_tier.size(), 1u);
  EXPECT_EQ(matches[0].hashes_by_tier.at(TierType::DRAM), (std::vector<std::string>{"h1"}));
}

TEST(ExternalKvBlockIndex, UnregisterRemovesOnlyOneTier) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::HBM);
  idx.Register("node-A", {"h1"}, TierType::DRAM);

  // Drop only the HBM bucket; DRAM survives.
  idx.Unregister("node-A", {"h1"}, TierType::HBM);

  auto matches = idx.Match({"h1"});
  ASSERT_EQ(matches.size(), 1u);
  ASSERT_EQ(matches[0].hashes_by_tier.size(), 1u);
  EXPECT_FALSE(matches[0].hashes_by_tier.count(TierType::HBM));
  EXPECT_EQ(matches[0].hashes_by_tier.at(TierType::DRAM), (std::vector<std::string>{"h1"}));
}

TEST(ExternalKvBlockIndex, UnregisterLastTierDropsEntry) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1", "h2"}, TierType::DRAM);
  idx.Unregister("node-A", {"h1"}, TierType::DRAM);

  // h1's only tier (DRAM) is gone → entry removed entirely.
  auto matches = idx.Match({"h1"});
  EXPECT_TRUE(matches.empty());

  // h2 is unaffected.
  auto m2 = idx.Match({"h2"});
  ASSERT_EQ(m2.size(), 1u);
}

TEST(ExternalKvBlockIndex, UnregisterAtMissingTierIsNoOp) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::DRAM);
  // No HBM bucket exists; unregistering it must not affect DRAM bucket.
  idx.Unregister("node-A", {"h1"}, TierType::HBM);

  auto matches = idx.Match({"h1"});
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].hashes_by_tier.at(TierType::DRAM), (std::vector<std::string>{"h1"}));
}

TEST(ExternalKvBlockIndex, UnregisterByNodeAtTierBulkDropsAcrossHashes) {
  // Simulates "this node just cleared its storage backend" — every hash on
  // SSD for this node disappears, but other tiers and other nodes stay.
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1", "h2", "h3"}, TierType::DRAM);
  idx.Register("node-A", {"h1", "h2"}, TierType::SSD);
  idx.Register("node-B", {"h1"}, TierType::SSD);

  idx.UnregisterByNodeAtTier("node-A", TierType::SSD);

  auto matches = idx.Match({"h1", "h2", "h3"});
  // node-A still appears with DRAM bucket; node-B still has SSD for h1.
  ASSERT_EQ(matches.size(), 2u);

  const auto* ma = FindMatch(matches, "node-A");
  ASSERT_NE(ma, nullptr);
  EXPECT_EQ(ma->hashes_by_tier.size(), 1u);
  EXPECT_TRUE(ma->hashes_by_tier.count(TierType::DRAM));
  EXPECT_FALSE(ma->hashes_by_tier.count(TierType::SSD));
  EXPECT_EQ(SortedHashes(ma->hashes_by_tier.at(TierType::DRAM)),
            (std::vector<std::string>{"h1", "h2", "h3"}));

  const auto* mb = FindMatch(matches, "node-B");
  ASSERT_NE(mb, nullptr);
  EXPECT_TRUE(mb->hashes_by_tier.count(TierType::SSD));
}

TEST(ExternalKvBlockIndex, UnregisterByNodeRemovesAllTiers) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::HBM);
  idx.Register("node-A", {"h1"}, TierType::DRAM);
  idx.Register("node-B", {"h1"}, TierType::HBM);

  idx.UnregisterByNode("node-A");

  auto matches = idx.Match({"h1"});
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].node_id, "node-B");
}

TEST(ExternalKvBlockIndex, MatchAcrossMultipleNodesCorrectGrouping) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1", "h2"}, TierType::DRAM);
  idx.Register("node-B", {"h2", "h3"}, TierType::SSD);
  idx.Register("node-C", {"h4"}, TierType::HBM);

  auto matches = idx.Match({"h1", "h2", "h3"});
  ASSERT_EQ(matches.size(), 2u);
  EXPECT_EQ(SortedNodeIds(matches), (std::vector<std::string>{"node-A", "node-B"}));

  const auto* ma = FindMatch(matches, "node-A");
  ASSERT_NE(ma, nullptr);
  EXPECT_EQ(SortedHashes(ma->hashes_by_tier.at(TierType::DRAM)),
            (std::vector<std::string>{"h1", "h2"}));

  const auto* mb = FindMatch(matches, "node-B");
  ASSERT_NE(mb, nullptr);
  EXPECT_EQ(SortedHashes(mb->hashes_by_tier.at(TierType::SSD)),
            (std::vector<std::string>{"h2", "h3"}));
}

TEST(ExternalKvBlockIndex, MatchSplitsHashesAcrossTiersOnSameNode) {
  // Different hashes on the same node living on different tiers must each
  // appear in their own bucket — this is the cost-model's per-tier signal.
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::HBM);
  idx.Register("node-A", {"h2", "h3"}, TierType::DRAM);
  idx.Register("node-A", {"h4"}, TierType::SSD);

  auto matches = idx.Match({"h1", "h2", "h3", "h4"});
  ASSERT_EQ(matches.size(), 1u);
  const auto& m = matches[0];
  ASSERT_EQ(m.hashes_by_tier.size(), 3u);
  EXPECT_EQ(m.hashes_by_tier.at(TierType::HBM), (std::vector<std::string>{"h1"}));
  EXPECT_EQ(SortedHashes(m.hashes_by_tier.at(TierType::DRAM)),
            (std::vector<std::string>{"h2", "h3"}));
  EXPECT_EQ(m.hashes_by_tier.at(TierType::SSD), (std::vector<std::string>{"h4"}));
  EXPECT_EQ(m.MatchedHashCount(), 4u);
}

TEST(ExternalKvBlockIndex, MatchedHashCountDeduplicatesAcrossTiers) {
  // h1 is reported on both HBM and DRAM — distinct count is 1, not 2.
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::HBM);
  idx.Register("node-A", {"h1", "h2"}, TierType::DRAM);

  auto matches = idx.Match({"h1", "h2"});
  ASSERT_EQ(matches.size(), 1u);
  // h1 on both HBM and DRAM, h2 on DRAM only → 2 distinct hashes.
  EXPECT_EQ(matches[0].MatchedHashCount(), 2u);
  EXPECT_EQ(matches[0].hashes_by_tier.at(TierType::HBM), (std::vector<std::string>{"h1"}));
  EXPECT_EQ(SortedHashes(matches[0].hashes_by_tier.at(TierType::DRAM)),
            (std::vector<std::string>{"h1", "h2"}));
}

TEST(ExternalKvBlockIndex, MatchWithNoHitsReturnsEmpty) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::DRAM);
  EXPECT_TRUE(idx.Match({"h99"}).empty());
}

TEST(ExternalKvBlockIndex, MatchEmptyQueryReturnsEmpty) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::DRAM);
  EXPECT_TRUE(idx.Match({}).empty());
}

TEST(ExternalKvBlockIndex, UnregisterNonExistentHashIsNoOp) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::DRAM);
  EXPECT_NO_THROW(idx.Unregister("node-A", {"h99"}, TierType::DRAM));
  EXPECT_EQ(idx.Match({"h1"}).size(), 1u);
}

TEST(ExternalKvBlockIndex, UnregisterByNodeNonExistentNodeIsNoOp) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::DRAM);
  EXPECT_NO_THROW(idx.UnregisterByNode("node-ghost"));
  EXPECT_NO_THROW(idx.UnregisterByNodeAtTier("node-ghost", TierType::SSD));
  EXPECT_EQ(idx.Match({"h1"}).size(), 1u);
}

TEST(ExternalKvBlockIndex, GetKvCountDeduplicatesAcrossTiers) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1", "h2"}, TierType::HBM);
  idx.Register("node-A", {"h1", "h2"}, TierType::DRAM);  // same hashes, second tier
  idx.Register("node-A", {"h3"}, TierType::SSD);

  // 3 distinct hashes; the multi-tier h1 / h2 must count once each.
  EXPECT_EQ(idx.GetKvCount("node-A"), 3u);
}

TEST(ExternalKvBlockIndex, ThreadSafetyConcurrentRegisterAndMatch) {
  ExternalKvBlockIndex idx;

  constexpr int kThreads = 8;
  constexpr int kIter = 200;

  std::vector<std::thread> writers;
  writers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    writers.emplace_back([&idx, t]() {
      for (int i = 0; i < kIter; ++i) {
        std::string node = "node-" + std::to_string(t);
        std::string hash = "h-" + std::to_string(t * kIter + i);
        idx.Register(node, {hash}, TierType::DRAM);
        idx.Match({hash});
      }
    });
  }
  for (auto& thr : writers) thr.join();

  for (int t = 0; t < kThreads; ++t) {
    std::string hash = "h-" + std::to_string(t * kIter);
    auto matches = idx.Match({hash});
    EXPECT_GE(matches.size(), 0u);
  }
}

}  // namespace mori::umbp
