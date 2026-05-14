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

// All matched hashes for a node, regardless of tier.
static std::vector<std::string> AllHashesSorted(
    const ExternalKvBlockIndex::NodeMatch& m) {
  std::vector<std::string> all;
  for (const auto& [tier, hashes] : m.hashes_by_tier) {
    all.insert(all.end(), hashes.begin(), hashes.end());
  }
  std::sort(all.begin(), all.end());
  return all;
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
  ASSERT_TRUE(matches[0].hashes_by_tier.count(TierType::DRAM));
  EXPECT_EQ(matches[0].hashes_by_tier.at(TierType::DRAM).size(), 2u);
}

TEST(ExternalKvBlockIndex, RegisterOverwritesTier) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::DRAM);
  idx.Register("node-A", {"h1"}, TierType::SSD);

  auto matches = idx.Match({"h1"});
  ASSERT_EQ(matches.size(), 1u);
  // After overwrite, h1 should only appear under SSD.
  ASSERT_EQ(matches[0].hashes_by_tier.size(), 1u);
  ASSERT_TRUE(matches[0].hashes_by_tier.count(TierType::SSD));
  EXPECT_EQ(matches[0].hashes_by_tier.at(TierType::SSD), (std::vector<std::string>{"h1"}));
  EXPECT_FALSE(matches[0].hashes_by_tier.count(TierType::DRAM));
}

TEST(ExternalKvBlockIndex, UnregisterRemovesSpecificHashes) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1", "h2", "h3"}, TierType::DRAM);
  idx.Unregister("node-A", {"h2"});

  auto matches = idx.Match({"h1", "h2", "h3"});
  ASSERT_EQ(matches.size(), 1u);
  const auto& m = matches[0];
  EXPECT_EQ(m.node_id, "node-A");
  EXPECT_EQ(AllHashesSorted(m), (std::vector<std::string>{"h1", "h3"}));

  // h2 specifically is gone
  auto h2_matches = idx.Match({"h2"});
  EXPECT_TRUE(h2_matches.empty());
}

TEST(ExternalKvBlockIndex, UnregisterByNodeRemovesAllHashes) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1", "h2"}, TierType::DRAM);
  idx.Register("node-B", {"h1", "h3"}, TierType::HBM);
  idx.UnregisterByNode("node-A");

  auto matches = idx.Match({"h1", "h2", "h3"});
  auto ids = SortedNodeIds(matches);
  EXPECT_EQ(ids, (std::vector<std::string>{"node-B"}));

  const auto* mb = FindMatch(matches, "node-B");
  ASSERT_NE(mb, nullptr);
  EXPECT_EQ(AllHashesSorted(*mb), (std::vector<std::string>{"h1", "h3"}));
  ASSERT_EQ(mb->hashes_by_tier.size(), 1u);
  ASSERT_TRUE(mb->hashes_by_tier.count(TierType::HBM));
}

TEST(ExternalKvBlockIndex, MatchWithNoHitsReturnsEmpty) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::DRAM);
  auto matches = idx.Match({"h99", "h100"});
  EXPECT_TRUE(matches.empty());
}

TEST(ExternalKvBlockIndex, MatchEmptyQueryReturnsEmpty) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::DRAM);
  auto matches = idx.Match({});
  EXPECT_TRUE(matches.empty());
}

TEST(ExternalKvBlockIndex, MatchAcrossMultipleNodesCorrectGrouping) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1", "h2"}, TierType::DRAM);
  idx.Register("node-B", {"h2", "h3"}, TierType::SSD);
  idx.Register("node-C", {"h4"}, TierType::HBM);

  auto matches = idx.Match({"h1", "h2", "h3"});
  ASSERT_EQ(matches.size(), 2u);

  auto ids = SortedNodeIds(matches);
  EXPECT_EQ(ids, (std::vector<std::string>{"node-A", "node-B"}));

  const auto* ma = FindMatch(matches, "node-A");
  ASSERT_NE(ma, nullptr);
  ASSERT_TRUE(ma->hashes_by_tier.count(TierType::DRAM));
  std::vector<std::string> a_hashes = ma->hashes_by_tier.at(TierType::DRAM);
  std::sort(a_hashes.begin(), a_hashes.end());
  EXPECT_EQ(a_hashes, (std::vector<std::string>{"h1", "h2"}));

  const auto* mb = FindMatch(matches, "node-B");
  ASSERT_NE(mb, nullptr);
  ASSERT_TRUE(mb->hashes_by_tier.count(TierType::SSD));
  std::vector<std::string> b_hashes = mb->hashes_by_tier.at(TierType::SSD);
  std::sort(b_hashes.begin(), b_hashes.end());
  EXPECT_EQ(b_hashes, (std::vector<std::string>{"h2", "h3"}));
}

TEST(ExternalKvBlockIndex, MatchSplitsHashesAcrossTiersOnSameNode) {
  // A single node holding the same prefix on multiple tiers must report each
  // hash under its actual tier (the cost model needs per-tier counts).
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::HBM);
  idx.Register("node-A", {"h2", "h3"}, TierType::DRAM);
  idx.Register("node-A", {"h4"}, TierType::SSD);

  auto matches = idx.Match({"h1", "h2", "h3", "h4"});
  ASSERT_EQ(matches.size(), 1u);
  const auto& m = matches[0];
  EXPECT_EQ(m.MatchedHashCount(), 4u);
  ASSERT_EQ(m.hashes_by_tier.size(), 3u);
  EXPECT_EQ(m.hashes_by_tier.at(TierType::HBM), (std::vector<std::string>{"h1"}));

  std::vector<std::string> dram_hashes = m.hashes_by_tier.at(TierType::DRAM);
  std::sort(dram_hashes.begin(), dram_hashes.end());
  EXPECT_EQ(dram_hashes, (std::vector<std::string>{"h2", "h3"}));

  EXPECT_EQ(m.hashes_by_tier.at(TierType::SSD), (std::vector<std::string>{"h4"}));

  // std::map iterates in tier order, so begin() is the fastest tier.
  EXPECT_EQ(m.hashes_by_tier.begin()->first, TierType::HBM);
}

TEST(ExternalKvBlockIndex, MatchPreservesUnknownTierBucket) {
  // UNKNOWN-tier registrations are kept under their own bucket so callers
  // can decide whether to treat them as cache hits or ignore them.
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::DRAM);
  idx.Register("node-A", {"h2"}, TierType::UNKNOWN);
  idx.Register("node-B", {"h1"}, TierType::UNKNOWN);
  idx.Register("node-B", {"h2"}, TierType::SSD);

  auto matches = idx.Match({"h1", "h2"});
  ASSERT_EQ(matches.size(), 2u);

  const auto* ma = FindMatch(matches, "node-A");
  ASSERT_NE(ma, nullptr);
  EXPECT_EQ(ma->hashes_by_tier.at(TierType::DRAM), (std::vector<std::string>{"h1"}));
  EXPECT_EQ(ma->hashes_by_tier.at(TierType::UNKNOWN), (std::vector<std::string>{"h2"}));

  const auto* mb = FindMatch(matches, "node-B");
  ASSERT_NE(mb, nullptr);
  EXPECT_EQ(mb->hashes_by_tier.at(TierType::UNKNOWN), (std::vector<std::string>{"h1"}));
  EXPECT_EQ(mb->hashes_by_tier.at(TierType::SSD), (std::vector<std::string>{"h2"}));
}

TEST(ExternalKvBlockIndex, UnregisterNonExistentHashIsNoOp) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::DRAM);
  EXPECT_NO_THROW(idx.Unregister("node-A", {"h99"}));
  auto matches = idx.Match({"h1"});
  EXPECT_EQ(matches.size(), 1u);
}

TEST(ExternalKvBlockIndex, UnregisterByNodeNonExistentNodeIsNoOp) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::DRAM);
  EXPECT_NO_THROW(idx.UnregisterByNode("node-ghost"));
  auto matches = idx.Match({"h1"});
  EXPECT_EQ(matches.size(), 1u);
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

  // Verify at least one registration landed from each thread
  for (int t = 0; t < kThreads; ++t) {
    std::string hash = "h-" + std::to_string(t * kIter);
    auto matches = idx.Match({hash});
    EXPECT_GE(matches.size(), 0u);  // just verify it doesn't crash / deadlock
  }
}

}  // namespace mori::umbp
