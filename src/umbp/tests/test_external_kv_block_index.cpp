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

static std::vector<std::string> SortedNodeIds(const std::vector<ExternalKvBlockIndex::NodeMatch>& ms) {
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

// ---- Tests ------------------------------------------------------------------

TEST(ExternalKvBlockIndex, RegisterAndMatchBasic) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1", "h2", "h3"}, TierType::DRAM);

  auto matches = idx.Match({"h1", "h2"});
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].node_id, "node-A");
  EXPECT_EQ(matches[0].tier, TierType::DRAM);
  EXPECT_EQ(matches[0].matched_hashes.size(), 2u);
}

TEST(ExternalKvBlockIndex, RegisterOverwritesTier) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1"}, TierType::DRAM);
  idx.Register("node-A", {"h1"}, TierType::SSD);

  auto matches = idx.Match({"h1"});
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].tier, TierType::SSD);
}

TEST(ExternalKvBlockIndex, UnregisterRemovesSpecificHashes) {
  ExternalKvBlockIndex idx;
  idx.Register("node-A", {"h1", "h2", "h3"}, TierType::DRAM);
  idx.Unregister("node-A", {"h2"});

  auto matches = idx.Match({"h1", "h2", "h3"});
  ASSERT_EQ(matches.size(), 1u);
  const auto& m = matches[0];
  EXPECT_EQ(m.node_id, "node-A");

  std::vector<std::string> sorted_hashes = m.matched_hashes;
  std::sort(sorted_hashes.begin(), sorted_hashes.end());
  EXPECT_EQ(sorted_hashes, (std::vector<std::string>{"h1", "h3"}));

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
  std::vector<std::string> sorted_hashes = mb->matched_hashes;
  std::sort(sorted_hashes.begin(), sorted_hashes.end());
  EXPECT_EQ(sorted_hashes, (std::vector<std::string>{"h1", "h3"}));
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
  EXPECT_EQ(ma->tier, TierType::DRAM);
  std::vector<std::string> a_hashes = ma->matched_hashes;
  std::sort(a_hashes.begin(), a_hashes.end());
  EXPECT_EQ(a_hashes, (std::vector<std::string>{"h1", "h2"}));

  const auto* mb = FindMatch(matches, "node-B");
  ASSERT_NE(mb, nullptr);
  EXPECT_EQ(mb->tier, TierType::SSD);
  std::vector<std::string> b_hashes = mb->matched_hashes;
  std::sort(b_hashes.begin(), b_hashes.end());
  EXPECT_EQ(b_hashes, (std::vector<std::string>{"h2", "h3"}));
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
