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
#include <vector>

#include "umbp/distributed/master/external_kv_block_index.h"

namespace mori::umbp {
namespace {

const ExternalKvBlockIndex::NodeMatch* FindMatch(
    const std::vector<ExternalKvBlockIndex::NodeMatch>& matches, const std::string& node_id) {
  for (const auto& match : matches) {
    if (match.node_id == node_id) return &match;
  }
  return nullptr;
}

std::vector<std::string> Sorted(std::vector<std::string> values) {
  std::sort(values.begin(), values.end());
  return values;
}

}  // namespace

TEST(ExternalKvBlockIndex, RegisterIsAdditiveAcrossTiersAndCountsMutations) {
  ExternalKvBlockIndex index;

  EXPECT_EQ(index.Register("node-A", {"h1"}, TierType::HBM), 1u);
  EXPECT_EQ(index.Register("node-A", {"h1"}, TierType::DRAM), 1u);
  EXPECT_EQ(index.Register("node-A", {"h1"}, TierType::DRAM), 0u);

  auto matches = index.Match({"h1"});
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].MatchedHashCount(), 1u);
  EXPECT_EQ(matches[0].hashes_by_tier.at(TierType::HBM), std::vector<std::string>({"h1"}));
  EXPECT_EQ(matches[0].hashes_by_tier.at(TierType::DRAM), std::vector<std::string>({"h1"}));
  EXPECT_EQ(index.GetKvCount("node-A"), 1u);
}

TEST(ExternalKvBlockIndex, UnregisterRemovesOnlyRequestedTier) {
  ExternalKvBlockIndex index;
  ASSERT_EQ(index.Register("node-A", {"h1", "h2"}, TierType::HBM), 2u);
  ASSERT_EQ(index.Register("node-A", {"h1"}, TierType::DRAM), 1u);

  EXPECT_EQ(index.Unregister("node-A", {"h1", "missing"}, TierType::HBM), 1u);
  EXPECT_EQ(index.Unregister("node-A", {"h1"}, TierType::HBM), 0u);

  auto matches = index.Match({"h1", "h2"});
  ASSERT_EQ(matches.size(), 1u);
  const auto& match = matches[0];
  EXPECT_EQ(match.hashes_by_tier.at(TierType::DRAM), std::vector<std::string>({"h1"}));
  EXPECT_EQ(match.hashes_by_tier.at(TierType::HBM), std::vector<std::string>({"h2"}));
  EXPECT_EQ(index.GetKvCount("node-A"), 2u);
}

TEST(ExternalKvBlockIndex, BulkUnregisterByTierAndNode) {
  ExternalKvBlockIndex index;
  ASSERT_EQ(index.Register("node-A", {"h1", "h2", "h3"}, TierType::DRAM), 3u);
  ASSERT_EQ(index.Register("node-A", {"h1", "h2"}, TierType::SSD), 2u);
  ASSERT_EQ(index.Register("node-B", {"h1"}, TierType::SSD), 1u);

  EXPECT_EQ(index.UnregisterByNodeAtTier("node-A", TierType::SSD), 2u);
  auto matches = index.Match({"h1", "h2", "h3"});
  ASSERT_EQ(matches.size(), 2u);
  const auto* node_a = FindMatch(matches, "node-A");
  ASSERT_NE(node_a, nullptr);
  ASSERT_EQ(node_a->hashes_by_tier.size(), 1u);
  EXPECT_EQ(Sorted(node_a->hashes_by_tier.at(TierType::DRAM)),
            (std::vector<std::string>{"h1", "h2", "h3"}));
  const auto* node_b = FindMatch(matches, "node-B");
  ASSERT_NE(node_b, nullptr);
  EXPECT_EQ(node_b->hashes_by_tier.at(TierType::SSD), std::vector<std::string>({"h1"}));

  EXPECT_EQ(index.UnregisterByNode("node-A"), 3u);
  matches = index.Match({"h1", "h2", "h3"});
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0].node_id, "node-B");
}

}  // namespace mori::umbp
