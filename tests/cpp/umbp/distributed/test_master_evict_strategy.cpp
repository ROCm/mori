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
// Master-side eviction policy plug-in.  Covers the default
// LruMasterEvictStrategy behavior: oldest-first within each per-(node,tier)
// byte budget.
#include <gtest/gtest.h>

#include <chrono>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "umbp/common/grpc_limits.h"
#include "umbp/distributed/master/evict_strategy.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {
namespace {

// EvictionCandidate::last_accessed_at is a system_clock time_point.
using Clock = std::chrono::system_clock;

EvictionCandidate MakeCandidate(const std::string& key, const std::string& node, TierType tier,
                                uint64_t size, Clock::time_point accessed) {
  EvictionCandidate c;
  c.key = key;
  c.location = Location{node, size, tier};
  c.last_accessed_at = accessed;
  c.size = size;
  return c;
}

TEST(LruMasterEvictStrategy, PicksOldestFirstUntilBudgetMet) {
  LruMasterEvictStrategy strategy;
  auto now = Clock::now();
  // newest -> oldest: c (now), a (now-2s), b (now-1s).  Oldest is `a`.
  std::vector<EvictionCandidate> candidates = {
      MakeCandidate("c", "n1", TierType::DRAM, 100, now),
      MakeCandidate("a", "n1", TierType::DRAM, 100, now - std::chrono::seconds(2)),
      MakeCandidate("b", "n1", TierType::DRAM, 100, now - std::chrono::seconds(1)),
  };
  std::unordered_map<std::string, std::map<TierType, int64_t>> budget;
  budget["n1"][TierType::DRAM] = 150;  // needs 2 victims of 100 each

  auto victims = strategy.SelectVictims(candidates, budget);
  ASSERT_EQ(victims.count("n1"), 1u);
  // Oldest-first: a, then b; c (newest) is spared once the 150-byte budget met.
  ASSERT_EQ(victims["n1"].size(), 2u);
  EXPECT_EQ(victims["n1"][0], "a");
  EXPECT_EQ(victims["n1"][1], "b");
}

TEST(LruMasterEvictStrategy, HonoursPerNodeTierBudgetIndependently) {
  LruMasterEvictStrategy strategy;
  auto now = Clock::now();
  std::vector<EvictionCandidate> candidates = {
      MakeCandidate("n1-old", "n1", TierType::DRAM, 100, now - std::chrono::seconds(5)),
      MakeCandidate("n1-new", "n1", TierType::DRAM, 100, now),
      MakeCandidate("n2-old", "n2", TierType::HBM, 100, now - std::chrono::seconds(5)),
  };
  std::unordered_map<std::string, std::map<TierType, int64_t>> budget;
  budget["n1"][TierType::DRAM] = 50;  // one victim
  budget["n2"][TierType::HBM] = 50;   // one victim

  auto victims = strategy.SelectVictims(candidates, budget);
  ASSERT_EQ(victims["n1"].size(), 1u);
  EXPECT_EQ(victims["n1"][0], "n1-old");
  ASSERT_EQ(victims["n2"].size(), 1u);
  EXPECT_EQ(victims["n2"][0], "n2-old");
}

TEST(LruMasterEvictStrategy, SkipsTiersWithNoBudget) {
  LruMasterEvictStrategy strategy;
  auto now = Clock::now();
  std::vector<EvictionCandidate> candidates = {
      MakeCandidate("k", "n1", TierType::DRAM, 100, now),
  };
  std::unordered_map<std::string, std::map<TierType, int64_t>> budget;  // empty: nothing to free
  auto victims = strategy.SelectVictims(candidates, budget);
  EXPECT_TRUE(victims.empty());
}

// ---------------------------------------------------------------------------
//  gRPC batch chunking (grpc_limits.h)
//
//  The eviction round that exposed this selected 155,157 victims -- 18.7 MB of
//  `repeated string` against a 4 MiB receive limit -- and protobuf refused the
//  whole message, so 81 consecutive rounds freed nothing and the tier could not
//  un-fill itself.  These pin the split, not the transport.
// ---------------------------------------------------------------------------

namespace {
std::vector<std::string> MakeKeys(size_t n, size_t key_len) {
  std::vector<std::string> keys;
  keys.reserve(n);
  for (size_t i = 0; i < n; ++i) keys.emplace_back(key_len, 'k');
  return keys;
}
}  // namespace

TEST(GrpcBatchChunkingTest, SmallListGoesInOneChunk) {
  const auto keys = MakeKeys(100, 117);
  EXPECT_EQ(GrpcMaxItemsPerBatch(keys, 0), 100u);
}

TEST(GrpcBatchChunkingTest, EveryChunkFitsUnderTheLimit) {
  // 155,157 keys of the length the real deployment used.
  const auto keys = MakeKeys(155157, 117);
  const size_t limit = GrpcMaxMessageBytes();

  size_t sent = 0, chunks = 0;
  while (sent < keys.size()) {
    const size_t take = GrpcMaxItemsPerBatch(keys, sent);
    ASSERT_GT(take, 0u) << "a zero-sized chunk would loop forever";

    size_t bytes = 0;
    for (size_t i = sent; i < sent + take; ++i) bytes += keys[i].size() + 8;
    EXPECT_LE(bytes, limit) << "chunk " << chunks << " would be refused";

    sent += take;
    ++chunks;
  }
  EXPECT_EQ(sent, keys.size()) << "chunking must cover every key exactly once";
  EXPECT_GE(chunks, 1u);
}

TEST(GrpcBatchChunkingTest, OversizedSingleItemStillMakesProgress) {
  // One key larger than the whole budget: returning 0 here would spin the
  // dispatch loop forever, which is worse than letting the RPC fail.
  const auto keys = MakeKeys(1, static_cast<size_t>(GrpcMaxMessageBytes()) + 1024);
  EXPECT_EQ(GrpcMaxItemsPerBatch(keys, 0), 1u);
}

TEST(GrpcBatchChunkingTest, LimitHasAFloorAndADefault) {
  // The default must not be below what the master surfaces already used, or
  // this change would silently lower a working limit.
  EXPECT_GE(GrpcMaxMessageBytes(), kMinGrpcMaxMessageBytes);
  EXPECT_GE(kDefaultGrpcMaxMessageBytes, 64u * 1024u * 1024u);
}

}  // namespace
}  // namespace mori::umbp
