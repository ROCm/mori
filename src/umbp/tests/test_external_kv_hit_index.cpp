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

#include <atomic>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/distributed/master/external_kv_hit_index.h"

namespace mori::umbp {
namespace {

std::unordered_map<std::string, uint64_t> LookupMap(ExternalKvHitIndex& index,
                                                    const std::vector<std::string>& hashes) {
  std::vector<std::pair<std::string, uint64_t>> entries;
  index.Lookup(hashes, &entries);
  std::unordered_map<std::string, uint64_t> out;
  for (const auto& [hash, total] : entries) out[hash] = total;
  return out;
}

TEST(ExternalKvHitIndexTest, IncrementAndLookup) {
  ExternalKvHitIndex index;
  index.IncrementHits({"h1", "h2"}, 100);

  auto counts = LookupMap(index, {"h1", "h2", "missing"});
  ASSERT_EQ(counts.size(), 2);
  EXPECT_EQ(counts["h1"], 1);
  EXPECT_EQ(counts["h2"], 1);
}

TEST(ExternalKvHitIndexTest, RepeatedIncrementsAccumulate) {
  ExternalKvHitIndex index;
  for (int i = 0; i < 10; ++i) index.IncrementHits({"hot"}, 100 + i);

  auto counts = LookupMap(index, {"hot"});
  ASSERT_EQ(counts.size(), 1);
  EXPECT_EQ(counts["hot"], 10);
}

TEST(ExternalKvHitIndexTest, LookupSkipsMissingAndDedupesRequestHashes) {
  ExternalKvHitIndex index;
  index.IncrementHits({"h1"}, 100);

  std::vector<std::pair<std::string, uint64_t>> entries;
  index.Lookup({"missing", "h1", "h1", "missing"}, &entries);
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].first, "h1");
  EXPECT_EQ(entries[0].second, 1);
}

TEST(ExternalKvHitIndexTest, GarbageCollectUsesLastSeenCutoff) {
  ExternalKvHitIndex index;
  index.IncrementHits({"old"}, 100);
  index.IncrementHits({"fresh"}, 200);

  EXPECT_EQ(index.GarbageCollect(150), 1);
  EXPECT_EQ(index.Size(), 1);

  auto counts = LookupMap(index, {"old", "fresh"});
  ASSERT_EQ(counts.size(), 1);
  EXPECT_EQ(counts["fresh"], 1);
}

TEST(ExternalKvHitIndexTest, ConcurrentCreationKeepsAllIncrements) {
  ExternalKvHitIndex index;
  constexpr int kThreads = 32;
  constexpr int kIterations = 1000;

  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int i = 0; i < kIterations; ++i) {
        index.IncrementHits({"shared"}, static_cast<uint64_t>(100 + i));
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& thread : threads) thread.join();

  auto counts = LookupMap(index, {"shared"});
  ASSERT_EQ(counts.size(), 1);
  EXPECT_EQ(counts["shared"], static_cast<uint64_t>(kThreads * kIterations));
}

}  // namespace
}  // namespace mori::umbp
