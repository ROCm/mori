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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "umbp/local/tiers/dram_tier.h"

namespace mori::umbp {
namespace {

TEST(DRAMTierConcurrencyTest, MixedBatchReadsWritesAndEvictionsNeverTearData) {
  constexpr size_t kKeyCount = 16;
  constexpr size_t kValueSize = 4096;
  constexpr int kReaderCount = 6;
  constexpr int kIterations = 100;

  // Keep the test's internal worker count modest: concurrency under test is
  // between tier calls, not the CopyJobs implementation inside one call.
  setenv("UMBP_DRAM_READ_THREADS", "2", 1);
  setenv("UMBP_DRAM_WRITE_THREADS", "2", 1);

  DRAMTier tier(2 * kKeyCount * kValueSize);
  std::vector<std::string> keys;
  std::vector<size_t> sizes(kKeyCount, kValueSize);
  keys.reserve(kKeyCount);
  for (size_t i = 0; i < kKeyCount; ++i) keys.push_back("concurrent-key-" + std::to_string(i));

  std::vector<std::vector<unsigned char>> patterns[2];
  for (int generation = 0; generation < 2; ++generation) {
    patterns[generation].reserve(kKeyCount);
    for (size_t i = 0; i < kKeyCount; ++i) {
      const auto value = static_cast<unsigned char>(0x20 + generation * 0x40 + i);
      patterns[generation].emplace_back(kValueSize, value);
    }
  }

  auto write_generation = [&](int generation) {
    std::vector<const void*> sources;
    sources.reserve(kKeyCount);
    for (const auto& pattern : patterns[generation]) sources.push_back(pattern.data());
    return tier.BatchWrite(keys, sources, sizes);
  };

  const auto initial = write_generation(0);
  ASSERT_EQ(initial.size(), kKeyCount);
  for (bool ok : initial) ASSERT_TRUE(ok);

  std::atomic<int> readers_ready{0};
  std::atomic<bool> start{false};
  std::atomic<bool> failed{false};
  std::vector<std::thread> readers;
  readers.reserve(kReaderCount);

  for (int reader_id = 0; reader_id < kReaderCount; ++reader_id) {
    readers.emplace_back([&]() {
      std::vector<std::vector<unsigned char>> buffers(kKeyCount,
                                                      std::vector<unsigned char>(kValueSize));
      std::vector<uintptr_t> destinations;
      destinations.reserve(kKeyCount);
      for (auto& buffer : buffers) {
        destinations.push_back(reinterpret_cast<uintptr_t>(buffer.data()));
      }

      readers_ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) std::this_thread::yield();

      for (int iteration = 0; iteration < kIterations && !failed.load(); ++iteration) {
        const auto results = tier.ReadBatchIntoPtr(keys, destinations, sizes);
        if (results.size() != kKeyCount) {
          failed.store(true);
          break;
        }
        for (size_t i = 0; i < kKeyCount; ++i) {
          if (!results[i]) continue;  // A concurrent eviction is a clean miss.
          const unsigned char observed = buffers[i][0];
          const unsigned char expected0 = patterns[0][i][0];
          const unsigned char expected1 = patterns[1][i][0];
          if (observed != expected0 && observed != expected1) {
            failed.store(true);
            break;
          }
          for (unsigned char byte : buffers[i]) {
            if (byte != observed) {
              failed.store(true);
              break;
            }
          }
        }
      }
    });
  }

  while (readers_ready.load(std::memory_order_acquire) != kReaderCount) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);

  for (int iteration = 0; iteration < kIterations && !failed.load(); ++iteration) {
    if (iteration % 7 == 0) tier.Evict(keys[static_cast<size_t>(iteration) % kKeyCount]);
    const auto results = write_generation(iteration & 1);
    if (results.size() != kKeyCount) {
      failed.store(true);
      break;
    }
    for (bool ok : results) {
      if (!ok) failed.store(true);
    }
  }

  for (auto& reader : readers) reader.join();
  ASSERT_FALSE(failed.load());

  const auto [used, capacity] = tier.Capacity();
  EXPECT_EQ(used, kKeyCount * kValueSize);
  EXPECT_EQ(capacity, 2 * kKeyCount * kValueSize);

  const auto candidates = tier.GetLRUCandidates(kKeyCount);
  EXPECT_EQ(candidates.size(), kKeyCount);
  EXPECT_EQ(std::unordered_set<std::string>(candidates.begin(), candidates.end()).size(),
            kKeyCount);
  for (const auto& key : keys) EXPECT_TRUE(tier.Exists(key));
}

}  // namespace
}  // namespace mori::umbp
