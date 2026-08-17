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
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "device_copy_run.h"
#include "umbp/local/tiers/dram_tier.h"
#include "umbp/local/tiers/local_storage_manager.h"

namespace mori::umbp {
namespace {

std::vector<const void*> ConstPtrs(const std::vector<std::vector<char>>& buffers) {
  std::vector<const void*> ptrs;
  ptrs.reserve(buffers.size());
  for (const auto& buffer : buffers) ptrs.push_back(buffer.data());
  return ptrs;
}

struct PlannerJob {
  void* dst;
  const void* src;
  size_t size;
};

TEST(DeviceCopyRunPlannerTest, MixedSizeAdjacentJobsStayOnLinearPath) {
  char src[64]{};
  char dst[64]{};
  std::vector<PlannerJob> jobs{{dst, src, 8}, {dst + 8, src + 8, 16}, {dst + 24, src + 24, 4}};
  std::vector<PlannerJob*> ordered{&jobs[0], &jobs[1], &jobs[2]};

  const auto run = detail::FindDeviceCopyRun(ordered, 0);
  EXPECT_EQ(run.kind, detail::DeviceCopyRunKind::kLinear);
  EXPECT_EQ(run.end, 3u);
  EXPECT_EQ(run.bytes, 28u);
}

TEST(DeviceCopyRunPlannerTest, UniformStridedJobsUseOnePitchedRun) {
  char src[96]{};
  char dst[144]{};
  std::vector<PlannerJob> jobs{{dst, src, 16}, {dst + 48, src + 32, 16}, {dst + 96, src + 64, 16}};
  std::vector<PlannerJob*> ordered{&jobs[0], &jobs[1], &jobs[2]};

  const auto run = detail::FindDeviceCopyRun(ordered, 0);
  EXPECT_EQ(run.kind, detail::DeviceCopyRunKind::kPitched);
  EXPECT_EQ(run.end, 3u);
  EXPECT_EQ(run.bytes, 48u);
  EXPECT_EQ(run.width, 16u);
  EXPECT_EQ(run.spitch, 32u);
  EXPECT_EQ(run.dpitch, 48u);
}

TEST(DeviceCopyRunPlannerTest, MixedSizeStridedJobsFallBackPerRange) {
  char src[96]{};
  char dst[120]{};
  std::vector<PlannerJob> jobs{{dst, src, 8}, {dst + 40, src + 32, 16}, {dst + 80, src + 64, 8}};
  std::vector<PlannerJob*> ordered{&jobs[0], &jobs[1], &jobs[2]};

  size_t submissions = 0;
  for (size_t begin = 0; begin < ordered.size();) {
    const auto run = detail::FindDeviceCopyRun(ordered, begin);
    EXPECT_EQ(run.kind, detail::DeviceCopyRunKind::kSingle);
    ++submissions;
    begin = run.end;
  }
  EXPECT_EQ(submissions, 3u);
}

TEST(DRAMTierRangesTest, ShuffledPutAndOverlappingGetAreByteCorrect) {
  DRAMTier tier(4096);
  std::vector<std::vector<char>> pieces{
      {'B', 'B', 'B', 'B'}, {'C', 'C', 'C', 'C'}, {'A', 'A', 'A', 'A'}};
  auto put = tier.BatchWriteRanges({"key"}, {12}, {ConstPtrs(pieces)}, {{4, 4, 4}}, {{4, 8, 0}});
  ASSERT_EQ(put, std::vector<bool>({true}));

  std::vector<char> suffix(4, 0);
  std::vector<char> prefix_a(4, 0);
  std::vector<char> prefix_b(4, 0);
  auto get = tier.ReadBatchRangesIntoPtr(
      {"key"},
      {{reinterpret_cast<uintptr_t>(suffix.data()), reinterpret_cast<uintptr_t>(prefix_a.data()),
        reinterpret_cast<uintptr_t>(prefix_b.data())}},
      {{4, 4, 4}}, {{8, 0, 0}});
  ASSERT_EQ(get, std::vector<bool>({true}));
  EXPECT_EQ(suffix, std::vector<char>(4, 'C'));
  EXPECT_EQ(prefix_a, std::vector<char>(4, 'A'));
  EXPECT_EQ(prefix_b, std::vector<char>(4, 'A'));
}

TEST(DRAMTierRangesTest, ShapeErrorsFailWholeBatch) {
  DRAMTier tier(4096);
  std::vector<char> value(8, 'x');
  ASSERT_TRUE(tier.Write("a", value.data(), value.size()));
  ASSERT_TRUE(tier.Write("b", value.data(), value.size()));
  std::vector<char> out_a(4), out_b(4);

  auto get = tier.ReadBatchRangesIntoPtr(
      {"a", "b"},
      {{reinterpret_cast<uintptr_t>(out_a.data())}, {reinterpret_cast<uintptr_t>(out_b.data())}},
      {{4, 4}, {4}}, {{0}, {0}});
  EXPECT_EQ(get, std::vector<bool>({false, false}));

  auto put = tier.BatchWriteRanges({"c", "d"}, {8}, {{value.data()}, {value.data()}}, {{8}, {8}},
                                   {{0}, {0}});
  EXPECT_EQ(put, std::vector<bool>({false, false}));
}

TEST(DRAMTierRangesTest, DataErrorsAreIsolatedPerEntry) {
  DRAMTier tier(4096);
  std::vector<char> value(8, 'v');
  ASSERT_TRUE(tier.Write("present", value.data(), value.size()));
  std::vector<char> good(4, 0), missing(4, 0), overflow(4, 0);

  auto result = tier.ReadBatchRangesIntoPtr({"present", "missing", "present"},
                                            {{reinterpret_cast<uintptr_t>(good.data())},
                                             {reinterpret_cast<uintptr_t>(missing.data())},
                                             {reinterpret_cast<uintptr_t>(overflow.data())}},
                                            {{4}, {4}, {4}}, {{2}, {0}, {6}});
  EXPECT_EQ(result, std::vector<bool>({true, false, false}));
  EXPECT_EQ(good, std::vector<char>(4, 'v'));
}

TEST(DRAMTierRangesTest, RejectsOverflowZeroAndEmptyEntries) {
  DRAMTier tier(4096);
  std::vector<char> value(16, 'z');
  ASSERT_TRUE(tier.Write("key", value.data(), value.size()));
  std::vector<char> out(64, 0);

  auto overflow = tier.ReadBatchRangesIntoPtr({"key"}, {{reinterpret_cast<uintptr_t>(out.data())}},
                                              {{64}}, {{std::numeric_limits<size_t>::max() - 8}});
  EXPECT_EQ(overflow, std::vector<bool>({false}));

  auto zero = tier.ReadBatchRangesIntoPtr({"key"}, {{reinterpret_cast<uintptr_t>(out.data())}},
                                          {{0}}, {{0}});
  EXPECT_EQ(zero, std::vector<bool>({false}));

  auto empty_get = tier.ReadBatchRangesIntoPtr({"key"}, {{}}, {{}}, {{}});
  EXPECT_EQ(empty_get, std::vector<bool>({false}));
  auto empty_put = tier.BatchWriteRanges({"empty"}, {0}, {{}}, {{}}, {{}});
  EXPECT_EQ(empty_put, std::vector<bool>({false}));
  EXPECT_FALSE(tier.Exists("empty"));
}

TEST(DRAMTierRangesTest, MissingTrailingRangeNeverReplacesExistingValue) {
  DRAMTier tier(4096);
  std::vector<char> original(12, 'o');
  ASSERT_TRUE(tier.Write("key", original.data(), original.size()));
  std::vector<std::vector<char>> pieces{{'n', 'n', 'n', 'n'}, {'m', 'm', 'm', 'm'}};

  // Keep object_size independent from emitted ranges; this is the §4.1 guard
  // against silently publishing [0, 8) as a complete 12-byte object.
  auto short_put = tier.BatchWriteRanges({"key"}, {12}, {ConstPtrs(pieces)}, {{4, 4}}, {{0, 4}});
  EXPECT_EQ(short_put, std::vector<bool>({false}));

  std::vector<char> readback(12, 0);
  ASSERT_TRUE(tier.ReadIntoPtr("key", reinterpret_cast<uintptr_t>(readback.data()), 12));
  EXPECT_EQ(readback, original);
}

TEST(DRAMTierRangesTest, AllocationFailureNeverReplacesExistingValue) {
  DRAMTier tier(12);
  std::vector<char> original(12, 'o');
  ASSERT_TRUE(tier.Write("key", original.data(), original.size()));
  std::vector<std::vector<char>> replacement{{'a', 'a', 'a', 'a', 'a', 'a'},
                                             {'b', 'b', 'b', 'b', 'b', 'b'}};

  auto put = tier.BatchWriteRanges({"key"}, {12}, {ConstPtrs(replacement)}, {{6, 6}}, {{0, 6}});
  EXPECT_EQ(put, std::vector<bool>({false}));

  std::vector<char> readback(12, 0);
  ASSERT_TRUE(tier.ReadIntoPtr("key", reinterpret_cast<uintptr_t>(readback.data()), 12));
  EXPECT_EQ(readback, original);
}

TEST(LocalStorageManagerRangesTest, FullTierEvictsAndRetriesWholeEntry) {
  UMBPConfig config;
  config.dram.capacity_bytes = 16;
  config.ssd.enabled = false;
  LocalStorageManager manager(config);
  std::vector<char> old_value(8, 'o');
  ASSERT_TRUE(manager.Write("old", old_value.data(), old_value.size()));

  std::vector<std::vector<char>> pieces{{'a', 'a', 'a', 'a', 'a', 'a'},
                                        {'b', 'b', 'b', 'b', 'b', 'b'}};
  std::vector<std::vector<uintptr_t>> ptrs{{reinterpret_cast<uintptr_t>(pieces[0].data()),
                                            reinterpret_cast<uintptr_t>(pieces[1].data())}};
  auto put = manager.WriteBatchRangesFromPtr({"new"}, {12}, ptrs, {{6, 6}}, {{0, 6}});
  ASSERT_EQ(put, std::vector<bool>({true}));
  EXPECT_FALSE(manager.Exists("old"));
  EXPECT_TRUE(manager.Exists("new"));

  std::vector<char> out_a(6, 0), out_b(6, 0);
  auto get = manager.ReadBatchRangesIntoPtr(
      {"new"},
      {{reinterpret_cast<uintptr_t>(out_a.data()), reinterpret_cast<uintptr_t>(out_b.data())}},
      {{6, 6}}, {{0, 6}});
  ASSERT_EQ(get, std::vector<bool>({true}));
  EXPECT_EQ(out_a, pieces[0]);
  EXPECT_EQ(out_b, pieces[1]);
}

TEST(LocalStorageManagerRangesTest, InvalidEntryDoesNotTriggerEvictionRetry) {
  UMBPConfig config;
  config.dram.capacity_bytes = 16;
  config.ssd.enabled = false;
  LocalStorageManager manager(config);
  std::vector<char> old_value(16, 'o');
  ASSERT_TRUE(manager.Write("old", old_value.data(), old_value.size()));

  std::vector<char> incomplete(8, 'n');
  auto put = manager.WriteBatchRangesFromPtr(
      {"invalid"}, {12}, {{reinterpret_cast<uintptr_t>(incomplete.data())}}, {{8}}, {{0}});
  EXPECT_EQ(put, std::vector<bool>({false}));
  EXPECT_TRUE(manager.Exists("old"));
  EXPECT_FALSE(manager.Exists("invalid"));
}

TEST(DRAMTierRangesTest, DuplicateKeysAreRejectedConsistentlyAcrossLayers) {
  std::vector<char> first(4, 'a');
  std::vector<char> second(4, 'b');
  std::vector<char> unique(4, 'u');

  DRAMTier tier(64);
  auto tier_result = tier.BatchWriteRanges({"dup", "dup", "unique"}, {4, 4, 4},
                                           {{first.data()}, {second.data()}, {unique.data()}},
                                           {{4}, {4}, {4}}, {{0}, {0}, {0}});
  EXPECT_EQ(tier_result, std::vector<bool>({false, false, true}));
  EXPECT_FALSE(tier.Exists("dup"));
  EXPECT_TRUE(tier.Exists("unique"));

  UMBPConfig config;
  config.dram.capacity_bytes = 64;
  config.ssd.enabled = false;
  LocalStorageManager manager(config);
  auto manager_result =
      manager.WriteBatchRangesFromPtr({"dup", "dup", "unique"}, {4, 4, 4},
                                      {{reinterpret_cast<uintptr_t>(first.data())},
                                       {reinterpret_cast<uintptr_t>(second.data())},
                                       {reinterpret_cast<uintptr_t>(unique.data())}},
                                      {{4}, {4}, {4}}, {{0}, {0}, {0}});
  EXPECT_EQ(manager_result, std::vector<bool>({false, false, true}));
  EXPECT_FALSE(manager.Exists("dup"));
  EXPECT_TRUE(manager.Exists("unique"));
}

TEST(DRAMTierRangesTest, DevicePitchedAndMixedWidthCopiesAreByteCorrect) {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    (void)hipGetLastError();
    GTEST_SKIP() << "No HIP device available";
  }

  constexpr size_t kRows = 3;
  constexpr size_t kWidth = 16;
  constexpr size_t kPitch = 32;
  std::vector<unsigned char> host(kRows * kPitch, 0);
  std::vector<char> expected(kRows * kWidth, 0);
  for (size_t row = 0; row < kRows; ++row) {
    for (size_t col = 0; col < kWidth; ++col) {
      const unsigned char value = static_cast<unsigned char>(row * 32 + col);
      host[row * kPitch + col] = value;
      expected[row * kWidth + col] = static_cast<char>(value);
    }
  }

  void* device = nullptr;
  ASSERT_EQ(hipMalloc(&device, host.size()), hipSuccess);
  ASSERT_EQ(hipMemcpy(device, host.data(), host.size(), hipMemcpyHostToDevice), hipSuccess);

  DRAMTier tier(4096);
  std::vector<const void*> sources;
  for (size_t row = 0; row < kRows; ++row) {
    sources.push_back(static_cast<const unsigned char*>(device) + row * kPitch);
  }
  ASSERT_EQ(tier.BatchWriteRanges({"pitched"}, {expected.size()}, {sources},
                                  {{kWidth, kWidth, kWidth}}, {{0, kWidth, 2 * kWidth}}),
            std::vector<bool>({true}));
  std::vector<char> readback(expected.size(), 0);
  ASSERT_TRUE(
      tier.ReadIntoPtr("pitched", reinterpret_cast<uintptr_t>(readback.data()), readback.size()));
  EXPECT_EQ(readback, expected);

  ASSERT_EQ(hipMemset(device, 0, host.size()), hipSuccess);
  std::vector<uintptr_t> destinations;
  for (size_t row = 0; row < kRows; ++row) {
    destinations.push_back(
        reinterpret_cast<uintptr_t>(static_cast<unsigned char*>(device) + row * kPitch));
  }
  ASSERT_EQ(tier.ReadBatchRangesIntoPtr({"pitched"}, {destinations}, {{kWidth, kWidth, kWidth}},
                                        {{0, kWidth, 2 * kWidth}}),
            std::vector<bool>({true}));
  ASSERT_EQ(hipMemcpy(host.data(), device, host.size(), hipMemcpyDeviceToHost), hipSuccess);
  for (size_t row = 0; row < kRows; ++row) {
    EXPECT_EQ(std::memcmp(host.data() + row * kPitch, expected.data() + row * kWidth, kWidth), 0);
  }

  // Different row widths cannot use the 2D path. Keep the source strided so
  // they also cannot accidentally form a flat 1D run.
  const std::vector<size_t> mixed_sizes{8, 16, 8};
  const std::vector<size_t> mixed_offsets{0, 8, 24};
  std::vector<char> mixed_expected(32, 0);
  std::fill(host.begin(), host.end(), 0);
  size_t cursor = 0;
  for (size_t row = 0; row < kRows; ++row) {
    for (size_t col = 0; col < mixed_sizes[row]; ++col) {
      const unsigned char value = static_cast<unsigned char>(0x80 + cursor + col);
      host[row * kPitch + col] = value;
      mixed_expected[cursor + col] = static_cast<char>(value);
    }
    cursor += mixed_sizes[row];
  }
  ASSERT_EQ(hipMemcpy(device, host.data(), host.size(), hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(tier.BatchWriteRanges({"mixed"}, {mixed_expected.size()}, {sources}, {mixed_sizes},
                                  {mixed_offsets}),
            std::vector<bool>({true}));
  std::vector<char> mixed_readback(mixed_expected.size(), 0);
  ASSERT_TRUE(tier.ReadIntoPtr("mixed", reinterpret_cast<uintptr_t>(mixed_readback.data()),
                               mixed_readback.size()));
  EXPECT_EQ(mixed_readback, mixed_expected);

  EXPECT_EQ(hipFree(device), hipSuccess);
}

}  // namespace
}  // namespace mori::umbp
