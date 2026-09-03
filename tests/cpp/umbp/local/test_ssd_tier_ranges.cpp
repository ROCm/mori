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
// Ranged (sub-object) reads against the local SSD tier.
//
// The shape under test is the one the sglang tree connector produces: one
// stored object holds a KV page across every layer, and a load asks for the
// byte slices belonging to one group of layers, each into its own buffer.
// Both the direct and buffered variants run, because O_DIRECT changes which
// reads can land in the caller's buffer and which have to bounce through an
// aligned window -- and the results must be indistinguishable.
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "umbp/local/tiers/segment/segment_format.h"
#include "umbp/local/tiers/ssd_tier.h"

namespace mori::umbp {
namespace {

namespace fs = std::filesystem;

// Point this at a real block-backed filesystem to exercise the O_DIRECT path;
// the default /tmp is overlayfs or tmpfs in most containers, which rejects
// O_DIRECT outright and makes the direct variants degrade to buffered.
std::string TestRoot() {
  const char* root = std::getenv("UMBP_TEST_SSD_RANGES_DIR");
  return root && *root ? root : "/tmp";
}

std::string MakeDir(const std::string& name) {
  const std::string dir = TestRoot() + "/umbp_test_ssd_ranges_" + name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

UMBPSsdConfig BaseConfig(const std::string& dir, bool direct_io) {
  UMBPSsdConfig cfg;
  cfg.enabled = true;
  cfg.storage_dir = dir;
  cfg.capacity_bytes = 64ULL * 1024 * 1024;
  cfg.segment_size_bytes = 16ULL * 1024 * 1024;
  cfg.io.backend = UMBPIoBackend::Posix;  // no io_uring dependency in unit tests
  cfg.direct_io = direct_io;
  cfg.verify_crc = true;  // ranged reads must not verify regardless
  cfg.tier_io_threads = 4;
  return cfg;
}

// Byte i of the object is a function of i, so any slice read back can be
// checked against the offset it claims to come from.
std::vector<char> PatternedValue(size_t size, int salt = 0) {
  std::vector<char> value(size);
  for (size_t i = 0; i < size; ++i) {
    value[i] = static_cast<char>((i * 31 + salt * 7) & 0xFF);
  }
  return value;
}

void ExpectSlice(const std::vector<char>& got, const std::vector<char>& object, size_t offset) {
  ASSERT_LE(offset + got.size(), object.size());
  EXPECT_TRUE(std::equal(got.begin(), got.end(), object.begin() + offset))
      << "slice at offset " << offset << " size " << got.size() << " does not match the object";
}

uintptr_t Ptr(std::vector<char>& buffer) { return reinterpret_cast<uintptr_t>(buffer.data()); }

// Skips rather than silently passing when a direct-I/O variant could not get
// direct I/O -- otherwise it re-runs the buffered path under a name that claims
// to cover alignment widening, which is the one part only O_DIRECT reaches.
#define SKIP_IF_DIRECT_UNAVAILABLE(tier, want_direct)                                   \
  do {                                                                                  \
    if ((want_direct) && !(tier).direct_io_active()) {                                  \
      GTEST_SKIP() << "filesystem rejects O_DIRECT; set UMBP_TEST_SSD_RANGES_DIR to a " \
                      "block-backed path to cover the aligned-window read path";        \
    }                                                                                   \
  } while (0)

class SsdTierRangesTest : public ::testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(BufferedAndDirect, SsdTierRangesTest, ::testing::Values(false, true),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "direct" : "buffered";
                         });

// The layer-group case: consecutive slices of one object, each into its own
// destination.  These coalesce into one device read and scatter on the way out,
// so a byte error here means the bounce bookkeeping is wrong.
TEST_P(SsdTierRangesTest, ContiguousSlicesScatterToSeparateBuffers) {
  const std::string dir = MakeDir(GetParam() ? "contig_direct" : "contig_buffered");
  auto cfg = BaseConfig(dir, GetParam());
  SSDTier tier(dir, cfg.capacity_bytes, cfg);
  SKIP_IF_DIRECT_UNAVAILABLE(tier, GetParam());

  constexpr size_t kLayer = 3000;  // deliberately not a kRecordAlign multiple
  constexpr size_t kLayers = 8;
  const auto object = PatternedValue(kLayer * kLayers);
  ASSERT_TRUE(tier.Write("page", object.data(), object.size()));

  std::vector<std::vector<char>> outs(kLayers, std::vector<char>(kLayer, 0));
  std::vector<uintptr_t> ptrs;
  std::vector<size_t> sizes;
  std::vector<size_t> offsets;
  for (size_t layer = 0; layer < kLayers; ++layer) {
    ptrs.push_back(Ptr(outs[layer]));
    sizes.push_back(kLayer);
    offsets.push_back(layer * kLayer);
  }

  ASSERT_EQ(tier.ReadBatchRangesIntoPtr({"page"}, {ptrs}, {sizes}, {offsets}),
            std::vector<bool>({true}));
  for (size_t layer = 0; layer < kLayers; ++layer) {
    ExpectSlice(outs[layer], object, layer * kLayer);
  }
}

// A single group out of the middle of the object, which is what layer-wise
// loading actually issues -- and the case that must NOT read the whole object.
TEST_P(SsdTierRangesTest, MiddleGroupReadsOnlyItsOwnSlices) {
  const std::string dir = MakeDir(GetParam() ? "mid_direct" : "mid_buffered");
  auto cfg = BaseConfig(dir, GetParam());
  SSDTier tier(dir, cfg.capacity_bytes, cfg);
  SKIP_IF_DIRECT_UNAVAILABLE(tier, GetParam());

  constexpr size_t kLayer = 4096;
  constexpr size_t kLayers = 16;
  const auto object = PatternedValue(kLayer * kLayers, 3);
  ASSERT_TRUE(tier.Write("page", object.data(), object.size()));

  std::vector<std::vector<char>> outs(4, std::vector<char>(kLayer, 0));
  std::vector<uintptr_t> ptrs;
  std::vector<size_t> sizes;
  std::vector<size_t> offsets;
  for (size_t j = 0; j < outs.size(); ++j) {
    ptrs.push_back(Ptr(outs[j]));
    sizes.push_back(kLayer);
    offsets.push_back((8 + j) * kLayer);
  }

  ASSERT_EQ(tier.ReadBatchRangesIntoPtr({"page"}, {ptrs}, {sizes}, {offsets}),
            std::vector<bool>({true}));
  for (size_t j = 0; j < outs.size(); ++j) {
    ExpectSlice(outs[j], object, (8 + j) * kLayer);
  }
}

// Ranges given out of order, non-adjacent, and overlapping.  None of these can
// coalesce, so every one becomes its own read; correctness must not depend on
// the caller sorting anything.
TEST_P(SsdTierRangesTest, ShuffledSparseAndOverlappingRangesAreByteCorrect) {
  const std::string dir = MakeDir(GetParam() ? "shuf_direct" : "shuf_buffered");
  auto cfg = BaseConfig(dir, GetParam());
  SSDTier tier(dir, cfg.capacity_bytes, cfg);
  SKIP_IF_DIRECT_UNAVAILABLE(tier, GetParam());

  const auto object = PatternedValue(10000, 5);
  ASSERT_TRUE(tier.Write("page", object.data(), object.size()));

  std::vector<char> tail(500, 0), head(500, 0), overlap(700, 0), lone(1, 0);
  const std::vector<size_t> offsets{9500, 0, 300, 4321};
  ASSERT_EQ(tier.ReadBatchRangesIntoPtr({"page"}, {{Ptr(tail), Ptr(head), Ptr(overlap), Ptr(lone)}},
                                        {{tail.size(), head.size(), overlap.size(), lone.size()}},
                                        {offsets}),
            std::vector<bool>({true}));
  ExpectSlice(tail, object, 9500);
  ExpectSlice(head, object, 0);
  ExpectSlice(overlap, object, 300);
  ExpectSlice(lone, object, 4321);
}

// Reading the whole object through the ranged path must agree with the
// whole-object path byte for byte -- this is the degenerate case the tree
// connector's prefetch strategy uses.
TEST_P(SsdTierRangesTest, FullCoverageMatchesWholeObjectRead) {
  const std::string dir = MakeDir(GetParam() ? "full_direct" : "full_buffered");
  auto cfg = BaseConfig(dir, GetParam());
  SSDTier tier(dir, cfg.capacity_bytes, cfg);
  SKIP_IF_DIRECT_UNAVAILABLE(tier, GetParam());

  const auto object = PatternedValue(20000, 9);
  ASSERT_TRUE(tier.Write("page", object.data(), object.size()));

  std::vector<char> whole(object.size(), 0);
  ASSERT_TRUE(tier.ReadIntoPtr("page", Ptr(whole), whole.size()));

  std::vector<std::vector<char>> parts{std::vector<char>(7000, 0), std::vector<char>(7000, 0),
                                       std::vector<char>(6000, 0)};
  ASSERT_EQ(tier.ReadBatchRangesIntoPtr({"page"}, {{Ptr(parts[0]), Ptr(parts[1]), Ptr(parts[2])}},
                                        {{7000, 7000, 6000}}, {{0, 7000, 14000}}),
            std::vector<bool>({true}));

  std::vector<char> stitched;
  for (const auto& part : parts) stitched.insert(stitched.end(), part.begin(), part.end());
  EXPECT_EQ(stitched, whole);
  EXPECT_EQ(stitched, object);
}

TEST_P(SsdTierRangesTest, ManyKeysInOneBatch) {
  const std::string dir = MakeDir(GetParam() ? "many_direct" : "many_buffered");
  auto cfg = BaseConfig(dir, GetParam());
  SSDTier tier(dir, cfg.capacity_bytes, cfg);
  SKIP_IF_DIRECT_UNAVAILABLE(tier, GetParam());

  constexpr size_t kKeys = 32;
  constexpr size_t kLayer = 1024;
  std::vector<std::string> keys;
  std::vector<std::vector<char>> objects;
  for (size_t k = 0; k < kKeys; ++k) {
    keys.push_back("page_" + std::to_string(k));
    objects.push_back(PatternedValue(kLayer * 4, static_cast<int>(k)));
    ASSERT_TRUE(tier.Write(keys.back(), objects.back().data(), objects.back().size()));
  }

  std::vector<std::vector<char>> outs(kKeys * 2, std::vector<char>(kLayer, 0));
  std::vector<std::vector<uintptr_t>> ptrs(kKeys);
  std::vector<std::vector<size_t>> sizes(kKeys), offsets(kKeys);
  for (size_t k = 0; k < kKeys; ++k) {
    ptrs[k] = {Ptr(outs[k * 2]), Ptr(outs[k * 2 + 1])};
    sizes[k] = {kLayer, kLayer};
    offsets[k] = {kLayer, kLayer * 2};  // layers 1 and 2: adjacent, so they coalesce
  }

  EXPECT_EQ(tier.ReadBatchRangesIntoPtr(keys, ptrs, sizes, offsets),
            std::vector<bool>(kKeys, true));
  for (size_t k = 0; k < kKeys; ++k) {
    ExpectSlice(outs[k * 2], objects[k], kLayer);
    ExpectSlice(outs[k * 2 + 1], objects[k], kLayer * 2);
  }
}

TEST_P(SsdTierRangesTest, ShapeErrorsFailWholeBatch) {
  const std::string dir = MakeDir(GetParam() ? "shape_direct" : "shape_buffered");
  auto cfg = BaseConfig(dir, GetParam());
  SSDTier tier(dir, cfg.capacity_bytes, cfg);
  SKIP_IF_DIRECT_UNAVAILABLE(tier, GetParam());

  const auto object = PatternedValue(4096);
  ASSERT_TRUE(tier.Write("a", object.data(), object.size()));
  ASSERT_TRUE(tier.Write("b", object.data(), object.size()));

  std::vector<char> out_a(64, 0), out_b(64, 0);
  // Two sizes but one pointer for "a": a malformed request, not a miss.
  EXPECT_EQ(tier.ReadBatchRangesIntoPtr({"a", "b"}, {{Ptr(out_a)}, {Ptr(out_b)}}, {{64, 64}, {64}},
                                        {{0}, {0}}),
            std::vector<bool>({false, false}));
}

TEST_P(SsdTierRangesTest, DataErrorsAreIsolatedPerEntry) {
  const std::string dir = MakeDir(GetParam() ? "iso_direct" : "iso_buffered");
  auto cfg = BaseConfig(dir, GetParam());
  SSDTier tier(dir, cfg.capacity_bytes, cfg);
  SKIP_IF_DIRECT_UNAVAILABLE(tier, GetParam());

  const auto object = PatternedValue(4096);
  ASSERT_TRUE(tier.Write("present", object.data(), object.size()));

  std::vector<char> good(64, 0), missing(64, 0), overflow(64, 0);
  EXPECT_EQ(tier.ReadBatchRangesIntoPtr({"present", "absent", "present"},
                                        {{Ptr(good)}, {Ptr(missing)}, {Ptr(overflow)}},
                                        {{64}, {64}, {64}}, {{128}, {0}, {4090}}),
            std::vector<bool>({true, false, false}));
  ExpectSlice(good, object, 128);
}

TEST_P(SsdTierRangesTest, RejectsOverflowZeroAndEmptyEntries) {
  const std::string dir = MakeDir(GetParam() ? "rej_direct" : "rej_buffered");
  auto cfg = BaseConfig(dir, GetParam());
  SSDTier tier(dir, cfg.capacity_bytes, cfg);
  SKIP_IF_DIRECT_UNAVAILABLE(tier, GetParam());

  const auto object = PatternedValue(4096);
  ASSERT_TRUE(tier.Write("key", object.data(), object.size()));
  std::vector<char> out(64, 0);

  EXPECT_EQ(tier.ReadBatchRangesIntoPtr({"key"}, {{Ptr(out)}}, {{64}},
                                        {{std::numeric_limits<size_t>::max() - 8}}),
            std::vector<bool>({false}));
  EXPECT_EQ(tier.ReadBatchRangesIntoPtr({"key"}, {{Ptr(out)}}, {{0}}, {{0}}),
            std::vector<bool>({false}));
  EXPECT_EQ(tier.ReadBatchRangesIntoPtr({"key"}, {{}}, {{}}, {{}}), std::vector<bool>({false}));
  EXPECT_EQ(tier.ReadBatchRangesIntoPtr({}, {}, {}, {}), std::vector<bool>({}));
}

// A value written with checksums on must still be readable through the ranged
// path, which does not (and cannot) verify them.
TEST_P(SsdTierRangesTest, RangedReadIgnoresRecordChecksum) {
  const std::string dir = MakeDir(GetParam() ? "crc_direct" : "crc_buffered");
  auto cfg = BaseConfig(dir, GetParam());
  ASSERT_TRUE(cfg.verify_crc);
  SSDTier tier(dir, cfg.capacity_bytes, cfg);
  SKIP_IF_DIRECT_UNAVAILABLE(tier, GetParam());

  const auto object = PatternedValue(8192, 11);
  ASSERT_TRUE(tier.Write("page", object.data(), object.size()));

  std::vector<char> slice(1000, 0);
  ASSERT_EQ(tier.ReadBatchRangesIntoPtr({"page"}, {{Ptr(slice)}}, {{1000}}, {{2000}}),
            std::vector<bool>({true}));
  ExpectSlice(slice, object, 2000);
}
// The tree connector loads straight into GPU buffers, so the tier has to accept
// a device destination.  read() cannot write one, which is why these have to
// bounce through host memory -- a regression here shows up as a silent miss
// (EFAULT reported as "key not found"), not as a crash.
TEST(SsdTierRangesDeviceTest, ScattersIntoDeviceBuffers) {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    GTEST_SKIP() << "no GPU available";
  }

  const std::string dir = MakeDir("device");
  auto cfg = BaseConfig(dir, /*direct_io=*/false);
  SSDTier tier(dir, cfg.capacity_bytes, cfg);

  constexpr size_t kLayer = 4096;
  constexpr size_t kLayers = 4;
  const auto object = PatternedValue(kLayer * kLayers, 29);
  ASSERT_TRUE(tier.Write("page", object.data(), object.size()));

  void* device_buffer = nullptr;
  ASSERT_EQ(hipMalloc(&device_buffer, kLayer * kLayers), hipSuccess);
  ASSERT_EQ(hipMemset(device_buffer, 0, kLayer * kLayers), hipSuccess);

  std::vector<uintptr_t> ptrs;
  std::vector<size_t> sizes;
  std::vector<size_t> offsets;
  for (size_t layer = 0; layer < kLayers; ++layer) {
    ptrs.push_back(reinterpret_cast<uintptr_t>(device_buffer) + layer * kLayer);
    sizes.push_back(kLayer);
    offsets.push_back(layer * kLayer);
  }

  ASSERT_EQ(tier.ReadBatchRangesIntoPtr({"page"}, {ptrs}, {sizes}, {offsets}),
            std::vector<bool>({true}));

  std::vector<char> host_result(kLayer * kLayers, 0);
  ASSERT_EQ(hipMemcpy(host_result.data(), device_buffer, host_result.size(), hipMemcpyDeviceToHost),
            hipSuccess);
  EXPECT_EQ(host_result, object);
  ASSERT_EQ(hipFree(device_buffer), hipSuccess);
}

// A single non-contiguous slice, which would otherwise take the "read straight
// into the caller" fast path -- the path a device pointer must be kept off.
TEST(SsdTierRangesDeviceTest, SingleSliceIntoDeviceBufferBounces) {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    GTEST_SKIP() << "no GPU available";
  }

  const std::string dir = MakeDir("device_single");
  auto cfg = BaseConfig(dir, /*direct_io=*/false);
  SSDTier tier(dir, cfg.capacity_bytes, cfg);

  const auto object = PatternedValue(20000, 31);
  ASSERT_TRUE(tier.Write("page", object.data(), object.size()));

  constexpr size_t kSlice = 4096;
  void* device_buffer = nullptr;
  ASSERT_EQ(hipMalloc(&device_buffer, kSlice), hipSuccess);
  ASSERT_EQ(hipMemset(device_buffer, 0, kSlice), hipSuccess);

  ASSERT_EQ(tier.ReadBatchRangesIntoPtr({"page"}, {{reinterpret_cast<uintptr_t>(device_buffer)}},
                                        {{kSlice}}, {{8192}}),
            std::vector<bool>({true}));

  std::vector<char> host_result(kSlice, 0);
  ASSERT_EQ(hipMemcpy(host_result.data(), device_buffer, kSlice, hipMemcpyDeviceToHost),
            hipSuccess);
  ExpectSlice(host_result, object, 8192);
  ASSERT_EQ(hipFree(device_buffer), hipSuccess);
}

}  // namespace
}  // namespace mori::umbp
