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
#include <stdexcept>
#include <string>
#include <vector>

#include "umbp/distributed/page_bitmap_allocator.h"
#include "umbp/distributed/types.h"

namespace {

using mori::umbp::AllocResult;
using mori::umbp::PageBitmapAllocator;
using mori::umbp::PageLocation;
using mori::umbp::ParsedDramLocation;
using mori::umbp::ParseDramLocationId;

constexpr uint64_t kPageSize = 1024;  // 1 KB pages — small for easier asserts

// Convenience: build allocator with `num_buffers` buffers of `pages_per_buffer`
// pages each.
PageBitmapAllocator MakeAllocator(size_t num_buffers, uint32_t pages_per_buffer) {
  std::vector<uint64_t> sizes(num_buffers, pages_per_buffer * kPageSize);
  return PageBitmapAllocator(kPageSize, sizes);
}

// Set of pages, for order-insensitive comparisons.
std::set<std::pair<uint32_t, uint32_t>> AsSet(const std::vector<PageLocation>& pages) {
  std::set<std::pair<uint32_t, uint32_t>> s;
  for (const auto& p : pages) s.insert({p.buffer_index, p.page_index});
  return s;
}

// =============================================================================
// PageBitmapAllocatorTest
// =============================================================================

TEST(PageBitmapAllocatorTest, ConstructorWithZeroPageSizeThrows) {
  EXPECT_THROW(PageBitmapAllocator(0, {1024}), std::invalid_argument);
}

TEST(PageBitmapAllocatorTest, ConstructorWithEmptyBuffersOk) {
  PageBitmapAllocator alloc(kPageSize, {});
  EXPECT_EQ(alloc.TotalBytes(), 0u);
  EXPECT_EQ(alloc.AvailableBytes(), 0u);
  EXPECT_EQ(alloc.UsedBytes(), 0u);
  EXPECT_EQ(alloc.NumBuffers(), 0u);

  EXPECT_FALSE(alloc.Allocate(1).has_value());
  EXPECT_FALSE(alloc.Allocate(1024).has_value());
}

TEST(PageBitmapAllocatorTest, ConstructorBufferSizeNotMultipleOfPageSizeWastesRemainder) {
  // 5000 bytes / 1024 = 4 pages (remainder 904 bytes wasted, by design).
  PageBitmapAllocator alloc(kPageSize, {5000});
  EXPECT_EQ(alloc.TotalBytes(), 4u * kPageSize);
  EXPECT_EQ(alloc.NumBuffers(), 1u);
  EXPECT_EQ(alloc.Buffers()[0].total_pages, 4u);
}

TEST(PageBitmapAllocatorTest, AllocateZeroPagesReturnsNullopt) {
  auto alloc = MakeAllocator(/*num_buffers=*/2, /*pages_per_buffer=*/4);
  EXPECT_FALSE(alloc.Allocate(0).has_value());
  EXPECT_EQ(alloc.AvailableBytes(), alloc.TotalBytes());
}

TEST(PageBitmapAllocatorTest, BasicAllocateSinglePageContinuous) {
  auto alloc = MakeAllocator(1, 4);
  uint64_t total = alloc.TotalBytes();

  auto r = alloc.Allocate(1);
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->location_id, "0:p0");
  ASSERT_EQ(r->pages.size(), 1u);
  EXPECT_EQ(r->pages[0], (PageLocation{0, 0}));

  EXPECT_EQ(alloc.AvailableBytes(), total - kPageSize);
  EXPECT_EQ(alloc.UsedBytes(), kPageSize);
}

TEST(PageBitmapAllocatorTest, AllocateContinuousMultiplePagesInOneBuffer) {
  auto alloc = MakeAllocator(2, 8);

  auto r = alloc.Allocate(3);
  ASSERT_TRUE(r.has_value());
  // Strategy 1 hit on buffer 0, starting at page 0.
  EXPECT_EQ(r->location_id, "0:p0,1,2");
  ASSERT_EQ(r->pages.size(), 3u);
  EXPECT_EQ(r->pages[0], (PageLocation{0, 0}));
  EXPECT_EQ(r->pages[1], (PageLocation{0, 1}));
  EXPECT_EQ(r->pages[2], (PageLocation{0, 2}));
}

TEST(PageBitmapAllocatorTest, AllocateDiscreteMultiplePagesInOneBuffer) {
  // Single buffer of 8 pages.  Allocate everything, then free pages 1, 3, 5
  // so the remaining free pages within that buffer are non-contiguous (and
  // there are exactly 3 free pages — no other buffer to fall through to).
  auto alloc = MakeAllocator(1, 8);
  auto all = alloc.Allocate(8);
  ASSERT_TRUE(all.has_value());

  alloc.Deallocate({{0, 1}, {0, 3}, {0, 5}});
  ASSERT_EQ(alloc.AvailableBytes(), 3u * kPageSize);

  // Strategy 1 fails (no contiguous 3-run).  Strategy 2 should pick up the
  // 3 discrete free pages within buffer 0.
  auto r = alloc.Allocate(3);
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->location_id, "0:p1,3,5");
  EXPECT_EQ(AsSet(r->pages), (std::set<std::pair<uint32_t, uint32_t>>{{0, 1}, {0, 3}, {0, 5}}));
  EXPECT_EQ(alloc.AvailableBytes(), 0u);
}

TEST(PageBitmapAllocatorTest, AllocateAcrossBuffersWhenSingleBufferInsufficient) {
  // Two buffers of 2 pages each; ask for 3 → requires Strategy 3.
  auto alloc = MakeAllocator(2, 2);
  uint64_t total = alloc.TotalBytes();

  auto r = alloc.Allocate(3);
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(r->location_id, "0:p0,1;1:p0");
  EXPECT_EQ(r->pages.size(), 3u);
  EXPECT_EQ(AsSet(r->pages), (std::set<std::pair<uint32_t, uint32_t>>{{0, 0}, {0, 1}, {1, 0}}));
  EXPECT_EQ(alloc.AvailableBytes(), total - 3u * kPageSize);
}

TEST(PageBitmapAllocatorTest, AllocateFailsWhenTotalCapacityInsufficient) {
  auto alloc = MakeAllocator(2, 2);  // 4 pages total
  uint64_t total = alloc.TotalBytes();

  auto r = alloc.Allocate(5);
  EXPECT_FALSE(r.has_value());
  // All-or-nothing: no bitmap was modified.
  EXPECT_EQ(alloc.AvailableBytes(), total);
  EXPECT_EQ(alloc.UsedBytes(), 0u);
  for (const auto& b : alloc.Buffers()) {
    EXPECT_EQ(b.free_count, b.total_pages);
    for (bool bit : b.bitmap) EXPECT_FALSE(bit);
  }
}

TEST(PageBitmapAllocatorTest, AllocateAllOrNothingPartialFitFails) {
  // Two buffers of 2 pages each (4 total).  Pre-allocate 2 pages so total
  // free drops to 2.  A subsequent request for 3 must fail entirely without
  // mutating any bitmap (verified by AvailableBytes parity).
  auto alloc = MakeAllocator(2, 2);
  auto a = alloc.Allocate(2);  // Strategy 1 hits buffer 0
  ASSERT_TRUE(a.has_value());
  uint64_t avail_before = alloc.AvailableBytes();

  auto r = alloc.Allocate(3);
  EXPECT_FALSE(r.has_value());
  EXPECT_EQ(alloc.AvailableBytes(), avail_before);
}

TEST(PageBitmapAllocatorTest, DeallocateRoundtrip) {
  auto alloc = MakeAllocator(2, 4);
  uint64_t total = alloc.TotalBytes();

  auto r = alloc.Allocate(5);  // forces Strategy 3 (>4)
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(alloc.AvailableBytes(), total - 5u * kPageSize);

  alloc.Deallocate(r->pages);
  EXPECT_EQ(alloc.AvailableBytes(), total);

  // Re-allocate — should be able to satisfy the same request shape.
  auto r2 = alloc.Allocate(5);
  ASSERT_TRUE(r2.has_value());
  EXPECT_EQ(AsSet(r2->pages), AsSet(r->pages));
}

TEST(PageBitmapAllocatorTest, DeallocateIdempotent) {
  auto alloc = MakeAllocator(1, 4);
  auto r = alloc.Allocate(2);
  ASSERT_TRUE(r.has_value());
  uint64_t avail_after_alloc = alloc.AvailableBytes();

  alloc.Deallocate(r->pages);
  uint64_t avail_after_first_free = alloc.AvailableBytes();
  EXPECT_GT(avail_after_first_free, avail_after_alloc);

  // Second Deallocate of the same set must not corrupt state.
  alloc.Deallocate(r->pages);
  EXPECT_EQ(alloc.AvailableBytes(), avail_after_first_free);
  for (const auto& b : alloc.Buffers()) {
    EXPECT_EQ(b.free_count, b.total_pages);
    for (bool bit : b.bitmap) EXPECT_FALSE(bit);
  }
}

TEST(PageBitmapAllocatorTest, DeallocateOutOfRangeNoOp) {
  auto alloc = MakeAllocator(2, 3);
  uint64_t total = alloc.TotalBytes();
  auto r = alloc.Allocate(2);
  ASSERT_TRUE(r.has_value());
  uint64_t avail_after_alloc = alloc.AvailableBytes();

  // Garbage entries: bad buffer_index, bad page_index, plus a valid free page.
  std::vector<PageLocation> bogus = {
      {99, 0},  // buffer_index out of range
      {0, 99},  // page_index out of range
      {1, 0},   // valid but currently free → idempotent skip
  };
  alloc.Deallocate(bogus);
  // Available unchanged: nothing was mutated.
  EXPECT_EQ(alloc.AvailableBytes(), avail_after_alloc);
  // free_count never underflows below total_pages.
  for (const auto& b : alloc.Buffers()) {
    EXPECT_LE(b.free_count, b.total_pages);
  }

  // Real free still works after the no-op call.
  alloc.Deallocate(r->pages);
  EXPECT_EQ(alloc.AvailableBytes(), total);
}

TEST(PageBitmapAllocatorTest, DeallocateByLocationIdHappy) {
  auto alloc = MakeAllocator(2, 4);
  uint64_t total = alloc.TotalBytes();

  auto r = alloc.Allocate(5);  // cross-buffer
  ASSERT_TRUE(r.has_value());
  EXPECT_LT(alloc.AvailableBytes(), total);

  alloc.DeallocateByLocationId(r->location_id);
  EXPECT_EQ(alloc.AvailableBytes(), total);
}

TEST(PageBitmapAllocatorTest, DeallocateByLocationIdMalformedNoOp) {
  auto alloc = MakeAllocator(1, 4);
  auto r = alloc.Allocate(1);
  ASSERT_TRUE(r.has_value());
  uint64_t avail_after_alloc = alloc.AvailableBytes();

  alloc.DeallocateByLocationId("");                // empty
  alloc.DeallocateByLocationId("not-a-location");  // no colon
  alloc.DeallocateByLocationId("0:x3");            // missing 'p'
  alloc.DeallocateByLocationId("0:p");             // empty page list
  alloc.DeallocateByLocationId("abc:p3");          // non-numeric buffer
  alloc.DeallocateByLocationId("0:pabc");          // non-numeric page

  EXPECT_EQ(alloc.AvailableBytes(), avail_after_alloc);
}

TEST(PageBitmapAllocatorTest, AllocateExhaustionThenSuccess) {
  // Sanity: after exhausting all pages, Allocate(1) fails; after one Deallocate
  // it succeeds again.
  auto alloc = MakeAllocator(1, 3);
  auto a = alloc.Allocate(3);
  ASSERT_TRUE(a.has_value());
  EXPECT_EQ(alloc.AvailableBytes(), 0u);
  EXPECT_FALSE(alloc.Allocate(1).has_value());

  alloc.Deallocate({a->pages[1]});
  auto b = alloc.Allocate(1);
  ASSERT_TRUE(b.has_value());
  EXPECT_EQ(b->pages[0], a->pages[1]);
}

// =============================================================================
// LocationIdSerializationTest
// =============================================================================

TEST(LocationIdSerializationTest, BuildSinglePage) {
  EXPECT_EQ(PageBitmapAllocator::BuildLocationId({{0, 3}}), "0:p3");
}

TEST(LocationIdSerializationTest, BuildSameBufferMultiPages) {
  EXPECT_EQ(PageBitmapAllocator::BuildLocationId({{0, 3}, {0, 4}}), "0:p3,4");
  // Out-of-order input must still produce ascending output.
  EXPECT_EQ(PageBitmapAllocator::BuildLocationId({{0, 4}, {0, 3}}), "0:p3,4");
}

TEST(LocationIdSerializationTest, BuildCrossBuffer) {
  EXPECT_EQ(PageBitmapAllocator::BuildLocationId({{0, 1}, {0, 2}, {1, 0}}), "0:p1,2;1:p0");
  // Out-of-order across buffers — buffer_index also ascending.
  EXPECT_EQ(PageBitmapAllocator::BuildLocationId({{1, 0}, {0, 2}, {0, 1}}), "0:p1,2;1:p0");
}

TEST(LocationIdSerializationTest, BuildEmpty) {
  EXPECT_EQ(PageBitmapAllocator::BuildLocationId({}), "");
}

TEST(LocationIdSerializationTest, RoundtripSinglePage) {
  std::vector<PageLocation> in = {{2, 7}};
  std::string s = PageBitmapAllocator::BuildLocationId(in);
  auto parsed = ParseDramLocationId(s);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->pages, in);
}

TEST(LocationIdSerializationTest, RoundtripCrossBuffer) {
  std::vector<PageLocation> in = {{0, 1}, {0, 2}, {1, 0}, {3, 5}};
  std::string s = PageBitmapAllocator::BuildLocationId(in);
  EXPECT_EQ(s, "0:p1,2;1:p0;3:p5");
  auto parsed = ParseDramLocationId(s);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->pages, in);
}

TEST(LocationIdSerializationTest, ParseEmptyReturnsNullopt) {
  EXPECT_FALSE(ParseDramLocationId("").has_value());
}

TEST(LocationIdSerializationTest, ParseMissingColonReturnsNullopt) {
  EXPECT_FALSE(ParseDramLocationId("0p3").has_value());
  EXPECT_FALSE(ParseDramLocationId("hello").has_value());
}

TEST(LocationIdSerializationTest, ParseMissingPPrefixReturnsNullopt) {
  EXPECT_FALSE(ParseDramLocationId("0:3").has_value());
  EXPECT_FALSE(ParseDramLocationId("0:x3").has_value());
}

TEST(LocationIdSerializationTest, ParseNonNumericReturnsNullopt) {
  EXPECT_FALSE(ParseDramLocationId("abc:p3").has_value());
  EXPECT_FALSE(ParseDramLocationId("0:pabc").has_value());
  EXPECT_FALSE(ParseDramLocationId("0:p3,abc").has_value());
  EXPECT_FALSE(ParseDramLocationId("0:p-1").has_value());
}

TEST(LocationIdSerializationTest, ParseDuplicatePageInSameBufferReturnsNullopt) {
  EXPECT_FALSE(ParseDramLocationId("0:p1,1").has_value());
  EXPECT_FALSE(ParseDramLocationId("0:p1,2,1").has_value());
}

TEST(LocationIdSerializationTest, ParseDuplicateBufferAcrossSegmentsReturnsNullopt) {
  // Canonical serialization always groups one buffer into one segment.
  EXPECT_FALSE(ParseDramLocationId("0:p1;0:p2").has_value());
}

TEST(LocationIdSerializationTest, ParseTrailingSemicolonReturnsNullopt) {
  EXPECT_FALSE(ParseDramLocationId("0:p1;").has_value());
  EXPECT_FALSE(ParseDramLocationId("0:p1;1:p0;").has_value());
  EXPECT_FALSE(ParseDramLocationId(";0:p1").has_value());
  EXPECT_FALSE(ParseDramLocationId("0:p1;;1:p0").has_value());
}

TEST(LocationIdSerializationTest, ParseEmptyPageListReturnsNullopt) {
  EXPECT_FALSE(ParseDramLocationId("0:p").has_value());
  EXPECT_FALSE(ParseDramLocationId("0:p,3").has_value());
  EXPECT_FALSE(ParseDramLocationId("0:p3,").has_value());
}

TEST(LocationIdSerializationTest, ParseWellFormedSingle) {
  auto parsed = ParseDramLocationId("0:p3");
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->pages.size(), 1u);
  EXPECT_EQ(parsed->pages[0], (PageLocation{0, 3}));
}

TEST(LocationIdSerializationTest, ParseWellFormedMulti) {
  auto parsed = ParseDramLocationId("0:p3,4");
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->pages.size(), 2u);
  EXPECT_EQ(parsed->pages[0], (PageLocation{0, 3}));
  EXPECT_EQ(parsed->pages[1], (PageLocation{0, 4}));
}

TEST(LocationIdSerializationTest, ParseWellFormedCrossBuffer) {
  auto parsed = ParseDramLocationId("0:p1,2;1:p0");
  ASSERT_TRUE(parsed.has_value());
  ASSERT_EQ(parsed->pages.size(), 3u);
  EXPECT_EQ(parsed->pages[0], (PageLocation{0, 1}));
  EXPECT_EQ(parsed->pages[1], (PageLocation{0, 2}));
  EXPECT_EQ(parsed->pages[2], (PageLocation{1, 0}));
}

}  // namespace
