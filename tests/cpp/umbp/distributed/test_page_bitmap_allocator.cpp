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

#include <set>
#include <stdexcept>
#include <vector>

#include "umbp/distributed/peer/peer_page_allocator.h"
#include "umbp/distributed/types.h"

namespace {

using mori::umbp::PageBitmapAllocator;
using mori::umbp::PageLocation;

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
  ASSERT_EQ(r->size(), 1u);
  EXPECT_EQ((*r)[0], (PageLocation{0, 0}));

  EXPECT_EQ(alloc.AvailableBytes(), total - kPageSize);
  EXPECT_EQ(alloc.UsedBytes(), kPageSize);
}

TEST(PageBitmapAllocatorTest, AllocateContinuousMultiplePagesInOneBuffer) {
  auto alloc = MakeAllocator(2, 8);

  auto r = alloc.Allocate(3);
  ASSERT_TRUE(r.has_value());
  // Strategy 1 hit on buffer 0, starting at page 0.
  ASSERT_EQ(r->size(), 3u);
  EXPECT_EQ((*r)[0], (PageLocation{0, 0}));
  EXPECT_EQ((*r)[1], (PageLocation{0, 1}));
  EXPECT_EQ((*r)[2], (PageLocation{0, 2}));
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
  // CollectFirstNFree scans low to high, so order is deterministic.
  EXPECT_EQ(*r, std::vector<PageLocation>({{0, 1}, {0, 3}, {0, 5}}));
  EXPECT_EQ(alloc.AvailableBytes(), 0u);
}

TEST(PageBitmapAllocatorTest, AllocateAcrossBuffersWhenSingleBufferInsufficient) {
  // Two buffers of 2 pages each; ask for 3 → requires Strategy 3.
  auto alloc = MakeAllocator(2, 2);
  uint64_t total = alloc.TotalBytes();

  auto r = alloc.Allocate(3);
  ASSERT_TRUE(r.has_value());
  // Strategy 3 greedy fill: buffer 0 (pages 0,1) then buffer 1 (page 0).
  EXPECT_EQ(*r, std::vector<PageLocation>({{0, 0}, {0, 1}, {1, 0}}));
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

  alloc.Deallocate(*r);
  EXPECT_EQ(alloc.AvailableBytes(), total);

  // Re-allocate — should be able to satisfy the same request shape.
  auto r2 = alloc.Allocate(5);
  ASSERT_TRUE(r2.has_value());
  EXPECT_EQ(AsSet(*r2), AsSet(*r));
}

TEST(PageBitmapAllocatorTest, DeallocateIdempotent) {
  auto alloc = MakeAllocator(1, 4);
  auto r = alloc.Allocate(2);
  ASSERT_TRUE(r.has_value());
  uint64_t avail_after_alloc = alloc.AvailableBytes();

  alloc.Deallocate(*r);
  uint64_t avail_after_first_free = alloc.AvailableBytes();
  EXPECT_GT(avail_after_first_free, avail_after_alloc);

  // Second Deallocate of the same set must not corrupt state.
  alloc.Deallocate(*r);
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
  alloc.Deallocate(*r);
  EXPECT_EQ(alloc.AvailableBytes(), total);
}

TEST(PageBitmapAllocatorTest, AllocateExhaustionThenSuccess) {
  // Sanity: after exhausting all pages, Allocate(1) fails; after one Deallocate
  // it succeeds again.
  auto alloc = MakeAllocator(1, 3);
  auto a = alloc.Allocate(3);
  ASSERT_TRUE(a.has_value());
  EXPECT_EQ(alloc.AvailableBytes(), 0u);
  EXPECT_FALSE(alloc.Allocate(1).has_value());

  alloc.Deallocate({(*a)[1]});
  auto b = alloc.Allocate(1);
  ASSERT_TRUE(b.has_value());
  EXPECT_EQ((*b)[0], (*a)[1]);
}

}  // namespace
