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

// ForEachRangePageFragment is the only place the object-range -> page-range
// mapping lives, for both the local medium and a peer's pages.  An error here
// does not fail: it reads or writes the wrong bytes of a KV block and the
// caller sees a successful transfer.  Until this file existed the arithmetic
// was only reachable through a live two-node fixture, so the awkward shapes
// (short last page, range ending exactly on the object's last byte, one byte
// past the end) went untested.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "umbp/distributed/range_map.h"

namespace mori::umbp {
namespace {

struct Fragment {
  size_t page_index;
  uint64_t tier_offset;
  size_t bytes;
  size_t copied;

  bool operator==(const Fragment& other) const {
    return page_index == other.page_index && tier_offset == other.tier_offset &&
           bytes == other.bytes && copied == other.copied;
  }
};

// Pages laid out as a contiguous run starting at `first_page` of buffer 0,
// which is what the allocator produces for a single-slot object.
std::vector<PageLocation> PageRun(size_t count, uint32_t first_page = 0, uint32_t buffer = 0) {
  std::vector<PageLocation> pages;
  pages.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    PageLocation page;
    page.buffer_index = buffer;
    page.page_index = first_page + static_cast<uint32_t>(i);
    pages.push_back(page);
  }
  return pages;
}

std::vector<Fragment> Collect(const std::vector<PageLocation>& pages, uint64_t page_size,
                              uint64_t stored_size, size_t offset, size_t size, bool* ok) {
  std::vector<Fragment> out;
  *ok = ForEachRangePageFragment(
      pages, page_size, stored_size, offset, size,
      [&](size_t page_index, uint64_t tier_offset, size_t bytes, size_t copied) {
        out.push_back(Fragment{page_index, tier_offset, bytes, copied});
        return true;
      });
  return out;
}

constexpr uint64_t kPage = 4096;

TEST(RangeMapTest, RangeInsideOnePage) {
  bool ok = false;
  const auto frags = Collect(PageRun(4), kPage, 4 * kPage, 100, 500, &ok);
  EXPECT_TRUE(ok);
  ASSERT_EQ(frags.size(), 1u);
  EXPECT_EQ(frags[0], (Fragment{0, 100, 500, 0}));
}

TEST(RangeMapTest, RangeStartsMidPageAndSpansTwo) {
  bool ok = false;
  const auto frags = Collect(PageRun(4), kPage, 4 * kPage, kPage - 300, 800, &ok);
  EXPECT_TRUE(ok);
  ASSERT_EQ(frags.size(), 2u);
  EXPECT_EQ(frags[0], (Fragment{0, kPage - 300, 300, 0}));
  EXPECT_EQ(frags[1], (Fragment{1, kPage, 500, 300}));
}

TEST(RangeMapTest, RangeSpansEveryPage) {
  bool ok = false;
  const auto frags = Collect(PageRun(3), kPage, 3 * kPage, 0, 3 * kPage, &ok);
  EXPECT_TRUE(ok);
  ASSERT_EQ(frags.size(), 3u);
  for (size_t p = 0; p < 3; ++p) {
    EXPECT_EQ(frags[p], (Fragment{p, p * kPage, kPage, p * kPage})) << "page " << p;
  }
}

// tier_offset is an offset into the BUFFER, so a slot that does not start at
// page 0 must be reported relative to the buffer, not to the object.
TEST(RangeMapTest, TierOffsetIsBufferRelative) {
  bool ok = false;
  const auto frags = Collect(PageRun(2, /*first_page=*/7), kPage, 2 * kPage, 64, 128, &ok);
  EXPECT_TRUE(ok);
  ASSERT_EQ(frags.size(), 1u);
  EXPECT_EQ(frags[0], (Fragment{0, 7 * kPage + 64, 128, 0}));
}

// The last page of an object is short whenever the object is not a page
// multiple, and the range must be clamped to the part the object owns.
TEST(RangeMapTest, ShortLastPageIsClamped) {
  constexpr uint64_t kStored = kPage + 1000;
  bool ok = false;
  const auto frags = Collect(PageRun(2), kPage, kStored, kPage - 200, 1200, &ok);
  EXPECT_TRUE(ok);
  ASSERT_EQ(frags.size(), 2u);
  EXPECT_EQ(frags[0], (Fragment{0, kPage - 200, 200, 0}));
  EXPECT_EQ(frags[1], (Fragment{1, kPage, 1000, 200}));
}

TEST(RangeMapTest, RangeEndingOnTheObjectsLastByte) {
  constexpr uint64_t kStored = kPage + 1000;
  bool ok = false;
  const auto frags = Collect(PageRun(2), kPage, kStored, kStored - 400, 400, &ok);
  EXPECT_TRUE(ok);
  ASSERT_EQ(frags.size(), 1u);
  EXPECT_EQ(frags[0], (Fragment{1, kPage + 600, 400, 0}));
}

TEST(RangeMapTest, WholeObjectWithShortLastPage) {
  constexpr uint64_t kStored = 2 * kPage + 1;
  bool ok = false;
  const auto frags = Collect(PageRun(3), kPage, kStored, 0, kStored, &ok);
  EXPECT_TRUE(ok);
  ASSERT_EQ(frags.size(), 3u);
  EXPECT_EQ(frags[2], (Fragment{2, 2 * kPage, 1, 2 * kPage}));
}

TEST(RangeMapTest, OneBytePastTheEndIsRejected) {
  constexpr uint64_t kStored = kPage + 1000;
  bool ok = true;
  const auto frags = Collect(PageRun(2), kPage, kStored, kStored - 400, 401, &ok);
  EXPECT_FALSE(ok);
  EXPECT_TRUE(frags.empty()) << "a rejected range must not emit fragments";
}

TEST(RangeMapTest, OffsetPastTheEndIsRejected) {
  bool ok = true;
  Collect(PageRun(2), kPage, 2 * kPage, 2 * kPage, 1, &ok);
  EXPECT_FALSE(ok);
}

TEST(RangeMapTest, ZeroSizeIsRejected) {
  bool ok = true;
  Collect(PageRun(2), kPage, 2 * kPage, 0, 0, &ok);
  EXPECT_FALSE(ok);
}

// A page count that cannot hold the object, or that is one page more than the
// object needs, means the caller paired a size with the wrong slot.
TEST(RangeMapTest, MismatchedPageCountIsRejected) {
  bool ok = true;
  Collect(PageRun(1), kPage, 2 * kPage, 0, 100, &ok);
  EXPECT_FALSE(ok) << "too few pages for the object";

  ok = true;
  Collect(PageRun(3), kPage, 2 * kPage, 0, 100, &ok);
  EXPECT_FALSE(ok) << "one more page than the object needs";
}

TEST(RangeMapTest, EmptyPagesOrZeroPageSizeIsRejected) {
  bool ok = true;
  Collect({}, kPage, kPage, 0, 100, &ok);
  EXPECT_FALSE(ok);

  ok = true;
  Collect(PageRun(1), 0, kPage, 0, 100, &ok);
  EXPECT_FALSE(ok);
}

// The walk stops at the first refusal instead of running the range to its end.
TEST(RangeMapTest, EmitRefusalAbortsTheWalk) {
  size_t calls = 0;
  const bool ok = ForEachRangePageFragment(PageRun(4), kPage, 4 * kPage, 0, 4 * kPage,
                                           [&](size_t, uint64_t, size_t, size_t) {
                                             ++calls;
                                             return calls < 2;
                                           });
  EXPECT_FALSE(ok);
  EXPECT_EQ(calls, 2u);
}

}  // namespace
}  // namespace mori::umbp
