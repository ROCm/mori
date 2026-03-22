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
#include <utility>

#include "umbp/common/pool_allocator.h"

namespace {

using mori::umbp::PoolAllocator;

PoolAllocator MakeDramAllocator(uint64_t total) {
  PoolAllocator alloc;
  alloc.total_size = total;
  alloc.offset_tracker = PoolAllocator::OffsetTracker{};
  return alloc;
}

PoolAllocator MakeSsdAllocator(uint64_t total) {
  PoolAllocator alloc;
  alloc.total_size = total;
  return alloc;
}

bool Overlaps(uint64_t a_off, uint64_t a_size, uint64_t b_off, uint64_t b_size) {
  return a_off < b_off + b_size && b_off < a_off + a_size;
}

TEST(PoolAllocatorTest, BasicAllocate) {
  auto alloc = MakeDramAllocator(1024);

  EXPECT_EQ(alloc.AvailableBytes(), 1024u);
  auto offset = alloc.Allocate(256);
  ASSERT_TRUE(offset.has_value());
  EXPECT_EQ(*offset, 0u);
  EXPECT_EQ(alloc.AvailableBytes(), 768u);
}

TEST(PoolAllocatorTest, AllocateBeyondCapacity) {
  auto alloc = MakeDramAllocator(100);

  EXPECT_FALSE(alloc.Allocate(200).has_value());
  EXPECT_EQ(alloc.AvailableBytes(), 100u);
}

TEST(PoolAllocatorTest, AllocateExactCapacity) {
  auto alloc = MakeDramAllocator(100);

  auto offset = alloc.Allocate(100);
  ASSERT_TRUE(offset.has_value());
  EXPECT_EQ(alloc.AvailableBytes(), 0u);

  EXPECT_FALSE(alloc.Allocate(1).has_value());
}

TEST(PoolAllocatorTest, DeallocateReclaim) {
  auto alloc = MakeDramAllocator(1024);

  auto offset = alloc.Allocate(256);
  ASSERT_TRUE(offset.has_value());
  EXPECT_EQ(alloc.AvailableBytes(), 768u);

  alloc.Deallocate(*offset, 256);
  EXPECT_EQ(alloc.AvailableBytes(), 1024u);
}

TEST(PoolAllocatorTest, FreeListReuse) {
  auto alloc = MakeDramAllocator(1024);

  auto offset1 = alloc.Allocate(256);
  ASSERT_TRUE(offset1.has_value());
  alloc.Allocate(128);

  alloc.Deallocate(*offset1, 256);
  auto offset2 = alloc.Allocate(256);
  ASSERT_TRUE(offset2.has_value());
  EXPECT_EQ(*offset1, *offset2);
}

TEST(PoolAllocatorTest, Fragmentation) {
  auto alloc = MakeDramAllocator(300);

  auto a = alloc.Allocate(100);
  auto b = alloc.Allocate(100);
  auto c = alloc.Allocate(100);
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  ASSERT_TRUE(c.has_value());
  EXPECT_EQ(alloc.AvailableBytes(), 0u);

  alloc.Deallocate(*a, 100);
  alloc.Deallocate(*c, 100);
  EXPECT_EQ(alloc.AvailableBytes(), 200u);

  EXPECT_FALSE(alloc.Allocate(150).has_value());
  EXPECT_EQ(alloc.AvailableBytes(), 200u);
}

TEST(PoolAllocatorTest, SsdModeAllocate) {
  auto alloc = MakeSsdAllocator(1024);

  auto offset = alloc.Allocate(256);
  ASSERT_TRUE(offset.has_value());
  EXPECT_EQ(*offset, 0u);
  EXPECT_EQ(alloc.AvailableBytes(), 768u);

  auto offset2 = alloc.Allocate(512);
  ASSERT_TRUE(offset2.has_value());
  EXPECT_EQ(*offset2, 0u);
  EXPECT_EQ(alloc.AvailableBytes(), 256u);
}

TEST(PoolAllocatorTest, SsdModeDeallocate) {
  auto alloc = MakeSsdAllocator(1024);

  alloc.Allocate(256);
  alloc.Allocate(512);
  EXPECT_EQ(alloc.AvailableBytes(), 256u);

  alloc.Deallocate(0, 256);
  EXPECT_EQ(alloc.AvailableBytes(), 512u);

  alloc.Deallocate(0, 512);
  EXPECT_EQ(alloc.AvailableBytes(), 1024u);
}

TEST(PoolAllocatorTest, SsdModeAllocateBeyondCapacity) {
  auto alloc = MakeSsdAllocator(100);

  EXPECT_FALSE(alloc.Allocate(200).has_value());
  EXPECT_EQ(alloc.AvailableBytes(), 100u);
}

TEST(PoolAllocatorTest, NonOverlappingOffsets) {
  auto alloc = MakeDramAllocator(1024);

  struct Region {
    uint64_t offset;
    uint64_t size;
  };
  std::vector<Region> regions;

  uint64_t sizes[] = {100, 200, 50, 300, 74};
  for (auto sz : sizes) {
    auto off = alloc.Allocate(sz);
    ASSERT_TRUE(off.has_value());
    regions.push_back({*off, sz});
  }

  for (size_t i = 0; i < regions.size(); ++i) {
    for (size_t j = i + 1; j < regions.size(); ++j) {
      EXPECT_FALSE(Overlaps(regions[i].offset, regions[i].size, regions[j].offset, regions[j].size))
          << "Region " << i << " [" << regions[i].offset << ", "
          << regions[i].offset + regions[i].size << ") overlaps with Region " << j << " ["
          << regions[j].offset << ", " << regions[j].offset + regions[j].size << ")";
    }
  }
}

TEST(PoolAllocatorTest, CoalesceAdjacentFreeBlocks) {
  auto alloc = MakeDramAllocator(300);

  auto a = alloc.Allocate(100);
  auto b = alloc.Allocate(100);
  auto c = alloc.Allocate(100);
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  ASSERT_TRUE(c.has_value());

  alloc.Deallocate(*a, 100);
  alloc.Deallocate(*b, 100);

  auto d = alloc.Allocate(200);
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(*d, 0u);
}

}  // namespace
