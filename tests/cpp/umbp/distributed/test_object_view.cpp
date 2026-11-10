// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <vector>

#include "umbp/distributed/object_view.h"
#include "umbp/engine/engine_types.h"

namespace mori::umbp {
namespace {

TEST(ObjectViewTest, IntersectsScatteredSpansWithoutTouchingGuardBytes) {
  std::vector<uint8_t> src_a(32, 0xee), src_b(32, 0xee);
  std::vector<uint8_t> dst_a(32, 0xcc), dst_b(32, 0xcc);
  for (uint8_t i = 0; i < 12; ++i) src_a[4 + i] = i;
  for (uint8_t i = 0; i < 8; ++i) src_b[7 + i] = 12 + i;

  ObjectView source{20,
                    "layout",
                    {{TransferRef::HostBytes(src_a.data(), src_a.size()), 4, 12, 0},
                     {TransferRef::HostBytes(src_b.data(), src_b.size()), 7, 8, 12}}};
  ObjectView target{20,
                    "layout",
                    {{TransferRef::HostBytes(dst_a.data(), dst_a.size()), 9, 5, 0},
                     {TransferRef::HostBytes(dst_b.data(), dst_b.size()), 3, 11, 5},
                     {TransferRef::HostBytes(dst_a.data(), dst_a.size()), 20, 4, 16}}};
  ASSERT_TRUE(ValidateObjectView(source));
  ASSERT_TRUE(ValidateObjectView(target));

  std::vector<TransferItem> items;
  ASSERT_TRUE(BuildObjectViewTransfers(source, target, &items));
  ASSERT_EQ(items.size(), 4u);
  for (const auto& item : items) {
    std::memcpy(static_cast<uint8_t*>(item.dst.host_ptr) + item.dst_offset,
                static_cast<const uint8_t*>(item.src.host_ptr) + item.src_offset, item.size);
  }
  std::vector<uint8_t> logical;
  logical.insert(logical.end(), dst_a.begin() + 9, dst_a.begin() + 14);
  logical.insert(logical.end(), dst_b.begin() + 3, dst_b.begin() + 14);
  logical.insert(logical.end(), dst_a.begin() + 20, dst_a.begin() + 24);
  ASSERT_EQ(logical.size(), 20u);
  for (uint8_t i = 0; i < logical.size(); ++i) EXPECT_EQ(logical[i], i);
  EXPECT_EQ(dst_a[8], 0xcc);
  EXPECT_EQ(dst_a[14], 0xcc);
  EXPECT_EQ(dst_b[2], 0xcc);
  EXPECT_EQ(dst_b[14], 0xcc);
  EXPECT_EQ(dst_a[19], 0xcc);
  EXPECT_EQ(dst_a[24], 0xcc);
}

TEST(ObjectViewTest, PageViewHandlesCrossBufferAndShortTail) {
  std::vector<uint8_t> first(4 * 4096), second(4 * 4096);
  std::vector<PageLocation> pages = {{0, 2}, {1, 1}};
  const auto view =
      BuildPageObjectView(pages, 4096, 4096 + 123, "page-layout", [&](uint32_t buffer) {
        return buffer == 0 ? TransferRef::HostBytes(first.data(), first.size())
                           : TransferRef::HostBytes(second.data(), second.size());
      });
  ASSERT_TRUE(ValidateObjectView(view));
  ASSERT_EQ(view.spans.size(), 2u);
  EXPECT_EQ(view.spans[0].offset, 2 * 4096u);
  EXPECT_EQ(view.spans[1].offset, 4096u);
  EXPECT_EQ(view.spans[1].length, 123u);
  EXPECT_EQ(view.spans[1].logical_offset, 4096u);
}

TEST(ObjectViewTest, RejectsLogicalHolesOverlapAndOutOfBoundsPhysicalRuns) {
  std::vector<uint8_t> bytes(32);
  auto ref = TransferRef::HostBytes(bytes.data(), bytes.size());
  EXPECT_FALSE(ValidateObjectView(ObjectView{8, "x", {{ref, 0, 4, 0}, {ref, 4, 3, 5}}}));
  EXPECT_FALSE(ValidateObjectView(ObjectView{8, "x", {{ref, 0, 5, 0}, {ref, 5, 4, 4}}}));
  EXPECT_FALSE(ValidateObjectView(ObjectView{8, "x", {{ref, 30, 8, 0}}}));
}

TEST(ObjectViewTest, PageGeometryValidatesLastPageCapacityAndMultiplicationOverflow) {
  std::vector<uint8_t> short_buffer(8);
  engine::PageGeometry geometry;
  geometry.page_tokens = 1;
  geometry.num_pages = 4;
  geometry.object_bytes = 4;
  geometry.regions = {{"short", TransferRef::HostBytes(short_buffer.data(), short_buffer.size()), 4,
                       1, 0, 0, "kv"}};
  EXPECT_TRUE(ValidateObjectView(geometry.View(1)));
  EXPECT_FALSE(geometry.Validate());

  std::vector<uint8_t> storage(64);
  geometry.regions = {{"overflow", TransferRef::HostBytes(storage.data(), storage.size()),
                       std::numeric_limits<uint64_t>::max(), 2, 0, 0, "kv"}};
  EXPECT_EQ(geometry.regions[0].PageBytes(), 0u);
  EXPECT_FALSE(geometry.Validate());

  geometry.regions = {{"offset-overflow", TransferRef::HostBytes(storage.data(), storage.size()),
                       std::numeric_limits<uint64_t>::max() / 2 + 1, 1, 0, 0, "kv"}};
  EXPECT_FALSE(geometry.Validate());
}

}  // namespace
}  // namespace mori::umbp
