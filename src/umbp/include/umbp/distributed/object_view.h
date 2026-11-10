// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "umbp/distributed/transfer/transfer_engine.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

// One physical run of an object. `offset` is relative to `buffer`; the logical
// interval is [logical_offset, logical_offset + length). ObjectView deliberately
// keeps the in-process TransferRef: the wire encoding added by the engine
// control plane will carry the same identity as backend/executor/region fields.
struct RegionSpan {
  TransferRef buffer;
  uint64_t offset = 0;
  uint64_t length = 0;
  uint64_t logical_offset = 0;
};

struct ObjectView {
  uint64_t size = 0;
  std::string layout_id;
  std::vector<RegionSpan> spans;

  bool empty() const { return spans.empty(); }
};

inline bool ValidateObjectView(const ObjectView& view) {
  if (view.size == 0 || view.spans.empty()) return false;
  std::vector<const RegionSpan*> ordered;
  ordered.reserve(view.spans.size());
  for (const auto& span : view.spans) {
    if (!span.buffer.Valid() || span.length == 0 || span.offset > span.buffer.size ||
        span.length > span.buffer.size - span.offset || span.logical_offset >= view.size ||
        span.length > view.size - span.logical_offset) {
      return false;
    }
    ordered.push_back(&span);
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const auto* a, const auto* b) { return a->logical_offset < b->logical_offset; });
  uint64_t next = 0;
  for (const auto* span : ordered) {
    if (span->logical_offset != next) return false;
    next += span->length;
  }
  return next == view.size;
}

// Construct the logical object view for a PageBackend allocation. Pages may be
// scattered within one buffer or across buffers; the final page is truncated to
// the bytes the object owns.
template <class BufferRef>
ObjectView BuildPageObjectView(const std::vector<PageLocation>& pages, uint64_t page_size,
                               uint64_t object_size, std::string layout_id,
                               BufferRef&& buffer_ref) {
  ObjectView view;
  view.size = object_size;
  view.layout_id = std::move(layout_id);
  if (page_size == 0 || object_size == 0 || pages.empty() ||
      object_size > pages.size() * page_size || object_size <= (pages.size() - 1) * page_size) {
    return view;
  }
  uint64_t logical = 0;
  for (const auto& page : pages) {
    TransferRef ref = buffer_ref(page.buffer_index);
    const uint64_t length = std::min(page_size, object_size - logical);
    if (!ref.Valid() || static_cast<uint64_t>(page.page_index) * page_size > ref.size ||
        length > ref.size - static_cast<uint64_t>(page.page_index) * page_size) {
      view.spans.clear();
      return view;
    }
    view.spans.push_back(RegionSpan{
        std::move(ref), static_cast<uint64_t>(page.page_index) * page_size, length, logical});
    logical += length;
  }
  if (!ValidateObjectView(view)) view.spans.clear();
  return view;
}

// Intersect two physical span lists in logical-byte order. No physical
// contiguity assumption is made; transfer engines may coalesce the resulting
// items later. `logical_offset/length == 0` means the whole object.
inline bool BuildObjectViewTransfers(const ObjectView& source, const ObjectView& target,
                                     std::vector<TransferItem>* items, uint64_t logical_offset = 0,
                                     uint64_t length = 0) {
  if (items == nullptr || !ValidateObjectView(source) || !ValidateObjectView(target) ||
      source.size != target.size || logical_offset > source.size) {
    return false;
  }
  if (length == 0) length = source.size - logical_offset;
  if (length == 0 || length > source.size - logical_offset) return false;

  std::vector<const RegionSpan*> src;
  std::vector<const RegionSpan*> dst;
  for (const auto& span : source.spans) src.push_back(&span);
  for (const auto& span : target.spans) dst.push_back(&span);
  const auto by_logical = [](const auto* a, const auto* b) {
    return a->logical_offset < b->logical_offset;
  };
  std::sort(src.begin(), src.end(), by_logical);
  std::sort(dst.begin(), dst.end(), by_logical);

  items->clear();
  const uint64_t end = logical_offset + length;
  size_t si = 0;
  size_t di = 0;
  uint64_t at = logical_offset;
  while (at < end) {
    while (si < src.size() && src[si]->logical_offset + src[si]->length <= at) ++si;
    while (di < dst.size() && dst[di]->logical_offset + dst[di]->length <= at) ++di;
    if (si == src.size() || di == dst.size() || src[si]->logical_offset > at ||
        dst[di]->logical_offset > at) {
      items->clear();
      return false;
    }
    const auto& s = *src[si];
    const auto& d = *dst[di];
    const uint64_t run =
        std::min({end - at, s.logical_offset + s.length - at, d.logical_offset + d.length - at});
    TransferItem item;
    item.src = s.buffer;
    item.dst = d.buffer;
    item.src_offset = s.offset + at - s.logical_offset;
    item.dst_offset = d.offset + at - d.logical_offset;
    item.size = run;
    item.tag = items->size();
    items->push_back(std::move(item));
    at += run;
  }
  return !items->empty();
}

}  // namespace mori::umbp
