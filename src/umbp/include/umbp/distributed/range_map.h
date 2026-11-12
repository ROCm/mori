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
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "umbp/common/range_utils.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

// Canonicalize a sparse request before using it as coverage or a cache key.
// Reject invalid input without publishing a partial union.
inline bool NormalizeObjectRanges(const std::vector<ObjectExtent>& input, uint64_t object_size,
                                  std::vector<ObjectExtent>* output) {
  auto sorted = input;
  for (const auto& range : sorted) {
    if (range.length == 0 || range.offset > object_size ||
        range.length > object_size - range.offset)
      return false;
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.offset < b.offset; });
  std::vector<ObjectExtent> merged;
  for (const auto& range : sorted) {
    if (!merged.empty() && range.offset - merged.back().offset <= merged.back().length) {
      merged.back().length =
          std::max(merged.back().length, range.offset - merged.back().offset + range.length);
    } else {
      merged.push_back(range);
    }
  }
  *output = std::move(merged);
  return true;
}

// Keep each range's original 16-byte alignment so compacting sparse input does
// not turn an otherwise vectorized FABRIC copy into a byte-copy fallback.
inline uint64_t PackedRangePadding(uint64_t cursor, uint64_t object_offset) {
  return (object_offset % 16 + 16 - cursor % 16) % 16;
}

inline uint64_t PackedRangeBytes(const std::vector<ObjectExtent>& ranges) {
  uint64_t bytes = 0;
  for (const auto& range : ranges) {
    const uint64_t padding = PackedRangePadding(bytes, range.offset);
    if (padding > std::numeric_limits<uint64_t>::max() - bytes) return 0;
    bytes += padding;
    if (range.length > std::numeric_limits<uint64_t>::max() - bytes) return 0;
    bytes += range.length;
  }
  return bytes;
}

// Translate only a fully covered range. Reading across a hole must fail even
// when its endpoints would fit inside the packed allocation.
inline bool RebaseRangeOntoPackedRanges(const std::vector<ObjectExtent>& ranges,
                                        size_t* object_offset, size_t size) {
  uint64_t packed = 0;
  for (const auto& range : ranges) {
    const uint64_t padding = PackedRangePadding(packed, range.offset);
    if (padding > std::numeric_limits<uint64_t>::max() - packed) return false;
    packed += padding;
    if (*object_offset >= range.offset && *object_offset - range.offset <= range.length &&
        size <= range.length - (*object_offset - range.offset)) {
      const uint64_t into = *object_offset - range.offset;
      if (packed > std::numeric_limits<size_t>::max() - into) return false;
      *object_offset = static_cast<size_t>(packed + into);
      return true;
    }
    if (range.length > std::numeric_limits<uint64_t>::max() - packed) return false;
    packed += range.length;
  }
  return false;
}

// The object-range -> page-range mapping, in one place.
//
// An object is stored as a page list; a caller names byte ranges of the object.
// Turning one into the other is the same arithmetic whether the pages live in
// this process's medium (PoolClient::BuildLocalRangeTransfers) or on a peer
// (PoolClient::BuildRemoteGetTransfers) — only the TransferRef the page index
// resolves to differs. It lives here so both callers share one implementation
// and so it can be unit-tested directly: a mistake here does not fail, it
// silently reads or writes the wrong bytes of a KV block.

// Offset of a page within its buffer, as a plain byte offset.
inline uint64_t PageOffset(const PageLocation& page, uint64_t page_size) {
  return static_cast<uint64_t>(page.page_index) * page_size;
}

// Bytes of page `i` that the object actually owns. Every page is full except
// the last, which is short whenever the object is not a page multiple.
inline uint64_t LogicalPageBytes(size_t i, size_t num_pages, uint64_t page_size,
                                 size_t total_size) {
  return (i + 1 == num_pages) ? (total_size - i * page_size) : page_size;
}

// Does `size` plausibly occupy exactly `num_pages` pages of `page_size`?
// One page fewer would not hold it; one more would not have been allocated.
inline bool SizeMatchesAllocation(uint64_t size, size_t num_pages, uint64_t page_size) {
  if (page_size == 0 || num_pages == 0 || size == 0) return false;
  if (size > num_pages * page_size) return false;
  if (size <= (num_pages - 1) * page_size) return false;
  return true;
}

// Moves an object range onto a staged window (ResolvedEntry::covered_offset /
// covered_size), so the walk below indexes the window's pages.  False, leaving
// `*object_offset` untouched, if any byte of the range lies outside the window.
inline bool RebaseRangeOntoWindow(uint64_t window_offset, uint64_t window_size,
                                  size_t* object_offset, size_t range_size) {
  const uint64_t offset = *object_offset;
  if (offset < window_offset) return false;
  const uint64_t into = offset - window_offset;
  if (into > window_size || range_size > window_size - into) return false;
  *object_offset = static_cast<size_t>(into);
  return true;
}

// Walk [object_offset, object_offset + range_size) across `pages`, calling
// `emit` once per page the range touches.
//
//   emit(page_index, tier_offset, fragment, copied_so_far) -> bool
//
// `tier_offset` is the byte offset of the fragment within pages[page_index]'s
// BUFFER (so the caller only has to supply the TransferRef for
// pages[page_index].buffer_index); `fragment` is how many bytes of this page
// the range covers; `copied_so_far` is how far into the range that fragment
// starts, which is what a contiguous destination wants as its own offset.
// Returning false from `emit` aborts the walk and makes this return false.
//
// Returns false — without emitting a partial result the caller could mistake
// for success — when the range runs off the end of the object, off the end of
// the page list, or into the part of the short last page the object does not
// own. Adjacent fragments are NOT merged here; the transfer engines coalesce
// while planning, and merging early would hide the page structure from them.
template <class Emit>
bool ForEachRangePageFragment(const std::vector<PageLocation>& pages, uint64_t page_size,
                              uint64_t stored_size, size_t object_offset, size_t range_size,
                              Emit&& emit) {
  if (range_size == 0) return false;
  if (!SizeMatchesAllocation(stored_size, pages.size(), page_size)) return false;
  if (IsObjectRangeOverflow(object_offset, range_size, static_cast<size_t>(stored_size))) {
    return false;
  }

  size_t offset = object_offset;
  size_t copied = 0;
  while (copied < range_size) {
    const size_t page_index = static_cast<size_t>(offset / page_size);
    const size_t within_page = static_cast<size_t>(offset % page_size);
    if (page_index >= pages.size()) return false;

    const uint64_t page_bytes =
        LogicalPageBytes(page_index, pages.size(), page_size, static_cast<size_t>(stored_size));
    if (within_page >= page_bytes) return false;

    const size_t fragment =
        std::min<size_t>(range_size - copied, static_cast<size_t>(page_bytes - within_page));
    if (!emit(page_index, PageOffset(pages[page_index], page_size) + within_page, fragment,
              copied)) {
      return false;
    }

    offset += fragment;
    copied += fragment;
  }
  return true;
}

}  // namespace mori::umbp
