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
#include <vector>

#include "umbp/common/range_utils.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

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
