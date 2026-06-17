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
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "umbp/distributed/types.h"

namespace mori::umbp {

// Page-granularity bitmap allocator, one instance per (node_id, tier) pair
// (owned by Master on the control plane, or by PeerDramAllocator on the peer).
// Internally holds N BufferState entries (one per physical buffer registered
// by the Client at RegisterClient time); each BufferState carries an
// independent `vector<bool>` bitmap.
//
// Allocation strategy:
//   1. same-buffer continuous run  (best — single-RDMA-friendly)
//   2. same-buffer discrete pages  (in-buffer scatter-gather)
//   3. cross-buffer discrete pages (cross-buffer scatter-gather, fallback)
//
// All-or-nothing semantic: if no strategy can satisfy `num_pages`, the call
// returns std::nullopt and **no bitmap bit is touched**.
//
// THREAD SAFETY: PageBitmapAllocator holds NO internal mutex.  Callers MUST
// serialize every method via the owning object's mutex (e.g.
// `ClientRegistry::mutex_` on master, `PeerDramAllocator::mutex_` on peer).
class PageBitmapAllocator {
 public:
  struct BufferState {
    uint32_t buffer_index = 0;
    uint64_t page_size = 0;  // bytes; constant after construction
    uint32_t total_pages = 0;
    std::vector<bool> bitmap;  // true = page is currently allocated
    uint32_t free_count = 0;
  };

  // Construct an allocator from a list of per-buffer byte sizes.
  //
  //   - `page_size` must be > 0 (throws std::invalid_argument otherwise).
  //   - `buffer_sizes` may be empty: that produces an allocator with zero
  //     capacity (valid; every Allocate(>0) returns nullopt).
  //   - Each buffer's `total_pages` is `buffer_sizes[i] / page_size` (integer
  //     division).  Any remainder bytes within a buffer are intentionally
  //     wasted — page-granularity allocation cannot address sub-page slack.
  //     Callers should size buffers as multiples of page_size.
  PageBitmapAllocator(uint64_t page_size, const std::vector<uint64_t>& buffer_sizes)
      : page_size_(page_size) {
    if (page_size == 0) {
      throw std::invalid_argument("PageBitmapAllocator: page_size must be > 0");
    }
    buffers_.reserve(buffer_sizes.size());
    for (size_t i = 0; i < buffer_sizes.size(); ++i) {
      BufferState bs;
      bs.buffer_index = static_cast<uint32_t>(i);
      bs.page_size = page_size;
      bs.total_pages = static_cast<uint32_t>(buffer_sizes[i] / page_size);
      bs.bitmap.assign(bs.total_pages, false);
      bs.free_count = bs.total_pages;
      buffers_.push_back(std::move(bs));
    }
  }

  // All-or-nothing allocate.  See class doc for strategy ordering; nullopt
  // leaves every bitmap bit untouched.
  std::optional<std::vector<PageLocation>> Allocate(uint32_t num_pages) {
    if (num_pages == 0) return std::nullopt;

    // Strategy 1: same-buffer continuous run.
    for (auto& buf : buffers_) {
      if (buf.free_count < num_pages) continue;
      auto run_start = FindContinuousFreeRun(buf, num_pages);
      if (run_start) {
        std::vector<PageLocation> pages;
        pages.reserve(num_pages);
        for (uint32_t k = 0; k < num_pages; ++k) {
          uint32_t idx = *run_start + k;
          buf.bitmap[idx] = true;
          pages.push_back({buf.buffer_index, idx});
        }
        buf.free_count -= num_pages;
        return pages;
      }
    }

    // Strategy 2: same-buffer discrete pages.  free_count >= num_pages is
    // sufficient to guarantee CollectFirstNFree() can yield exactly num_pages.
    for (auto& buf : buffers_) {
      if (buf.free_count < num_pages) continue;
      auto idxs = CollectFirstNFree(buf, num_pages);
      std::vector<PageLocation> pages;
      pages.reserve(num_pages);
      for (auto idx : idxs) {
        buf.bitmap[idx] = true;
        pages.push_back({buf.buffer_index, idx});
      }
      buf.free_count -= num_pages;
      return pages;
    }

    // Strategy 3: cross-buffer discrete pages (greedy, in buffer-index order).
    // First a guarded total-capacity check so that we never partially mutate
    // bitmaps when the request can't be satisfied globally.
    uint64_t total_free = 0;
    for (const auto& b : buffers_) total_free += b.free_count;
    if (total_free < num_pages) return std::nullopt;

    std::vector<PageLocation> pages;
    pages.reserve(num_pages);
    uint32_t remaining = num_pages;
    for (auto& buf : buffers_) {
      if (remaining == 0) break;
      if (buf.free_count == 0) continue;
      uint32_t take = std::min<uint32_t>(remaining, buf.free_count);
      auto idxs = CollectFirstNFree(buf, take);
      for (auto idx : idxs) {
        buf.bitmap[idx] = true;
        pages.push_back({buf.buffer_index, idx});
      }
      buf.free_count -= take;
      remaining -= take;
    }
    return pages;
  }

  // Idempotent free: for each entry, only flip true -> false.  Out-of-range
  // buffer_index / page_index and already-free pages are silently skipped
  // (do NOT throw, do NOT underflow free_count).
  void Deallocate(const std::vector<PageLocation>& pages) {
    for (const auto& p : pages) {
      if (p.buffer_index >= buffers_.size()) continue;
      auto& buf = buffers_[p.buffer_index];
      if (p.page_index >= buf.total_pages) continue;
      if (!buf.bitmap[p.page_index]) continue;  // already free — idempotent no-op
      buf.bitmap[p.page_index] = false;
      ++buf.free_count;
    }
  }

  uint64_t TotalBytes() const {
    uint64_t sum = 0;
    for (const auto& b : buffers_) {
      sum += static_cast<uint64_t>(b.total_pages) * page_size_;
    }
    return sum;
  }

  uint64_t AvailableBytes() const {
    uint64_t free_pages = 0;
    for (const auto& b : buffers_) free_pages += b.free_count;
    return free_pages * page_size_;
  }

  uint64_t UsedBytes() const { return TotalBytes() - AvailableBytes(); }

  uint64_t PageSize() const { return page_size_; }
  size_t NumBuffers() const { return buffers_.size(); }

  // Read-only access to per-buffer state (e.g. for diagnostics / Heartbeat
  // status reporting).  Returned reference is invalidated by any subsequent
  // mutating call; callers holding the owning object's mutex are safe.
  const std::vector<BufferState>& Buffers() const { return buffers_; }

 private:
  // Find a contiguous run of `n` free pages in `buf`.  Returns the starting
  // page_index on success, nullopt if no such run exists.  O(total_pages).
  static std::optional<uint32_t> FindContinuousFreeRun(const BufferState& buf, uint32_t n) {
    if (n == 0 || n > buf.total_pages) return std::nullopt;
    uint32_t run = 0;
    for (uint32_t i = 0; i < buf.total_pages; ++i) {
      if (!buf.bitmap[i]) {
        ++run;
        if (run == n) return i + 1 - n;
      } else {
        run = 0;
      }
    }
    return std::nullopt;
  }

  // Collect the first `n` free page indices in `buf` (scanning low to high).
  // Caller MUST guarantee buf.free_count >= n.
  static std::vector<uint32_t> CollectFirstNFree(const BufferState& buf, uint32_t n) {
    std::vector<uint32_t> result;
    result.reserve(n);
    for (uint32_t i = 0; i < buf.total_pages && result.size() < n; ++i) {
      if (!buf.bitmap[i]) result.push_back(i);
    }
    return result;
  }

  uint64_t page_size_;
  std::vector<BufferState> buffers_;
};

}  // namespace mori::umbp
