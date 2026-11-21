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

#include "umbp/distributed/peer/backend/page_pool.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

// Page-granularity bitmap allocator, one instance per (node_id, tier) pair
// (owned by Master on the control plane, or by PageBackend on the peer).
// Internally holds N BufferState entries (one per physical buffer registered
// by the Client at RegisterClient time); each BufferState carries an
// independent bitmap of 64-bit words.
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
// `ClientRegistry::mutex_` on master, `PageBackend::mutex_` on peer).
//
// This is the default PagePool implementation.  It is a bitmap with a next-fit
// cursor: cheap to reason about and allocation-free, but its search is still
// proportional to the run it examines rather than O(1).  See page_pool.h for
// why that seam exists and what a faster implementation would look like.
class PageBitmapAllocator : public PagePool {
 public:
  struct BufferState {
    uint32_t buffer_index = 0;
    uint64_t page_size = 0;  // bytes; constant after construction
    uint32_t total_pages = 0;
    // Bit (i % 64) of words[i / 64] is set while page i is allocated.  Bits past
    // total_pages in the last word stay set, so no search reports them free.
    //
    // Words rather than vector<bool>: a search skips a word of allocated (or
    // free) pages with one bit scan.  The cursor below does not help a search
    // that fails, and in a full pool that eviction has fragmented, the run
    // search for a large object fails in buffer after buffer.  On 1,799 GB of
    // 64 KiB pages in four buffers (27 million pages), Allocate of a 19 MB
    // object took 3.6-7.7 ms bit by bit and takes 40-75 us by words; a put
    // batch allocates all of its objects under one hold of PageBackend::mutex_.
    std::vector<uint64_t> words;
    uint32_t free_count = 0;
    // Next-fit cursor: where the last successful search STOPPED.
    //
    // Both searches used to restart at index 0 on every Allocate, which is
    // O(total_pages) per call regardless of how much was already allocated.
    // That is invisible on a coarse pool and crippling on a fine one: a 64 GiB
    // pool of 1728 B pages -- what the sglang direct linker mandates, since its
    // page must not exceed the smallest per-layer object -- has 39.8 MILLION
    // pages, so every allocation rescanned tens of millions of bits and got
    // slower as the pool filled.  Measured on an 8-rank embedded deployment:
    // BatchAllocateSlots was 93.8% of remote put time at ~280 us per page.
    //
    // The cursor makes the common case proportional to the run actually
    // examined rather than to the pool.  Correctness is unaffected: both
    // searches wrap and still cover every page before reporting failure, so
    // they find a run exactly when the old scan would have.  Only the CHOSEN
    // run differs, and no caller depends on which free pages it gets.
    uint32_t cursor = 0;
    // Smallest run proven absent since this buffer last freed a page.  A failed
    // search leaves the cursor in place, so on a nearly full buffer it would
    // rescan the whole bitmap for every key of a batch, under the owner's lock.
    // Allocating cannot create a run, so this stays true until a free.
    uint32_t no_run_at_least = UINT32_MAX;

    bool allocated(uint32_t page) const { return (words[page / 64] >> (page % 64)) & 1; }
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
      bs.words.assign((static_cast<size_t>(bs.total_pages) + 63) / 64, 0);
      if (bs.total_pages % 64 != 0) bs.words.back() = ~uint64_t{0} << (bs.total_pages % 64);
      bs.free_count = bs.total_pages;
      buffers_.push_back(std::move(bs));
    }
  }

  // All-or-nothing allocate.  See class doc for strategy ordering; nullopt
  // leaves every bitmap bit untouched.
  std::optional<std::vector<PageLocation>> Allocate(
      uint32_t num_pages, std::optional<size_t> preferred_buffer = std::nullopt) override {
    if (num_pages == 0) return std::nullopt;

    if (preferred_buffer && *preferred_buffer < buffers_.size()) {
      auto& local = buffers_[*preferred_buffer];
      if (local.free_count >= num_pages) {
        if (auto pages = AllocateInBuffer(local, num_pages, true)) return pages;
        return AllocateInBuffer(local, num_pages, false);
      }
      return AllocateAcrossBuffers(num_pages, *preferred_buffer, true);
    }

    // Preserve the original order when there is no valid preference.
    for (auto& buf : buffers_) {
      if (buf.free_count < num_pages) continue;
      if (auto pages = AllocateInBuffer(buf, num_pages, true)) return pages;
    }
    for (auto& buf : buffers_) {
      if (buf.free_count >= num_pages) return AllocateInBuffer(buf, num_pages, false);
    }

    return AllocateAcrossBuffers(num_pages, 0, false);
  }
  // Idempotent free: for each entry, only flip true -> false.  Out-of-range
  // buffer_index / page_index and already-free pages are silently skipped
  // (do NOT throw, do NOT underflow free_count).
  void Deallocate(const std::vector<PageLocation>& pages) override {
    for (const auto& p : pages) {
      if (p.buffer_index >= buffers_.size()) continue;
      auto& buf = buffers_[p.buffer_index];
      if (p.page_index >= buf.total_pages) continue;
      if (!buf.allocated(p.page_index)) continue;  // already free — idempotent no-op
      buf.words[p.page_index / 64] &= ~(uint64_t{1} << (p.page_index % 64));
      ++buf.free_count;
      buf.no_run_at_least = UINT32_MAX;
    }
  }

  uint64_t TotalBytes() const override {
    uint64_t sum = 0;
    for (const auto& b : buffers_) {
      sum += static_cast<uint64_t>(b.total_pages) * page_size_;
    }
    return sum;
  }

  uint64_t AvailableBytes() const override {
    uint64_t free_pages = 0;
    for (const auto& b : buffers_) free_pages += b.free_count;
    return free_pages * page_size_;
  }

  uint64_t PageSize() const override { return page_size_; }
  size_t NumBuffers() const override { return buffers_.size(); }
  uint32_t BufferPageCount(size_t buffer_index) const override {
    return buffer_index < buffers_.size() ? buffers_[buffer_index].total_pages : 0;
  }
  uint32_t BufferFreePages(size_t buffer_index) const override {
    return buffer_index < buffers_.size() ? buffers_[buffer_index].free_count : 0;
  }

  // Read-only access to per-buffer state (e.g. for diagnostics / Heartbeat
  // status reporting).  Returned reference is invalidated by any subsequent
  // mutating call; callers holding the owning object's mutex are safe.
  const std::vector<BufferState>& Buffers() const { return buffers_; }

 private:
  static std::optional<std::vector<PageLocation>> AllocateInBuffer(BufferState& buf, uint32_t count,
                                                                   bool contiguous) {
    std::optional<uint32_t> start;
    if (contiguous) {
      if (count >= buf.no_run_at_least) return std::nullopt;
      start = FindContinuousFreeRun(buf, count);
      if (!start) {
        buf.no_run_at_least = count;
        return std::nullopt;
      }
    }
    const auto indices = contiguous ? std::vector<uint32_t>{} : CollectFirstNFree(buf, count);
    std::vector<PageLocation> pages;
    pages.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t index = start ? *start + i : indices[i];
      MarkAllocated(buf, index);
      pages.push_back({buf.buffer_index, index});
    }
    buf.free_count -= count;
    buf.cursor = (pages.back().page_index + 1) % buf.total_pages;
    return pages;
  }

  std::optional<std::vector<PageLocation>> AllocateAcrossBuffers(uint32_t num_pages, size_t first,
                                                                 bool advance_cursors) {
    // Check before mutating any bitmap. A local fragment is useful even when
    // another buffer could independently satisfy the whole request.
    // First a guarded total-capacity check so that we never partially mutate
    // bitmaps when the request can't be satisfied globally.
    uint64_t total_free = 0;
    for (const auto& b : buffers_) total_free += b.free_count;
    if (total_free < num_pages) return std::nullopt;

    std::vector<PageLocation> pages;
    pages.reserve(num_pages);
    uint32_t remaining = num_pages;
    for (size_t i = 0; i < buffers_.size(); ++i) {
      auto& buf = buffers_[(first + i) % buffers_.size()];
      if (remaining == 0) break;
      if (buf.free_count == 0) continue;
      uint32_t take = std::min<uint32_t>(remaining, buf.free_count);
      auto idxs = CollectFirstNFree(buf, take);
      for (auto idx : idxs) {
        MarkAllocated(buf, idx);
        pages.push_back({buf.buffer_index, idx});
      }
      buf.free_count -= take;
      // NUMA spill becomes the normal path when a local buffer fills. Keep
      // next-fit progress there instead of rescanning all earlier spills on
      // every request. The no-preference path retains its original ordering.
      if (advance_cursors) buf.cursor = (idxs.back() + 1) % buf.total_pages;
      remaining -= take;
    }
    return pages;
  }

  static void MarkAllocated(BufferState& buf, uint32_t page) {
    buf.words[page / 64] |= uint64_t{1} << (page % 64);
  }

  // First page in [from, end) that is allocated (or free, for allocated ==
  // false), or `end` if there is none.  One bit scan per word.
  static uint32_t NextPage(const BufferState& buf, uint32_t from, uint32_t end, bool allocated) {
    if (from >= end) return end;
    size_t word = from / 64;
    const size_t last = (static_cast<size_t>(end) - 1) / 64;
    uint64_t bits =
        (allocated ? buf.words[word] : ~buf.words[word]) & (~uint64_t{0} << (from % 64));
    while (bits == 0) {
      if (++word > last) return end;
      bits = allocated ? buf.words[word] : ~buf.words[word];
    }
    return static_cast<uint32_t>(std::min<uint64_t>(word * 64 + __builtin_ctzll(bits), end));
  }

  // The first run of `n` free pages that starts at or after `from` and ends
  // by `end`.
  static std::optional<uint32_t> FindRun(const BufferState& buf, uint32_t n, uint32_t from,
                                         uint32_t end) {
    if (n >= kLongRun) return FindLongRun(buf, n, from, end);
    while (from < end) {
      const uint32_t first_free = NextPage(buf, from, end, false);
      if (end - first_free < n) return std::nullopt;
      const uint32_t next_allocated = NextPage(buf, first_free, end, true);
      if (next_allocated - first_free >= n) return first_free;
      from = next_allocated;
    }
    return std::nullopt;
  }

  // Any run of kLongRun or more free pages covers a whole free word, so for
  // such requests only words equal to zero are worth a look.  This keeps a
  // failing search cheap in a pool whose free pages are scattered one or two
  // per word, where stepping from run to run would still visit every one.
  static constexpr uint32_t kLongRun = 127;

  static std::optional<uint32_t> FindLongRun(const BufferState& buf, uint32_t n, uint32_t from,
                                             uint32_t end) {
    const size_t first = from / 64;
    const size_t last = (static_cast<size_t>(end) - 1) / 64;
    for (size_t word = first; word <= last; ++word) {
      if (buf.words[word] != 0) continue;
      // This is the run's first whole free word, so the word before it (if it
      // is in range) holds an allocated page; its free top bits join the run.
      uint64_t run_start = word * 64;
      if (word > first) run_start -= __builtin_clzll(buf.words[word - 1]);
      run_start = std::max<uint64_t>(run_start, from);
      const uint32_t run_end = NextPage(buf, static_cast<uint32_t>(word * 64), end, true);
      if (run_end - run_start >= n) return static_cast<uint32_t>(run_start);
      word = run_end / 64;
    }
    return std::nullopt;
  }

  // Find a contiguous run of `n` free pages in `buf`.  Returns the starting
  // page_index on success, nullopt if no such run exists.  O(total_pages / 64).
  // Next-fit: start where the last search stopped and wrap once.  A run may
  // straddle the wrap point only if it starts before it, so the second pass
  // scans [0, cursor + n) -- the overlap is what keeps this exhaustive.
  static std::optional<uint32_t> FindContinuousFreeRun(const BufferState& buf, uint32_t n) {
    if (n == 0 || n > buf.total_pages) return std::nullopt;
    const uint32_t start = buf.cursor < buf.total_pages ? buf.cursor : 0;
    if (auto run = FindRun(buf, n, start, buf.total_pages)) return run;
    const uint32_t wrap_end = std::min<uint64_t>(static_cast<uint64_t>(start) + n,
                                                 static_cast<uint64_t>(buf.total_pages));
    return FindRun(buf, n, 0, wrap_end);
  }

  // Append the free pages of [from, end) to `out`, low to high, until it holds `n`.
  static void CollectFree(const BufferState& buf, uint32_t from, uint32_t end, uint32_t n,
                          std::vector<uint32_t>* out) {
    if (from >= end) return;
    const size_t last = (static_cast<size_t>(end) - 1) / 64;
    for (size_t word = from / 64; word <= last && out->size() < n; ++word) {
      uint64_t free_bits = ~buf.words[word];
      if (word == from / 64) free_bits &= ~uint64_t{0} << (from % 64);
      for (; free_bits != 0 && out->size() < n; free_bits &= free_bits - 1) {
        const uint64_t page = word * 64 + __builtin_ctzll(free_bits);
        if (page >= end) return;
        out->push_back(static_cast<uint32_t>(page));
      }
    }
  }

  // Collect the first `n` free page indices in `buf` (scanning low to high).
  // Caller MUST guarantee buf.free_count >= n.
  static std::vector<uint32_t> CollectFirstNFree(const BufferState& buf, uint32_t n) {
    std::vector<uint32_t> result;
    result.reserve(n);
    // Same next-fit start as the run search, and the same exhaustiveness
    // argument: the wrap pass covers [0, start), so together they visit every
    // page.  free_count >= n is the caller's guarantee that this succeeds.
    const uint32_t start = buf.cursor < buf.total_pages ? buf.cursor : 0;
    CollectFree(buf, start, buf.total_pages, n, &result);
    CollectFree(buf, 0, start, n, &result);
    return result;
  }

  uint64_t page_size_;
  std::vector<BufferState> buffers_;
};

}  // namespace mori::umbp
