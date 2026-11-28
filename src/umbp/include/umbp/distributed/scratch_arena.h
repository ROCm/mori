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

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <utility>

namespace mori::umbp {

// The registered host buffer that remote ranged calls stage through, shared by
// every such call in flight.  A call leases one contiguous slice for all of its
// rounds: all it asks for when it is alone, otherwise an equal share among the
// calls holding or awaiting a slice, but never less than one caller's share of
// the arena nor than its largest unit, and it waits until that much is free
// rather than staging through a sliver in many more rounds.  So a lone call gets
// the whole arena, as many calls as it is for overlap, and anything up to the
// arena's size stays servable.  Calls are admitted in arrival order, so a large
// one is never passed over.
//
// Not the storage page pool (PagePool): that hands out page lists that need not
// be contiguous, never waits, and counts toward the tier's capacity; a slice has
// to be one contiguous extent of this buffer, and a call waits for one.
class ScratchArena {
 public:
  // Slices start and end on this, so two never share a cache line.
  static constexpr size_t kAlignment = 64;

  // Non-owning: the buffer belongs to the caller (DistributedClient).  `callers`
  // is how many calls it is for at once: one per GPU that stages through it.
  ScratchArena(void* base, size_t bytes, int numa_node, size_t callers);
  char* Base() const { return base_; }
  size_t Bytes() const { return bytes_; }
  // The node the buffer was placed on; -1 when it was not placed by node.
  int NumaNode() const { return numa_node_; }

  // A slice, given back on destruction, so every early exit releases it.
  class Lease {
   public:
    Lease() = default;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept { *this = std::move(other); }
    Lease& operator=(Lease&& other) noexcept;
    ~Lease();
    char* base() const { return owner_ != nullptr ? owner_->base_ + offset_ : nullptr; }
    size_t size() const { return size_; }

   private:
    friend class ScratchArena;
    Lease(ScratchArena* owner, size_t offset, size_t size)
        : owner_(owner), offset_(offset), size_(size) {}
    ScratchArena* owner_ = nullptr;
    size_t offset_ = 0;
    size_t size_ = 0;
  };

  // Blocks until every earlier caller is admitted and its slice is free, then
  // leases one of at least `min_bytes` and at most `want_bytes`.  Empty when the
  // arena has no buffer or could never hold `min_bytes`.
  Lease Acquire(size_t min_bytes, size_t want_bytes);

  // Calls waiting to be admitted.
  size_t Queued() const;

 private:
  void Release(size_t offset);
  // The largest free extent between live slices, as (offset, size).
  std::pair<size_t, size_t> LargestGapLocked() const;

  char* const base_;
  const size_t bytes_;
  const int numa_node_;
  const size_t floor_;  // one caller's share
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<uint64_t> waiters_;  // tickets, in arrival order
  uint64_t next_ticket_ = 0;
  std::map<size_t, size_t> slices_;  // offset -> size of every live slice
};

}  // namespace mori::umbp
