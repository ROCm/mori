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
#include "umbp/distributed/scratch_arena.h"

#include <algorithm>

namespace mori::umbp {
namespace {

constexpr size_t AlignDown(size_t value) { return value & ~(ScratchArena::kAlignment - 1); }
constexpr size_t AlignUp(size_t value) { return AlignDown(value + ScratchArena::kAlignment - 1); }

}  // namespace

ScratchArena::ScratchArena(void* base, size_t bytes, int numa_node, size_t callers)
    : base_(static_cast<char*>(base)),
      bytes_(base != nullptr ? AlignDown(bytes) : 0),
      numa_node_(numa_node),
      floor_(AlignDown(bytes_ / std::max<size_t>(callers, 1))) {}

ScratchArena::Lease& ScratchArena::Lease::operator=(Lease&& other) noexcept {
  if (this != &other) {
    if (owner_ != nullptr) owner_->Release(offset_);
    owner_ = std::exchange(other.owner_, nullptr);
    offset_ = other.offset_;
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

ScratchArena::Lease::~Lease() {
  if (owner_ != nullptr) owner_->Release(offset_);
}

ScratchArena::Lease ScratchArena::Acquire(size_t min_bytes, size_t want_bytes) {
  if (bytes_ == 0 || min_bytes > bytes_) return Lease{};
  min_bytes = std::max(AlignUp(min_bytes), kAlignment);
  want_bytes = std::clamp(AlignUp(std::min(want_bytes, bytes_)), min_bytes, bytes_);
  // Never less than one caller's share, unless it asks for less.
  const size_t floor = std::min(want_bytes, std::max(min_bytes, floor_));

  std::unique_lock<std::mutex> lock(mutex_);
  const uint64_t ticket = next_ticket_++;
  waiters_.push_back(ticket);
  std::pair<size_t, size_t> gap;
  cv_.wait(lock, [&] {
    if (waiters_.front() != ticket) return false;
    gap = LargestGapLocked();
    return gap.second >= floor;
  });
  // An equal share with every call holding a slice or still queued, this one included.
  const size_t share = AlignDown(bytes_ / (slices_.size() + waiters_.size()));
  const size_t size = std::min({want_bytes, gap.second, std::max(share, floor)});
  waiters_.pop_front();
  slices_.emplace(gap.first, size);
  cv_.notify_all();  // the next in line may fit in what is left
  return Lease(this, gap.first, size);
}

size_t ScratchArena::Queued() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return waiters_.size();
}

void ScratchArena::Release(size_t offset) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    slices_.erase(offset);
  }
  cv_.notify_all();
}

std::pair<size_t, size_t> ScratchArena::LargestGapLocked() const {
  std::pair<size_t, size_t> best{0, 0};
  size_t end = 0;
  for (const auto& [offset, size] : slices_) {
    if (offset - end > best.second) best = {end, offset - end};
    end = offset + size;
  }
  if (bytes_ - end > best.second) best = {end, bytes_ - end};
  return best;
}

}  // namespace mori::umbp
