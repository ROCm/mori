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
#include <utility>
#include <vector>

namespace mori::umbp {

struct PoolAllocator {
  uint64_t total_size = 0;
  uint64_t used_size = 0;

  struct OffsetTracker {
    uint64_t bump = 0;
    std::vector<std::pair<uint64_t, uint64_t>> free_list;  // {offset, size}
  };
  std::optional<OffsetTracker> offset_tracker;

  std::optional<uint64_t> Allocate(uint64_t size) {
    if (size == 0 || size > AvailableBytes()) {
      return std::nullopt;
    }

    if (!offset_tracker) {
      used_size += size;
      return uint64_t{0};
    }

    auto& tracker = *offset_tracker;

    for (auto it = tracker.free_list.begin(); it != tracker.free_list.end(); ++it) {
      if (it->second >= size) {
        uint64_t offset = it->first;
        if (it->second == size) {
          tracker.free_list.erase(it);
        } else {
          it->first += size;
          it->second -= size;
        }
        used_size += size;
        return offset;
      }
    }

    if (tracker.bump + size <= total_size) {
      uint64_t offset = tracker.bump;
      tracker.bump += size;
      used_size += size;
      return offset;
    }

    return std::nullopt;
  }

  void Deallocate(uint64_t offset, uint64_t size) {
    if (size == 0) return;
    if (size > used_size) {
      used_size = 0;
      return;
    }
    used_size -= size;

    if (!offset_tracker) return;

    auto& fl = offset_tracker->free_list;
    auto pos = std::lower_bound(fl.begin(), fl.end(), std::make_pair(offset, uint64_t{0}));
    pos = fl.insert(pos, {offset, size});

    auto next = std::next(pos);
    if (next != fl.end() && pos->first + pos->second == next->first) {
      pos->second += next->second;
      fl.erase(next);
    }

    if (pos != fl.begin()) {
      auto prev = std::prev(pos);
      if (prev->first + prev->second == pos->first) {
        prev->second += pos->second;
        fl.erase(pos);
      }
    }
  }

  uint64_t AvailableBytes() const { return total_size - used_size; }
};

}  // namespace mori::umbp
