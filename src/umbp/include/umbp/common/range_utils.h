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
#include <numeric>
#include <vector>

namespace mori::umbp {

inline bool IsObjectRangeOverflow(size_t offset, size_t size, size_t limit) {
  return size > limit || offset > limit - size;
}

template <typename Ptr>
inline bool RangeBatchShapeValid(size_t entry_count, const std::vector<std::vector<Ptr>>& ptrs,
                                 const std::vector<std::vector<size_t>>& sizes,
                                 const std::vector<std::vector<size_t>>& offsets) {
  if (ptrs.size() != entry_count || sizes.size() != entry_count || offsets.size() != entry_count) {
    return false;
  }
  for (size_t i = 0; i < entry_count; ++i) {
    if (ptrs[i].size() != sizes[i].size() || ptrs[i].size() != offsets[i].size()) {
      return false;
    }
  }
  return true;
}

inline bool RangesTileObject(size_t object_size, const std::vector<size_t>& sizes,
                             const std::vector<size_t>& offsets) {
  if (object_size == 0 || sizes.empty() || sizes.size() != offsets.size()) return false;

  std::vector<size_t> order(sizes.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return offsets[a] < offsets[b]; });

  size_t cursor = 0;
  for (size_t index : order) {
    const size_t size = sizes[index];
    const size_t offset = offsets[index];
    if (size == 0 || offset != cursor || IsObjectRangeOverflow(offset, size, object_size)) {
      return false;
    }
    cursor = offset + size;  // Safe after IsObjectRangeOverflow.
  }
  return cursor == object_size;
}

}  // namespace mori::umbp
