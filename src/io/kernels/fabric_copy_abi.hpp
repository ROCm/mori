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

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace mori::io::fabric {
inline constexpr int kSingleK = 1;
inline constexpr int kSmallK = 16;
inline constexpr int kLargeK = 512;
inline constexpr uint64_t kMinChunk = 4096;

template <int K>
struct FabricCopyArgs {
  uint32_t numSegs;
  uint32_t chunkShift;  // log2(chunk bytes); 64 represents a conceptual 2^64-byte chunk
  uint64_t src[K];
  uint64_t dst[K];
  uint64_t size[K];
  uint32_t chunkEnd[K];
};

template <int K>
constexpr bool ValidCopyAbi() {
  using A = FabricCopyArgs<K>;
  static_assert(std::is_standard_layout_v<A> && std::is_trivially_copyable_v<A>);
  static_assert(alignof(A) == 8);
  static_assert(offsetof(A, numSegs) == 0 && offsetof(A, chunkShift) == 4);
  static_assert(offsetof(A, src) == 8);
  static_assert(offsetof(A, dst) == 8 + 8 * K);
  static_assert(offsetof(A, size) == 8 + 16 * K);
  static_assert(offsetof(A, chunkEnd) == 8 + 24 * K);
  static_assert(sizeof(A) == (8 + 28 * K + 7) / 8 * 8);
  return true;
}
static_assert(ValidCopyAbi<kSingleK>() && sizeof(FabricCopyArgs<kSingleK>) == 40);
static_assert(ValidCopyAbi<kSmallK>() && sizeof(FabricCopyArgs<kSmallK>) == 456);
static_assert(ValidCopyAbi<kLargeK>() && sizeof(FabricCopyArgs<kLargeK>) == 14344);
}  // namespace mori::io::fabric
