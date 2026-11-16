// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace mori::io::fabric {
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
  static_assert(sizeof(A) == 8 + 28 * K);
  return true;
}
static_assert(ValidCopyAbi<kSmallK>() && sizeof(FabricCopyArgs<kSmallK>) == 456);
static_assert(ValidCopyAbi<kLargeK>() && sizeof(FabricCopyArgs<kLargeK>) == 14344);
}  // namespace mori::io::fabric
