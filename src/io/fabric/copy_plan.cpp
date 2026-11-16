// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include "src/io/fabric/copy_plan.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mori::io::fabric {
namespace {
constexpr uint64_t kMax64 = std::numeric_limits<uint64_t>::max();
constexpr uint64_t kMax32 = std::numeric_limits<uint32_t>::max();
uint64_t CeilDiv(uint64_t n, uint64_t d) { return n / d + (n % d != 0); }
uint32_t ChunkShift(uint64_t bytes, uint32_t maxBlocks) {
  uint64_t target = std::max(kMinChunk, CeilDiv(bytes, maxBlocks));
  uint32_t shift = 12;
  // shift == 64 represents a conceptual 2^64-byte chunk (one chunk per segment).
  while (shift < 64 && (uint64_t{1} << shift) < target) ++shift;
  return shift;
}
uint64_t SegmentChunks(uint64_t size, uint32_t shift) {
  return shift == 64 ? 1 : CeilDiv(size, uint64_t{1} << shift);
}
template <int K>
CopyDispatch Dispatch(const std::vector<CopySegment>& segs, size_t begin, size_t end,
                      uint64_t bytes, uint32_t maxBlocks) {
  FabricCopyArgs<K> args{};
  args.numSegs = static_cast<uint32_t>(end - begin);
  args.chunkShift = ChunkShift(bytes, maxBlocks);
  uint64_t chunks = 0;
  for (size_t i = begin; i < end; ++i) {
    size_t j = i - begin;
    args.src[j] = segs[i].src;
    args.dst[j] = segs[i].dst;
    args.size[j] = segs[i].size;
    chunks += SegmentChunks(segs[i].size, args.chunkShift);
    args.chunkEnd[j] = static_cast<uint32_t>(chunks);
  }
  return {args, static_cast<uint32_t>(std::min<uint64_t>(chunks, maxBlocks))};
}
}  // namespace

CopyPlan PlanCopy(std::vector<CopySegment> segments, uint32_t maxBlocks, bool kernelLoaded,
                  bool remoteIsSrc) {
  if (maxBlocks == 0) throw std::invalid_argument("Fabric maxBlocks must be positive");
  CopyPlan plan;
  plan.segments.reserve(segments.size());
  auto key = [remoteIsSrc](const CopySegment& s) { return remoteIsSrc ? s.src : s.dst; };
  segments.erase(std::remove_if(segments.begin(), segments.end(),
                                [](const CopySegment& s) { return s.size == 0; }),
                 segments.end());
  auto less = [&](const CopySegment& a, const CopySegment& b) { return key(a) < key(b); };
  if (!std::is_sorted(segments.begin(), segments.end(), less))
    std::sort(segments.begin(), segments.end(), less);
  for (const auto& s : segments) {
    if (!plan.segments.empty()) {
      auto& last = plan.segments.back();
      if (last.src <= kMax64 - last.size && last.dst <= kMax64 - last.size &&
          last.size <= kMax64 - s.size && last.src + last.size == s.src &&
          last.dst + last.size == s.dst) {
        last.size += s.size;
        continue;
      }
    }
    plan.segments.push_back(s);
  }
  if (!kernelLoaded) {
    for (const auto& s : plan.segments) {
      if (s.size >= 4ull * 1024 * 1024) {
        plan.error =
            "FABRIC: fabric_copy kernel not loaded; segments >= 4 MiB cannot fall back "
            "to hipMemcpyAsync (may livelock on imported fabric pointers). "
            "Load the fabric_copy module first.";
        break;
      }
    }
    return plan;
  }
  size_t begin = 0;
  while (begin < plan.segments.size()) {
    size_t end = begin;
    uint64_t bytes = 0;
    while (end < plan.segments.size() && end - begin < kLargeK &&
           bytes <= kMax64 - plan.segments[end].size) {
      bytes += plan.segments[end++].size;
    }
    // Normally one linear pass per group. Only extreme chunk-count inputs need
    // trimming and a new adaptive chunk calculation for the shortened dispatch.
    for (;;) {
      uint32_t shift = ChunkShift(bytes, maxBlocks);
      uint64_t chunks = 0;
      size_t split = begin;
      for (; split < end; ++split) {
        chunks += SegmentChunks(plan.segments[split].size, shift);
        if (chunks > kMax32) break;
      }
      if (split == end) break;
      while (end > split) bytes -= plan.segments[--end].size;
    }
    if (end - begin <= kSmallK) {
      plan.dispatches.push_back(Dispatch<kSmallK>(plan.segments, begin, end, bytes, maxBlocks));
    } else {
      plan.dispatches.push_back(Dispatch<kLargeK>(plan.segments, begin, end, bytes, maxBlocks));
    }
    begin = end;
  }
  return plan;
}
}  // namespace mori::io::fabric
