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

// Shared pieces for TransferPlanBuilders; the rules are on TransferPlanBuilder.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp {

// [base, base + len) ends exactly at `next`, without overflow.
inline bool AdjacentNoOverflow(size_t base, size_t len, size_t next) {
  return len <= std::numeric_limits<size_t>::max() - base && base + len == next;
}

// [offset, offset + size) fits in `capacity` bytes.
inline bool RangeFits(uint64_t offset, uint64_t size, uint64_t capacity) {
  return size <= capacity && offset <= capacity - size;
}

// Both of `item`'s ranges fit their buffers.
inline bool ItemInBounds(const TransferItem& item) {
  return RangeFits(item.src_offset, item.size, item.src.size) &&
         RangeFits(item.dst_offset, item.size, item.dst.size);
}

inline size_t HashCombine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

// An empty plan between `item`'s endpoints.
inline TransferPlan PlanBetween(const TransferItem& item, TransferDirection dir) {
  TransferPlan plan;
  plan.src = item.src;
  plan.dst = item.dst;
  plan.dir = dir;
  return plan;
}

// Appends a segment, extending the last one when both sides are contiguous,
// `merge` allows it and the result stays <= max_merged; adds `tag` once.
inline void AppendSegment(TransferPlan* plan, size_t src_off, size_t dst_off, size_t size,
                          size_t tag, bool merge = true,
                          size_t max_merged = std::numeric_limits<size_t>::max()) {
  const bool extend = merge && !plan->sizes.empty() &&
                      AdjacentNoOverflow(plan->src_offsets.back(), plan->sizes.back(), src_off) &&
                      AdjacentNoOverflow(plan->dst_offsets.back(), plan->sizes.back(), dst_off) &&
                      size <= max_merged && plan->sizes.back() <= max_merged - size;
  if (extend) {
    plan->sizes.back() += size;
  } else {
    plan->src_offsets.push_back(src_off);
    plan->dst_offsets.push_back(dst_off);
    plan->sizes.push_back(size);
  }
  if (plan->tags.empty() || plan->tags.back() != tag) plan->tags.push_back(tag);
}

// File CPU I/O groups by memory/file range; GDS keeps one plan per GPU destination.
template <typename Engine>
class FilePlanBuilder final : public TransferPlanBuilder {
 public:
  FilePlanBuilder(const Engine& engine, size_t expected_items) : engine_(engine) {
    out_.plans.reserve(expected_items);
  }
  void Add(const TransferItem& item) override {
    if (item.size == 0) return;
    const bool gpu = item.dst.loc == mori::io::MemoryLocationType::GPU;
    if (!engine_.CanHandle(item.src, item.dst) ||
        !(gpu ? RangeFits(item.dst_offset, item.size, item.dst.size) : ItemInBounds(item))) {
      out_.rejected_tags.push_back(item.tag);
      return;
    }
    if (gpu) {
      auto plan = PlanBetween(item, TransferDirection::kLocal);
      AppendSegment(&plan, item.src_offset, item.dst_offset, item.size, item.tag);
      out_.plans.push_back(std::move(plan));
      return;
    }
    const bool write = item.dst.IsFile();
    const auto& file = write ? item.dst : item.src;
    const auto* mem = write ? item.src.host_ptr : item.dst.host_ptr;
    auto [it, fresh] = plan_of_.emplace(std::make_tuple(mem, file.file_fd, file.file_offset, write),
                                        out_.plans.size());
    if (fresh) out_.plans.push_back(PlanBetween(item, TransferDirection::kLocal));
    auto& plan = out_.plans[it->second];
    // Each item was checked against its own view; retain the widest valid view.
    plan.src.size = std::max(plan.src.size, item.src.size);
    plan.dst.size = std::max(plan.dst.size, item.dst.size);
    AppendSegment(&plan, item.src_offset, item.dst_offset, item.size, item.tag);
  }
  TransferPlanSet Finish() override { return std::move(out_); }

 private:
  const Engine& engine_;
  TransferPlanSet out_;
  std::map<std::tuple<const void*, int, uint64_t, bool>, size_t> plan_of_;
};

template <typename Engine>
std::unique_ptr<TransferPlanBuilder> MakeFilePlanBuilder(const Engine& engine,
                                                         size_t expected_items = 0) {
  return std::make_unique<FilePlanBuilder<Engine>>(engine, expected_items);
}

// Real and simulated GPU copies share the registration boundaries and gather cutoff.
inline bool GatherShapeEligible(size_t fragments, size_t bytes) {
  return fragments >= 2 && bytes / fragments < (128ULL << 10);
}

// Merges adjacent ranges, but a segment never crosses a host registration piece.
inline void AppendWithinHostPieces(TransferPlan* plan, const TransferItem& item, size_t piece) {
  const bool host_is_src = item.src.loc != mori::io::MemoryLocationType::GPU;
  const bool host_is_dst = !host_is_src && item.dst.loc != mori::io::MemoryLocationType::GPU;
  const uintptr_t host_base = host_is_src   ? reinterpret_cast<uintptr_t>(item.src.host_ptr)
                              : host_is_dst ? reinterpret_cast<uintptr_t>(item.dst.host_ptr)
                                            : 0;
  const auto host_piece = [&](size_t src_offset, size_t dst_offset) {
    return (host_base + (host_is_src ? src_offset : dst_offset)) / piece;
  };
  size_t so = static_cast<size_t>(item.src_offset);
  size_t dof = static_cast<size_t>(item.dst_offset);
  size_t remaining = static_cast<size_t>(item.size);
  while (remaining > 0) {
    size_t sz = remaining;
    if (host_base != 0) {
      const uintptr_t host = host_base + (host_is_src ? so : dof);
      sz = std::min<size_t>(sz, piece - host % piece);
    }
    const bool same_piece =
        host_base == 0 || plan->sizes.empty() ||
        host_piece(plan->src_offsets.back(), plan->dst_offsets.back()) == host_piece(so, dof);
    AppendSegment(plan, so, dof, sz, item.tag, same_piece);
    so += sz;
    dof += sz;
    remaining -= sz;
  }
}

// Builder for engines with one plan per (src, dst) buffer pair.  Admits what
// CanHandle() and ItemInBounds() accept; `append(plan, item)` adds segments.
// `Engine` is the final class, so CanHandle() is not a virtual call.
template <typename Engine, typename AppendFn>
class PairPlanBuilder final : public TransferPlanBuilder {
 public:
  PairPlanBuilder(const Engine& engine, AppendFn append)
      : engine_(engine), append_(std::move(append)) {}

  void Add(const TransferItem& item) override {
    if (item.size == 0) return;
    if (!engine_.CanHandle(item.src, item.dst) || !ItemInBounds(item)) {
      out_.rejected_tags.push_back(item.tag);
      return;
    }
    const Pair pair{item.src.host_ptr, item.dst.host_ptr};
    // Consecutive items usually share a pair: skip the lookup.
    if (out_.plans.empty() || !(pair == last_pair_)) {
      last_index_ = plan_of_.try_emplace(pair, out_.plans.size()).first->second;
      last_pair_ = pair;
      if (last_index_ == out_.plans.size()) {
        out_.plans.push_back(PlanBetween(item, TransferDirection::kLocal));
      }
    }
    append_(&out_.plans[last_index_], item);
  }

  TransferPlanSet Finish() override { return std::move(out_); }

 private:
  struct Pair {
    const void* src;
    const void* dst;
    bool operator==(const Pair& o) const noexcept { return src == o.src && dst == o.dst; }
  };
  struct PairHash {
    size_t operator()(const Pair& p) const noexcept {
      return HashCombine(std::hash<const void*>{}(p.src), std::hash<const void*>{}(p.dst));
    }
  };

  const Engine& engine_;
  AppendFn append_;
  TransferPlanSet out_;
  std::unordered_map<Pair, size_t, PairHash> plan_of_;
  Pair last_pair_{nullptr, nullptr};
  size_t last_index_ = 0;
};

template <typename Engine, typename AppendFn>
std::unique_ptr<TransferPlanBuilder> MakePairPlanBuilder(const Engine& engine, AppendFn append) {
  return std::make_unique<PairPlanBuilder<Engine, AppendFn>>(engine, std::move(append));
}

}  // namespace mori::umbp
