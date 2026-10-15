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

// Shared pieces for engines' TransferPlanBuilders (see TransferPlanBuilder for
// the rules every builder keeps).  PairPlanBuilder is a whole builder for an
// engine with one plan per (src, dst) buffer pair; the helpers serve any other.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/distributed/transfer/transfer_engine.h"

namespace mori::umbp {

// True iff [next, ...) starts exactly where [base, base + len) ends, with no
// overflow in base + len.
inline bool AdjacentNoOverflow(size_t base, size_t len, size_t next) {
  return len <= std::numeric_limits<size_t>::max() - base && base + len == next;
}

// Does [offset, offset + size) fit a buffer of `capacity` bytes?
inline bool RangeFits(uint64_t offset, uint64_t size, uint64_t capacity) {
  return size <= capacity && offset <= capacity - size;
}

// Both sides of `item` within their buffers.  Checked once per item: a range
// past the end is a bug in the caller's slot bookkeeping, and its key must be
// failed rather than copied past the end.
inline bool ItemInBounds(const TransferItem& item) {
  return RangeFits(item.src_offset, item.size, item.src.size) &&
         RangeFits(item.dst_offset, item.size, item.dst.size);
}

inline size_t HashCombine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

// A new plan between `item`'s endpoints, with no segments yet.
inline TransferPlan PlanBetween(const TransferItem& item, TransferDirection dir) {
  TransferPlan plan;
  plan.src = item.src;
  plan.dst = item.dst;
  plan.dir = dir;
  return plan;
}

// Appends one segment to `plan`, extending its last segment instead when both
// sides continue it, `merge` allows it, and the result stays <= max_merged.
// Records `tag` once per run of segments from the same item.
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

// The builder of an engine with one local plan per (src, dst) buffer pair:
// the memcpy engines.  It takes an item when the engine's CanHandle() takes the
// pair and both ranges are in bounds, and `append(plan, item)` adds the item's
// segment(s) to its pair's plan.  `Engine` is the final class, so CanHandle()
// is not a virtual call per item.
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
    // A key's pages arrive back to back, so most items skip the lookup.
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
  // Grows with distinct pairs, not with the items of the batch.
  std::unordered_map<Pair, size_t, PairHash> plan_of_;
  Pair last_pair_{nullptr, nullptr};
  size_t last_index_ = 0;
};

template <typename Engine, typename AppendFn>
std::unique_ptr<TransferPlanBuilder> MakePairPlanBuilder(const Engine& engine, AppendFn append) {
  return std::make_unique<PairPlanBuilder<Engine, AppendFn>>(engine, std::move(append));
}

}  // namespace mori::umbp
