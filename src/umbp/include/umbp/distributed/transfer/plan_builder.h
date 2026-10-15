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

// What every engine's incremental TransferPlanBuilder shares.  An engine
// describes only what is its own -- which items it takes, how it groups them,
// how a segment is appended -- in a small Traits struct, and
// GroupedPlanBuilder<Traits> does the rest.  Plan() is then PlanViaBuilder(),
// so batch and incremental planning cannot drift apart.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
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

// Groups by the two endpoints' addresses: the plan key of the memcpy engines.
struct PtrPairKey {
  const void* src;
  const void* dst;
  bool operator==(const PtrPairKey& o) const noexcept { return src == o.src && dst == o.dst; }
};
struct PtrPairKeyHash {
  size_t operator()(const PtrPairKey& k) const noexcept {
    return HashCombine(std::hash<const void*>{}(k.src), std::hash<const void*>{}(k.dst));
  }
};

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

// Plan() of an engine whose CreatePlanBuilder() is incremental: that builder
// over a vector.  Never call it from an engine that keeps the default
// (buffered) builder, whose Finish() calls Plan().
inline TransferPlanSet PlanViaBuilder(const TransferEngine& engine,
                                      const std::vector<TransferItem>& items) {
  auto builder = engine.CreatePlanBuilder(items.size());
  for (const auto& item : items) builder->Add(item);
  return builder->Finish();
}

// An incremental builder from a Traits struct.  The builder skips empty items,
// collects rejected tags, keeps plans in first-seen order, and hands the set
// over in Finish().  Traits supplies:
//
//   static constexpr bool kGroups;  // false: one plan per item, no lookup
//   using Key; using KeyHash;
//   // The item's group, or nullopt to reject it.
//   std::optional<Key> Admit(const TransferItem&);
//   // May plan `index` (already in the item's group) take the item?
//   bool Joins(size_t index, const TransferPlan&, const Key&, const TransferItem&);
//   TransferPlan NewPlan(const Key&, const TransferItem&);
//   void Append(size_t index, TransferPlan*, const Key&, const TransferItem&);
//
// A group may hold several plans: a key that only buckets endpoints (MORI-IO's
// per-engine memory ids), or a plan that can fill up (a bounce pool).  Joins
// picks among them; the first that accepts takes the item, else a new plan
// opens in the group.
template <typename Traits>
class GroupedPlanBuilder final : public TransferPlanBuilder {
 public:
  explicit GroupedPlanBuilder(Traits traits, size_t expected_items = 0)
      : traits_(std::move(traits)) {
    if constexpr (!Traits::kGroups) out_.plans.reserve(expected_items);
  }

  void Add(const TransferItem& item) override {
    if (item.size == 0) return;
    const std::optional<typename Traits::Key> key = traits_.Admit(item);
    if (!key.has_value()) {
      out_.rejected_tags.push_back(item.tag);
      return;
    }
    size_t index = out_.plans.size();
    if constexpr (Traits::kGroups) {
      // Grows with distinct groups, not with the items of the batch.
      std::vector<size_t>& group = groups_[*key];
      bool joined = false;
      for (size_t candidate : group) {
        if (traits_.Joins(candidate, out_.plans[candidate], *key, item)) {
          index = candidate;
          joined = true;
          break;
        }
      }
      if (!joined) group.push_back(index);
    }
    if (index == out_.plans.size()) out_.plans.push_back(traits_.NewPlan(*key, item));
    traits_.Append(index, &out_.plans[index], *key, item);
  }

  TransferPlanSet Finish() override { return std::move(out_); }

 private:
  Traits traits_;
  TransferPlanSet out_;
  std::unordered_map<typename Traits::Key, std::vector<size_t>, typename Traits::KeyHash> groups_;
};

}  // namespace mori::umbp
