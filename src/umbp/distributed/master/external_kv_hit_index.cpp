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
#include "umbp/distributed/master/external_kv_hit_index.h"

#include <mutex>
#include <shared_mutex>
#include <unordered_set>

namespace mori::umbp {

size_t ExternalKvHitIndex::ShardIdx(std::string_view hash) {
  return std::hash<std::string_view>{}(hash) % kShards;
}

void ExternalKvHitIndex::UpdateLastSeen(Entry* entry, uint64_t now_ns) {
  uint64_t old = entry->last_seen_ns.load(std::memory_order_relaxed);
  while (old < now_ns && !entry->last_seen_ns.compare_exchange_weak(
                             old, now_ns, std::memory_order_relaxed, std::memory_order_relaxed)) {
  }
}

void ExternalKvHitIndex::IncrementHits(const std::vector<std::string>& unique_hashes,
                                       uint64_t now_ns) {
  for (const auto& hash : unique_hashes) {
    auto& shard = shards_[ShardIdx(hash)];
    {
      std::shared_lock lock(shard.mu);
      auto it = shard.entries.find(hash);
      if (it != shard.entries.end()) {
        Entry* entry = it->second.get();
        entry->total.fetch_add(1, std::memory_order_relaxed);
        UpdateLastSeen(entry, now_ns);
        continue;
      }
    }

    std::unique_lock lock(shard.mu);
    auto [it, inserted] = shard.entries.try_emplace(hash);
    if (inserted) {
      auto entry = std::make_unique<Entry>();
      entry->total.store(1, std::memory_order_relaxed);
      entry->last_seen_ns.store(now_ns, std::memory_order_relaxed);
      it->second = std::move(entry);
    } else {
      Entry* entry = it->second.get();
      entry->total.fetch_add(1, std::memory_order_relaxed);
      UpdateLastSeen(entry, now_ns);
    }
  }
}

size_t ExternalKvHitIndex::Lookup(const std::vector<std::string>& hashes,
                                  std::vector<std::pair<std::string, uint64_t>>* out) const {
  if (out == nullptr) return 0;
  std::unordered_set<std::string_view> seen;
  seen.reserve(hashes.size());
  size_t filled = 0;
  for (const auto& hash : hashes) {
    if (!seen.insert(hash).second) continue;
    const auto& shard = shards_[ShardIdx(hash)];
    std::shared_lock lock(shard.mu);
    auto it = shard.entries.find(hash);
    if (it == shard.entries.end()) continue;
    out->push_back({hash, it->second->total.load(std::memory_order_relaxed)});
    ++filled;
  }
  return filled;
}

size_t ExternalKvHitIndex::GarbageCollect(uint64_t cutoff_ns) {
  size_t dropped = 0;
  for (auto& shard : shards_) {
    std::unique_lock lock(shard.mu);
    auto it = shard.entries.begin();
    while (it != shard.entries.end()) {
      const uint64_t last_seen = it->second->last_seen_ns.load(std::memory_order_relaxed);
      if (last_seen < cutoff_ns) {
        it = shard.entries.erase(it);
        ++dropped;
      } else {
        ++it;
      }
    }
  }
  return dropped;
}

size_t ExternalKvHitIndex::Size() const {
  size_t size = 0;
  for (const auto& shard : shards_) {
    std::shared_lock lock(shard.mu);
    size += shard.entries.size();
  }
  return size;
}

}  // namespace mori::umbp
