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
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "umbp/distributed/peer/backend/medium_backend.h"

namespace mori::umbp {

// Private, incomplete local replicas. The cache owns allocation lifetime, not
// a TTL reaper. A shared_ptr held by each fetch/scatter fences reuse; only idle
// entries may be committed or discarded. Network transfers run outside the cache lock.
class DirectPoolGetCache {
 public:
  using Interval = std::pair<size_t, size_t>;  // [begin, end)
  struct Entry {
    MediumBackend* backend = nullptr;
    std::string key;
    AllocateResult slot;
    std::mutex mutex;
    std::condition_variable changed;
    bool fetching = false;
    bool failed = false;
    std::vector<Interval> ready;  // sorted, disjoint, coalesced
    std::chrono::steady_clock::time_point touched;

    bool Complete() const {
      return !failed && ready.size() == 1 && ready[0] == Interval{0, slot.size};
    }
    ~Entry() {
      if (!backend || slot.outcome != AllocateOutcome::kSuccessAllocated) return;
      // Last reader has completed its copy before releasing this reference.
      // Commit may deduplicate and free this allocation immediately.
      try {
        if (Complete()) {
          const auto committed = backend->BatchCommit({{slot.slot_id, key}});
          if (!committed.empty() && committed.front().success) return;
        }
      } catch (...) {
        // Destruction must still try to release an uncommitted slot.
      }
      try {
        backend->BatchAbort({slot.slot_id});
      } catch (...) {
        // Backend cleanup is best effort; do not terminate during unwinding.
      }
    }
    std::vector<Interval> Missing(const std::vector<Interval>& wanted) const {
      std::vector<Interval> missing;
      for (const auto& range : wanted) {
        size_t cursor = range.first;
        for (const auto& valid : ready) {
          if (valid.second <= cursor) continue;
          if (valid.first >= range.second) break;
          if (valid.first > cursor) missing.emplace_back(cursor, valid.first);
          cursor = std::max(cursor, valid.second);
          if (cursor >= range.second) break;
        }
        if (cursor < range.second) missing.emplace_back(cursor, range.second);
      }
      return missing;
    }
    void MarkReady(const std::vector<Interval>& ranges) {
      ready.insert(ready.end(), ranges.begin(), ranges.end());
      Normalize(&ready);
    }
    static void Normalize(std::vector<Interval>* ranges) {
      std::sort(ranges->begin(), ranges->end());
      size_t n = 0;
      for (const auto range : *ranges) {
        if (range.first == range.second) continue;
        if (n && range.first <= (*ranges)[n - 1].second) {
          (*ranges)[n - 1].second = std::max((*ranges)[n - 1].second, range.second);
        } else {
          (*ranges)[n++] = range;
        }
      }
      ranges->resize(n);
    }
  };

  explicit DirectPoolGetCache(size_t budget) : budget_(budget) {}

  std::shared_ptr<Entry> Acquire(MediumBackend* backend, const std::string& key, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = entries_.find(key);
    if (found != entries_.end()) {
      if (found->second->slot.size != size || found->second->backend != backend) return {};
      found->second->touched = std::chrono::steady_clock::now();
      return found->second;
    }
    const size_t page_size = backend->PageSize();
    if (size == 0 || page_size == 0 || size > budget_) return {};
    const size_t pages = (size - 1) / page_size + 1;
    if (pages > budget_ / page_size) return {};
    const size_t needed = pages * page_size;
    if (bytes_ > budget_ - needed) PruneLocked(true);
    if (bytes_ > budget_ - needed) return {};  // all remaining slots have readers
    // Establish ownership before allocating a slot that has no reaper deadline.
    auto entry = std::make_shared<Entry>();
    entry->backend = backend;
    entry->key = key;
    AllocateRequest request{key, size};
    request.local_lifetime = true;
    auto slots = backend->BatchAllocate({request});
    if (!slots.empty() && slots[0].outcome == AllocateOutcome::kFailedNoSpace) {
      PruneLocked(true);  // discard idle partial replicas before failing
      slots = backend->BatchAllocate({request});
    }
    if (slots.empty() || slots[0].outcome != AllocateOutcome::kSuccessAllocated) return {};
    entry->slot = std::move(slots[0]);
    const size_t charged = entry->slot.pages.size() * entry->slot.page_size;
    if (charged > budget_ || bytes_ > budget_ - charged) {
      return {};
    }
    entry->touched = std::chrono::steady_clock::now();
    entries_.emplace(key, entry);
    bytes_ += charged;
    return entry;
  }

  void Trim() {
    std::lock_guard<std::mutex> lock(mutex_);
    PruneLocked(false);
  }
  // Called after front-end reads have drained, before backend Clear/Shutdown.
  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [key, entry] : entries_) entry->failed = true;
    entries_.clear();
    bytes_ = 0;
  }
  size_t Bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return bytes_;
  }

 private:
  void PruneLocked(bool pressure) {
    const auto now = std::chrono::steady_clock::now();
    for (auto it = entries_.begin(); it != entries_.end();) {
      const auto& entry = it->second;
      bool discard = false;
      if (entry.use_count() == 1) {
        std::lock_guard<std::mutex> entry_lock(entry->mutex);
        discard = pressure || entry->failed || entry->Complete() ||
                  now - entry->touched >= std::chrono::seconds(30);
      }
      if (discard) {
        bytes_ -= entry->slot.pages.size() * entry->slot.page_size;
        it = entries_.erase(it);
      } else {
        ++it;
      }
    }
  }
  const size_t budget_;
  mutable std::mutex mutex_;
  size_t bytes_ = 0;
  std::unordered_map<std::string, std::shared_ptr<Entry>> entries_;
};

}  // namespace mori::umbp
