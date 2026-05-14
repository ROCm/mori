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
#include "umbp/distributed/peer/peer_dram_allocator.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

namespace {

// Round size up to whole pages of `page_size`.  Returns 0 only when
// size == 0; the allocator treats num_pages == 0 as ENOSPC anyway.
uint32_t SizeToPages(uint64_t size, uint64_t page_size) {
  if (page_size == 0 || size == 0) return 0;
  uint64_t pages = (size + page_size - 1) / page_size;
  // Cap at uint32_t — PageBitmapAllocator uses uint32_t for page counts.
  if (pages > std::numeric_limits<uint32_t>::max()) {
    return 0;
  }
  return static_cast<uint32_t>(pages);
}

}  // namespace

PeerDramAllocator::PeerDramAllocator(uint64_t page_size, TierConfig dram, TierConfig hbm,
                                     std::chrono::milliseconds pending_ttl,
                                     std::chrono::milliseconds read_lease_ttl,
                                     std::chrono::milliseconds reaper_interval)
    : page_size_(page_size),
      pending_ttl_(pending_ttl),
      read_lease_ttl_(read_lease_ttl),
      reaper_interval_(reaper_interval) {
  if (page_size == 0) {
    throw std::invalid_argument("PeerDramAllocator: page_size must be > 0");
  }
  auto install_tier = [&](TierType tier, TierConfig& cfg) {
    if (cfg.buffer_sizes.empty() && cfg.buffer_descs.empty()) return;
    if (cfg.buffer_sizes.size() != cfg.buffer_descs.size()) {
      throw std::invalid_argument("PeerDramAllocator: buffer_sizes / buffer_descs length mismatch");
    }
    allocators_.emplace(tier, std::make_unique<PageBitmapAllocator>(page_size, cfg.buffer_sizes));
    tier_descs_.emplace(tier, std::move(cfg.buffer_descs));
  };
  install_tier(TierType::DRAM, dram);
  install_tier(TierType::HBM, hbm);
}

PeerDramAllocator::~PeerDramAllocator() { StopReaper(); }

PageBitmapAllocator* PeerDramAllocator::AllocatorForLocked(TierType tier) {
  auto it = allocators_.find(tier);
  return it == allocators_.end() ? nullptr : it->second.get();
}
const PageBitmapAllocator* PeerDramAllocator::AllocatorForLocked(TierType tier) const {
  auto it = allocators_.find(tier);
  return it == allocators_.end() ? nullptr : it->second.get();
}

std::optional<PeerDramAllocator::PendingSlot> PeerDramAllocator::Allocate(uint64_t size,
                                                                          TierType tier) {
  const uint32_t num_pages = SizeToPages(size, page_size_);
  if (num_pages == 0) return std::nullopt;

  // Refuse new allocations between ClearLocal() and the first acked
  // full-sync empty heartbeat — any owned key created in that window
  // would not appear in the snapshot we are about to ship to master.
  if (clear_full_sync_pending_.load(std::memory_order_acquire)) return std::nullopt;

  std::lock_guard<std::mutex> lock(mutex_);
  PageBitmapAllocator* alloc = AllocatorForLocked(tier);
  if (alloc == nullptr) return std::nullopt;

  auto result = alloc->Allocate(num_pages);
  if (!result) return std::nullopt;

  PendingSlot slot;
  slot.slot_id = next_slot_id_.fetch_add(1, std::memory_order_relaxed);
  slot.tier = tier;
  slot.pages = std::move(result->pages);
  slot.size = size;
  slot.deadline = std::chrono::steady_clock::now() + pending_ttl_;
  pending_[slot.slot_id] = slot;
  return slot;
}

bool PeerDramAllocator::Commit(uint64_t slot_id, const std::string& key,
                               uint64_t& bytes_committed) {
  bytes_committed = 0;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_.find(slot_id);
  if (it == pending_.end()) return false;  // reaped, aborted, or never existed

  // Slot was alive when ClearLocal() ran: release its pages now and
  // tell the writer the Put failed.  No ADD is queued, so the
  // post-clear empty snapshot stays authoritative.
  if (it->second.cancelled_by_clear) {
    if (auto* alloc = AllocatorForLocked(it->second.tier)) {
      alloc->Deallocate(it->second.pages);
    }
    pending_.erase(it);
    return false;
  }

  // If the key was already owned (legitimate idempotent retry from the
  // writer), free the prior slot's pages first so we don't leak — then
  // overwrite with the new slot.  This matches the doc's "commit raced
  // reaper" recovery path: writer retries and we converge.
  auto existing = owned_.find(key);
  if (existing != owned_.end()) {
    if (auto* prev_alloc = AllocatorForLocked(existing->second.tier)) {
      prev_alloc->Deallocate(existing->second.pages);
    }
    pending_events_.push_back(KvEvent{KvEvent::Kind::REMOVE, key, existing->second.tier, 0});
    owned_.erase(existing);
  }

  OwnedSlot owned;
  owned.tier = it->second.tier;
  owned.pages = std::move(it->second.pages);
  owned.size = it->second.size;
  pending_events_.push_back(KvEvent{KvEvent::Kind::ADD, key, owned.tier, owned.size});
  owned_[key] = std::move(owned);
  pending_.erase(it);
  bytes_committed = owned_[key].size;
  return true;
}

bool PeerDramAllocator::Abort(uint64_t slot_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_.find(slot_id);
  if (it == pending_.end()) return true;  // already reaped / aborted — idempotent
  if (auto* alloc = AllocatorForLocked(it->second.tier)) {
    alloc->Deallocate(it->second.pages);
  }
  pending_.erase(it);
  return true;
}

PeerDramAllocator::ResolveResult PeerDramAllocator::Resolve(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  ResolveResult r;
  auto it = owned_.find(key);
  if (it == owned_.end()) return r;
  r.found = true;
  r.tier = it->second.tier;
  r.pages = it->second.pages;
  r.size = it->second.size;
  // Bump read lease so concurrent Evict reports bytes_freed=0 for this key.
  read_leases_[key].push_back(std::chrono::steady_clock::now() + read_lease_ttl_);
  return r;
}

std::vector<PeerDramAllocator::EvictResult> PeerDramAllocator::Evict(
    const std::vector<std::string>& keys) {
  std::vector<EvictResult> out;
  out.reserve(keys.size());
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& key : keys) {
    EvictResult r;
    r.key = key;
    auto it = owned_.find(key);
    if (it == owned_.end()) {
      out.push_back(std::move(r));
      continue;
    }
    if (HasActiveReadLeaseLocked(key)) {
      // Master will retry next round once the lease expires.  Emit no event.
      out.push_back(std::move(r));
      continue;
    }
    if (auto* alloc = AllocatorForLocked(it->second.tier)) {
      alloc->Deallocate(it->second.pages);
    }
    r.bytes_freed = it->second.size;
    pending_events_.push_back(KvEvent{KvEvent::Kind::REMOVE, key, it->second.tier, 0});
    owned_.erase(it);
    out.push_back(std::move(r));
  }
  return out;
}

void PeerDramAllocator::ClearLocal() {
  std::lock_guard<std::mutex> lock(mutex_);
  // Owned pages: release back to the bitmap.  No REMOVE events — the
  // upcoming full-sync empty snapshot will collapse master's index
  // for this node in one shot.
  for (auto& kv : owned_) {
    if (auto* alloc = AllocatorForLocked(kv.second.tier)) {
      alloc->Deallocate(kv.second.pages);
    }
  }
  owned_.clear();

  // Pending pages: do NOT free them — an in-flight RDMA write from a
  // peer could still land on them.  Mark them so the eventual Commit
  // fails (and frees the pages) without enqueueing an ADD.  The
  // reaper's existing TTL path also handles dead writers.
  for (auto& kv : pending_) {
    kv.second.cancelled_by_clear = true;
  }

  read_leases_.clear();
  // Drop any queued ADD/REMOVE that the heartbeat hasn't shipped yet:
  // the snapshot we're about to send is the authoritative state.
  pending_events_.clear();

  clear_full_sync_pending_.store(true, std::memory_order_release);
  MORI_UMBP_INFO("[PeerDramAllocator] ClearLocal — pending writes will be rejected until ack");
}

void PeerDramAllocator::ClearFullSyncAcked() {
  clear_full_sync_pending_.store(false, std::memory_order_release);
}

void PeerDramAllocator::QueueExternalEvent(KvEvent ev) {
  std::lock_guard<std::mutex> lock(mutex_);
  pending_events_.push_back(std::move(ev));
}

std::vector<KvEvent> PeerDramAllocator::DrainPendingEvents() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<KvEvent> drained;
  drained.swap(pending_events_);
  return drained;
}

std::vector<KvEvent> PeerDramAllocator::SnapshotOwnedKeys() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<KvEvent> out;
  out.reserve(owned_.size());
  for (const auto& kv : owned_) {
    out.push_back(KvEvent{KvEvent::Kind::ADD, kv.first, kv.second.tier, kv.second.size});
  }
  return out;
}

std::map<TierType, TierCapacity> PeerDramAllocator::TierCapacitiesSnapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::map<TierType, TierCapacity> out;
  for (const auto& kv : allocators_) {
    TierCapacity cap;
    cap.total_bytes = kv.second->TotalBytes();
    cap.available_bytes = kv.second->AvailableBytes();
    out[kv.first] = cap;
  }
  return out;
}

std::vector<BufferMemoryDescBytes> PeerDramAllocator::AllBufferDescs(TierType tier) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<BufferMemoryDescBytes> out;
  auto it = tier_descs_.find(tier);
  if (it == tier_descs_.end()) return out;
  out.reserve(it->second.size());
  for (size_t i = 0; i < it->second.size(); ++i) {
    out.push_back({static_cast<uint32_t>(i), it->second[i]});
  }
  return out;
}

std::vector<BufferMemoryDescBytes> PeerDramAllocator::BufferDescsForPages(
    TierType tier, const std::vector<PageLocation>& pages) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return BuildBufferDescsLocked(tier, pages);
}

std::vector<BufferMemoryDescBytes> PeerDramAllocator::BuildBufferDescsLocked(
    TierType tier, const std::vector<PageLocation>& pages) const {
  std::vector<BufferMemoryDescBytes> out;
  auto it = tier_descs_.find(tier);
  if (it == tier_descs_.end()) return out;
  std::vector<uint32_t> seen;
  seen.reserve(pages.size());
  for (const auto& p : pages) {
    if (std::find(seen.begin(), seen.end(), p.buffer_index) != seen.end()) continue;
    if (p.buffer_index >= it->second.size()) continue;  // defensive: skip dangling page refs
    seen.push_back(p.buffer_index);
  }
  std::sort(seen.begin(), seen.end());
  out.reserve(seen.size());
  for (uint32_t idx : seen) {
    out.push_back({idx, it->second[idx]});
  }
  return out;
}

bool PeerDramAllocator::HasActiveReadLeaseLocked(const std::string& key) {
  auto it = read_leases_.find(key);
  if (it == read_leases_.end()) return false;
  const auto now = std::chrono::steady_clock::now();
  while (!it->second.empty() && it->second.front() <= now) it->second.pop_front();
  if (it->second.empty()) {
    read_leases_.erase(it);
    return false;
  }
  return true;
}

void PeerDramAllocator::StartReaper() {
  bool expected = false;
  if (!reaper_running_.compare_exchange_strong(expected, true)) return;
  reaper_thread_ = std::thread([this] { ReaperLoop(); });
}

void PeerDramAllocator::StopReaper() {
  if (!reaper_running_.exchange(false)) return;
  reaper_cv_.notify_all();
  if (reaper_thread_.joinable()) reaper_thread_.join();
}

void PeerDramAllocator::ReaperLoop() {
  while (reaper_running_.load()) {
    {
      std::unique_lock<std::mutex> lk(reaper_cv_mutex_);
      reaper_cv_.wait_for(lk, reaper_interval_, [this] { return !reaper_running_.load(); });
    }
    if (!reaper_running_.load()) break;
    ReaperSweep();
  }
}

void PeerDramAllocator::ReaperSweep() {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto now = std::chrono::steady_clock::now();

  // Expire pending slots whose deadline has passed.  No event is
  // emitted: the slot was never owned, master never indexed it.
  for (auto it = pending_.begin(); it != pending_.end();) {
    if (it->second.deadline <= now) {
      if (auto* alloc = AllocatorForLocked(it->second.tier)) {
        alloc->Deallocate(it->second.pages);
      }
      MORI_UMBP_DEBUG("[PeerDramAllocator] reaped pending slot {} ({} bytes)", it->first,
                      it->second.size);
      it = pending_.erase(it);
    } else {
      ++it;
    }
  }

  // Trim expired read leases so they stop blocking eviction.  Empty
  // entries are dropped from the map to keep its size bounded.
  for (auto it = read_leases_.begin(); it != read_leases_.end();) {
    while (!it->second.empty() && it->second.front() <= now) it->second.pop_front();
    if (it->second.empty()) {
      it = read_leases_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace mori::umbp
