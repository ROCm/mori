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
  slot.generation = allocator_generation_;
  pending_[slot.slot_id] = slot;
  return slot;
}

bool PeerDramAllocator::Commit(uint64_t slot_id, const std::string& key,
                               uint64_t& bytes_committed) {
  bytes_committed = 0;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_.find(slot_id);
  if (it == pending_.end()) return false;  // reaped, aborted, or never existed

  // Pre-clear pending slot: free pages, report Put failure, no ADD.
  if (it->second.generation != allocator_generation_) {
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
  // Extend the read lease so concurrent Evict reports bytes_freed=0 for
  // this key.  steady_clock is monotonic and read_lease_ttl_ is fixed,
  // so this assignment is always >= any previous deadline for the key.
  read_lease_until_[key] = std::chrono::steady_clock::now() + read_lease_ttl_;
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
  const auto now = std::chrono::steady_clock::now();

  // Pending slots become pre-clear via generation mismatch; their
  // pages stay reserved (RDMA write may still be in flight) and are
  // freed by the writer's Commit or by the reaper's TTL path.
  ++allocator_generation_;

  // Owned: defer pages with an active read lease (RDMA read may still
  // be in flight); free others immediately.  No REMOVE events — the
  // upcoming full-sync empty snapshot collapses master's index.
  for (auto& [key, slot] : owned_) {
    auto lease_it = read_lease_until_.find(key);
    if (lease_it != read_lease_until_.end() && lease_it->second > now) {
      DeferredFree df;
      df.key = key;
      df.tier = slot.tier;
      df.pages = std::move(slot.pages);
      df.release_at = lease_it->second;
      deferred_frees_.push_back(std::move(df));
      continue;
    }
    if (auto* alloc = AllocatorForLocked(slot.tier)) {
      alloc->Deallocate(slot.pages);
    }
  }
  owned_.clear();

  // Active deadlines that mattered are already in deferred_frees_.
  read_lease_until_.clear();

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

std::map<TierType, uint64_t> PeerDramAllocator::OwnedKeyCountByTier() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::map<TierType, uint64_t> result;
  for (TierType t : {TierType::HBM, TierType::DRAM, TierType::SSD}) result[t] = 0;
  for (const auto& [key, slot] : owned_) result[slot.tier]++;
  return result;
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
  auto it = read_lease_until_.find(key);
  if (it == read_lease_until_.end()) return false;
  if (it->second <= std::chrono::steady_clock::now()) {
    read_lease_until_.erase(it);
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

  // Drop expired read leases so they stop blocking eviction and the
  // map size stays bounded.
  for (auto it = read_lease_until_.begin(); it != read_lease_until_.end();) {
    if (it->second <= now) {
      it = read_lease_until_.erase(it);
    } else {
      ++it;
    }
  }

  // Release ClearLocal()-deferred pages whose lease deadline has passed.
  for (auto it = deferred_frees_.begin(); it != deferred_frees_.end();) {
    if (it->release_at <= now) {
      if (auto* alloc = AllocatorForLocked(it->tier)) {
        alloc->Deallocate(it->pages);
      }
      MORI_UMBP_DEBUG("[PeerDramAllocator] released deferred key='{}' pages={}", it->key,
                      it->pages.size());
      it = deferred_frees_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace mori::umbp
