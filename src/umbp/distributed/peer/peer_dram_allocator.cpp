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
#include <cassert>
#include <functional>
#include <stdexcept>
#include <utility>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

// ---------------------------------------------------------------------------
//  File-local helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
//  Construction
// ---------------------------------------------------------------------------

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
    // buffer_bases is optional (deployments / tests that never pin leave it
    // empty); when present it must line up with the buffers one-to-one.
    if (!cfg.buffer_bases.empty() && cfg.buffer_bases.size() != cfg.buffer_sizes.size()) {
      throw std::invalid_argument("PeerDramAllocator: buffer_bases / buffer_sizes length mismatch");
    }
    allocators_.emplace(tier, std::make_unique<PageBitmapAllocator>(page_size, cfg.buffer_sizes));
    tier_descs_.emplace(tier, std::move(cfg.buffer_descs));
    tier_bases_.emplace(tier, std::move(cfg.buffer_bases));
  };
  install_tier(TierType::DRAM, dram);
  install_tier(TierType::HBM, hbm);
}

PeerDramAllocator::~PeerDramAllocator() { StopReaper(); }

// ---------------------------------------------------------------------------
//  Allocation (reserve pages into a pending slot)
// ---------------------------------------------------------------------------

PageBitmapAllocator* PeerDramAllocator::AllocatorForLocked(TierType tier) {
  auto it = allocators_.find(tier);
  return it == allocators_.end() ? nullptr : it->second.get();
}
const PageBitmapAllocator* PeerDramAllocator::AllocatorForLocked(TierType tier) const {
  auto it = allocators_.find(tier);
  return it == allocators_.end() ? nullptr : it->second.get();
}

PeerDramAllocator::AllocateResult PeerDramAllocator::Allocate(const std::string& key, uint64_t size,
                                                              TierType tier) {
  std::lock_guard<std::mutex> lock(mutex_);
  return AllocateLocked(key, size, tier);
}

PeerDramAllocator::AllocateResult PeerDramAllocator::AllocateLocked(const std::string& key,
                                                                    uint64_t size, TierType tier) {
  auto fail = [&](Outcome outcome, const char* reason) {
    MORI_UMBP_WARN("[PeerDramAllocator] Allocate reason={} key='{}' size={} tier={}", reason, key,
                   size, static_cast<int>(tier));
    AllocateResult r;
    r.outcome = outcome;
    return r;
  };

  const uint32_t num_pages = SizeToPages(size, page_size_);
  if (num_pages == 0) return fail(Outcome::kFailed, "ZERO_SIZE");

  // Block new allocations between ClearLocal() and full-sync ack — any
  // owned key created here would miss the empty snapshot to master.
  if (clear_full_sync_pending_.load(std::memory_order_acquire)) {
    return fail(Outcome::kFailed, "CLEAR_PENDING");
  }

  // owned_ dedup (master-index-lag fallback).  pending_ deliberately
  // not checked — same-key pending race is absorbed by Commit().
  if (owned_.find(key) != owned_.end()) {
    AllocateResult out;
    out.outcome = Outcome::kSuccessAlreadyExists;
    return out;
  }

  PageBitmapAllocator* alloc = AllocatorForLocked(tier);
  if (alloc == nullptr) return fail(Outcome::kFailed, "BAD_TIER");

  auto pages = alloc->Allocate(num_pages);
  if (!pages) return fail(Outcome::kFailedNoSpace, "NO_SPACE");

  AllocateResult out;

  PendingSlot slot;
  slot.slot_id = next_slot_id_.fetch_add(1, std::memory_order_relaxed);
  slot.tier = tier;
  slot.pages = std::move(*pages);
  slot.size = size;
  slot.deadline = std::chrono::steady_clock::now() + pending_ttl_;
  slot.generation = allocator_generation_;
  pending_[slot.slot_id] = slot;
  out.outcome = Outcome::kSuccessAllocated;
  out.slot = std::move(slot);
  return out;
}

std::vector<PeerDramAllocator::BatchAllocateResult> PeerDramAllocator::BatchAllocate(
    const std::vector<AllocateRequest>& entries) {
  std::vector<BatchAllocateResult> out(entries.size());
  if (entries.empty()) return out;
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& entry = entries[i];
    auto result = AllocateLocked(entry.key, entry.size, entry.tier);
    out[i].outcome = result.outcome;
    out[i].slot = std::move(result.slot);
    if (out[i].outcome == Outcome::kSuccessAllocated && out[i].slot.has_value()) {
      out[i].descs = BuildBufferDescsLocked(out[i].slot->tier, out[i].slot->pages);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
//  Commit / abort (promote a pending slot to owned, or release it)
// ---------------------------------------------------------------------------

bool PeerDramAllocator::Commit(uint64_t slot_id, const std::string& key,
                               uint64_t& bytes_committed) {
  std::lock_guard<std::mutex> lock(mutex_);
  return CommitLocked(slot_id, key, bytes_committed);
}

bool PeerDramAllocator::CommitLocked(uint64_t slot_id, const std::string& key,
                                     uint64_t& bytes_committed) {
  bytes_committed = 0;
  auto it = pending_.find(slot_id);
  if (it == pending_.end()) {
    MORI_UMBP_WARN("[PeerDramAllocator] Commit reason=SLOT_GONE key='{}' slot_id={}", key, slot_id);
    return false;
  }

  // Pre-clear pending slot: free pages, report Put failure, no ADD.
  if (it->second.generation != allocator_generation_) {
    MORI_UMBP_WARN("[PeerDramAllocator] Commit reason=PRE_CLEAR key='{}' slot_id={}", key, slot_id);
    if (auto* alloc = AllocatorForLocked(it->second.tier)) {
      alloc->Deallocate(it->second.pages);
    }
    pending_.erase(it);
    return false;
  }

  // Race-window safety net: two writers passed Allocate() before either
  // committed.  Keep first, drop new pages, idempotent success.
  auto existing = owned_.find(key);
  if (existing != owned_.end()) {
    MORI_UMBP_WARN(
        "[PeerDramAllocator] duplicate Commit for key='{}' "
        "(existing tier={} size={}, new size={}) — keeping prior slot",
        key, static_cast<int>(existing->second.tier), existing->second.size, it->second.size);
    if (auto* alloc = AllocatorForLocked(it->second.tier)) {
      alloc->Deallocate(it->second.pages);
    }
    bytes_committed = existing->second.size;
    pending_.erase(it);
    return true;
  }

  OwnedSlot owned;
  owned.tier = it->second.tier;
  owned.pages = std::move(it->second.pages);
  owned.size = it->second.size;
  QueueEventLocked(KvEvent{KvEvent::Kind::ADD, key, owned.tier, owned.size});
  owned_[key] = std::move(owned);
  pending_.erase(it);
  bytes_committed = owned_[key].size;
  return true;
}

std::vector<PeerDramAllocator::CommitResult> PeerDramAllocator::BatchCommit(
    const std::vector<CommitRequest>& entries) {
  std::vector<CommitResult> out(entries.size());
  if (entries.empty()) return out;
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& entry = entries[i];
    out[i].success = CommitLocked(entry.slot_id, entry.key, out[i].bytes_committed);
  }
  return out;
}

bool PeerDramAllocator::Abort(uint64_t slot_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return AbortLocked(slot_id);
}

bool PeerDramAllocator::AbortLocked(uint64_t slot_id) {
  auto it = pending_.find(slot_id);
  if (it == pending_.end()) return true;  // already reaped / aborted — idempotent
  if (auto* alloc = AllocatorForLocked(it->second.tier)) {
    alloc->Deallocate(it->second.pages);
  }
  pending_.erase(it);
  return true;
}

std::vector<bool> PeerDramAllocator::BatchAbort(const std::vector<uint64_t>& slot_ids) {
  std::vector<bool> out(slot_ids.size(), false);
  if (slot_ids.empty()) return out;
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < slot_ids.size(); ++i) {
    out[i] = AbortLocked(slot_ids[i]);
  }
  return out;
}

// ---------------------------------------------------------------------------
//  Resolve (read path; grants a short read lease to fence eviction)
// ---------------------------------------------------------------------------

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

std::vector<PeerDramAllocator::ResolvedEntry> PeerDramAllocator::BatchResolve(
    const std::vector<std::string>& keys, bool include_descs) {
  std::vector<ResolvedEntry> out(keys.size());
  if (keys.empty()) return out;
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < keys.size(); ++i) {
    const auto& key = keys[i];
    auto it = owned_.find(key);
    if (it == owned_.end()) continue;
    auto& entry = out[i];
    entry.found = true;
    entry.tier = it->second.tier;
    entry.pages = it->second.pages;
    entry.size = it->second.size;
    if (include_descs) {
      entry.descs = BuildBufferDescsLocked(it->second.tier, it->second.pages);
    }
    // Per-key now(): matches single-key Resolve() so the last key in
    // a large batch isn't shortchanged by earlier keys' work.
    read_lease_until_[key] = std::chrono::steady_clock::now() + read_lease_ttl_;
  }
  return out;
}

// ---------------------------------------------------------------------------
//  Eviction (skips leased / copy-pinned keys; emits REMOVE)
// ---------------------------------------------------------------------------

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
    if (HasActivePinLocked(key)) {
      // An SSD copy worker is reading these pages.  Do NOT free them, do
      // NOT emit REMOVE DRAM, keep the key owned.  bytes_freed=0 tells
      // master to retry; the pin is released when the copy finishes.
      out.push_back(std::move(r));
      continue;
    }
    if (auto* alloc = AllocatorForLocked(it->second.tier)) {
      alloc->Deallocate(it->second.pages);
    }
    r.bytes_freed = it->second.size;
    QueueEventLocked(KvEvent{KvEvent::Kind::REMOVE, key, it->second.tier, 0});
    owned_.erase(it);
    out.push_back(std::move(r));
  }
  return out;
}

// ---------------------------------------------------------------------------
//  DRAM copy pins (protect owned pages while the SSD copy pipeline reads them)
// ---------------------------------------------------------------------------

std::optional<PeerDramAllocator::DramCopyPin> PeerDramAllocator::AcquireDramCopyPin(
    const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = owned_.find(key);
  if (it == owned_.end()) return std::nullopt;              // already evicted -> drop task
  if (pins_.find(key) != pins_.end()) return std::nullopt;  // duplicate task

  DramCopyPin pin;
  pin.total_size = it->second.size;
  pin.segments = BuildCopySegmentsLocked(it->second.tier, it->second.pages, it->second.size);
  pin.pin_token = next_pin_token_++;
  pins_[key] = PinState{pin.pin_token, std::chrono::steady_clock::now()};
  return pin;
}

void PeerDramAllocator::ReleaseDramCopyPin(const std::string& key, uint64_t pin_token) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pins_.find(key);
  if (it == pins_.end() || it->second.token != pin_token) return;  // tolerate late/dup release
  pins_.erase(it);
}

bool PeerDramAllocator::HasActivePinLocked(const std::string& key) const {
  return pins_.find(key) != pins_.end();
}

std::vector<std::pair<const void*, size_t>> PeerDramAllocator::BuildCopySegmentsLocked(
    TierType tier, const std::vector<PageLocation>& pages, uint64_t total_size) const {
  std::vector<std::pair<const void*, size_t>> segments;
  auto it = tier_bases_.find(tier);
  if (it == tier_bases_.end() || it->second.empty()) return segments;
  const auto& bases = it->second;
  segments.reserve(pages.size());
  uint64_t remaining = total_size;
  for (const auto& p : pages) {
    if (p.buffer_index >= bases.size() || bases[p.buffer_index] == nullptr) {
      MORI_UMBP_ERROR("[PeerDramAllocator] copy segment: bad buffer_index {} (bases={})",
                      p.buffer_index, bases.size());
      return {};
    }
    // Last page may be partial; earlier pages are full page_size.
    const uint64_t bytes = std::min<uint64_t>(page_size_, remaining);
    const char* base = static_cast<const char*>(bases[p.buffer_index]);
    segments.emplace_back(base + static_cast<uint64_t>(p.page_index) * page_size_, bytes);
    remaining -= bytes;
  }
  return segments;
}

// ---------------------------------------------------------------------------
//  Clear (distributed clear; gates writes until the full-sync ack)
// ---------------------------------------------------------------------------

void PeerDramAllocator::ClearLocal() {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto now = std::chrono::steady_clock::now();

  // Pending slots become pre-clear via generation mismatch; their
  // pages stay reserved (RDMA write may still be in flight) and are
  // freed by the writer's Commit or by the reaper's TTL path.
  ++allocator_generation_;

  // pins_ should be empty because PoolClient::Clear quiesces the copy pipeline.
  // If not, this is a caller bug; log loudly.  We do not support clearing with
  // active copy pins and make no attempt to salvage them here.  A debug assert
  // turns this into a hard failure under test/CI; release builds keep running
  // (the freed pages cannot be reused until ClearFullSyncAcked re-enables
  // Allocate, so this stays UAF-safe in practice).
  if (!pins_.empty()) {
    MORI_UMBP_ERROR(
        "[PeerDramAllocator] ClearLocal with {} active copy pin(s) — caller did not quiesce "
        "the copy pipeline (bug)",
        pins_.size());
    assert(pins_.empty() && "ClearLocal called with active copy pins; quiesce the pipeline first");
  }

  // Owned: defer pages with an active read lease (an RDMA read may still be in
  // flight) until their lease deadline; free the rest immediately.  No REMOVE
  // events — the upcoming full-sync empty snapshot collapses master's index.
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

  // Active read-lease deadlines that mattered are already in deferred_frees_.
  read_lease_until_.clear();
  pins_.clear();

  // Drop any queued ADD/REMOVE that the heartbeat hasn't shipped yet:
  // the snapshot we're about to send is the authoritative state.
  pending_events_.clear();

  clear_full_sync_pending_.store(true, std::memory_order_release);
  MORI_UMBP_INFO("[PeerDramAllocator] ClearLocal — pending writes will be rejected until ack");
}

void PeerDramAllocator::ClearFullSyncAcked() {
  clear_full_sync_pending_.store(false, std::memory_order_release);
}

// ---------------------------------------------------------------------------
//  OwnedLocationSource (heartbeat events: queue / drain / snapshot)
// ---------------------------------------------------------------------------

void PeerDramAllocator::QueueEventLocked(KvEvent event) {
  pending_events_.push_back(std::move(event));
  // Wake the heartbeat the moment the outbox first reaches the threshold.  Events
  // are appended one at a time, so `==` fires at most once per batch.  Called
  // under mutex_: auto_flush_cb_ must be cheap and MUST NOT re-enter the
  // allocator (it only signals the heartbeat thread).  Exactness isn't required —
  // the heartbeat interval is the backstop.
  if (pending_events_.size() == auto_flush_threshold_ && auto_flush_cb_) {
    auto_flush_cb_();
  }
}

std::vector<KvEvent> PeerDramAllocator::DrainPendingEvents() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<KvEvent> drained;
  drained.swap(pending_events_);
  return drained;
}

void PeerDramAllocator::SetAutoFlushHook(size_t threshold, std::function<void()> cb) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto_flush_threshold_ = threshold;
  auto_flush_cb_ = std::move(cb);
}

std::vector<KvEvent> PeerDramAllocator::SnapshotOwnedKeysLocked() const {
  std::vector<KvEvent> out;
  out.reserve(owned_.size());
  for (const auto& kv : owned_) {
    out.push_back(KvEvent{KvEvent::Kind::ADD, kv.first, kv.second.tier, kv.second.size});
  }
  return out;
}

// Test-only: a pure read-only snapshot with no outbox side effects.  Production
// full-sync uses SnapshotOwnedKeysForFullSync() below; tests use this to assert
// owned state without disturbing the event outbox.  Kept to satisfy the
// OwnedLocationSource interface and the unit tests; not on any production path.
std::vector<KvEvent> PeerDramAllocator::SnapshotOwnedKeys() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return SnapshotOwnedKeysLocked();
}

std::vector<KvEvent> PeerDramAllocator::SnapshotOwnedKeysForFullSync() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto out = SnapshotOwnedKeysLocked();
  // The snapshot is now authoritative: drop the queued delta (already reflected
  // in it) so the next delta carries only new events.
  pending_events_.clear();
  return out;
}

// ---------------------------------------------------------------------------
//  Capacity & buffer-descriptor queries
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
//  Read-lease helper
// ---------------------------------------------------------------------------

bool PeerDramAllocator::HasActiveReadLeaseLocked(const std::string& key) {
  auto it = read_lease_until_.find(key);
  if (it == read_lease_until_.end()) return false;
  if (it->second <= std::chrono::steady_clock::now()) {
    read_lease_until_.erase(it);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
//  Reaper (background sweep: pending TTL, expired leases, deferred frees)
// ---------------------------------------------------------------------------

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

  // Warn about copy pins held far longer than any healthy copy should
  // take.  We never force-free a pin (a worker may still be reading its
  // segments — freeing would be a use-after-free); this is purely an
  // observability signal that a copy worker is stuck.
  constexpr std::chrono::seconds kLongRunningPinWarn{30};
  for (const auto& [key, pin] : pins_) {
    if (now - pin.acquired_at > kLongRunningPinWarn) {
      MORI_UMBP_WARN("[PeerDramAllocator] copy pin for key='{}' held >{}s — copy worker stuck?",
                     key, kLongRunningPinWarn.count());
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
