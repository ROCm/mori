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
#include "umbp/distributed/peer/backend/page_backend.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>
#include <msgpack.hpp>
#include <stdexcept>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/transfer/transfer_engine.h"

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

PageBackend::PageBackend(TierType tier, uint64_t page_size, TierConfig cfg,
                         std::chrono::milliseconds pending_ttl,
                         std::chrono::milliseconds read_lease_ttl,
                         std::chrono::milliseconds reaper_interval)
    : tier_(tier),
      page_size_(page_size),
      pending_ttl_(pending_ttl),
      read_lease_ttl_(read_lease_ttl),
      reaper_interval_(reaper_interval) {
  if (page_size == 0) {
    throw std::invalid_argument("PageBackend: page_size must be > 0");
  }
  InstallTierConfig(std::move(cfg));
}

// Sugar: build the host source these knobs describe and delegate, so there is
// one self-allocating constructor body and one Init path.
PageBackend::PageBackend(TierType tier, uint64_t page_size, OwnershipConfig ownership,
                         std::chrono::milliseconds pending_ttl,
                         std::chrono::milliseconds read_lease_ttl,
                         std::chrono::milliseconds reaper_interval)
    : PageBackend(tier, page_size,
                  std::make_unique<HostPageMemorySource>(HostPageMemorySource::Options{
                      ownership.use_hugepages, ownership.hugepage_size, ownership.numa_node,
                      ownership.prefault}),
                  ownership.buffer_sizes, pending_ttl, read_lease_ttl, reaper_interval) {}

PageBackend::PageBackend(TierType tier, uint64_t page_size,
                         std::unique_ptr<PageMemorySource> source,
                         std::vector<uint64_t> buffer_sizes, std::chrono::milliseconds pending_ttl,
                         std::chrono::milliseconds read_lease_ttl,
                         std::chrono::milliseconds reaper_interval)
    : tier_(tier),
      page_size_(page_size),
      pending_ttl_(pending_ttl),
      read_lease_ttl_(read_lease_ttl),
      reaper_interval_(reaper_interval),
      source_(std::move(source)),
      source_buffer_sizes_(std::move(buffer_sizes)) {
  if (page_size == 0) {
    throw std::invalid_argument("PageBackend: page_size must be > 0");
  }
  if (source_ == nullptr) {
    throw std::invalid_argument("PageBackend: PageMemorySource must not be null");
  }
  // Captured now rather than read through source_ at BuildBufferRefs time: that
  // runs for the TierConfig constructor too, where there is no source.
  buffer_loc_ = source_->LocationType();
  buffer_device_ = source_->Device();
  // Not yet configured — allocator_ stays null until Init() self-allocates.
}

PageBackend::~PageBackend() {
  StopReaper();
  Shutdown();
}

void PageBackend::InstallTierConfig(TierConfig cfg) {
  if (cfg.buffer_sizes.empty() && cfg.buffer_descs.empty()) {
    allocator_ = std::make_unique<PageBitmapAllocator>(page_size_, std::vector<uint64_t>{});
    return;
  }
  if (cfg.buffer_sizes.size() != cfg.buffer_descs.size()) {
    throw std::invalid_argument("PageBackend: buffer_sizes / buffer_descs length mismatch");
  }
  // buffer_bases is optional (deployments / tests that never pin leave it
  // empty); when present it must line up with the buffers one-to-one.
  if (!cfg.buffer_bases.empty() && cfg.buffer_bases.size() != cfg.buffer_sizes.size()) {
    throw std::invalid_argument("PageBackend: buffer_bases / buffer_sizes length mismatch");
  }
  allocator_ = std::make_unique<PageBitmapAllocator>(page_size_, cfg.buffer_sizes);
  buffer_descs_ = std::move(cfg.buffer_descs);
  buffer_bases_ = std::move(cfg.buffer_bases);
  BuildBufferRefs();
}

// Materialize the per-buffer transfer endpoints once, so BufferRef() on the
// local hot path is an index rather than an msgpack unpack.  The published size
// is the ALLOCATABLE size (total_pages * page_size), not the mapped size: it is
// the bound a same-process copy is checked against, and it is the tighter of
// the two.
void PageBackend::BuildBufferRefs() {
  buffer_refs_.clear();
  if (!allocator_) return;
  const auto& buffers = allocator_->Buffers();
  buffer_refs_.resize(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    TransferRef ref;
    ref.size = static_cast<uint64_t>(buffers[i].total_pages) * page_size_;
    // From the source that allocated these bytes, not assumed.  This is what
    // makes a local transfer against this backend select the right engine:
    // HbmCopyEngine claims the pair on loc == GPU, LocalCopyEngine on both CPU.
    ref.loc = buffer_loc_;
    ref.device = buffer_device_;
    if (i < buffer_bases_.size()) ref.host_ptr = buffer_bases_[i];
    if (i < buffer_descs_.size() && !buffer_descs_[i].empty()) {
      try {
        auto handle = msgpack::unpack(reinterpret_cast<const char*>(buffer_descs_[i].data()),
                                      buffer_descs_[i].size());
        ref.mem = handle.get().as<mori::io::MemoryDesc>();
      } catch (const std::exception& e) {
        MORI_UMBP_ERROR("[PageBackend] BuildBufferRefs: bad desc for buffer {}: {}", i, e.what());
      }
    }
    buffer_refs_[i] = std::move(ref);
  }
}

// ---------------------------------------------------------------------------
//  MediumBackend — identity + ownership
// ---------------------------------------------------------------------------

bool PageBackend::Init(MemoryRegistrar* registrar) {
  if (source_ == nullptr) {
    // Constructed with a pre-built TierConfig (test / legacy-direct path) —
    // already configured, nothing to do.
    return true;
  }
  if (owns_memory_) return true;  // idempotent

  registrar_ = registrar;

  // The medium-specific half: what kind of memory, and where.  Everything after
  // this call is identical for DRAM, HBM, or any future paged medium.
  std::vector<PageMemorySource::Buffer> buffers;
  if (!source_->Allocate(source_buffer_sizes_, &buffers)) {
    MORI_UMBP_ERROR("[PageBackend] Init: {} allocation failed for tier={}", source_->Name(),
                    static_cast<int>(tier_));
    registrar_ = nullptr;
    return false;
  }

  const mori::io::MemoryLocationType loc = source_->LocationType();
  const int device = source_->Device();

  TierConfig cfg;
  cfg.buffer_sizes.reserve(buffers.size());
  cfg.buffer_descs.reserve(buffers.size());
  cfg.buffer_bases.reserve(buffers.size());

  for (const auto& buffer : buffers) {
    std::vector<uint8_t> desc_bytes;
    if (registrar != nullptr) {
      // The backend supplies the facts a descriptor cannot recover — location
      // type and device — because the backend is the allocator (design doc §6).
      // They come from the source, which is the thing that actually knows.
      TransferRef ref = registrar->RegisterMemory(buffer.base, buffer.size, loc, device);
      owned_refs_.push_back(ref);
      if (ref.HasMemoryDesc()) {
        msgpack::sbuffer sbuf;
        msgpack::pack(sbuf, ref.mem);
        desc_bytes.assign(sbuf.data(), sbuf.data() + sbuf.size());
      }
    }

    cfg.buffer_sizes.push_back(buffer.size);
    cfg.buffer_descs.push_back(std::move(desc_bytes));
    cfg.buffer_bases.push_back(buffer.base);
  }

  const size_t buffer_count = cfg.buffer_sizes.size();
  InstallTierConfig(std::move(cfg));
  owns_memory_ = true;
  StartReaper();
  MORI_UMBP_INFO("[PageBackend] Init tier={} source={} loc={} device={} buffers={} total_bytes={}",
                 static_cast<int>(tier_), source_->Name(), static_cast<int>(loc), device,
                 buffer_count, Capacity().total_bytes);
  return true;
}

void PageBackend::Shutdown() {
  if (!owns_memory_) return;
  if (registrar_ != nullptr) {
    for (const auto& ref : owned_refs_) registrar_->Deregister(ref);
  }
  owned_refs_.clear();

  // Deregistration must precede release: the registrar may still hold an MR
  // over these pages, and freeing under it is what the ordering here prevents.
  if (source_ != nullptr) source_->Release();

  owns_memory_ = false;
  registrar_ = nullptr;
}

// ---------------------------------------------------------------------------
//  HostPageMemorySource — host DRAM, verbatim what Init() used to do inline
// ---------------------------------------------------------------------------

bool HostPageMemorySource::Allocate(const std::vector<uint64_t>& sizes, std::vector<Buffer>* out) {
  HostMemAllocator allocator;
  HostBufferOptions opts;
  opts.backing =
      opts_.use_hugepages ? HostBufferBacking::kAnonymousHugetlb : HostBufferBacking::kAnonymous;
  opts.hugepage_size = opts_.hugepage_size;
  opts.numa_node = opts_.numa_node;
  opts.prefault = opts_.prefault;

  std::vector<HostBufferHandle> taken;
  std::vector<Buffer> staged;
  for (uint64_t size : sizes) {
    if (size == 0) continue;
    HostBufferHandle handle = allocator.Alloc(size, opts);
    if (!handle.valid()) {
      MORI_UMBP_ERROR("[HostPageMemorySource] host allocation failed for size={}", size);
      // All-or-nothing: unwind only what THIS call took, leaving any earlier
      // successful Allocate (and `out`) untouched.
      for (auto& h : taken) allocator.Free(h);
      return false;
    }
    // mapped_size, not the request: hugepage rounding makes the extra usable.
    staged.push_back(Buffer{handle.ptr, handle.mapped_size});
    taken.push_back(handle);
  }

  handles_.insert(handles_.end(), taken.begin(), taken.end());
  out->insert(out->end(), staged.begin(), staged.end());
  return true;
}

void HostPageMemorySource::Release() {
  HostMemAllocator allocator;
  for (auto& handle : handles_) allocator.Free(handle);
  handles_.clear();
}

// ---------------------------------------------------------------------------
//  Allocation (reserve pages into a pending slot)
// ---------------------------------------------------------------------------

AllocateResult PageBackend::Allocate(const std::string& key, uint64_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  return AllocateLocked(key, size);
}

AllocateResult PageBackend::AllocateLocked(const std::string& key, uint64_t size) {
  auto fail = [&](AllocateOutcome outcome, const char* reason) {
    MORI_UMBP_WARN("[PageBackend] Allocate reason={} key='{}' size={} tier={}", reason, key, size,
                   static_cast<int>(tier_));
    AllocateResult r;
    r.outcome = outcome;
    return r;
  };

  const uint32_t num_pages = SizeToPages(size, page_size_);
  if (num_pages == 0) return fail(AllocateOutcome::kFailed, "ZERO_SIZE");

  // Block new allocations between ClearLocal() and full-sync ack — any
  // owned key created here would miss the empty snapshot to master.
  if (clear_full_sync_pending_.load(std::memory_order_acquire)) {
    return fail(AllocateOutcome::kFailed, "CLEAR_PENDING");
  }

  // owned_ dedup (master-index-lag fallback).  pending_ deliberately
  // not checked — same-key pending race is absorbed by Commit().
  if (owned_.find(key) != owned_.end()) {
    AllocateResult out;
    out.outcome = AllocateOutcome::kSuccessAlreadyExists;
    return out;
  }

  if (!allocator_) return fail(AllocateOutcome::kFailed, "NOT_CONFIGURED");
  auto pages = allocator_->Allocate(num_pages);
  if (!pages) return fail(AllocateOutcome::kFailedNoSpace, "NO_SPACE");

  PendingSlot slot;
  slot.slot_id = next_slot_id_.fetch_add(1, std::memory_order_relaxed);
  slot.pages = std::move(*pages);
  slot.size = size;
  slot.deadline = std::chrono::steady_clock::now() + pending_ttl_;
  slot.generation = allocator_generation_;
  const uint64_t slot_id = slot.slot_id;
  const auto slot_pages = slot.pages;
  pending_[slot_id] = std::move(slot);

  AllocateResult out;
  out.outcome = AllocateOutcome::kSuccessAllocated;
  out.slot_id = slot_id;
  out.pages = slot_pages;
  out.size = size;
  out.page_size = page_size_;
  out.pending_ttl_ms = PendingTtlMs();
  return out;
}

std::vector<AllocateResult> PageBackend::BatchAllocate(
    const std::vector<AllocateRequest>& entries) {
  std::vector<AllocateResult> out(entries.size());
  if (entries.empty()) return out;
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < entries.size(); ++i) {
    out[i] = AllocateLocked(entries[i].key, entries[i].size);
    if (out[i].outcome == AllocateOutcome::kSuccessAllocated) {
      out[i].descs = BuildBufferDescsLocked(out[i].pages);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
//  Commit / abort (promote a pending slot to owned, or release it)
// ---------------------------------------------------------------------------

bool PageBackend::Commit(uint64_t slot_id, const std::string& key, uint64_t& bytes_committed) {
  std::lock_guard<std::mutex> lock(mutex_);
  return CommitLocked(slot_id, key, bytes_committed);
}

bool PageBackend::CommitLocked(uint64_t slot_id, const std::string& key,
                               uint64_t& bytes_committed) {
  bytes_committed = 0;
  auto it = pending_.find(slot_id);
  if (it == pending_.end()) {
    MORI_UMBP_WARN("[PageBackend] Commit reason=SLOT_GONE key='{}' slot_id={}", key, slot_id);
    return false;
  }

  // Pre-clear pending slot: free pages, report Put failure, no ADD.
  if (it->second.generation != allocator_generation_) {
    MORI_UMBP_WARN("[PageBackend] Commit reason=PRE_CLEAR key='{}' slot_id={}", key, slot_id);
    if (allocator_) allocator_->Deallocate(it->second.pages);
    pending_.erase(it);
    return false;
  }

  // Race-window safety net: two writers passed Allocate() before either
  // committed.  Keep first, drop new pages, idempotent success.
  auto existing = owned_.find(key);
  if (existing != owned_.end()) {
    MORI_UMBP_WARN(
        "[PageBackend] duplicate Commit for key='{}' (existing size={}, new size={}) — keeping "
        "prior slot",
        key, existing->second.size, it->second.size);
    if (allocator_) allocator_->Deallocate(it->second.pages);
    bytes_committed = existing->second.size;
    pending_.erase(it);
    return true;
  }

  OwnedSlot owned;
  owned.pages = std::move(it->second.pages);
  owned.size = it->second.size;
  QueueEventLocked(KvEvent{KvEvent::Kind::ADD, key, tier_, owned.size});
  owned_[key] = std::move(owned);
  pending_.erase(it);
  bytes_committed = owned_[key].size;
  return true;
}

std::vector<CommitResult> PageBackend::BatchCommit(const std::vector<CommitRequest>& entries) {
  std::vector<CommitResult> out(entries.size());
  if (entries.empty()) return out;
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < entries.size(); ++i) {
    out[i].success = CommitLocked(entries[i].slot_id, entries[i].key, out[i].bytes_committed);
  }
  return out;
}

bool PageBackend::Abort(uint64_t slot_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return AbortLocked(slot_id);
}

bool PageBackend::AbortLocked(uint64_t slot_id) {
  auto it = pending_.find(slot_id);
  if (it == pending_.end()) return true;  // already reaped / aborted — idempotent
  if (allocator_) allocator_->Deallocate(it->second.pages);
  pending_.erase(it);
  return true;
}

std::vector<bool> PageBackend::BatchAbort(const std::vector<uint64_t>& slot_ids) {
  std::vector<bool> out(slot_ids.size(), false);
  if (slot_ids.empty()) return out;
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < slot_ids.size(); ++i) out[i] = AbortLocked(slot_ids[i]);
  return out;
}

// ---------------------------------------------------------------------------
//  Resolve (read path; grants a short read lease to fence eviction)
// ---------------------------------------------------------------------------

ResolvedEntry PageBackend::Resolve(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  ResolvedEntry r;
  auto it = owned_.find(key);
  if (it == owned_.end()) return r;
  r.found = true;
  r.pages = it->second.pages;
  r.size = it->second.size;
  r.page_size = page_size_;
  // Extend the read lease so concurrent Evict reports bytes_freed=0 for
  // this key.  steady_clock is monotonic and read_lease_ttl_ is fixed,
  // so this assignment is always >= any previous deadline for the key.
  read_lease_until_[key] = std::chrono::steady_clock::now() + read_lease_ttl_;
  return r;
}

std::vector<ResolvedEntry> PageBackend::BatchResolve(const std::vector<std::string>& keys,
                                                     bool include_descs) {
  std::vector<ResolvedEntry> out(keys.size());
  if (keys.empty()) return out;
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < keys.size(); ++i) {
    const auto& key = keys[i];
    auto it = owned_.find(key);
    if (it == owned_.end()) continue;
    auto& entry = out[i];
    entry.found = true;
    entry.pages = it->second.pages;
    entry.size = it->second.size;
    entry.page_size = page_size_;
    if (include_descs) entry.descs = BuildBufferDescsLocked(it->second.pages);
    // Per-key now(): matches single-key Resolve() so the last key in
    // a large batch isn't shortchanged by earlier keys' work.
    read_lease_until_[key] = std::chrono::steady_clock::now() + read_lease_ttl_;
  }
  return out;
}

bool PageBackend::Contains(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return owned_.find(key) != owned_.end();
}

// ---------------------------------------------------------------------------
//  Eviction (skips leased / copy-pinned keys; emits REMOVE)
// ---------------------------------------------------------------------------

std::vector<EvictResult> PageBackend::Evict(const std::vector<std::string>& keys) {
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
      // NOT emit REMOVE, keep the key owned.  bytes_freed=0 tells master to
      // retry; the pin is released when the copy finishes.
      out.push_back(std::move(r));
      continue;
    }
    if (allocator_) allocator_->Deallocate(it->second.pages);
    r.bytes_freed = it->second.size;
    QueueEventLocked(KvEvent{KvEvent::Kind::REMOVE, key, tier_, 0});
    owned_.erase(it);
    out.push_back(std::move(r));
  }
  return out;
}

// ---------------------------------------------------------------------------
//  DRAM copy pins (protect owned pages while the SSD copy pipeline reads them)
// ---------------------------------------------------------------------------

std::optional<PageBackend::DramCopyPin> PageBackend::AcquireDramCopyPin(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = owned_.find(key);
  if (it == owned_.end()) return std::nullopt;              // already evicted -> drop task
  if (pins_.find(key) != pins_.end()) return std::nullopt;  // duplicate task

  DramCopyPin pin;
  pin.total_size = it->second.size;
  pin.segments = BuildCopySegmentsLocked(it->second.pages, it->second.size);
  pin.pin_token = next_pin_token_++;
  pins_[key] = PinState{pin.pin_token, std::chrono::steady_clock::now()};
  return pin;
}

void PageBackend::ReleaseDramCopyPin(const std::string& key, uint64_t pin_token) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pins_.find(key);
  if (it == pins_.end() || it->second.token != pin_token) return;  // tolerate late/dup release
  pins_.erase(it);
}

bool PageBackend::HasActivePinLocked(const std::string& key) const {
  return pins_.find(key) != pins_.end();
}

std::vector<std::pair<const void*, size_t>> PageBackend::BuildCopySegmentsLocked(
    const std::vector<PageLocation>& pages, uint64_t total_size) const {
  std::vector<std::pair<const void*, size_t>> segments;
  if (buffer_bases_.empty()) return segments;
  segments.reserve(pages.size());
  uint64_t remaining = total_size;
  for (const auto& p : pages) {
    if (p.buffer_index >= buffer_bases_.size() || buffer_bases_[p.buffer_index] == nullptr) {
      MORI_UMBP_ERROR("[PageBackend] copy segment: bad buffer_index {} (bases={})", p.buffer_index,
                      buffer_bases_.size());
      return {};
    }
    // Last page may be partial; earlier pages are full page_size.
    const uint64_t bytes = std::min<uint64_t>(page_size_, remaining);
    const char* base = static_cast<const char*>(buffer_bases_[p.buffer_index]);
    segments.emplace_back(base + static_cast<uint64_t>(p.page_index) * page_size_, bytes);
    remaining -= bytes;
  }
  return segments;
}

// ---------------------------------------------------------------------------
//  Clear (distributed clear; gates writes until the full-sync ack)
// ---------------------------------------------------------------------------

void PageBackend::ClearLocal() {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto now = std::chrono::steady_clock::now();

  // Pending slots become pre-clear via generation mismatch; their pages
  // stay reserved (RDMA write may still be in flight) and are freed by the
  // writer's Commit or by the reaper's TTL path.
  ++allocator_generation_;

  // pins_ should be empty because PoolClient::Clear quiesces the copy
  // pipeline.  If not, this is a caller bug; log loudly.  We do not
  // support clearing with active copy pins and make no attempt to salvage
  // them here.  A debug assert turns this into a hard failure under
  // test/CI; release builds keep running (the freed pages cannot be
  // reused until ClearFullSyncAcked re-enables Allocate, so this stays
  // UAF-safe in practice).
  if (!pins_.empty()) {
    MORI_UMBP_ERROR(
        "[PageBackend] ClearLocal with {} active copy pin(s) — caller did not quiesce the copy "
        "pipeline (bug)",
        pins_.size());
    assert(pins_.empty() && "ClearLocal called with active copy pins; quiesce the pipeline first");
  }

  // Owned: defer pages with an active read lease (an RDMA read may still be
  // in flight) until their lease deadline; free the rest immediately.  No
  // REMOVE events — the upcoming full-sync empty snapshot collapses
  // master's index.
  for (auto& [key, slot] : owned_) {
    auto lease_it = read_lease_until_.find(key);
    if (lease_it != read_lease_until_.end() && lease_it->second > now) {
      DeferredFree df;
      df.key = key;
      df.pages = std::move(slot.pages);
      df.release_at = lease_it->second;
      deferred_frees_.push_back(std::move(df));
      continue;
    }
    if (allocator_) allocator_->Deallocate(slot.pages);
  }
  owned_.clear();

  // Active read-lease deadlines that mattered are already in deferred_frees_.
  read_lease_until_.clear();
  pins_.clear();

  // Drop any queued ADD/REMOVE that the heartbeat hasn't shipped yet: the
  // snapshot we're about to send is the authoritative state.
  pending_events_.clear();

  clear_full_sync_pending_.store(true, std::memory_order_release);
  MORI_UMBP_INFO("[PageBackend] ClearLocal — pending writes will be rejected until ack");
}

void PageBackend::ClearFullSyncAcked() {
  clear_full_sync_pending_.store(false, std::memory_order_release);
}

// ---------------------------------------------------------------------------
//  Heartbeat events (drain / snapshot) — MediumBackend + OwnedLocationSource
// ---------------------------------------------------------------------------

void PageBackend::QueueEventLocked(KvEvent event) {
  pending_events_.push_back(std::move(event));
  // Wake the heartbeat the moment the outbox first reaches the threshold.
  // Events are appended one at a time, so `==` fires at most once per
  // batch.  Called under mutex_: auto_flush_cb_ must be cheap and MUST NOT
  // re-enter this object (it only signals the heartbeat thread).
  // Exactness isn't required — the heartbeat interval is the backstop.
  // threshold_ == 0 disables size-based auto-flush entirely (ADDs then
  // ship only on the interval / explicit Flush).
  if (auto_flush_threshold_ > 0 && pending_events_.size() == auto_flush_threshold_ &&
      auto_flush_cb_) {
    auto_flush_cb_();
  }
}

std::vector<KvEvent> PageBackend::DrainPendingEvents() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<KvEvent> drained;
  drained.swap(pending_events_);
  return drained;
}

void PageBackend::SetAutoFlushHook(size_t threshold, std::function<void()> cb) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto_flush_threshold_ = threshold;
  auto_flush_cb_ = std::move(cb);
}

std::vector<KvEvent> PageBackend::SnapshotOwnedKeysLocked() const {
  std::vector<KvEvent> out;
  out.reserve(owned_.size());
  for (const auto& kv : owned_) {
    out.push_back(KvEvent{KvEvent::Kind::ADD, kv.first, tier_, kv.second.size});
  }
  return out;
}

// Test-only: a pure read-only snapshot with no outbox side effects.
// Production full-sync uses SnapshotOwnedKeysForFullSync() below; tests use
// this to assert owned state without disturbing the event outbox.
std::vector<KvEvent> PageBackend::SnapshotOwnedKeys() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return SnapshotOwnedKeysLocked();
}

std::vector<KvEvent> PageBackend::SnapshotOwnedKeysForFullSync() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto out = SnapshotOwnedKeysLocked();
  // The snapshot is now authoritative: drop the queued delta (already
  // reflected in it) so the next delta carries only new events.
  pending_events_.clear();
  return out;
}

// ---------------------------------------------------------------------------
//  Capacity & buffer-descriptor queries
// ---------------------------------------------------------------------------

uint64_t PageBackend::OwnedKeyCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return owned_.size();
}

TierCapacity PageBackend::Capacity() const {
  std::lock_guard<std::mutex> lock(mutex_);
  TierCapacity cap;
  if (allocator_) {
    cap.total_bytes = allocator_->TotalBytes();
    cap.available_bytes = allocator_->AvailableBytes();
  }
  return cap;
}

std::vector<BufferMemoryDescBytes> PageBackend::AllBufferDescs() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<BufferMemoryDescBytes> out;
  out.reserve(buffer_descs_.size());
  for (size_t i = 0; i < buffer_descs_.size(); ++i) {
    out.push_back({static_cast<uint32_t>(i), buffer_descs_[i]});
  }
  return out;
}

std::vector<BufferMemoryDescBytes> PageBackend::BufferDescsForPages(
    const std::vector<PageLocation>& pages) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return BuildBufferDescsLocked(pages);
}

// No lock: buffer_refs_ is built once by InstallTierConfig and immutable
// afterwards, and the local copy path must not contend for mutex_ (the same
// lock every Allocate/Commit/Resolve and the heartbeat snapshot take) per page.
TransferRef PageBackend::BufferRef(uint32_t buffer_index) const {
  if (buffer_index >= buffer_refs_.size()) return TransferRef{};
  return buffer_refs_[buffer_index];
}

std::vector<BufferMemoryDescBytes> PageBackend::BuildBufferDescsLocked(
    const std::vector<PageLocation>& pages) const {
  std::vector<BufferMemoryDescBytes> out;
  std::vector<uint32_t> seen;
  seen.reserve(pages.size());
  for (const auto& p : pages) {
    if (std::find(seen.begin(), seen.end(), p.buffer_index) != seen.end()) continue;
    if (p.buffer_index >= buffer_descs_.size()) continue;  // defensive: skip dangling page refs
    seen.push_back(p.buffer_index);
  }
  std::sort(seen.begin(), seen.end());
  out.reserve(seen.size());
  for (uint32_t idx : seen) out.push_back({idx, buffer_descs_[idx]});
  return out;
}

// ---------------------------------------------------------------------------
//  Read-lease helper
// ---------------------------------------------------------------------------

bool PageBackend::HasActiveReadLeaseLocked(const std::string& key) {
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

void PageBackend::StartReaper() {
  bool expected = false;
  if (!reaper_running_.compare_exchange_strong(expected, true)) return;
  reaper_thread_ = std::thread([this] { ReaperLoop(); });
}

void PageBackend::StopReaper() {
  if (!reaper_running_.exchange(false)) return;
  reaper_cv_.notify_all();
  if (reaper_thread_.joinable()) reaper_thread_.join();
}

void PageBackend::ReaperLoop() {
  while (reaper_running_.load()) {
    {
      std::unique_lock<std::mutex> lk(reaper_cv_mutex_);
      reaper_cv_.wait_for(lk, reaper_interval_, [this] { return !reaper_running_.load(); });
    }
    if (!reaper_running_.load()) break;
    ReaperSweep();
  }
}

void PageBackend::ReaperSweep() {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto now = std::chrono::steady_clock::now();

  // Expire pending slots whose deadline has passed.  No event is emitted:
  // the slot was never owned, master never indexed it.
  for (auto it = pending_.begin(); it != pending_.end();) {
    if (it->second.deadline <= now) {
      if (allocator_) allocator_->Deallocate(it->second.pages);
      MORI_UMBP_DEBUG("[PageBackend] reaped pending slot {} ({} bytes)", it->first,
                      it->second.size);
      it = pending_.erase(it);
    } else {
      ++it;
    }
  }

  // Drop expired read leases so they stop blocking eviction and the map
  // size stays bounded.
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
      MORI_UMBP_WARN("[PageBackend] copy pin for key='{}' held >{}s — copy worker stuck?", key,
                     kLongRunningPinWarn.count());
    }
  }

  // Release ClearLocal()-deferred pages whose lease deadline has passed.
  for (auto it = deferred_frees_.begin(); it != deferred_frees_.end();) {
    if (it->release_at <= now) {
      if (allocator_) allocator_->Deallocate(it->pages);
      MORI_UMBP_DEBUG("[PageBackend] released deferred key='{}' pages={}", it->key,
                      it->pages.size());
      it = deferred_frees_.erase(it);
    } else {
      ++it;
    }
  }
}

// ---------------------------------------------------------------------------
//  Factory
// ---------------------------------------------------------------------------

std::unique_ptr<MediumBackend> MakePageBackend(TierType tier, uint64_t page_size,
                                               PageBackend::OwnershipConfig ownership,
                                               std::chrono::milliseconds pending_ttl,
                                               std::chrono::milliseconds read_lease_ttl) {
  return std::make_unique<PageBackend>(tier, page_size, std::move(ownership), pending_ttl,
                                       read_lease_ttl);
}

}  // namespace mori::umbp
