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
#include "umbp/distributed/peer/backend/ssd_backend.h"

#include <algorithm>
#include <msgpack.hpp>
#include <stdexcept>
#include <utility>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {

namespace {
constexpr uint32_t kNoPage = UINT32_MAX;
constexpr uint32_t kStagingBufferIndex = 0;
}  // namespace

SsdBackend::SsdBackend(Config cfg) : cfg_(std::move(cfg)) {
  if (cfg_.page_size == 0) {
    throw std::invalid_argument("SsdBackend: page_size must be > 0");
  }
  if (cfg_.staging_pages == 0) {
    throw std::invalid_argument("SsdBackend: staging_pages must be > 0");
  }
  ssd_ = std::make_unique<PeerSsdManager>(cfg_.ssd);
}

SsdBackend::~SsdBackend() {
  StopReaper();
  Shutdown();
}

// ---------------------------------------------------------------------------
//  Ownership
// ---------------------------------------------------------------------------

bool SsdBackend::Init(MemoryRegistrar* registrar) {
  if (initialized_) return true;
  if (!cfg_.ssd.enabled) {
    MORI_UMBP_ERROR("[SsdBackend] Init: SSD tier is disabled in config");
    return false;
  }

  registrar_ = registrar;

  // ONE buffer covering every staging page, so each page is contiguous and a
  // single registration serves the arena.  Host memory deliberately: this is
  // what the backend publishes, and publishing ordinary registered DRAM is the
  // entire reason SSD needs no transfer-layer change (see the header).
  staging_size_ = static_cast<uint64_t>(cfg_.staging_pages) * cfg_.page_size;
  staging_source_ = std::make_unique<HostPageMemorySource>(HostPageMemorySource::Options{});

  std::vector<PageMemorySource::Buffer> buffers;
  if (!staging_source_->Allocate({staging_size_}, &buffers) || buffers.empty()) {
    MORI_UMBP_ERROR("[SsdBackend] Init: staging arena allocation failed ({} bytes)", staging_size_);
    registrar_ = nullptr;
    return false;
  }
  staging_base_ = buffers[0].base;
  staging_size_ = buffers[0].size;

  if (registrar != nullptr) {
    buffer_ref_ = registrar->RegisterMemory(
        staging_base_, staging_size_, staging_source_->LocationType(), staging_source_->Device());
    if (buffer_ref_.HasMemoryDesc()) {
      msgpack::sbuffer sbuf;
      msgpack::pack(sbuf, buffer_ref_.mem);
      buffer_desc_.assign(sbuf.data(), sbuf.data() + sbuf.size());
    }
  } else {
    buffer_ref_ = TransferRef::HostBytes(
        staging_base_, staging_size_, staging_source_->LocationType(), staging_source_->Device());
  }

  // Free list, high index first so page 0 is handed out first (nicer logs).
  const uint32_t usable_pages = static_cast<uint32_t>(staging_size_ / cfg_.page_size);
  free_pages_.reserve(usable_pages);
  for (uint32_t i = usable_pages; i > 0; --i) free_pages_.push_back(i - 1);

  // Crash-restart leftover: metadata is gone but files may remain, so used
  // capacity would diverge from owned_.  Must run before any IO.
  ssd_->DiscardLeftoverOnStartup();

  initialized_ = true;
  StartReaper();
  MORI_UMBP_INFO("[SsdBackend] Init staging_pages={} page_size={} arena_bytes={}", usable_pages,
                 cfg_.page_size, staging_size_);
  return true;
}

void SsdBackend::Shutdown() {
  if (!initialized_) return;
  StopReaper();

  if (registrar_ != nullptr && buffer_ref_.Valid()) registrar_->Deregister(buffer_ref_);
  buffer_ref_ = TransferRef{};
  buffer_desc_.clear();

  // Deregister before release, same ordering rule as PageBackend::Shutdown.
  if (staging_source_ != nullptr) staging_source_->Release();
  staging_source_.reset();
  staging_base_ = nullptr;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.clear();
    for (auto& [key, lease] : migration_reads_) ReleaseStagingPageLocked(lease.page_index);
    migration_reads_.clear();
    read_leases_.clear();
    free_pages_.clear();
  }

  initialized_ = false;
  registrar_ = nullptr;
}

// ---------------------------------------------------------------------------
//  Staging arena
// ---------------------------------------------------------------------------

uint32_t SsdBackend::AcquireStagingPageLocked() {
  if (free_pages_.empty()) return kNoPage;
  const uint32_t page = free_pages_.back();
  free_pages_.pop_back();
  return page;
}

void SsdBackend::ReleaseStagingPageLocked(uint32_t page_index) {
  if (page_index == kNoPage) return;
  free_pages_.push_back(page_index);
}

void* SsdBackend::StagingPagePtr(uint32_t page_index) const {
  if (staging_base_ == nullptr) return nullptr;
  return static_cast<char*>(staging_base_) + static_cast<uint64_t>(page_index) * cfg_.page_size;
}

// ---------------------------------------------------------------------------
//  Control plane
// ---------------------------------------------------------------------------

TierCapacity SsdBackend::Capacity() const {
  const auto [used, total] = ssd_->Capacity();
  TierCapacity cap;
  cap.total_bytes = static_cast<uint64_t>(total);
  cap.available_bytes = total > used ? static_cast<uint64_t>(total - used) : 0;
  cap.max_allocatable_bytes = std::min<uint64_t>(cap.available_bytes, cfg_.page_size);
  return cap;
}

uint64_t SsdBackend::OwnedKeyCount() const {
  return static_cast<uint64_t>(ssd_->SnapshotOwnedKeys().size());
}

std::vector<KvEvent> SsdBackend::DrainPendingEvents() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    unshipped_events_ = 0;
  }
  return ssd_->DrainPendingEvents();
}

std::vector<KvEvent> SsdBackend::SnapshotOwnedKeys() const { return ssd_->SnapshotOwnedKeys(); }

std::vector<KvEvent> SsdBackend::SnapshotOwnedKeysForFullSync() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    unshipped_events_ = 0;
  }
  return ssd_->SnapshotOwnedKeysForFullSync();
}

void SsdBackend::ClearLocal() {
  std::vector<std::string> migration_keys;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Invalidate outstanding slots so a late Commit fails rather than
    // resurrecting a key the cluster believes is gone.
    for (auto& [slot_id, slot] : pending_) ReleaseStagingPageLocked(slot.page_index);
    pending_.clear();
    for (auto& [key, lease] : read_leases_) ReleaseStagingPageLocked(lease.page_index);
    read_leases_.clear();
    for (auto& [key, lease] : migration_reads_) {
      ReleaseStagingPageLocked(lease.page_index);
      migration_keys.push_back(key);
    }
    migration_reads_.clear();
    unshipped_events_ = 0;
  }
  for (const auto& key : migration_keys) ssd_->UnpinForMigration(key);
  clear_full_sync_pending_.store(true, std::memory_order_release);
  ssd_->ClearLocal();
}

void SsdBackend::ClearFullSyncAcked() {
  clear_full_sync_pending_.store(false, std::memory_order_release);
}

void SsdBackend::SetAutoFlushHook(size_t threshold, std::function<void()> cb) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto_flush_threshold_ = threshold == 0 ? 1 : threshold;
  auto_flush_cb_ = std::move(cb);
}

// Counted here rather than read out of PeerSsdManager: its outbox is private
// and the hook only needs to know "enough has happened to be worth a flush".
void SsdBackend::NoteEventQueuedLocked() {
  ++unshipped_events_;
  if (auto_flush_cb_ && unshipped_events_ >= auto_flush_threshold_) {
    unshipped_events_ = 0;
    auto_flush_cb_();
  }
}

// ---------------------------------------------------------------------------
//  Slot lifecycle
// ---------------------------------------------------------------------------

std::vector<AllocateResult> SsdBackend::BatchAllocate(const std::vector<AllocateRequest>& entries) {
  std::vector<AllocateResult> results(entries.size());
  if (clear_full_sync_pending_.load(std::memory_order_acquire)) {
    // Gated until master has acked the empty full sync, same rule as
    // PageBackend: no new owned key may appear in that window.
    return results;
  }

  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);

  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& req = entries[i];
    AllocateResult& out = results[i];

    if (req.size == 0 || req.size > cfg_.page_size) {
      // One key, one page — see the header.  Not kFailedNoSpace: no other peer
      // would do better, so the writer must not keep retrying elsewhere.
      MORI_UMBP_WARN("[SsdBackend] key '{}' size {} exceeds page_size {}", req.key, req.size,
                     cfg_.page_size);
      out.outcome = AllocateOutcome::kFailed;
      continue;
    }
    if (ssd_->Exists(req.key)) {
      out.outcome = AllocateOutcome::kSuccessAlreadyExists;
      continue;
    }

    const uint32_t page = AcquireStagingPageLocked();
    if (page == kNoPage) {
      // The arena is a real capacity limit, so this IS retry-elsewhere-worthy.
      out.outcome = AllocateOutcome::kFailedNoSpace;
      continue;
    }

    const uint64_t slot_id = next_slot_id_.fetch_add(1, std::memory_order_relaxed);
    pending_.emplace(slot_id,
                     PendingSlot{slot_id, req.key, page, req.size, now + cfg_.pending_ttl});

    out.outcome = AllocateOutcome::kSuccessAllocated;
    out.slot_id = slot_id;
    out.pages = {PageLocation{kStagingBufferIndex, page}};
    out.size = req.size;
    out.page_size = cfg_.page_size;
    out.pending_ttl_ms = static_cast<uint64_t>(cfg_.pending_ttl.count());
    if (!buffer_desc_.empty()) {
      out.descs = {BufferMemoryDescBytes{kStagingBufferIndex, buffer_desc_}};
    }
  }
  return results;
}

std::vector<CommitResult> SsdBackend::BatchCommit(const std::vector<CommitRequest>& entries) {
  std::vector<CommitResult> results(entries.size());

  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& req = entries[i];

    // Take the slot out under the lock, then do the SSD write OUTSIDE it: a
    // spill is real IO and holding the arena lock across it would stall every
    // concurrent Allocate and Resolve.
    PendingSlot slot;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = pending_.find(req.slot_id);
      if (it == pending_.end()) {
        results[i].success = false;  // reaped, aborted, cleared, or never existed
        continue;
      }
      slot = it->second;
      pending_.erase(it);
    }

    const void* src = StagingPagePtr(slot.page_index);
    const bool ok = src != nullptr && ssd_->Write(req.key, {{src, slot.size}}, slot.size);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      // The staging page is borrowed for the write only; the key's home is the
      // SSD.  Returned on success AND failure.
      ReleaseStagingPageLocked(slot.page_index);
      if (ok) NoteEventQueuedLocked();
    }

    if (!ok) {
      MORI_UMBP_ERROR("[SsdBackend] Commit: SSD write failed for key '{}' ({} bytes)", req.key,
                      slot.size);
      results[i].success = false;
      continue;
    }
    results[i].success = true;
    results[i].bytes_committed = slot.size;
  }
  return results;
}

std::vector<bool> SsdBackend::BatchAbort(const std::vector<uint64_t>& slot_ids) {
  std::vector<bool> results(slot_ids.size(), true);
  std::lock_guard<std::mutex> lock(mutex_);
  for (uint64_t slot_id : slot_ids) {
    auto it = pending_.find(slot_id);
    if (it == pending_.end()) continue;  // idempotent: already reaped or unknown
    ReleaseStagingPageLocked(it->second.page_index);
    pending_.erase(it);
  }
  return results;
}

std::vector<ResolvedEntry> SsdBackend::BatchResolve(const std::vector<std::string>& keys,
                                                    bool include_descs) {
  std::vector<ResolvedEntry> results(keys.size());
  const auto now = std::chrono::steady_clock::now();

  for (size_t i = 0; i < keys.size(); ++i) {
    const std::string& key = keys[i];
    ResolvedEntry& out = results[i];

    // An existing lease means the bytes are already staged: extend it and reuse
    // the page rather than re-reading the SSD for a concurrent second reader.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = read_leases_.find(key);
      if (it != read_leases_.end()) {
        it->second.expires_at = now + cfg_.read_lease_ttl;
        out.found = true;
        out.pages = {PageLocation{kStagingBufferIndex, it->second.page_index}};
        out.size = it->second.size;
        out.page_size = cfg_.page_size;
        if (include_descs && !buffer_desc_.empty()) {
          out.descs = {BufferMemoryDescBytes{kStagingBufferIndex, buffer_desc_}};
        }
        continue;
      }
    }

    const uint64_t size = ssd_->SizeOf(key);
    if (size == 0 || size > cfg_.page_size) continue;  // found = false

    uint32_t page;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      page = AcquireStagingPageLocked();
    }
    if (page == kNoPage) {
      // Staging exhausted.  Reported as a miss, which is the wrong shape for
      // "this node has it, come back" — see the header's note on the rejected
      // retry state.  Sizing the arena for read concurrency is the mitigation.
      MORI_UMBP_WARN("[SsdBackend] Resolve: staging arena exhausted, key '{}' degraded to miss",
                     key);
      continue;
    }

    // The SSD read runs outside the lock: PeerSsdManager marks the key
    // in-flight for the window, so eviction already skips it.
    const SsdReadOutcome outcome = ssd_->PrepareRead(key, StagingPagePtr(page), cfg_.page_size);
    if (outcome.status != SsdReadStatus::kOk) {
      std::lock_guard<std::mutex> lock(mutex_);
      ReleaseStagingPageLocked(page);
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      // A concurrent Resolve of the same key may have staged it while this read
      // was in flight; keep the winner and give this page back.
      auto [it, inserted] =
          read_leases_.emplace(key, ReadLease{page, outcome.size, now + cfg_.read_lease_ttl});
      if (!inserted) {
        ReleaseStagingPageLocked(page);
        it->second.expires_at = now + cfg_.read_lease_ttl;
      }
      out.found = true;
      out.pages = {PageLocation{kStagingBufferIndex, it->second.page_index}};
      out.size = it->second.size;
      out.page_size = cfg_.page_size;
      if (include_descs && !buffer_desc_.empty()) {
        out.descs = {BufferMemoryDescBytes{kStagingBufferIndex, buffer_desc_}};
      }
    }
  }
  return results;
}

bool SsdBackend::AcquireMigrationRead(const std::string& key,
                                      ResolvedEntry* resolved) {
  if (resolved == nullptr) return false;
  if (!ssd_->PinForMigration(key)) return false;
  const uint64_t size = ssd_->SizeOf(key);
  if (size == 0 || size > cfg_.page_size) {
    ssd_->UnpinForMigration(key);
    return false;
  }

  uint32_t page = kNoPage;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (migration_reads_.count(key) != 0) {
      ssd_->UnpinForMigration(key);
      return false;
    }
    page = AcquireStagingPageLocked();
    if (page == kNoPage) {
      ssd_->UnpinForMigration(key);
      return false;
    }
    migration_reads_[key] =
        ReadLease{page, size, std::chrono::steady_clock::time_point::max()};
  }
  const SsdReadOutcome outcome =
      ssd_->PrepareRead(key, StagingPagePtr(page), cfg_.page_size);
  if (outcome.status != SsdReadStatus::kOk) {
    ReleaseMigrationRead(key);
    return false;
  }
  resolved->found = true;
  resolved->pages = {PageLocation{kStagingBufferIndex, page}};
  resolved->size = outcome.size;
  resolved->page_size = cfg_.page_size;
  return true;
}

void SsdBackend::ReleaseMigrationRead(const std::string& key) {
  bool released = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = migration_reads_.find(key);
    if (it == migration_reads_.end()) return;
    ReleaseStagingPageLocked(it->second.page_index);
    migration_reads_.erase(it);
    released = true;
  }
  if (released) ssd_->UnpinForMigration(key);
}

bool SsdBackend::Contains(const std::string& key) const {
  return ssd_ != nullptr && ssd_->Exists(key);
}

std::vector<EvictResult> SsdBackend::Evict(const std::vector<std::string>& keys) {
  std::vector<EvictResult> results;
  results.reserve(keys.size());

  for (const std::string& key : keys) {
    EvictResult r;
    r.key = key;

    // A staged read holds the bytes live for its lease; PeerSsdManager::Evict
    // would refuse anyway (inflight_reads_), but reporting 0 here keeps the
    // master's retry behavior explicit rather than incidental.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (read_leases_.count(key) != 0 || migration_reads_.count(key) != 0) {
        results.push_back(r);  // bytes_freed = 0 -> master retries next round
        continue;
      }
    }

    const uint64_t size = ssd_->SizeOf(key);
    if (size != 0 && ssd_->Evict(key)) {
      r.bytes_freed = size;
      std::lock_guard<std::mutex> lock(mutex_);
      NoteEventQueuedLocked();  // PeerSsdManager queued the REMOVE
    }
    results.push_back(r);
  }
  return results;
}

// ---------------------------------------------------------------------------
//  Bootstrap
// ---------------------------------------------------------------------------

std::vector<BufferMemoryDescBytes> SsdBackend::AllBufferDescs() const {
  if (buffer_desc_.empty()) return {};
  return {BufferMemoryDescBytes{kStagingBufferIndex, buffer_desc_}};
}

TransferRef SsdBackend::BufferRef(uint32_t buffer_index) const {
  if (buffer_index != kStagingBufferIndex) return TransferRef{};
  return buffer_ref_;
}

// ---------------------------------------------------------------------------
//  Reaper
// ---------------------------------------------------------------------------

void SsdBackend::StartReaper() {
  bool expected = false;
  if (!reaper_running_.compare_exchange_strong(expected, true)) return;
  reaper_thread_ = std::thread([this] { ReaperLoop(); });
}

void SsdBackend::StopReaper() {
  if (!reaper_running_.exchange(false)) return;
  reaper_cv_.notify_all();
  if (reaper_thread_.joinable()) reaper_thread_.join();
}

void SsdBackend::ReaperLoop() {
  while (reaper_running_.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lock(reaper_cv_mutex_);
      reaper_cv_.wait_for(lock, cfg_.reaper_interval,
                          [this] { return !reaper_running_.load(std::memory_order_acquire); });
    }
    if (!reaper_running_.load(std::memory_order_acquire)) break;
    ReaperSweep();
  }
}

// Returns staging pages whose borrower is gone: a writer that died between
// Allocate and Commit, and a reader whose lease has elapsed.  Both are pure
// arena bookkeeping — no SSD state is touched, because neither case ever put
// anything on the SSD.
void SsdBackend::ReaperSweep() {
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto it = pending_.begin(); it != pending_.end();) {
    if (it->second.deadline <= now) {
      MORI_UMBP_WARN("[SsdBackend] reaping expired slot {} (key '{}')", it->second.slot_id,
                     it->second.key);
      ReleaseStagingPageLocked(it->second.page_index);
      it = pending_.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = read_leases_.begin(); it != read_leases_.end();) {
    if (it->second.expires_at <= now) {
      ReleaseStagingPageLocked(it->second.page_index);
      it = read_leases_.erase(it);
    } else {
      ++it;
    }
  }
}

std::unique_ptr<MediumBackend> MakeSsdBackend(SsdBackend::Config cfg) {
  return std::make_unique<SsdBackend>(std::move(cfg));
}

}  // namespace mori::umbp
