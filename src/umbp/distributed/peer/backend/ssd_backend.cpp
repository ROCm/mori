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
#include <limits>
#include <msgpack.hpp>
#include <stdexcept>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/master/master_metrics.h"

namespace mori::umbp {

namespace {
constexpr uint32_t kStagingBufferIndex = 0;

uint32_t SizeToPages(uint64_t size, uint64_t page_size) {
  if (size == 0 || page_size == 0) return 0;
  const uint64_t pages = 1 + (size - 1) / page_size;
  return pages > UINT32_MAX ? 0 : static_cast<uint32_t>(pages);
}
}  // namespace

SsdBackend::SsdBackend(Config cfg) : cfg_(std::move(cfg)) {
  if (cfg_.page_size == 0) {
    throw std::invalid_argument("SsdBackend: page_size must be > 0");
  }
  if (cfg_.staging_pages == 0) {
    throw std::invalid_argument("SsdBackend: staging_pages must be > 0");
  }
  if (cfg_.page_size > std::numeric_limits<uint64_t>::max() / cfg_.staging_pages ||
      cfg_.page_size > std::numeric_limits<size_t>::max() / cfg_.staging_pages) {
    throw std::invalid_argument("SsdBackend: staging arena size overflows");
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

  // Opt-in switch for the file->GPU (GDS) read path.  Off by default even when
  // the build has hipfile; set UMBP_ENABLE_GDS=1 to route O_DIRECT SSD reads
  // through the GdsEngine instead of the staging arena (no rebuild needed).
  if (const char* env = std::getenv("UMBP_ENABLE_GDS")) {
    const std::string v(env);
    gds_enabled_ = (v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "ON");
  }

  // Extent staging (D1) wherever the tier reads ranges; UMBP_SSD_EXTENT_STAGING=0 disables it.
  extent_staging_ = ssd_->SupportsRangedRead();
  if (const char* env = std::getenv("UMBP_SSD_EXTENT_STAGING")) {
    const std::string v(env);
    if (v == "0" || v == "false" || v == "FALSE" || v == "off" || v == "OFF")
      extent_staging_ = false;
  }

  // ONE buffer covering every staging page, so each page is contiguous and a
  // single registration serves the arena.  Host memory deliberately: this is
  // what the backend publishes, and publishing ordinary registered DRAM is the
  // entire reason SSD needs no transfer-layer change (see the header).
  staging_size_ = static_cast<uint64_t>(cfg_.staging_pages) * cfg_.page_size;
  // Every byte this backend moves crosses the arena twice (device <-> arena,
  // arena <-> wire), so hugepage backing matters here more than for an ordinary
  // buffer.  HostMemAllocator falls back to 4 KiB pages on its own when no
  // hugetlb pages are free, which is what we want: a node must still come up.
  HostPageMemorySource::Options staging_opts;
  staging_opts.use_hugepages = cfg_.staging_use_hugepages;
  staging_opts.hugepage_size = cfg_.staging_hugepage_size;
  staging_source_ = std::make_unique<HostPageMemorySource>(staging_opts);

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

  // A bitmap lets a multi-page object reserve one contiguous run while keeping
  // the transfer-visible page size unchanged.
  const uint32_t usable_pages =
      static_cast<uint32_t>(std::min<uint64_t>(cfg_.staging_pages, staging_size_ / cfg_.page_size));
  staging_page_used_.assign(usable_pages, false);

  // Crash-restart leftover: metadata is gone but files may remain, so used
  // capacity would diverge from owned_.  Must run before any IO.
  ssd_->DiscardLeftoverOnStartup();

  // The file tier deletes a segment once nothing in it is live.  A GDS reader
  // holds a FileRef (fd + offset) with no lease, so keep a deleted segment's
  // fd open as long as a staged read may lag its resolve, and drop the cached
  // handle before the fd closes and its number can be reused.
  if (gds_enabled_) {
    ssd_->SetRetiredFileHook([this](int fd) { DropGdsHandle(fd); }, cfg_.read_lease_ttl);
  }

  initialized_ = true;
  StartReaper();
  MORI_UMBP_INFO(
      "[SsdBackend] Init staging_pages={} page_size={} arena_bytes={} hugepages={} "
      "single_flight={} extent_staging={}",
      usable_pages, cfg_.page_size, staging_size_, cfg_.staging_use_hugepages,
      cfg_.ssd.ssd.single_flight_reads, extent_staging_);
  return true;
}

void SsdBackend::Shutdown() {
  if (!initialized_) return;
  StopReaper();

  if (registrar_ != nullptr && buffer_ref_.Valid()) registrar_->Deregister(buffer_ref_);
  buffer_ref_ = TransferRef{};
  buffer_desc_.clear();

  // Release the GDS file handles obtained lazily during resolves.  Unhook
  // first: the tier outlives this call and must not call back into it.
  if (ssd_) ssd_->SetRetiredFileHook(nullptr, std::chrono::milliseconds{0});
  {
    std::lock_guard<std::mutex> lock(gds_mutex_);
    if (registrar_ != nullptr) {
      for (auto& [fd, handle] : gds_handles_) {
        if (handle != nullptr) registrar_->Deregister(TransferRef::File(fd, 0, 0, handle));
      }
    }
    gds_handles_.clear();
  }

  // Deregister before release, same ordering rule as PageBackend::Shutdown.
  if (staging_source_ != nullptr) staging_source_->Release();
  staging_source_.reset();
  staging_base_ = nullptr;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.clear();
    migration_reads_.clear();
    read_leases_.clear();
    staging_page_used_.clear();
  }

  initialized_ = false;
  registrar_ = nullptr;
}

// ---------------------------------------------------------------------------
//  Staging arena
// ---------------------------------------------------------------------------

SsdBackend::StagingSpan SsdBackend::AcquireStagingSpanLocked(uint32_t page_count) {
  if (page_count == 0 || page_count > staging_page_used_.size()) return {};

  const auto first_fit = [&]() -> StagingSpan {
    uint32_t run_start = 0;
    uint32_t run_length = 0;
    for (uint32_t i = 0; i < staging_page_used_.size(); ++i) {
      if (staging_page_used_[i]) {
        run_length = 0;
        continue;
      }
      if (run_length == 0) run_start = i;
      if (++run_length != page_count) continue;
      for (uint32_t page = run_start; page < run_start + page_count; ++page) {
        staging_page_used_[page] = true;
      }
      // Mirrored for the metrics tick, which must not take mutex_. Correctness
      // still lives entirely in staging_page_used_.
      const uint64_t in_use =
          staging_pages_in_use_.fetch_add(page_count, std::memory_order_relaxed) + page_count;
      if (in_use > staging_pages_peak_.load(std::memory_order_relaxed)) {
        staging_pages_peak_.store(in_use, std::memory_order_relaxed);
      }
      return StagingSpan{run_start, page_count};
    }
    return {};
  };
  StagingSpan span = first_fit();
  // Finished local reads linger until the reaper; take their pages back now.
  if (span.page_count == 0 && ReclaimReadLeasesLocked(std::chrono::steady_clock::now()) > 0) {
    span = first_fit();
  }
  return span;
}

void SsdBackend::ReleaseStagingSpanLocked(StagingSpan span) {
  if (span.page_count == 0 || span.first_page >= staging_page_used_.size()) return;
  const uint32_t end = std::min<uint64_t>(staging_page_used_.size(),
                                          static_cast<uint64_t>(span.first_page) + span.page_count);
  for (uint32_t page = span.first_page; page < end; ++page) {
    staging_page_used_[page] = false;
  }
  staging_pages_in_use_.fetch_sub(end - span.first_page, std::memory_order_relaxed);
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
  const uint64_t staging_capacity =
      static_cast<uint64_t>(staging_page_used_.size()) * cfg_.page_size;
  cap.max_allocatable_bytes = std::min({cap.available_bytes, cap.total_bytes, staging_capacity});
  return cap;
}

uint64_t SsdBackend::OwnedKeyCount() const {
  // Metrics poll this every second; a snapshot would copy every key under mutex_.
  return static_cast<uint64_t>(ssd_->OwnedKeyCount());
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
    for (auto& [slot_id, slot] : pending_) ReleaseStagingSpanLocked(slot.span);
    pending_.clear();
    // A span a local reader still copies out of goes back with its last hold.
    for (auto& [key, lease] : read_leases_)
      extent_stages_.push_back(
          ExtentStage{lease.span, ObjectExtent{0, lease.size}, std::move(lease.hold)});
    read_leases_.clear();
    std::vector<ExtentStage> held;
    for (auto& stage : extent_stages_) {
      if (stage.hold.use_count() > 1) {
        held.push_back(std::move(stage));
      } else {
        ReleaseStagingSpanLocked(stage.span);
      }
    }
    extent_stages_ = std::move(held);
    for (auto& [key, lease] : migration_reads_) {
      ReleaseStagingSpanLocked(lease.span);
      migration_keys.push_back(key);
    }
    migration_reads_.clear();
    unshipped_events_ = 0;
  }
  for (const auto& key : migration_keys) ssd_->UnpinForMigration(key);
  // The SSD wipe below can close and reopen segment fds, so drop the cached GDS
  // handles (they are re-registered lazily on the next resolve) rather than risk
  // a stale handle if an fd number is later reused.
  {
    std::lock_guard<std::mutex> lock(gds_mutex_);
    if (registrar_ != nullptr) {
      for (auto& [fd, handle] : gds_handles_) {
        if (handle != nullptr) registrar_->Deregister(TransferRef::File(fd, 0, 0, handle));
      }
    }
    gds_handles_.clear();
  }
  clear_full_sync_pending_.store(true, std::memory_order_release);
  ssd_->ClearLocal();
}

void SsdBackend::ClearFullSyncAcked() {
  clear_full_sync_pending_.store(false, std::memory_order_release);
}

void SsdBackend::SetEventPublishing(bool enabled) {
  if (ssd_) ssd_->SetEventPublishing(enabled);
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

  const uint64_t ssd_capacity = static_cast<uint64_t>(ssd_->Capacity().second);
  const uint64_t staging_capacity =
      static_cast<uint64_t>(staging_page_used_.size()) * cfg_.page_size;
  const auto fits = [&](uint64_t size) {
    const uint32_t pages = SizeToPages(size, cfg_.page_size);
    return pages != 0 && size <= ssd_capacity && pages <= staging_page_used_.size();
  };

  // A local file engine writes straight into a reserved file range, so it
  // needs no staging; the reservation's disk write runs outside mutex_.
  std::vector<std::optional<RecordReservation>> records(entries.size());
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& req = entries[i];
    if (req.allow_file_ref && fits(req.size) && !ssd_->Exists(req.key)) {
      records[i] = ssd_->Reserve(req.key, req.size);
    }
  }

  std::vector<size_t> unused_records;
  const auto now = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lock(mutex_);
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& req = entries[i];
    AllocateResult& out = results[i];

    const uint32_t page_count = SizeToPages(req.size, cfg_.page_size);
    if (!fits(req.size)) {
      // The object can never fit this backend's staging arena. This is a shape
      // failure, not transient arena pressure, so another identical peer would
      // not do better.
      MORI_UMBP_WARN("[SsdBackend] key '{}' size {} exceeds permanent limit {}", req.key, req.size,
                     std::min(ssd_capacity, staging_capacity));
      out.outcome = AllocateOutcome::kFailed;
      continue;
    }
    if (ssd_->Exists(req.key)) {
      if (records[i]) unused_records.push_back(i);
      out.outcome = AllocateOutcome::kSuccessAlreadyExists;
      continue;
    }
    if (records[i]) {
      const RecordLocation& loc = records[i]->location;
      const uint64_t slot_id = next_slot_id_.fetch_add(1, std::memory_order_relaxed);
      PendingSlot slot{slot_id, req.key, StagingSpan{}, req.size, now + cfg_.pending_ttl};
      slot.record = records[i];
      pending_.emplace(slot_id, std::move(slot));
      out.outcome = AllocateOutcome::kSuccessAllocated;
      out.slot_id = slot_id;
      out.size = req.size;
      out.page_size = cfg_.page_size;
      out.pending_ttl_ms = static_cast<uint64_t>(cfg_.pending_ttl.count());
      out.file_ref = TransferRef::File(loc.fd, loc.value_offset, loc.readable_size);
      continue;
    }

    const StagingSpan span = AcquireStagingSpanLocked(page_count);
    if (span.page_count == 0) {
      // The arena is a real capacity limit, so this IS retry-elsewhere-worthy.
      out.outcome = AllocateOutcome::kFailedNoSpace;
      continue;
    }

    const uint64_t slot_id = next_slot_id_.fetch_add(1, std::memory_order_relaxed);
    pending_.emplace(slot_id,
                     PendingSlot{slot_id, req.key, span, req.size, now + cfg_.pending_ttl});

    out.outcome = AllocateOutcome::kSuccessAllocated;
    out.slot_id = slot_id;
    out.pages.reserve(span.page_count);
    for (uint32_t page = span.first_page; page < span.first_page + span.page_count; ++page) {
      out.pages.push_back(PageLocation{kStagingBufferIndex, page});
    }
    out.size = req.size;
    out.page_size = cfg_.page_size;
    out.pending_ttl_ms = static_cast<uint64_t>(cfg_.pending_ttl.count());
    if (!buffer_desc_.empty()) {
      out.descs = {BufferMemoryDescBytes{kStagingBufferIndex, buffer_desc_}};
    }
  }
  lock.unlock();
  for (size_t i : unused_records) ssd_->AbortReserved(entries[i].key, *records[i]);
  return results;
}

std::vector<CommitResult> SsdBackend::BatchCommit(const std::vector<CommitRequest>& entries) {
  std::vector<CommitResult> results(entries.size());
  if (entries.empty()) return results;

  // Phase 1 (ONE lock): take every slot out of pending_.  An entry whose slot is
  // gone — reaped, aborted, cleared, or never there — fails here and takes no
  // further part.
  std::vector<PendingSlot> slots(entries.size());
  std::vector<size_t> todo;
  todo.reserve(entries.size());
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < entries.size(); ++i) {
      auto it = pending_.find(entries[i].slot_id);
      if (it == pending_.end()) continue;  // results[i].success stays false
      slots[i] = it->second;
      pending_.erase(it);
      todo.push_back(i);
    }
  }
  if (todo.empty()) return results;

  // Phase 2 (no lock): ONE batched SSD write.  A per-key Write loop would idle
  // every drive but one of a multi-drive tier, so keeping the batch intact all
  // the way down to the backend is the whole point of this path.  An entry whose
  // staging page cannot be resolved is dropped from the batch and fails.
  std::vector<std::string> keys;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  std::vector<size_t> batched;  // indices into `entries`, parallel to the above
  std::vector<size_t> written;  // reserved records, already on the drive
  keys.reserve(todo.size());
  srcs.reserve(todo.size());
  sizes.reserve(todo.size());
  batched.reserve(todo.size());
  for (size_t i : todo) {
    if (slots[i].record) {
      if (ssd_->CommitReserved(entries[i].key, *slots[i].record)) written.push_back(i);
      continue;
    }
    const void* src = StagingPagePtr(slots[i].span.first_page);
    if (src == nullptr) continue;
    keys.push_back(entries[i].key);
    srcs.push_back(src);
    sizes.push_back(slots[i].size);
    batched.push_back(i);
  }

  std::vector<bool> ok;
  if (!batched.empty()) {
    ok = ssd_->WriteBatch(keys, srcs, sizes);
    ok.resize(batched.size(), false);
  }

  // Phase 3 (ONE lock): return every borrowed staging page — on success AND on
  // failure, since the page was only ever borrowed for the write — and queue one
  // event per key that landed.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i : todo) ReleaseStagingSpanLocked(slots[i].span);
    for (size_t j = 0; j < batched.size(); ++j) {
      if (ok[j]) NoteEventQueuedLocked();
    }
    for (size_t k = 0; k < written.size(); ++k) NoteEventQueuedLocked();
  }
  for (size_t i : written) {
    results[i].success = true;
    results[i].bytes_committed = slots[i].size;
  }

  for (size_t j = 0; j < batched.size(); ++j) {
    const size_t i = batched[j];
    if (!ok[j]) {
      MORI_UMBP_ERROR("[SsdBackend] Commit: SSD write failed for key '{}' ({} bytes)",
                      entries[i].key, slots[i].size);
      continue;
    }
    results[i].success = true;
    results[i].bytes_committed = slots[i].size;
  }
  return results;
}

std::vector<bool> SsdBackend::BatchAbort(const std::vector<uint64_t>& slot_ids) {
  std::vector<bool> results(slot_ids.size(), true);
  std::vector<PendingSlot> reserved;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (uint64_t slot_id : slot_ids) {
      auto it = pending_.find(slot_id);
      if (it == pending_.end()) continue;  // idempotent: already reaped or unknown
      ReleaseStagingSpanLocked(it->second.span);
      if (it->second.record) reserved.push_back(std::move(it->second));
      pending_.erase(it);
    }
  }
  for (const auto& slot : reserved) ssd_->AbortReserved(slot.key, *slot.record);
  return results;
}

std::vector<ResolvedEntry> SsdBackend::BatchResolve(const std::vector<std::string>& keys,
                                                    bool include_descs, bool allow_file_refs,
                                                    ResolveReader reader,
                                                    const std::vector<ObjectExtent>* extents) {
  const bool local = reader == ResolveReader::kLocal;
  std::vector<ResolvedEntry> results(keys.size());
  if (keys.empty()) return results;
  const auto now = std::chrono::steady_clock::now();
  // Stage only the bytes an in-process reader names (design-ssd-ranged-io.md D1).
  // A remote reader's extent does not travel on the wire (D2), so it stages whole.
  const bool use_extents =
      extent_staging_ && extents != nullptr && extents->size() == keys.size() && local;

  // Fills `out` from a lease the caller has already located.  Caller holds
  // mutex_.  Factored out because the lease-hit and read-completed paths publish
  // byte-identical results and drifting between them would be a silent bug.
  // `window` is the object bytes the span holds when it is not all of them.
  auto publish_span = [&](ResolvedEntry& out, StagingSpan span, uint64_t size,
                          const std::shared_ptr<const void>& hold, ObjectExtent window) {
    if (local) out.hold = hold;
    out.outcome = ResolveOutcome::kFound;
    out.found = true;
    out.pages.reserve(span.page_count);
    for (uint32_t page = span.first_page; page < span.first_page + span.page_count; ++page) {
      out.pages.push_back(PageLocation{kStagingBufferIndex, page});
    }
    out.size = size;
    out.page_size = cfg_.page_size;
    out.covered_offset = window.offset;
    out.covered_size = window.length;
    if (include_descs && !buffer_desc_.empty()) {
      out.descs = {BufferMemoryDescBytes{kStagingBufferIndex, buffer_desc_}};
    }
  };
  auto publish = [&](ResolvedEntry& out, const ReadLease& lease) {
    publish_span(out, lease.span, lease.size, lease.hold, ObjectExtent{});
  };

  // Phase 1 (ONE lock): serve every key that is already staged, size every
  // unique remaining key, then reserve the spans.  A remote reader retries a
  // partial resolve whole, so its leases would pin the subset while retries
  // compete for the rest: it gets every span or, on transient pressure, kBusy
  // with no new reservation kept.  A local reader copies its part first.
  std::vector<size_t> todo;                           // keys needing a device read
  std::vector<uint32_t> page_counts;                  // parallel to todo
  std::vector<ObjectExtent> read_extents;             // parallel to todo; length 0 = whole
  std::vector<StagingSpan> spans;                     // parallel to todo
  std::vector<void*> dsts;                            // parallel to todo
  std::vector<size_t> caps;                           // parallel to todo
  std::vector<std::string> read_keys;                 // parallel to todo
  std::vector<std::vector<size_t>> result_groups;     // duplicate input indices per device read
  std::unordered_map<std::string, size_t> scheduled;  // key (+ window) -> its todo slot
  todo.reserve(keys.size());
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
      auto it = read_leases_.find(keys[i]);
      if (it != read_leases_.end()) {  // a whole-object lease serves any extent too
        if (!local) it->second.expires_at = now + cfg_.read_lease_ttl;
        publish(results[i], it->second);
        continue;
      }

      // Zero-copy GDS (UMBP_ENABLE_GDS): when the caller can read a file ref,
      // the switch is on, a file engine is present, and the record is on an
      // O_DIRECT fd, publish a FileRef and skip the staging arena — the reader
      // (GdsEngine) DMAs the range into device memory.
      //
      // allow_file_refs is tested FIRST on purpose.  A caller that cannot use a
      // file ref — a peer serving the wire, or an existence check reading only
      // `found` — must fall through to the staging path and get real pages, and
      // testing it first also spares it LocateRecord and the hipFile
      // registration GdsHandleForFd performs as a side effect.
      std::optional<RecordLocation> loc;
      if (allow_file_refs && gds_enabled_ && (loc = ssd_->LocateRecord(keys[i])) &&
          loc->direct_io && loc->fd >= 0) {
        if (void* handle = GdsHandleForFd(loc->fd)) {
          results[i].outcome = ResolveOutcome::kFound;
          results[i].found = true;
          results[i].size = loc->value_size;
          results[i].page_size = cfg_.page_size;
          results[i].file_ref =
              TransferRef::File(loc->fd, loc->value_offset, loc->readable_size, handle);
          continue;
        }
      }

      const uint64_t size = ssd_->SizeOf(keys[i]);
      if (size == 0) continue;

      // The extent widened to the 4 KiB record alignment (so O_DIRECT lands in
      // the page-aligned span) and clamped to the object.  One outside the
      // object, or one that widens to all of it, is a whole read.
      ObjectExtent window;
      if (use_extents && (*extents)[i].length != 0) {
        const ObjectExtent& want = (*extents)[i];
        if (want.offset < size && want.length <= size - want.offset) {
          const uint64_t lo = want.offset & ~(kExtentAlign - 1);
          const uint64_t hi = std::min<uint64_t>(
              size, (want.offset + want.length + kExtentAlign - 1) & ~(kExtentAlign - 1));
          if (hi - lo < size) window = ObjectExtent{lo, hi - lo};
        }
      }
      // Duplicates share a read only when they want the same bytes.
      std::string schedule_key = keys[i];
      if (window.length != 0) {
        schedule_key += '\0' + std::to_string(window.offset) + ':' + std::to_string(window.length);
      }
      auto duplicate = scheduled.find(schedule_key);
      if (duplicate != scheduled.end()) {
        result_groups[duplicate->second].push_back(i);
        continue;
      }

      const uint64_t read_size = window.length != 0 ? window.length : size;
      const uint32_t page_count = SizeToPages(read_size, cfg_.page_size);
      if (page_count == 0 || page_count > staging_page_used_.size()) {
        results[i].outcome = ResolveOutcome::kFailed;
        MORI_UMBP_ERROR("[SsdBackend] Resolve: key '{}' size {} cannot fit staging arena", keys[i],
                        read_size);
        continue;
      }
      todo.push_back(i);
      page_counts.push_back(page_count);
      read_extents.push_back(window);
      read_keys.push_back(keys[i]);
      scheduled.emplace(std::move(schedule_key), todo.size() - 1);
      result_groups.push_back({i});
    }

    uint64_t requested_pages = 0;
    for (uint32_t page_count : page_counts) requested_pages += page_count;
    if (local) {
      // Up to half the arena, leaving the rest to concurrent readers; the first
      // key goes past that so every call progresses.  The rest is kBusy.
      const uint64_t budget = std::max<size_t>(staging_page_used_.size() / 2, 1);
      uint64_t taken = 0;
      size_t staged = 0;
      for (; staged < todo.size(); ++staged) {
        if (staged > 0 && taken + page_counts[staged] > budget) break;
        const StagingSpan span = AcquireStagingSpanLocked(page_counts[staged]);
        if (span.page_count == 0) {
          slot_full_rejects_.fetch_add(1, std::memory_order_relaxed);
          break;
        }
        spans.push_back(span);
        taken += page_counts[staged];
      }
      for (size_t j = staged; j < todo.size(); ++j) {
        for (size_t index : result_groups[j]) results[index].outcome = ResolveOutcome::kBusy;
      }
      todo.resize(staged);
      read_extents.resize(staged);
      read_keys.resize(staged);
      result_groups.resize(staged);
    } else {
      if (requested_pages > staging_page_used_.size()) {
        for (const auto& group : result_groups) {
          for (size_t index : group) results[index].outcome = ResolveOutcome::kFailed;
        }
        MORI_UMBP_ERROR(
            "[SsdBackend] Resolve: batch requires {} staging pages but arena has {}; not retryable",
            requested_pages, staging_page_used_.size());
        return results;
      }
      spans.reserve(todo.size());
      for (size_t j = 0; j < todo.size(); ++j) {
        const StagingSpan span = AcquireStagingSpanLocked(page_counts[j]);
        if (span.page_count == 0) {
          for (const StagingSpan reserved : spans) ReleaseStagingSpanLocked(reserved);
          for (const auto& group : result_groups) {
            for (size_t index : group) results[index].outcome = ResolveOutcome::kBusy;
          }
          slot_full_rejects_.fetch_add(1, std::memory_order_relaxed);
          MORI_UMBP_WARN(
              "[SsdBackend] Resolve: staging arena busy; rolled back {} reservations for {} keys",
              spans.size(), todo.size());
          return results;
        }
        spans.push_back(span);
      }
    }
    for (const StagingSpan span : spans) {
      dsts.push_back(StagingPagePtr(span.first_page));
      caps.push_back(static_cast<size_t>(span.page_count) * cfg_.page_size);
    }
  }
  if (todo.empty()) return results;

  // Phase 2 (no lock): ONE batched SSD read.  ShardedSsdTier turns this into
  // concurrent IO on every drive holding part of the batch.  Duplicate keys in
  // this call were collapsed above; PeerSsdManager additionally coalesces reads
  // racing across calls.  It marks each key in-flight for the window, so
  // eviction already skips them.
  const std::vector<SsdReadOutcome> outcomes =
      ssd_->PrepareReadBatch(read_keys, dsts, caps, &read_extents);

  // Phase 3 (ONE lock): install a lease per key that landed, hand back the pages
  // of the ones that did not.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t j = 0; j < todo.size(); ++j) {
      const size_t primary = todo[j];
      if (outcomes[j].status != SsdReadStatus::kOk) {
        ReleaseStagingSpanLocked(spans[j]);
        const ResolveOutcome outcome = outcomes[j].status == SsdReadStatus::kNotFound
                                           ? ResolveOutcome::kMissing
                                           : ResolveOutcome::kFailed;
        for (size_t result_index : result_groups[j]) {
          results[result_index].outcome = outcome;
        }
        continue;
      }
      if (read_extents[j].length != 0) {
        // Private to its readers and freed with their holds; never a read lease,
        // which promises the WHOLE object to the next resolve of the key.
        extent_stages_.push_back(ExtentStage{spans[j], read_extents[j], std::make_shared<char>()});
        for (size_t result_index : result_groups[j]) {
          publish_span(results[result_index], spans[j], outcomes[j].size,
                       extent_stages_.back().hold, read_extents[j]);
        }
        extent_stages_total_.fetch_add(1, std::memory_order_relaxed);
        extent_stage_bytes_.fetch_add(read_extents[j].length, std::memory_order_relaxed);
        continue;
      }
      // A concurrent Resolve of the same key may have staged it while this read
      // was in flight; keep the winner and give this page back.  A local
      // reader's lease starts expired: only its hold keeps the page.
      const auto lease_deadline = local ? std::chrono::steady_clock::time_point{}
                                        : std::chrono::steady_clock::now() + cfg_.read_lease_ttl;
      auto [it, inserted] = read_leases_.emplace(
          keys[primary],
          ReadLease{spans[j], outcomes[j].size, lease_deadline, std::make_shared<char>()});
      if (!inserted) {
        ReleaseStagingSpanLocked(spans[j]);
        if (!local) it->second.expires_at = lease_deadline;
      }
      for (size_t result_index : result_groups[j]) {
        publish(results[result_index], it->second);
      }
    }
  }
  return results;
}

bool SsdBackend::AcquireMigrationRead(const std::string& key, ResolvedEntry* resolved,
                                      bool allow_file_ref) {
  if (resolved == nullptr) return false;
  if (!ssd_->PinForMigration(key)) return false;
  const uint64_t size = ssd_->SizeOf(key);

  // The caller reads the record in place: a pin, no staging.
  const auto loc = allow_file_ref ? ssd_->LocateRecord(key) : std::nullopt;
  if (loc) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!migration_reads_
               .emplace(key, ReadLease{StagingSpan{}, size,
                                       std::chrono::steady_clock::time_point::max()})
               .second) {
        ssd_->UnpinForMigration(key);
        return false;
      }
    }
    resolved->outcome = ResolveOutcome::kFound;
    resolved->found = true;
    resolved->size = loc->value_size;
    resolved->page_size = cfg_.page_size;
    resolved->file_ref = TransferRef::File(loc->fd, loc->value_offset, loc->readable_size);
    migrations_in_place_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  const uint32_t page_count = SizeToPages(size, cfg_.page_size);
  if (page_count == 0 || page_count > staging_page_used_.size()) {
    ssd_->UnpinForMigration(key);
    return false;
  }

  StagingSpan span;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (migration_reads_.count(key) != 0) {
      ssd_->UnpinForMigration(key);
      return false;
    }
    span = AcquireStagingSpanLocked(page_count);
    if (span.page_count == 0) {
      ssd_->UnpinForMigration(key);
      return false;
    }
    migration_reads_[key] = ReadLease{span, size, std::chrono::steady_clock::time_point::max()};
  }
  const SsdReadOutcome outcome = ssd_->PrepareRead(
      key, StagingPagePtr(span.first_page), static_cast<size_t>(span.page_count) * cfg_.page_size);
  if (outcome.status != SsdReadStatus::kOk) {
    ReleaseMigrationRead(key);
    return false;
  }
  resolved->outcome = ResolveOutcome::kFound;
  resolved->found = true;
  resolved->pages.reserve(span.page_count);
  for (uint32_t page = span.first_page; page < span.first_page + span.page_count; ++page) {
    resolved->pages.push_back(PageLocation{kStagingBufferIndex, page});
  }
  resolved->size = outcome.size;
  resolved->page_size = cfg_.page_size;
  migrations_staged_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void SsdBackend::ReleaseMigrationRead(const std::string& key) {
  bool released = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = migration_reads_.find(key);
    if (it == migration_reads_.end()) return;
    ReleaseStagingSpanLocked(it->second.span);
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

bool SsdBackend::OwnsEvictionOrder() const { return ssd_ != nullptr && ssd_->OrdersEviction(); }

std::vector<EvictionOffer> SsdBackend::EvictionCandidates(size_t max_count) {
  if (ssd_ == nullptr || max_count == 0) return {};
  auto candidates = ssd_->EvictionCandidates(max_count);
  std::lock_guard<std::mutex> lock(mutex_);
  if (read_leases_.empty() && migration_reads_.empty()) return candidates;
  candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                  [&](const EvictionOffer& offer) {
                                    return read_leases_.count(offer.key) != 0 ||
                                           migration_reads_.count(offer.key) != 0;
                                  }),
                   candidates.end());
  return candidates;
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

void* SsdBackend::GdsHandleForFd(int fd) {
  std::lock_guard<std::mutex> lock(gds_mutex_);
  auto it = gds_handles_.find(fd);
  if (it != gds_handles_.end()) return it->second;  // cached, possibly nullptr
  // A zero-length RegisterFile just obtains (and ref-counts) the fd's handle;
  // the per-key ranges are built by BatchResolve.  An invalid ref means no file
  // engine is configured — cache nullptr so the fd is not probed again.
  void* handle = nullptr;
  if (registrar_ != nullptr) {
    TransferRef ref = registrar_->RegisterFile(fd, 0, 0);
    if (ref.IsFile()) handle = ref.gds_handle;
  }
  gds_handles_[fd] = handle;
  return handle;
}

void SsdBackend::DropGdsHandle(int fd) {
  std::lock_guard<std::mutex> lock(gds_mutex_);
  auto it = gds_handles_.find(fd);
  if (it == gds_handles_.end()) return;
  if (it->second != nullptr && registrar_ != nullptr) {
    registrar_->Deregister(TransferRef::File(fd, 0, 0, it->second));
  }
  gds_handles_.erase(it);
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

// Returns what a gone borrower held: a writer that died between Allocate and
// Commit (its staging span, or its reserved record), and a reader whose lease
// has elapsed and whose holds are gone.
void SsdBackend::ReaperSweep() {
  const auto now = std::chrono::steady_clock::now();
  std::vector<PendingSlot> expired_records;
  std::unique_lock<std::mutex> lock(mutex_);

  for (auto it = pending_.begin(); it != pending_.end();) {
    if (it->second.deadline <= now) {
      MORI_UMBP_WARN("[SsdBackend] reaping expired slot {} (key '{}')", it->second.slot_id,
                     it->second.key);
      ReleaseStagingSpanLocked(it->second.span);
      if (it->second.record) {
        expired_records.push_back(std::move(it->second));
      } else {
        staging_expired_reclaims_.fetch_add(1, std::memory_order_relaxed);
      }
      it = pending_.erase(it);
    } else {
      ++it;
    }
  }

  ReclaimReadLeasesLocked(now);
  lock.unlock();
  for (const auto& slot : expired_records) ssd_->AbortReserved(slot.key, *slot.record);
}

size_t SsdBackend::ReclaimReadLeasesLocked(std::chrono::steady_clock::time_point now) {
  size_t reclaimed = 0;
  for (auto it = read_leases_.begin(); it != read_leases_.end();) {
    // use_count is exact here: holds are only copied under mutex_.
    if (it->second.expires_at <= now && it->second.hold.use_count() <= 1) {
      ReleaseStagingSpanLocked(it->second.span);
      it = read_leases_.erase(it);
      ++reclaimed;
    } else {
      ++it;
    }
  }
  for (size_t k = 0; k < extent_stages_.size();) {
    if (extent_stages_[k].hold.use_count() <= 1) {
      ReleaseStagingSpanLocked(extent_stages_[k].span);
      extent_stages_[k] = std::move(extent_stages_.back());
      extent_stages_.pop_back();
      ++reclaimed;
    } else {
      ++k;
    }
  }
  staging_expired_reclaims_.fetch_add(reclaimed, std::memory_order_relaxed);
  return reclaimed;
}

std::unique_ptr<MediumBackend> MakeSsdBackend(SsdBackend::Config cfg) {
  return std::make_unique<SsdBackend>(std::move(cfg));
}

std::vector<MetricSample> SsdBackend::SampleMetrics() const {
  // Everything a caller can see from OUTSIDE this class — resolve hits and
  // misses, bytes handed out, eviction volume, per-call latency — is already
  // measured by InstrumentedBackend against the MediumBackend interface, and is
  // NOT repeated here.  What is left is exactly what the interface hides: what
  // the drive itself did, and how full the staging arena is.
  //
  // These go out under the generic MORI_UMBP_METRIC_BACKEND_MEDIUM_* names with
  // the specifics in an event= / state= label, which is what lets one dashboard
  // panel render this medium beside every other one.  tier= and backend= are
  // added by the publisher.
  std::vector<MetricSample> out;
  if (ssd_ == nullptr) return out;

  auto event = [&out](const char* name, uint64_t v) {
    if (v == 0) return;
    out.push_back(MetricSample{MORI_UMBP_METRIC_BACKEND_MEDIUM_EVENTS_TOTAL,
                               MORI_UMBP_METRIC_BACKEND_MEDIUM_EVENTS_TOTAL_HELP,
                               {{"event", name}},
                               v});
  };
  auto bytes = [&out](const char* name, uint64_t v) {
    if (v == 0) return;
    out.push_back(MetricSample{MORI_UMBP_METRIC_BACKEND_MEDIUM_BYTES_TOTAL,
                               MORI_UMBP_METRIC_BACKEND_MEDIUM_BYTES_TOTAL_HELP,
                               {{"event", name}},
                               v});
  };
  auto state = [&out](const char* name, uint64_t v) {
    out.push_back(MetricSample{MORI_UMBP_METRIC_BACKEND_MEDIUM_STATE,
                               MORI_UMBP_METRIC_BACKEND_MEDIUM_STATE_HELP,
                               {{"state", name}},
                               v,
                               MetricKind::kGauge});
  };

  // Drive-level read outcomes.  These differ from the resolve counts the
  // decorator produces: a resolve that a single-flight merge served never
  // reached the drive, and a size_too_large never became a miss upstream.
  event("device_read_ok", ssd_->ReadOk());
  event("device_read_not_found", ssd_->ReadNotFound());
  event("device_read_size_too_large", ssd_->ReadSizeTooLarge());
  event("device_read_error", ssd_->ReadError());

  // Single-flight: dup is the opportunity, merged is what was actually taken.
  // dup - merged flatlining at dup means coalescing is off or the merge window
  // is being missed, which is invisible from the read outcomes alone.
  event("single_flight_lead", ssd_->ReadLead());
  event("single_flight_dup", ssd_->ReadDup());
  event("single_flight_merged", ssd_->ReadMerged());

  // Evictions are counted by the decorator and the pool; only a delete the
  // drive refused is counted here.
  event("eviction_backend_failed", ssd_->EvictionBackendFailures());

  // Staging pressure.  slot_full_reject is the one to watch: it counts resolves
  // that reported a miss for a key this node HOLDS, purely because the arena
  // was full — indistinguishable from a real miss anywhere else.
  event("staging_slot_full_reject", slot_full_rejects_.load(std::memory_order_relaxed));
  event("staging_expired_reclaim", staging_expired_reclaims_.load(std::memory_order_relaxed));
  // Reads that staged only an extent (D1), and their bytes.
  event("extent_stage", extent_stages_total_.load(std::memory_order_relaxed));
  bytes("extent_stage", extent_stage_bytes_.load(std::memory_order_relaxed));

  // Migrations: in place (file <-> DRAM through the file engine) or staged.
  event("migration_in_place", migrations_in_place_.load(std::memory_order_relaxed));
  event("migration_staged", migrations_staged_.load(std::memory_order_relaxed));
  event("write_in_place", ssd_->WritesInPlace());
  event("write_in_place_aborted", ssd_->WriteInPlaceAborts());

  // Bytes that actually crossed the device boundary, as opposed to the logical
  // bytes in mori_umbp_backend_bytes_total.  Reads exclude single-flight merges,
  // so this stays a true measure of device traffic.
  bytes("device_write", ssd_->CopyBytes());
  bytes("device_read", ssd_->ReadBytes());

  // Disk comes back a whole segment file at a time; capacity used counts dead
  // records too, so live_bytes says how much of it still holds keys.
  const TierSpaceStats space = ssd_->SpaceStats();
  event("segment_deleted", space.files_deleted);
  bytes("segment_deleted", space.bytes_deleted);
  event("segment_no_room", space.no_room);
  state("live_bytes", space.live_bytes);
  state("segment_files", space.files);

  state("staging_pages_in_use", staging_pages_in_use_.load(std::memory_order_relaxed));
  state("staging_pages_peak", staging_pages_peak_.load(std::memory_order_relaxed));
  state("staging_pages_total", cfg_.staging_pages);
  return out;
}

}  // namespace mori::umbp
