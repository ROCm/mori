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

#include <msgpack.hpp>
#include <stdexcept>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/distributed/master/master_metrics.h"

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

  // Runtime kill-switch for the file->GPU (GDS) read path.  On by default when
  // the build has hipfile; UMBP_ENABLE_GDS=0 forces every SSD read back onto the
  // staging arena without a rebuild (for A/B against GDS, or as a safety valve).
  if (const char* env = std::getenv("UMBP_ENABLE_GDS")) {
    const std::string v(env);
    gds_enabled_ = !(v == "0" || v == "false" || v == "FALSE");
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

  // Free list, high index first so page 0 is handed out first (nicer logs).
  const uint32_t usable_pages = static_cast<uint32_t>(staging_size_ / cfg_.page_size);
  free_pages_.reserve(usable_pages);
  for (uint32_t i = usable_pages; i > 0; --i) free_pages_.push_back(i - 1);

  // Crash-restart leftover: metadata is gone but files may remain, so used
  // capacity would diverge from owned_.  Must run before any IO.
  ssd_->DiscardLeftoverOnStartup();

  initialized_ = true;
  StartReaper();
  MORI_UMBP_INFO(
      "[SsdBackend] Init staging_pages={} page_size={} arena_bytes={} hugepages={} "
      "single_flight={}",
      usable_pages, cfg_.page_size, staging_size_, cfg_.staging_use_hugepages,
      cfg_.ssd.ssd.single_flight_reads);
  return true;
}

void SsdBackend::Shutdown() {
  if (!initialized_) return;
  StopReaper();

  if (registrar_ != nullptr && buffer_ref_.Valid()) registrar_->Deregister(buffer_ref_);
  buffer_ref_ = TransferRef{};
  buffer_desc_.clear();

  // Release the GDS file handles obtained lazily during resolves.
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
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Invalidate outstanding slots so a late Commit fails rather than
    // resurrecting a key the cluster believes is gone.
    for (auto& [slot_id, slot] : pending_) ReleaseStagingPageLocked(slot.page_index);
    pending_.clear();
    for (auto& [key, lease] : read_leases_) ReleaseStagingPageLocked(lease.page_index);
    read_leases_.clear();
    unshipped_events_ = 0;
  }
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
  keys.reserve(todo.size());
  srcs.reserve(todo.size());
  sizes.reserve(todo.size());
  batched.reserve(todo.size());
  for (size_t i : todo) {
    const void* src = StagingPagePtr(slots[i].page_index);
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
    for (size_t i : todo) ReleaseStagingPageLocked(slots[i].page_index);
    for (size_t j = 0; j < batched.size(); ++j) {
      if (ok[j]) NoteEventQueuedLocked();
    }
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
  if (keys.empty()) return results;
  const auto now = std::chrono::steady_clock::now();

  // Fills `out` from a lease the caller has already located.  Caller holds
  // mutex_.  Factored out because the lease-hit and read-completed paths publish
  // byte-identical results and drifting between them would be a silent bug.
  auto publish = [&](ResolvedEntry& out, const ReadLease& lease) {
    out.found = true;
    out.pages = {PageLocation{kStagingBufferIndex, lease.page_index}};
    out.size = lease.size;
    out.page_size = cfg_.page_size;
    if (include_descs && !buffer_desc_.empty()) {
      out.descs = {BufferMemoryDescBytes{kStagingBufferIndex, buffer_desc_}};
    }
  };

  // Phase 1 (ONE lock): serve every key that is already staged, then size and
  // claim a staging page for the rest.  An existing lease means the bytes are
  // there: extend it and reuse the page rather than re-reading the SSD for a
  // concurrent second reader.
  std::vector<size_t> todo;            // keys needing a device read
  std::vector<uint32_t> pages;         // parallel to todo
  std::vector<void*> dsts;             // parallel to todo
  std::vector<size_t> caps;            // parallel to todo
  std::vector<std::string> read_keys;  // parallel to todo
  todo.reserve(keys.size());
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
      auto it = read_leases_.find(keys[i]);
      if (it != read_leases_.end()) {
        it->second.expires_at = now + cfg_.read_lease_ttl;
        publish(results[i], it->second);
        continue;
      }

      const uint64_t size = ssd_->SizeOf(keys[i]);
      if (size == 0 || size > cfg_.page_size) continue;  // found = false

      // Zero-copy GDS (UMBP_ENABLE_GDS): when enabled, a file engine is present,
      // and the record is on an O_DIRECT fd, publish a FileRef and skip the
      // staging arena — the reader (GdsEngine) DMAs the range into device memory.
      std::optional<RecordLocation> loc;
      if (gds_enabled_ && (loc = ssd_->LocateRecord(keys[i])) && loc->direct_io && loc->fd >= 0) {
        if (void* handle = GdsHandleForFd(loc->fd)) {
          results[i].found = true;
          results[i].size = loc->value_size;
          results[i].page_size = cfg_.page_size;
          results[i].file_ref =
              TransferRef::File(loc->fd, loc->value_offset, loc->readable_size, handle);
          continue;
        }
      }

      const uint32_t page = AcquireStagingPageLocked();
      if (page == kNoPage) {
        // Staging exhausted.  Reported as a miss, which is the wrong shape for
        // "this node has it, come back" — see the header's note on the rejected
        // retry state.  Sizing the arena for read concurrency is the mitigation.
        slot_full_rejects_.fetch_add(1, std::memory_order_relaxed);
        MORI_UMBP_WARN("[SsdBackend] Resolve: staging arena exhausted, key '{}' degraded to miss",
                       keys[i]);
        continue;
      }
      todo.push_back(i);
      pages.push_back(page);
      dsts.push_back(StagingPagePtr(page));
      caps.push_back(cfg_.page_size);
      read_keys.push_back(keys[i]);
    }
  }
  if (todo.empty()) return results;

  // Phase 2 (no lock): ONE batched SSD read.  ShardedSsdTier turns this into
  // concurrent IO on every drive holding part of the batch, and PeerSsdManager
  // coalesces any same-key duplicates within it; the per-key PrepareRead loop
  // this replaces could do neither.  PeerSsdManager marks each key in-flight for
  // the window, so eviction already skips them.
  const std::vector<SsdReadOutcome> outcomes = ssd_->PrepareReadBatch(read_keys, dsts, caps);

  // Phase 3 (ONE lock): install a lease per key that landed, hand back the pages
  // of the ones that did not.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t j = 0; j < todo.size(); ++j) {
      const size_t i = todo[j];
      if (outcomes[j].status != SsdReadStatus::kOk) {
        ReleaseStagingPageLocked(pages[j]);
        continue;
      }
      // A concurrent Resolve of the same key may have staged it while this read
      // was in flight; keep the winner and give this page back.
      auto [it, inserted] = read_leases_.emplace(
          keys[i], ReadLease{pages[j], outcomes[j].size, now + cfg_.read_lease_ttl});
      if (!inserted) {
        ReleaseStagingPageLocked(pages[j]);
        it->second.expires_at = now + cfg_.read_lease_ttl;
      }
      publish(results[i], it->second);
    }
  }
  return results;
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
      if (read_leases_.count(key) != 0) {
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
      staging_expired_reclaims_.fetch_add(1, std::memory_order_relaxed);
      it = pending_.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = read_leases_.begin(); it != read_leases_.end();) {
    if (it->second.expires_at <= now) {
      ReleaseStagingPageLocked(it->second.page_index);
      staging_expired_reclaims_.fetch_add(1, std::memory_order_relaxed);
      it = read_leases_.erase(it);
    } else {
      ++it;
    }
  }
}

std::unique_ptr<MediumBackend> MakeSsdBackend(SsdBackend::Config cfg) {
  return std::make_unique<SsdBackend>(std::move(cfg));
}

std::vector<MediumCounter> SsdBackend::Counters() const {
  // Every value here is a monotonic snapshot; PoolClient ships the delta since
  // its last tick.  Read outcomes and single-flight accounting share one metric
  // name and separate by label, so a dashboard can sum reads or split them
  // without knowing which statuses exist.
  std::vector<MediumCounter> out;
  if (ssd_ == nullptr) return out;

  auto read = [&](const char* status, uint64_t v) {
    out.push_back(MediumCounter{MORI_UMBP_METRIC_SSD_READ_TOTAL,
                                MORI_UMBP_METRIC_SSD_READ_TOTAL_HELP,
                                {{"status", status}},
                                v});
  };
  read("ok", ssd_->ReadOk());
  read("not_found", ssd_->ReadNotFound());
  read("size_too_large", ssd_->ReadSizeTooLarge());
  read("error", ssd_->ReadError());
  // Single-flight: dup is the opportunity, merged is what was actually taken.
  // dup - merged flatlining at dup means coalescing is off or the merge window
  // is being missed, which is invisible from the read outcomes alone.
  read("lead", ssd_->ReadLead());
  read("dup", ssd_->ReadDup());
  read("merged", ssd_->ReadMerged());

  auto simple = [&](const char* name, const char* help, uint64_t v) {
    out.push_back(MediumCounter{name, help, {}, v});
  };
  simple(MORI_UMBP_METRIC_SSD_COPY_BYTES_TOTAL, MORI_UMBP_METRIC_SSD_COPY_BYTES_TOTAL_HELP,
         ssd_->CopyBytes());
  simple(MORI_UMBP_METRIC_SSD_READ_BYTES_TOTAL, MORI_UMBP_METRIC_SSD_READ_BYTES_TOTAL_HELP,
         ssd_->ReadBytes());
  simple(MORI_UMBP_METRIC_SSD_EVICTION_ROUNDS_TOTAL,
         MORI_UMBP_METRIC_SSD_EVICTION_ROUNDS_TOTAL_HELP, ssd_->EvictionRounds());
  simple(MORI_UMBP_METRIC_SSD_EVICTION_VICTIMS_TOTAL,
         MORI_UMBP_METRIC_SSD_EVICTION_VICTIMS_TOTAL_HELP, ssd_->EvictionVictims());
  simple(MORI_UMBP_METRIC_SSD_EVICTION_BYTES_FREED_TOTAL,
         MORI_UMBP_METRIC_SSD_EVICTION_BYTES_FREED_TOTAL_HELP, ssd_->EvictionBytesFreed());
  simple(MORI_UMBP_METRIC_SSD_EVICTION_BACKEND_FAILED_TOTAL,
         MORI_UMBP_METRIC_SSD_EVICTION_BACKEND_FAILED_TOTAL_HELP, ssd_->EvictionBackendFailures());
  simple(MORI_UMBP_METRIC_SSD_STAGING_SLOT_FULL_REJECTS_TOTAL,
         MORI_UMBP_METRIC_SSD_STAGING_SLOT_FULL_REJECTS_TOTAL_HELP,
         slot_full_rejects_.load(std::memory_order_relaxed));
  simple(MORI_UMBP_METRIC_SSD_STAGING_EXPIRED_RECLAIMS_TOTAL,
         MORI_UMBP_METRIC_SSD_STAGING_EXPIRED_RECLAIMS_TOTAL_HELP,
         staging_expired_reclaims_.load(std::memory_order_relaxed));
  return out;
}

}  // namespace mori::umbp
