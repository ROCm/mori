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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/peer/ssd/peer_ssd_manager.h"

namespace mori::umbp {

// SSD as a MediumBackend, staged through registered host DRAM.
//
// WHY STAGED, AND NOT A FileRef.  transfer_engine.h reserves a file endpoint
// for exactly this backend ("SsdBackend is what needs a FileRef, so SsdBackend
// brings it").  This implementation deliberately does NOT take that option, and
// the reason is worth stating because it is the whole design:
//
//   SSD bytes are not addressable.  Every endpoint UMBP moves bytes between is
//   a memory range — a raw pointer locally, an RDMA MR remotely.  A file offset
//   is neither.  Two ways to close that gap:
//
//     (a) teach the transfer layer a file endpoint — a new TransferRef kind, a
//         PosixFileEngine, and CHAINING in CompositeTransferEngine for the
//         remote reader (file -> bounce -> wire), which that class explicitly
//         does not implement today;
//     (b) make the SSD's published endpoint ordinary registered host memory,
//         and move bytes between it and the device inside the backend.
//
//   (b) is what this class does.  The cost is a host-DRAM copy on each side;
//   the benefit is that SSD reaches the data plane with ZERO new transfer-layer
//   concepts — no new endpoint kind, no new engine, no chaining — and a remote
//   peer reading from this node's SSD sees a perfectly ordinary registered
//   buffer and needs no code at all.  A FileRef backend remains the right
//   answer for a zero-copy / GDS path; it is a strictly larger change and this
//   one does not block it.
//
// SO: THE STAGING ARENA IS THE MEDIUM, as far as everything outside this class
// is concerned.  The pages this backend publishes are host DRAM.  What makes it
// SSD is where the bytes live BETWEEN a Commit and the next Resolve, which is
// the only window in which a cache tier's capacity actually means anything.
//
// LIFECYCLE OF A PUT
//   1. BatchAllocate(key, size) reserves a staging page and publishes it.
//   2. The writer RDMAs (or memcpys) into that page — normal data plane, no
//      SSD awareness anywhere.
//   3. BatchCommit spills the staging page to the SSD via PeerSsdManager::Write,
//      frees the page, and queues an ADD event.  The staging page is NOT the
//      key's home; it is borrowed for the duration of the write.
//
// LIFECYCLE OF A GET
//   1. BatchResolve fills a staging page from the SSD (PeerSsdManager::
//      PrepareRead) and publishes that page under a READ LEASE.
//   2. The reader moves bytes out of it, exactly as it would from DRAM.
//   3. The reaper frees the staging page when the lease expires.
//
//   This is the one place SSD's asymmetry with DRAM shows through the
//   interface: a Resolve does real IO and can fail, and it consumes a scarce
//   resource (staging pages) that a DRAM resolve does not.  A transient
//   shortfall is reported as kBusy, after rolling back every reservation made
//   by that backend batch.  PoolClient retries the whole response so no page
//   location survives across attempts.  A batch whose own working set exceeds
//   the arena is kFailed instead: unchanged retries can never make it fit.
//
// A key may span several staging pages.  Runs are contiguous inside the one
// registered arena because PeerSsdManager::PrepareRead takes one (ptr,
// capacity), while AllocateResult / ResolvedEntry publish every page so the
// ordinary transfer path can address the object at page granularity.
class SsdBackend : public MediumBackend {
 public:
  struct Config {
    // Must match the DRAM/HBM page size the node registered with master: a
    // batch spanning media stays self-describing only if every medium agrees.
    uint64_t page_size = 2ULL * 1024 * 1024;

    // Staging pages held in host DRAM.  This is the backend's aggregate
    // in-flight capacity: an object of N pages consumes N entries until its
    // write commits or its read lease expires.
    uint32_t staging_pages = 64;

    // Back the staging arena with hugetlbfs pages.  Every byte this backend
    // moves crosses the arena twice (device <-> arena, arena <-> wire), so its
    // TLB behaviour is on the critical path in a way an ordinary buffer's is
    // not.  Falls back to 4 KiB pages with a warning when hugepages cannot be
    // reserved — a failure here must not stop the node coming up.
    bool staging_use_hugepages = false;
    uint64_t staging_hugepage_size = 2ULL * 1024 * 1024;

    PeerSsdConfig ssd;

    std::chrono::milliseconds pending_ttl{30000};
    std::chrono::milliseconds read_lease_ttl{3000};
    std::chrono::milliseconds reaper_interval{200};
  };

  explicit SsdBackend(Config cfg);
  ~SsdBackend() override;

  SsdBackend(const SsdBackend&) = delete;
  SsdBackend& operator=(const SsdBackend&) = delete;

  // ==================== MediumBackend ====================

  TierType Tier() const override { return TierType::SSD; }
  const char* Name() const override { return "SsdBackend"; }

  // Allocates + registers the staging arena, discards crash leftovers, and
  // starts the reaper.  Fails if the SSD tier itself could not be opened.
  bool Init(MemoryRegistrar* registrar) override;
  void Shutdown() override;

  // From the SSD tier, NOT from the staging arena: capacity is a statement
  // about what this node can hold for the cluster, and the staging arena holds
  // nothing across a Commit.
  TierCapacity Capacity() const override;
  uint64_t OwnedKeyCount() const override;

  // Forwarded to PeerSsdManager, which already emits TierType::SSD events and
  // whose header anticipated exactly this adapter.
  std::vector<KvEvent> DrainPendingEvents() override;
  std::vector<KvEvent> SnapshotOwnedKeys() const override;
  std::vector<KvEvent> SnapshotOwnedKeysForFullSync() override;

  void ClearLocal() override;
  void ClearFullSyncAcked() override;
  bool IsClearFullSyncPending() const override {
    return clear_full_sync_pending_.load(std::memory_order_acquire);
  }

  std::vector<AllocateResult> BatchAllocate(const std::vector<AllocateRequest>& entries) override;
  std::vector<CommitResult> BatchCommit(const std::vector<CommitRequest>& entries) override;
  std::vector<bool> BatchAbort(const std::vector<uint64_t>& slot_ids) override;
  std::vector<ResolvedEntry> BatchResolve(const std::vector<std::string>& keys,
                                          bool include_descs) override;
  bool AcquireMigrationRead(const std::string& key, ResolvedEntry* resolved) override;
  void ReleaseMigrationRead(const std::string& key) override;
  bool Contains(const std::string& key) const override;
  std::vector<EvictResult> Evict(const std::vector<std::string>& keys) override;

  void SetAutoFlushHook(size_t threshold, std::function<void()> cb) override;

  // MediumBackend contract; forwarded to the manager that owns the outbox.
  void SetEventPublishing(bool enabled) override;

  // PeerSsdManager's read/eviction counters plus this class's own staging-arena
  // counters, as monotonic values.  PoolClient ships the deltas.
  std::vector<MetricSample> SampleMetrics() const override;

  uint64_t PageSize() const override { return cfg_.page_size; }
  std::vector<BufferMemoryDescBytes> AllBufferDescs() const override;

  size_t BufferCount() const override { return buffer_ref_.Valid() ? 1 : 0; }
  TransferRef BufferRef(uint32_t buffer_index) const override;

  // Test seam: run one reaper sweep synchronously without the thread.
  void RunReaperOnceForTest() { ReaperSweep(); }

 private:
  struct StagingSpan {
    uint32_t first_page = 0;
    uint32_t page_count = 0;
  };

  // A contiguous staging span borrowed for a write (pending slot) or read.
  struct PendingSlot {
    uint64_t slot_id = 0;
    std::string key;
    StagingSpan span;
    uint64_t size = 0;
    std::chrono::steady_clock::time_point deadline;
  };

  struct ReadLease {
    StagingSpan span;
    uint64_t size = 0;
    std::chrono::steady_clock::time_point expires_at;
  };

  // Caller MUST hold mutex_.  A zero-length result means no contiguous run of
  // the requested size is currently available.
  StagingSpan AcquireStagingSpanLocked(uint32_t page_count);
  void ReleaseStagingSpanLocked(StagingSpan span);

  // Local address of a staging page.  Valid after Init.
  void* StagingPagePtr(uint32_t page_index) const;

  // hipFile handle for a segment fd, obtained lazily from the registrar's file
  // engine (GdsEngine) and shared by every key on that segment.  Returns nullptr
  // when no file engine is configured; that result is cached so a fd is probed
  // at most once.  Deregistered in Shutdown.
  void* GdsHandleForFd(int fd);

  // Caller MUST hold mutex_.  Bumps the unshipped-event count and fires the
  // auto-flush hook once it crosses the threshold.
  void NoteEventQueuedLocked();

  void StartReaper();
  void StopReaper();
  void ReaperLoop();
  void ReaperSweep();

  Config cfg_;

  mutable std::mutex mutex_;

  std::unique_ptr<PeerSsdManager> ssd_;

  // The staging arena: ONE host buffer of staging_pages * page_size, so every
  // page is contiguous (PrepareRead's requirement) and one registration covers
  // the lot.  buffer_index is therefore always 0.
  std::unique_ptr<PageMemorySource> staging_source_;
  void* staging_base_ = nullptr;
  uint64_t staging_size_ = 0;
  TransferRef buffer_ref_;
  std::vector<uint8_t> buffer_desc_;
  std::vector<bool> staging_page_used_;

  std::unordered_map<uint64_t, PendingSlot> pending_;
  std::unordered_map<std::string, ReadLease> read_leases_;
  std::unordered_map<std::string, ReadLease> migration_reads_;

  // Runtime kill-switch (UMBP_ENABLE_GDS=0): keep every SSD read on the staging
  // arena even when the build has hipfile and a GdsEngine is registered.  Read
  // once at Init; correctness never depends on it (hipfile's own compat mode
  // already covers fastpath misses), so this is for A/B and operational safety.
  bool gds_enabled_ = true;

  // GDS file handles by segment fd (see GdsHandleForFd).  On its own lock, off
  // mutex_, so a resolve holding mutex_ can register a handle without widening
  // what mutex_ guards.
  std::mutex gds_mutex_;
  std::unordered_map<int, void*> gds_handles_;

  size_t unshipped_events_ = 0;
  size_t auto_flush_threshold_ = SIZE_MAX;
  std::function<void()> auto_flush_cb_;

  std::atomic<uint64_t> next_slot_id_{1};
  std::atomic<bool> clear_full_sync_pending_{false};

  // Staging-arena observability.  Relaxed atomics, never correctness state.
  // slot_full_rejects_ is the one to watch: it counts resolve batches that
  // returned BUSY because the arena was temporarily full.
  std::atomic<uint64_t> slot_full_rejects_{0};
  std::atomic<uint64_t> staging_expired_reclaims_{0};
  // Live arena occupancy, mirrored out of staging_page_used_ so SampleMetrics
  // can read it without taking mutex_ (the metrics tick must never contend with
  // a read).
  std::atomic<uint64_t> staging_pages_in_use_{0};
  bool initialized_ = false;
  MemoryRegistrar* registrar_ = nullptr;

  std::thread reaper_thread_;
  std::atomic<bool> reaper_running_{false};
  std::mutex reaper_cv_mutex_;
  std::condition_variable reaper_cv_;
};

// The one place a caller outside this file obtains an SSD backend — same Phase 5
// Rule A shape as MakePageBackend / MakeHbmBackend.
std::unique_ptr<MediumBackend> MakeSsdBackend(SsdBackend::Config cfg);

}  // namespace mori::umbp
