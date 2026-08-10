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
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "mori/io/engine.hpp"
#include "umbp/distributed/peer/backend/medium_backend.h"
#include "umbp/distributed/peer/backend/peer_page_allocator.h"
#include "umbp/distributed/types.h"
#include "umbp/local/host_mem_allocator.h"

namespace mori::umbp {

// One medium's (DRAM or HBM) peer-owned allocator + key map.  One instance ==
// one medium: the tier is the identity of the object (design doc §3), which is
// why no method below takes a TierType — PeerDramAllocator's internal
// map<TierType, ...> is gone, relocated to BackendRegistry (one PageBackend
// instance per registry entry).
//
// In the master-as-advisor design, this object is the canonical owner of
// every per-key page set on the local node — master's GlobalBlockIndex is a
// downstream projection of the events this object emits.
//
// Lifecycle of a Put on this peer:
//   1. Allocate(size) -> AllocateResult { slot_id, pages, ... }
//   2. (Writer RDMAs into the slot's pages.)
//   3. Commit(slot_id, key) -> moves pending -> owned, queues an ADD event.
//      OR Abort(slot_id) -> drops pending; no event.
// If the writer dies between (1) and (3), the reaper drops the pending
// slot at `pending_ttl` and no event is ever emitted.
//
// Lifecycle of an Evict (master-driven):
//   1. Master sends EvictKey(keys[]) — handled by the peer service.
//   2. Peer service calls Evict(keys) here.
//   3. For each key actually freed, a REMOVE event is queued.
//   4. Heartbeat ships the REMOVE events and master drops the index entry.
//
// MediumBackend is the only base: the Phase 2 transitional second base
// (OwnedLocationSource) is gone, along with the type itself — MasterClient now
// consumes BackendRegistry directly and MediumBackend carries the heartbeat
// event contract (design doc §3, "this interface absorbs the existing
// OwnedLocationSource").
class PageBackend : public MediumBackend {
 public:
  // Buffers already known (bases/descs supplied by the caller at
  // construction).  Used directly by tests that want to exercise the
  // allocator's bookkeeping without a real TransferEngine/IOEngine —
  // Init() on an instance built this way is a no-op success (already
  // configured).  Production backends use the OwnershipConfig constructor
  // below instead, so no buffer pointer ever crosses PoolClientConfig
  // (design doc §1 item 4 / Phase 2b).
  struct TierConfig {
    std::vector<uint64_t> buffer_sizes;
    std::vector<std::vector<uint8_t>> buffer_descs;  // packed mori::io::MemoryDesc bytes
    // Local host base pointer of each buffer (same order as buffer_sizes).
    // Used by AcquireDramCopyPin to resolve a (buffer_index, page_index)
    // page to a directly-readable local pointer for the SSD copy worker.
    // Empty is allowed for tiers/deployments with no DRAM copy (e.g. unit
    // tests that never pin); pin acquisition then yields no segments.
    std::vector<void*> buffer_bases;
  };

  // Ownership config: sizes only.  Init(TransferEngine*) self-allocates one
  // HostMemAllocator buffer per entry and registers it with the engine —
  // PageBackend is the allocator now, not a bookkeeper over memory handed to
  // it from outside (design doc §1 item 4).
  struct OwnershipConfig {
    std::vector<uint64_t> buffer_sizes;
    bool use_hugepages = false;
    uint64_t hugepage_size = 2ULL * 1024 * 1024;
    int numa_node = -1;
    bool prefault = true;
  };

  // `pending_ttl` is the only TTL in the system.  After this elapses
  // without a matching Commit / Abort, the reaper frees the slot's
  // pages.  `read_lease_ttl` is how long a single Resolve protects its
  // key from concurrent Evict.  `reaper_interval` is the wakeup cadence
  // for sweeping expired pendings and read leases.
  PageBackend(TierType tier, uint64_t page_size, TierConfig cfg,
              std::chrono::milliseconds pending_ttl,
              std::chrono::milliseconds read_lease_ttl = std::chrono::milliseconds{500},
              std::chrono::milliseconds reaper_interval = std::chrono::milliseconds{200});

  // Production constructor: no memory yet.  Init(TransferEngine*) performs
  // the actual self-allocation + registration.
  PageBackend(TierType tier, uint64_t page_size, OwnershipConfig ownership,
              std::chrono::milliseconds pending_ttl,
              std::chrono::milliseconds read_lease_ttl = std::chrono::milliseconds{500},
              std::chrono::milliseconds reaper_interval = std::chrono::milliseconds{200});

  ~PageBackend() override;

  PageBackend(const PageBackend&) = delete;
  PageBackend& operator=(const PageBackend&) = delete;

  // ==================== MediumBackend ====================

  TierType Tier() const override { return tier_; }
  const char* Name() const override { return "PageBackend"; }

  // No-op success if constructed with a pre-built TierConfig (already
  // configured; the caller drives the reaper itself via RunReaperOnceForTest).
  // Otherwise self-allocates OwnershipConfig::buffer_sizes via HostMemAllocator,
  // registers each buffer with `registrar`, and starts the reaper — a backend
  // is fully live when Init returns, so no caller has to know it has a reaper.
  bool Init(MemoryRegistrar* registrar) override;
  void Shutdown() override;

  TierCapacity Capacity() const override;
  uint64_t OwnedKeyCount() const override;

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
  std::vector<EvictResult> Evict(const std::vector<std::string>& keys) override;

  uint64_t PageSize() const override { return page_size_; }
  std::vector<BufferMemoryDescBytes> AllBufferDescs() const override;

  size_t BufferCount() const override { return buffer_refs_.size(); }
  TransferRef BufferRef(uint32_t buffer_index) const override;

  // ==================== Single-key convenience ====================
  // NOT part of MediumBackend (design doc §3: "single-key Allocate/Commit/
  // Abort/Resolve — the single-key RPCs are served by one-element batches").
  // Kept as dedicated (non-wrapping) methods for the hot single-key path in
  // PoolClient/PeerServiceServer — same shape as before, minus the tier
  // parameter.  `descs` is left empty on the AllocateResult/ResolvedEntry
  // returned here; callers that need descs call BufferDescsForPages
  // separately (mirrors pre-refactor behavior).

  AllocateResult Allocate(const std::string& key, uint64_t size);
  bool Commit(uint64_t slot_id, const std::string& key, uint64_t& bytes_committed);
  bool Abort(uint64_t slot_id);
  ResolvedEntry Resolve(const std::string& key);

  uint64_t PendingTtlMs() const { return static_cast<uint64_t>(pending_ttl_.count()); }

  // Returns the deduplicated, ascending-by-buffer_index list of descs
  // referenced by `pages`.
  std::vector<BufferMemoryDescBytes> BufferDescsForPages(
      const std::vector<PageLocation>& pages) const;

  // ==================== DRAM copy pin (SSD copy-on-commit) ====================

  // A copy pin protects a committed key's pages from eviction while the SSD
  // copy worker reads them.  It is an in-process lifetime guard (NOT a
  // master lease): pages stay readable until ReleaseDramCopyPin.  `segments`
  // are directly-readable local pointers + lengths (pages may be
  // non-contiguous across buffers).
  struct DramCopyPin {
    std::vector<std::pair<const void*, size_t>> segments;
    size_t total_size = 0;
    uint64_t pin_token = 0;  // release-time validation
  };

  // Atomically (under mutex_) confirm `key` is owned, mark it pinned, and
  // resolve its pages to local segments.  Returns nullopt if the key is
  // not owned (already evicted -> worker drops the task) or is already
  // pinned (duplicate task).  While pinned, Evict(key) reports
  // bytes_freed=0 and emits no REMOVE (master retries next round).  There
  // is NO TTL: the pin lives until ReleaseDramCopyPin.  The caller MUST
  // release (use a RAII guard) so a worker exit always frees the pin.
  std::optional<DramCopyPin> AcquireDramCopyPin(const std::string& key);

  // Release a pin.  No-op if the key is not pinned or the token does not
  // match (tolerates duplicate / late release).
  void ReleaseDramCopyPin(const std::string& key, uint64_t pin_token);

  // ==================== Heartbeat auto-flush ====================

  // MediumBackend contract.  The hook is invoked while the allocator lock is
  // held, so `cb` must not re-enter this object.  `threshold` should be >= 1.
  void SetAutoFlushHook(size_t threshold, std::function<void()> cb) override;

  // ==================== Reaper ====================

  // Test seam: run one reaper sweep synchronously without the thread.
  void RunReaperOnceForTest() { ReaperSweep(); }

 private:
  // Lifecycle is owned by Init/Shutdown, not by the caller.  Both are
  // idempotent.
  void StartReaper();
  void StopReaper();

  struct PendingSlot {
    uint64_t slot_id = 0;
    std::vector<PageLocation> pages;
    uint64_t size = 0;
    std::chrono::steady_clock::time_point deadline;
    // Snapshot of allocator_generation_ at Allocate().  Commit() rejects
    // slots whose generation no longer matches the current.
    uint64_t generation = 0;
  };

  struct OwnedSlot {
    std::vector<PageLocation> pages;
    uint64_t size = 0;
  };

  AllocateResult AllocateLocked(const std::string& key, uint64_t size);
  bool CommitLocked(uint64_t slot_id, const std::string& key, uint64_t& bytes_committed);
  bool AbortLocked(uint64_t slot_id);

  // Single entry point for appending to the event outbox; when the outbox
  // reaches auto_flush_threshold_ it invokes auto_flush_cb_ (under the lock)
  // to wake the heartbeat.  Caller MUST hold `mutex_`.
  void QueueEventLocked(KvEvent event);

  // Build the full ADD list for every owned key.  Caller MUST hold `mutex_`.
  std::vector<KvEvent> SnapshotOwnedKeysLocked() const;

  // Caller MUST hold `mutex_`.  True iff `key`'s read-lease deadline is
  // still in the future.  Drops the entry if it has expired.
  bool HasActiveReadLeaseLocked(const std::string& key);

  // Caller MUST hold `mutex_`.  True iff `key` currently has an active copy
  // pin (protects its pages from eviction).
  bool HasActivePinLocked(const std::string& key) const;

  // Caller MUST hold `mutex_`.  Resolve `pages` to directly-readable local
  // segments via buffer_bases_.  Empty if no bases are configured.
  std::vector<std::pair<const void*, size_t>> BuildCopySegmentsLocked(
      const std::vector<PageLocation>& pages, uint64_t total_size) const;

  // Caller MUST hold `mutex_`.
  std::vector<BufferMemoryDescBytes> BuildBufferDescsLocked(
      const std::vector<PageLocation>& pages) const;

  // Build buffer_refs_ from the installed bases + descs.  Called once, from
  // InstallTierConfig.
  void BuildBufferRefs();

  // Install `cfg` as this backend's live buffers (used by both constructors:
  // directly for the TierConfig ctor, and from Init() after self-allocation
  // for the OwnershipConfig ctor).
  void InstallTierConfig(TierConfig cfg);

  void ReaperLoop();
  void ReaperSweep();

  // Owned pages held back at ClearLocal() because of an active read lease.
  // Released by ReaperSweep() when release_at <= now.
  struct DeferredFree {
    std::string key;  // for debug log only.
    std::vector<PageLocation> pages;
    std::chrono::steady_clock::time_point release_at;
  };

  TierType tier_;
  mutable std::mutex mutex_;
  uint64_t page_size_;
  std::chrono::milliseconds pending_ttl_;
  std::chrono::milliseconds read_lease_ttl_;
  std::chrono::milliseconds reaper_interval_;

  std::unique_ptr<PageBitmapAllocator> allocator_;
  std::vector<std::vector<uint8_t>> buffer_descs_;
  // Local host base pointer per buffer_index.  Source of truth for page ->
  // local pointer, used by AcquireDramCopyPin.
  std::vector<void*> buffer_bases_;
  // The same buffers as transfer endpoints, built once at InstallTierConfig so
  // BufferRef() costs an index rather than an msgpack unpack.  Immutable after
  // Init, hence readable without mutex_.
  std::vector<TransferRef> buffer_refs_;

  std::unordered_map<uint64_t, PendingSlot> pending_;
  std::unordered_map<std::string, OwnedSlot> owned_;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point> read_lease_until_;
  std::vector<KvEvent> pending_events_;

  // Auto-flush state (see QueueEventLocked):
  size_t auto_flush_threshold_ = SIZE_MAX;  // events before a flush; default = never
  std::function<void()> auto_flush_cb_;     // set once at startup; wakes the heartbeat

  // Active copy pins.  An entry means `key`'s owned pages are protected
  // from eviction until ReleaseDramCopyPin.  No TTL: lifetime is bound to
  // the worker via a RAII guard, not a deadline (force-freeing under an
  // in-flight backend Write would be a use-after-free).
  struct PinState {
    uint64_t token = 0;
    std::chrono::steady_clock::time_point acquired_at;  // for long-running warning only
  };
  std::unordered_map<std::string, PinState> pins_;
  uint64_t next_pin_token_ = 1;

  std::vector<DeferredFree> deferred_frees_;

  // Bumped by ClearLocal(); snapshotted into each PendingSlot.  Commit()
  // rejects pre-clear slots via mismatch.  Local-only.
  uint64_t allocator_generation_ = 0;

  std::atomic<uint64_t> next_slot_id_{1};

  // Set by ClearLocal(), cleared by ClearFullSyncAcked().  Gates Allocate()
  // so no new owned key appears between local clear and the first acked
  // full-sync empty heartbeat.
  std::atomic<bool> clear_full_sync_pending_{false};

  std::thread reaper_thread_;
  std::atomic<bool> reaper_running_{false};
  std::mutex reaper_cv_mutex_;
  std::condition_variable reaper_cv_;

  // ---- Ownership (only set when constructed via OwnershipConfig) ----
  std::optional<OwnershipConfig> ownership_;
  bool owns_memory_ = false;  // true once Init() has self-allocated
  std::vector<HostBufferHandle> owned_buffer_handles_;
  std::vector<TransferRef> owned_refs_;
  MemoryRegistrar* registrar_ = nullptr;  // non-owning; set at Init(), used at Shutdown()
};

// The one place a caller outside this file obtains a PageBackend.
//
// Phase 5 Rule A ("only one place may name a concrete backend") was blocked on
// PoolClient::Init needing LocalBufferViews() for the local memcpy path; Phase 6
// deleted that call, so construction can go through a factory returning the
// interface.  The class itself stays in this header because its unit tests
// drive the allocator bookkeeping directly through the TierConfig constructor —
// what matters for the acceptance test is that the PRODUCTION path names no
// concrete type, and it no longer does.
std::unique_ptr<MediumBackend> MakePageBackend(TierType tier, uint64_t page_size,
                                               PageBackend::OwnershipConfig ownership,
                                               std::chrono::milliseconds pending_ttl,
                                               std::chrono::milliseconds read_lease_ttl);

}  // namespace mori::umbp
