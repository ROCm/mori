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
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/peer/owned_location_source.h"
#include "umbp/distributed/peer/peer_page_allocator.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

// Peer-owned DRAM/HBM allocator + key map.  In the master-as-advisor
// design, this object is the canonical owner of every per-key page set
// on the local node — master's GlobalBlockIndex is a downstream
// projection of the events this object emits.
//
// Lifecycle of a Put on this peer:
//   1. Allocate(size, tier) -> PendingSlot { slot_id, pages, ... }
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
class PeerDramAllocator : public OwnedLocationSource {
 public:
  // Per-tier inputs: each i'th buffer's size in bytes, and the matching
  // packed mori::io::MemoryDesc that the writer will use to RDMA into
  // that buffer.  `descs.size()` must equal `buffer_sizes.size()` (or
  // both be empty, meaning "this tier is not configured").
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

  // `pending_ttl` is the only TTL in the system.  After this elapses
  // without a matching Commit / Abort, the reaper frees the slot's
  // pages.  `read_lease_ttl` is how long a single Resolve protects its
  // key from concurrent Evict.  `reaper_interval`
  // is the wakeup cadence for sweeping expired pendings and read
  // leases.
  PeerDramAllocator(uint64_t page_size, TierConfig dram, TierConfig hbm,
                    std::chrono::milliseconds pending_ttl,
                    std::chrono::milliseconds read_lease_ttl = std::chrono::milliseconds{500},
                    std::chrono::milliseconds reaper_interval = std::chrono::milliseconds{200});
  ~PeerDramAllocator();

  PeerDramAllocator(const PeerDramAllocator&) = delete;
  PeerDramAllocator& operator=(const PeerDramAllocator&) = delete;

  // -------- RPC entry points --------

  struct PendingSlot {
    uint64_t slot_id = 0;
    TierType tier = TierType::UNKNOWN;
    std::vector<PageLocation> pages;
    uint64_t size = 0;
    std::chrono::steady_clock::time_point deadline;
    // Snapshot of allocator_generation_ at Allocate().  Commit()
    // rejects slots whose generation no longer matches the current.
    uint64_t generation = 0;
  };

  struct OwnedSlot {
    TierType tier = TierType::UNKNOWN;
    std::vector<PageLocation> pages;
    uint64_t size = 0;
  };

  struct ResolveResult {
    bool found = false;
    TierType tier = TierType::UNKNOWN;
    std::vector<PageLocation> pages;
    uint64_t size = 0;
  };

  // ResolveResult + descs built under the same lock.
  struct ResolvedEntry {
    bool found = false;
    TierType tier = TierType::UNKNOWN;
    std::vector<PageLocation> pages;
    uint64_t size = 0;
    std::vector<BufferMemoryDescBytes> descs;
  };

  struct EvictResult {
    std::string key;
    uint64_t bytes_freed = 0;  // 0 if key was unknown / already freed / read-leased
  };

  // Allocate() outcome. Only kSuccessAllocated populates slot.
  // kFailedNoSpace is split out so the writer can keep retrying on
  // other peers; every other failure collapses into kFailed and is
  // diagnosed via the WARN log emitted inside Allocate().
  enum class Outcome {
    kSuccessAllocated,
    kSuccessAlreadyExists,  // owned_[key] dedup hit
    kFailed,                // generic — see allocator log for reason
    kFailedNoSpace,         // tier exhausted; retry on another peer
  };

  struct AllocateResult {
    Outcome outcome = Outcome::kFailed;
    std::optional<PendingSlot> slot;  // populated iff outcome == kSuccessAllocated
  };

  struct AllocateRequest {
    std::string key;
    uint64_t size = 0;
    TierType tier = TierType::UNKNOWN;
  };

  struct BatchAllocateResult {
    Outcome outcome = Outcome::kFailed;
    std::optional<PendingSlot> slot;
    std::vector<BufferMemoryDescBytes> descs;
  };

  struct CommitRequest {
    uint64_t slot_id = 0;
    std::string key;
  };

  struct CommitResult {
    bool success = false;
    uint64_t bytes_committed = 0;
  };

  // Reserve `size` bytes on `tier` for `key`.  `key` enables owned_
  // dedup (master-index-lag fallback; primary dedup is at BatchRoutePut).
  AllocateResult Allocate(const std::string& key, uint64_t size, TierType tier);

  std::vector<BatchAllocateResult> BatchAllocate(const std::vector<AllocateRequest>& entries);

  // Move pending -> owned and queue an ADD event.  Returns false if
  // slot_id is unknown (already reaped, already aborted, or never
  // existed) — the writer treats false as a Put failure.
  bool Commit(uint64_t slot_id, const std::string& key, uint64_t& bytes_committed);

  std::vector<CommitResult> BatchCommit(const std::vector<CommitRequest>& entries);

  // Drop a pending slot.  Idempotent: returns true if the slot was
  // dropped here OR was already gone (reaped or never existed).  False
  // is reserved for a state we don't currently produce (kept for future
  // contract changes).
  bool Abort(uint64_t slot_id);

  std::vector<bool> BatchAbort(const std::vector<uint64_t>& slot_ids);

  // Look up a key the writer was just routed to.  Extends the
  // read-lease deadline for `key` to now + `read_lease_ttl_`;
  // concurrent Evict requests for that key during the lease window
  // report bytes_freed=0.
  ResolveResult Resolve(const std::string& key);

  // Batched Resolve + BufferDescsForPages under a single mutex_ hold.
  // Per-key behavior byte-identical to Resolve() + BufferDescsForPages().
  std::vector<ResolvedEntry> BatchResolve(const std::vector<std::string>& keys);

  // Master-driven eviction.  Idempotent: keys that are unknown or
  // already gone produce zero-bytes entries; keys with active read
  // leases produce zero-bytes entries (master will retry next round).
  // For every key actually freed, a REMOVE event is queued.
  std::vector<EvictResult> Evict(const std::vector<std::string>& keys);

  // -------- DRAM copy pin (SSD copy-on-commit) --------

  // A copy pin protects a committed key's DRAM pages from eviction while
  // the SSD copy worker reads them.  It is an in-process lifetime guard
  // (NOT a master lease): pages stay readable until ReleaseDramCopyPin.
  // `segments` are directly-readable local pointers + lengths (pages may
  // be non-contiguous across buffers).
  struct DramCopyPin {
    std::vector<std::pair<const void*, size_t>> segments;
    size_t total_size = 0;
    uint64_t pin_token = 0;  // release-time validation
  };

  // Atomically (under mutex_) confirm `key` is owned, mark it pinned, and
  // resolve its pages to local segments.  Returns nullopt if the key is
  // not owned (already evicted -> worker drops the task) or is already
  // pinned (duplicate task).  While pinned, Evict(key) reports
  // bytes_freed=0 and emits no REMOVE DRAM (master retries next round).
  // There is NO TTL: the pin lives until ReleaseDramCopyPin.  The caller
  // MUST release (use a RAII guard) so a worker exit always frees the pin.
  std::optional<DramCopyPin> AcquireDramCopyPin(const std::string& key);

  // Release a pin.  No-op if the key is not pinned or the token does not
  // match (tolerates duplicate / late release).
  void ReleaseDramCopyPin(const std::string& key, uint64_t pin_token);

  // -------- Distributed Clear --------

  // Drop owned/lease state, bump allocator_generation_ so pre-clear
  // pending slots fail at Commit(), and clear the event outbox.  Owned
  // pages with an active read lease are deferred to the reaper; others
  // are freed immediately.  Allocate() returns nullopt until
  // ClearFullSyncAcked().
  void ClearLocal();

  // Called by the heartbeat thread after the first full-sync empty
  // snapshot is acked by master.  Re-enables Allocate().
  void ClearFullSyncAcked();

  bool IsClearFullSyncPending() const {
    return clear_full_sync_pending_.load(std::memory_order_acquire);
  }

  // -------- Heartbeat helpers --------

  // Drain the outbox of events queued since the last call.  Called by
  // the heartbeat shipper; clears the buffer.  OwnedLocationSource.
  std::vector<KvEvent> DrainPendingEvents() override;

  // Full snapshot of every owned key as ADD events.  Used when master
  // requests a full sync (seq gap or master restart).  OwnedLocationSource.
  std::vector<KvEvent> SnapshotOwnedKeys() const override;

  // Live owned-key count per tier.  O(tiers) — cheap to call every
  // heartbeat.  Used by the heartbeat shipper for per-client metrics.
  std::map<TierType, uint64_t> OwnedKeyCountByTier() const;

  // Live capacity per tier — derived directly from the underlying
  // bitmap allocators, so always reflects pending+owned correctly.
  std::map<TierType, TierCapacity> TierCapacitiesSnapshot() const;

  // -------- Wire-side helpers (used by peer service handlers) --------

  uint64_t PageSize() const { return page_size_; }
  uint64_t PendingTtlMs() const { return static_cast<uint64_t>(pending_ttl_.count()); }

  // Returns descs for all configured buffers on `tier` (sorted by
  // buffer_index ascending), ready to drop into a peer-service RPC
  // response.  Empty if the tier is not configured.  Used by
  // GetPeerInfo for first-contact bootstrap; AllocateSlot / ResolveKey
  // need only the descs that actually appear in `pages` and use the
  // overload below.
  std::vector<BufferMemoryDescBytes> AllBufferDescs(TierType tier) const;

  // Returns the deduplicated, ascending-by-buffer_index list of descs
  // referenced by `pages`.
  std::vector<BufferMemoryDescBytes> BufferDescsForPages(
      TierType tier, const std::vector<PageLocation>& pages) const;

  // -------- Reaper --------

  void StartReaper();
  void StopReaper();

  // Test seam: run one reaper sweep synchronously without the thread.
  // Public so unit tests can drive deterministic TTL expiry without
  // racing the background thread.
  void RunReaperOnceForTest() { ReaperSweep(); }

 private:
  // Caller MUST hold `mutex_`.
  PageBitmapAllocator* AllocatorForLocked(TierType tier);
  const PageBitmapAllocator* AllocatorForLocked(TierType tier) const;

  AllocateResult AllocateLocked(const std::string& key, uint64_t size, TierType tier);
  bool CommitLocked(uint64_t slot_id, const std::string& key, uint64_t& bytes_committed);
  bool AbortLocked(uint64_t slot_id);

  // Caller MUST hold `mutex_`.  True iff `key`'s read-lease deadline
  // is still in the future.  Drops the entry if it has expired.
  bool HasActiveReadLeaseLocked(const std::string& key);

  // Caller MUST hold `mutex_`.  True iff `key` currently has an active
  // copy pin (protects its pages from eviction).
  bool HasActivePinLocked(const std::string& key) const;

  // Caller MUST hold `mutex_`.  Resolve `pages` to directly-readable
  // local segments via the per-tier buffer bases.  Empty if the tier has
  // no bases configured.
  std::vector<std::pair<const void*, size_t>> BuildCopySegmentsLocked(
      TierType tier, const std::vector<PageLocation>& pages, uint64_t total_size) const;

  // Caller MUST hold `mutex_`.
  std::vector<BufferMemoryDescBytes> BuildBufferDescsLocked(
      TierType tier, const std::vector<PageLocation>& pages) const;

  void ReaperLoop();
  void ReaperSweep();

  // Owned pages held back at ClearLocal() because of an active read
  // lease.  Released by ReaperSweep() when release_at <= now.
  struct DeferredFree {
    std::string key;  // for debug log only.
    TierType tier = TierType::UNKNOWN;
    std::vector<PageLocation> pages;
    std::chrono::steady_clock::time_point release_at;
  };

  mutable std::mutex mutex_;
  uint64_t page_size_;
  std::chrono::milliseconds pending_ttl_;
  std::chrono::milliseconds read_lease_ttl_;
  std::chrono::milliseconds reaper_interval_;

  std::map<TierType, std::unique_ptr<PageBitmapAllocator>> allocators_;
  std::map<TierType, std::vector<std::vector<uint8_t>>> tier_descs_;
  // Per-tier local host base pointer per buffer_index (parallel to the
  // allocator's buffers).  Source of truth for page -> local pointer used
  // by AcquireDramCopyPin.
  std::map<TierType, std::vector<void*>> tier_bases_;

  std::unordered_map<uint64_t, PendingSlot> pending_;
  std::unordered_map<std::string, OwnedSlot> owned_;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point> read_lease_until_;
  std::vector<KvEvent> pending_events_;

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

  // Bumped by ClearLocal(); snapshotted into each PendingSlot.
  // Commit() rejects pre-clear slots via mismatch.  Local-only.
  uint64_t allocator_generation_ = 0;

  std::atomic<uint64_t> next_slot_id_{1};

  // Set by ClearLocal(), cleared by ClearFullSyncAcked().  Gates
  // Allocate() so no new owned key appears between local clear and the
  // first acked full-sync empty heartbeat.
  std::atomic<bool> clear_full_sync_pending_{false};

  std::thread reaper_thread_;
  std::atomic<bool> reaper_running_{false};
  std::mutex reaper_cv_mutex_;
  std::condition_variable reaper_cv_;
};

}  // namespace mori::umbp
