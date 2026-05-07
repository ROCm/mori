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
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

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
class PeerDramAllocator {
 public:
  // Per-tier inputs: each i'th buffer's size in bytes, and the matching
  // packed mori::io::MemoryDesc that the writer will use to RDMA into
  // that buffer.  `descs.size()` must equal `buffer_sizes.size()` (or
  // both be empty, meaning "this tier is not configured").
  struct TierConfig {
    std::vector<uint64_t> buffer_sizes;
    std::vector<std::vector<uint8_t>> buffer_descs;  // packed mori::io::MemoryDesc bytes
  };

  // `pending_ttl` is the only TTL in the system.  After this elapses
  // without a matching Commit / Abort, the reaper frees the slot's
  // pages.  `read_lease_ttl` is how long a single Resolve protects its
  // key from concurrent Evict (Bug #7 resolution).  `reaper_interval`
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

  struct EvictResult {
    std::string key;
    uint64_t bytes_freed = 0;  // 0 if key was unknown / already freed / read-leased
  };

  // Reserve `size` bytes on `tier`.  Returns nullopt on ENOSPC or if
  // the tier is not configured.  The slot is in the pending state
  // until Commit (or Abort, or the reaper expires it).
  std::optional<PendingSlot> Allocate(uint64_t size, TierType tier);

  // Move pending -> owned and queue an ADD event.  Returns false if
  // slot_id is unknown (already reaped, already aborted, or never
  // existed) — the writer treats false as a Put failure.
  bool Commit(uint64_t slot_id, const std::string& key);

  // Drop a pending slot.  Idempotent: returns true if the slot was
  // dropped here OR was already gone (reaped or never existed).  False
  // is reserved for a state we don't currently produce (kept for future
  // contract changes).
  bool Abort(uint64_t slot_id);

  // Look up a key the writer was just routed to.  Bumps the read-lease
  // counter for `key` for `read_lease_ttl_`; concurrent Evict requests
  // for that key during the lease window report bytes_freed=0.
  ResolveResult Resolve(const std::string& key);

  // Master-driven eviction.  Idempotent: keys that are unknown or
  // already gone produce zero-bytes entries; keys with active read
  // leases produce zero-bytes entries (master will retry next round).
  // For every key actually freed, a REMOVE event is queued.
  std::vector<EvictResult> Evict(const std::vector<std::string>& keys);

  // -------- Heartbeat helpers --------

  // Drain the outbox of events queued since the last call.  Called by
  // the heartbeat shipper; clears the buffer.
  std::vector<KvEvent> DrainPendingEvents();

  // Full snapshot of every owned key as ADD events.  Used when master
  // requests a full sync (seq gap or master restart).
  std::vector<KvEvent> SnapshotOwnedKeys() const;

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

  // Caller MUST hold `mutex_`.  True iff `key` has at least one
  // unexpired read-lease entry.  Trims expired entries while at it.
  bool HasActiveReadLeaseLocked(const std::string& key);

  // Caller MUST hold `mutex_`.
  std::vector<BufferMemoryDescBytes> BuildBufferDescsLocked(
      TierType tier, const std::vector<PageLocation>& pages) const;

  void ReaperLoop();
  void ReaperSweep();

  mutable std::mutex mutex_;
  uint64_t page_size_;
  std::chrono::milliseconds pending_ttl_;
  std::chrono::milliseconds read_lease_ttl_;
  std::chrono::milliseconds reaper_interval_;

  std::map<TierType, std::unique_ptr<PageBitmapAllocator>> allocators_;
  std::map<TierType, std::vector<std::vector<uint8_t>>> tier_descs_;

  std::unordered_map<uint64_t, PendingSlot> pending_;
  std::unordered_map<std::string, OwnedSlot> owned_;
  std::unordered_map<std::string, std::deque<std::chrono::steady_clock::time_point>> read_leases_;
  std::vector<KvEvent> pending_events_;

  std::atomic<uint64_t> next_slot_id_{1};

  std::thread reaper_thread_;
  std::atomic<bool> reaper_running_{false};
  std::mutex reaper_cv_mutex_;
  std::condition_variable reaper_cv_;
};

}  // namespace mori::umbp
