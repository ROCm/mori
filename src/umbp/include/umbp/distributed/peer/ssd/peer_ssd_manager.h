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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "umbp/distributed/config.h"  // PeerSsdConfig
#include "umbp/distributed/types.h"
// RecordLocation is returned by value from LocateRecord, so the tier interface
// is included here rather than forward-declared. peer/ssd is the SSD adapter
// that already builds on the local tier (Phase 5 lint exempts this directory),
// so this crosses no new boundary.
#include "umbp/local/tiers/tier_backend.h"

namespace mori::umbp {

enum class SsdReadStatus { kOk, kNotFound, kSizeTooLarge, kError };
struct SsdReadOutcome {
  SsdReadStatus status = SsdReadStatus::kError;
  size_t size = 0;
};

// Peer-side owner of the local SSD tier: one SSD TierBackend + the
// key->SSD-location map, capacity, the owned-location event outbox, the
// read-prepare path, and Evict().  Reached through SsdBackend.  It never
// decides to evict; it only offers candidates, in the tier's order when the
// tier keeps one and in its own LRU otherwise.
//
// It reuses ONLY the low-level TierBackend (SSDTier).  Peer DRAM is owned by
// PageBackend, and a second thing here that also held DRAM and demoted between
// the two would scramble that ownership — which is what the deleted
// LocalStorageManager did, and part of why it went.
class PeerSsdManager {
 public:
  explicit PeerSsdManager(const PeerSsdConfig& cfg);

  // Test-only: inject a fake backend.
  explicit PeerSsdManager(std::unique_ptr<TierBackend> backend, bool single_flight = true);

  ~PeerSsdManager();

  PeerSsdManager(const PeerSsdManager&) = delete;
  PeerSsdManager& operator=(const PeerSsdManager&) = delete;

  // {used_bytes, total_bytes}.  Reported via heartbeat as TierType::SSD.
  std::pair<size_t, size_t> Capacity() const;

  bool Exists(const std::string& key) const;

  // Stored size of `key`, or 0 if unknown.  Needed by SsdBackend::BatchResolve:
  // PrepareRead wants a staging buffer chosen BEFORE the read, so the reader has
  // to know how big the key is while it still only holds the key's name.
  // `owned_` has carried this all along; only the accessor was missing.
  uint64_t SizeOf(const std::string& key) const;

  // Hold a key against eviction across a peer-local migration.  Uses the same
  // refcount as active reads so eviction has one protection rule.
  bool PinForMigration(const std::string& key);
  void UnpinForMigration(const std::string& key);

  // Physical location of |key|'s value for a zero-copy GDS read, or nullopt when
  // the key is unknown or being evicted (mirroring PrepareRead's kNotFound).
  // Forwards to the SSD tier; does no device IO.
  std::optional<RecordLocation> LocateRecord(const std::string& key) const;

  // Batched form of Write for contiguous sources — the SsdBackend::BatchCommit
  // path.  One dedup pass, ONE backend BatchWrite (which a multi-drive
  // ShardedSsdTier fans across every drive in parallel), then one recording
  // pass.  This is what makes an SSD-target batch put scale with drive count
  // instead of serializing key-by-key; a per-key Write loop would idle every
  // drive but one.
  //
  // Result length and order match @p keys.  An already-owned key reports true
  // without any device IO (LRU refreshed).  A key with no room fails; this
  // never evicts to make room.
  std::vector<bool> WriteBatch(const std::vector<std::string>& keys,
                               const std::vector<const void*>& srcs,
                               const std::vector<size_t>& sizes);

  // Write-in-place for a local file engine: reserve room, fill
  // `location`, then CommitReserved records the key (ADD event) or
  // AbortReserved drops it.  nullopt when the tier cannot (no file, no room).
  std::optional<RecordReservation> Reserve(const std::string& key, uint64_t size);
  bool CommitReserved(const std::string& key, const RecordReservation& r);
  void AbortReserved(const std::string& key, const RecordReservation& r);

  // Frees one key; false if it is being read or pinned (inflight_reads_ > 0).
  // Marks the key in evicting_ under the lock, evicts outside it, and only on
  // success removes owned_/lru_ and queues a REMOVE (never while the bytes
  // still exist; no double evict).  PeerPool::Evict holds the lifecycle lock,
  // so this never overlaps a Clear.
  bool Evict(const std::string& key);

  // Coldest first, skipping keys being read, pinned or evicted.  Frees nothing.
  // When the tier orders its own eviction (OrdersEviction()), its order is used
  // and each offer carries the bytes the tier credits it with; otherwise the
  // order is this class's LRU and an offer carries the key's size.
  std::vector<EvictionOffer> EvictionCandidates(size_t max_count);

  // True when only the tier's own order frees space (the segment-log SSD tier
  // returns disk a whole segment at a time), so no other ranking should be
  // used to pick its victims.
  bool OrdersEviction() const;

  // Forwards to the tier: see TierBackend::SetRetiredFileHook.
  void SetRetiredFileHook(std::function<void(int fd)> on_close, std::chrono::milliseconds grace);
  TierSpaceStats SpaceStats() const { return backend_ ? backend_->SpaceStats() : TierSpaceStats{}; }

  // Distributed Clear: drop the logical owned-location map + undrained events,
  // then delete the physical SSD bytes (a user Clear means the cache is no
  // longer wanted).  Read priority: clears the logical map first (so new reads
  // immediately miss with kNotFound), waits for any in-flight PrepareRead to
  // finish (SSD reads cannot be safely aborted), and only then wipes the
  // backend.  No eviction runs concurrently: see Evict.
  // Precondition: callers (PoolClient::Clear) MUST quiesce the SSD copy
  // pipeline first so no in-flight copy re-populates owned_ right after this
  // returns.  Crash-restart leftover (metadata gone, files remain) is a
  // known follow-up.
  void ClearLocal();

  // Read the key's bytes into a staging slot.  Returns kNotFound when
  // the key is unknown OR currently being evicted (evicting_); kSizeTooLarge
  // when the key is bigger than staging_cap; otherwise reads the bytes and
  // returns kOk.  The backend IO runs outside the lock; the key is marked
  // in-flight (inflight_reads_) across that window so eviction skips it.
  SsdReadOutcome PrepareRead(const std::string& key, void* staging_ptr, size_t staging_cap);

  // Batched form of PrepareRead: resolve + mark every key in flight under one
  // lock, issue ONE backend BatchReadIntoPtr (fanned across drives by
  // ShardedSsdTier), then release the marks in one pass.  Per-key semantics are
  // identical to PrepareRead — including kNotFound for an evicting key and
  // kSizeTooLarge rejected before any device IO.
  //
  // This is the throughput path: the per-key form costs one device read per
  // key and leaves every drive but one idle, so a multi-drive tier can only be
  // saturated through here.
  //
  // @p dsts and @p caps are parallel to @p keys; the result is too.  A key
  // whose @p extents entry has a nonzero length reads only those object bytes
  // (unverified: a record's CRC covers the whole value); its outcome `size` is
  // still the object's, and an extent outside the object is kError.  Single
  // flight attaches a reader only to a read that covers all of its bytes.
  std::vector<SsdReadOutcome> PrepareReadBatch(const std::vector<std::string>& keys,
                                               const std::vector<void*>& dsts,
                                               const std::vector<size_t>& caps,
                                               const std::vector<ObjectExtent>* extents = nullptr);

  // Whether PrepareReadBatch can take extents: the tier reads sub-object ranges.
  bool SupportsRangedRead() const;

  // Same shape as OwnedLocationSource (see the class comment above) — all
  // events carry TierType::SSD.
  std::vector<KvEvent> DrainPendingEvents();

  // False stops recording ADD/REMOVE and drops what is queued.  The outbox has
  // exactly one consumer -- MasterClient's heartbeat -- so a node with no
  // master would otherwise grow it by one entry per written key forever.
  void SetEventPublishing(bool enabled);
  std::vector<KvEvent> SnapshotOwnedKeys() const;
  size_t OwnedKeyCount() const;

  // Full-sync snapshot that also atomically drops the event outbox under the
  // same lock.
  std::vector<KvEvent> SnapshotOwnedKeysForFullSync();

  // Crash-restart leftover policy (discard): after a crash owned_ is empty but
  // physical SSD bytes may remain, diverging used capacity from owned_.  This
  // best-effort wipes them at startup for a clean, consistent tier (cache is
  // re-fetchable).  Call before the copy pipeline starts / before any IO (not
  // synchronized against Write/PrepareRead).  No-op when SSD is disabled.
  void DiscardLeftoverOnStartup();

  // Prometheus-only observability snapshots (see metrics_ below), sampled by
  // SsdBackend::SampleMetrics; never drive correctness.
  uint64_t ReadOk() const { return metrics_.read_ok.load(std::memory_order_relaxed); }
  uint64_t ReadNotFound() const { return metrics_.read_not_found.load(std::memory_order_relaxed); }
  uint64_t ReadSizeTooLarge() const {
    return metrics_.read_size_too_large.load(std::memory_order_relaxed);
  }
  uint64_t ReadError() const { return metrics_.read_error.load(std::memory_order_relaxed); }
  // Single-flight coalescing: leads issue device IO, dups are the concurrent
  // same-key requests, merged is the subset of dups actually served from a
  // leader's buffer.  dup - merged is the residue that still hit the drive.
  uint64_t ReadLead() const { return metrics_.read_lead.load(std::memory_order_relaxed); }
  uint64_t ReadDup() const { return metrics_.read_dup.load(std::memory_order_relaxed); }
  uint64_t ReadMerged() const { return metrics_.read_merged.load(std::memory_order_relaxed); }
  // Byte counters for SSD IO bandwidth (rate() in Grafana = bytes/s).
  uint64_t CopyBytes() const { return metrics_.copy_bytes.load(std::memory_order_relaxed); }
  uint64_t ReadBytes() const { return metrics_.read_bytes.load(std::memory_order_relaxed); }
  uint64_t EvictionBackendFailures() const {
    return metrics_.evict_backend_failures.load(std::memory_order_relaxed);
  }
  // Records written in place through a reservation, and reservations given up.
  uint64_t WritesInPlace() const {
    return metrics_.writes_in_place.load(std::memory_order_relaxed);
  }
  uint64_t WriteInPlaceAborts() const {
    return metrics_.write_in_place_aborts.load(std::memory_order_relaxed);
  }

 private:
  // One owned SSD key: its size plus a hook into the LRU recency list so that
  // a touch is an O(1) splice and a victim lookup is an O(1) walk from the tail.
  struct OwnedEntry {
    uint64_t size = 0;
    std::list<std::string>::iterator lru_it;  // position of this key in lru_
  };

  // Splice |key| to the MRU (front) of the recency list.  Caller holds mutex_
  // and must have already inserted the key into owned_.
  void TouchLocked(const std::string& key);

  // A write of `key` landed: own it (ADD event), or just touch it if a
  // concurrent writer already did.  Caller holds mutex_.
  void RecordWrittenLocked(const std::string& key, uint64_t size);

  // Build the full ADD list for every owned SSD key.  Caller MUST hold mutex_.
  std::vector<KvEvent> SnapshotOwnedKeysLocked() const;

  mutable std::mutex mutex_;
  std::unique_ptr<TierBackend> backend_;  // null when cfg.enabled == false

  // key -> {size, lru position}.  The authoritative owned-location map.
  std::unordered_map<std::string, OwnedEntry> owned_;
  std::list<std::string> lru_;  // front = most-recently-used, back = LRU
  std::vector<KvEvent> pending_events_;
  // Guarded by mutex_ alongside pending_events_.  See SetEventPublishing.
  bool event_publishing_ = true;

  // One follower attached to a leader's read.  Delivery is a memcpy into
  // {dst, cap} out of the leader's staging slot.
  //
  // Held by shared_ptr on both sides: the leader swaps the follower list out of
  // the episode when it seals, so a follower cannot read its own outcome out of
  // that list afterwards — it keeps its own handle instead.
  //
  // Buffer lifetime: both leader and followers are blocked inside
  // PrepareRead/PrepareReadBatch for the whole copy, so neither side's staging
  // slot can be recycled underneath the memcpy.
  struct FollowerTarget {
    void* dst = nullptr;
    // Object bytes [offset, offset + length) the leader copies out of its buffer.
    uint64_t offset = 0;
    uint64_t length = 0;
    // Written by the leader before it sets episode->done, read by the follower
    // after it wakes.
    bool ok = false;
  };

  // One leader's read, from claim to fan-out.  Followers capture a shared_ptr
  // at attach time and wait on *this* object, never on mutable per-key state:
  // otherwise a leader that finishes and is immediately replaced by a new
  // leader on the same key would reset the flag a not-yet-woken follower is
  // waiting on, and that follower would silently start tracking a read whose
  // fan-out list it is not in.  Holding the shared_ptr also lets the episode
  // outlive its slot, so the last follower to wake still finds valid state.
  //
  // Note there is deliberately no episode-level `ok`: the leader's read outcome
  // reaches each follower through that follower's own FollowerTarget::ok.
  // `done` is the only thing followers wait on.
  struct ReadEpisode {
    bool sealed = false;  // snapshot taken; late arrivals must not attach
    bool done = false;    // leader finished (read + fan-out)
    // Object bytes the leader reads; only a reader whose bytes lie inside attaches.
    uint64_t offset = 0;
    uint64_t length = 0;
    bool Covers(uint64_t want_offset, uint64_t want_length) const {
      return want_offset >= offset && want_offset - offset <= length &&
             want_length <= length - (want_offset - offset);
    }
    std::vector<std::shared_ptr<FollowerTarget>> followers;
    std::condition_variable cv;  // waited on under mutex_
  };

  struct InflightRead {
    int refs = 0;  // leader + followers + independents holding this key
    // Non-null exactly while a leader is mid-read; the leader clears it on
    // completion so the next arrival starts a fresh episode.
    std::shared_ptr<ReadEpisode> episode;
  };

  // Read priority + eviction coordination (all guarded by mutex_):
  //   inflight_reads_: key -> InflightRead (entry exists only while refs > 0).
  //     Eviction skips keys with a live read.
  //   evicting_: keys currently inside Evict's backend-evict window; new reads
  //     of these miss (kNotFound) and EvictionCandidates skips them.
  std::unordered_map<std::string, std::unique_ptr<InflightRead>> inflight_reads_;
  std::unordered_set<std::string> evicting_;
  std::condition_variable reads_drained_cv_;  // notified when inflight_reads_ empties

  // Coalesce concurrent same-key reads (UMBPSsdConfig::single_flight_reads).
  // Off => every requester reads the device, and read_dup still reports the
  // headroom that turning it on would recover.
  bool single_flight_ = true;

  // Cap on how long a follower waits for its leader.  A wedged or crashed
  // leader must surface as a read error rather than hanging the RPC; the caller
  // treats it as a miss and refetches from source.
  static constexpr int kFollowerWaitSeconds = 30;

  // Prometheus-only observability counters: relaxed atomics bumped at discrete
  // events, read once per metrics tick.  NOT correctness state
  // (owned_/lru_/inflight_reads_ are authoritative); deletable with the provider.
  struct MetricsCounters {
    std::atomic<uint64_t> read_ok{0};
    std::atomic<uint64_t> read_not_found{0};
    std::atomic<uint64_t> read_size_too_large{0};
    std::atomic<uint64_t> read_error{0};
    // Single-flight accounting.  A requester that finds no in-flight read of
    // the key is a "lead" (it is the one that issues device IO); one that finds
    // an in-flight read is a "dup" — a concurrent request for the identical key
    // that could attach to it instead.  MLA + TP is the case that produces them
    // (all attention-TP ranks GET a byte-identical key); MHA keys carry a
    // per-rank suffix and should never dup.
    std::atomic<uint64_t> read_lead{0};
    std::atomic<uint64_t> read_dup{0};
    // Of the read_dup requests, those actually served by memcpy from a leader
    // (single_flight_ on and the merge window not missed).  read_dup minus this
    // is the residue that still hit the drive.
    std::atomic<uint64_t> read_merged{0};
    std::atomic<uint64_t> copy_bytes{0};  // bytes written to SSD (write IO)
    std::atomic<uint64_t> read_bytes{0};  // bytes read from SSD (read IO)
    std::atomic<uint64_t> evict_backend_failures{0};
    std::atomic<uint64_t> writes_in_place{0};
    std::atomic<uint64_t> write_in_place_aborts{0};
  };
  MetricsCounters metrics_;
};

}  // namespace mori::umbp
