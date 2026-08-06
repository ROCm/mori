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
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mori/io/common.hpp"         // MemoryDesc (push fan-out destinations)
#include "umbp/distributed/config.h"  // PeerSsdConfig
#include "umbp/distributed/peer/owned_location_source.h"
#include "umbp/distributed/types.h"

namespace mori::io {
class IOEngine;
}

namespace mori::umbp {

class TierBackend;  // umbp/local/tiers/tier_backend.h — kept out of this header.

enum class SsdReadStatus { kOk, kNotFound, kSizeTooLarge, kError };
struct SsdReadOutcome {
  SsdReadStatus status = SsdReadStatus::kError;
  size_t size = 0;
};

// Per-key RDMA plumbing for the single-flight fan-out push (see
// PrepareReadBatch).  Every field is per-key and supplied by every caller,
// because a caller does not know before the call which of its keys it will lead
// and which it will follow:
//
//   * local_desc/local_offset describe where THIS caller's own dsts[i] lives.
//     Used when this caller ends up LEADING: they are the RDMA source of the
//     push to its followers.
//   * remote_desc/remote_offset describe where THIS caller ultimately wants the
//     bytes.  Used when this caller ends up FOLLOWING: they are the RDMA
//     destination its leader writes into, instead of memcpy-ing into dsts[i].
//
// push_eligible == false (the default) opts the key out entirely and is the
// exact pre-push behaviour: the caller is memcpy-served into dsts[i] and pulls
// from there.  Anything unresolvable upstream (unknown node, stale region id,
// expired registration, no IO engine) must arrive here as push_eligible=false
// rather than as an invalid descriptor.
struct ReadPushSlot {
  bool push_eligible = false;
  mori::io::MemoryDesc local_desc{};
  size_t local_offset = 0;
  mori::io::MemoryDesc remote_desc{};
  size_t remote_offset = 0;
};

// Peer-side owner of the local SSD tier in the master-as-advisor design.
// Single responsibility: manage one SSD TierBackend + the key->SSD-location
// map + capacity + the owned-location event outbox + the read-prepare and
// local-eviction paths.  It deliberately reuses ONLY the low-level TierBackend
// (SSDTier); it must NOT pull in LocalStorageManager / LocalBlockIndex (which
// carry their own DRAM tier + demote/promote) — peer DRAM is owned by
// PeerDramAllocator and two DRAM concepts would scramble ownership.
class PeerSsdManager : public OwnedLocationSource {
 public:
  // @p io_engine (non-owning, may be null) enables the RDMA-WRITE fan-out push
  // in PrepareReadBatch.  Null — the default, and what every existing caller and
  // test gets — leaves the manager memcpy-only, byte-for-byte the prior
  // behaviour.  PoolClient owns the engine and constructs it before this.
  explicit PeerSsdManager(const PeerSsdConfig& cfg, mori::io::IOEngine* io_engine = nullptr);

  // Test-only: inject a ready-made backend and explicit watermarks so unit
  // tests can drive eviction with a controllable (e.g. blocking) fake backend.
  // Production code must use the config constructor.
  PeerSsdManager(std::unique_ptr<TierBackend> backend, double high_watermark, double low_watermark,
                 bool single_flight = true, mori::io::IOEngine* io_engine = nullptr);

  ~PeerSsdManager() override;

  PeerSsdManager(const PeerSsdManager&) = delete;
  PeerSsdManager& operator=(const PeerSsdManager&) = delete;

  // {used_bytes, total_bytes}.  Reported via heartbeat as TierType::SSD.
  std::pair<size_t, size_t> Capacity() const;

  bool Exists(const std::string& key) const;

  // Write the key's bytes (assembled from possibly non-contiguous DRAM source
  // segments) to the SSD backend.  On success records the SSD location and
  // queues an ADD SSD event; on failure records nothing and queues nothing
  // (best-effort clean).
  bool Write(const std::string& key, const std::vector<std::pair<const void*, size_t>>& segments,
             size_t total_size);

  // Batched form of Write for contiguous sources — the direct-SSD put path.
  // One dedup pass, ONE backend BatchWrite (which a multi-drive ShardedSsdTier
  // fans across every drive in parallel), then one recording pass.  This is
  // what makes an SSD-target BatchPut scale with drive count instead of
  // serializing key-by-key; a per-key Write loop would idle every drive but one.
  //
  // Result length and order match @p keys.  An already-owned key reports true
  // without any device IO (LRU refreshed).  On a partial failure the failed keys
  // are retried once after a single eviction round, so a batch that trips the
  // high watermark mid-way still lands.
  std::vector<bool> WriteBatch(const std::vector<std::string>& keys,
                               const std::vector<const void*>& srcs,
                               const std::vector<size_t>& sizes);

  // Local eviction of a single key.  Read priority: a key with an
  // in-flight PrepareRead (inflight_reads_ > 0) is NOT evicted (returns false).
  // Concurrency: marks the key in evicting_ under the lock, runs the backend
  // evict outside the lock, and only on backend success removes owned_/lru_ and
  // queues a REMOVE SSD event (so REMOVE is never emitted while the bytes still
  // exist, and two workers cannot double-evict the same victim).
  //
  // Does NOT itself hold eviction_mu_ (EvictToLowWatermark already holds it
  // while looping over victims, so taking it here would deadlock).  The normal
  // caller is EvictToLowWatermark; a direct caller must not run concurrently
  // with ClearLocal (production relies on PoolClient::Clear quiescing the copy
  // pipeline first, so no eviction is in flight during a Clear).
  bool Evict(const std::string& key);

  // Local LRU victim selection (oldest first), skipping keys that are
  // being read (inflight_reads_ > 0) or already being evicted (evicting_).
  // Accumulates sizes until >= bytes_to_free; returns fewer if not enough
  // free-able keys exist (never blocks).
  //
  // Not pluggable yet (only master-side eviction is).  A real SSD plugin must
  // own the per-algorithm state that lives here in lru_/owned_, not just the
  // selection step; target design is a stateful policy with
  // OnAdd/OnTouch/OnRemove/Clear hooks plus a SelectVictims called under mutex_.
  std::vector<std::string> SelectVictims(size_t bytes_to_free);

  // Distributed Clear: drop the logical owned-location map + undrained events,
  // then delete the physical SSD bytes (a user Clear means the cache is no
  // longer wanted).  Read priority: clears the logical map first (so new reads
  // immediately miss with kNotFound), waits for any in-flight PrepareRead to
  // finish (SSD reads cannot be safely aborted), and only then wipes the
  // backend.  Serializes against eviction rounds via eviction_mu_.
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
  // @p dsts and @p caps are parallel to @p keys; the result is too.
  //
  // Fan-out push (opt-in): when @p push_slots is non-empty (parallel to @p keys)
  // and an IOEngine was supplied, a key this call FOLLOWS whose slot is
  // push_eligible is served by the leader RDMA-WRITE-ing straight into
  // push_slots[i].remote_desc instead of memcpy-ing into dsts[i].  @p pushed (if
  // non-null, sized to @p keys) reports which keys took that path: those keys'
  // dsts[i] were never touched, so the caller must not pull from them — and the
  // staging slot it claimed for them holds nothing and can be released.
  // Anything not push-eligible, unresolvable, or not followed falls through to
  // the unchanged memcpy path.
  std::vector<SsdReadOutcome> PrepareReadBatch(const std::vector<std::string>& keys,
                                               const std::vector<void*>& dsts,
                                               const std::vector<size_t>& caps,
                                               const std::vector<ReadPushSlot>& push_slots = {},
                                               std::vector<bool>* pushed = nullptr);

  // OwnedLocationSource — all events carry TierType::SSD.
  std::vector<KvEvent> DrainPendingEvents() override;
  std::vector<KvEvent> SnapshotOwnedKeys() const override;

  // Full-sync snapshot that also atomically drops the event outbox under the
  // same lock.  See OwnedLocationSource.
  std::vector<KvEvent> SnapshotOwnedKeysForFullSync() override;

  // Crash-restart leftover policy (discard): after a crash owned_ is empty but
  // physical SSD bytes may remain, diverging used capacity from owned_.  This
  // best-effort wipes them at startup for a clean, consistent tier (cache is
  // re-fetchable).  Call before the copy pipeline starts / before any IO (not
  // synchronized against Write/PrepareRead).  No-op when SSD is disabled.
  void DiscardLeftoverOnStartup();

  // Prometheus-only observability snapshots (see metrics_ below); sampled once
  // per metrics tick by PublishSsdMetrics(), never drive correctness.
  uint64_t ReadOk() const { return metrics_.read_ok.load(std::memory_order_relaxed); }
  uint64_t ReadNotFound() const { return metrics_.read_not_found.load(std::memory_order_relaxed); }
  uint64_t ReadSizeTooLarge() const {
    return metrics_.read_size_too_large.load(std::memory_order_relaxed);
  }
  uint64_t ReadError() const { return metrics_.read_error.load(std::memory_order_relaxed); }
  // Single-flight probe (see read_lead / read_dup below).
  uint64_t ReadLead() const { return metrics_.read_lead.load(std::memory_order_relaxed); }
  uint64_t ReadDup() const { return metrics_.read_dup.load(std::memory_order_relaxed); }
  uint64_t ReadMerged() const { return metrics_.read_merged.load(std::memory_order_relaxed); }
  // Of the merged (follower-served) reads, those delivered by RDMA WRITE
  // straight into the requester's buffer rather than by memcpy into its staging
  // slot.  read_pushed_failed counts push transfers that were posted and did not
  // complete — those followers get a read error, nobody else is affected.
  uint64_t ReadPushed() const { return metrics_.read_pushed.load(std::memory_order_relaxed); }
  uint64_t ReadPushFailed() const {
    return metrics_.read_push_failed.load(std::memory_order_relaxed);
  }
  // Byte counters for SSD IO bandwidth (rate() in Grafana = bytes/s).
  uint64_t CopyBytes() const { return metrics_.copy_bytes.load(std::memory_order_relaxed); }
  uint64_t ReadBytes() const { return metrics_.read_bytes.load(std::memory_order_relaxed); }
  uint64_t EvictionRounds() const { return metrics_.evict_rounds.load(std::memory_order_relaxed); }
  uint64_t EvictionVictims() const {
    return metrics_.evict_victims.load(std::memory_order_relaxed);
  }
  uint64_t EvictionBytesFreed() const {
    return metrics_.evict_bytes_freed.load(std::memory_order_relaxed);
  }
  uint64_t EvictionBackendFailures() const {
    return metrics_.evict_backend_failures.load(std::memory_order_relaxed);
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

  // Build the full ADD list for every owned SSD key.  Caller MUST hold mutex_.
  std::vector<KvEvent> SnapshotOwnedKeysLocked() const;

  // Evict oldest keys until used <= low_watermark * total.  Runs the backend
  // IO outside mutex_ (via Evict); serialized by eviction_mu_ so concurrent
  // copy workers do not run overlapping eviction rounds (and never over-evict).
  void EvictToLowWatermark();

  // Serializes eviction rounds (EvictToLowWatermark) and excludes ClearLocal.
  // Always acquired BEFORE mutex_ to keep a single lock order.
  std::mutex eviction_mu_;

  mutable std::mutex mutex_;
  std::unique_ptr<TierBackend> backend_;  // null when cfg.enabled == false
  double high_watermark_ = 0.9;
  double low_watermark_ = 0.7;

  // key -> {size, lru position}.  The authoritative owned-location map.
  std::unordered_map<std::string, OwnedEntry> owned_;
  std::list<std::string> lru_;  // front = most-recently-used, back = LRU
  std::vector<KvEvent> pending_events_;

  // Per-key state for one or more concurrent reads.  Serves two purposes:
  //
  //  1. Read priority / eviction safety (the original role): an entry exists
  //     exactly while refs > 0, and eviction skips any key with a live read.
  //
  //  2. Single flight: the first requester of a key becomes the *leader* and is
  //     the only one to touch the drive.  Requesters that arrive while it is
  //     still reading become *followers* — they register a destination buffer,
  //     block on cv, and are filled from the leader's buffer when it completes:
  //     by memcpy into their own staging slot, or (when they registered a push
  //     destination and the leader can post RDMA) by a WRITE straight into their
  //     final buffer, which skips the copy and the pull entirely.  This is what
  //     collapses the tp_size-way read fan-out that MLA + TP produces (every
  //     attention-TP rank GETs a byte-identical key).
  //
  // A requester that arrives after the leader has sealed its follower snapshot
  // is *independent*: it missed the merge window and issues its own read, which
  // is exactly the pre-single-flight behaviour and always correct.
  //
  // Buffer lifetime: both leader and followers are blocked inside
  // PrepareRead/PrepareReadBatch for the whole copy, so neither side's staging
  // slot can be recycled underneath the memcpy.
  // One leader's read, from claim to fan-out.  Followers capture a shared_ptr
  // at attach time and wait on *this* object, never on mutable per-key state:
  // otherwise a leader that finishes and is immediately replaced by a new
  // leader on the same key would reset the flag a not-yet-woken follower is
  // waiting on, and that follower would silently start tracking a read whose
  // fan-out list it is not in.  Holding the shared_ptr also lets the episode
  // outlive its slot, so the last follower to wake still finds valid state.
  // One follower attached to a leader's read.
  //
  // Delivery is either a memcpy into {dst, cap} (the original path) or an RDMA
  // WRITE into {remote, remote_offset} capped at `cap` (the push path).
  //
  // `ok` is PER FOLLOWER, not shared.  With memcpy a shared episode-level bit
  // was safe: copying from valid host memory cannot fail for one follower and
  // succeed for another.  An RDMA WRITE can — a remote QP can be down or an
  // rkey stale for exactly one destination — so one push-follower's failure
  // must not be allowed to taint the leader's own result, the other
  // push-followers, or the memcpy-followers on the same episode.
  //
  // Held by shared_ptr on both sides: the leader swaps the follower list out of
  // the episode when it seals, so a follower cannot read its own outcome out of
  // that list afterwards — it keeps its own handle instead.
  struct FollowerTarget {
    bool wants_push = false;        // requester registered a push destination
    void* dst = nullptr;            // memcpy path; ALWAYS valid, push or not
    size_t cap = 0;                 // byte cap for both paths
    mori::io::MemoryDesc remote{};  // push path
    size_t remote_offset = 0;
    // Both written by the leader before it sets episode->done, and read by the
    // follower after it wakes.  `pushed` reports what the leader ACTUALLY did,
    // not what the follower asked for: a leader with no IO engine or no
    // registered source (e.g. the single-key PrepareRead path, which never
    // pushes) serves a push-wanting follower by memcpy into `dst` and reports
    // pushed=false.  The follower must never assume delivery it did not get.
    bool ok = false;
    bool pushed = false;
  };

  // Note there is deliberately no episode-level `ok`.  The leader's read outcome
  // reaches each follower through that follower's own FollowerTarget::ok, so a
  // per-destination delivery failure cannot be confused with the read result —
  // see FollowerTarget.  `done` is the only thing followers wait on.
  struct ReadEpisode {
    bool sealed = false;  // snapshot taken; late arrivals must not attach
    bool done = false;    // leader finished (read + fan-out)
    std::vector<std::shared_ptr<FollowerTarget>> followers;
    std::condition_variable cv;  // waited on under mutex_
  };

  struct InflightRead {
    int refs = 0;  // leader + followers + independents holding this key
    // Non-null exactly while a leader is mid-read; the leader clears it on
    // completion so the next arrival starts a fresh episode.
    std::shared_ptr<ReadEpisode> episode;
  };

  //   inflight_reads_: key -> InflightRead (entry exists only while refs > 0).
  //   evicting_: keys currently inside Evict's backend-evict window; new reads
  //     of these miss (kNotFound) and SelectVictims skips them.
  std::unordered_map<std::string, std::unique_ptr<InflightRead>> inflight_reads_;
  std::unordered_set<std::string> evicting_;
  std::condition_variable reads_drained_cv_;  // notified when inflight_reads_ empties

  // Coalesce concurrent same-key reads (UMBPSsdConfig::single_flight_reads).
  // Off => every requester reads the device, and read_dup still reports the
  // headroom that turning it on would recover.
  bool single_flight_ = true;

  // Non-owning; null disables the fan-out push entirely (memcpy-only).  The peer
  // is an RDMA *initiator* through this handle — the one place in UMBP where
  // that direction is used.
  mori::io::IOEngine* io_engine_ = nullptr;

  // Post one BatchWrite for every push-follower in @p fanout and wait each
  // posted transfer INDIVIDUALLY, setting that follower's own `ok`.  Returns the
  // number of followers actually pushed.  Never throws to the caller: a failure
  // to post leaves the affected followers ok=false, which the caller surfaces as
  // that follower's own read error.
  size_t PushFanout(const std::vector<std::shared_ptr<FollowerTarget>>& fanout,
                    const ReadPushSlot& leader_slot, size_t size, uint64_t* pushed_bytes);

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
    // Single-flight probe.  A read that finds inflight_reads_[key] == 0 "leads"
    // (it is the one that would issue device IO under single-flight); one that
    // finds > 0 is a "dup" — a concurrent request for the identical key that
    // could have attached to the in-flight read instead of issuing its own.
    // Today every requester reads, so read_dup counts device reads that a
    // single-flight merge would eliminate.  MLA + TP is the case that produces
    // them (all attention-TP ranks GET a byte-identical key); MHA keys carry a
    // per-rank suffix and should never dup.
    std::atomic<uint64_t> read_lead{0};
    std::atomic<uint64_t> read_dup{0};
    // Of the read_dup requests, those actually served by memcpy from a leader
    // (single_flight_ on and the merge window not missed).  read_dup minus this
    // is the residue that still hit the drive.
    std::atomic<uint64_t> read_merged{0};
    // Of read_merged, the followers served by RDMA WRITE instead of memcpy, and
    // the subset whose write did not complete.
    std::atomic<uint64_t> read_pushed{0};
    std::atomic<uint64_t> read_push_failed{0};
    std::atomic<uint64_t> copy_bytes{0};  // bytes written to SSD (write IO)
    std::atomic<uint64_t> read_bytes{0};  // bytes read from SSD (read IO)
    std::atomic<uint64_t> evict_rounds{0};
    std::atomic<uint64_t> evict_victims{0};
    std::atomic<uint64_t> evict_bytes_freed{0};
    std::atomic<uint64_t> evict_backend_failures{0};
  };
  MetricsCounters metrics_;
};

}  // namespace mori::umbp
