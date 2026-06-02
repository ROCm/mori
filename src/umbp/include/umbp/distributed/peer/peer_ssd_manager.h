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

#include "umbp/distributed/config.h"  // PeerSsdConfig
#include "umbp/distributed/peer/owned_location_source.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

class TierBackend;  // umbp/local/tiers/tier_backend.h — kept out of this header.

enum class SsdReadStatus { kOk, kNotFound, kSizeTooLarge, kError };
struct SsdReadOutcome {
  SsdReadStatus status = SsdReadStatus::kError;
  size_t size = 0;
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
  explicit PeerSsdManager(const PeerSsdConfig& cfg);

  // Test-only: inject a ready-made backend and explicit watermarks so unit
  // tests can drive eviction with a controllable (e.g. blocking) fake backend.
  // Production code must use the config constructor.
  PeerSsdManager(std::unique_ptr<TierBackend> backend, double high_watermark, double low_watermark);

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

  // OwnedLocationSource — all events carry TierType::SSD.
  std::vector<KvEvent> DrainPendingEvents() override;
  std::vector<KvEvent> SnapshotOwnedKeys() const override;

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

  // Read priority + eviction coordination (all guarded by mutex_):
  //   inflight_reads_: key -> active PrepareRead count (entry exists only while
  //     > 0).  Eviction skips keys with a live read.
  //   evicting_: keys currently inside Evict's backend-evict window; new reads
  //     of these miss (kNotFound) and SelectVictims skips them.
  std::unordered_map<std::string, int> inflight_reads_;
  std::unordered_set<std::string> evicting_;
  std::condition_variable reads_drained_cv_;  // notified when inflight_reads_ empties

  // Prometheus-only observability counters: relaxed atomics bumped at discrete
  // events, read once per metrics tick.  NOT correctness state
  // (owned_/lru_/inflight_reads_ are authoritative); deletable with the provider.
  struct MetricsCounters {
    std::atomic<uint64_t> read_ok{0};
    std::atomic<uint64_t> read_not_found{0};
    std::atomic<uint64_t> read_size_too_large{0};
    std::atomic<uint64_t> read_error{0};
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
