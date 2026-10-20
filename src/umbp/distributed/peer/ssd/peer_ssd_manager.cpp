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
#include "umbp/distributed/peer/ssd/peer_ssd_manager.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/ssd_perf.h"
#include "umbp/local/tiers/spdk_proxy_tier.h"
#include "umbp/local/tiers/ssd_backend_factory.h"
#include "umbp/local/tiers/ssd_tier.h"
#include "umbp/local/tiers/tier_backend.h"

namespace mori::umbp {

// ---------------------------------------------------------------------------
//  Construction
// ---------------------------------------------------------------------------

PeerSsdManager::PeerSsdManager(const PeerSsdConfig& cfg) {
  if (!cfg.enabled) {
    MORI_UMBP_INFO("[PeerSsdManager] constructed disabled (no SSD backend)");
    return;
  }
  const auto& ssd = cfg.ssd;
  // `ssd`'s watermarks drive PeerPool's local eviction, not this class.
  single_flight_ = ssd.single_flight_reads;

  // Explicit backend selection — an unknown value is a configuration error, not
  // a reason to silently fall back to the file backend.
  if (ssd.ssd_backend == "file") {
    backend_ = MakeFileSsdBackend(ssd);
    MORI_UMBP_INFO("[PeerSsdManager] SSDTier ready dir={} capacity={}B", ssd.storage_dir,
                   ssd.capacity_bytes);
    return;
  }

  // The distributed peer shares one physical device across processes, so SPDK
  // is reached through the spdk_proxy daemon, never the single-process direct
  // SpdkSsdTier.  Both "spdk" and "spdk_proxy" therefore map to SpdkProxyTier:
  // "spdk" here means "use SPDK via the proxy".
  if (ssd.ssd_backend == "spdk" || ssd.ssd_backend == "spdk_proxy") {
    MORI_UMBP_INFO("[PeerSsdManager] distributed ssd_backend={} uses SpdkProxyTier",
                   ssd.ssd_backend);
    auto proxy = std::make_unique<SpdkProxyTier>(ssd);
    // PeerSsdManager only connects to an already-ready proxy; it does not spawn
    // the daemon.  A connect failure with an explicit SPDK config is fatal — we
    // must not silently disable SSD or fall back to POSIX.
    if (!proxy->IsValid()) {
      throw std::runtime_error("[PeerSsdManager] ssd_backend=" + ssd.ssd_backend +
                               ": SPDK proxy connect failed (shm=" + ssd.spdk_proxy_shm_name +
                               "). Ensure the spdk_proxy daemon is running and READY.");
    }
    backend_ = std::move(proxy);
    MORI_UMBP_INFO("[PeerSsdManager] SpdkProxyTier ready (shm={})", ssd.spdk_proxy_shm_name);
    return;
  }

  throw std::runtime_error("[PeerSsdManager] unknown ssd_backend='" + ssd.ssd_backend +
                           "' (expected one of: file, spdk, spdk_proxy)");
}

PeerSsdManager::PeerSsdManager(std::unique_ptr<TierBackend> backend, bool single_flight)
    : backend_(std::move(backend)), single_flight_(single_flight) {}

PeerSsdManager::~PeerSsdManager() = default;

// ---------------------------------------------------------------------------
//  Capacity & queries
// ---------------------------------------------------------------------------

std::pair<size_t, size_t> PeerSsdManager::Capacity() const {
  if (!backend_) return {0, 0};
  return backend_->Capacity();
}

bool PeerSsdManager::Exists(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return owned_.find(key) != owned_.end();
}

uint64_t PeerSsdManager::SizeOf(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = owned_.find(key);
  return it == owned_.end() ? 0 : it->second.size;
}

bool PeerSsdManager::PinForMigration(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (owned_.find(key) == owned_.end() || evicting_.count(key) != 0) return false;
  auto& slot = inflight_reads_[key];
  if (!slot) slot = std::make_unique<InflightRead>();
  ++slot->refs;
  return true;
}

void PeerSsdManager::UnpinForMigration(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = inflight_reads_.find(key);
  if (it != inflight_reads_.end() && --it->second->refs <= 0) inflight_reads_.erase(it);
  if (inflight_reads_.empty()) reads_drained_cv_.notify_all();
}

std::optional<RecordLocation> PeerSsdManager::LocateRecord(const std::string& key) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (backend_ == nullptr) return std::nullopt;
  // A key mid-eviction is a miss, exactly as PrepareRead treats it.
  if (evicting_.count(key) != 0) return std::nullopt;
  if (owned_.find(key) == owned_.end()) return std::nullopt;
  return backend_->LocateRecord(key);
}

void PeerSsdManager::TouchLocked(const std::string& key) {
  auto it = owned_.find(key);
  if (it == owned_.end()) return;
  lru_.splice(lru_.begin(), lru_, it->second.lru_it);  // move-to-front; iterator stays valid
  it->second.lru_it = lru_.begin();
}

std::vector<bool> PeerSsdManager::WriteBatch(const std::vector<std::string>& keys,
                                             const std::vector<const void*>& srcs,
                                             const std::vector<size_t>& sizes) {
  std::vector<bool> results(keys.size(), false);
  if (!backend_ || keys.empty()) return results;
  if (srcs.size() != keys.size() || sizes.size() != keys.size()) {
    MORI_UMBP_ERROR("[PeerSsdManager] WriteBatch: mismatched input lengths ({} / {} / {})",
                    keys.size(), srcs.size(), sizes.size());
    return results;
  }

  // Stage timers for the [SsdPerf/peer] PUT breakdown (no-ops unless
  // UMBP_SSD_TIMING is set).
  const auto t_begin = ssdperf::Now();
  double dedup_ms = 0.0, backend_ms = 0.0, record_ms = 0.0;

  // Pass 1 — dedup under one lock.  Already-owned keys succeed with no IO (same
  // rationale as Write: a sequential re-put must not repeat the device write).
  std::vector<size_t> todo;
  todo.reserve(keys.size());
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
      if (sizes[i] == 0) continue;  // meaningless put; leave results[i] == false
      if (owned_.find(keys[i]) != owned_.end()) {
        TouchLocked(keys[i]);
        results[i] = true;
        continue;
      }
      todo.push_back(i);
    }
  }
  dedup_ms = ssdperf::MsSince(t_begin);
  const size_t dedup_hits = keys.size() - todo.size();
  if (todo.empty()) return results;

  // Pass 2 — one batched backend write outside the lock.  A failed key
  // (ENOSPC included) is recorded nowhere; this never evicts.
  const auto t_backend = ssdperf::Now();
  std::vector<std::string> batch_keys;
  std::vector<const void*> batch_srcs;
  std::vector<size_t> batch_sizes;
  batch_keys.reserve(todo.size());
  batch_srcs.reserve(todo.size());
  batch_sizes.reserve(todo.size());
  for (size_t i : todo) {
    batch_keys.push_back(keys[i]);
    batch_srcs.push_back(srcs[i]);
    batch_sizes.push_back(sizes[i]);
  }
  auto ok = backend_->BatchWrite(batch_keys, batch_srcs, batch_sizes);
  ok.resize(todo.size(), false);
  backend_ms = ssdperf::MsSince(t_backend);

  // Pass 3 — record locations + ADD events under one lock.  The owned_ re-check
  // matches Write: a concurrent writer may have landed the same content-addressed
  // bytes, and only the first recorder emits ADD SSD.
  const auto t_record = ssdperf::Now();
  uint64_t written_bytes = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t j = 0; j < todo.size(); ++j) {
      const size_t i = todo[j];
      if (!ok[j]) continue;
      results[i] = true;
      written_bytes += sizes[i];
      RecordWrittenLocked(keys[i], sizes[i]);
    }
  }
  metrics_.copy_bytes.fetch_add(written_bytes, std::memory_order_relaxed);
  record_ms = ssdperf::MsSince(t_record);

  for (size_t j = 0; j < todo.size(); ++j) {
    if (!ok[j]) {
      MORI_UMBP_WARN("[PeerSsdManager] WriteBatch: backend write failed key={} size={}",
                     keys[todo[j]], sizes[todo[j]]);
    }
  }

  if (ssdperf::Enabled()) {
    const double total_ms = ssdperf::MsSince(t_begin);
    const auto [used, total] = Capacity();
    MORI_UMBP_INFO(
        "[SsdPerf/peer] PUT keys={} dedup_hits={} written={} bytes={} total_ms={:.3f} GB_s={:.2f} "
        "| dedup={:.3f}ms backend={:.3f}ms ({:.0f}%) record={:.3f}ms | used={}/{}B",
        keys.size(), dedup_hits, todo.size(), written_bytes, total_ms,
        ssdperf::GbPerSec(written_bytes, total_ms), dedup_ms, backend_ms,
        ssdperf::Pct(backend_ms, total_ms), record_ms, used, total);
  }
  return results;
}

void PeerSsdManager::RecordWrittenLocked(const std::string& key, uint64_t size) {
  if (owned_.find(key) != owned_.end()) {
    TouchLocked(key);
    return;
  }
  lru_.push_front(key);
  owned_.emplace(key, OwnedEntry{size, lru_.begin()});
  if (event_publishing_) {
    pending_events_.push_back(KvEvent{KvEvent::Kind::ADD, key, TierType::SSD, size});
  }
}

std::optional<RecordReservation> PeerSsdManager::Reserve(const std::string& key, uint64_t size) {
  if (!backend_ || size == 0) return std::nullopt;
  return backend_->ReserveRecord(key, static_cast<size_t>(size));
}

bool PeerSsdManager::CommitReserved(const std::string& key, const RecordReservation& r) {
  if (!backend_ || !backend_->CommitRecord(key, r)) return false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    RecordWrittenLocked(key, r.location.value_size);
  }
  metrics_.copy_bytes.fetch_add(r.location.value_size, std::memory_order_relaxed);
  metrics_.writes_in_place.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void PeerSsdManager::AbortReserved(const std::string& key, const RecordReservation& r) {
  if (backend_) backend_->AbortRecord(key, r);
  metrics_.write_in_place_aborts.fetch_add(1, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
//  Eviction (the medium's one implementation; read-priority)
// ---------------------------------------------------------------------------

bool PeerSsdManager::Evict(const std::string& key) {
  if (!backend_) return false;

  // Reserve under the lock: skip if a read is in flight (read priority), if the
  // key is gone, or if another worker is already evicting it.  owned_ is kept
  // until the backend confirms the delete.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (owned_.find(key) == owned_.end()) return false;
    if (inflight_reads_.count(key) != 0) return false;
    if (!evicting_.insert(key).second) return false;
  }

  bool ok = backend_->Evict(key);  // backend IO outside the lock

  // Commit only if the backend freed the bytes; on failure keep owned_ for a
  // later retry, so REMOVE SSD is never emitted while the bytes still exist.
  std::lock_guard<std::mutex> lock(mutex_);
  evicting_.erase(key);
  if (!ok) {
    metrics_.evict_backend_failures.fetch_add(1, std::memory_order_relaxed);
    MORI_UMBP_WARN("[PeerSsdManager] backend Evict failed key={} — keeping for retry", key);
    return false;
  }
  auto it = owned_.find(key);
  if (it != owned_.end()) {  // still present (a racing ClearLocal could have dropped it)
    lru_.erase(it->second.lru_it);
    owned_.erase(it);
    if (event_publishing_) {
      pending_events_.push_back(KvEvent{KvEvent::Kind::REMOVE, key, TierType::SSD, 0});
    }
  }
  return true;
}

bool PeerSsdManager::OrdersEviction() const {
  return backend_ != nullptr && backend_->Capabilities().orders_eviction;
}

void PeerSsdManager::SetRetiredFileHook(std::function<void(int fd)> on_close,
                                        std::chrono::milliseconds grace) {
  if (backend_) backend_->SetRetiredFileHook(std::move(on_close), grace);
}

std::vector<EvictionOffer> PeerSsdManager::EvictionCandidates(size_t max_count) {
  std::vector<EvictionOffer> candidates;
  if (max_count == 0) return candidates;

  if (OrdersEviction()) {
    // Ask the tier outside mutex_ (it ranks every segment), for enough extra
    // keys that the ones filtered out below do not shorten the answer -- a
    // short answer reads to the pool as "nothing more to offer".
    size_t skipped = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      skipped = inflight_reads_.size() + evicting_.size();
    }
    const auto hints = backend_->EvictionOrder(max_count + skipped);
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& hint : hints) {
      if (candidates.size() >= max_count) break;
      if (inflight_reads_.count(hint.key) != 0) continue;
      if (evicting_.count(hint.key) != 0) continue;
      // Not recorded yet (its write is still landing) or already gone.
      if (owned_.find(hint.key) == owned_.end()) continue;
      candidates.push_back(EvictionOffer{hint.key, hint.reclaim_bytes});
    }
    return candidates;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  // Oldest first (LRU tail -> MRU front), skipping keys being read or evicted.
  for (auto it = lru_.rbegin(); it != lru_.rend() && candidates.size() < max_count; ++it) {
    const std::string& key = *it;
    if (inflight_reads_.count(key) != 0) continue;
    if (evicting_.count(key) != 0) continue;
    auto owned_it = owned_.find(key);
    if (owned_it == owned_.end()) continue;  // defensive; lru_/owned_ stay in sync
    candidates.push_back(EvictionOffer{key, owned_it->second.size});
  }
  return candidates;
}

// ---------------------------------------------------------------------------
//  Clear (drop metadata + wipe physical bytes)
// ---------------------------------------------------------------------------

void PeerSsdManager::ClearLocal() {
  // CALLER INVARIANT: quiesce the copy pipeline before calling this.  We drain
  // in-flight reads (below) but not in-flight Writes, so a racing Write could
  // re-add owned_/ADD SSD after backend->Clear() wipes its bytes.
  // PoolClient::Clear() Quiesce()s first; new callers must too.
  // No Evict overlaps this: PeerPool::Evict holds the lifecycle lock that
  // PeerPool::ClearLocal takes exclusively.
  {
    std::unique_lock<std::mutex> lock(mutex_);
    owned_.clear();
    lru_.clear();
    pending_events_.clear();
    evicting_.clear();
    // Read priority: new reads already miss (owned_ is empty), but a read that
    // already started holds inflight_reads_ and is doing backend IO.  Wait for
    // it to finish before wiping the bytes — SSD reads cannot be safely aborted.
    reads_drained_cv_.wait(lock, [this] { return inflight_reads_.empty(); });
  }
  if (backend_) backend_->Clear();  // delete physical SSD bytes (user Clear = discard cache)
}

void PeerSsdManager::DiscardLeftoverOnStartup() {
  // owned_ is empty on a fresh process, so there is nothing to reconcile: just
  // wipe any orphan bytes the backend loaded from disk so used capacity starts
  // at 0 (cache is re-fetchable; safe to drop).  See header for the policy.
  if (!backend_) return;
  auto [used, total] = backend_->Capacity();
  if (used == 0) {
    MORI_UMBP_INFO("[PeerSsdManager] startup discard: no SSD leftover (used=0)");
    return;
  }
  MORI_UMBP_INFO("[PeerSsdManager] startup discard: wiping {}B SSD leftover (total={}B)", used,
                 total);
  backend_->Clear();
}

// ---------------------------------------------------------------------------
//  Read
// ---------------------------------------------------------------------------

SsdReadOutcome PeerSsdManager::PrepareRead(const std::string& key, void* staging_ptr,
                                           size_t staging_cap) {
  if (!backend_) {
    metrics_.read_not_found.fetch_add(1, std::memory_order_relaxed);
    return SsdReadOutcome{SsdReadStatus::kNotFound, 0};
  }

  // Resolve size and mark the read in flight under the lock, then run the
  // blocking SSD IO outside it (so a concurrent copy Write is not serialized).
  size_t size = 0;
  bool leading = false;    // we own the device read and the fan-out
  bool following = false;  // a leader will fill our buffer; we only wait
  std::shared_ptr<ReadEpisode> episode;
  // Our own slot in the leader's fan-out list, kept because the leader swaps
  // that list out of the episode when it seals.
  std::shared_ptr<FollowerTarget> my_target;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = owned_.find(key);
    if (it == owned_.end()) {
      metrics_.read_not_found.fetch_add(1, std::memory_order_relaxed);
      return SsdReadOutcome{SsdReadStatus::kNotFound, 0};
    }
    // Being evicted: bytes about to vanish — treat a new read as a stale-route
    // miss rather than racing the backend delete.
    if (evicting_.count(key) != 0) {
      metrics_.read_not_found.fetch_add(1, std::memory_order_relaxed);
      return SsdReadOutcome{SsdReadStatus::kNotFound, 0};
    }
    size = it->second.size;
    // Reject over-capacity before touching the device (no in-flight mark, no IO).
    if (size > staging_cap) {
      metrics_.read_size_too_large.fetch_add(1, std::memory_order_relaxed);
      MORI_UMBP_WARN("[PeerSsdManager] SSD read key={} size={}B exceeds its {}B buffer", key, size,
                     staging_cap);
      return SsdReadOutcome{SsdReadStatus::kSizeTooLarge, size};
    }
    // Same leader/follower protocol as PrepareReadBatch, over the same map, so
    // a single-key read and a batched one coalesce with each other.  No log
    // here: this path is one key per call, so the counters are the only sane
    // reporting channel.
    auto& slot = inflight_reads_[key];
    if (!slot) slot = std::make_unique<InflightRead>();
    const int prior = slot->refs++;
    metrics_.read_dup.fetch_add(prior > 0 ? 1 : 0, std::memory_order_relaxed);
    metrics_.read_lead.fetch_add(prior > 0 ? 0 : 1, std::memory_order_relaxed);
    if (!slot->episode) {
      slot->episode = std::make_shared<ReadEpisode>();
      slot->episode->length = size;  // a whole-object read: [0, size)
      episode = slot->episode;
      leading = true;
    } else if (single_flight_ && !slot->episode->sealed && slot->episode->Covers(0, size)) {
      // An extent leader lacks the rest of the object: then read it ourselves.
      my_target = std::make_shared<FollowerTarget>();
      my_target->dst = staging_ptr;
      my_target->length = size;  // size <= staging_cap, checked above
      slot->episode->followers.push_back(my_target);
      episode = slot->episode;
      following = true;
      metrics_.read_merged.fetch_add(1, std::memory_order_relaxed);
    }
    // else: missed the merge window — fall through and read it ourselves.
  }

  bool read_ok = false;
  if (following) {
    std::unique_lock<std::mutex> lock(mutex_);
    const bool signalled = episode->cv.wait_for(lock, std::chrono::seconds(kFollowerWaitSeconds),
                                                [&episode] { return episode->done; });
    if (!signalled) {
      MORI_UMBP_WARN(
          "[PeerSsdManager] single-flight follower timed out after {}s waiting on leader key={}",
          kFollowerWaitSeconds, key);
    }
    read_ok = signalled && my_target->ok;
  } else {
    read_ok = backend_->ReadIntoPtr(key, reinterpret_cast<uintptr_t>(staging_ptr), size);
    if (leading) {
      // Publish to whoever attached while we were on the device.  Failures are
      // published too, so a follower is never left to time out.
      std::vector<std::shared_ptr<FollowerTarget>> fanout;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        episode->sealed = true;
        fanout.swap(episode->followers);
        auto it = inflight_reads_.find(key);
        if (it != inflight_reads_.end() && it->second->episode == episode) {
          it->second->episode.reset();
        }
      }
      for (const auto& f : fanout) {
        f->ok = read_ok;
        if (read_ok) {
          std::memcpy(f->dst, static_cast<const char*>(staging_ptr) + (f->offset - episode->offset),
                      f->length);
        }
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        episode->done = true;
        episode->cv.notify_all();
      }
    }
  }

  // Always release the in-flight mark (even on error), refresh LRU on success,
  // and wake a waiting ClearLocal once the last read drains.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto rit = inflight_reads_.find(key);
    if (rit != inflight_reads_.end() && --rit->second->refs <= 0) inflight_reads_.erase(rit);
    if (read_ok && owned_.find(key) != owned_.end()) TouchLocked(key);
    if (inflight_reads_.empty()) reads_drained_cv_.notify_all();
  }

  if (!read_ok) {
    // owned_ had the key but the backend couldn't serve it (e.g. a local evict
    // raced us, or a corrupt record): kError, not a definitive miss.
    metrics_.read_error.fetch_add(1, std::memory_order_relaxed);
    MORI_UMBP_WARN("[PeerSsdManager] PrepareRead backend read failed key={} size={}", key, size);
    return SsdReadOutcome{SsdReadStatus::kError, size};
  }
  metrics_.read_ok.fetch_add(1, std::memory_order_relaxed);
  metrics_.read_bytes.fetch_add(size, std::memory_order_relaxed);  // SSD read IO bandwidth source
  return SsdReadOutcome{SsdReadStatus::kOk, size};
}

std::vector<SsdReadOutcome> PeerSsdManager::PrepareReadBatch(
    const std::vector<std::string>& keys, const std::vector<void*>& dsts,
    const std::vector<size_t>& caps, const std::vector<ObjectExtent>* extents) {
  std::vector<SsdReadOutcome> out(keys.size(), SsdReadOutcome{SsdReadStatus::kNotFound, 0});
  if (keys.empty()) return out;
  if (!backend_) {
    metrics_.read_not_found.fetch_add(keys.size(), std::memory_order_relaxed);
    return out;
  }
  if (dsts.size() != keys.size() || caps.size() != keys.size() ||
      (extents != nullptr && extents->size() != keys.size())) {
    MORI_UMBP_ERROR("[PeerSsdManager] PrepareReadBatch: mismatched input lengths ({} / {} / {})",
                    keys.size(), dsts.size(), caps.size());
    for (auto& o : out) o = SsdReadOutcome{SsdReadStatus::kError, 0};
    return out;
  }
  const auto is_ranged = [&](size_t i) { return extents != nullptr && (*extents)[i].length != 0; };
  std::vector<ObjectExtent> wants(keys.size());  // what each key reads: its extent, or [0, size)

  // Stage timers for the [SsdPerf/peer] GET breakdown (no-ops unless
  // UMBP_SSD_TIMING is set).
  const auto t_begin = ssdperf::Now();
  double resolve_ms = 0.0, backend_ms = 0.0, release_ms = 0.0;
  // fanout_ms: Pass 2b, single-threaded memcpy + per-key lock/unlock/notify_all
  // serving OTHER concurrent callers' follower reads off of our own device read.
  // wait_ms: Pass 2c, time spent blocked on someone else's episode as a follower.
  // Both are 0 for calls that neither lead anyone nor follow anyone.
  double fanout_ms = 0.0, wait_ms = 0.0;
  uint64_t fanout_bytes = 0;
  size_t fanout_count = 0;

  // Pass 1 — resolve sizes and mark every servable key in flight under ONE lock
  // acquisition (the per-key path pays two per key).
  std::vector<size_t> todo;   // I lead (or independently read) these
  std::vector<size_t> waits;  // I follow these; a leader will fill my buffer
  std::vector<bool> is_leader(keys.size(), false);
  // Per-request handle on the episode we lead or follow.  Held as a shared_ptr
  // so it stays valid even after the slot moves on to a new leader.
  std::vector<std::shared_ptr<ReadEpisode>> episodes(keys.size());
  // Our own entry in a leader's fan-out list, for the keys we follow.  Kept
  // per-request because the leader swaps its list out of the episode on seal.
  std::vector<std::shared_ptr<FollowerTarget>> my_targets(keys.size());
  todo.reserve(keys.size());
  // dup = requests that found an identical read already in flight;
  // merged = those actually served without touching the drive;
  // max_concurrent = largest same-key fan-out seen this batch.
  size_t dup = 0, merged = 0;
  int max_concurrent = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
      auto it = owned_.find(keys[i]);
      if (it == owned_.end() || evicting_.count(keys[i]) != 0) {
        metrics_.read_not_found.fetch_add(1, std::memory_order_relaxed);
        continue;  // stays kNotFound
      }
      const size_t size = it->second.size;
      wants[i] = is_ranged(i) ? (*extents)[i] : ObjectExtent{0, size};
      if (wants[i].offset > size || wants[i].length > size - wants[i].offset) {
        metrics_.read_error.fetch_add(1, std::memory_order_relaxed);
        MORI_UMBP_WARN("[PeerSsdManager] SSD read key={} extent [{}, +{}) outside size {}", keys[i],
                       wants[i].offset, wants[i].length, size);
        out[i] = SsdReadOutcome{SsdReadStatus::kError, size};
        continue;
      }
      if (wants[i].length > caps[i]) {
        metrics_.read_size_too_large.fetch_add(1, std::memory_order_relaxed);
        MORI_UMBP_WARN("[PeerSsdManager] SSD read key={} size={}B exceeds its {}B buffer", keys[i],
                       wants[i].length, caps[i]);
        out[i] = SsdReadOutcome{SsdReadStatus::kSizeTooLarge, size};
        continue;
      }
      out[i].size = size;
      auto& slot = inflight_reads_[keys[i]];
      if (!slot) slot = std::make_unique<InflightRead>();
      const int prior = slot->refs++;
      if (prior > 0) {
        ++dup;
        if (prior > max_concurrent) max_concurrent = prior;
      }
      if (!slot->episode) {
        // First in: this request owns the device read and the fan-out.
        slot->episode = std::make_shared<ReadEpisode>();
        slot->episode->offset = wants[i].offset;
        slot->episode->length = wants[i].length;
        episodes[i] = slot->episode;
        is_leader[i] = true;
        todo.push_back(i);
      } else if (single_flight_ && !slot->episode->sealed &&
                 slot->episode->Covers(wants[i].offset, wants[i].length)) {
        // Merge window open and the leader reads our bytes: hand it our buffer
        // and wait on its episode.
        my_targets[i] = std::make_shared<FollowerTarget>();
        my_targets[i]->dst = dsts[i];
        my_targets[i]->offset = wants[i].offset;
        my_targets[i]->length = wants[i].length;
        slot->episode->followers.push_back(my_targets[i]);
        episodes[i] = slot->episode;
        waits.push_back(i);
        ++merged;
      } else {
        // Missed the window, the leader reads other bytes, or single flight is
        // off: read it ourselves.  Always correct, just not merged.
        todo.push_back(i);
      }
    }
  }
  resolve_ms = ssdperf::MsSince(t_begin);
  metrics_.read_dup.fetch_add(dup, std::memory_order_relaxed);
  metrics_.read_merged.fetch_add(merged, std::memory_order_relaxed);
  metrics_.read_lead.fetch_add(todo.size() + waits.size() - dup, std::memory_order_relaxed);
  const size_t not_owned = keys.size() - todo.size() - waits.size();
  if (todo.empty() && waits.empty()) {
    if (ssdperf::Enabled()) {
      MORI_UMBP_INFO(
          "[SsdPerf/peer] GET keys={} served=0 not_owned={} total_ms={:.3f} | resolve={:.3f}ms "
          "(no backend IO: nothing owned locally)",
          keys.size(), not_owned, ssdperf::MsSince(t_begin), resolve_ms);
    }
    return out;
  }

  // Pass 2 — batched backend reads outside the lock, one for whole objects and
  // one for extents; ShardedSsdTier turns each into concurrent IO on every
  // drive holding part of the batch.
  std::vector<bool> ok(todo.size(), false);
  if (!todo.empty()) {
    const auto t_backend = ssdperf::Now();
    std::vector<size_t> whole, ranged;  // positions in todo
    for (size_t j = 0; j < todo.size(); ++j) (is_ranged(todo[j]) ? ranged : whole).push_back(j);
    if (!whole.empty()) {
      std::vector<std::string> k;
      std::vector<uintptr_t> d;
      std::vector<size_t> z;
      for (size_t j : whole) {
        k.push_back(keys[todo[j]]);
        d.push_back(reinterpret_cast<uintptr_t>(dsts[todo[j]]));
        z.push_back(out[todo[j]].size);
      }
      const auto got = backend_->BatchReadIntoPtr(k, d, z);
      for (size_t w = 0; w < whole.size() && w < got.size(); ++w) ok[whole[w]] = got[w];
    }
    if (!ranged.empty()) {
      std::vector<std::string> k;
      std::vector<std::vector<uintptr_t>> d;
      std::vector<std::vector<size_t>> z, o;
      for (size_t j : ranged) {
        const size_t i = todo[j];
        k.push_back(keys[i]);
        d.push_back({reinterpret_cast<uintptr_t>(dsts[i])});
        z.push_back({static_cast<size_t>(wants[i].length)});
        o.push_back({static_cast<size_t>(wants[i].offset)});
      }
      const auto got = backend_->ReadBatchRangesIntoPtr(k, d, z, o);
      for (size_t r = 0; r < ranged.size() && r < got.size(); ++r) ok[ranged[r]] = got[r];
    }
    backend_ms = ssdperf::MsSince(t_backend);
  }

  // Pass 2b — fan out to followers.  MUST run before Pass 2c: if this batch
  // leads key A while another thread leads B and follows A, waiting first would
  // close a cycle.  Issuing all our own reads and publishing their results
  // before we block on anyone else's makes that cycle impossible.
  //
  // A leader whose read FAILED still publishes (ok=false) — followers must be
  // released with an error, never left to time out.
  const auto t_fanout = ssdperf::Now();
  for (size_t j = 0; j < todo.size(); ++j) {
    const size_t i = todo[j];
    if (!is_leader[i]) continue;  // independent read: nobody is waiting on it
    const auto& ep = episodes[i];
    std::vector<std::shared_ptr<FollowerTarget>> fanout;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ep->sealed = true;  // late arrivals now go independent instead of waiting
      fanout.swap(ep->followers);
      // Retire the episode so the next arrival starts a fresh one rather than
      // attaching to a read that is already past its snapshot.
      auto it = inflight_reads_.find(keys[i]);
      if (it != inflight_reads_.end() && it->second->episode == ep) it->second->episode.reset();
    }
    // Deliver outside the lock: holding mutex_ across a page-sized memcpy would
    // stall every other read/evict here.
    for (const auto& f : fanout) {
      f->ok = ok[j];
      if (!ok[j]) continue;
      std::memcpy(f->dst, static_cast<const char*>(dsts[i]) + (f->offset - ep->offset), f->length);
      fanout_bytes += f->length;
      ++fanout_count;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ep->done = true;
      ep->cv.notify_all();
    }
  }
  fanout_ms = ssdperf::MsSince(t_fanout);

  // Pass 2c — collect the keys we followed.  Our staging buffer is still alive
  // (we never returned), so a leader's memcpy landed in a valid destination.
  const auto t_wait = ssdperf::Now();
  std::vector<bool> wait_ok(waits.size(), false);
  for (size_t w = 0; w < waits.size(); ++w) {
    const size_t i = waits[w];
    const auto& ep = episodes[i];
    std::unique_lock<std::mutex> lock(mutex_);
    const bool signalled = ep->cv.wait_for(lock, std::chrono::seconds(kFollowerWaitSeconds),
                                           [&ep] { return ep->done; });
    if (!signalled) {
      MORI_UMBP_WARN(
          "[PeerSsdManager] single-flight follower timed out after {}s waiting on leader key={}",
          kFollowerWaitSeconds, keys[i]);
    }
    wait_ok[w] = signalled && my_targets[i]->ok;
  }
  wait_ms = ssdperf::MsSince(t_wait);

  // Pass 3 — release the in-flight marks, refresh LRU for the reads that landed,
  // and wake a waiting ClearLocal once the last read drains.
  const auto t_release = ssdperf::Now();
  uint64_t read_bytes = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto release = [&](const std::string& key, bool landed) {
      auto rit = inflight_reads_.find(key);
      if (rit != inflight_reads_.end() && --rit->second->refs <= 0) inflight_reads_.erase(rit);
      if (landed && owned_.find(key) != owned_.end()) TouchLocked(key);
    };
    for (size_t j = 0; j < todo.size(); ++j) release(keys[todo[j]], ok[j]);
    for (size_t w = 0; w < waits.size(); ++w) release(keys[waits[w]], wait_ok[w]);
    if (inflight_reads_.empty()) reads_drained_cv_.notify_all();
  }

  auto record = [&](size_t i, bool landed, bool from_device) {
    if (landed) {
      out[i].status = SsdReadStatus::kOk;
      // read_bytes is device traffic only, so merged reads are excluded — the
      // counter stays a true measure of what the drive moved.
      if (from_device) read_bytes += wants[i].length;
      metrics_.read_ok.fetch_add(1, std::memory_order_relaxed);
    } else {
      // owned_ had the key but it couldn't be served (a local evict raced us, a
      // corrupt record, or our leader failed): kError, not a definitive miss.
      out[i].status = SsdReadStatus::kError;
      metrics_.read_error.fetch_add(1, std::memory_order_relaxed);
      MORI_UMBP_WARN("[PeerSsdManager] PrepareReadBatch read failed key={} size={} (merged={})",
                     keys[i], out[i].size, !from_device);
    }
  };
  for (size_t j = 0; j < todo.size(); ++j) record(todo[j], ok[j], /*from_device=*/true);
  for (size_t w = 0; w < waits.size(); ++w) record(waits[w], wait_ok[w], /*from_device=*/false);
  metrics_.read_bytes.fetch_add(read_bytes, std::memory_order_relaxed);
  release_ms = ssdperf::MsSince(t_release);

  if (ssdperf::Enabled()) {
    const double total_ms = ssdperf::MsSince(t_begin);
    size_t served = 0;
    for (size_t j = 0; j < todo.size(); ++j) {
      if (ok[j]) ++served;
    }
    for (size_t w = 0; w < waits.size(); ++w) {
      if (wait_ok[w]) ++served;
    }
    const uint64_t dup_total = metrics_.read_dup.load(std::memory_order_relaxed);
    const uint64_t lead_total = metrics_.read_lead.load(std::memory_order_relaxed);
    const uint64_t merged_total = metrics_.read_merged.load(std::memory_order_relaxed);
    const uint64_t probe_total = dup_total + lead_total;
    MORI_UMBP_INFO(
        "[SsdPerf/peer] GET keys={} device_reads={} served={} not_owned={} bytes={} "
        "total_ms={:.3f} GB_s={:.2f} | resolve={:.3f}ms backend={:.3f}ms ({:.0f}%) "
        "fanout={:.3f}ms ({:.0f}%) wait={:.3f}ms ({:.0f}%) release_lru={:.3f}ms | "
        "fanout_bytes={} fanout_count={} fanout_GB_s={:.2f} | "
        "singleflight={} dup={} merged={} max_concurrent={} "
        "dup_total={}/{} ({:.1f}%) merged_total={}",
        keys.size(), todo.size(), served, not_owned, read_bytes, total_ms,
        ssdperf::GbPerSec(read_bytes, total_ms), resolve_ms, backend_ms,
        ssdperf::Pct(backend_ms, total_ms), fanout_ms, ssdperf::Pct(fanout_ms, total_ms), wait_ms,
        ssdperf::Pct(wait_ms, total_ms), release_ms, fanout_bytes, fanout_count,
        ssdperf::GbPerSec(fanout_bytes, fanout_ms), single_flight_ ? "on" : "off", dup, merged,
        max_concurrent, dup_total, probe_total,
        ssdperf::Pct(static_cast<double>(dup_total), probe_total), merged_total);
  }
  return out;
}

bool PeerSsdManager::SupportsRangedRead() const {
  return backend_ != nullptr && backend_->Capabilities().ranged_read;
}

// ---------------------------------------------------------------------------
//  OwnedLocationSource (heartbeat event drain / snapshot)
// ---------------------------------------------------------------------------

void PeerSsdManager::SetEventPublishing(bool enabled) {
  std::lock_guard<std::mutex> lock(mutex_);
  event_publishing_ = enabled;
  if (!enabled) pending_events_.clear();
}

std::vector<KvEvent> PeerSsdManager::DrainPendingEvents() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<KvEvent> drained;
  drained.swap(pending_events_);
  return drained;
}

std::vector<KvEvent> PeerSsdManager::SnapshotOwnedKeysLocked() const {
  std::vector<KvEvent> out;
  out.reserve(owned_.size());
  for (const auto& [key, entry] : owned_) {
    out.push_back(KvEvent{KvEvent::Kind::ADD, key, TierType::SSD, entry.size});
  }
  return out;
}

std::vector<KvEvent> PeerSsdManager::SnapshotOwnedKeys() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return SnapshotOwnedKeysLocked();
}

size_t PeerSsdManager::OwnedKeyCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return owned_.size();
}

std::vector<KvEvent> PeerSsdManager::SnapshotOwnedKeysForFullSync() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto out = SnapshotOwnedKeysLocked();
  // Snapshot is authoritative now: drop the queued delta (already reflected in
  // it) so the next delta carries only events produced after this snapshot.
  pending_events_.clear();
  return out;
}

}  // namespace mori::umbp
