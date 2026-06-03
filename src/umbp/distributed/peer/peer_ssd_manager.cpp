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
#include "umbp/distributed/peer/peer_ssd_manager.h"

#include <cstring>
#include <stdexcept>

#include "mori/utils/mori_log.hpp"
#include "umbp/local/tiers/spdk_proxy_tier.h"
#include "umbp/local/tiers/ssd_tier.h"
#include "umbp/local/tiers/tier_backend.h"

namespace mori::umbp {

namespace {
// Watermarks must satisfy 0 < low < high <= 1.  We fail fast on a bad value
// rather than silently clamping, so a misconfigured env surfaces immediately.
bool WatermarksValid(double high, double low) {
  return high > 0.0 && high <= 1.0 && low > 0.0 && low < high;
}
}  // namespace

// ---------------------------------------------------------------------------
//  Construction
// ---------------------------------------------------------------------------

PeerSsdManager::PeerSsdManager(const PeerSsdConfig& cfg) {
  if (!cfg.enabled) {
    MORI_UMBP_INFO("[PeerSsdManager] constructed disabled (no SSD backend)");
    return;
  }
  const auto& ssd = cfg.ssd;

  if (!WatermarksValid(ssd.high_watermark, ssd.low_watermark)) {
    throw std::runtime_error(
        "[PeerSsdManager] invalid SSD watermarks (require 0 < low_watermark < "
        "high_watermark <= 1)");
  }
  high_watermark_ = ssd.high_watermark;
  low_watermark_ = ssd.low_watermark;

  // Explicit backend selection — an unknown value is a configuration error, not
  // a reason to silently fall back to POSIX.
  if (ssd.ssd_backend == "posix") {
    backend_ = std::make_unique<SSDTier>(ssd.storage_dir, ssd.capacity_bytes, ssd);
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
                           "' (expected one of: posix, spdk, spdk_proxy)");
}

PeerSsdManager::PeerSsdManager(std::unique_ptr<TierBackend> backend, double high_watermark,
                               double low_watermark)
    : backend_(std::move(backend)), high_watermark_(high_watermark), low_watermark_(low_watermark) {
  if (!WatermarksValid(high_watermark_, low_watermark_)) {
    throw std::runtime_error(
        "[PeerSsdManager] invalid SSD watermarks (require 0 < low_watermark < "
        "high_watermark <= 1)");
  }
}

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

void PeerSsdManager::TouchLocked(const std::string& key) {
  auto it = owned_.find(key);
  if (it == owned_.end()) return;
  lru_.splice(lru_.begin(), lru_, it->second.lru_it);  // move-to-front; iterator stays valid
  it->second.lru_it = lru_.begin();
}

// ---------------------------------------------------------------------------
//  Write (copy-on-commit landing)
// ---------------------------------------------------------------------------

bool PeerSsdManager::Write(const std::string& key,
                           const std::vector<std::pair<const void*, size_t>>& segments,
                           size_t total_size) {
  if (!backend_) return false;

  // Optimization, not just defense: the DRAM pin only dedups *concurrent*
  // copies, so a sequential re-put of an already-resident key would otherwise
  // repeat the whole SSD write.  Skip it and refresh LRU instead.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (owned_.find(key) != owned_.end()) {
      TouchLocked(key);
      return true;
    }
  }

  // Assemble (possibly non-contiguous) DRAM source segments into one contiguous
  // buffer for the backend.  A single right-sized segment is written directly;
  // otherwise we memcpy into scratch (a writev-style path could avoid the copy).
  const void* data = nullptr;
  std::vector<char> scratch;
  if (segments.size() == 1 && segments[0].second == total_size) {
    data = segments[0].first;
  } else {
    scratch.resize(total_size);
    size_t off = 0;
    for (const auto& [ptr, len] : segments) {
      if (off + len > total_size) {
        MORI_UMBP_ERROR("[PeerSsdManager] Write key={} segments exceed total_size={}", key,
                        total_size);
        return false;
      }
      if (len > 0) std::memcpy(scratch.data() + off, ptr, len);
      off += len;
    }
    if (off != total_size) {
      MORI_UMBP_ERROR("[PeerSsdManager] Write key={} assembled {} != total_size={}", key, off,
                      total_size);
      return false;
    }
    data = scratch.data();
  }

  // Backend IO outside our mutex_ — SSDTier is internally synchronized.  A
  // failure may be ENOSPC: run one eviction round to reclaim space and retry
  // once before giving up (best-effort, no event/record on final failure).
  if (!backend_->Write(key, data, total_size)) {
    EvictToLowWatermark();
    if (!backend_->Write(key, data, total_size)) {
      MORI_UMBP_WARN("[PeerSsdManager] backend Write failed key={} size={}", key, total_size);
      return false;
    }
  }

  // Bytes physically written to the SSD device (write IO bandwidth source).
  // Counted on every successful backend Write, including the rare dup-content
  // re-write by a second copy worker (real device IO happened either way).
  metrics_.copy_bytes.fetch_add(total_size, std::memory_order_relaxed);

  // Record the location.  Re-check owned_: with > 1 copy worker, two of them
  // could have both seen "absent" above and written the same (identical,
  // content-addressed) bytes — only the first records it + emits ADD SSD.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (owned_.find(key) != owned_.end()) {
      TouchLocked(key);
    } else {
      lru_.push_front(key);
      owned_.emplace(key, OwnedEntry{total_size, lru_.begin()});
      pending_events_.push_back(KvEvent{KvEvent::Kind::ADD, key, TierType::SSD, total_size});
    }
  }

  // Check-after-write trigger, on this copy worker (no dedicated thread).
  auto [used, total] = Capacity();
  if (total > 0 && static_cast<double>(used) >= high_watermark_ * static_cast<double>(total)) {
    EvictToLowWatermark();
  }
  return true;
}

// ---------------------------------------------------------------------------
//  Eviction (local, read-priority)
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
    metrics_.evict_victims.fetch_add(1, std::memory_order_relaxed);
    metrics_.evict_bytes_freed.fetch_add(it->second.size, std::memory_order_relaxed);
    lru_.erase(it->second.lru_it);
    owned_.erase(it);
    pending_events_.push_back(KvEvent{KvEvent::Kind::REMOVE, key, TierType::SSD, 0});
  }
  return true;
}

std::vector<std::string> PeerSsdManager::SelectVictims(size_t bytes_to_free) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> victims;
  if (bytes_to_free == 0) return victims;

  // Oldest first (LRU tail -> MRU front), skipping keys being read or evicted.
  size_t freed = 0;
  for (auto it = lru_.rbegin(); it != lru_.rend(); ++it) {
    const std::string& key = *it;
    if (inflight_reads_.count(key) != 0) continue;
    if (evicting_.count(key) != 0) continue;
    auto owned_it = owned_.find(key);
    if (owned_it == owned_.end()) continue;  // defensive; lru_/owned_ stay in sync
    victims.push_back(key);
    freed += owned_it->second.size;
    if (freed >= bytes_to_free) break;
  }
  return victims;
}

void PeerSsdManager::EvictToLowWatermark() {
  if (!backend_) return;

  // Only one eviction round at a time: a second concurrent worker (or a worker
  // racing ClearLocal) backs off instead of over-evicting.  try_lock keeps the
  // copy path non-blocking.
  std::unique_lock<std::mutex> round(eviction_mu_, std::try_to_lock);
  if (!round.owns_lock()) return;

  auto [used, total] = Capacity();
  if (total == 0) return;
  double low_bytes = low_watermark_ * static_cast<double>(total);
  if (static_cast<double>(used) <= low_bytes) return;
  size_t bytes_to_free = used - static_cast<size_t>(low_bytes);

  // A real round is about to run (we own the round lock and are over the low
  // watermark).  Count it before selecting victims.
  metrics_.evict_rounds.fetch_add(1, std::memory_order_relaxed);

  // Single pass, no retry loop: if everything reclaimable is in use we free
  // what we can and stop, so a fully-pinned tier cannot starve the worker.
  for (const auto& key : SelectVictims(bytes_to_free)) {
    Evict(key);
  }
}

// ---------------------------------------------------------------------------
//  Clear (drop metadata + wipe physical bytes)
// ---------------------------------------------------------------------------

void PeerSsdManager::ClearLocal() {
  // CALLER INVARIANT: quiesce the copy pipeline before calling this.  We drain
  // in-flight reads (below) but not in-flight Writes, so a racing Write could
  // re-add owned_/ADD SSD after backend->Clear() wipes its bytes.
  // PoolClient::Clear() Quiesce()s first; new callers must too.
  //
  // Exclude eviction rounds for the whole operation (lock order: eviction_mu_
  // before mutex_, matching EvictToLowWatermark).
  std::lock_guard<std::mutex> round(eviction_mu_);
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
      // staging_cap = per-slot cap (ssd_staging_buffer_size / ssd_read_slots).
      MORI_UMBP_WARN(
          "[PeerSsdManager] remote SSD read key={} size={}B exceeds per-slot staging cap {}B; "
          "raise ssd_staging_buffer_size or lower ssd_read_slots",
          key, size, staging_cap);
      return SsdReadOutcome{SsdReadStatus::kSizeTooLarge, size};
    }
    ++inflight_reads_[key];
  }

  bool read_ok = backend_->ReadIntoPtr(key, reinterpret_cast<uintptr_t>(staging_ptr), size);

  // Always release the in-flight mark (even on error), refresh LRU on success,
  // and wake a waiting ClearLocal once the last read drains.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto rit = inflight_reads_.find(key);
    if (rit != inflight_reads_.end() && --rit->second <= 0) inflight_reads_.erase(rit);
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

// ---------------------------------------------------------------------------
//  OwnedLocationSource (heartbeat event drain / snapshot)
// ---------------------------------------------------------------------------

std::vector<KvEvent> PeerSsdManager::DrainPendingEvents() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<KvEvent> drained;
  drained.swap(pending_events_);
  return drained;
}

std::vector<KvEvent> PeerSsdManager::SnapshotOwnedKeys() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<KvEvent> out;
  out.reserve(owned_.size());
  for (const auto& [key, entry] : owned_) {
    out.push_back(KvEvent{KvEvent::Kind::ADD, key, TierType::SSD, entry.size});
  }
  return out;
}

}  // namespace mori::umbp
