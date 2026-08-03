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
#include <tuple>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/ssd_perf.h"
#include "umbp/local/tiers/spdk_proxy_tier.h"
#include "umbp/local/tiers/ssd_backend_factory.h"
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
  // a reason to silently fall back to the file backend.
  if (ssd.ssd_backend == "file") {
    // One directory -> SSDTier; several (comma-separated) -> ShardedSsdTier
    // spreading keys across the drives, so batch IO runs on all of them at once.
    backend_ = MakeFileSsdBackend(ssd);
    MORI_UMBP_INFO("[PeerSsdManager] file backend ready dirs=[{}] drives={} capacity={}B",
                   ssd.storage_dir, ssd.StorageDirs().size(), ssd.capacity_bytes);
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
  double dedup_ms = 0.0, backend_ms = 0.0, retry_ms = 0.0, record_ms = 0.0, evict_ms = 0.0;

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

  auto gather = [&](const std::vector<size_t>& idxs) {
    std::vector<std::string> k;
    std::vector<const void*> d;
    std::vector<size_t> z;
    k.reserve(idxs.size());
    d.reserve(idxs.size());
    z.reserve(idxs.size());
    for (size_t i : idxs) {
      k.push_back(keys[i]);
      d.push_back(srcs[i]);
      z.push_back(sizes[i]);
    }
    return std::make_tuple(std::move(k), std::move(d), std::move(z));
  };

  // Pass 2 — one batched backend write outside the lock.  Failures are most
  // likely ENOSPC, so retry exactly those once after a single eviction round
  // (mirrors Write's evict-and-retry, amortized over the whole batch).
  const auto t_backend = ssdperf::Now();
  auto [bk, bd, bz] = gather(todo);
  auto ok = backend_->BatchWrite(bk, bd, bz);
  ok.resize(todo.size(), false);
  backend_ms = ssdperf::MsSince(t_backend);

  std::vector<size_t> retry;
  for (size_t j = 0; j < todo.size(); ++j) {
    if (!ok[j]) retry.push_back(todo[j]);
  }
  if (!retry.empty()) {
    const auto t_retry = ssdperf::Now();
    EvictToLowWatermark();
    auto [rk, rd, rz] = gather(retry);
    auto retry_ok = backend_->BatchWrite(rk, rd, rz);
    retry_ok.resize(retry.size(), false);
    for (size_t j = 0, r = 0; j < todo.size(); ++j) {
      if (!ok[j]) ok[j] = retry_ok[r++];
    }
    retry_ms = ssdperf::MsSince(t_retry);
  }

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
      if (owned_.find(keys[i]) != owned_.end()) {
        TouchLocked(keys[i]);
      } else {
        lru_.push_front(keys[i]);
        owned_.emplace(keys[i], OwnedEntry{sizes[i], lru_.begin()});
        pending_events_.push_back(KvEvent{KvEvent::Kind::ADD, keys[i], TierType::SSD, sizes[i]});
      }
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

  // One check-after-write for the whole batch instead of one per key.
  const auto t_evict = ssdperf::Now();
  auto [used, total] = Capacity();
  if (total > 0 && static_cast<double>(used) >= high_watermark_ * static_cast<double>(total)) {
    EvictToLowWatermark();
  }
  evict_ms = ssdperf::MsSince(t_evict);

  if (ssdperf::Enabled()) {
    const double total_ms = ssdperf::MsSince(t_begin);
    MORI_UMBP_INFO(
        "[SsdPerf/peer] PUT keys={} dedup_hits={} written={} bytes={} total_ms={:.3f} GB_s={:.2f} "
        "| "
        "dedup={:.3f}ms backend={:.3f}ms ({:.0f}%) evict_retry={:.3f}ms record={:.3f}ms "
        "evict_check={:.3f}ms | used={}/{}B",
        keys.size(), dedup_hits, todo.size(), written_bytes, total_ms,
        ssdperf::GbPerSec(written_bytes, total_ms), dedup_ms, backend_ms,
        ssdperf::Pct(backend_ms, total_ms), retry_ms, record_ms, evict_ms, used, total);
  }
  return results;
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
      // staging_cap = per-slot cap (ssd_staging_buffer_size / ssd_staging_buffer_slots).
      MORI_UMBP_WARN(
          "[PeerSsdManager] remote SSD read key={} size={}B exceeds per-slot staging cap {}B; "
          "raise ssd_staging_buffer_size or lower ssd_staging_buffer_slots",
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

std::vector<SsdReadOutcome> PeerSsdManager::PrepareReadBatch(const std::vector<std::string>& keys,
                                                             const std::vector<void*>& dsts,
                                                             const std::vector<size_t>& caps) {
  std::vector<SsdReadOutcome> out(keys.size(), SsdReadOutcome{SsdReadStatus::kNotFound, 0});
  if (keys.empty()) return out;
  if (!backend_) {
    metrics_.read_not_found.fetch_add(keys.size(), std::memory_order_relaxed);
    return out;
  }
  if (dsts.size() != keys.size() || caps.size() != keys.size()) {
    MORI_UMBP_ERROR("[PeerSsdManager] PrepareReadBatch: mismatched input lengths ({} / {} / {})",
                    keys.size(), dsts.size(), caps.size());
    for (auto& o : out) o = SsdReadOutcome{SsdReadStatus::kError, 0};
    return out;
  }

  // Stage timers for the [SsdPerf/peer] GET breakdown (no-ops unless
  // UMBP_SSD_TIMING is set).
  const auto t_begin = ssdperf::Now();
  double resolve_ms = 0.0, backend_ms = 0.0, release_ms = 0.0;

  // Pass 1 — resolve sizes and mark every servable key in flight under ONE lock
  // acquisition (the per-key path pays two per key).
  std::vector<size_t> todo;
  todo.reserve(keys.size());
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
      auto it = owned_.find(keys[i]);
      if (it == owned_.end() || evicting_.count(keys[i]) != 0) {
        metrics_.read_not_found.fetch_add(1, std::memory_order_relaxed);
        continue;  // stays kNotFound
      }
      const size_t size = it->second.size;
      if (size > caps[i]) {
        metrics_.read_size_too_large.fetch_add(1, std::memory_order_relaxed);
        MORI_UMBP_WARN(
            "[PeerSsdManager] SSD read key={} size={}B exceeds per-slot staging cap {}B; "
            "raise ssd_staging_buffer_size or lower ssd_staging_buffer_slots",
            keys[i], size, caps[i]);
        out[i] = SsdReadOutcome{SsdReadStatus::kSizeTooLarge, size};
        continue;
      }
      out[i].size = size;
      ++inflight_reads_[keys[i]];
      todo.push_back(i);
    }
  }
  resolve_ms = ssdperf::MsSince(t_begin);
  const size_t not_owned = keys.size() - todo.size();
  if (todo.empty()) {
    if (ssdperf::Enabled()) {
      MORI_UMBP_INFO(
          "[SsdPerf/peer] GET keys={} served=0 not_owned={} total_ms={:.3f} | resolve={:.3f}ms "
          "(no backend IO: nothing owned locally)",
          keys.size(), not_owned, ssdperf::MsSince(t_begin), resolve_ms);
    }
    return out;
  }

  // Pass 2 — one batched backend read outside the lock; ShardedSsdTier turns
  // this into concurrent IO on every drive holding part of the batch.
  std::vector<std::string> k;
  std::vector<uintptr_t> d;
  std::vector<size_t> z;
  k.reserve(todo.size());
  d.reserve(todo.size());
  z.reserve(todo.size());
  for (size_t i : todo) {
    k.push_back(keys[i]);
    d.push_back(reinterpret_cast<uintptr_t>(dsts[i]));
    z.push_back(out[i].size);
  }
  const auto t_backend = ssdperf::Now();
  auto ok = backend_->BatchReadIntoPtr(k, d, z);
  ok.resize(todo.size(), false);
  backend_ms = ssdperf::MsSince(t_backend);

  // Pass 3 — release the in-flight marks, refresh LRU for the reads that landed,
  // and wake a waiting ClearLocal once the last read drains.
  const auto t_release = ssdperf::Now();
  uint64_t read_bytes = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t j = 0; j < todo.size(); ++j) {
      const size_t i = todo[j];
      auto rit = inflight_reads_.find(keys[i]);
      if (rit != inflight_reads_.end() && --rit->second <= 0) inflight_reads_.erase(rit);
      if (ok[j] && owned_.find(keys[i]) != owned_.end()) TouchLocked(keys[i]);
    }
    if (inflight_reads_.empty()) reads_drained_cv_.notify_all();
  }

  for (size_t j = 0; j < todo.size(); ++j) {
    const size_t i = todo[j];
    if (ok[j]) {
      out[i].status = SsdReadStatus::kOk;
      read_bytes += out[i].size;
      metrics_.read_ok.fetch_add(1, std::memory_order_relaxed);
    } else {
      // owned_ had the key but the backend couldn't serve it (a local evict
      // raced us, or a corrupt record): kError, not a definitive miss.
      out[i].status = SsdReadStatus::kError;
      metrics_.read_error.fetch_add(1, std::memory_order_relaxed);
      MORI_UMBP_WARN("[PeerSsdManager] PrepareReadBatch backend read failed key={} size={}",
                     keys[i], out[i].size);
    }
  }
  metrics_.read_bytes.fetch_add(read_bytes, std::memory_order_relaxed);
  release_ms = ssdperf::MsSince(t_release);

  if (ssdperf::Enabled()) {
    const double total_ms = ssdperf::MsSince(t_begin);
    size_t served = 0;
    for (size_t j = 0; j < todo.size(); ++j) {
      if (ok[j]) ++served;
    }
    MORI_UMBP_INFO(
        "[SsdPerf/peer] GET keys={} attempted={} served={} not_owned={} bytes={} total_ms={:.3f} "
        "GB_s={:.2f} | resolve={:.3f}ms backend={:.3f}ms ({:.0f}%) release_lru={:.3f}ms",
        keys.size(), todo.size(), served, not_owned, read_bytes, total_ms,
        ssdperf::GbPerSec(read_bytes, total_ms), resolve_ms, backend_ms,
        ssdperf::Pct(backend_ms, total_ms), release_ms);
  }
  return out;
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

std::vector<KvEvent> PeerSsdManager::SnapshotOwnedKeysForFullSync() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto out = SnapshotOwnedKeysLocked();
  // Snapshot is authoritative now: drop the queued delta (already reflected in
  // it) so the next delta carries only events produced after this snapshot.
  pending_events_.clear();
  return out;
}

}  // namespace mori::umbp
