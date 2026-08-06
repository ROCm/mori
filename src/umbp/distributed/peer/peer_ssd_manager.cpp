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

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "mori/io/engine.hpp"
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

// A default-constructed MemoryDesc has size 0 and addresses nothing.  Same test
// PoolClient uses for "this descriptor was actually hydrated".
bool IsValidDesc(const mori::io::MemoryDesc& d) { return d.size > 0; }
}  // namespace

// ---------------------------------------------------------------------------
//  Construction
// ---------------------------------------------------------------------------

PeerSsdManager::PeerSsdManager(const PeerSsdConfig& cfg, mori::io::IOEngine* io_engine)
    : io_engine_(io_engine) {
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
  single_flight_ = ssd.single_flight_reads;

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
                               double low_watermark, bool single_flight,
                               mori::io::IOEngine* io_engine)
    : backend_(std::move(backend)), high_watermark_(high_watermark), low_watermark_(low_watermark) {
  single_flight_ = single_flight;
  io_engine_ = io_engine;
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
      // staging_cap = per-slot cap (ssd_staging_buffer_size / ssd_staging_buffer_slots).
      MORI_UMBP_WARN(
          "[PeerSsdManager] remote SSD read key={} size={}B exceeds per-slot staging cap {}B; "
          "raise ssd_staging_buffer_size or lower ssd_staging_buffer_slots",
          key, size, staging_cap);
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
      episode = slot->episode;
      leading = true;
    } else if (single_flight_ && !slot->episode->sealed) {
      // Single-key requesters never register a push destination (see the
      // "explicit scope" note on PrepareReadBatch): they attach as a plain
      // memcpy follower into their own staging slot.
      my_target = std::make_shared<FollowerTarget>();
      my_target->dst = staging_ptr;
      my_target->cap = staging_cap;
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
      // memcpy only: this path has no RDMA source of its own (staging_ptr's
      // MemoryDesc is not plumbed through the single-key RPC), so even a
      // follower that wanted a push is served the classic way and told so.
      for (const auto& f : fanout) {
        f->ok = read_ok;
        f->pushed = false;
        if (read_ok) std::memcpy(f->dst, staging_ptr, std::min(size, f->cap));
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

// RDMA-WRITE the leader's just-read bytes into each push-follower's own
// destination buffer.  This is the whole point of the fast path: a WR post costs
// microseconds where the memcpy it replaces costs a full page copy per follower,
// serialized on one thread.
//
// Every follower gets its OWN TransferStatus and is waited individually.  An
// aggregate "first failure wins" wait would let one dead remote QP fail
// destinations that actually landed — and, worse, would leave the caller unable
// to say which. The leader's own read outcome is not represented here at all;
// it already succeeded or this function was not called.
size_t PeerSsdManager::PushFanout(const std::vector<std::shared_ptr<FollowerTarget>>& fanout,
                                  const ReadPushSlot& leader_slot, size_t size,
                                  uint64_t* pushed_bytes) {
  const size_t n = fanout.size();
  // One transfer per follower.  They share a local descriptor (the leader's
  // staging region) but each has a distinct remote descriptor, and BatchWrite
  // pairs the two vectors element-wise, so grouping buys nothing here.
  mori::io::MemDescVec local_descs(n, leader_slot.local_desc);
  mori::io::MemDescVec remote_descs;
  mori::io::BatchSizeVec local_offsets(n), remote_offsets(n), sizes(n);
  remote_descs.reserve(n);
  std::vector<size_t> bytes(n, 0);
  for (size_t k = 0; k < n; ++k) {
    const auto& f = fanout[k];
    bytes[k] = std::min(size, f->cap);
    remote_descs.push_back(f->remote);
    local_offsets[k].push_back(leader_slot.local_offset);
    remote_offsets[k].push_back(f->remote_offset);
    sizes[k].push_back(bytes[k]);
  }

  std::vector<mori::io::TransferStatus> statuses(n);
  mori::io::TransferStatusPtrVec status_ptrs(n);
  mori::io::TransferUniqueIdVec ids(n);
  for (size_t k = 0; k < n; ++k) {
    status_ptrs[k] = &statuses[k];
    ids[k] = io_engine_->AllocateTransferUniqueId();
  }
  io_engine_->BatchWrite(local_descs, local_offsets, remote_descs, remote_offsets, sizes,
                         status_ptrs, ids);

  size_t ok_count = 0;
  for (size_t k = 0; k < n; ++k) {
    statuses[k].Wait();
    const bool ok = statuses[k].Succeeded();
    fanout[k]->ok = ok;
    // pushed reflects the delivery mechanism, and is meaningful only alongside
    // ok: a failed push is reported to that follower as a read error, and it
    // must NOT then go looking in a staging slot nobody filled.
    fanout[k]->pushed = ok;
    if (ok) {
      ++ok_count;
      if (pushed_bytes) *pushed_bytes += bytes[k];
    } else {
      metrics_.read_push_failed.fetch_add(1, std::memory_order_relaxed);
      MORI_UMBP_WARN(
          "[PeerSsdManager] fan-out push failed for one follower (size={}B); that requester gets a "
          "read error, the leader and every other follower are unaffected",
          bytes[k]);
    }
  }
  metrics_.read_pushed.fetch_add(ok_count, std::memory_order_relaxed);
  return ok_count;
}

std::vector<SsdReadOutcome> PeerSsdManager::PrepareReadBatch(
    const std::vector<std::string>& keys, const std::vector<void*>& dsts,
    const std::vector<size_t>& caps, const std::vector<ReadPushSlot>& push_slots,
    std::vector<bool>* pushed) {
  std::vector<SsdReadOutcome> out(keys.size(), SsdReadOutcome{SsdReadStatus::kNotFound, 0});
  if (pushed) pushed->assign(keys.size(), false);
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
  // Push is strictly opt-in and strictly additive: no engine, no slots, or a
  // wrong-length slot vector all mean "memcpy exactly as before".  A length
  // mismatch is a caller bug, so say so rather than silently half-pushing.
  bool push_enabled = io_engine_ != nullptr && !push_slots.empty();
  if (!push_slots.empty() && push_slots.size() != keys.size()) {
    MORI_UMBP_ERROR("[PeerSsdManager] PrepareReadBatch: push_slots={} but keys={}; push disabled",
                    push_slots.size(), keys.size());
    push_enabled = false;
  }
  const ReadPushSlot kNoPush{};
  auto slot_for = [&](size_t i) -> const ReadPushSlot& {
    return push_enabled ? push_slots[i] : kNoPush;
  };

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
  // Of fanout_count, the followers served by RDMA WRITE instead of memcpy.
  // pushed_here counts what THIS call pushed as a leader; followed_pushed counts
  // the keys THIS call had pushed to it as a follower (what the RPC reports).
  uint64_t push_bytes = 0;
  size_t pushed_here = 0, followed_pushed = 0;

  // Pass 1 — resolve sizes and mark every servable key in flight under ONE lock
  // acquisition (the per-key path pays two per key).
  std::vector<size_t> todo;   // I lead (or independently read) these
  std::vector<size_t> waits;  // I follow these; a leader will fill my buffer
  std::vector<bool> is_leader(keys.size(), false);
  // Per-request handle on the episode we lead or follow.  Held as a shared_ptr
  // so it stays valid even after the slot moves on to a new leader.
  std::vector<std::shared_ptr<ReadEpisode>> episodes(keys.size());
  // Our own entry in a leader's fan-out list, for the keys we follow.  Kept
  // per-request because the leader swaps its list out of the episode on seal,
  // and because each follower's outcome is now its own (RDMA WRITE can fail for
  // one destination and succeed for another).
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
        episodes[i] = slot->episode;
        is_leader[i] = true;
        todo.push_back(i);
      } else if (single_flight_ && !slot->episode->sealed) {
        // Merge window open: hand the leader our buffer and wait on its episode.
        // We always hand it a memcpy destination (our staging slot), and
        // additionally a push destination when we have one — the leader picks,
        // and tells us which it used, so a leader that cannot push still serves
        // us correctly.
        const auto& ps = slot_for(i);
        my_targets[i] = std::make_shared<FollowerTarget>();
        my_targets[i]->dst = dsts[i];
        my_targets[i]->cap = caps[i];
        my_targets[i]->wants_push = ps.push_eligible;
        my_targets[i]->remote = ps.remote_desc;
        my_targets[i]->remote_offset = ps.remote_offset;
        slot->episode->followers.push_back(my_targets[i]);
        episodes[i] = slot->episode;
        waits.push_back(i);
        ++merged;
      } else {
        // Missed the window (or single flight off): read it ourselves.  This is
        // the pre-single-flight path and is always correct, just not merged.
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
  std::vector<bool> ok;
  if (!todo.empty()) {
    const auto t_backend = ssdperf::Now();
    ok = backend_->BatchReadIntoPtr(k, d, z);
    ok.resize(todo.size(), false);
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
    // Deliver outside the lock: holding mutex_ across a page-sized memcpy (let
    // alone an RDMA round trip) would stall every other read/evict here.
    //
    // Our read failed: publish the failure to everyone and post nothing.  A
    // follower must be released with an error, never left to time out, and a
    // BatchWrite of bytes we never read would be worse than useless.
    if (!ok[j]) {
      for (const auto& f : fanout) {
        f->ok = false;
        f->pushed = false;
      }
    } else {
      // Split the fan-out.  A follower is pushed only if it asked to be AND we
      // can be an RDMA source (engine present, our own staging pointer
      // registered); otherwise it drops into the unchanged memcpy loop below.
      const auto& lead_slot = slot_for(i);
      const bool can_push = io_engine_ != nullptr && IsValidDesc(lead_slot.local_desc);
      std::vector<std::shared_ptr<FollowerTarget>> push_group;
      for (const auto& f : fanout) {
        if (can_push && f->wants_push && IsValidDesc(f->remote)) {
          push_group.push_back(f);
          continue;
        }
        // Unchanged path: single-threaded memcpy into the follower's own slot.
        const size_t n = std::min(out[i].size, f->cap);
        std::memcpy(f->dst, dsts[i], n);
        f->ok = true;
        f->pushed = false;
        fanout_bytes += n;
        ++fanout_count;
      }
      // One BatchWrite for this key's whole push group; each posted transfer is
      // waited individually so one bad destination cannot fail the others.
      // fanout_bytes/fanout_count deliberately stay memcpy-only: they exist to
      // measure the copy cost this path is here to remove, so pushed followers
      // are reported under pushed/push_bytes instead.
      if (!push_group.empty()) {
        pushed_here += PushFanout(push_group, lead_slot, out[i].size, &push_bytes);
      }
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ep->done = true;
      ep->cv.notify_all();
    }
  }
  fanout_ms = ssdperf::MsSince(t_fanout);

  // Pass 2c — collect the keys we followed.  Our staging buffer is still alive
  // (we never returned), so a leader's memcpy landed in a valid destination; so
  // did an RDMA WRITE into our own registered destination buffer, which we also
  // never released.  Read our OWN target's outcome, not the episode's: with push
  // in play the leader's read can succeed while one destination's write fails.
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
    if (wait_ok[w] && my_targets[i]->pushed) {
      if (pushed) (*pushed)[i] = true;
      ++followed_pushed;
    }
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
      if (from_device) read_bytes += out[i].size;
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
        "pushed={} push_bytes={} followed_pushed={} push_total={} push_failed_total={} | "
        "singleflight={} dup={} merged={} max_concurrent={} "
        "dup_total={}/{} ({:.1f}%) merged_total={}",
        keys.size(), todo.size(), served, not_owned, read_bytes, total_ms,
        ssdperf::GbPerSec(read_bytes, total_ms), resolve_ms, backend_ms,
        ssdperf::Pct(backend_ms, total_ms), fanout_ms, ssdperf::Pct(fanout_ms, total_ms), wait_ms,
        ssdperf::Pct(wait_ms, total_ms), release_ms, fanout_bytes, fanout_count,
        ssdperf::GbPerSec(fanout_bytes, fanout_ms), pushed_here, push_bytes, followed_pushed,
        metrics_.read_pushed.load(std::memory_order_relaxed),
        metrics_.read_push_failed.load(std::memory_order_relaxed), single_flight_ ? "on" : "off",
        dup, merged, max_concurrent, dup_total, probe_total,
        ssdperf::Pct(static_cast<double>(dup_total), probe_total), merged_total);
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
