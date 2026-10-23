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

#include <chrono>
#include <cstdint>
#include <mutex>

namespace mori::umbp {

// Where a local resolve spends its time, filled across PoolClient, PeerPool and
// SsdBackend on the calling thread. Profiling only: null unless ranged debug is on.
struct ResolveProf {
  double pool = 0;           // PeerPool::BatchResolve, whole
  double pool_wait = 0;      // operation_mutex_ / lifecycle_mutex_ acquisition
  double pool_hold = 0;      // operation_mutex_ held
  double promote = 0;        // PromoteForRead
  double backend = 0;        // every backend BatchResolve
  double ssd = 0;            // SsdBackend::BatchResolve, whole
  double ssd_wait = 0;       // SsdBackend mutex_ acquisition
  double ssd_hold = 0;       // SsdBackend mutex_ held
  double ssd_scan = 0;       // AcquireStagingSpanLocked, incl. reclaim
  double ssd_sizeof = 0;     // per-key SizeOf under mutex_
  double ssd_read = 0;       // PrepareReadBatch
  double busy_sleep = 0;     // BUSY backoff sleeps
  uint64_t rounds = 0;       // staged parts per call
  uint64_t attempts = 0;     // PeerPool::BatchResolve calls
  uint64_t busy_keys = 0;    // keys SsdBackend answered kBusy
  uint64_t staged_keys = 0;  // keys SsdBackend staged
  uint64_t scan_pages = 0;   // pages first-fit walked
  uint64_t reclaims = 0;     // spans freed by an on-demand reclaim
  uint64_t occupancy = 0;    // staging pages in use, summed at each phase-1 entry
  uint64_t occupancy_n = 0;

  void Add(const ResolveProf& o) {
    pool += o.pool, pool_wait += o.pool_wait, pool_hold += o.pool_hold, promote += o.promote;
    backend += o.backend, ssd += o.ssd, ssd_wait += o.ssd_wait, ssd_hold += o.ssd_hold;
    ssd_scan += o.ssd_scan, ssd_sizeof += o.ssd_sizeof, ssd_read += o.ssd_read;
    busy_sleep += o.busy_sleep, rounds += o.rounds, attempts += o.attempts;
    busy_keys += o.busy_keys, staged_keys += o.staged_keys, scan_pages += o.scan_pages;
    reclaims += o.reclaims, occupancy += o.occupancy, occupancy_n += o.occupancy_n;
  }
};

inline thread_local ResolveProf* t_resolve_prof = nullptr;

// Points t_resolve_prof at `prof` for a scope.
class ScopedResolveProf {
 public:
  explicit ScopedResolveProf(ResolveProf* prof) : prev_(t_resolve_prof) { t_resolve_prof = prof; }
  ~ScopedResolveProf() { t_resolve_prof = prev_; }
  ScopedResolveProf(const ScopedResolveProf&) = delete;
  ScopedResolveProf& operator=(const ScopedResolveProf&) = delete;

 private:
  ResolveProf* prev_;
};

inline double ProfSeconds(std::chrono::steady_clock::time_point a,
                          std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// Adds its lifetime to one ResolveProf field of the active profile.
class ProfTimer {
 public:
  explicit ProfTimer(double ResolveProf::* field)
      : prof_(t_resolve_prof),
        field_(field),
        start_(prof_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}) {
  }
  ~ProfTimer() {
    if (prof_ != nullptr) prof_->*field_ += ProfSeconds(start_, std::chrono::steady_clock::now());
  }
  ProfTimer(const ProfTimer&) = delete;
  ProfTimer& operator=(const ProfTimer&) = delete;

 private:
  ResolveProf* prof_;
  double ResolveProf::* field_;
  std::chrono::steady_clock::time_point start_;
};

// A lock_guard that books the acquisition and the hold separately.
template <typename Mutex>
class ProfLockGuard {
 public:
  ProfLockGuard(Mutex& m, double ResolveProf::* wait, double ResolveProf::* hold)
      : m_(m), prof_(t_resolve_prof), hold_(hold) {
    if (prof_ == nullptr) {
      m_.lock();
      return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    m_.lock();
    acquired_ = std::chrono::steady_clock::now();
    prof_->*wait += ProfSeconds(t0, acquired_);
  }
  ~ProfLockGuard() {
    if (prof_ != nullptr) prof_->*hold_ += ProfSeconds(acquired_, std::chrono::steady_clock::now());
    m_.unlock();
  }
  ProfLockGuard(const ProfLockGuard&) = delete;
  ProfLockGuard& operator=(const ProfLockGuard&) = delete;

 private:
  Mutex& m_;
  ResolveProf* prof_;
  double ResolveProf::* hold_;
  std::chrono::steady_clock::time_point acquired_;
};

}  // namespace mori::umbp
