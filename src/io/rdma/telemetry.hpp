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

#include <x86intrin.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <unordered_map>

#include "mori/io/telemetry.hpp"
#include "src/io/rdma/common.hpp"

namespace mori {
namespace io {

// ── Global config (read-only after init) ─────────────────────────
class TelemetryConfig {
 public:
  static void Init();
  static void Apply(const TelemetryInitOptions& opts);
  static bool Enabled() { return enabled_.load(std::memory_order_relaxed); }
  static double TscGhz() { return tsc_ghz_; }
  static void SetTscGhz(double v) { tsc_ghz_ = v; }

 private:
  static std::atomic<bool> enabled_;
  static std::atomic<bool> initialized_;
  static double tsc_ghz_;
};

#define MORI_IO_TELEM_UNLIKELY(expr) __builtin_expect(!!(expr), 0)

// ── Per-EP verbs counters (cache-line aligned) ───────────────────
struct alignas(64) VerbsCounterSet {
  // --- cache line 1: post path (written by post threads) ---
  std::atomic<uint64_t> bytes_posted{0};
  std::atomic<uint64_t> ops_posted{0};
  std::atomic<uint64_t> sq_full_events{0};
  std::atomic<uint64_t> post_send_errors{0};
  char pad0_[64 - 4 * sizeof(std::atomic<uint64_t>)];

  // --- cache line 2: completion path (written by CQ poll thread) ---
  std::atomic<uint64_t> bytes_completed{0};
  std::atomic<uint64_t> ops_completed{0};
  std::atomic<uint64_t> cqe_errors{0};
  char pad1_[64 - 3 * sizeof(std::atomic<uint64_t>)];

  // --- cache line 3: CQ poll stats (CQ poll thread only) ---
  std::atomic<uint64_t> cq_poll_count{0};
  std::atomic<uint64_t> cq_empty_poll_count{0};
  char pad2_[64 - 2 * sizeof(std::atomic<uint64_t>)];

  // --- cache line 4: SQ depth high watermark (written by post threads) ---
  std::atomic<int> sq_depth_hwm{0};
  char pad3_[64 - sizeof(std::atomic<int>)];
};

// ── API-level counters ───────────────────────────────────────────
struct alignas(64) ApiCounterSet {
  std::atomic<uint64_t> read_calls{0};
  std::atomic<uint64_t> write_calls{0};
  std::atomic<uint64_t> batch_read_calls{0};
  std::atomic<uint64_t> batch_write_calls{0};
  std::atomic<uint64_t> rejected_calls{0};
  std::atomic<uint64_t> total_bytes_read{0};
  std::atomic<uint64_t> total_bytes_written{0};
};

// ── Per-EP RDMA latency stats (EWMA + min/max) ──────────────────
//    Updated by CQ poll thread only — no cross-thread contention.
struct LatencyStats {
  std::atomic<uint64_t> ewma_tsc{0};
  std::atomic<uint64_t> min_tsc{UINT64_MAX};
  std::atomic<uint64_t> max_tsc{0};
  std::atomic<uint64_t> count{0};

  void Update(uint64_t delta_tsc);
};

// ── Shared per-EP telemetry state ────────────────────────────────
struct EpTelemetryState {
  EndpointId id{0};
  VerbsCounterSet counters;
  LatencyStats stats;
  std::shared_ptr<std::atomic<int>> sq_depth;
  int sq_depth_max{0};
};

// ── Per-backend registry ─────────────────────────────────────────
class TelemetryRegistry {
 public:
  ApiCounterSet& ApiCounters() { return api_counters_; }
  PollCqMode GetPollMode() const { return poll_mode_; }
  void SetPollMode(PollCqMode m) { poll_mode_ = m; }

  std::shared_ptr<EpTelemetryState> CreateEpTelemetry(
      EndpointId id, const std::shared_ptr<std::atomic<int>>& sq_depth, int sq_depth_max);

  // May be called concurrently from multiple CQ poll threads.
  void RecordJct(uint64_t jct_tsc, uint64_t rdma_wall_tsc, uint64_t sw_overhead_tsc);

  TelemetrySnapshot Snapshot() const;

 private:
  ApiCounterSet api_counters_;
  PollCqMode poll_mode_{PollCqMode::POLLING};

  mutable std::shared_mutex mu_;
  std::unordered_map<EndpointId, std::shared_ptr<EpTelemetryState>> ep_telemetry_;

  std::atomic<uint64_t> jct_ewma_tsc_{0};
  std::atomic<uint64_t> jct_min_tsc_{UINT64_MAX};
  std::atomic<uint64_t> jct_max_tsc_{0};
  std::atomic<uint64_t> jct_count_{0};

  std::atomic<uint64_t> rdma_wall_ewma_tsc_{0};
  std::atomic<uint64_t> sw_overhead_ewma_tsc_{0};

  static void UpdateEwmaCas(std::atomic<uint64_t>& ewma, uint64_t sample);
};

inline double TscToUs(uint64_t tsc) {
  return static_cast<double>(tsc) / (TelemetryConfig::TscGhz() * 1000.0);
}

}  // namespace io
}  // namespace mori
