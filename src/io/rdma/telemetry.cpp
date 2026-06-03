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
#include "src/io/rdma/telemetry.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>

#include "mori/io/logging.hpp"
#include "mori/utils/env_utils.hpp"

namespace mori {
namespace io {

// ── TelemetryConfig statics ─────────────────────────────────────

std::atomic<bool> TelemetryConfig::enabled_{false};
std::atomic<bool> TelemetryConfig::initialized_{false};
double TelemetryConfig::tsc_ghz_{0.0};

static void CalibrateIfNeeded() {
  if (TelemetryConfig::TscGhz() > 0.0) return;

  auto t0 = std::chrono::steady_clock::now();
  uint64_t tsc0 = __rdtsc();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  uint64_t tsc1 = __rdtsc();
  auto t1 = std::chrono::steady_clock::now();

  double elapsed_ns =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
  TelemetryConfig::SetTscGhz(elapsed_ns > 0 ? static_cast<double>(tsc1 - tsc0) / elapsed_ns : 1.0);
  MORI_IO_INFO("Telemetry: TSC calibrated at {:.3f} GHz", TelemetryConfig::TscGhz());
}

void TelemetryConfig::Init() {
  bool expected = false;
  if (!initialized_.compare_exchange_strong(expected, true)) return;

  if (!env::IsEnvVarEnabled("MORI_IO_TELEMETRY_ENABLED")) return;
  enabled_.store(true, std::memory_order_relaxed);

  bool has_constant_tsc = false;
  bool has_nonstop_tsc = false;
  {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
      if (line.find("flags") == std::string::npos) continue;
      has_constant_tsc = line.find("constant_tsc") != std::string::npos;
      has_nonstop_tsc = line.find("nonstop_tsc") != std::string::npos;
      break;
    }
  }
  if (!has_constant_tsc || !has_nonstop_tsc) {
    MORI_IO_WARN(
        "Telemetry: CPU lacks constant_tsc ({}) or nonstop_tsc ({}); "
        "TSC-based latency measurements may drift",
        has_constant_tsc, has_nonstop_tsc);
  }

  CalibrateIfNeeded();
}

void TelemetryConfig::Apply(const TelemetryInitOptions& opts) {
  if (opts.enabled.has_value()) {
    enabled_.store(opts.enabled.value(), std::memory_order_relaxed);
    if (opts.enabled.value()) {
      CalibrateIfNeeded();
    }
  }
}

void InitTelemetry(const TelemetryInitOptions& opts) {
  TelemetryConfig::Init();
  TelemetryConfig::Apply(opts);
}

// ── LatencyStats ────────────────────────────────────────────────

void LatencyStats::Update(uint64_t delta_tsc) {
  uint64_t prev = ewma_tsc.load(std::memory_order_relaxed);
  uint64_t next;
  if (prev == 0) {
    next = delta_tsc;
  } else {
    next = prev + (static_cast<int64_t>(delta_tsc - prev) >> 4);
  }
  ewma_tsc.store(next, std::memory_order_relaxed);

  uint64_t cur_min = min_tsc.load(std::memory_order_relaxed);
  while (delta_tsc < cur_min &&
         !min_tsc.compare_exchange_weak(cur_min, delta_tsc, std::memory_order_relaxed)) {
  }

  uint64_t cur_max = max_tsc.load(std::memory_order_relaxed);
  while (delta_tsc > cur_max &&
         !max_tsc.compare_exchange_weak(cur_max, delta_tsc, std::memory_order_relaxed)) {
  }

  count.fetch_add(1, std::memory_order_relaxed);
}

// ── TelemetryRegistry ───────────────────────────────────────────

void TelemetryRegistry::UpdateEwmaCas(std::atomic<uint64_t>& ewma, uint64_t sample) {
  uint64_t prev = ewma.load(std::memory_order_relaxed);
  uint64_t next;
  do {
    if (prev == 0) {
      next = sample;
    } else {
      next = prev + (static_cast<int64_t>(sample - prev) >> 4);
    }
  } while (!ewma.compare_exchange_weak(prev, next, std::memory_order_relaxed));
}

void TelemetryRegistry::RecordJct(uint64_t jct_tsc, uint64_t rdma_wall_tsc,
                                  uint64_t sw_overhead_tsc) {
  UpdateEwmaCas(jct_ewma_tsc_, jct_tsc);
  UpdateEwmaCas(rdma_wall_ewma_tsc_, rdma_wall_tsc);
  UpdateEwmaCas(sw_overhead_ewma_tsc_, sw_overhead_tsc);

  uint64_t cur_min = jct_min_tsc_.load(std::memory_order_relaxed);
  while (jct_tsc < cur_min &&
         !jct_min_tsc_.compare_exchange_weak(cur_min, jct_tsc, std::memory_order_relaxed)) {
  }

  uint64_t cur_max = jct_max_tsc_.load(std::memory_order_relaxed);
  while (jct_tsc > cur_max &&
         !jct_max_tsc_.compare_exchange_weak(cur_max, jct_tsc, std::memory_order_relaxed)) {
  }

  jct_count_.fetch_add(1, std::memory_order_relaxed);
}

std::shared_ptr<EpTelemetryState> TelemetryRegistry::CreateEpTelemetry(
    EndpointId id, const std::shared_ptr<std::atomic<int>>& sq_depth, int sq_depth_max) {
  auto state = std::make_shared<EpTelemetryState>();
  state->id = id;
  state->sq_depth = sq_depth;
  state->sq_depth_max = sq_depth_max;

  std::unique_lock lock(mu_);
  ep_telemetry_[id] = state;
  return state;
}

TelemetrySnapshot TelemetryRegistry::Snapshot() const {
  TelemetrySnapshot snap{};
  snap.snapshot_tsc = __rdtsc();
  snap.tsc_freq_ghz = TelemetryConfig::TscGhz();
  auto relaxed = std::memory_order_relaxed;

  // API counters
  snap.api.read_calls = api_counters_.read_calls.load(relaxed);
  snap.api.write_calls = api_counters_.write_calls.load(relaxed);
  snap.api.batch_read_calls = api_counters_.batch_read_calls.load(relaxed);
  snap.api.batch_write_calls = api_counters_.batch_write_calls.load(relaxed);
  snap.api.rejected_calls = api_counters_.rejected_calls.load(relaxed);
  snap.api.total_bytes_read = api_counters_.total_bytes_read.load(relaxed);
  snap.api.total_bytes_written = api_counters_.total_bytes_written.load(relaxed);

  // JCT stats
  uint64_t jc = jct_count_.load(relaxed);
  snap.api.jct_sample_count = jc;
  if (jc > 0) {
    snap.api.jct_ewma_us = TscToUs(jct_ewma_tsc_.load(relaxed));
    snap.api.jct_min_us = TscToUs(jct_min_tsc_.load(relaxed));
    snap.api.jct_max_us = TscToUs(jct_max_tsc_.load(relaxed));
  }

  // JCT breakdown
  if (jc > 0) {
    snap.breakdown.api_total_ewma_us = snap.api.jct_ewma_us;
    snap.breakdown.rdma_wall_ewma_us = TscToUs(rdma_wall_ewma_tsc_.load(relaxed));
    snap.breakdown.sw_overhead_ewma_us = TscToUs(sw_overhead_ewma_tsc_.load(relaxed));
  }

  // Per-EP aggregation
  snap.verbs.poll_mode = poll_mode_;

  uint64_t total_latency_samples = 0;
  double weighted_ewma_sum = 0.0;
  double agg_min = 0.0;
  double agg_max = 0.0;
  bool has_any_latency = false;

  {
    std::shared_lock lock(mu_);
    for (const auto& [id, ept] : ep_telemetry_) {
      TelemetrySnapshot::EpDetail detail;
      detail.ep_id = id;
      detail.verbs.bytes_posted = ept->counters.bytes_posted.load(relaxed);
      detail.verbs.bytes_completed = ept->counters.bytes_completed.load(relaxed);
      detail.verbs.ops_posted = ept->counters.ops_posted.load(relaxed);
      detail.verbs.ops_completed = ept->counters.ops_completed.load(relaxed);
      detail.verbs.cqe_errors = ept->counters.cqe_errors.load(relaxed);
      detail.verbs.sq_full_events = ept->counters.sq_full_events.load(relaxed);
      detail.verbs.post_send_errors = ept->counters.post_send_errors.load(relaxed);
      detail.verbs.cq_polls = ept->counters.cq_poll_count.load(relaxed);
      detail.verbs.cq_empty_polls = ept->counters.cq_empty_poll_count.load(relaxed);
      detail.verbs.poll_mode = poll_mode_;
      detail.verbs.sq_depth = ept->sq_depth ? ept->sq_depth->load(relaxed) : 0;
      detail.verbs.sq_depth_hwm = ept->counters.sq_depth_hwm.load(relaxed);
      detail.verbs.sq_depth_max = ept->sq_depth_max;

      if (detail.verbs.cq_polls > 0) {
        detail.verbs.cq_utilization =
            1.0 - static_cast<double>(detail.verbs.cq_empty_polls) / detail.verbs.cq_polls;
      }

      uint64_t sc = ept->stats.count.load(relaxed);
      detail.verbs.rdma_latency_sample_count = sc;
      if (sc > 0) {
        detail.verbs.rdma_latency_ewma_us = TscToUs(ept->stats.ewma_tsc.load(relaxed));
        detail.verbs.rdma_latency_min_us = TscToUs(ept->stats.min_tsc.load(relaxed));
        detail.verbs.rdma_latency_max_us = TscToUs(ept->stats.max_tsc.load(relaxed));

        weighted_ewma_sum += detail.verbs.rdma_latency_ewma_us * sc;
        total_latency_samples += sc;
        if (!has_any_latency) {
          agg_min = detail.verbs.rdma_latency_min_us;
          agg_max = detail.verbs.rdma_latency_max_us;
          has_any_latency = true;
        } else {
          agg_min = std::min(agg_min, detail.verbs.rdma_latency_min_us);
          agg_max = std::max(agg_max, detail.verbs.rdma_latency_max_us);
        }
      }

      // Aggregate counters
      snap.verbs.bytes_posted += detail.verbs.bytes_posted;
      snap.verbs.bytes_completed += detail.verbs.bytes_completed;
      snap.verbs.ops_posted += detail.verbs.ops_posted;
      snap.verbs.ops_completed += detail.verbs.ops_completed;
      snap.verbs.cqe_errors += detail.verbs.cqe_errors;
      snap.verbs.sq_full_events += detail.verbs.sq_full_events;
      snap.verbs.post_send_errors += detail.verbs.post_send_errors;
      snap.verbs.cq_polls += detail.verbs.cq_polls;
      snap.verbs.cq_empty_polls += detail.verbs.cq_empty_polls;
      snap.verbs.sq_depth += detail.verbs.sq_depth;
      snap.verbs.sq_depth_hwm += detail.verbs.sq_depth_hwm;
      snap.verbs.sq_depth_max += detail.verbs.sq_depth_max;

      snap.per_ep.push_back(detail);
    }
  }

  // Aggregated CQ utilization
  if (snap.verbs.cq_polls > 0) {
    snap.verbs.cq_utilization =
        1.0 - static_cast<double>(snap.verbs.cq_empty_polls) / snap.verbs.cq_polls;
  }

  // Aggregated RDMA latency
  snap.verbs.rdma_latency_sample_count = total_latency_samples;
  if (total_latency_samples > 0) {
    snap.verbs.rdma_latency_ewma_us = weighted_ewma_sum / total_latency_samples;
    snap.verbs.rdma_latency_min_us = agg_min;
    snap.verbs.rdma_latency_max_us = agg_max;
  }

  return snap;
}

}  // namespace io
}  // namespace mori
