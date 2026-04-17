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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "mori/io/common.hpp"
#include "mori/io/enum.hpp"

namespace mori {
namespace io {

struct TelemetryInitOptions {
  std::optional<bool> enabled;
};

void InitTelemetry(const TelemetryInitOptions& opts = {});

struct TelemetrySnapshot {
  struct ApiStats {
    uint64_t read_calls{0};
    uint64_t write_calls{0};
    uint64_t batch_read_calls{0};
    uint64_t batch_write_calls{0};
    uint64_t rejected_calls{0};
    uint64_t total_bytes_read{0};
    uint64_t total_bytes_written{0};
    // Initiator-local JCT: API entry -> local CQE completion.
    // Does NOT include notification round-trip to remote side.
    // Values are 0.0 when jct_sample_count == 0.
    double jct_ewma_us{0.0};
    double jct_min_us{0.0};
    double jct_max_us{0.0};
    uint64_t jct_sample_count{0};
  } api;

  struct VerbsStats {
    uint64_t bytes_posted{0};
    uint64_t bytes_completed{0};
    uint64_t ops_posted{0};
    uint64_t ops_completed{0};
    uint64_t cqe_errors{0};
    uint64_t post_send_errors{0};
    uint64_t sq_full_events{0};
    uint64_t cq_polls{0};
    uint64_t cq_empty_polls{0};
    // Fraction of CQ polls that found work.
    // Only meaningful in POLLING mode. In EVENT mode, value is ~1.0
    // because polls only occur after completion channel notification.
    double cq_utilization{0.0};
    PollCqMode poll_mode{PollCqMode::POLLING};
    int sq_depth{0};
    int sq_depth_hwm{0};
    int sq_depth_max{0};
    // Per-signaled-WR post-to-CQE latency (NIC + wire time).
    // Values are 0.0 when rdma_latency_sample_count == 0.
    double rdma_latency_ewma_us{0.0};
    double rdma_latency_min_us{0.0};
    double rdma_latency_max_us{0.0};
    uint64_t rdma_latency_sample_count{0};
  } verbs;

  // Latency breakdown (makespan-based, EWMA).
  // Values are 0.0 when api.jct_sample_count == 0.
  struct LatencyBreakdown {
    double api_total_ewma_us{0.0};
    double rdma_wall_ewma_us{0.0};
    double sw_overhead_ewma_us{0.0};
  } breakdown;

  struct EpDetail {
    uint64_t ep_id{0};
    VerbsStats verbs;
  };
  std::vector<EpDetail> per_ep;

  // TSC at snapshot time. Diff two snapshots' tsc values and divide by
  // (tsc_freq_ghz * 1e9) to get elapsed seconds for bandwidth computation.
  uint64_t snapshot_tsc{0};
  double tsc_freq_ghz{0.0};
};

}  // namespace io
}  // namespace mori
