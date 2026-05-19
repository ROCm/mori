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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/distributed/master/client_counter_rate_view.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>

#include "mori/utils/mori_log.hpp"
#include "umbp/common/env_time.h"
#include "umbp/distributed/master/master_metrics.h"

namespace mori::umbp {

namespace {

uint64_t MillisecondsToNanoseconds(uint64_t ms) { return ms * 1000ULL * 1000ULL; }

uint64_t MinSampleGapNs() {
  static const uint64_t v = MillisecondsToNanoseconds(
      static_cast<uint64_t>(GetEnvMilliseconds("UMBP_TRANSFER_RATE_MIN_SAMPLE_GAP_MS",
                                               std::chrono::milliseconds(200), /*min_allowed=*/1)
                                .count()));
  return v;
}

uint64_t MaxSampleAgeNs() {
  static const uint64_t v = MillisecondsToNanoseconds(
      static_cast<uint64_t>(GetEnvMilliseconds("UMBP_TRANSFER_RATE_MAX_SAMPLE_AGE_MS",
                                               std::chrono::milliseconds(15000), /*min_allowed=*/1)
                                .count()));
  return v;
}

uint64_t TickMinGapNs() {
  static const uint64_t v = MillisecondsToNanoseconds(
      static_cast<uint64_t>(GetEnvMilliseconds("UMBP_TRANSFER_RATE_TICK_MIN_GAP_MS",
                                               std::chrono::milliseconds(2000), /*min_allowed=*/1)
                                .count()));
  return v;
}

uint32_t EwmaAlphaPermille() {
  static const uint32_t v = [] {
    const uint32_t raw =
        GetEnvUint32("UMBP_TRANSFER_RATE_EWMA_ALPHA_PERMILLE", 300, /*min_allowed=*/1);
    if (raw <= 1000) return raw;
    MORI_UMBP_WARN(
        "env UMBP_TRANSFER_RATE_EWMA_ALPHA_PERMILLE: above 1000 (value='{}'); using default", raw);
    return 300U;
  }();
  return v;
}

double EwmaAlpha() { return static_cast<double>(EwmaAlphaPermille()) / 1000.0; }

std::string ToString(std::string_view value) { return std::string(value.data(), value.size()); }

std::size_t DirectionIndex(HiCacheTransfer direction) {
  return static_cast<std::size_t>(direction);
}

}  // namespace

bool ClientCounterRateView::Accept(std::string_view metric_name) const {
  return metric_name == MORI_UMBP_METRIC_HICACHE_TRANSFER_BYTES_TOTAL;
}

void ClientCounterRateView::OnSample(std::string_view node_id, const ::umbp::MetricSample& sample,
                                     uint64_t now_ns) {
  try {
    if (sample.value_case() != ::umbp::MetricSample::kCounterDelta) {
      Drop(DropReason::kNonCounter);
      return;
    }

    const double value = sample.counter_delta();
    if (!std::isfinite(value) || value < 0.0 ||
        value > static_cast<double>(std::numeric_limits<uint64_t>::max())) {
      Drop(DropReason::kInvalidValue);
      return;
    }

    int direction_count = 0;
    std::optional<HiCacheTransfer> direction;
    for (const auto& label : sample.labels()) {
      if (label.name() != "direction") continue;
      ++direction_count;
      direction = ParseDirectionLabel(label.value());
    }
    if (direction_count != 1) {
      Drop(DropReason::kMissingOrDuplicateDirection);
      return;
    }
    if (!direction.has_value() || *direction == HiCacheTransfer::UNKNOWN) {
      Drop(DropReason::kUnknownDirection);
      return;
    }

    OnSampleInner(node_id, *direction, static_cast<uint64_t>(value), now_ns);
  } catch (const std::exception& e) {
    MORI_UMBP_WARN("[RateView] dropping sample after exception: {}", e.what());
    Drop(DropReason::kInvalidValue);
  } catch (...) {
    MORI_UMBP_WARN("[RateView] dropping sample after unknown exception");
    Drop(DropReason::kInvalidValue);
  }
}

void ClientCounterRateView::OnSampleInner(std::string_view node_id, HiCacheTransfer direction,
                                          uint64_t delta_bytes, uint64_t now_ns) {
  std::unique_lock lock(mu_);
  auto& states = nodes_[ToString(node_id)];
  auto& state = states[DirectionIndex(direction)];

  if (!state.has_baseline) {
    state.has_baseline = true;
    state.last_sample_ns = now_ns;
    state.last_rate_update_ns = now_ns;
    state.pending_delta_bytes = 0;
    return;
  }

  if (std::numeric_limits<uint64_t>::max() - state.pending_delta_bytes < delta_bytes) {
    state.pending_delta_bytes = std::numeric_limits<uint64_t>::max();
  } else {
    state.pending_delta_bytes += delta_bytes;
  }
  if (now_ns <= state.last_sample_ns) return;

  const uint64_t elapsed_ns = now_ns - state.last_sample_ns;
  if (elapsed_ns < MinSampleGapNs()) return;

  const double instant_bps =
      static_cast<double>(state.pending_delta_bytes) * 1e9 / static_cast<double>(elapsed_ns);
  if (!std::isfinite(instant_bps) || instant_bps < 0.0) {
    state.pending_delta_bytes = 0;
    Drop(DropReason::kInvalidValue);
    return;
  }

  const double alpha = EwmaAlpha();
  if (!state.has_rate) {
    state.ewma_bps = instant_bps;
    state.has_rate = true;
  } else {
    state.ewma_bps = alpha * instant_bps + (1.0 - alpha) * state.ewma_bps;
  }
  state.window_ms = elapsed_ns / 1000000ULL;
  state.last_sample_ns = now_ns;
  state.last_rate_update_ns = now_ns;
  state.pending_delta_bytes = 0;
}

void ClientCounterRateView::Tick(std::string_view node_id, uint64_t now_ns) {
  std::unique_lock lock(mu_);
  auto it = nodes_.find(ToString(node_id));
  if (it == nodes_.end()) return;

  const uint64_t tick_min_gap_ns = TickMinGapNs();
  const double decay = 1.0 - EwmaAlpha();
  for (HiCacheTransfer direction : kDirections) {
    auto& state = it->second[DirectionIndex(direction)];
    if (!state.has_rate) continue;
    if (now_ns < state.last_sample_ns || now_ns - state.last_sample_ns < tick_min_gap_ns) {
      continue;
    }
    if (state.last_tick_ns != 0 &&
        (now_ns < state.last_tick_ns || now_ns - state.last_tick_ns < tick_min_gap_ns)) {
      continue;
    }
    state.ewma_bps = std::max(0.0, state.ewma_bps * decay);
    state.last_rate_update_ns = now_ns;
    state.last_tick_ns = now_ns;
  }
}

void ClientCounterRateView::Forget(std::string_view node_id) {
  std::unique_lock lock(mu_);
  nodes_.erase(ToString(node_id));
}

std::vector<HiCacheTransferRate> ClientCounterRateView::Snapshot(std::string_view node_id,
                                                                 uint64_t now_ns) const {
  std::shared_lock lock(mu_);
  auto it = nodes_.find(ToString(node_id));
  if (it == nodes_.end()) return {};

  std::vector<HiCacheTransferRate> out;
  out.reserve(kDirections.size());
  const uint64_t max_age_ns = MaxSampleAgeNs();
  for (HiCacheTransfer direction : kDirections) {
    const auto& state = it->second[DirectionIndex(direction)];
    if (!state.has_rate) continue;
    const uint64_t age_ns =
        now_ns > state.last_rate_update_ns ? now_ns - state.last_rate_update_ns : 0;
    if (age_ns > max_age_ns) continue;
    HiCacheTransferRate rate;
    rate.direction = direction;
    rate.bytes_per_sec = state.ewma_bps;
    rate.rate_age_ms = age_ns / 1000000ULL;
    rate.window_ms = state.window_ms;
    out.push_back(rate);
  }
  return out;
}

uint64_t ClientCounterRateView::DroppedCount(DropReason reason) const {
  const auto index = static_cast<std::size_t>(reason);
  if (index >= dropped_counts_.size()) return 0;
  return dropped_counts_[index].load(std::memory_order_relaxed);
}

std::optional<HiCacheTransfer> ClientCounterRateView::ParseDirectionLabel(std::string_view label) {
  if (label == "l1_to_l2") return HiCacheTransfer::L1_TO_L2;
  if (label == "l2_to_l1") return HiCacheTransfer::L2_TO_L1;
  if (label == "l2_to_l3") return HiCacheTransfer::L2_TO_L3;
  if (label == "l3_to_l2") return HiCacheTransfer::L3_TO_L2;
  return std::nullopt;
}

void ClientCounterRateView::Drop(DropReason reason) {
  const auto index = static_cast<std::size_t>(reason);
  if (index >= dropped_counts_.size()) return;
  dropped_counts_[index].fetch_add(1, std::memory_order_relaxed);
}

}  // namespace mori::umbp
