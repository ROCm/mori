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
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "umbp/distributed/master/client_metric_dispatcher.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {

class ClientCounterRateView : public ClientMetricConsumer {
 public:
  bool Accept(std::string_view metric_name) const override;
  void OnSample(std::string_view node_id, const ::umbp::MetricSample& sample,
                uint64_t now_ns) override;

  void Tick(std::string_view node_id, uint64_t now_ns);
  void Forget(std::string_view node_id);
  std::vector<HiCacheTransferRate> Snapshot(std::string_view node_id, uint64_t now_ns) const;

  enum class DropReason : std::size_t {
    kNonCounter = 0,
    kInvalidValue = 1,
    kMissingOrDuplicateDirection = 2,
    kUnknownDirection = 3,
    kCount = 4,
  };

  uint64_t DroppedCount(DropReason reason) const;

 private:
  struct State {
    uint64_t last_sample_ns = 0;
    uint64_t last_rate_update_ns = 0;
    uint64_t last_tick_ns = 0;
    uint64_t pending_delta_bytes = 0;
    double ewma_bps = 0.0;
    uint64_t window_ms = 0;
    bool has_baseline = false;
    bool has_rate = false;
  };

  static constexpr std::size_t kDirectionSlots = 5;
  static constexpr std::array<HiCacheTransfer, 4> kDirections = {
      HiCacheTransfer::L1_TO_L2,
      HiCacheTransfer::L2_TO_L1,
      HiCacheTransfer::L2_TO_L3,
      HiCacheTransfer::L3_TO_L2,
  };

  static std::optional<HiCacheTransfer> ParseDirectionLabel(std::string_view label);
  void OnSampleInner(std::string_view node_id, HiCacheTransfer direction, uint64_t delta_bytes,
                     uint64_t now_ns);
  void Drop(DropReason reason);

  mutable std::shared_mutex mu_;
  std::unordered_map<std::string, std::array<State, kDirectionSlots>> nodes_;
  std::array<std::atomic<uint64_t>, static_cast<std::size_t>(DropReason::kCount)> dropped_counts_{};
};

}  // namespace mori::umbp
