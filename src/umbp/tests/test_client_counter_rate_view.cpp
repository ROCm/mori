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
#include <gtest/gtest.h>

#include <cmath>

#include "umbp.pb.h"
#include "umbp/distributed/master/client_counter_rate_view.h"
#include "umbp/distributed/master/master_metrics.h"

namespace mori::umbp {
namespace {

constexpr uint64_t kMs = 1000ULL * 1000ULL;

::umbp::MetricSample MakeTransferSample(const char* direction, double delta) {
  ::umbp::MetricSample sample;
  sample.set_name(MORI_UMBP_METRIC_HICACHE_TRANSFER_BYTES_TOTAL);
  sample.set_help(MORI_UMBP_METRIC_HICACHE_TRANSFER_BYTES_TOTAL_HELP);
  auto* label = sample.add_labels();
  label->set_name("direction");
  label->set_value(direction);
  sample.set_counter_delta(delta);
  return sample;
}

TEST(ClientCounterRateView, BuildsBaselineThenEwmaAndIdleDecay) {
  ClientCounterRateView view;
  const uint64_t t0 = 1000ULL * kMs;

  view.OnSample("node-a", MakeTransferSample("l2_to_l1", 1000.0), t0);
  EXPECT_TRUE(view.Snapshot("node-a", t0).empty());

  view.OnSample("node-a", MakeTransferSample("l2_to_l1", 2000.0), t0 + 1000 * kMs);
  auto rates = view.Snapshot("node-a", t0 + 1000 * kMs);
  ASSERT_EQ(rates.size(), 1u);
  EXPECT_EQ(rates[0].direction, HiCacheTransfer::L2_TO_L1);
  EXPECT_NEAR(rates[0].bytes_per_sec, 2000.0, 1e-6);
  EXPECT_EQ(rates[0].rate_age_ms, 0u);
  EXPECT_EQ(rates[0].window_ms, 1000u);

  view.OnSample("node-a", MakeTransferSample("l2_to_l1", 4000.0), t0 + 2000 * kMs);
  rates = view.Snapshot("node-a", t0 + 2000 * kMs);
  ASSERT_EQ(rates.size(), 1u);
  EXPECT_NEAR(rates[0].bytes_per_sec, 2600.0, 1e-6);

  view.Tick("node-a", t0 + 2500 * kMs);
  rates = view.Snapshot("node-a", t0 + 2500 * kMs);
  ASSERT_EQ(rates.size(), 1u);
  EXPECT_NEAR(rates[0].bytes_per_sec, 2600.0, 1e-6);

  view.Tick("node-a", t0 + 5000 * kMs);
  rates = view.Snapshot("node-a", t0 + 5000 * kMs);
  ASSERT_EQ(rates.size(), 1u);
  EXPECT_NEAR(rates[0].bytes_per_sec, 1820.0, 1e-6);
  EXPECT_EQ(rates[0].rate_age_ms, 0u);

  view.Tick("node-a", t0 + 5500 * kMs);
  rates = view.Snapshot("node-a", t0 + 5500 * kMs);
  ASSERT_EQ(rates.size(), 1u);
  EXPECT_NEAR(rates[0].bytes_per_sec, 1820.0, 1e-6);

  view.Forget("node-a");
  EXPECT_TRUE(view.Snapshot("node-a", t0 + 5500 * kMs).empty());
}

TEST(ClientCounterRateView, MinGapAccumulatesAcceptedDeltas) {
  ClientCounterRateView view;
  const uint64_t t0 = 2000ULL * kMs;

  view.OnSample("node-a", MakeTransferSample("l1_to_l2", 1000.0), t0);
  view.OnSample("node-a", MakeTransferSample("l1_to_l2", 100.0), t0 + 100 * kMs);
  EXPECT_TRUE(view.Snapshot("node-a", t0 + 100 * kMs).empty());

  view.OnSample("node-a", MakeTransferSample("l1_to_l2", 150.0), t0 + 250 * kMs);
  auto rates = view.Snapshot("node-a", t0 + 250 * kMs);
  ASSERT_EQ(rates.size(), 1u);
  EXPECT_EQ(rates[0].direction, HiCacheTransfer::L1_TO_L2);
  EXPECT_NEAR(rates[0].bytes_per_sec, 1000.0, 1e-6);
  EXPECT_EQ(rates[0].window_ms, 250u);
}

TEST(ClientCounterRateView, TickAntiBurstMergesCloseHeartbeats) {
  ClientCounterRateView view;
  const uint64_t t0 = 2500ULL * kMs;

  view.OnSample("node-a", MakeTransferSample("l1_to_l2", 1000.0), t0);
  view.OnSample("node-a", MakeTransferSample("l1_to_l2", 2000.0), t0 + 1000 * kMs);
  auto rates = view.Snapshot("node-a", t0 + 1000 * kMs);
  ASSERT_EQ(rates.size(), 1u);
  const double before = rates[0].bytes_per_sec;

  view.Tick("node-a", t0 + 4000 * kMs);
  rates = view.Snapshot("node-a", t0 + 4000 * kMs);
  ASSERT_EQ(rates.size(), 1u);
  const double after_first = rates[0].bytes_per_sec;
  EXPECT_LT(after_first, before);

  view.Tick("node-a", t0 + 4100 * kMs);
  rates = view.Snapshot("node-a", t0 + 4100 * kMs);
  ASSERT_EQ(rates.size(), 1u);
  EXPECT_DOUBLE_EQ(rates[0].bytes_per_sec, after_first);
}

TEST(ClientCounterRateView, DropsInvalidInputsWithoutStatePollution) {
  ClientCounterRateView view;
  const uint64_t now = 3000ULL * kMs;

  ::umbp::MetricSample gauge = MakeTransferSample("l2_to_l3", 1.0);
  gauge.clear_counter_delta();
  gauge.set_gauge_value(1.0);
  view.OnSample("node-a", gauge, now);
  EXPECT_EQ(view.DroppedCount(ClientCounterRateView::DropReason::kNonCounter), 1u);

  view.OnSample("node-a", MakeTransferSample("l2_to_l3", -1.0), now);
  view.OnSample("node-a", MakeTransferSample("l2_to_l3", std::nan("")), now);
  EXPECT_EQ(view.DroppedCount(ClientCounterRateView::DropReason::kInvalidValue), 2u);

  ::umbp::MetricSample missing;
  missing.set_name(MORI_UMBP_METRIC_HICACHE_TRANSFER_BYTES_TOTAL);
  missing.set_counter_delta(1.0);
  view.OnSample("node-a", missing, now);

  ::umbp::MetricSample duplicate = MakeTransferSample("l2_to_l3", 1.0);
  auto* label = duplicate.add_labels();
  label->set_name("direction");
  label->set_value("l3_to_l2");
  view.OnSample("node-a", duplicate, now);
  EXPECT_EQ(view.DroppedCount(ClientCounterRateView::DropReason::kMissingOrDuplicateDirection), 2u);

  view.OnSample("node-a", MakeTransferSample("bad_direction", 1.0), now);
  EXPECT_EQ(view.DroppedCount(ClientCounterRateView::DropReason::kUnknownDirection), 1u);
  EXPECT_TRUE(view.Snapshot("node-a", now).empty());
}

}  // namespace
}  // namespace mori::umbp
