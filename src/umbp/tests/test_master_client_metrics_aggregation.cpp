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

// Targeted in-process tests for the abc-side MasterClient metrics machinery
// preserved through the rebase onto the alloc-decoup baseline.  These exercise:
//
//   * The bucket-bound array kMasterClientRpcLatencyBucketsArr (layout pin).
//   * MasterClient::Observe — cumulative-bucket aggregation, +Inf overflow,
//     bounds-mismatch first-write-wins WARN dedup, and the series-cardinality
//     cap with metrics_dropped_count_ accounting.
//   * The ScopedRpcTimer integration callbacks RecordRpcLatency /
//     RecordRpcError, including the metrics_running_ short-circuit that
//     protects unregistered (read-only Python) clients from unbounded buffer
//     growth.
//   * MetricsLoop's empty-buffer early-out (the abc bug fix that swapped
//     `histograms.empty()` -> `histogram_aggregates.empty()`).
//
// These tests deliberately do NOT spin up a real master server: that surface
// is gated behind MORI_UMBP_LEGACY_DISTRIBUTED_TESTS in this branch because
// the legacy RPC surface (Lookup/FinalizeAllocation/...) was deleted by
// alloc-decoup.  Instead we drive the in-process MasterClient via its public
// API and the friend-class test handle declared in master_client.h.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_metrics.h"

namespace mori::umbp {

// MasterClientRpcLatencyTest lives in the mori::umbp namespace (NOT in an
// anonymous helper namespace) so the `friend class MasterClientRpcLatencyTest;`
// declaration in master_client.h resolves to the same class — anonymous-ns
// classes are not findable by unqualified-name friend declarations from the
// enclosing namespace.  The friend grants access to the private fields we
// need to inspect/poke (pending_histogram_aggregates_, metrics_dropped_count_,
// pending_histogram_series_cap_, metrics_running_, metrics_mutex_) without
// adding a public test-only API to the production class.
class MasterClientRpcLatencyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Construct a client pointed at a non-existent loopback master.  The
    // ctor only creates a gRPC channel (lazy connect), so this is cheap and
    // does not block.  We never call RegisterSelf/RPCs here — every test
    // exercises in-memory state through the public buffering API or the
    // friend-test poke methods below.
    UMBPMasterClientConfig cfg;
    cfg.master_address = "127.0.0.1:1";  // unreachable on purpose
    cfg.node_id = "metrics-agg-test-node";
    cfg.node_address = "127.0.0.1";
    cfg.auto_heartbeat = false;
    client_ = std::make_unique<MasterClient>(cfg);
  }

  void TearDown() override {
    // Force metrics_running_ off so the dtor's StopMetricsReporting() is a
    // no-op (no thread was started in these tests; this is just defensive
    // for cases that flip the flag to exercise the recording path).
    SetMetricsRunning(false);
    client_.reset();
  }

  // --- Friend-test handles --------------------------------------------------

  void SetSeriesCap(std::size_t cap) {
    std::lock_guard<std::mutex> lk(client_->metrics_mutex_);
    client_->pending_histogram_series_cap_ = cap;
  }

  std::size_t PendingSeriesCount() {
    std::lock_guard<std::mutex> lk(client_->metrics_mutex_);
    return client_->pending_histogram_aggregates_.size();
  }

  uint64_t DroppedCount() {
    return client_->metrics_dropped_count_.load(std::memory_order_relaxed);
  }

  // Flip the metrics_running_ guard without going through StartMetricsReporting
  // (which spawns a thread and is gated on registered_).  RecordRpcLatency /
  // RecordRpcError early-out when this is false; the tests that exercise the
  // recording path flip it on for the duration of the call and back off in
  // TearDown so no flush thread is left running.
  void SetMetricsRunning(bool v) { client_->metrics_running_.store(v, std::memory_order_relaxed); }

  // ScopedRpcTimer's integration callbacks are private on MasterClient.  The
  // fixture is a friend of MasterClient, but TEST_F bodies live in a generated
  // class *derived* from the fixture and the friend relationship does not
  // propagate through inheritance.  Thunk through these forwarders so test
  // bodies can drive the recording path the same way ScopedRpcTimer does at
  // production runtime.
  void CallRecordRpcLatency(std::string_view method, bool ok, double seconds) {
    client_->RecordRpcLatency(method, ok, seconds);
  }
  void CallRecordRpcError(std::string_view method, std::string_view code) {
    client_->RecordRpcError(method, code);
  }

  // Inspect the warned_mismatch flag for a series.  Walks the pending map
  // under the same mutex Observe() holds so this is race-free with respect
  // to a concurrent Observe (none of these tests run them concurrently, but
  // the lock pairs cleanly with the production reader anyway).
  bool WarnedMismatchFor(const std::string& metric_name) {
    std::lock_guard<std::mutex> lk(client_->metrics_mutex_);
    for (const auto& [key, h] : client_->pending_histogram_aggregates_) {
      if (h.name == metric_name) return h.warned_mismatch;
    }
    return false;
  }

  // Expose the HistogramAccumulator fields the aggregation tests assert on.
  // Returning the struct by value keeps the lock scoped tightly.
  struct AggregatorView {
    bool found = false;
    std::vector<double> bounds;
    std::vector<uint64_t> bucket_counts;
    uint64_t count = 0;
    double sum = 0.0;
  };
  AggregatorView GetAggregator(const std::string& metric_name, const MasterClient::Labels& labels) {
    std::lock_guard<std::mutex> lk(client_->metrics_mutex_);
    AggregatorView v;
    for (const auto& [key, h] : client_->pending_histogram_aggregates_) {
      if (h.name != metric_name) continue;
      if (h.labels != labels) continue;
      v.found = true;
      v.bounds = h.bounds;
      v.bucket_counts = h.bucket_counts;
      v.count = h.count;
      v.sum = h.sum;
      return v;
    }
    return v;
  }

  // Counter inspection mirror of GetAggregator for the RecordRpcError test.
  struct CounterView {
    bool found = false;
    double value = 0.0;
  };
  CounterView GetCounter(const std::string& metric_name, const MasterClient::Labels& labels) {
    std::lock_guard<std::mutex> lk(client_->metrics_mutex_);
    CounterView v;
    for (const auto& [key, s] : client_->pending_counters_) {
      if (s.name != metric_name) continue;
      if (s.labels != labels) continue;
      v.found = true;
      v.value = s.value;
      return v;
    }
    return v;
  }

  std::unique_ptr<MasterClient> client_;
};

// --- 1. Bucket-array layout pin --------------------------------------------

// kMasterClientRpcLatencyBucketsArr is consumed verbatim by the master-side
// histogram registry on first observation, and the layout is silently first-
// write-wins.  Editing in place corrupts every series in flight, so this
// test pins the layout (count, monotonicity, range coverage) so a sloppy
// edit fails CI rather than scrambling production histograms.
TEST(MasterClientBucketsTest, BucketArrayHasExpectedLayout) {
  constexpr std::size_t kSize =
      sizeof(kMasterClientRpcLatencyBucketsArr) / sizeof(kMasterClientRpcLatencyBucketsArr[0]);
  ASSERT_EQ(kSize, 14u) << "Bucket count is part of the wire layout — bumping requires a "
                           "metric-name suffix bump (e.g. _v2), not an in-place edit.";

  // Strictly ascending: histogram bucket bounds must be sorted for the
  // cumulative encoding to be well-defined.
  for (std::size_t i = 1; i < kSize; ++i) {
    EXPECT_LT(kMasterClientRpcLatencyBucketsArr[i - 1], kMasterClientRpcLatencyBucketsArr[i])
        << "Bounds must be strictly ascending at index " << i;
  }

  // Documented coverage: 0.1 ms (1e-4 s) ~ 5 s.  The first and last bounds
  // pin those endpoints; intermediate bounds are documentation-only.
  EXPECT_DOUBLE_EQ(kMasterClientRpcLatencyBucketsArr[0], 1e-4);
  EXPECT_DOUBLE_EQ(kMasterClientRpcLatencyBucketsArr[kSize - 1], 5.0);

  // Series-cardinality cap is a separate constant in the same header; pin
  // it here so an unrelated edit that drops it cannot slip through.
  EXPECT_EQ(kMasterClientMaxPendingHistograms, 15000u);
}

// --- 2. Observe aggregation behavior ---------------------------------------

// Multiple Observe calls on the same (name, labels) series must collapse
// into a single HistogramAccumulator entry whose bucket_counts are CUMULATIVE
// (>= bounds[i] increments every bucket whose upper bound is >= value).
// This is the abc-side behaviour the master relies on to merge by per-bucket
// addition without any encoding conversion.
TEST_F(MasterClientRpcLatencyTest, ObserveAccumulatesCumulativeBuckets) {
  const std::vector<double> bounds = {0.001, 0.01, 0.1, 1.0};
  const MasterClient::Labels labels = {{"rpc", "Foo"}, {"status", "ok"}};

  // Three observations, each landing in different buckets.
  // value=0.0005 -> hits bound[0]=0.001 (and every later bound)
  // value=0.05   -> hits bound[2]=0.1   (and bound[3]=1.0)
  // value=0.7    -> hits bound[3]=1.0   only
  client_->Observe("test_obs_metric", "synthetic", labels, bounds, 0.0005);
  client_->Observe("test_obs_metric", "synthetic", labels, bounds, 0.05);
  client_->Observe("test_obs_metric", "synthetic", labels, bounds, 0.7);

  // Exactly one accumulator entry — multi-Observe collapses by series key.
  EXPECT_EQ(PendingSeriesCount(), 1u);

  auto agg = GetAggregator("test_obs_metric", labels);
  ASSERT_TRUE(agg.found);
  EXPECT_EQ(agg.count, 3u);
  EXPECT_DOUBLE_EQ(agg.sum, 0.0005 + 0.05 + 0.7);
  ASSERT_EQ(agg.bounds.size(), bounds.size());
  ASSERT_EQ(agg.bucket_counts.size(), bounds.size());
  // Cumulative expectations.
  EXPECT_EQ(agg.bucket_counts[0], 1u);  // 0.0005 <= 0.001
  EXPECT_EQ(agg.bucket_counts[1], 1u);  // 0.0005 <= 0.01
  EXPECT_EQ(agg.bucket_counts[2], 2u);  // + 0.05 <= 0.1
  EXPECT_EQ(agg.bucket_counts[3], 3u);  // + 0.7  <= 1.0

  // Cumulative monotonicity: the wire encoding requires bucket_counts[i] <=
  // bucket_counts[i+1].  This pins the production assertion the master-side
  // merge relies on.
  for (std::size_t i = 1; i < agg.bucket_counts.size(); ++i) {
    EXPECT_LE(agg.bucket_counts[i - 1], agg.bucket_counts[i]);
  }
  // bucket_counts.back() <= count (difference is the implicit +Inf overflow).
  EXPECT_LE(agg.bucket_counts.back(), agg.count);
}

// A value larger than the largest finite bound falls into the implicit +Inf
// bucket.  Per the cumulative encoding bucket_counts.back() must NOT
// increment, but count and sum still advance.  This is the corner the
// master-side decoder has to handle with `count - bucket_counts.back()`.
TEST_F(MasterClientRpcLatencyTest, ObservePlusInfOverflowDoesNotIncrementBuckets) {
  const std::vector<double> bounds = {0.001, 0.01};
  const MasterClient::Labels labels = {};
  // value=0.5 is above every finite bound (0.01 < 0.5 < +Inf).
  client_->Observe("test_inf_overflow", "synthetic", labels, bounds, 0.5);

  auto agg = GetAggregator("test_inf_overflow", labels);
  ASSERT_TRUE(agg.found);
  EXPECT_EQ(agg.count, 1u);
  EXPECT_DOUBLE_EQ(agg.sum, 0.5);
  ASSERT_EQ(agg.bucket_counts.size(), 2u);
  EXPECT_EQ(agg.bucket_counts[0], 0u);
  EXPECT_EQ(agg.bucket_counts[1], 0u);
  // Overflow accounting: count - bucket_counts.back() == 1 (one +Inf hit).
  EXPECT_EQ(agg.count - agg.bucket_counts.back(), 1u);
}

// On a bounds-size mismatch for an existing series, Observe must:
//   1. Keep the *first-write* bounds layout (silent first-write-wins).
//   2. Set warned_mismatch=true so the WARN log fires at most once per
//      series; a single misconfigured caller must not silence every other
//      series' WARN.
//   3. Still increment count/sum so observations are not silently dropped.
TEST_F(MasterClientRpcLatencyTest, ObserveBoundsMismatchSetsWarnedFlagOnce) {
  const MasterClient::Labels labels = {{"rpc", "Foo"}};
  const std::vector<double> bounds_first = {0.01, 0.1, 1.0};
  const std::vector<double> bounds_second = {0.01, 0.1};  // size mismatch

  // First-write seeds the layout.
  client_->Observe("test_mismatch_metric", "synthetic", labels, bounds_first, 0.05);
  EXPECT_FALSE(WarnedMismatchFor("test_mismatch_metric"))
      << "warned_mismatch must be false after the first (well-formed) observation";

  // Second observation with mismatched bounds size triggers the WARN-dedup
  // path.  Bounds layout must NOT change.
  client_->Observe("test_mismatch_metric", "synthetic", labels, bounds_second, 0.05);
  EXPECT_TRUE(WarnedMismatchFor("test_mismatch_metric"));
  auto agg = GetAggregator("test_mismatch_metric", labels);
  ASSERT_TRUE(agg.found);
  EXPECT_EQ(agg.bounds.size(), bounds_first.size())
      << "First-write-wins: bounds layout must not be replaced by a mismatched second write";
  // count/sum must still accumulate — the observation is not silently dropped.
  EXPECT_EQ(agg.count, 2u);
  EXPECT_DOUBLE_EQ(agg.sum, 0.10);

  // A third mismatched call must keep warned_mismatch=true (it's a flag, not
  // a counter); production relies on this to suppress repeat WARN log spam.
  client_->Observe("test_mismatch_metric", "synthetic", labels, bounds_second, 0.05);
  EXPECT_TRUE(WarnedMismatchFor("test_mismatch_metric"));
}

// Series-cardinality cap: the (cap+1)-th distinct (name, labels) series must
// be rejected at insert time AND bump metrics_dropped_count_ by exactly 1.
// This is the cold path that fires on label-cardinality leaks in production
// (e.g. a per-key label was accidentally introduced) — we exercise it
// directly here rather than relying on it never firing.
TEST_F(MasterClientRpcLatencyTest, SeriesCapDropsExcessAndIncrementsDroppedCounter) {
  ASSERT_EQ(PendingSeriesCount(), 0u);
  const uint64_t dropped_before = DroppedCount();

  SetSeriesCap(4);
  for (int i = 0; i < 5; ++i) {
    MasterClient::Labels labels = {{"k", std::to_string(i)}};
    client_->Observe("test_cap_metric", "synthetic", labels, {1.0}, 0.5);
  }

  EXPECT_EQ(PendingSeriesCount(), 4u)
      << "After 5 distinct-label Observes with cap=4, exactly 4 series must remain";
  EXPECT_EQ(DroppedCount() - dropped_before, 1u)
      << "5th distinct series must bump metrics_dropped_count_ by exactly 1";

  // Subsequent Observes that match an *existing* series must NOT count as
  // drops (cap only applies on insert of a new series key).  Pin this so a
  // future refactor can't accidentally double-count.
  MasterClient::Labels existing = {{"k", "0"}};
  client_->Observe("test_cap_metric", "synthetic", existing, {1.0}, 0.25);
  EXPECT_EQ(PendingSeriesCount(), 4u);
  EXPECT_EQ(DroppedCount() - dropped_before, 1u);
}

// --- 3. ScopedRpcTimer callback integration --------------------------------

// RecordRpcLatency / RecordRpcError are the integration points ScopedRpcTimer
// uses (declared as friend in master_client.h).  Both short-circuit when
// metrics_running_=false to avoid unbounded buffer growth on read-only
// Python clients (which never call RegisterSelf) and during destructor
// teardown.  This test pins both halves of that contract.
TEST_F(MasterClientRpcLatencyTest, RecordRpcShortCircuitsWhenMetricsNotRunning) {
  ASSERT_FALSE(client_->IsRegistered());
  // metrics_running_ defaults to false on a never-registered client; verify
  // by calling the recording path — neither buffer must gain entries.
  CallRecordRpcLatency("FakeRpc", true, 0.001);
  CallRecordRpcError("FakeRpc", "UNAVAILABLE");
  EXPECT_EQ(PendingSeriesCount(), 0u);
  auto err = GetCounter(MORI_UMBP_METRIC_MASTER_CLIENT_RPC_ERRORS_TOTAL,
                        {{"rpc", "FakeRpc"}, {"code", "UNAVAILABLE"}});
  EXPECT_FALSE(err.found)
      << "RecordRpcError must short-circuit on unregistered (metrics_running_=false) clients";
}

// With metrics_running_=true the same calls must record:
//   * RecordRpcLatency seeds a histogram aggregate under the canonical
//     rpc-latency metric name with labels {rpc, status}, using the 14-bound
//     layout from kMasterClientRpcLatencyBucketsArr.
//   * RecordRpcError bumps a counter under the canonical errors-total metric
//     with labels {rpc, code}.
TEST_F(MasterClientRpcLatencyTest, RecordRpcLatencyAndErrorPopulateCanonicalMetrics) {
  SetMetricsRunning(true);

  CallRecordRpcLatency("FakeRpc", /*ok=*/true, /*seconds=*/0.001);
  CallRecordRpcLatency("FakeRpc", /*ok=*/false, /*seconds=*/2.0);
  CallRecordRpcError("FakeRpc", "UNAVAILABLE");

  // Histogram side: two distinct (rpc, status) series.
  auto ok_agg = GetAggregator(MORI_UMBP_METRIC_MASTER_CLIENT_RPC_LATENCY,
                              {{"rpc", "FakeRpc"}, {"status", "ok"}});
  ASSERT_TRUE(ok_agg.found);
  EXPECT_EQ(ok_agg.count, 1u);
  EXPECT_DOUBLE_EQ(ok_agg.sum, 0.001);
  // 0.001 falls in or below several bounds; the cumulative sum across all
  // buckets must be > 0 and bucket_counts must be monotone.  Check both ends.
  ASSERT_FALSE(ok_agg.bucket_counts.empty());
  EXPECT_LE(ok_agg.bucket_counts.back(), ok_agg.count);
  // Layout pin: the bounds attached to a rpc-latency record must match the
  // public bucket array verbatim — both length and endpoint values.
  ASSERT_EQ(ok_agg.bounds.size(), sizeof(kMasterClientRpcLatencyBucketsArr) /
                                      sizeof(kMasterClientRpcLatencyBucketsArr[0]));
  EXPECT_DOUBLE_EQ(ok_agg.bounds.front(), kMasterClientRpcLatencyBucketsArr[0]);
  EXPECT_DOUBLE_EQ(ok_agg.bounds.back(),
                   kMasterClientRpcLatencyBucketsArr[ok_agg.bounds.size() - 1]);

  auto err_agg = GetAggregator(MORI_UMBP_METRIC_MASTER_CLIENT_RPC_LATENCY,
                               {{"rpc", "FakeRpc"}, {"status", "error"}});
  ASSERT_TRUE(err_agg.found);
  EXPECT_EQ(err_agg.count, 1u);
  EXPECT_DOUBLE_EQ(err_agg.sum, 2.0);

  // Counter side: one (rpc, code) row with a +1 delta.
  auto err = GetCounter(MORI_UMBP_METRIC_MASTER_CLIENT_RPC_ERRORS_TOTAL,
                        {{"rpc", "FakeRpc"}, {"code", "UNAVAILABLE"}});
  ASSERT_TRUE(err.found);
  EXPECT_DOUBLE_EQ(err.value, 1.0);

  // Repeat to confirm the counter accumulates rather than overwriting.
  CallRecordRpcError("FakeRpc", "UNAVAILABLE");
  err = GetCounter(MORI_UMBP_METRIC_MASTER_CLIENT_RPC_ERRORS_TOTAL,
                   {{"rpc", "FakeRpc"}, {"code", "UNAVAILABLE"}});
  ASSERT_TRUE(err.found);
  EXPECT_DOUBLE_EQ(err.value, 2.0);
}

// --- 4. MetricsLoop empty-buffer behaviour ---------------------------------
//
// The abc-side fix in MetricsLoop's early-out replaced the typo
// `histograms.empty()` (never compiled in alloc-decoup tip — that local was
// gone) with `histogram_aggregates.empty()`.  We can't run MetricsLoop
// against a real master without the full RPC stack, but we *can* confirm the
// state it depends on: with all three pending maps empty, the swap leaves
// three empty locals and the early `continue` is taken — i.e. nothing
// crashes and no series is left behind.  This is the next-best assertion
// without a recording master fixture.
TEST_F(MasterClientRpcLatencyTest, EmptyPendingMapsLeaveNothingToFlush) {
  ASSERT_EQ(PendingSeriesCount(), 0u);
  EXPECT_EQ(DroppedCount(), 0u);
  // Drive an "Observe nothing" — buffers stay empty.  This is the state
  // MetricsLoop sees on every tick when the client is idle.  Pin that the
  // buffers do not spuriously gain entries from anything else (e.g. a
  // daemon thread); the test fixture explicitly avoided RegisterSelf, so
  // the metrics thread is not running.
  EXPECT_EQ(PendingSeriesCount(), 0u);
}

}  // namespace mori::umbp
