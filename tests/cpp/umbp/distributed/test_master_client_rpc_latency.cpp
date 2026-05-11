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

// Tests for ScopedRpcTimer / MasterClient RPC latency instrumentation.
//
// Uses a fake recording gRPC server (heartbeat interval set to 100ms so the
// MasterClient's flush thread runs the same cadence) and inspects the
// captured ReportMetrics requests directly — that is the layer at which
// the timer's effect is visible from outside the client process.

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "umbp.grpc.pb.h"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_metrics.h"

namespace mori::umbp {
namespace {

static uint16_t AllocPort() {
  static std::atomic<uint16_t> next{0};
  if (next.load() == 0) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    next.store(static_cast<uint16_t>(60000 + (std::rand() % 2000)));
  }
  return next.fetch_add(10);
}

// Records every ReportMetrics request and replies OK to RegisterClient with
// a configurable heartbeat interval (which the client also reuses as its
// metrics flush interval, see MetricsReportIntervalMs).
class RecordingMasterService final : public ::umbp::UMBPMaster::Service {
 public:
  explicit RecordingMasterService(int heartbeat_interval_ms)
      : heartbeat_interval_ms_(heartbeat_interval_ms) {}

  grpc::Status RegisterClient(grpc::ServerContext*, const ::umbp::RegisterClientRequest*,
                              ::umbp::RegisterClientResponse* resp) override {
    resp->set_heartbeat_interval_ms(heartbeat_interval_ms_);
    return grpc::Status::OK;
  }

  grpc::Status UnregisterClient(grpc::ServerContext*, const ::umbp::UnregisterClientRequest*,
                                ::umbp::UnregisterClientResponse*) override {
    return grpc::Status::OK;
  }

  grpc::Status Heartbeat(grpc::ServerContext*, const ::umbp::HeartbeatRequest*,
                         ::umbp::HeartbeatResponse* resp) override {
    resp->set_status(::umbp::CLIENT_STATUS_ALIVE);
    return grpc::Status::OK;
  }

  // RouteGet is the workhorse RPC we exercise for instrumentation tests:
  // it has a single-key shape and a found=false-on-OK return that lets us
  // hammer it without changing master state.  When fail_route_get_=true it
  // returns UNAVAILABLE deterministically so the ErrorPath test does not
  // depend on TCP/channel reconnection timing.
  grpc::Status RouteGet(grpc::ServerContext*, const ::umbp::RouteGetRequest*,
                        ::umbp::RouteGetResponse* resp) override {
    if (fail_route_get_.load()) {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE, "injected by test");
    }
    resp->set_found(false);
    return grpc::Status::OK;
  }

  void SetFailRouteGet(bool fail) { fail_route_get_.store(fail); }

  // Used to test that read-only Python-style clients (no RegisterSelf) don't
  // leak entries into pending_histogram_aggregates_.
  grpc::Status MatchExternalKv(grpc::ServerContext*, const ::umbp::MatchExternalKvRequest*,
                               ::umbp::MatchExternalKvResponse*) override {
    return grpc::Status::OK;
  }

  grpc::Status ReportMetrics(grpc::ServerContext*, const ::umbp::ReportMetricsRequest* req,
                             ::umbp::ReportMetricsResponse*) override {
    std::lock_guard<std::mutex> lock(mu_);
    requests_.push_back(*req);
    cv_.notify_all();
    return grpc::Status::OK;
  }

  bool WaitForReport(std::chrono::milliseconds timeout = std::chrono::milliseconds(2000)) {
    std::unique_lock<std::mutex> lock(mu_);
    return cv_.wait_for(lock, timeout, [this] { return !requests_.empty(); });
  }

  std::vector<::umbp::ReportMetricsRequest> Requests() {
    std::lock_guard<std::mutex> lock(mu_);
    return requests_;
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mu_);
    requests_.clear();
  }

 private:
  int heartbeat_interval_ms_;
  std::atomic<bool> fail_route_get_{false};
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<::umbp::ReportMetricsRequest> requests_;
};

static std::vector<::umbp::MetricSample> CollectSamples(
    const std::vector<::umbp::ReportMetricsRequest>& reqs) {
  std::vector<::umbp::MetricSample> out;
  for (const auto& r : reqs)
    for (const auto& s : r.metrics()) out.push_back(s);
  return out;
}

static std::string LabelValue(const ::umbp::MetricSample& s, const std::string& key) {
  for (const auto& l : s.labels()) {
    if (l.name() == key) return l.value();
  }
  return {};
}

}  // namespace

// MasterClientRpcLatencyTest lives in the mori::umbp namespace (not the
// anonymous helpers ns above) so the `friend class MasterClientRpcLatencyTest;`
// declaration in master_client.h resolves to the same class — anonymous-ns
// classes are not findable by unqualified-name friend declarations from the
// enclosing namespace.
class MasterClientRpcLatencyTest : public ::testing::Test {
 protected:
  static constexpr int kFlushIntervalMs = 100;

  void SetUp() override {
    port_ = AllocPort();
    address_ = "127.0.0.1:" + std::to_string(port_);
    service_ = std::make_unique<RecordingMasterService>(kFlushIntervalMs);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address_, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);
  }

  void TearDown() override {
    client_.reset();
    if (server_) {
      server_->Shutdown(std::chrono::system_clock::now() + std::chrono::milliseconds(500));
      server_->Wait();
    }
  }

  void StartClientAndRegister(const std::string& node_id = "rpclat-test-node") {
    UMBPMasterClientConfig cfg;
    cfg.node_id = node_id;
    cfg.node_address = "127.0.0.1";
    cfg.master_address = address_;
    client_ = std::make_unique<MasterClient>(cfg);
    std::map<TierType, TierCapacity> caps;
    caps[TierType::DRAM] = {1 << 20, 1 << 20};
    auto status = client_->RegisterSelf(caps);
    ASSERT_TRUE(status.ok()) << status.error_message();
  }

  // Test-only hook: shrink MasterClient's series-cardinality cap so the
  // CapEnforcedAndDropCounterIncrements test exercises the cold drop branch
  // deterministically.  This fixture is `friend class` of MasterClient (see
  // master_client.h), so this method has access to the private field.
  // TEST_F bodies live in a generated class derived from this fixture; that
  // derived class is NOT itself a friend, so it must go through this helper.
  // Holds metrics_mutex_ during the write so it pairs cleanly with the
  // production readers (Observe + MetricsLoop) under TSan.
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
  // Quiesce the MetricsLoop thread so a test can fire Observes and inspect
  // pending_histogram_aggregates_ / metrics_dropped_count_ in memory without
  // racing the 100 ms flush tick.  Observe()/AddCounter()/SetGauge() do not
  // gate on metrics_running_, so they keep functioning while the flush loop
  // is paused; the buffered state survives the join and is picked up by the
  // next flush after StartMetricsThreadForTesting() restarts the loop.
  void StopMetricsThreadForTesting() { client_->StopMetricsReporting(); }
  void StartMetricsThreadForTesting() { client_->StartMetricsReporting(); }

  uint16_t port_ = 0;
  std::string address_;
  std::unique_ptr<RecordingMasterService> service_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<MasterClient> client_;
};

// N successful RouteGets must produce a single histogram_aggregate sample
// labelled rpc=RouteGet,status=ok with count() >= N (client-side aggregation
// collapses per-observation samples into one per series), and ReportMetrics
// itself must never appear (self-feedback would re-bias the metric).
TEST_F(MasterClientRpcLatencyTest, RouteGetHistogramObservedAndReportMetricsExcluded) {
  StartClientAndRegister();
  service_->Clear();

  constexpr int N = 5;
  for (int i = 0; i < N; ++i) {
    std::optional<RouteGetResult> out;
    auto status = client_->RouteGet("test-key-" + std::to_string(i), {}, &out);
    ASSERT_TRUE(status.ok()) << status.error_message();
  }

  ASSERT_TRUE(service_->WaitForReport(std::chrono::milliseconds(2000)));
  std::this_thread::sleep_for(std::chrono::milliseconds(2 * kFlushIntervalMs));
  auto samples = CollectSamples(service_->Requests());

  uint64_t route_get_ok_count = 0;
  bool saw_report_metrics_self_metric = false;
  for (const auto& s : samples) {
    if (s.value_case() != ::umbp::MetricSample::kHistogramAggregate) continue;
    if (s.name() != MORI_UMBP_METRIC_MASTER_CLIENT_RPC_LATENCY) continue;
    const auto rpc = LabelValue(s, "rpc");
    const auto status = LabelValue(s, "status");
    if (rpc == "RouteGet" && status == "ok") {
      route_get_ok_count += s.histogram_aggregate().count();
    }
    if (rpc == "ReportMetrics") saw_report_metrics_self_metric = true;
  }
  EXPECT_GE(route_get_ok_count, static_cast<uint64_t>(N))
      << "Expected aggregated count >= " << N << " for RouteGet latency";
  EXPECT_FALSE(saw_report_metrics_self_metric)
      << "ReportMetrics RPC must not be self-instrumented (would create a feedback loop)";
}

// On a non-OK status the timer must emit both a status=error histogram
// observation and a counter row tagged with the gRPC code name.  Failures
// are injected deterministically by toggling RecordingMasterService into
// fail_route_get mode (RegisterClient and ReportMetrics still succeed, so the
// flush thread can deliver the error sample to the test sink).
TEST_F(MasterClientRpcLatencyTest, ErrorPathRecordsErrorCounter) {
  StartClientAndRegister();
  service_->Clear();
  service_->SetFailRouteGet(true);

  constexpr int N = 5;
  for (int i = 0; i < N; ++i) {
    std::optional<RouteGetResult> out;
    auto status = client_->RouteGet("err-test-" + std::to_string(i), {}, &out);
    EXPECT_FALSE(status.ok()) << "Injected fail_route_get must surface as non-OK at " << i;
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
  }

  ASSERT_TRUE(service_->WaitForReport(std::chrono::milliseconds(3000)));
  std::this_thread::sleep_for(std::chrono::milliseconds(2 * kFlushIntervalMs));
  auto samples = CollectSamples(service_->Requests());

  int error_counter_total = 0;
  bool saw_route_get_unavailable = false;
  uint64_t route_get_error_count = 0;
  for (const auto& s : samples) {
    if (s.name() == MORI_UMBP_METRIC_MASTER_CLIENT_RPC_ERRORS_TOTAL &&
        s.value_case() == ::umbp::MetricSample::kCounterDelta) {
      error_counter_total += static_cast<int>(s.counter_delta());
      if (LabelValue(s, "rpc") == "RouteGet" && LabelValue(s, "code") == "UNAVAILABLE") {
        saw_route_get_unavailable = true;
      }
    }
    if (s.name() == MORI_UMBP_METRIC_MASTER_CLIENT_RPC_LATENCY &&
        s.value_case() == ::umbp::MetricSample::kHistogramAggregate &&
        LabelValue(s, "rpc") == "RouteGet" && LabelValue(s, "status") == "error") {
      route_get_error_count += s.histogram_aggregate().count();
    }
  }
  EXPECT_GE(error_counter_total, N) << "Each injected failure must bump the error counter";
  EXPECT_TRUE(saw_route_get_unavailable)
      << "Expected RouteGet error counter sample with code=UNAVAILABLE";
  EXPECT_GE(route_get_error_count, static_cast<uint64_t>(N))
      << "Expected aggregated count >= " << N << " for status=error latency";
}

// A MasterClient that never calls RegisterSelf must not buffer latency
// samples — otherwise pending_histogram_aggregates_ would grow forever on
// Python's read-only UMBPMasterClient path.
TEST_F(MasterClientRpcLatencyTest, UnregisteredClientDoesNotBufferSamples) {
  // Build a client but never RegisterSelf.
  UMBPMasterClientConfig cfg;
  cfg.node_id = "unregistered-test-node";
  cfg.node_address = "127.0.0.1";
  cfg.master_address = address_;
  auto unreg_client = std::make_unique<MasterClient>(cfg);

  // Fire an RPC that does not require registration (MatchExternalKv).
  std::vector<MasterClient::ExternalKvNodeMatch> matches;
  for (int i = 0; i < 50; ++i) {
    (void)unreg_client->MatchExternalKv({"hash-" + std::to_string(i)}, &matches);
  }

  // Wait long enough for any flush thread to have run (it shouldn't have
  // started since registered_=false), then verify nothing was reported.
  service_->Clear();
  std::this_thread::sleep_for(std::chrono::milliseconds(3 * kFlushIntervalMs));
  auto reqs = service_->Requests();
  EXPECT_TRUE(reqs.empty()) << "Unregistered MasterClient must not flush metrics; got "
                            << reqs.size() << " request(s)";

  unreg_client.reset();
}

// N=200 successful RouteGets produce a single histogram_aggregate sample per
// (rpc, status) series with count() == N, sum() > 0, and cumulative-monotone
// bucket_counts.  Verifies the wire encoding and the master-side merge
// invariants (cumulative + bucket_counts.back() <= count for the +Inf
// overflow).
TEST_F(MasterClientRpcLatencyTest, AggregatedFlushCountSumAndCumulativeBuckets) {
  StartClientAndRegister();
  service_->Clear();

  constexpr int N = 200;
  for (int i = 0; i < N; ++i) {
    std::optional<RouteGetResult> out;
    auto status = client_->RouteGet("agg-key-" + std::to_string(i), {}, &out);
    ASSERT_TRUE(status.ok()) << status.error_message();
  }

  ASSERT_TRUE(service_->WaitForReport(std::chrono::milliseconds(2000)));
  std::this_thread::sleep_for(std::chrono::milliseconds(3 * kFlushIntervalMs));
  auto samples = CollectSamples(service_->Requests());

  // Sum aggregated counts across flush cycles for the (rpc=RouteGet,
  // status=ok) series; we may have multiple flushes if the test ran across
  // cycle boundaries.
  uint64_t total_count = 0;
  double total_sum = 0;
  bool saw_any_aggregate = false;
  std::vector<uint64_t> last_bucket_counts;
  for (const auto& s : samples) {
    if (s.value_case() != ::umbp::MetricSample::kHistogramAggregate) continue;
    if (s.name() != MORI_UMBP_METRIC_MASTER_CLIENT_RPC_LATENCY) continue;
    if (LabelValue(s, "rpc") != "RouteGet" || LabelValue(s, "status") != "ok") continue;
    saw_any_aggregate = true;
    const auto& a = s.histogram_aggregate();
    total_count += a.count();
    total_sum += a.sum();
    last_bucket_counts.assign(a.bucket_counts().begin(), a.bucket_counts().end());
    // Cumulative monotonicity per-sample: bucket_counts[i] <= bucket_counts[i+1].
    for (int i = 0; i + 1 < a.bucket_counts_size(); ++i) {
      EXPECT_LE(a.bucket_counts(i), a.bucket_counts(i + 1))
          << "bucket_counts must be cumulative (monotone non-decreasing)";
    }
    // bounds.size() must equal bucket_counts.size().
    EXPECT_EQ(a.bounds_size(), a.bucket_counts_size());
    // bucket_counts.back() <= count (difference is the +Inf overflow).
    if (a.bucket_counts_size() > 0) {
      EXPECT_LE(a.bucket_counts(a.bucket_counts_size() - 1), a.count());
    }
  }
  EXPECT_TRUE(saw_any_aggregate) << "No histogram_aggregate sample for (RouteGet, ok) found";
  EXPECT_GE(total_count, static_cast<uint64_t>(N));
  EXPECT_GT(total_sum, 0.0);
}

// A value above every finite bound falls into the implicit +Inf bucket:
// bucket_counts must stay all-zero while count and sum still increment.
// Exercises the corner the cumulative encoding has to handle correctly.
TEST_F(MasterClientRpcLatencyTest, AggregatedFlushPlusInfOverflow) {
  StartClientAndRegister();
  service_->Clear();

  client_->Observe("test_inf_overflow", "synthetic", {}, {0.001, 0.01}, 0.5);

  ASSERT_TRUE(service_->WaitForReport(std::chrono::milliseconds(2000)));
  std::this_thread::sleep_for(std::chrono::milliseconds(2 * kFlushIntervalMs));
  auto samples = CollectSamples(service_->Requests());

  bool found = false;
  for (const auto& s : samples) {
    if (s.value_case() != ::umbp::MetricSample::kHistogramAggregate) continue;
    if (s.name() != "test_inf_overflow") continue;
    found = true;
    const auto& a = s.histogram_aggregate();
    ASSERT_EQ(a.bounds_size(), 2);
    EXPECT_DOUBLE_EQ(a.bounds(0), 0.001);
    EXPECT_DOUBLE_EQ(a.bounds(1), 0.01);
    ASSERT_EQ(a.bucket_counts_size(), 2);
    EXPECT_EQ(a.bucket_counts(0), 0u);
    EXPECT_EQ(a.bucket_counts(1), 0u);
    EXPECT_EQ(a.count(), 1u);
    EXPECT_DOUBLE_EQ(a.sum(), 0.5);
    break;
  }
  EXPECT_TRUE(found) << "test_inf_overflow histogram_aggregate not seen";
}

// Directly exercises the series-cardinality cap branch: shrink the cap via
// the friend hook, fire cap+1 distinct-label Observes, and assert exactly
// `cap` series make it onto the wire plus a metrics_dropped_total bump.
// This is the cold path that fires only on label-cardinality leaks in
// production, so we must verify the erase + ++dropped logic directly rather
// than relying on it never firing.
//
// Determinism note: the heartbeat thread fires its own Observe per tick and
// would consume cap slots, leaving an unpredictable number of test slots.
// We stop both the heartbeat AND the metrics-flush thread before the
// in-memory assertion block so neither can mutate pending_*/_dropped state
// during inspection; the metrics thread is restarted afterwards so the
// wire-level checks below can rely on the regular flush path.
TEST_F(MasterClientRpcLatencyTest, CapEnforcedAndDropCounterIncrements) {
  StartClientAndRegister();
  client_->StopHeartbeat();
  // One full flush cycle for any inflight heartbeat-Observe to leave the
  // pending map; then drop the captured wire history.
  std::this_thread::sleep_for(std::chrono::milliseconds(3 * kFlushIntervalMs));
  service_->Clear();

  // Pause the flush loop so PendingSeriesCount()/DroppedCount() are stable
  // for the in-memory assertions.  Observe() does not gate on
  // metrics_running_, so the cap path still executes; the buffered state
  // sits in pending_* until the loop is restarted below.
  StopMetricsThreadForTesting();
  ASSERT_EQ(PendingSeriesCount(), 0u) << "Pending must be empty before cap test";
  SetSeriesCap(4);
  for (int i = 0; i < 5; ++i) {
    MasterClient::Labels labels = {{"k", std::to_string(i)}};
    client_->Observe("test_cap_metric", "synthetic", labels, {1.0}, 0.5);
  }
  // Now safe — flush thread is parked.
  EXPECT_EQ(PendingSeriesCount(), 4u)
      << "After 5 Observes with cap=4, pending must hold exactly 4 series";
  EXPECT_GE(DroppedCount(), 1u) << "5th Observe must have bumped metrics_dropped_count_";

  // Resume the flush loop; the 4 buffered series and the dropped counter
  // delta will be drained in the next ReportMetrics RPC.
  StartMetricsThreadForTesting();

  ASSERT_TRUE(service_->WaitForReport(std::chrono::milliseconds(2000)));
  std::this_thread::sleep_for(std::chrono::milliseconds(3 * kFlushIntervalMs));
  auto samples = CollectSamples(service_->Requests());

  int test_cap_metric_series = 0;
  uint64_t dropped_total_delta = 0;
  for (const auto& s : samples) {
    if (s.name() == "test_cap_metric" &&
        s.value_case() == ::umbp::MetricSample::kHistogramAggregate) {
      ++test_cap_metric_series;
    }
    if (s.name() == MORI_UMBP_METRIC_MASTER_CLIENT_METRICS_DROPPED_TOTAL &&
        s.value_case() == ::umbp::MetricSample::kCounterDelta) {
      dropped_total_delta += static_cast<uint64_t>(s.counter_delta());
    }
  }
  EXPECT_EQ(test_cap_metric_series, 4)
      << "Cap=4 must let exactly 4 distinct series through, got " << test_cap_metric_series;
  EXPECT_GE(dropped_total_delta, 1u)
      << "5th distinct series must bump metrics_dropped_total counter";
}

}  // namespace mori::umbp

int main(int argc, char** argv) {
  // Match the metrics flush cadence to the test's heartbeat interval so flush
  // cycles are deterministic in test-time.  MetricsReportIntervalMs() caches
  // the value statically on first read, so this must be set before any
  // MasterClient is constructed.
  ::setenv("UMBP_METRICS_REPORT_INTERVAL_MS", "100", 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
