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

  // Lookup is the workhorse RPC we exercise for instrumentation tests.
  // When fail_lookup_=true it returns UNAVAILABLE deterministically so the
  // ErrorPath test does not depend on TCP/channel reconnection timing.
  grpc::Status Lookup(grpc::ServerContext*, const ::umbp::LookupRequest*,
                      ::umbp::LookupResponse* resp) override {
    if (fail_lookup_.load()) {
      return grpc::Status(grpc::StatusCode::UNAVAILABLE, "injected by test");
    }
    resp->set_found(false);
    return grpc::Status::OK;
  }

  void SetFailLookup(bool fail) { fail_lookup_.store(fail); }

  // Used to test that read-only Python-style clients (no RegisterSelf) don't
  // leak entries into pending_histograms_.
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
  std::atomic<bool> fail_lookup_{false};
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
    cfg.auto_heartbeat = true;
    client_ = std::make_unique<MasterClient>(cfg);
    std::map<TierType, TierCapacity> caps;
    caps[TierType::DRAM] = {1 << 20, 1 << 20};
    auto status = client_->RegisterSelf(caps);
    ASSERT_TRUE(status.ok()) << status.error_message();
  }

  uint16_t port_ = 0;
  std::string address_;
  std::unique_ptr<RecordingMasterService> service_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<MasterClient> client_;
};

// N successful Lookups must produce >=N latency histogram observations
// labelled rpc=Lookup,status=ok, and ReportMetrics itself must never appear
// (self-feedback would re-bias the metric).
TEST_F(MasterClientRpcLatencyTest, LookupHistogramObservedAndReportMetricsExcluded) {
  StartClientAndRegister();
  service_->Clear();

  constexpr int N = 5;
  for (int i = 0; i < N; ++i) {
    bool found = false;
    auto status = client_->Lookup("test-key-" + std::to_string(i), &found);
    ASSERT_TRUE(status.ok()) << status.error_message();
  }

  ASSERT_TRUE(service_->WaitForReport(std::chrono::milliseconds(2000)));
  std::this_thread::sleep_for(std::chrono::milliseconds(2 * kFlushIntervalMs));
  auto samples = CollectSamples(service_->Requests());

  int lookup_hist_count = 0;
  bool saw_report_metrics_self_metric = false;
  for (const auto& s : samples) {
    if (s.value_case() != ::umbp::MetricSample::kHistogram) continue;
    if (s.name() != MORI_UMBP_METRIC_MASTER_CLIENT_RPC_LATENCY) continue;
    const auto rpc = LabelValue(s, "rpc");
    const auto status = LabelValue(s, "status");
    if (rpc == "Lookup" && status == "ok") {
      ++lookup_hist_count;
    }
    if (rpc == "ReportMetrics") saw_report_metrics_self_metric = true;
  }
  EXPECT_GE(lookup_hist_count, N) << "Expected >=" << N << " Lookup latency samples";
  EXPECT_FALSE(saw_report_metrics_self_metric)
      << "ReportMetrics RPC must not be self-instrumented (would create a feedback loop)";
}

// On a non-OK status the timer must emit both a status=error histogram
// observation and a counter row tagged with the gRPC code name.  Failures
// are injected deterministically by toggling RecordingMasterService into
// fail_lookup mode (RegisterClient and ReportMetrics still succeed, so the
// flush thread can deliver the error sample to the test sink).
TEST_F(MasterClientRpcLatencyTest, ErrorPathRecordsErrorCounter) {
  StartClientAndRegister();
  service_->Clear();
  service_->SetFailLookup(true);

  bool found = false;
  constexpr int N = 5;
  for (int i = 0; i < N; ++i) {
    auto status = client_->Lookup("err-test-" + std::to_string(i), &found);
    EXPECT_FALSE(status.ok()) << "Injected fail_lookup must surface as non-OK at " << i;
    EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
  }

  ASSERT_TRUE(service_->WaitForReport(std::chrono::milliseconds(3000)));
  std::this_thread::sleep_for(std::chrono::milliseconds(2 * kFlushIntervalMs));
  auto samples = CollectSamples(service_->Requests());

  int error_counter_total = 0;
  bool saw_lookup_unavailable = false;
  int lookup_error_hist = 0;
  for (const auto& s : samples) {
    if (s.name() == MORI_UMBP_METRIC_MASTER_CLIENT_RPC_ERRORS_TOTAL &&
        s.value_case() == ::umbp::MetricSample::kCounterDelta) {
      error_counter_total += static_cast<int>(s.counter_delta());
      if (LabelValue(s, "rpc") == "Lookup" && LabelValue(s, "code") == "UNAVAILABLE") {
        saw_lookup_unavailable = true;
      }
    }
    if (s.name() == MORI_UMBP_METRIC_MASTER_CLIENT_RPC_LATENCY &&
        s.value_case() == ::umbp::MetricSample::kHistogram && LabelValue(s, "rpc") == "Lookup" &&
        LabelValue(s, "status") == "error") {
      ++lookup_error_hist;
    }
  }
  EXPECT_GE(error_counter_total, N) << "Each injected failure must bump the error counter";
  EXPECT_TRUE(saw_lookup_unavailable)
      << "Expected Lookup error counter sample with code=UNAVAILABLE";
  EXPECT_GE(lookup_error_hist, N) << "Expected >=" << N << " latency histograms with status=error";
}

// A MasterClient that never calls RegisterSelf must not buffer latency
// samples — otherwise pending_histograms_ would grow forever on Python's
// read-only UMBPMasterClient path.
TEST_F(MasterClientRpcLatencyTest, UnregisteredClientDoesNotBufferSamples) {
  // Build a client but never RegisterSelf.
  UMBPMasterClientConfig cfg;
  cfg.node_id = "unregistered-test-node";
  cfg.node_address = "127.0.0.1";
  cfg.master_address = address_;
  cfg.auto_heartbeat = false;
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

}  // namespace
}  // namespace mori::umbp

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
