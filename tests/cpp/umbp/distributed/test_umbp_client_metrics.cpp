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

// Two test suites:
//
// 1. MasterClientMetricsTest — unit tests for MasterClient::AddCounter,
//    SetGauge, and Observe buffering.  Uses a fake recording server that
//    returns a 100ms heartbeat interval, allowing the flush to be verified
//    within a short wall-clock window.
//
// 2. PoolClientLocalByteTrackingTest — integration test verifying that a
//    single-node PoolClient reports correct Put/Get byte counts (traffic=local)
//    through the full pipeline: PoolClient → MasterClient buffer →
//    ReportMetrics RPC → MasterServer → Prometheus HTTP endpoint.
//
// main() sets UMBP_HEARTBEAT_TTL_SEC=1 before any test runs so that the real
// MasterServer in suite 2 returns ≈500ms heartbeat/metrics flush intervals.

#include <arpa/inet.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "umbp.grpc.pb.h"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"

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

// ============================================================
//  Fake gRPC server that captures ReportMetrics calls.
//  RegisterClient returns a configurable heartbeat interval so
//  the metrics flush cadence is under test control.
// ============================================================
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
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<::umbp::ReportMetricsRequest> requests_;
};

// Flatten all MetricSamples across all captured requests.
static std::vector<::umbp::MetricSample> CollectSamples(
    const std::vector<::umbp::ReportMetricsRequest>& reqs) {
  std::vector<::umbp::MetricSample> out;
  for (const auto& r : reqs)
    for (const auto& s : r.metrics()) out.push_back(s);
  return out;
}

// Sum counter_delta across samples matching name and optional traffic label.
static double SumCounterDelta(const std::vector<::umbp::MetricSample>& samples,
                              const std::string& name, const std::string& traffic = "") {
  double total = 0.0;
  for (const auto& s : samples) {
    if (s.name() != name) continue;
    if (s.value_case() != ::umbp::MetricSample::kCounterDelta) continue;
    if (!traffic.empty()) {
      bool match = false;
      for (const auto& l : s.labels())
        if (l.name() == "traffic" && l.value() == traffic) {
          match = true;
          break;
        }
      if (!match) continue;
    }
    total += s.counter_delta();
  }
  return total;
}

// ============================================================
//  Suite 1: MasterClient metrics API buffering (unit tests)
// ============================================================
class MasterClientMetricsTest : public ::testing::Test {
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

    UMBPMasterClientConfig cfg;
    cfg.node_id = "metrics-test-node";
    cfg.node_address = "127.0.0.1";
    cfg.master_address = address_;
    cfg.auto_heartbeat = true;
    client_ = std::make_unique<MasterClient>(cfg);

    std::map<TierType, TierCapacity> caps;
    caps[TierType::DRAM] = {1 << 20, 1 << 20};
    auto status = client_->RegisterSelf(caps);
    ASSERT_TRUE(status.ok()) << status.error_message();
  }

  void TearDown() override {
    client_.reset();
    if (server_) {
      server_->Shutdown(std::chrono::system_clock::now() + std::chrono::milliseconds(500));
      server_->Wait();
    }
  }

  uint16_t port_ = 0;
  std::string address_;
  std::unique_ptr<RecordingMasterService> service_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<MasterClient> client_;
};

// Three AddCounter calls with the same name+labels are accumulated and sent
// as a single delta equal to their sum in the next flush.
TEST_F(MasterClientMetricsTest, AddCounterAccumulatesDeltas) {
  service_->Clear();
  client_->AddCounter("test_counter", "help", {{"traffic", "local"}}, 100.0);
  client_->AddCounter("test_counter", "help", {{"traffic", "local"}}, 200.0);
  client_->AddCounter("test_counter", "help", {{"traffic", "local"}}, 50.0);

  ASSERT_TRUE(service_->WaitForReport(std::chrono::milliseconds(2000)));
  auto samples = CollectSamples(service_->Requests());
  EXPECT_DOUBLE_EQ(SumCounterDelta(samples, "test_counter", "local"), 350.0);
}

// AddCounter calls with different label values produce separate entries.
TEST_F(MasterClientMetricsTest, AddCounterDistinctLabelsAreSeparate) {
  service_->Clear();
  client_->AddCounter("test_bytes", "help", {{"traffic", "local"}}, 1024.0);
  client_->AddCounter("test_bytes", "help", {{"traffic", "remote"}}, 2048.0);

  ASSERT_TRUE(service_->WaitForReport(std::chrono::milliseconds(2000)));
  auto samples = CollectSamples(service_->Requests());
  EXPECT_DOUBLE_EQ(SumCounterDelta(samples, "test_bytes", "local"), 1024.0);
  EXPECT_DOUBLE_EQ(SumCounterDelta(samples, "test_bytes", "remote"), 2048.0);
}

// Multiple SetGauge calls for the same series collapse: only the last value
// is present in the flushed request.
TEST_F(MasterClientMetricsTest, SetGaugeLastWriteWins) {
  service_->Clear();
  client_->SetGauge("test_gauge", "help", {}, 10.0);
  client_->SetGauge("test_gauge", "help", {}, 20.0);
  client_->SetGauge("test_gauge", "help", {}, 30.0);

  ASSERT_TRUE(service_->WaitForReport(std::chrono::milliseconds(2000)));
  auto samples = CollectSamples(service_->Requests());

  int count = 0;
  double last_val = 0.0;
  for (const auto& s : samples) {
    if (s.name() == "test_gauge" && s.value_case() == ::umbp::MetricSample::kGaugeValue) {
      last_val = s.gauge_value();
      ++count;
    }
  }
  EXPECT_EQ(count, 1) << "Three SetGauge calls must collapse to one sample";
  EXPECT_DOUBLE_EQ(last_val, 30.0);
}

// Observe flushes the histogram with correct bounds and value.
TEST_F(MasterClientMetricsTest, ObserveHistogramFlushed) {
  service_->Clear();
  client_->Observe("test_hist", "help", {}, {1.0, 5.0, 10.0}, 3.5);

  ASSERT_TRUE(service_->WaitForReport(std::chrono::milliseconds(2000)));
  auto samples = CollectSamples(service_->Requests());

  bool found = false;
  for (const auto& s : samples) {
    if (s.name() == "test_hist" && s.value_case() == ::umbp::MetricSample::kHistogram) {
      EXPECT_DOUBLE_EQ(s.histogram().value(), 3.5);
      ASSERT_EQ(s.histogram().bounds_size(), 3);
      EXPECT_DOUBLE_EQ(s.histogram().bounds(0), 1.0);
      EXPECT_DOUBLE_EQ(s.histogram().bounds(2), 10.0);
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Histogram sample not found in flushed request";
}

// Every ReportMetrics request must carry the client's own node_id so the
// master can inject it as the Prometheus "node" label.
TEST_F(MasterClientMetricsTest, NodeIdSetInRequest) {
  service_->Clear();
  client_->AddCounter("probe_counter", "help", {}, 1.0);

  ASSERT_TRUE(service_->WaitForReport(std::chrono::milliseconds(2000)));
  for (const auto& req : service_->Requests()) {
    EXPECT_EQ(req.node_id(), "metrics-test-node");
  }
}

// ============================================================
//  Suite 2: PoolClient local-path byte tracking (integration)
//  Uses a real MasterServer with Prometheus metrics port.
//  UMBP_HEARTBEAT_TTL_SEC=1 (set in main) → ≈500ms flush interval.
// ============================================================

static std::string FetchPrometheusMetrics(int port) {
  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) return "";
  struct timeval tv{3, 0};
  ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  if (::connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(sock);
    return "";
  }
  const char* req = "GET /metrics HTTP/1.0\r\nHost: localhost\r\n\r\n";
  ::send(sock, req, strlen(req), 0);
  std::string resp;
  char buf[8192];
  ssize_t n;
  while ((n = ::recv(sock, buf, sizeof(buf), 0)) > 0) resp.append(buf, n);
  ::close(sock);
  return resp;
}

// Scan Prometheus text for a line matching name and label_substr.
// Returns the trailing numeric value, or -1.0 if not found.
static double ParseMetricValue(const std::string& body, const std::string& name,
                               const std::string& label_substr) {
  size_t pos = 0;
  while (pos < body.size()) {
    size_t nl = body.find('\n', pos);
    std::string line = body.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
    pos = (nl == std::string::npos) ? body.size() : nl + 1;
    if (line.empty() || line.front() == '#') continue;
    if (line.find(name) == std::string::npos) continue;
    if (!label_substr.empty() && line.find(label_substr) == std::string::npos) continue;
    size_t sp = line.rfind(' ');
    if (sp == std::string::npos) continue;
    try {
      return std::stod(line.substr(sp + 1));
    } catch (...) {
    }
  }
  return -1.0;
}

constexpr size_t kLocalPageSize = 4096;
constexpr size_t kLocalBufSize = 8 << 20;

class PoolClientLocalByteTrackingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    uint16_t base = AllocPort();
    master_port_ = base;
    metrics_port_ = base + 1;
    io_port_ = base + 2;

    buf_ = std::malloc(kLocalBufSize);
    src_ = std::malloc(kLocalPageSize);
    dst_ = std::malloc(kLocalPageSize);
    ASSERT_NE(buf_, nullptr);
    ASSERT_NE(src_, nullptr);
    ASSERT_NE(dst_, nullptr);
    std::memset(buf_, 0, kLocalBufSize);
    std::memset(src_, 0xAB, kLocalPageSize);
    std::memset(dst_, 0, kLocalPageSize);

    MasterServerConfig master_cfg;
    // Short TTL so the recommended heartbeat/metrics interval is ≈100ms.
    master_cfg.registry_config.heartbeat_ttl = std::chrono::seconds(1);
    master_cfg.listen_address = "0.0.0.0:" + std::to_string(master_port_);
    master_cfg.metrics_port = metrics_port_;
    master_ = std::make_unique<MasterServer>(std::move(master_cfg));
    server_thread_ = std::thread([this] { master_->Run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    PoolClientConfig cfg;
    cfg.master_config.node_id = "node-local";
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = "localhost:" + std::to_string(master_port_);
    cfg.io_engine.host = "0.0.0.0";
    cfg.io_engine.port = io_port_;
    cfg.dram_page_size = kLocalPageSize;
    cfg.dram_buffers = {{buf_, kLocalBufSize}};
    cfg.tier_capacities = {{TierType::DRAM, {kLocalBufSize, kLocalBufSize}}};
    client_ = std::make_unique<PoolClient>(std::move(cfg));
    ASSERT_TRUE(client_->Init());
  }

  void TearDown() override {
    if (client_) client_->Shutdown();
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
    std::free(buf_);
    std::free(src_);
    std::free(dst_);
  }

  uint16_t master_port_ = 0;
  uint16_t metrics_port_ = 0;
  uint16_t io_port_ = 0;
  void* buf_ = nullptr;
  void* src_ = nullptr;
  void* dst_ = nullptr;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> client_;
};

// Single-node Put + Get: both must appear as local traffic in Prometheus.
// With UMBP_HEARTBEAT_TTL_SEC=1, the flush interval is ≈500ms; sleeping 1.5s
// ensures at least one complete flush cycle.
TEST_F(PoolClientLocalByteTrackingTest, LocalPutGetBytesCounted) {
  const std::string key = "metric-tracking-key";

  ASSERT_TRUE(client_->Put(key, src_, kLocalPageSize));
  ASSERT_TRUE(client_->Get(key, dst_, kLocalPageSize));

  std::this_thread::sleep_for(std::chrono::milliseconds(1500));

  std::string body = FetchPrometheusMetrics(metrics_port_);
  ASSERT_FALSE(body.empty()) << "Could not fetch Prometheus metrics from port " << metrics_port_;

  double put_local = ParseMetricValue(body, "mori_umbp_client_put_bytes_total", "local");
  double get_local = ParseMetricValue(body, "mori_umbp_client_get_bytes_total", "local");

  EXPECT_GE(put_local, static_cast<double>(kLocalPageSize))
      << "Expected >= " << kLocalPageSize << " local put bytes; Prometheus shows " << put_local;
  EXPECT_GE(get_local, static_cast<double>(kLocalPageSize))
      << "Expected >= " << kLocalPageSize << " local get bytes; Prometheus shows " << get_local;

  // Single-node setup must not produce any remote traffic counters.
  EXPECT_EQ(ParseMetricValue(body, "mori_umbp_client_put_bytes_total", "remote"), -1.0)
      << "Unexpected remote put bytes in single-node setup";
  EXPECT_EQ(ParseMetricValue(body, "mori_umbp_client_get_bytes_total", "remote"), -1.0)
      << "Unexpected remote get bytes in single-node setup";
}

}  // namespace
}  // namespace mori::umbp

int main(int argc, char** argv) {
  // Must be set before any GetEnvSeconds("UMBP_HEARTBEAT_TTL_SEC") static
  // is initialized.  TTL=1s → recommended interval ≈ 500ms, which the
  // metrics flush thread reuses.
  ::setenv("UMBP_HEARTBEAT_TTL_SEC", "1", 1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
