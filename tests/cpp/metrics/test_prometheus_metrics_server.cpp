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
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mori/metrics/prometheus_metrics_server.hpp"

namespace mori::metrics {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Perform a single HTTP GET on the given path and return the full response.
static std::string httpGet(int port, const std::string& path) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  EXPECT_GE(fd, 0);

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

  EXPECT_EQ(::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);

  std::string request = "GET " + path + " HTTP/1.0\r\nHost: localhost\r\n\r\n";
  ::write(fd, request.data(), request.size());

  std::string response;
  char buf[4096];
  ssize_t n;
  while ((n = ::read(fd, buf, sizeof(buf))) > 0) response.append(buf, static_cast<std::size_t>(n));

  ::close(fd);
  return response;
}

// Extract the HTTP body (everything after the blank line).
static std::string httpBody(const std::string& response) {
  auto pos = response.find("\r\n\r\n");
  if (pos == std::string::npos) return response;
  return response.substr(pos + 4);
}

// Extract the HTTP status line (first line).
static std::string httpStatus(const std::string& response) {
  auto pos = response.find("\r\n");
  return response.substr(0, pos);
}

// Pick an ephemeral port by binding to :0 and immediately closing.
static int freePort() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  int opt = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = 0;
  ::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));

  socklen_t len = sizeof(addr);
  ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
  int port = ntohs(addr.sin_port);
  ::close(fd);
  return port;
}

// ---------------------------------------------------------------------------
// Fixture: one server instance per test, on a fresh ephemeral port.
// ---------------------------------------------------------------------------
class PrometheusMetricsServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    port_ = freePort();
    server_ = std::make_unique<MetricsServer>(port_);
    // Give the accept loop a moment to start.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  void TearDown() override { server_.reset(); }

  std::string scrape() { return httpBody(httpGet(port_, "/metrics")); }
  std::string scrapeRaw() { return httpGet(port_, "/metrics"); }
  std::string scrapeRaw(const std::string& path) { return httpGet(port_, path); }

  int port_{};
  std::unique_ptr<MetricsServer> server_;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST_F(PrometheusMetricsServerTest, StartsAndStopsCleanly) {
  EXPECT_TRUE(server_->running());
  EXPECT_EQ(server_->port(), port_);
}

TEST_F(PrometheusMetricsServerTest, EmptyResponseIsValid) {
  // An empty registry should still return 200 OK with an empty body.
  auto raw = scrapeRaw();
  EXPECT_NE(raw.find("200 OK"), std::string::npos);
  EXPECT_TRUE(httpBody(raw).empty());
}

// ---------------------------------------------------------------------------
// GET routing
// ---------------------------------------------------------------------------

TEST_F(PrometheusMetricsServerTest, UnknownPathReturns404) {
  auto raw = scrapeRaw("/unknown");
  EXPECT_NE(httpStatus(raw).find("404"), std::string::npos);
}

TEST_F(PrometheusMetricsServerTest, RootPathReturns404) {
  auto raw = scrapeRaw("/");
  EXPECT_NE(httpStatus(raw).find("404"), std::string::npos);
}

TEST_F(PrometheusMetricsServerTest, MetricsPathReturns200) {
  auto raw = scrapeRaw("/metrics");
  EXPECT_NE(httpStatus(raw).find("200"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Gauge
// ---------------------------------------------------------------------------

TEST_F(PrometheusMetricsServerTest, GaugeAppearsInOutput) {
  server_->setGauge("test_gauge", "A test gauge", 42.0);
  auto body = scrape();
  EXPECT_NE(body.find("# TYPE test_gauge gauge"), std::string::npos);
  EXPECT_NE(body.find("test_gauge 42"), std::string::npos);
}

TEST_F(PrometheusMetricsServerTest, GaugeHelpAppearsInOutput) {
  server_->setGauge("test_gauge_help", "My help text", 1.0);
  auto body = scrape();
  EXPECT_NE(body.find("# HELP test_gauge_help My help text"), std::string::npos);
}

TEST_F(PrometheusMetricsServerTest, GaugeOverwritesPreviousValue) {
  server_->setGauge("overwrite_gauge", "Gauge", 1.0);
  server_->setGauge("overwrite_gauge", "Gauge", 99.5);
  auto body = scrape();
  EXPECT_NE(body.find("overwrite_gauge 99.5"), std::string::npos);
  // Old value must not appear.
  EXPECT_EQ(body.find("overwrite_gauge 1"), std::string::npos);
}

TEST_F(PrometheusMetricsServerTest, MultipleGauges) {
  server_->setGauge("gauge_a", "Gauge A", 1.0);
  server_->setGauge("gauge_b", "Gauge B", 2.0);
  auto body = scrape();
  EXPECT_NE(body.find("gauge_a"), std::string::npos);
  EXPECT_NE(body.find("gauge_b"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Counter
// ---------------------------------------------------------------------------

TEST_F(PrometheusMetricsServerTest, CounterAppearsInOutput) {
  server_->addCounter("test_counter", "A test counter");
  auto body = scrape();
  EXPECT_NE(body.find("# TYPE test_counter counter"), std::string::npos);
  EXPECT_NE(body.find("test_counter 1"), std::string::npos);
}

TEST_F(PrometheusMetricsServerTest, CounterDefaultDeltaIsOne) {
  server_->addCounter("delta_counter", "Delta counter");
  server_->addCounter("delta_counter", "Delta counter");
  auto body = scrape();
  EXPECT_NE(body.find("delta_counter 2"), std::string::npos);
}

TEST_F(PrometheusMetricsServerTest, CounterCustomDelta) {
  server_->addCounter("big_counter", "Big counter", 100);
  auto body = scrape();
  EXPECT_NE(body.find("big_counter 100"), std::string::npos);
}

TEST_F(PrometheusMetricsServerTest, CounterAccumulatesAcrossCalls) {
  server_->addCounter("accum_counter", "Accum", 3);
  server_->addCounter("accum_counter", "Accum", 7);
  auto body = scrape();
  EXPECT_NE(body.find("accum_counter 10"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------

TEST_F(PrometheusMetricsServerTest, HistogramAppearsInOutput) {
  server_->observe("test_hist", "A histogram", {0.1, 1.0, 10.0}, 0.5);
  auto body = scrape();
  EXPECT_NE(body.find("# TYPE test_hist histogram"), std::string::npos);
  EXPECT_NE(body.find("test_hist_bucket"), std::string::npos);
  EXPECT_NE(body.find("test_hist_sum"), std::string::npos);
  EXPECT_NE(body.find("test_hist_count"), std::string::npos);
}

TEST_F(PrometheusMetricsServerTest, HistogramInfBucketEqualsCount) {
  server_->observe("inf_hist", "Inf bucket", {0.1, 1.0}, 0.5);
  server_->observe("inf_hist", "Inf bucket", {0.1, 1.0}, 0.05);
  auto body = scrape();
  EXPECT_NE(body.find("inf_hist_bucket{le=\"+Inf\"} 2"), std::string::npos);
  EXPECT_NE(body.find("inf_hist_count 2"), std::string::npos);
}

TEST_F(PrometheusMetricsServerTest, HistogramBucketsAreCumulative) {
  // Observation 0.05 falls into bucket le=0.1 but not le=0.01.
  server_->observe("cum_hist", "Cumulative", {0.01, 0.1, 1.0}, 0.05);
  auto body = scrape();
  EXPECT_NE(body.find("cum_hist_bucket{le=\"0.01\"} 0"), std::string::npos);
  EXPECT_NE(body.find("cum_hist_bucket{le=\"0.1\"} 1"), std::string::npos);
  EXPECT_NE(body.find("cum_hist_bucket{le=\"1\"} 1"), std::string::npos);
}

TEST_F(PrometheusMetricsServerTest, HistogramSumAndCount) {
  server_->observe("sum_hist", "Sum/count", {1.0, 10.0}, 2.0);
  server_->observe("sum_hist", "Sum/count", {1.0, 10.0}, 3.0);
  auto body = scrape();
  EXPECT_NE(body.find("sum_hist_sum 5"), std::string::npos);
  EXPECT_NE(body.find("sum_hist_count 2"), std::string::npos);
}

TEST_F(PrometheusMetricsServerTest, HistogramBoundsInitialisedOnFirstCall) {
  // Subsequent calls with different bounds should use the original layout.
  server_->observe("init_hist", "Init", {1.0}, 0.5);
  // Pass different bounds — should be silently ignored.
  server_->observe("init_hist", "Init", {999.0}, 0.5);
  auto body = scrape();
  // The original bound 1.0 must be present, 999 must not.
  EXPECT_NE(body.find("le=\"1\""), std::string::npos);
  EXPECT_EQ(body.find("le=\"999\""), std::string::npos);
}

// ---------------------------------------------------------------------------
// Thread safety (smoke test)
// ---------------------------------------------------------------------------

TEST_F(PrometheusMetricsServerTest, ConcurrentUpdatesDoNotCrash) {
  constexpr int kThreads = 8;
  constexpr int kIters = 500;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < kIters; ++i) {
        server_->setGauge("concurrent_gauge", "Concurrent gauge",
                          static_cast<double>(t * kIters + i));
        server_->addCounter("concurrent_counter", "Concurrent counter");
        server_->observe("concurrent_hist", "Concurrent histogram", {0.1, 1.0, 10.0},
                         static_cast<double>(i % 10) * 0.1);
      }
    });
  }

  for (auto& th : threads) th.join();

  // After all writers finish, a scrape must succeed without crashing.
  auto body = scrape();
  EXPECT_NE(body.find("concurrent_gauge"), std::string::npos);
  EXPECT_NE(body.find("concurrent_counter"), std::string::npos);
  EXPECT_NE(body.find("concurrent_hist"), std::string::npos);

  // Counter must equal kThreads * kIters.
  const uint64_t expected = static_cast<uint64_t>(kThreads) * kIters;
  EXPECT_NE(body.find("concurrent_counter " + std::to_string(expected)), std::string::npos);
}

// ---------------------------------------------------------------------------
// Multiple sequential scrapes
// ---------------------------------------------------------------------------

TEST_F(PrometheusMetricsServerTest, MultipleScrapes) {
  server_->setGauge("stable_gauge", "Stable", 7.0);
  for (int i = 0; i < 5; ++i) {
    auto body = scrape();
    EXPECT_NE(body.find("stable_gauge 7"), std::string::npos);
  }
}

// ---------------------------------------------------------------------------
// Labeled series, escaping, removal
// ---------------------------------------------------------------------------

TEST_F(PrometheusMetricsServerTest, LabelValueIsEscaped) {
  server_->addCounter("esc_total", "help", {{"path", "a\"b\\c\nd"}}, 1);
  auto body = httpBody(httpGet(port_, "/metrics"));
  EXPECT_NE(body.find("esc_total{path=\"a\\\"b\\\\c\\nd\"} 1"), std::string::npos) << body;
}

TEST_F(PrometheusMetricsServerTest, HelpIsEscaped) {
  server_->setGauge("esc_gauge", "line one\nline \\ two", 1.0);
  auto body = httpBody(httpGet(port_, "/metrics"));
  EXPECT_NE(body.find("# HELP esc_gauge line one\\nline \\\\ two\n"), std::string::npos) << body;
}

TEST_F(PrometheusMetricsServerTest, RemoveSeriesDropsOnlyThatSeries) {
  server_->addCounter("rm_total", "help", {{"gpu", "0"}}, 3);
  server_->addCounter("rm_total", "help", {{"gpu", "1"}}, 4);
  EXPECT_EQ(server_->removeSeries("rm_total", {{"gpu", "0"}}), 1);
  auto body = httpBody(httpGet(port_, "/metrics"));
  EXPECT_EQ(body.find("rm_total{gpu=\"0\"}"), std::string::npos) << body;
  EXPECT_NE(body.find("rm_total{gpu=\"1\"} 4"), std::string::npos) << body;
  EXPECT_NE(body.find("# TYPE rm_total counter"), std::string::npos) << body;
}

TEST_F(PrometheusMetricsServerTest, RemoveLastSeriesDropsTheFamily) {
  server_->setGauge("rm_gauge", "help", {{"gpu", "0"}}, 3);
  server_->observeAggregated("rm_hist", "help", {{"gpu", "0"}}, {1.0, 2.0}, {1, 2}, 2, 2.5);
  EXPECT_EQ(server_->removeSeries("rm_gauge", {{"gpu", "0"}}), 1);
  EXPECT_EQ(server_->removeSeries("rm_hist", {{"gpu", "0"}}), 1);
  EXPECT_EQ(server_->removeSeries("rm_hist", {{"gpu", "0"}}), 0);
  auto body = httpBody(httpGet(port_, "/metrics"));
  EXPECT_EQ(body.find("rm_gauge"), std::string::npos) << body;
  EXPECT_EQ(body.find("rm_hist"), std::string::npos) << body;
}

TEST_F(PrometheusMetricsServerTest, RemovedCounterStartsAgainAtZero) {
  server_->addCounter("again_total", "help", {{"gpu", "0"}}, 9);
  server_->removeSeries("again_total", {{"gpu", "0"}});
  server_->addCounter("again_total", "help", {{"gpu", "0"}}, 2);
  auto body = httpBody(httpGet(port_, "/metrics"));
  EXPECT_NE(body.find("again_total{gpu=\"0\"} 2"), std::string::npos) << body;
}

// ---------------------------------------------------------------------------
// Bind address, port 0, misbehaving clients
// ---------------------------------------------------------------------------

TEST(PrometheusMetricsServerBind, PortZeroReportsTheChosenPort) {
  MetricsServer server(0, "127.0.0.1");
  EXPECT_GT(server.port(), 0);
  server.setGauge("bind_gauge", "help", 1.0);
  auto body = httpBody(httpGet(server.port(), "/metrics"));
  EXPECT_NE(body.find("bind_gauge 1"), std::string::npos) << body;
}

TEST(PrometheusMetricsServerBind, BadAddressThrows) {
  EXPECT_THROW(MetricsServer(0, "not-an-address"), std::runtime_error);
}

TEST(PrometheusMetricsServerBind, PortInUseThrows) {
  MetricsServer first(0, "127.0.0.1");
  EXPECT_THROW(MetricsServer(first.port(), "127.0.0.1"), std::runtime_error);
}

TEST(PrometheusMetricsServerClients, SilentClientDoesNotBlockTheNextScrape) {
  MetricsServer server(0, "127.0.0.1");
  server.setGauge("alive", "help", 1.0);

  int silent = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(server.port()));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  ASSERT_EQ(::connect(silent, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);

  auto start = std::chrono::steady_clock::now();
  auto body = httpBody(httpGet(server.port(), "/metrics"));
  auto waited = std::chrono::steady_clock::now() - start;
  EXPECT_NE(body.find("alive 1"), std::string::npos) << body;
  EXPECT_LT(waited, std::chrono::milliseconds(MetricsServer::kClientTimeoutMs + 2000));
  ::close(silent);
}

// Must not SIGPIPE: the body exceeds a socket buffer, so the server is still sending.
TEST(PrometheusMetricsServerClients, ClientHangingUpMidResponseIsHarmless) {
  MetricsServer server(0, "127.0.0.1");
  for (int i = 0; i < 20000; ++i) {
    server.addCounter("big_total", "help", {{"series", std::to_string(i)}}, 1);
  }
  for (int round = 0; round < 3; ++round) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(server.port()));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    ASSERT_EQ(::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
    const std::string request = "GET /metrics HTTP/1.0\r\n\r\n";
    ASSERT_EQ(::write(fd, request.data(), request.size()), static_cast<ssize_t>(request.size()));
    // Linger 0: close() sends a reset instead of waiting to drain.
    linger lg{1, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_LINGER, &lg, sizeof(lg));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    ::close(fd);
  }
  auto body = httpBody(httpGet(server.port(), "/metrics"));
  EXPECT_NE(body.find("big_total{series=\"19999\"} 1"), std::string::npos);
}

// With SIGUSR1 blocked in this thread it stays pending unless the accept thread takes it.
static std::atomic<int> g_server_signal_runs{0};
static void CountServerSignal(int) { g_server_signal_runs.fetch_add(1); }

TEST(PrometheusMetricsServerClients, AcceptThreadDoesNotTakeProcessSignals) {
  MetricsServer server(0, "127.0.0.1");

  struct sigaction sa{};
  struct sigaction old_sa{};
  sa.sa_handler = CountServerSignal;
  ASSERT_EQ(sigaction(SIGUSR1, &sa, &old_sa), 0);
  sigset_t usr1, old_mask;
  sigemptyset(&usr1);
  sigaddset(&usr1, SIGUSR1);
  ASSERT_EQ(pthread_sigmask(SIG_BLOCK, &usr1, &old_mask), 0);

  g_server_signal_runs.store(0);
  ASSERT_EQ(kill(getpid(), SIGUSR1), 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_EQ(g_server_signal_runs.load(), 0) << "the accept thread handled a process signal";

  ASSERT_EQ(pthread_sigmask(SIG_SETMASK, &old_mask, nullptr), 0);  // now it is ours to take
  for (int i = 0; i < 200 && g_server_signal_runs.load() == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_EQ(g_server_signal_runs.load(), 1);
  sigaction(SIGUSR1, &old_sa, nullptr);
}

}  // namespace mori::metrics
