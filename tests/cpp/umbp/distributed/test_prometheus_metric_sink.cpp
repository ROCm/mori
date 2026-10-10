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

// ---------------------------------------------------------------------------
//  The masterless observability path.
//
//  What used to happen: everything a peer measured went to MasterClient, which
//  shipped it to a master.  A node with no master measured all of it and then
//  had nowhere to put it, so the standalone server — the deployment sglang's
//  direct external linker actually runs — served no /metrics at all and no
//  flag could make it.
//
//  These tests cover the two halves of the fix that can be asserted without
//  standing up a server: the sink that writes into a local MetricsServer, and
//  the port-resolution rule that decides whether one is created.  The wiring
//  (PoolClient choosing a sink, StandaloneServer owning it) is exercised by
//  the standalone server's own integration tests.
// ---------------------------------------------------------------------------

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <string>

#include "mori/metrics/prometheus_metrics_server.hpp"
#include "umbp/distributed/metrics/prometheus_metric_sink.h"
#include "umbp/standalone/standalone_server.h"

namespace mori::umbp {
namespace {

// Bind an ephemeral port, note it, and let it go: gives the test a port that
// is free right now without hard-coding one that a parallel ctest job may hold.
int PickFreePort() {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  EXPECT_GE(fd, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  EXPECT_EQ(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
  socklen_t len = sizeof(addr);
  EXPECT_EQ(::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len), 0);
  const int port = ntohs(addr.sin_port);
  ::close(fd);
  return port;
}

std::string Scrape(int port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  EXPECT_GE(fd, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return {};
  }
  const std::string req = "GET /metrics HTTP/1.0\r\n\r\n";
  EXPECT_GT(::send(fd, req.data(), req.size(), 0), 0);
  std::string out;
  char buf[4096];
  ssize_t n = 0;
  while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) out.append(buf, static_cast<size_t>(n));
  ::close(fd);
  return out;
}

// RAII for an env var, so a failing assertion cannot leak it into the next test.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    const char* prev = std::getenv(name);
    had_prev_ = prev != nullptr;
    if (had_prev_) prev_ = prev;
    if (value == nullptr) {
      ::unsetenv(name);
    } else {
      ::setenv(name, value, 1);
    }
  }
  ~ScopedEnv() {
    if (had_prev_) {
      ::setenv(name_, prev_.c_str(), 1);
    } else {
      ::unsetenv(name_);
    }
  }

 private:
  const char* name_;
  bool had_prev_ = false;
  std::string prev_;
};

// --------------------------------------------------------------------------
//  PrometheusMetricSink
// --------------------------------------------------------------------------

TEST(PrometheusMetricSink, ServesCountersGaugesAndHistogramsItWasGiven) {
  PrometheusMetricSink sink(PickFreePort(), "node-a");

  sink.AddCounter("mori_umbp_test_bytes_total", "help", {{"op", "commit"}}, 4096);
  sink.AddCounter("mori_umbp_test_bytes_total", "help", {{"op", "commit"}}, 1024);
  sink.SetGauge("mori_umbp_test_pages", "help", {{"tier", "DRAM"}}, 17);
  sink.Observe("mori_umbp_test_latency_seconds", "help", {{"rpc", "Get"}}, {0.1, 1.0}, 0.5);

  const std::string body = Scrape(sink.port());
  ASSERT_FALSE(body.empty()) << "metrics endpoint did not answer";

  // Counter deltas accumulate rather than overwrite.
  EXPECT_NE(body.find("mori_umbp_test_bytes_total"), std::string::npos);
  EXPECT_NE(body.find("5120"), std::string::npos);
  EXPECT_NE(body.find("mori_umbp_test_pages"), std::string::npos);
  EXPECT_NE(body.find("17"), std::string::npos);
  // Histograms land as a real Prometheus histogram, not a bare number.
  EXPECT_NE(body.find("mori_umbp_test_latency_seconds_bucket"), std::string::npos);
  EXPECT_NE(body.find("mori_umbp_test_latency_seconds_count"), std::string::npos);
}

TEST(PrometheusMetricSink, StampsNodeLabelSoTwoServersDoNotCollapse) {
  PrometheusMetricSink sink(PickFreePort(), "node-a");
  sink.AddCounter("mori_umbp_test_ops_total", "help", {{"op", "resolve"}}, 1);

  const std::string body = Scrape(sink.port());
  ASSERT_FALSE(body.empty());
  EXPECT_NE(body.find("node=\"node-a\""), std::string::npos)
      << "the node label master used to add on ingest must survive the masterless path";
}

TEST(PrometheusMetricSink, DoesNotOverwriteANodeLabelTheComponentSupplied) {
  PrometheusMetricSink sink(PickFreePort(), "node-a");
  // A series that already speaks for another node — the sink is forwarding it,
  // not authoring it, so it must not relabel it as its own.
  sink.SetGauge("mori_umbp_test_peer_pages", "help", {{"node", "node-b"}}, 3);

  const std::string body = Scrape(sink.port());
  ASSERT_FALSE(body.empty());
  EXPECT_NE(body.find("node=\"node-b\""), std::string::npos);
  EXPECT_EQ(body.find("node=\"node-a\""), std::string::npos);
}

TEST(PrometheusMetricSink, EmptyNodeIdStampsNothing) {
  PrometheusMetricSink sink(PickFreePort(), "");
  sink.AddCounter("mori_umbp_test_ops_total", "help", {{"op", "evict"}}, 1);

  const std::string body = Scrape(sink.port());
  ASSERT_FALSE(body.empty());
  EXPECT_EQ(body.find("node=\"\""), std::string::npos);
}

// --------------------------------------------------------------------------
//  Port resolution
//
//  The distinction these pin down: "serve nothing" and "the operator typed
//  something impossible" must not produce the same outcome, because the first
//  is a choice and the second is a typo that would otherwise be discovered as
//  a blank dashboard days later.
// --------------------------------------------------------------------------

TEST(StandaloneMetricsPort, UnsetUsesTheDefault) {
  ScopedEnv env("UMBP_STANDALONE_METRICS_PORT", nullptr);
  bool ok = false;
  std::string error;
  EXPECT_EQ(standalone::ResolveMetricsPortFromEnv(&ok, &error),
            standalone::kDefaultStandaloneMetricsPort);
  EXPECT_TRUE(ok);
  EXPECT_TRUE(error.empty());
}

TEST(StandaloneMetricsPort, DefaultIsNotTheMasterPort) {
  // A node may run a master and a standalone server; sharing 9091 would make
  // one of them fail to bind for reasons neither reports clearly.
  EXPECT_NE(standalone::kDefaultStandaloneMetricsPort, 9091);
}

TEST(StandaloneMetricsPort, ExplicitPortIsHonored) {
  ScopedEnv env("UMBP_STANDALONE_METRICS_PORT", "19099");
  bool ok = false;
  std::string error;
  EXPECT_EQ(standalone::ResolveMetricsPortFromEnv(&ok, &error), 19099);
  EXPECT_TRUE(ok);
}

TEST(StandaloneMetricsPort, ZeroAndWordsDisableIt) {
  bool ok = false;
  std::string error;
  {
    ScopedEnv env("UMBP_STANDALONE_METRICS_PORT", "0");
    EXPECT_EQ(standalone::ResolveMetricsPortFromEnv(&ok, &error), 0);
    EXPECT_TRUE(ok);
  }
  {
    ScopedEnv env("UMBP_STANDALONE_METRICS_PORT", "OFF");
    EXPECT_EQ(standalone::ResolveMetricsPortFromEnv(&ok, &error), 0);
    EXPECT_TRUE(ok) << "case-insensitive: an operator writing OFF meant off";
  }
  {
    ScopedEnv env("UMBP_STANDALONE_METRICS_PORT", "false");
    EXPECT_EQ(standalone::ResolveMetricsPortFromEnv(&ok, &error), 0);
    EXPECT_TRUE(ok);
  }
}

TEST(StandaloneMetricsPort, GarbageIsRejectedNotSilentlyDisabled) {
  bool ok = true;
  std::string error;
  {
    ScopedEnv env("UMBP_STANDALONE_METRICS_PORT", "9O92");  // letter O, not zero
    EXPECT_EQ(standalone::ResolveMetricsPortFromEnv(&ok, &error), 0);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(error.empty());
  }
  {
    ok = true;
    error.clear();
    ScopedEnv env("UMBP_STANDALONE_METRICS_PORT", "70000");
    EXPECT_EQ(standalone::ResolveMetricsPortFromEnv(&ok, &error), 0);
    EXPECT_FALSE(ok);
    EXPECT_FALSE(error.empty());
  }
}

}  // namespace
}  // namespace mori::umbp
