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
// Tests for the client-tag feature:
//
//  Suite 1 — ClientRegistryTagsTest
//    Unit tests directly on ClientRegistry: verify tags are stored on
//    RegisterClient and returned verbatim by GetClientTags.
//
//  Suite 2 — MasterClientTagsE2ETest
//    Integration test: MasterClient registers with tags via gRPC, the real
//    MasterServer stores them, and ReportMetrics injects them as Prometheus
//    labels on top of the usual {node=...} base.
//
// The test uses a free ephemeral port to avoid collisions with other tests.

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
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "umbp.grpc.pb.h"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/client_registry.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/types.h"

namespace mori::umbp {
namespace {

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------

static uint16_t AllocPort() {
  // Bind to :0 and let the kernel pick a free port, then close immediately.
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return 50400;
  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = 0;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return 50400;
  }
  socklen_t len = sizeof(addr);
  ::getsockname(fd, reinterpret_cast<struct sockaddr*>(&addr), &len);
  ::close(fd);
  return ntohs(addr.sin_port);
}

static bool WaitFor(std::function<bool()> pred, std::chrono::milliseconds timeout,
                    std::chrono::milliseconds poll = std::chrono::milliseconds(50)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::sleep_for(poll);
  }
  return pred();
}

// ---------------------------------------------------------------------------
//  Suite 1: ClientRegistry unit tests (no gRPC)
// ---------------------------------------------------------------------------

TEST(ClientRegistryTagsTest, TagsStoredOnRegister) {
  ClientRegistry reg(ClientRegistryConfig{});
  const std::vector<std::string> tags = {"sgl_role=prefill", "env=test"};
  ASSERT_TRUE(reg.RegisterClient("n1", "127.0.0.1:9001", {}, /*peer=*/"",
                                 /*engine=*/{}, tags));
  EXPECT_EQ(reg.GetClientTags("n1"), tags);
}

TEST(ClientRegistryTagsTest, EmptyTagsReturnedForUnknownNode) {
  ClientRegistry reg(ClientRegistryConfig{});
  EXPECT_TRUE(reg.GetClientTags("ghost").empty());
}

TEST(ClientRegistryTagsTest, EmptyTagsWhenNoneProvided) {
  ClientRegistry reg(ClientRegistryConfig{});
  ASSERT_TRUE(reg.RegisterClient("n1", "127.0.0.1:9002", {}));
  EXPECT_TRUE(reg.GetClientTags("n1").empty());
}

TEST(ClientRegistryTagsTest, TagsUnchangedByHeartbeat) {
  ClientRegistry reg(ClientRegistryConfig{});
  const std::vector<std::string> tags = {"sgl_role=decode"};
  ASSERT_TRUE(reg.RegisterClient("n1", "127.0.0.1:9003", {}, "", {}, tags));

  uint64_t acked = 0;
  bool request_full_sync = false;
  reg.Heartbeat("n1", {}, {}, /*is_full_sync=*/false, 0, &acked, &request_full_sync);

  EXPECT_EQ(reg.GetClientTags("n1"), tags);
}

TEST(ClientRegistryTagsTest, TagsClearedAfterUnregister) {
  ClientRegistry reg(ClientRegistryConfig{});
  ASSERT_TRUE(reg.RegisterClient("n1", "127.0.0.1:9004", {}, "", {}, {"sgl_role=prefill"}));
  reg.UnregisterClient("n1");
  EXPECT_TRUE(reg.GetClientTags("n1").empty());
}

TEST(ClientRegistryTagsTest, MultipleNodesHaveIndependentTags) {
  ClientRegistry reg(ClientRegistryConfig{});
  ASSERT_TRUE(reg.RegisterClient("p", "127.0.0.1:9005", {}, "", {}, {"sgl_role=prefill"}));
  ASSERT_TRUE(reg.RegisterClient("d", "127.0.0.1:9006", {}, "", {}, {"sgl_role=decode"}));

  EXPECT_EQ(reg.GetClientTags("p"), std::vector<std::string>{"sgl_role=prefill"});
  EXPECT_EQ(reg.GetClientTags("d"), std::vector<std::string>{"sgl_role=decode"});
}

// ---------------------------------------------------------------------------
//  Suite 2: End-to-end via MasterClient + MasterServer
// ---------------------------------------------------------------------------

// Captures every ReportMetrics request body verbatim.
class CapturingMasterService final : public ::umbp::UMBPMaster::Service {
 public:
  grpc::Status RegisterClient(grpc::ServerContext*, const ::umbp::RegisterClientRequest* req,
                              ::umbp::RegisterClientResponse* resp) override {
    resp->set_heartbeat_interval_ms(50);
    std::lock_guard<std::mutex> lock(mu_);
    last_reg_tags_.assign(req->tags().begin(), req->tags().end());
    registered_.store(true);
    return grpc::Status::OK;
  }

  grpc::Status Heartbeat(grpc::ServerContext*, const ::umbp::HeartbeatRequest*,
                         ::umbp::HeartbeatResponse* resp) override {
    resp->set_status(::umbp::CLIENT_STATUS_ALIVE);
    resp->set_acked_seq(1);
    return grpc::Status::OK;
  }

  grpc::Status UnregisterClient(grpc::ServerContext*, const ::umbp::UnregisterClientRequest*,
                                ::umbp::UnregisterClientResponse*) override {
    return grpc::Status::OK;
  }

  grpc::Status ReportMetrics(grpc::ServerContext*, const ::umbp::ReportMetricsRequest* req,
                             ::umbp::ReportMetricsResponse*) override {
    std::lock_guard<std::mutex> lock(mu_);
    report_requests_.push_back(*req);
    return grpc::Status::OK;
  }

  // Stubs for unused RPCs
  grpc::Status RouteGet(grpc::ServerContext*, const ::umbp::RouteGetRequest*,
                        ::umbp::RouteGetResponse*) override {
    return grpc::Status::OK;
  }
  grpc::Status RoutePut(grpc::ServerContext*, const ::umbp::RoutePutRequest*,
                        ::umbp::RoutePutResponse*) override {
    return grpc::Status::OK;
  }
  grpc::Status BatchRouteGet(grpc::ServerContext*, const ::umbp::BatchRouteGetRequest*,
                             ::umbp::BatchRouteGetResponse*) override {
    return grpc::Status::OK;
  }
  grpc::Status BatchRoutePut(grpc::ServerContext*, const ::umbp::BatchRoutePutRequest*,
                             ::umbp::BatchRoutePutResponse*) override {
    return grpc::Status::OK;
  }
  grpc::Status MatchExternalKv(grpc::ServerContext*, const ::umbp::MatchExternalKvRequest*,
                               ::umbp::MatchExternalKvResponse*) override {
    return grpc::Status::OK;
  }

  std::vector<std::string> LastRegTags() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_reg_tags_;
  }

  std::vector<::umbp::ReportMetricsRequest> ReportRequests() const {
    std::lock_guard<std::mutex> lock(mu_);
    return report_requests_;
  }

  bool Registered() const { return registered_.load(); }

 private:
  mutable std::mutex mu_;
  std::vector<std::string> last_reg_tags_;
  std::vector<::umbp::ReportMetricsRequest> report_requests_;
  std::atomic<bool> registered_{false};
};

class MasterClientTagsE2ETest : public ::testing::Test {
 protected:
  void SetUp() override {
    port_ = AllocPort();
    addr_ = "127.0.0.1:" + std::to_string(port_);

    svc_ = std::make_unique<CapturingMasterService>();
    grpc::ServerBuilder builder;
    builder.AddListeningPort(addr_, grpc::InsecureServerCredentials());
    builder.RegisterService(svc_.get());
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);
  }

  void TearDown() override {
    client_.reset();
    server_->Shutdown(std::chrono::system_clock::now() + std::chrono::seconds(2));
  }

  UMBPMasterClientConfig MakeConfig(std::vector<std::string> tags = {}) {
    UMBPMasterClientConfig cfg;
    cfg.master_address = addr_;
    cfg.node_id = "test-node";
    cfg.node_address = "127.0.0.1:9999";
    cfg.auto_heartbeat = false;
    cfg.tags = std::move(tags);
    return cfg;
  }

  uint16_t port_;
  std::string addr_;
  std::unique_ptr<CapturingMasterService> svc_;
  std::unique_ptr<grpc::Server> server_;
  std::unique_ptr<MasterClient> client_;
};

TEST_F(MasterClientTagsE2ETest, TagsSentOnRegisterSelf) {
  client_ = std::make_unique<MasterClient>(MakeConfig({"sgl_role=prefill", "env=ci"}));
  auto status = client_->RegisterSelf({});
  ASSERT_TRUE(status.ok()) << status.error_message();

  ASSERT_TRUE(WaitFor([this] { return svc_->Registered(); }, std::chrono::seconds(3)));

  const auto tags = svc_->LastRegTags();
  ASSERT_EQ(tags.size(), 2u);
  EXPECT_EQ(tags[0], "sgl_role=prefill");
  EXPECT_EQ(tags[1], "env=ci");
}

TEST_F(MasterClientTagsE2ETest, EmptyTagsSentWhenNoneConfigured) {
  client_ = std::make_unique<MasterClient>(MakeConfig());
  auto status = client_->RegisterSelf({});
  ASSERT_TRUE(status.ok()) << status.error_message();

  ASSERT_TRUE(WaitFor([this] { return svc_->Registered(); }, std::chrono::seconds(3)));

  EXPECT_TRUE(svc_->LastRegTags().empty());
}

// Verify that MasterClient::AddCounter forwards labels to the server and
// that a capturing server would see the node label.  The real tag-injection
// into ReportMetrics base labels is exercised in the real-MasterServer suite
// below because CapturingMasterService doesn't run ClientRegistry.
TEST_F(MasterClientTagsE2ETest, ReportMetricsCarriesNodeId) {
  setenv("UMBP_METRICS_REPORT_INTERVAL_MS", "50", 1);
  client_ = std::make_unique<MasterClient>(MakeConfig({"sgl_role=decode"}));
  auto status = client_->RegisterSelf({});
  ASSERT_TRUE(status.ok()) << status.error_message();

  client_->AddCounter("test_counter", "help", {{"op", "put"}}, 1.0);

  // Wait for the metrics flush thread to run
  const bool flushed =
      WaitFor([this] { return !svc_->ReportRequests().empty(); }, std::chrono::seconds(3));
  ASSERT_TRUE(flushed) << "No ReportMetrics RPC received";

  const auto reqs = svc_->ReportRequests();
  ASSERT_FALSE(reqs.empty());
  EXPECT_EQ(reqs[0].node_id(), "test-node");
}

// ---------------------------------------------------------------------------
//  Suite 3: Real MasterServer — tags injected into ReportMetrics labels
// ---------------------------------------------------------------------------

class RealMasterServerTagsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("UMBP_HEARTBEAT_TTL_SEC", "2", 1);
    setenv("UMBP_METRICS_REPORT_INTERVAL_MS", "50", 1);

    port_ = AllocPort();
    metrics_port_ = AllocPort();
    addr_ = "127.0.0.1:" + std::to_string(port_);

    MasterServerConfig cfg = MasterServerConfig::FromEnvironment();
    cfg.listen_address = addr_;
    cfg.metrics_port = metrics_port_;
    server_ = std::make_unique<MasterServer>(std::move(cfg));
    server_thread_ = std::thread([this] { server_->Run(); });

    // Wait for the gRPC port to be bound
    ASSERT_TRUE(WaitFor([this] { return server_->GetBoundPort() != 0; }, std::chrono::seconds(5)));
  }

  void TearDown() override {
    client_.reset();
    server_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
    unsetenv("UMBP_HEARTBEAT_TTL_SEC");
    unsetenv("UMBP_METRICS_REPORT_INTERVAL_MS");
  }

  UMBPMasterClientConfig MakeConfig(std::vector<std::string> tags = {}) {
    UMBPMasterClientConfig cfg;
    cfg.master_address = addr_;
    cfg.node_id = "real-node";
    cfg.node_address = "127.0.0.1:9998";
    cfg.auto_heartbeat = false;
    cfg.tags = std::move(tags);
    return cfg;
  }

  // Fetch raw Prometheus text from the metrics HTTP server.
  std::string FetchMetrics() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return "";
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(metrics_port_);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
      ::close(fd);
      return "";
    }
    const char* req = "GET /metrics HTTP/1.0\r\nHost: localhost\r\n\r\n";
    ::send(fd, req, strlen(req), 0);
    std::string resp;
    char buf[4096];
    ssize_t n;
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) resp.append(buf, static_cast<size_t>(n));
    ::close(fd);
    // Strip HTTP headers
    const auto sep = resp.find("\r\n\r\n");
    return sep != std::string::npos ? resp.substr(sep + 4) : resp;
  }

  uint16_t port_;
  uint16_t metrics_port_;
  std::string addr_;
  std::unique_ptr<MasterServer> server_;
  std::thread server_thread_;
  std::unique_ptr<MasterClient> client_;
};

TEST_F(RealMasterServerTagsTest, TagsAppearInReportedMetricLabels) {
  client_ = std::make_unique<MasterClient>(MakeConfig({"sgl_role=prefill"}));
  ASSERT_TRUE(client_->RegisterSelf({}).ok());

  // Emit a counter and let the flush thread send it to the real MasterServer.
  client_->AddCounter("mori_test_tags_counter", "tags test", {{"op", "put"}}, 42.0);

  // Wait for the metric to appear in Prometheus output with the tag label.
  const bool found = WaitFor(
      [this] {
        const auto body = FetchMetrics();
        return body.find("sgl_role=\"prefill\"") != std::string::npos &&
               body.find("mori_test_tags_counter") != std::string::npos;
      },
      std::chrono::seconds(5));

  EXPECT_TRUE(found) << "Expected sgl_role=prefill label in /metrics output:\n" << FetchMetrics();
}

TEST_F(RealMasterServerTagsTest, NoTagsWhenNoneRegistered) {
  client_ = std::make_unique<MasterClient>(MakeConfig());  // no tags
  ASSERT_TRUE(client_->RegisterSelf({}).ok());

  client_->AddCounter("mori_test_notags_counter", "no tags test", {}, 1.0);

  // Wait for the metric to land; confirm sgl_role label is absent.
  bool metric_arrived = WaitFor(
      [this] { return FetchMetrics().find("mori_test_notags_counter") != std::string::npos; },
      std::chrono::seconds(5));
  ASSERT_TRUE(metric_arrived) << "Metric never appeared";

  const auto body = FetchMetrics();
  EXPECT_EQ(body.find("sgl_role="), std::string::npos) << "Unexpected sgl_role label in /metrics:\n"
                                                       << body;
}

}  // namespace
}  // namespace mori::umbp
