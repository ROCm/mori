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
#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_metrics.h"
#include "umbp/distributed/master/master_server.h"

namespace mori::umbp {
namespace {

uint16_t AllocPort() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return 50400;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = 0;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return 50400;
  }
  socklen_t len = sizeof(addr);
  ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
  ::close(fd);
  return ntohs(addr.sin_port);
}

bool WaitFor(const std::function<bool()>& pred, std::chrono::milliseconds timeout,
             std::chrono::milliseconds poll = std::chrono::milliseconds(20)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::sleep_for(poll);
  }
  return pred();
}

class MasterServerTransferRatesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("UMBP_METRICS_REPORT_INTERVAL_MS", "50", 1);
    setenv("UMBP_TRANSFER_RATE_MIN_SAMPLE_GAP_MS", "10", 1);
    setenv("UMBP_TRANSFER_RATE_MAX_SAMPLE_AGE_MS", "5000", 1);
    setenv("UMBP_TRANSFER_RATE_TICK_MIN_GAP_MS", "20", 1);
    setenv("UMBP_HEARTBEAT_TTL_SEC", "2", 1);

    port_ = AllocPort();
    addr_ = "127.0.0.1:" + std::to_string(port_);

    MasterServerConfig server_cfg = MasterServerConfig::FromEnvironment();
    server_cfg.listen_address = addr_;
    server_cfg.metrics_port = 0;
    server_ = std::make_unique<MasterServer>(std::move(server_cfg));
    server_thread_ = std::thread([this] { server_->Run(); });

    ASSERT_TRUE(WaitFor([this] { return server_->GetBoundPort() != 0; }, std::chrono::seconds(5)));
  }

  void TearDown() override {
    client_.reset();
    server_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();

    unsetenv("UMBP_METRICS_REPORT_INTERVAL_MS");
    unsetenv("UMBP_TRANSFER_RATE_MIN_SAMPLE_GAP_MS");
    unsetenv("UMBP_TRANSFER_RATE_MAX_SAMPLE_AGE_MS");
    unsetenv("UMBP_TRANSFER_RATE_TICK_MIN_GAP_MS");
    unsetenv("UMBP_HEARTBEAT_TTL_SEC");
  }

  UMBPMasterClientConfig MakeClientConfig() const {
    UMBPMasterClientConfig cfg;
    cfg.master_address = addr_;
    cfg.node_id = "rate-node";
    cfg.node_address = "127.0.0.1:9997";
    cfg.auto_heartbeat = false;
    cfg.tags = {"sgl_role=prefill"};
    return cfg;
  }

  uint16_t port_ = 0;
  std::string addr_;
  std::unique_ptr<MasterServer> server_;
  std::thread server_thread_;
  std::unique_ptr<MasterClient> client_;
};

TEST_F(MasterServerTransferRatesTest, ReportMetricsFeedsTransferRateRpcWithPrometheusDisabled) {
  client_ = std::make_unique<MasterClient>(MakeClientConfig());
  ASSERT_TRUE(client_->RegisterSelf(/*tier_capacities=*/{}, /*peer_address=*/"peer:1234").ok());

  std::vector<MasterClient::ClientTransferRates> clients;
  ASSERT_TRUE(client_->GetClientTransferRates({}, &clients).ok());
  ASSERT_EQ(clients.size(), 1u);
  EXPECT_EQ(clients[0].node_id, "rate-node");
  EXPECT_EQ(clients[0].peer_address, "peer:1234");
  EXPECT_EQ(clients[0].tags, std::vector<std::string>({"sgl_role=prefill"}));
  EXPECT_TRUE(clients[0].rates.empty());

  client_->AddCounter(MORI_UMBP_METRIC_HICACHE_TRANSFER_BYTES_TOTAL,
                      MORI_UMBP_METRIC_HICACHE_TRANSFER_BYTES_TOTAL_HELP,
                      {{"direction", "l2_to_l1"}}, 1000.0);
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  client_->AddCounter(MORI_UMBP_METRIC_HICACHE_TRANSFER_BYTES_TOTAL,
                      MORI_UMBP_METRIC_HICACHE_TRANSFER_BYTES_TOTAL_HELP,
                      {{"direction", "l2_to_l1"}}, 2000.0);

  const bool saw_rate = WaitFor(
      [this] {
        std::vector<MasterClient::ClientTransferRates> out;
        if (!client_->GetClientTransferRates({"rate-node"}, &out).ok()) return false;
        if (out.size() != 1 || out[0].rates.size() != 1) return false;
        const auto& rate = out[0].rates[0];
        return rate.direction == HiCacheTransfer::L2_TO_L1 && rate.bytes_per_sec > 0.0 &&
               rate.rate_age_ms < 2000 && rate.window_ms >= 10;
      },
      std::chrono::seconds(5));
  ASSERT_TRUE(saw_rate);

  ASSERT_TRUE(client_->UnregisterSelf().ok());
  clients.clear();
  ASSERT_TRUE(client_->GetClientTransferRates({"rate-node"}, &clients).ok());
  EXPECT_TRUE(clients.empty());
}

}  // namespace
}  // namespace mori::umbp
