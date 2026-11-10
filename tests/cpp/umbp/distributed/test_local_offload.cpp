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

#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"

namespace mori::umbp {
namespace {
constexpr size_t kPage = 4096;
constexpr size_t kObject = 2 * kPage;

// Real metadata RPCs and local byte copies; no GPU or RDMA hardware.
class LocalOffloadTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    if (const char* value = std::getenv("UMBP_DISTRIBUTED_OFFLOAD")) old_env_ = value;
    for (size_t i = 0; i < data_.size(); ++i) data_[i] = static_cast<char>(i * 37);
    MasterServerConfig cfg;
    cfg.listen_address = "127.0.0.1:0";
    cfg.eviction_config.high_watermark = 2.0;
    master_ = std::make_unique<MasterServer>(std::move(cfg));
    thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 500 && master_->GetBoundPort() == 0; ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ASSERT_NE(master_->GetBoundPort(), 0);
  }
  void TearDown() override {
    if (master_) master_->Shutdown();
    if (thread_.joinable()) thread_.join();
    if (old_env_)
      setenv("UMBP_DISTRIBUTED_OFFLOAD", old_env_->c_str(), 1);
    else
      unsetenv("UMBP_DISTRIBUTED_OFFLOAD");
  }
  std::unique_ptr<PoolClient> Start(const char* node, const char* flag,
                                    size_t capacity = 2 * kObject) {
    if (flag)
      setenv("UMBP_DISTRIBUTED_OFFLOAD", flag, 1);
    else
      unsetenv("UMBP_DISTRIBUTED_OFFLOAD");
    PoolClientConfig cfg;
    cfg.master_config.node_id = node;
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = "127.0.0.1:" + std::to_string(master_->GetBoundPort());
    cfg.io_engine.host.clear();
    cfg.auto_peer_service_port = true;
    cfg.dram_page_size = kPage;
    cfg.dram.buffer_sizes = {capacity};
    cfg.local_evict_high_watermark = 2.0;
    auto client = std::make_unique<PoolClient>(std::move(cfg));
    if (!client->Init()) return nullptr;
    return client;
  }
  std::vector<bool> Put(PoolClient& client, const std::vector<std::string>& keys) {
    if (GetParam()) {
      return client.BatchPutRanges(
          keys, std::vector<size_t>(keys.size(), kObject),
          std::vector<std::vector<const void*>>(keys.size(), {data_.data(), data_.data() + 1500}),
          std::vector<std::vector<size_t>>(keys.size(), {1500, kObject - 1500}),
          std::vector<std::vector<size_t>>(keys.size(), {0, 1500}));
    }
    return client.BatchPut(keys, std::vector<const void*>(keys.size(), data_.data()),
                           std::vector<size_t>(keys.size(), kObject));
  }
  bool Local(PoolClient& client, const std::string& key) {
    return client.Backends().Get(TierType::DRAM)->Contains(key);
  }
  bool WaitExists(PoolClient& client, const std::string& key) {
    for (int i = 0; i < 500; ++i) {
      if (client.Exists(key)) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
  }
  std::unique_ptr<MasterServer> master_;
  std::thread thread_;
  std::optional<std::string> old_env_;
  std::vector<char> data_ = std::vector<char>(kObject);
};

TEST_P(LocalOffloadTest, LocalPlacementRetainsSharingAndDoesNotSpill) {
  auto target = Start("target", "0", 4 * kObject);
  ASSERT_NE(target, nullptr);
  ASSERT_EQ(Put(*target, {"remote", "remote-full"}), std::vector<bool>({true, true}));
  target->Master().FlushHeartbeat();
  auto caller = Start("default", nullptr);
  ASSERT_NE(caller, nullptr);
  ASSERT_TRUE(WaitExists(*caller, "remote"));
  ASSERT_EQ(Put(*caller, {"remote"}), std::vector<bool>({true}));
  EXPECT_FALSE(Local(*caller, "remote"));
  caller.reset();

  caller = Start("local", "0");
  ASSERT_NE(caller, nullptr);
  ASSERT_EQ(Put(*caller, {"remote"}), std::vector<bool>({true}));
  ASSERT_TRUE(Local(*caller, "remote"));
  std::vector<char> out(kObject);
  ASSERT_TRUE(caller->Get("remote", out.data(), out.size()));
  EXPECT_EQ(out, data_);
  const auto results = Put(*caller, {"first", "second"});
  ASSERT_EQ(results.size(), 2);
  ASSERT_NE(results[0], results[1]);
  for (size_t i = 0; i < results.size(); ++i) {
    const std::string key = i == 0 ? "first" : "second";
    EXPECT_EQ(Local(*caller, key), results[i]);
    EXPECT_FALSE(Local(*target, key));
  }
  EXPECT_EQ(Put(*caller, {"remote-full"}), std::vector<bool>({false}));
  EXPECT_FALSE(Local(*caller, "remote-full"));
  caller->Master().FlushHeartbeat();
  const std::string resident = results[0] ? "first" : "second";
  ASSERT_TRUE(WaitExists(*target, resident));
  std::vector<std::optional<RouteGetResult>> routes;
  ASSERT_TRUE(target->Master().BatchRouteGet({resident}, {}, &routes).ok());
  ASSERT_EQ(routes.size(), 1);
  ASSERT_TRUE(routes[0]);
  EXPECT_EQ(routes[0]->node_id, "local");
}

INSTANTIATE_TEST_SUITE_P(PutApis, LocalOffloadTest, ::testing::Bool());
}  // namespace
}  // namespace mori::umbp
