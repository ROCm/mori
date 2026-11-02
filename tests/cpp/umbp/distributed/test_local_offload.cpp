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

class ScopedOffloadEnv {
 public:
  explicit ScopedOffloadEnv(const char* value) {
    if (const char* old = std::getenv("UMBP_DISTRIBUTED_OFFLOAD")) old_ = old;
    if (value)
      setenv("UMBP_DISTRIBUTED_OFFLOAD", value, 1);
    else
      unsetenv("UMBP_DISTRIBUTED_OFFLOAD");
  }
  ~ScopedOffloadEnv() {
    if (old_)
      setenv("UMBP_DISTRIBUTED_OFFLOAD", old_->c_str(), 1);
    else
      unsetenv("UMBP_DISTRIBUTED_OFFLOAD");
  }

 private:
  std::optional<std::string> old_;
};

// Real Master/metadata RPCs and local byte copies; no GPU or RDMA hardware.
class LocalOffloadTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    MasterServerConfig cfg;
    cfg.listen_address = "127.0.0.1:0";
    cfg.eviction_config.high_watermark = 2.0;
    cfg.eviction_config.low_watermark = 1.5;
    master_ = std::make_unique<MasterServer>(std::move(cfg));
    thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 500 && master_->GetBoundPort() == 0; ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    ASSERT_NE(master_->GetBoundPort(), 0);
    address_ = "127.0.0.1:" + std::to_string(master_->GetBoundPort());
    target_ = Start("target", "0");
    ASSERT_NE(target_, nullptr);
    for (size_t i = 0; i < data_.size(); ++i) data_[i] = static_cast<char>(i * 37);
  }
  void TearDown() override {
    clients_.clear();
    if (master_) master_->Shutdown();
    if (thread_.joinable()) thread_.join();
  }
  PoolClient* Start(const std::string& name, const char* flag, size_t capacity = 4 * kObject) {
    ScopedOffloadEnv env(flag);
    PoolClientConfig cfg;
    cfg.master_config.node_id = name;
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = address_;
    cfg.io_engine.host.clear();
    cfg.auto_peer_service_port = true;
    cfg.dram_page_size = kPage;
    cfg.dram.buffer_sizes = {capacity};
    cfg.cache_remote_fetches = false;
    cfg.ranged_locality_prefetch = false;
    cfg.local_evict_high_watermark = 2.0;
    cfg.local_evict_low_watermark = 1.5;
    auto client = std::make_unique<PoolClient>(std::move(cfg));
    if (!client->Init()) return nullptr;
    auto* result = client.get();
    clients_.push_back(std::move(client));
    return result;
  }
  std::vector<bool> Put(PoolClient* client, const std::vector<std::string>& keys) {
    if (GetParam()) {
      return client->BatchPutRanges(
          keys, std::vector<size_t>(keys.size(), kObject),
          std::vector<std::vector<const void*>>(keys.size(), {data_.data(), data_.data() + 1500}),
          std::vector<std::vector<size_t>>(keys.size(), {1500, kObject - 1500}),
          std::vector<std::vector<size_t>>(keys.size(), {0, 1500}));
    }
    return client->BatchPut(keys, std::vector<const void*>(keys.size(), data_.data()),
                            std::vector<size_t>(keys.size(), kObject));
  }
  bool Local(PoolClient* client, const std::string& key) {
    return client->Backends().Get(TierType::DRAM)->Contains(key);
  }
  bool WaitExists(PoolClient* client, const std::string& key) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      if (client->Exists(key)) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
  }
  void CheckBytes(PoolClient* client, const std::string& key) {
    std::vector<char> out(kObject);
    ASSERT_TRUE(client->Get(key, out.data(), out.size()));
    EXPECT_EQ(out, data_);
  }
  std::unique_ptr<MasterServer> master_;
  std::thread thread_;
  std::string address_;
  std::vector<std::unique_ptr<PoolClient>> clients_;
  PoolClient* target_ = nullptr;
  std::vector<char> data_ = std::vector<char>(kObject);
};

TEST_P(LocalOffloadTest, DefaultAndEnabledPreserveRemoteDedup) {
  ASSERT_EQ(Put(target_, {"remote"}), std::vector<bool>({true}));
  target_->Master().FlushHeartbeat();
  int id = 0;
  for (const char* flag : {static_cast<const char*>(nullptr), "1", "true", "ON"}) {
    auto* caller = Start("enabled-" + std::to_string(id++), flag);
    ASSERT_NE(caller, nullptr);
    ASSERT_TRUE(WaitExists(caller, "remote"));
    EXPECT_EQ(Put(caller, {"remote"}), std::vector<bool>({true}));
    EXPECT_FALSE(Local(caller, "remote"));
  }
}

TEST_P(LocalOffloadTest, LocalWritesPublishAndIgnoreRemoteDedup) {
  ASSERT_EQ(Put(target_, {"remote"}), std::vector<bool>({true}));
  target_->Master().FlushHeartbeat();
  int id = 0;
  for (const char* flag : {"0", "false", "OFF"}) {
    const std::string key = "local-" + std::to_string(id++);
    auto* caller = Start(key, flag);
    ASSERT_NE(caller, nullptr);
    ASSERT_TRUE(caller->HasMaster());
    ASSERT_TRUE(WaitExists(caller, "remote"));
    ASSERT_EQ(Put(caller, {key, "remote"}), std::vector<bool>({true, true}));
    ASSERT_TRUE(Local(caller, key));
    CheckBytes(caller, key);
    CheckBytes(caller, "remote");
    EXPECT_TRUE(Local(caller, "remote"));
    EXPECT_FALSE(Local(target_, key));
    EXPECT_EQ(Put(caller, {"remote"}), std::vector<bool>({true}));
    caller->Master().FlushHeartbeat();
    ASSERT_TRUE(WaitExists(target_, key));
    std::vector<std::optional<RouteGetResult>> routes;
    ASSERT_TRUE(target_->Master().BatchRouteGet({key}, {}, &routes).ok());
    ASSERT_EQ(routes.size(), 1);
    ASSERT_TRUE(routes[0].has_value());
    EXPECT_EQ(routes[0]->node_id, key);
  }
}

TEST_P(LocalOffloadTest, CapacityFailuresStayLocalAndArePerKey) {
  auto* caller = Start("full-local", "0", kObject);
  ASSERT_NE(caller, nullptr);
  const auto results = Put(caller, {"first", "second"});
  ASSERT_EQ(results.size(), 2);
  EXPECT_NE(results[0], results[1]);
  for (size_t i = 0; i < results.size(); ++i) {
    const std::string key = i == 0 ? "first" : "second";
    EXPECT_EQ(Local(caller, key), results[i]);
    EXPECT_FALSE(Local(target_, key));
    if (results[i]) CheckBytes(caller, key);
  }
  ASSERT_EQ(Put(target_, {"remote"}), std::vector<bool>({true}));
  target_->Master().FlushHeartbeat();
  ASSERT_TRUE(WaitExists(caller, "remote"));
  EXPECT_EQ(Put(caller, {"remote", "overflow"}), std::vector<bool>({false, false}));
  EXPECT_FALSE(Local(caller, "remote"));
  EXPECT_FALSE(Local(caller, "overflow"));
  EXPECT_FALSE(Local(target_, "overflow"));
}

TEST(LocalOffloadEnvTest, InvalidValuesFailInitialization) {
  for (const char* flag : {"", "2", "disabled", "flase"}) {
    ScopedOffloadEnv env(flag);
    PoolClient client(PoolClientConfig{});
    EXPECT_FALSE(client.Init());
    EXPECT_FALSE(client.IsInitialized());
  }
}

INSTANTIATE_TEST_SUITE_P(PutApis, LocalOffloadTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "Ranges" : "Contiguous";
                         });
}  // namespace
}  // namespace mori::umbp
