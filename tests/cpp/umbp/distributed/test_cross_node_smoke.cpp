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

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"

namespace mori::umbp {
namespace {

constexpr size_t kBufSize = 1 << 20;
constexpr size_t kBlockSize = 4096;
static uint16_t AllocPort() {
  static std::atomic<uint16_t> next{0};
  if (next.load() == 0) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    next.store(static_cast<uint16_t>(51000 + (std::rand() % 5000)));
  }
  return next.fetch_add(100);
}

class CrossNodeSmoke : public ::testing::Test {
 protected:
  void SetUp() override {
    uint16_t base = AllocPort();
    master_port_ = base;
    io_port_a_ = base + 1;
    io_port_b_ = base + 2;

    buf_a_ = std::malloc(kBufSize);
    buf_b_ = std::malloc(kBufSize);
    caller_buf_ = std::malloc(kBufSize);
    read_buf_ = std::malloc(kBufSize);
    ASSERT_NE(buf_a_, nullptr);
    ASSERT_NE(buf_b_, nullptr);
    ASSERT_NE(caller_buf_, nullptr);
    ASSERT_NE(read_buf_, nullptr);
    std::memset(buf_a_, 0, kBufSize);
    std::memset(buf_b_, 0, kBufSize);
    std::memset(caller_buf_, 0, kBufSize);
    std::memset(read_buf_, 0, kBufSize);

    MasterServerConfig master_cfg;
    master_cfg.listen_address = "0.0.0.0:" + std::to_string(master_port_);
    master_ = std::make_unique<MasterServer>(std::move(master_cfg));
    server_thread_ = std::thread([this] { master_->Run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::string master_addr = "localhost:" + std::to_string(master_port_);

    PoolClientConfig cfg_a;
    cfg_a.master_config.node_id = "node-a";
    cfg_a.master_config.node_address = "127.0.0.1";
    cfg_a.master_config.master_address = master_addr;
    cfg_a.io_engine_host = "0.0.0.0";
    cfg_a.io_engine_port = io_port_a_;
    cfg_a.dram_buffers = {{buf_a_, kBlockSize}};
    cfg_a.tier_capacities = {{TierType::DRAM, {kBlockSize, kBlockSize}}};
    client_a_ = std::make_unique<PoolClient>(std::move(cfg_a));
    ASSERT_TRUE(client_a_->Init());

    PoolClientConfig cfg_b;
    cfg_b.master_config.node_id = "node-b";
    cfg_b.master_config.node_address = "127.0.0.1";
    cfg_b.master_config.master_address = master_addr;
    cfg_b.io_engine_host = "0.0.0.0";
    cfg_b.io_engine_port = io_port_b_;
    cfg_b.dram_buffers = {{buf_b_, kBufSize}};
    cfg_b.tier_capacities = {{TierType::DRAM, {kBufSize, kBufSize}}};
    client_b_ = std::make_unique<PoolClient>(std::move(cfg_b));
    ASSERT_TRUE(client_b_->Init());

    client_a_->RegisterMemory(caller_buf_, kBufSize);
    client_b_->RegisterMemory(read_buf_, kBufSize);
  }

  void TearDown() override {
    if (client_b_) client_b_->Shutdown();
    if (client_a_) client_a_->Shutdown();
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
    std::free(buf_a_);
    std::free(buf_b_);
    std::free(caller_buf_);
    std::free(read_buf_);
  }

  uint16_t master_port_ = 0;
  uint16_t io_port_a_ = 0;
  uint16_t io_port_b_ = 0;
  void* buf_a_ = nullptr;
  void* buf_b_ = nullptr;
  void* caller_buf_ = nullptr;
  void* read_buf_ = nullptr;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> client_a_;
  std::unique_ptr<PoolClient> client_b_;
};

TEST_F(CrossNodeSmoke, PutGetWithRDMA) {
  std::memset(caller_buf_, 0xAB, kBlockSize);

  ASSERT_TRUE(client_a_->PutRemote("rdma-key", caller_buf_, kBlockSize));
  EXPECT_TRUE(client_a_->ExistsRemote("rdma-key"));
  EXPECT_TRUE(client_b_->ExistsRemote("rdma-key"));

  std::memset(read_buf_, 0, kBlockSize);
  ASSERT_TRUE(client_b_->GetRemote("rdma-key", read_buf_, kBlockSize));
  EXPECT_EQ(std::memcmp(caller_buf_, read_buf_, kBlockSize), 0);
}

TEST_F(CrossNodeSmoke, BatchPutGetWithRDMA) {
  auto* src1 = static_cast<char*>(caller_buf_);
  auto* src2 = src1 + kBlockSize;
  auto* src3 = src2 + kBlockSize;
  std::memset(src1, 0x11, kBlockSize);
  std::memset(src2, 0x22, kBlockSize);
  std::memset(src3, 0x33, kBlockSize);

  std::vector<std::string> keys = {"bk1", "bk2", "bk3"};
  std::vector<const void*> srcs = {src1, src2, src3};
  std::vector<size_t> sizes = {kBlockSize, kBlockSize, kBlockSize};

  auto put_results = client_a_->BatchPutRemote(keys, srcs, sizes);
  ASSERT_EQ(put_results.size(), 3u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(put_results[i]) << "put failed for " << keys[i];
  }

  for (const auto& key : keys) {
    EXPECT_TRUE(client_a_->ExistsRemote(key));
    EXPECT_TRUE(client_b_->ExistsRemote(key));
  }

  auto* dst1 = static_cast<char*>(read_buf_);
  auto* dst2 = dst1 + kBlockSize;
  auto* dst3 = dst2 + kBlockSize;
  std::memset(read_buf_, 0, kBlockSize * 3);

  std::vector<void*> dsts = {dst1, dst2, dst3};
  auto get_results = client_b_->BatchGetRemote(keys, dsts, sizes);
  ASSERT_EQ(get_results.size(), 3u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(get_results[i]) << "get failed for " << keys[i];
  }
  EXPECT_EQ(std::memcmp(src1, dst1, kBlockSize), 0);
  EXPECT_EQ(std::memcmp(src2, dst2, kBlockSize), 0);
  EXPECT_EQ(std::memcmp(src3, dst3, kBlockSize), 0);
}

TEST_F(CrossNodeSmoke, FinalizeIdempotentE2E) {
  std::memset(caller_buf_, 0xCD, kBlockSize);
  ASSERT_TRUE(client_a_->PutRemote("idem-key", caller_buf_, kBlockSize));
  EXPECT_TRUE(client_a_->ExistsRemote("idem-key"));

  ASSERT_TRUE(client_a_->PutRemote("idem-key", caller_buf_, kBlockSize));
  EXPECT_TRUE(client_b_->ExistsRemote("idem-key"));

  std::memset(read_buf_, 0, kBlockSize);
  ASSERT_TRUE(client_b_->GetRemote("idem-key", read_buf_, kBlockSize));
  EXPECT_EQ(std::memcmp(caller_buf_, read_buf_, kBlockSize), 0);
}

TEST_F(CrossNodeSmoke, DepthPropagation) {
  auto* src1 = static_cast<char*>(caller_buf_);
  auto* src2 = src1 + kBlockSize;
  auto* src3 = src2 + kBlockSize;
  std::memset(src1, 0x41, kBlockSize);
  std::memset(src2, 0x42, kBlockSize);
  std::memset(src3, 0x43, kBlockSize);

  std::vector<std::string> keys = {"d1", "d2", "d3"};
  std::vector<const void*> srcs = {src1, src2, src3};
  std::vector<size_t> sizes = {kBlockSize, kBlockSize, kBlockSize};
  std::vector<int> depths = {5, 10, 15};

  auto results = client_a_->BatchPutRemote(keys, srcs, sizes, depths);
  ASSERT_EQ(results.size(), 3u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(results[i]) << "put with depth failed for " << keys[i];
    EXPECT_TRUE(client_a_->ExistsRemote(keys[i]));
  }

  auto* dst1 = static_cast<char*>(read_buf_);
  std::memset(dst1, 0, kBlockSize);
  ASSERT_TRUE(client_b_->GetRemote("d1", dst1, kBlockSize));
  EXPECT_EQ(std::memcmp(src1, dst1, kBlockSize), 0);
}

}  // namespace
}  // namespace mori::umbp
