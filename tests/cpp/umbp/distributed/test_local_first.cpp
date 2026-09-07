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

// PoolClientConfig::local_first — resolve on this node's own media before
// asking the master, and skip the master entirely when the whole batch is here.
//
// The claim under test is not "fewer RPCs" but "NO RPC", so the assertions do
// not count calls: they STOP THE MASTER and then read.  A read that still
// succeeds against a dead master provably never needed it, which is exactly the
// single-node case the flag exists for.  The `local_first=false` twin is what
// keeps that honest -- it must fail on the same data, or the test is passing
// for some reason other than the flag.

#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"

namespace mori::umbp {
namespace {

constexpr size_t kPageSize = 4096;
constexpr size_t kDramCap = 16ULL * 1024 * 1024;
constexpr size_t kObjectSize = 8192;

uint16_t NextPeerServicePort() {
  static std::atomic<uint16_t> next{
      static_cast<uint16_t>(51000 + (static_cast<unsigned>(::getpid()) % 3000))};
  return next.fetch_add(1);
}

std::vector<char> Pattern(size_t size, int seed) {
  std::vector<char> out(size);
  for (size_t i = 0; i < size; ++i) out[i] = static_cast<char>((i * 31 + seed) & 0xFF);
  return out;
}

class LocalFirstTest : public ::testing::Test {
 protected:
  void SetUp() override {
    MasterServerConfig master_cfg;
    master_cfg.listen_address = "0.0.0.0:0";
    master_ = std::make_unique<MasterServer>(std::move(master_cfg));
    server_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 50 && master_->GetBoundPort() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_NE(master_->GetBoundPort(), 0) << "Master failed to start";
    master_addr_ = "localhost:" + std::to_string(master_->GetBoundPort());
  }

  void TearDown() override {
    for (auto& c : clients_) {
      if (c) c->Shutdown();
    }
    clients_.clear();
    StopMaster();
  }

  // Idempotent: a test that stops the master mid-body must not have TearDown
  // stop it again.
  void StopMaster() {
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
    master_.reset();
  }

  PoolClientConfig DramConfig(const std::string& node_id, bool local_first) {
    PoolClientConfig cfg;
    cfg.master_config.node_id = node_id;
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = master_addr_;
    cfg.io_engine.host = "0.0.0.0";
    cfg.io_engine.port = 0;
    cfg.peer_service_port = NextPeerServicePort();
    cfg.dram_page_size = kPageSize;
    cfg.medium = TierType::DRAM;
    cfg.dram.buffer_sizes = {kDramCap};
    cfg.cache_remote_fetches = false;
    cfg.local_first = local_first;
    return cfg;
  }

  PoolClient* Start(PoolClientConfig cfg) {
    auto client = std::make_unique<PoolClient>(std::move(cfg));
    if (!client->Init()) return nullptr;
    clients_.push_back(std::move(client));
    return clients_.back().get();
  }

  std::string master_addr_;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::vector<std::unique_ptr<PoolClient>> clients_;
};

// The single-node case the flag exists for: everything this process put is
// still readable once the master is gone.
TEST_F(LocalFirstTest, FullyLocalBatchGetSurvivesAMasterOutage) {
  auto* node = Start(DramConfig("local-first-get", /*local_first=*/true));
  ASSERT_NE(node, nullptr);

  const std::vector<std::string> keys = {"lf-a", "lf-b", "lf-c"};
  std::vector<std::vector<char>> objects;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < keys.size(); ++i) {
    objects.push_back(Pattern(kObjectSize, static_cast<int>(i)));
    srcs.push_back(objects.back().data());
    sizes.push_back(kObjectSize);
  }
  ASSERT_EQ(node->BatchPut(keys, srcs, sizes), std::vector<bool>(keys.size(), true));

  StopMaster();

  std::vector<std::vector<char>> reads(keys.size(), std::vector<char>(kObjectSize, 0));
  std::vector<void*> dsts;
  for (auto& r : reads) dsts.push_back(r.data());

  EXPECT_EQ(node->BatchGet(keys, dsts, sizes), std::vector<bool>(keys.size(), true));
  for (size_t i = 0; i < keys.size(); ++i) {
    EXPECT_EQ(std::memcmp(reads[i].data(), objects[i].data(), kObjectSize), 0)
        << "wrong bytes for " << keys[i];
  }
}

// Same data, same outage, flag off.  If this passed, the test above would prove
// nothing about local_first.
TEST_F(LocalFirstTest, RouteFirstBatchGetDoesNotSurviveAMasterOutage) {
  auto* node = Start(DramConfig("route-first-get", /*local_first=*/false));
  ASSERT_NE(node, nullptr);

  const std::vector<std::string> keys = {"rf-a"};
  const auto object = Pattern(kObjectSize, 7);
  const std::vector<const void*> srcs = {object.data()};
  const std::vector<size_t> sizes = {kObjectSize};
  ASSERT_EQ(node->BatchPut(keys, srcs, sizes), std::vector<bool>({true}));

  StopMaster();

  std::vector<char> read(kObjectSize, 0);
  const std::vector<void*> dsts = {read.data()};
  EXPECT_EQ(node->BatchGet(keys, dsts, sizes), std::vector<bool>({false}))
      << "route-first must go through the master, so a dead master is a miss";
}

TEST_F(LocalFirstTest, LocallyHeldExistsSurvivesAMasterOutage) {
  auto* node = Start(DramConfig("local-first-exists", /*local_first=*/true));
  ASSERT_NE(node, nullptr);

  const std::vector<std::string> keys = {"lfe-a", "lfe-b"};
  const auto object = Pattern(kObjectSize, 3);
  const std::vector<const void*> srcs = {object.data(), object.data()};
  const std::vector<size_t> sizes = {kObjectSize, kObjectSize};
  ASSERT_EQ(node->BatchPut(keys, srcs, sizes), std::vector<bool>(keys.size(), true));

  StopMaster();

  EXPECT_EQ(node->BatchExists(keys), std::vector<bool>(keys.size(), true));
  EXPECT_TRUE(node->Exists(keys[0]));
}

// A local MISS is not conclusive, so it must still be taken to the master --
// and when the master cannot answer, the honest result is false rather than a
// fabricated hit.  This is the half of the contract that keeps local-first from
// turning "I do not have it" into "nobody has it".
TEST_F(LocalFirstTest, UnknownKeyIsNotClaimedToExist) {
  auto* node = Start(DramConfig("local-first-miss", /*local_first=*/true));
  ASSERT_NE(node, nullptr);

  const std::vector<std::string> present = {"lfm-here"};
  const auto object = Pattern(kObjectSize, 11);
  ASSERT_EQ(node->BatchPut(present, {object.data()}, {kObjectSize}), std::vector<bool>({true}));

  // Master alive: a key nobody ever put is absent, and the local hit beside it
  // is unaffected by the lookup that answers for the miss.
  const std::vector<std::string> mixed = {"lfm-here", "lfm-never-put"};
  EXPECT_EQ(node->BatchExists(mixed), std::vector<bool>({true, false}));

  std::vector<char> read(kObjectSize, 0);
  std::vector<char> unused(kObjectSize, 0);
  const std::vector<void*> dsts = {read.data(), unused.data()};
  EXPECT_EQ(node->BatchGet(mixed, dsts, {kObjectSize, kObjectSize}),
            std::vector<bool>({true, false}));
  EXPECT_EQ(std::memcmp(read.data(), object.data(), kObjectSize), 0);
}

}  // namespace
}  // namespace mori::umbp
