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

// Eviction modes end to end: a real master and a DRAM PoolClient over gRPC.
// The master's own rounds run every 5 s, so anything evicted well inside that
// window was evicted by the node itself.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"

namespace mori::umbp {
namespace {

constexpr size_t kPage = 4096;
constexpr size_t kPages = 16;

std::string Key(int i) { return "pde-" + std::to_string(i); }

bool WaitFor(const std::function<bool()>& done, std::chrono::milliseconds budget) {
  const auto deadline = std::chrono::steady_clock::now() + budget;
  while (std::chrono::steady_clock::now() < deadline) {
    if (done()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return done();
}

class PeerDrivenEvictionTest : public ::testing::Test {
 protected:
  void TearDown() override { Stop(); }

  void Stop() {
    if (client_) client_->Shutdown();
    client_.reset();
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
    master_.reset();
  }

  // A master (advertising peer-driven eviction or not) and one 16-page DRAM
  // node with 0.9 / 0.5 watermarks.
  PoolClient* Start(bool master_supports, PeerEvictionMode mode) {
    MasterServerConfig master_cfg;
    master_cfg.listen_address = "0.0.0.0:0";
    master_cfg.registry_config.advertise_peer_driven_eviction = master_supports;
    master_ = std::make_unique<MasterServer>(std::move(master_cfg));
    server_thread_ = std::thread([this] { master_->Run(); });
    if (!WaitFor([&] { return master_->GetBoundPort() != 0; }, std::chrono::seconds(5))) {
      return nullptr;
    }

    PoolClientConfig cfg;
    cfg.master_config.node_id = "peer-driven";
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = "localhost:" + std::to_string(master_->GetBoundPort());
    cfg.io_engine.host = "0.0.0.0";
    cfg.io_engine.port = 0;
    cfg.auto_peer_service_port = true;
    cfg.dram_page_size = kPage;
    cfg.medium = TierType::DRAM;
    cfg.dram.buffer_sizes = {kPages * kPage};
    cfg.cache_remote_fetches = false;
    cfg.local_evict_high_watermark = 0.9;
    cfg.local_evict_low_watermark = 0.5;
    cfg.eviction_mode = mode;
    client_ = std::make_unique<PoolClient>(std::move(cfg));
    if (!client_->Init()) client_.reset();
    return client_.get();
  }

  void PutKeys(int first, int count) {
    const std::vector<char> value(kPage, 'x');
    for (int i = first; i < first + count; ++i) {
      ASSERT_TRUE(client_->Put(Key(i), value.data(), value.size())) << i;
    }
  }

  MediumBackend* Dram() { return client_->Backends().Get(TierType::DRAM); }

  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> client_;
};

TEST_F(PeerDrivenEvictionTest, MasterRankedNodeEvictsTheMastersVictims) {
  auto* node = Start(/*master_supports=*/true, PeerEvictionMode::kMasterRanked);
  ASSERT_NE(node, nullptr);
  ASSERT_TRUE(node->Master().PeerDrivenEviction());

  // Half full, and known to the master, before crossing the watermark.
  PutKeys(0, 8);
  std::vector<std::string> known;
  for (int i = 0; i < 8; ++i) known.push_back(Key(i));
  node->Master().FlushHeartbeat();
  ASSERT_TRUE(WaitFor(
      [&] {
        std::vector<bool> found;
        return node->Master().BatchLookup(known, &found).ok() &&
               std::all_of(found.begin(), found.end(), [](bool f) { return f; });
      },
      std::chrono::seconds(5)));

  PutKeys(8, 7);  // 15 of 16 pages
  ASSERT_TRUE(
      WaitFor([&] { return Dram()->OwnedKeyCount() <= kPages / 2; }, std::chrono::seconds(3)))
      << "the node did not evict for itself";
  EXPECT_EQ(Dram()->OwnedKeyCount(), kPages / 2);
  EXPECT_EQ(node->LocalEvictionStats().master_keys, 7u) << "victims did not come from the master";
  for (int i = 8; i < 15; ++i) EXPECT_TRUE(Dram()->Contains(Key(i))) << i;
}

TEST_F(PeerDrivenEvictionTest, EvictsLocallyUnlessTheModeIsMaster) {
  struct Case {
    bool master_supports;
    PeerEvictionMode mode;
    bool self_evicts;
  };
  // The default; master_ranked against an older master (falls back to local
  // LRU); and master mode, which leaves DRAM to the master's rounds.
  const std::vector<Case> cases = {{true, PeerEvictionMode::kLocal, true},
                                   {false, PeerEvictionMode::kMasterRanked, true},
                                   {true, PeerEvictionMode::kMaster, false}};
  for (const auto& c : cases) {
    SCOPED_TRACE(::testing::Message() << "master_supports=" << c.master_supports
                                      << " mode=" << static_cast<int>(c.mode));
    auto* node = Start(c.master_supports, c.mode);
    ASSERT_NE(node, nullptr);
    PutKeys(0, 15);  // 15 of 16 pages
    if (c.self_evicts) {
      EXPECT_TRUE(
          WaitFor([&] { return Dram()->OwnedKeyCount() <= kPages / 2; }, std::chrono::seconds(3)));
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1500));
      EXPECT_EQ(Dram()->OwnedKeyCount(), 15u);
    }
    EXPECT_EQ(node->LocalEvictionStats().master_keys, 0u) << "asked the master";
    Stop();
  }
}

}  // namespace
}  // namespace mori::umbp
