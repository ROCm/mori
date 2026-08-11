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

// ONE MEDIUM PER NODE.
//
// PoolClient::Init used to register DRAM unconditionally and then add HBM/SSD
// beside it, which made "--tier ssd" mean "DRAM *and* SSD".  That is not a tier
// stack: the routing plane treats every advertised tier as an equally valid put
// target (Phase 4 deleted the hardcoded tier orders), so the second medium
// mirrors the first rather than sitting behind it.  UMBPDistributedConfig::
// medium now selects exactly one.
//
// Covered here:
//   1. The lowering  — UMBPMedium -> PoolClientConfig::medium, and the SSD
//      opt-in coming from the selector rather than UMBPConfig::ssd.enabled.
//   2. Validation    — the selected medium's sizing is checked, the others'
//                      are not.
//   3. The registry  — a live PoolClient holds exactly one backend, and an
//      SSD node has NO DRAM tier to fall back to.
//   4. The wire      — a cross-node BatchPut lands on a peer whose only medium
//      is SSD.  PartitionBatchPutTargets used to drop any route that was not
//      DRAM/HBM, so this is the case that silently failed.

#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"

namespace mori::umbp {
namespace {

namespace fs = std::filesystem;

constexpr size_t kPageSize = 4096;
constexpr size_t kDramCap = 8 << 20;

// ---------------------------------------------------------------------------
//  1. Lowering
// ---------------------------------------------------------------------------

UMBPDistributedConfig MinimalDistributedConfig() {
  UMBPDistributedConfig dc;
  dc.master_config.master_address = "localhost:1";
  dc.master_config.node_id = "n0";
  dc.master_config.node_address = "127.0.0.1";
  return dc;
}

TEST(MediumLoweringTest, DefaultsToDram) {
  auto pc = ToPoolClientConfig(MinimalDistributedConfig(), DramOwnershipConfig{}, PeerSsdConfig{});
  EXPECT_EQ(pc.medium, TierType::DRAM);
  // The SSD tier stays shut unless the selector names it, whatever the SSD
  // block says — UMBPSsdConfig::enabled defaults to true and describes the
  // LOCAL-mode tier, so keying off it would opt every deployment in.
  PeerSsdConfig ssd;
  ssd.enabled = true;
  ssd.ssd.enabled = true;
  auto pc2 = ToPoolClientConfig(MinimalDistributedConfig(), DramOwnershipConfig{}, ssd);
  EXPECT_EQ(pc2.medium, TierType::DRAM);
  EXPECT_FALSE(pc2.ssd.enabled);
}

TEST(MediumLoweringTest, SelectorPicksTierAndOptsSsdIn) {
  auto dc = MinimalDistributedConfig();

  dc.medium = UMBPMedium::HBM;
  dc.hbm.device = 3;
  dc.hbm.capacity_bytes = 1 << 20;
  auto hbm = ToPoolClientConfig(dc, DramOwnershipConfig{}, PeerSsdConfig{});
  EXPECT_EQ(hbm.medium, TierType::HBM);
  EXPECT_EQ(hbm.hbm.device, 3);
  ASSERT_EQ(hbm.hbm.buffer_sizes.size(), 1u);
  EXPECT_EQ(hbm.hbm.buffer_sizes[0], 1u << 20);
  EXPECT_FALSE(hbm.ssd.enabled);

  dc.medium = UMBPMedium::SSD;
  auto ssd = ToPoolClientConfig(dc, DramOwnershipConfig{}, PeerSsdConfig{});
  EXPECT_EQ(ssd.medium, TierType::SSD);
  EXPECT_TRUE(ssd.ssd.enabled) << "selecting SSD is what opts the tier in";
}

// ---------------------------------------------------------------------------
//  2. Validation — only the selected medium is sized-checked
// ---------------------------------------------------------------------------

UMBPConfig MinimalUmbpConfig() {
  UMBPConfig cfg;
  cfg.distributed = MinimalDistributedConfig();
  return cfg;
}

TEST(MediumValidationTest, UnselectedMediaAreNotChecked) {
  auto cfg = MinimalUmbpConfig();
  // hbm.capacity_bytes is 0 here, and that is fine: this is a DRAM node.
  std::string err;
  EXPECT_TRUE(cfg.Validate(&err)) << err;
}

TEST(MediumValidationTest, SelectedHbmNeedsCapacity) {
  auto cfg = MinimalUmbpConfig();
  cfg.distributed->medium = UMBPMedium::HBM;
  std::string err;
  EXPECT_FALSE(cfg.Validate(&err));
  EXPECT_NE(err.find("distributed.hbm.capacity_bytes"), std::string::npos) << err;

  cfg.distributed->hbm.capacity_bytes = 1 << 20;
  EXPECT_TRUE(cfg.Validate(&err)) << err;
}

TEST(MediumValidationTest, SelectedSsdNeedsCapacityEvenWhenTierFlagIsOff) {
  auto cfg = MinimalUmbpConfig();
  cfg.distributed->medium = UMBPMedium::SSD;
  cfg.ssd.enabled = false;  // must NOT short-circuit the sizing check
  cfg.ssd.capacity_bytes = 0;
  std::string err;
  EXPECT_FALSE(cfg.Validate(&err));
  EXPECT_NE(err.find("ssd.capacity_bytes"), std::string::npos) << err;
}

// ---------------------------------------------------------------------------
//  3 + 4. Live PoolClient: one registered backend, and a remote put onto it
// ---------------------------------------------------------------------------

// Free ephemeral port; PoolClient binds and advertises this verbatim, so a
// hardcoded base would collide with concurrent test processes.
uint16_t NextPeerServicePort() {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd >= 0) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = 0;
    socklen_t len = sizeof(addr);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0 &&
        ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0) {
      uint16_t port = ntohs(addr.sin_port);
      ::close(fd);
      return port;
    }
    ::close(fd);
  }
  static std::atomic<uint16_t> next{
      static_cast<uint16_t>(54000 + (static_cast<unsigned>(::getpid()) % 4000))};
  return next.fetch_add(1);
}

class MediumSelectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<uint64_t> counter{0};
    ssd_dir_ = fs::temp_directory_path() / ("umbp_medium_sel_" + std::to_string(::getpid()) + "_" +
                                            std::to_string(counter.fetch_add(1)));
    fs::remove_all(ssd_dir_);

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
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
    std::error_code ec;
    fs::remove_all(ssd_dir_, ec);
  }

  PoolClientConfig BaseConfig(const std::string& node_id) {
    PoolClientConfig cfg;
    cfg.master_config.node_id = node_id;
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = master_addr_;
    cfg.io_engine.host = "0.0.0.0";
    cfg.io_engine.port = 0;
    cfg.peer_service_port = NextPeerServicePort();
    cfg.dram_page_size = kPageSize;
    // Off so a remote read stays remote and the assertions below describe the
    // medium under test, not a re-cached copy on the reader.
    cfg.cache_remote_fetches = false;
    return cfg;
  }

  PoolClientConfig DramConfig(const std::string& node_id, size_t capacity) {
    auto cfg = BaseConfig(node_id);
    cfg.medium = TierType::DRAM;
    cfg.dram.buffer_sizes = {capacity};
    return cfg;
  }

  PoolClientConfig SsdConfig(const std::string& node_id) {
    auto cfg = BaseConfig(node_id);
    cfg.medium = TierType::SSD;
    // DRAM sizing is still present and must be ignored — a node that selected
    // SSD gets no host pool, whatever this says.
    cfg.dram.buffer_sizes = {kDramCap};
    cfg.ssd_staging_buffer_slots = 8;
    cfg.ssd.ssd.storage_dir = ssd_dir_.string();
    cfg.ssd.ssd.capacity_bytes = 64ULL * 1024 * 1024;
    cfg.ssd.ssd.io.backend = UMBPIoBackend::Posix;  // avoid io_uring container flakiness
    return cfg;
  }

  PoolClient* Start(PoolClientConfig cfg) {
    auto client = std::make_unique<PoolClient>(std::move(cfg));
    if (!client->Init()) return nullptr;
    clients_.push_back(std::move(client));
    return clients_.back().get();
  }

  fs::path ssd_dir_;
  std::string master_addr_;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::vector<std::unique_ptr<PoolClient>> clients_;
};

TEST_F(MediumSelectionTest, DramNodeRegistersOnlyDram) {
  auto* node = Start(DramConfig("node-dram", kDramCap));
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->Medium(), TierType::DRAM);
  EXPECT_EQ(node->Backends().All().size(), 1u);
  EXPECT_NE(node->Backends().Get(TierType::DRAM), nullptr);
  EXPECT_EQ(node->Backends().Get(TierType::SSD), nullptr);
  EXPECT_EQ(node->Backends().Get(TierType::HBM), nullptr);
}

TEST_F(MediumSelectionTest, SsdNodeHasNoDramTier) {
  auto* node = Start(SsdConfig("node-ssd"));
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->Medium(), TierType::SSD);
  ASSERT_EQ(node->Backends().All().size(), 1u) << "an SSD node must not also carry a DRAM pool";
  EXPECT_EQ(node->Backends().All()[0]->Tier(), TierType::SSD);
  EXPECT_EQ(node->Backends().Get(TierType::DRAM), nullptr);
}

TEST_F(MediumSelectionTest, SsdOnlyNodeRoundTripsLocally) {
  auto* node = Start(SsdConfig("node-ssd-local"));
  ASSERT_NE(node, nullptr);

  std::vector<char> payload(kPageSize, 0x5A);
  ASSERT_TRUE(node->Put("ssd-local-key", payload.data(), payload.size()));

  std::vector<char> readback(kPageSize, 0);
  ASSERT_TRUE(node->Get("ssd-local-key", readback.data(), readback.size()));
  EXPECT_EQ(std::memcmp(payload.data(), readback.data(), payload.size()), 0);
}

// The regression that motivated the tier filter's removal: master routes the
// batch to the only node with capacity, whose only medium is SSD.  The old
// DRAM/HBM allowlist in PartitionBatchPutTargets dropped exactly these items,
// so every key came back false with nothing on the wire.
TEST_F(MediumSelectionTest, CrossNodeBatchPutLandsOnSsdPeer) {
  // One page of DRAM ~= no capacity, so the batch must route to the peer.
  auto* caller = Start(DramConfig("node-caller", kPageSize));
  ASSERT_NE(caller, nullptr);
  auto* target = Start(SsdConfig("node-ssd-target"));
  ASSERT_NE(target, nullptr);

  constexpr size_t kKeys = 4;
  std::vector<std::string> keys;
  std::vector<std::vector<char>> buffers;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  for (size_t i = 0; i < kKeys; ++i) {
    keys.push_back("ssd-remote-" + std::to_string(i));
    buffers.emplace_back(kPageSize, static_cast<char>(0x20 + i));
    sizes.push_back(kPageSize);
  }
  for (auto& b : buffers) srcs.push_back(b.data());

  auto put_results = caller->BatchPut(keys, srcs, sizes);
  ASSERT_EQ(put_results.size(), kKeys);
  for (size_t i = 0; i < kKeys; ++i) {
    EXPECT_TRUE(put_results[i]) << "key " << keys[i] << " was not placed on the SSD peer";
  }

  // And it really is on the peer's SSD, readable back over the wire.  Retried:
  // the peer publishes its committed keys to master through the heartbeat, so a
  // read issued immediately after the put can legitimately find no route yet.
  std::vector<std::vector<char>> reads(kKeys, std::vector<char>(kPageSize, 0));
  std::vector<void*> dsts;
  for (auto& r : reads) dsts.push_back(r.data());
  std::vector<bool> get_results;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  do {
    get_results = caller->BatchGet(keys, dsts, sizes);
    ASSERT_EQ(get_results.size(), kKeys);
    bool all = true;
    for (size_t i = 0; i < kKeys; ++i) all = all && get_results[i];
    if (all) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  } while (std::chrono::steady_clock::now() < deadline);

  for (size_t i = 0; i < kKeys; ++i) {
    EXPECT_TRUE(get_results[i]) << "key " << keys[i] << " unreadable from the SSD peer";
    if (get_results[i]) {
      EXPECT_EQ(std::memcmp(buffers[i].data(), reads[i].data(), kPageSize), 0);
    }
  }
}

}  // namespace
}  // namespace mori::umbp
