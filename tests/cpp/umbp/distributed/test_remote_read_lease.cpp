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
// A remote read is only safe while the owning peer's read lease holds: past it
// the peer may evict the key and reuse its pages, and the read still succeeds on
// the wire with another key's bytes.  The reader must turn such a read into a
// miss.
//
// node-a owns "victim"; node-b reads it over the fabric through an engine that
// delays the submit, standing in for a network or scheduling stall, while
// node-a stays under put pressure so its own eviction frees and reuses pages.
// Needs a fabric, like test_cross_node_smoke.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/distributed/pool_platform.h"

namespace mori::umbp {
namespace {

constexpr size_t kPage = 4096;
constexpr size_t kPages = 64;
constexpr uint64_t kVictimTag = 0xFFFFFFFFFFFFFFFFull;

void Fill(void* buf, uint64_t tag) {
  auto* words = static_cast<uint64_t*>(buf);
  for (size_t i = 0; i < kPage / sizeof(uint64_t); ++i) words[i] = tag;
}

bool HoldsOnly(const void* buf, uint64_t tag) {
  const auto* words = static_cast<const uint64_t*>(buf);
  for (size_t i = 0; i < kPage / sizeof(uint64_t); ++i) {
    if (words[i] != tag) return false;
  }
  return true;
}

// Delays the next Submit by the armed amount, then is transparent again.
class StallingEngine final : public TransferEngine {
 public:
  StallingEngine(std::unique_ptr<TransferEngine> inner, std::shared_ptr<std::atomic<int>> stall_ms)
      : inner_(std::move(inner)), stall_ms_(std::move(stall_ms)) {}

  const char* Name() const override { return inner_->Name(); }
  std::vector<MetricSample> SampleMetrics() const override { return inner_->SampleMetrics(); }
  TransferRef RegisterMemory(void* base, size_t size, mori::io::MemoryLocationType loc,
                             int device) override {
    return inner_->RegisterMemory(base, size, loc, device);
  }
  TransferRef RegisterFile(int fd, uint64_t offset, uint64_t size) override {
    return inner_->RegisterFile(fd, offset, size);
  }
  void Deregister(const TransferRef& ref) override { inner_->Deregister(ref); }
  bool CanHandle(const TransferRef& src, const TransferRef& dst) const override {
    return inner_->CanHandle(src, dst);
  }
  std::unique_ptr<TransferPlanBuilder> CreatePlanBuilder(size_t expected_items) const override {
    return inner_->CreatePlanBuilder(expected_items);
  }
  std::unique_ptr<TransferHandle> Submit(std::vector<TransferPlan> plans) override {
    if (const int ms = stall_ms_->exchange(0); ms > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
    return inner_->Submit(std::move(plans));
  }

 private:
  std::unique_ptr<TransferEngine> inner_;
  std::shared_ptr<std::atomic<int>> stall_ms_;
};

// The real platform, with its transfer engine wrapped.
class StallingPlatform final : public PoolPlatform {
 public:
  explicit StallingPlatform(std::shared_ptr<std::atomic<int>> stall_ms)
      : inner_(MakePoolPlatform(UMBPSimConfig{})), stall_ms_(std::move(stall_ms)) {}

  Engines CreateTransferEngine(const PoolClientConfig& config,
                               const std::vector<BackendInstanceConfig>& backends) override {
    Engines engines = inner_->CreateTransferEngine(config, backends);
    engines.transfer = std::make_unique<StallingEngine>(std::move(engines.transfer), stall_ms_);
    return engines;
  }
  std::unique_ptr<MediumBackend> CreateBackend(const BackendInstanceConfig& backend,
                                               uint64_t page_size,
                                               const PoolClientConfig& config) override {
    return inner_->CreateBackend(backend, page_size, config);
  }
  TransferRef LocalUserBytes(void* ptr, size_t size,
                             const TransferRef& registered_ref) const override {
    return inner_->LocalUserBytes(ptr, size, registered_ref);
  }
  HostBufferHandle AllocateHost(size_t size, const HostBufferOptions& options) override {
    return inner_->AllocateHost(size, options);
  }
  void FreeHost(HostBufferHandle& buffer) override { inner_->FreeHost(buffer); }
  void AddHostGatherRegion(void* ptr, size_t size) override {
    inner_->AddHostGatherRegion(ptr, size);
  }
  void ClearHostGatherRegions() override { inner_->ClearHostGatherRegions(); }

 private:
  std::shared_ptr<PoolPlatform> inner_;
  std::shared_ptr<std::atomic<int>> stall_ms_;
};

struct Case {
  int stall_ms;   // against the default 500 ms DRAM read lease
  bool pressure;  // the owner keeps evicting and reusing pages meanwhile
  bool expect_hit;
};

class RemoteReadLease : public ::testing::TestWithParam<Case> {
 protected:
  void SetUp() override {
    // A node's puts stay on that node, so "victim" lives on the owner.
    setenv("UMBP_ROUTE_PUT_NODE_AFFINITY", "local", 1);
    write_buf_ = std::aligned_alloc(kPage, kPage);
    read_buf_ = std::aligned_alloc(kPage, kPage);

    MasterServerConfig master_cfg = MasterServerConfig::FromEnvironment();
    master_cfg.listen_address = "0.0.0.0:0";
    master_cfg.registry_config.heartbeat_ttl = std::chrono::seconds{1};
    master_ = std::make_unique<MasterServer>(std::move(master_cfg));
    server_thread_ = std::thread([this] { master_->Run(); });
    for (int i = 0; i < 100 && master_->GetBoundPort() == 0; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_NE(master_->GetBoundPort(), 0);
    master_addr_ = "localhost:" + std::to_string(master_->GetBoundPort());

    owner_ = std::make_unique<PoolClient>(NodeConfig("node-a"));
    ASSERT_TRUE(owner_->Init());
    reader_ = std::make_unique<PoolClient>(NodeConfig("node-b"),
                                           std::make_shared<StallingPlatform>(stall_ms_));
    ASSERT_TRUE(reader_->Init());
    owner_->RegisterMemory(write_buf_, kPage);
    reader_->RegisterMemory(read_buf_, kPage);
  }

  void TearDown() override {
    if (reader_) reader_->Shutdown();
    if (owner_) owner_->Shutdown();
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
    std::free(write_buf_);
    std::free(read_buf_);
    unsetenv("UMBP_ROUTE_PUT_NODE_AFFINITY");
  }

  PoolClientConfig NodeConfig(const char* node) {
    PoolClientConfig cfg;
    cfg.master_config.node_id = node;
    cfg.master_config.node_address = "127.0.0.1";
    cfg.master_config.master_address = master_addr_;
    cfg.io_engine.host = "0.0.0.0";
    cfg.io_engine.port = 0;
    cfg.auto_peer_service_port = true;
    cfg.dram.buffer_sizes = {kPages * kPage};
    cfg.dram_page_size = kPage;
    return cfg;
  }

  // The node the master routes a read of `key` to ("" when it knows none).
  std::string OwnerOf(const std::string& key) {
    UMBPMasterClientConfig cfg;
    cfg.node_id = "probe";
    cfg.node_address = "127.0.0.1";
    cfg.master_address = master_addr_;
    MasterClient probe(cfg);
    std::optional<RouteGetResult> route;
    if (!probe.RouteGet(key, {}, &route).ok() || !route.has_value()) return "";
    return route->node_id;
  }

  void* write_buf_ = nullptr;
  void* read_buf_ = nullptr;
  std::string master_addr_;
  std::shared_ptr<std::atomic<int>> stall_ms_ = std::make_shared<std::atomic<int>>(0);
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> owner_;
  std::unique_ptr<PoolClient> reader_;
};

TEST_P(RemoteReadLease, AReadPastTheLeaseIsAMissNeverAnotherKeysBytes) {
  const Case c = GetParam();

  Fill(write_buf_, kVictimTag);
  ASSERT_TRUE(owner_->Put("victim", write_buf_, kPage));
  std::string home;
  for (int i = 0; i < 200 && home.empty(); ++i) {  // the master learns it on a heartbeat
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    home = OwnerOf("victim");
  }
  ASSERT_EQ(home, "node-a") << "the read below must be a remote one";

  std::memset(read_buf_, 0, kPage);
  stall_ms_->store(c.stall_ms);
  bool got = false;
  const auto start = std::chrono::steady_clock::now();
  std::thread get_thread([&] { got = reader_->Get("victim", read_buf_, kPage); });

  if (c.pressure) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // let the resolve land first
    const auto until = start + std::chrono::milliseconds(c.stall_ms + 100);
    for (uint64_t i = 0; std::chrono::steady_clock::now() < until; ++i) {
      Fill(write_buf_, i);
      owner_->Put("fill-" + std::to_string(i), write_buf_, kPage);
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
  get_thread.join();
  ASSERT_EQ(stall_ms_->load(), 0) << "the read never reached the transfer engine";

  EXPECT_EQ(got, c.expect_hit);
  if (got) {
    EXPECT_TRUE(HoldsOnly(read_buf_, kVictimTag))
        << "a successful Get returned bytes that are not the key's";
  }
}

INSTANTIATE_TEST_SUITE_P(
    Dram, RemoteReadLease,
    ::testing::Values(Case{50, true, true},      // settles inside the lease
                      Case{900, true, false},    // past it, pages reused meanwhile
                      Case{900, false, false}),  // past it: unprovable either way, so a miss too
    [](const ::testing::TestParamInfo<Case>& info) {
      return "stall" + std::to_string(info.param.stall_ms) + "ms" +
             (info.param.pressure ? "_pressure" : "_idle");
    });

}  // namespace
}  // namespace mori::umbp
