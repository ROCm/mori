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

// A remote read of a DRAM key takes a read lease on the owner, and the lease is
// a timer.  The reader reports back when its copy is done (ValidateReads with
// `release`), so the owner need not wait the timer out: the key must be
// evictable as soon as the reader's Get returns (#749).
//
// node-a owns the keys, node-b reads them over the fabric, and the test plays
// the master: it asks node-a's peer service to evict.  The DRAM read lease is
// 60 s here, so a key that is freed was freed by the report, not by the timer.
// Needs a fabric, like test_cross_node_smoke.
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

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

#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/master/master_server.h"
#include "umbp/distributed/pool_client.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {
namespace {

constexpr size_t kPage = 4096;
constexpr size_t kPages = 64;
constexpr size_t kKeys = 8;

// Both are read once per process, so they are set before the first test.  A
// value already in the environment wins, which is how the lease is shortened to
// time the old behaviour by hand.
class LongLeaseEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    setenv("UMBP_DRAM_READ_LEASE_MS", "60000", /*overwrite=*/0);
    // A node's puts stay on that node, so the owner's keys live on the owner.
    setenv("UMBP_ROUTE_PUT_NODE_AFFINITY", "local", /*overwrite=*/1);
  }
};
const ::testing::Environment* const kLongLease =
    ::testing::AddGlobalTestEnvironment(new LongLeaseEnvironment);

class RemoteReadRelease : public ::testing::Test {
 protected:
  void SetUp() override {
    write_buf_ = static_cast<char*>(std::aligned_alloc(kPage, kKeys * kPage));
    read_buf_ = static_cast<char*>(std::aligned_alloc(kPage, kKeys * kPage));

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
    reader_ = std::make_unique<PoolClient>(NodeConfig("node-b"));
    ASSERT_TRUE(reader_->Init());
    owner_->RegisterMemory(write_buf_, kKeys * kPage);
    reader_->RegisterMemory(read_buf_, kKeys * kPage);
  }

  void TearDown() override {
    if (reader_) reader_->Shutdown();
    if (owner_) owner_->Shutdown();
    if (master_) master_->Shutdown();
    if (server_thread_.joinable()) server_thread_.join();
    std::free(write_buf_);
    std::free(read_buf_);
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
    // Nothing here fills a tier; the test is the only one that evicts.
    cfg.eviction_mode = PeerEvictionMode::kMaster;
    return cfg;
  }

  // Puts `count` keys on the owner and waits until the master routes every one
  // of them to it, so the reads below are remote.  Opens a stub to the owner.
  void OwnerPuts(const std::string& prefix, size_t count, std::vector<std::string>* keys) {
    for (size_t i = 0; i < count; ++i) {
      keys->push_back(prefix + std::to_string(i));
      std::memset(write_buf_ + i * kPage, static_cast<int>(0x10 + i), kPage);
      ASSERT_TRUE(owner_->Put(keys->back(), write_buf_ + i * kPage, kPage));
    }
    UMBPMasterClientConfig cfg;
    cfg.node_id = "probe";
    cfg.node_address = "127.0.0.1";
    cfg.master_address = master_addr_;
    MasterClient probe(cfg);
    std::string peer_address;
    for (const auto& key : *keys) {
      std::optional<RouteGetResult> route;
      for (int i = 0; i < 400; ++i) {  // the master learns a key on a heartbeat
        if (probe.RouteGet(key, {}, &route).ok() && route.has_value()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
      }
      ASSERT_TRUE(route.has_value()) << key;
      ASSERT_EQ(route->node_id, "node-a") << "the read of " << key << " must be remote";
      peer_address = route->peer_address;
    }
    owner_peer_ = ::umbp::UMBPPeer::NewStub(
        grpc::CreateChannel(peer_address, grpc::InsecureChannelCredentials()));
  }

  // node-b reads every key from node-a and checks the bytes.
  void ReaderGets(const std::vector<std::string>& keys) {
    std::memset(read_buf_, 0, kKeys * kPage);
    std::vector<void*> dsts;
    std::vector<size_t> sizes(keys.size(), kPage);
    for (size_t i = 0; i < keys.size(); ++i) dsts.push_back(read_buf_ + i * kPage);
    const auto got = reader_->BatchGet(keys, dsts, sizes);
    ASSERT_EQ(got.size(), keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      ASSERT_TRUE(got[i]) << keys[i];
      ASSERT_EQ(std::memcmp(read_buf_ + i * kPage, write_buf_ + i * kPage, kPage), 0) << keys[i];
    }
  }

  // A reader that resolves on the owner and never reports back: an older
  // build, or one that died before its copy finished.
  void SilentReaderResolves(const std::vector<std::string>& keys) {
    ::umbp::BatchResolveKeysRequest req;
    for (const auto& key : keys) req.add_keys(key);
    ::umbp::BatchResolveKeysResponse resp;
    grpc::ClientContext ctx;
    ASSERT_TRUE(owner_peer_->BatchResolveKeys(&ctx, req, &resp).ok());
  }

  // One eviction round from the master: bytes freed per key, 0 for a key that
  // is still leased.
  std::vector<uint64_t> OwnerEvicts(const std::vector<std::string>& keys) {
    ::umbp::EvictKeyRequest req;
    for (const auto& key : keys) req.add_keys(key);
    ::umbp::EvictKeyResponse resp;
    grpc::ClientContext ctx;
    std::vector<uint64_t> freed(keys.size(), 0);
    if (!owner_peer_->EvictKey(&ctx, req, &resp).ok()) return freed;
    for (int i = 0; i < resp.evicted_size() && i < static_cast<int>(keys.size()); ++i) {
      freed[i] = resp.evicted(i).bytes_freed();
    }
    return freed;
  }

  char* write_buf_ = nullptr;
  char* read_buf_ = nullptr;
  std::string master_addr_;
  std::unique_ptr<MasterServer> master_;
  std::thread server_thread_;
  std::unique_ptr<PoolClient> owner_;
  std::unique_ptr<PoolClient> reader_;
  std::unique_ptr<::umbp::UMBPPeer::Stub> owner_peer_;
};

TEST_F(RemoteReadRelease, AKeyIsEvictableAsSoonAsItsRemoteReadReturns) {
  std::vector<std::string> keys;
  OwnerPuts("read-", kKeys, &keys);
  ReaderGets(keys);
  const auto read_done = std::chrono::steady_clock::now();

  std::vector<uint64_t> freed = OwnerEvicts(keys);
  size_t held = 0;
  for (uint64_t bytes : freed) held += bytes == 0 ? 1 : 0;
  if (held == 0) return;

  // Failed.  Say how long the keys stayed held, which is the lease's length
  // when only the timer frees them.
  const auto give_up = read_done + std::chrono::seconds{3};
  while (held != 0 && std::chrono::steady_clock::now() < give_up) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    const auto again = OwnerEvicts(keys);
    held = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (freed[i] == 0) freed[i] = again[i];
      held += freed[i] == 0 ? 1 : 0;
    }
  }
  const auto waited_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - read_done)
                             .count();
  if (held != 0) {
    FAIL() << held << " of " << keys.size() << " keys were still unevictable " << waited_ms
           << " ms after the remote read returned";
  }
  FAIL() << "keys became evictable only " << waited_ms << " ms after the remote read returned";
}

// The control for the test above: the lease is real in this setup, and it is
// the reader's report that ends it.  A reader that never reports keeps it.
TEST_F(RemoteReadRelease, AReaderThatNeverReportsKeepsItsLease) {
  std::vector<std::string> keys;
  OwnerPuts("silent-", 2, &keys);
  SilentReaderResolves(keys);

  for (uint64_t bytes : OwnerEvicts(keys)) EXPECT_EQ(bytes, 0u);
}

// A report counts one reader out, not the key: another reader may still be
// copying the same pages.
TEST_F(RemoteReadRelease, OneReadersReportDoesNotEndAnothersLease) {
  std::vector<std::string> keys;
  OwnerPuts("shared-", 2, &keys);
  SilentReaderResolves(keys);
  ReaderGets(keys);

  for (uint64_t bytes : OwnerEvicts(keys)) EXPECT_EQ(bytes, 0u);
}

}  // namespace
}  // namespace mori::umbp
