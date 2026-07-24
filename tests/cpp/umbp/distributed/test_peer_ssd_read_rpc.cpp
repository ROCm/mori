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
//
// RPC-level integration test for the SSD read path: a real PeerServiceServer
// (backed by a real PeerSsdManager / POSIX SSDTier) served over a gRPC loopback
// channel.  It exercises prepare -> read-from-staging -> release / TTL and,
// crucially, asserts that OK / NOT_FOUND / NO_SLOT / SIZE_TOO_LARGE are each
// reported as distinct statuses so a transient failure is never collapsed into
// a NOT_FOUND miss.  RDMA is intentionally out of scope here (the staging buffer
// is read directly); the full BatchGet -> RDMA path needs a live cluster.
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/distributed/config.h"
#include "umbp/distributed/peer/peer_service.h"
#include "umbp/distributed/peer/peer_ssd_manager.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {
namespace {

namespace fs = std::filesystem;

constexpr size_t kStagingSize = 4096;
constexpr int kNumReadSlots = 4;  // -> 1024 B per slot
constexpr int kLeaseTimeoutS = 2;

uint16_t AllocPort() {
  static std::atomic<uint16_t> next{51300};
  return next.fetch_add(1);
}

class PeerSsdReadRpcTest : public ::testing::Test {
 protected:
  void SetUp() override {
    staging_buffer_ = std::malloc(kStagingSize);
    ASSERT_NE(staging_buffer_, nullptr);
    std::memset(staging_buffer_, 0, kStagingSize);

    dir_ = fs::temp_directory_path() /
           ("umbp_ssd_rpc_" + std::to_string(::getpid()) + "_" + std::to_string(AllocPort()));
    fs::remove_all(dir_);

    PeerSsdConfig cfg;
    cfg.enabled = true;
    cfg.ssd.enabled = true;
    cfg.ssd.storage_dir = dir_.string();
    cfg.ssd.capacity_bytes = 1 << 20;
    cfg.ssd.io.backend = UMBPIoBackend::Posix;  // avoid io_uring container flakiness
    peer_ssd_ = std::make_unique<PeerSsdManager>(cfg);

    // Fake staging MemoryDesc bytes — GetPeerInfo just echoes them; this test
    // reads the staging buffer directly rather than RDMA-ing it.
    staging_desc_ = {0xAB, 0xCD};

    port_ = AllocPort();
    server_ = std::make_unique<PeerServiceServer>(
        /*dram_alloc=*/nullptr, peer_ssd_.get(), staging_buffer_, kStagingSize, staging_desc_,
        kNumReadSlots, std::chrono::seconds(kLeaseTimeoutS));
    ASSERT_TRUE(server_->Start(port_));
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    auto channel = grpc::CreateChannel("localhost:" + std::to_string(port_),
                                       grpc::InsecureChannelCredentials());
    stub_ = ::umbp::UMBPPeer::NewStub(channel);
  }

  void TearDown() override {
    server_->Stop();
    server_.reset();
    peer_ssd_.reset();
    std::free(staging_buffer_);
    fs::remove_all(dir_);
  }

  void WriteSsd(const std::string& key, const std::string& data) {
    ASSERT_TRUE(peer_ssd_->Write(key, {{data.data(), data.size()}}, data.size()));
  }

  ::umbp::PrepareSsdReadResponse Prepare(const std::string& key, uint64_t max_size) {
    ::umbp::PrepareSsdReadRequest req;
    req.set_key(key);
    req.set_max_size(max_size);
    ::umbp::PrepareSsdReadResponse resp;
    grpc::ClientContext ctx;
    EXPECT_TRUE(stub_->PrepareSsdRead(&ctx, req, &resp).ok());
    return resp;
  }

  void* staging_buffer_ = nullptr;
  fs::path dir_;
  std::vector<uint8_t> staging_desc_;
  uint16_t port_ = 0;
  std::unique_ptr<PeerSsdManager> peer_ssd_;
  std::unique_ptr<PeerServiceServer> server_;
  std::unique_ptr<::umbp::UMBPPeer::Stub> stub_;
};

TEST_F(PeerSsdReadRpcTest, OkReadsBytesIntoStaging) {
  const std::string data = "ssd-read-rpc-ok";
  WriteSsd("k-ok", data);

  auto resp = Prepare("k-ok", data.size());
  ASSERT_EQ(resp.status(), ::umbp::SSD_READ_OK);
  EXPECT_EQ(resp.size(), data.size());
  EXPECT_LT(resp.staging_offset(), kStagingSize);
  EXPECT_GT(resp.lease_id(), 0u);
  std::string loaded(static_cast<const char*>(staging_buffer_) + resp.staging_offset(),
                     resp.size());
  EXPECT_EQ(loaded, data);
}

TEST_F(PeerSsdReadRpcTest, NotFoundIsADistinctMiss) {
  auto resp = Prepare("absent", 64);
  EXPECT_EQ(resp.status(), ::umbp::SSD_READ_NOT_FOUND);
}

TEST_F(PeerSsdReadRpcTest, SizeTooLargeIsDistinct) {
  // A key bigger than one slot (1024 B) must report SIZE_TOO_LARGE, not OK and
  // not NOT_FOUND.
  const std::string big(2048, 'q');
  WriteSsd("k-big", big);
  auto resp = Prepare("k-big", kStagingSize);
  EXPECT_EQ(resp.status(), ::umbp::SSD_READ_SIZE_TOO_LARGE);
}

// The key assertion the review asked for: slot exhaustion is NO_SLOT
// (retryable), never collapsed into NOT_FOUND.  A present key and an absent key
// under exhaustion are distinguishable.
TEST_F(PeerSsdReadRpcTest, NoSlotIsDistinctFromNotFound) {
  std::vector<::umbp::PrepareSsdReadResponse> held;
  for (int i = 0; i < kNumReadSlots; ++i) {
    const std::string key = "hold-" + std::to_string(i);
    WriteSsd(key, "payload");
    auto resp = Prepare(key, 64);
    ASSERT_EQ(resp.status(), ::umbp::SSD_READ_OK);
    held.push_back(resp);  // keep the lease held (no release) so slots stay busy
  }

  // A present key with all slots busy -> NO_SLOT (retryable), NOT a miss.
  WriteSsd("present-extra", "payload");
  EXPECT_EQ(Prepare("present-extra", 64).status(), ::umbp::SSD_READ_NO_SLOT);

  // An absent key under the same exhaustion is still NO_SLOT (slot check
  // precedes the key lookup), so the caller cannot mistake exhaustion for a
  // definitive miss.
  EXPECT_EQ(Prepare("absent-extra", 64).status(), ::umbp::SSD_READ_NO_SLOT);
}

// Many concurrent readers contend for a fixed pool of staging slots: at most
// kNumReadSlots win OK, the rest get NO_SLOT (a retryable transient), and NEVER
// NOT_FOUND for a present key.  Also exercises the staging observability
// (slot_full_rejects counter + the in-use gauge accessor).
TEST_F(PeerSsdReadRpcTest, ConcurrentReadersExhaustSlotsWithoutFalseMiss) {
  const std::string data = "concurrent-payload";
  for (int i = 0; i < 32; ++i) WriteSsd("ck-" + std::to_string(i), data);

  const uint64_t slot_full_before = server_->Metrics().slot_full_rejects.load();

  constexpr int kReaders = 24;  // >> kNumReadSlots, leases held (never released)
  std::atomic<int> ok{0}, no_slot{0}, other{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < kReaders; ++i) {
    threads.emplace_back([&, i] {
      auto resp = Prepare("ck-" + std::to_string(i), data.size());
      switch (resp.status()) {
        case ::umbp::SSD_READ_OK:
          ok.fetch_add(1);
          break;
        case ::umbp::SSD_READ_NO_SLOT:
          no_slot.fetch_add(1);
          break;
        default:
          other.fetch_add(1);  // NOT_FOUND / SIZE_TOO_LARGE / ERROR must never happen here
          break;
      }
    });
  }
  for (auto& t : threads) t.join();

  EXPECT_EQ(other.load(), 0) << "present keys never report a false miss under contention";
  EXPECT_LE(ok.load(), kNumReadSlots) << "at most one OK per staging slot";
  EXPECT_EQ(ok.load() + no_slot.load(), kReaders);
  EXPECT_GT(no_slot.load(), 0) << "with more readers than slots, some must see NO_SLOT";

  // The NO_SLOT rejections were counted, and the gauge sees the held leases.
  EXPECT_GE(server_->Metrics().slot_full_rejects.load() - slot_full_before,
            static_cast<uint64_t>(no_slot.load()));
  EXPECT_EQ(server_->SnapshotReadSlotsInUse(), static_cast<size_t>(ok.load()));
}

// A best-effort release frees the slot; double release reports false.
TEST_F(PeerSsdReadRpcTest, ReleaseFreesSlotAndIsBestEffort) {
  WriteSsd("k-rel", "payload");
  auto resp = Prepare("k-rel", 64);
  ASSERT_EQ(resp.status(), ::umbp::SSD_READ_OK);

  ::umbp::ReleaseSsdLeaseRequest rel;
  rel.set_lease_id(resp.lease_id());
  ::umbp::ReleaseSsdLeaseResponse rel_resp;
  grpc::ClientContext ctx;
  ASSERT_TRUE(stub_->ReleaseSsdLease(&ctx, rel, &rel_resp).ok());
  EXPECT_TRUE(rel_resp.success());

  ::umbp::ReleaseSsdLeaseResponse rel_resp2;
  grpc::ClientContext ctx2;
  ASSERT_TRUE(stub_->ReleaseSsdLease(&ctx2, rel, &rel_resp2).ok());
  EXPECT_FALSE(rel_resp2.success()) << "double release is a no-op";
}

// Leased slots are reclaimed by TTL even without a release, so a fresh prepare
// succeeds after the lease elapses (slot lifecycle: Leased -> reclaimed).
TEST_F(PeerSsdReadRpcTest, LeasedSlotsReclaimedByTtl) {
  for (int i = 0; i < kNumReadSlots; ++i) {
    const std::string key = "ttl-" + std::to_string(i);
    WriteSsd(key, "payload");
    ASSERT_EQ(Prepare(key, 64).status(), ::umbp::SSD_READ_OK);  // never released
  }
  EXPECT_EQ(Prepare("ttl-0", 64).status(), ::umbp::SSD_READ_NO_SLOT);  // all busy

  std::this_thread::sleep_for(std::chrono::seconds(kLeaseTimeoutS + 1));

  WriteSsd("ttl-after", "payload");
  EXPECT_EQ(Prepare("ttl-after", 64).status(), ::umbp::SSD_READ_OK) << "TTL should reclaim a slot";
}

}  // namespace
}  // namespace mori::umbp
