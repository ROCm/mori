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
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/common/env_time.h"
#include "umbp/distributed/config.h"
#include "umbp/distributed/peer/peer_dram_allocator.h"
#include "umbp/distributed/peer/peer_service.h"
#include "umbp/distributed/peer/peer_ssd_manager.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {
namespace {

constexpr size_t kStagingSize = 4096;
constexpr uint64_t kPageSize = 4096;
constexpr uint16_t kBasePort = 50200;
constexpr int kNumReadSlots = 4;
constexpr int kLeaseTimeoutS = 2;

// Ask the kernel for a currently-free ephemeral port.  PeerServiceServer binds
// this port directly, so a hard-coded base (kBasePort) collides with concurrent
// test processes / leftover servers on a shared (self-hosted CI) host and makes
// the suite flaky.
static uint16_t AllocPort() {
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
      static_cast<uint16_t>(kBasePort + (static_cast<unsigned>(::getpid()) % 4000))};
  return next.fetch_add(1);
}

class PeerServiceSlotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    staging_buffer_ = std::malloc(kStagingSize);
    ASSERT_NE(staging_buffer_, nullptr);
    std::memset(staging_buffer_, 0, kStagingSize);

    ssd_dir_ = std::filesystem::temp_directory_path() /
               ("umbp_test_ssd_" + std::to_string(getpid()) + "_" + std::to_string(AllocPort()));
    std::filesystem::create_directories(ssd_dir_);

    ssd_staging_mem_desc_ = {0xD0, 0xE0, 0xF0};

    // Peer-side SSD tier owner (file SSDTier on a temp dir; Posix I/O backend to
    // avoid io_uring availability differences inside the build container).
    PeerSsdConfig ssd_cfg;
    ssd_cfg.enabled = true;
    ssd_cfg.ssd.enabled = true;
    ssd_cfg.ssd.storage_dir = ssd_dir_.string();
    ssd_cfg.ssd.capacity_bytes = 1 << 20;
    ssd_cfg.ssd.io.backend = UMBPIoBackend::Posix;
    peer_ssd_ = std::make_unique<PeerSsdManager>(ssd_cfg);

    // Standalone DRAM allocator: the SSD-path RPCs exercised here never
    // allocate DRAM, but PeerServiceServer requires a non-null allocator.
    // An empty TierConfig (no DRAM/HBM buffers) is sufficient and avoids
    // standing up a PoolClient / connecting to a master.
    dram_alloc_ = std::make_unique<PeerDramAllocator>(kPageSize, PeerDramAllocator::TierConfig{},
                                                      PeerDramAllocator::TierConfig{},
                                                      std::chrono::milliseconds{1000});

    port_ = AllocPort();

    server_ = std::make_unique<PeerServiceServer>(
        dram_alloc_.get(), peer_ssd_.get(), staging_buffer_, kStagingSize, ssd_staging_mem_desc_,
        kNumReadSlots, std::chrono::seconds(kLeaseTimeoutS), std::vector<uint8_t>{},
        /*master_client=*/nullptr);
    server_->Start(port_);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto channel = grpc::CreateChannel("localhost:" + std::to_string(port_),
                                       grpc::InsecureChannelCredentials());
    stub_ = ::umbp::UMBPPeer::NewStub(channel);
  }

  void TearDown() override {
    server_->Stop();
    server_.reset();
    dram_alloc_.reset();
    peer_ssd_.reset();
    std::free(staging_buffer_);
    std::filesystem::remove_all(ssd_dir_);
  }

  void WriteTestDataToSsd(const std::string& key, const std::string& data) {
    peer_ssd_->Write(key, {{data.data(), data.size()}}, data.size());
  }

  void* staging_buffer_ = nullptr;
  std::filesystem::path ssd_dir_;
  std::vector<uint8_t> ssd_staging_mem_desc_;
  uint16_t port_ = 0;
  std::unique_ptr<PeerSsdManager> peer_ssd_;
  std::unique_ptr<PeerDramAllocator> dram_alloc_;
  std::unique_ptr<PeerServiceServer> server_;
  std::unique_ptr<::umbp::UMBPPeer::Stub> stub_;
};

// --- GetPeerInfo ---

TEST_F(PeerServiceSlotTest, GetPeerInfoReturnsStagingInfo) {
  ::umbp::GetPeerInfoRequest request;
  ::umbp::GetPeerInfoResponse response;
  grpc::ClientContext context;

  auto status = stub_->GetPeerInfo(&context, request, &response);
  ASSERT_TRUE(status.ok()) << status.error_message();
  EXPECT_EQ(response.ssd_staging_mem_desc(),
            std::string(ssd_staging_mem_desc_.begin(), ssd_staging_mem_desc_.end()));
  EXPECT_EQ(response.ssd_staging_size(), kStagingSize);
}

// --- PrepareSsdRead ---

TEST_F(PeerServiceSlotTest, PrepareSsdReadSuccess) {
  const std::string data = "read me from ssd";
  WriteTestDataToSsd("block_read", data);

  ::umbp::PrepareSsdReadRequest req;
  req.set_key("block_read");
  req.set_max_size(data.size());
  ::umbp::PrepareSsdReadResponse resp;
  grpc::ClientContext ctx;

  auto status = stub_->PrepareSsdRead(&ctx, req, &resp);
  ASSERT_TRUE(status.ok()) << status.error_message();
  ASSERT_EQ(resp.status(), ::umbp::SSD_READ_OK);
  EXPECT_LT(resp.staging_offset(), kStagingSize);
  EXPECT_EQ(resp.size(), data.size());
  EXPECT_GT(resp.lease_id(), 0u);
  EXPECT_GT(resp.lease_ttl_ms(), 0u);

  std::string loaded(static_cast<const char*>(staging_buffer_) + resp.staging_offset(),
                     resp.size());
  EXPECT_EQ(loaded, data);
}

TEST_F(PeerServiceSlotTest, PrepareSsdReadNotFound) {
  ::umbp::PrepareSsdReadRequest req;
  req.set_key("nonexistent");
  req.set_max_size(64);
  ::umbp::PrepareSsdReadResponse resp;
  grpc::ClientContext ctx;

  auto status = stub_->PrepareSsdRead(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(resp.status(), ::umbp::SSD_READ_NOT_FOUND);
}

TEST_F(PeerServiceSlotTest, PrepareSsdReadTooLarge) {
  // A key whose actual size exceeds the per-slot size must report
  // SIZE_TOO_LARGE (distinct from NOT_FOUND), not be silently served.
  const std::string big(kStagingSize, 'x');  // larger than one slot (kStagingSize/kNumReadSlots)
  WriteTestDataToSsd("block_big", big);

  ::umbp::PrepareSsdReadRequest req;
  req.set_key("block_big");
  req.set_max_size(kStagingSize);
  ::umbp::PrepareSsdReadResponse resp;
  grpc::ClientContext ctx;

  auto status = stub_->PrepareSsdRead(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(resp.status(), ::umbp::SSD_READ_SIZE_TOO_LARGE);
}

TEST_F(PeerServiceSlotTest, PrepareSsdReadExhaustSlots) {
  for (int i = 0; i < kNumReadSlots; ++i) {
    const std::string key = "block_" + std::to_string(i);
    const std::string data = "data_" + std::to_string(i);
    WriteTestDataToSsd(key, data);
  }

  for (int i = 0; i < kNumReadSlots; ++i) {
    const std::string key = "block_" + std::to_string(i);
    ::umbp::PrepareSsdReadRequest req;
    req.set_key(key);
    req.set_max_size(64);
    ::umbp::PrepareSsdReadResponse resp;
    grpc::ClientContext ctx;
    auto status = stub_->PrepareSsdRead(&ctx, req, &resp);
    ASSERT_TRUE(status.ok());
    ASSERT_EQ(resp.status(), ::umbp::SSD_READ_OK) << "Slot " << i << " should succeed";
  }

  WriteTestDataToSsd("block_extra", "extra");
  ::umbp::PrepareSsdReadRequest req;
  req.set_key("block_extra");
  req.set_max_size(64);
  ::umbp::PrepareSsdReadResponse resp;
  grpc::ClientContext ctx;
  auto status = stub_->PrepareSsdRead(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  // Transient slot exhaustion is NO_SLOT (retryable) — never a NOT_FOUND miss.
  EXPECT_EQ(resp.status(), ::umbp::SSD_READ_NO_SLOT) << "Should be NO_SLOT when slots exhausted";
}

// --- ReleaseSsdLease ---

TEST_F(PeerServiceSlotTest, ReleaseSsdLeaseSuccess) {
  const std::string data = "release test";
  WriteTestDataToSsd("block_rel", data);

  uint64_t lease_id;
  {
    ::umbp::PrepareSsdReadRequest req;
    req.set_key("block_rel");
    req.set_max_size(data.size());
    ::umbp::PrepareSsdReadResponse resp;
    grpc::ClientContext ctx;
    stub_->PrepareSsdRead(&ctx, req, &resp);
    ASSERT_EQ(resp.status(), ::umbp::SSD_READ_OK);
    lease_id = resp.lease_id();
  }

  {
    ::umbp::ReleaseSsdLeaseRequest req;
    req.set_lease_id(lease_id);
    ::umbp::ReleaseSsdLeaseResponse resp;
    grpc::ClientContext ctx;
    auto status = stub_->ReleaseSsdLease(&ctx, req, &resp);
    ASSERT_TRUE(status.ok());
    EXPECT_TRUE(resp.success());
  }

  {
    ::umbp::ReleaseSsdLeaseRequest req;
    req.set_lease_id(lease_id);
    ::umbp::ReleaseSsdLeaseResponse resp;
    grpc::ClientContext ctx;
    auto status = stub_->ReleaseSsdLease(&ctx, req, &resp);
    ASSERT_TRUE(status.ok());
    EXPECT_FALSE(resp.success()) << "Double release should fail";
  }
}

TEST_F(PeerServiceSlotTest, ReleaseSsdLeaseInvalid) {
  ::umbp::ReleaseSsdLeaseRequest req;
  req.set_lease_id(9999);
  ::umbp::ReleaseSsdLeaseResponse resp;
  grpc::ClientContext ctx;
  auto status = stub_->ReleaseSsdLease(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(resp.success());
}

// --- Slot isolation: different slots get different offsets ---

TEST_F(PeerServiceSlotTest, MultipleReadSlotsDifferentOffsets) {
  for (int i = 0; i < kNumReadSlots; ++i) {
    const std::string key = "iso_" + std::to_string(i);
    const std::string data(16, 'a' + i);
    WriteTestDataToSsd(key, data);
  }

  std::vector<uint64_t> offsets;
  for (int i = 0; i < kNumReadSlots; ++i) {
    const std::string key = "iso_" + std::to_string(i);
    ::umbp::PrepareSsdReadRequest req;
    req.set_key(key);
    req.set_max_size(16);
    ::umbp::PrepareSsdReadResponse resp;
    grpc::ClientContext ctx;
    stub_->PrepareSsdRead(&ctx, req, &resp);
    ASSERT_EQ(resp.status(), ::umbp::SSD_READ_OK);
    offsets.push_back(resp.staging_offset());
  }

  for (size_t i = 0; i < offsets.size(); ++i) {
    for (size_t j = i + 1; j < offsets.size(); ++j) {
      EXPECT_NE(offsets[i], offsets[j]) << "Slots " << i << " and " << j << " overlap";
    }
  }
}

// --- TTL reclaim for read slots ---

TEST_F(PeerServiceSlotTest, ReadSlotTtlReclaim) {
  for (int i = 0; i < kNumReadSlots; ++i) {
    const std::string key = "ttl_" + std::to_string(i);
    const std::string data = "ttl_data_" + std::to_string(i);
    WriteTestDataToSsd(key, data);
  }

  for (int i = 0; i < kNumReadSlots; ++i) {
    const std::string key = "ttl_" + std::to_string(i);
    ::umbp::PrepareSsdReadRequest req;
    req.set_key(key);
    req.set_max_size(10);
    ::umbp::PrepareSsdReadResponse resp;
    grpc::ClientContext ctx;
    stub_->PrepareSsdRead(&ctx, req, &resp);
    ASSERT_EQ(resp.status(), ::umbp::SSD_READ_OK);
  }

  std::this_thread::sleep_for(std::chrono::seconds(kLeaseTimeoutS + 1));

  const std::string key = "ttl_0";
  ::umbp::PrepareSsdReadRequest req;
  req.set_key(key);
  req.set_max_size(10);
  ::umbp::PrepareSsdReadResponse resp;
  grpc::ClientContext ctx;
  auto status = stub_->PrepareSsdRead(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(resp.status(), ::umbp::SSD_READ_OK) << "Should succeed after TTL reclaim";
}

// ---- sync-server poller sizing ----------------------------------------------
//
// The values themselves only surface as RPC latency under fan-in, which a unit
// test cannot observe; what is testable is that the env plumbing resolves and
// that an inconsistent pair can never reach gRPC, which requires max >= min.

class PeerPollerConfigTest : public ::testing::Test {
 protected:
  void SetUp() override { Clear(); }
  void TearDown() override { Clear(); }
  static void Clear() {
    ::unsetenv("UMBP_PEER_MIN_POLLERS");
    ::unsetenv("UMBP_PEER_MAX_POLLERS");
    ResetEnvWarnStateForTesting();
  }
};

TEST_F(PeerPollerConfigTest, DefaultsCoverTpFanIn) {
  const PeerPollerConfig cfg = ResolvePeerPollerConfig();
  // >= 8 so a TP8 rank group asking for the same MLA key does not serialize on
  // thread spawns, which is the whole point of overriding gRPC's 1/2.
  EXPECT_GE(cfg.min_pollers, 8);
  EXPECT_GE(cfg.max_pollers, cfg.min_pollers);
}

TEST_F(PeerPollerConfigTest, EnvOverridesBothBounds) {
  ::setenv("UMBP_PEER_MIN_POLLERS", "3", 1);
  ::setenv("UMBP_PEER_MAX_POLLERS", "17", 1);
  const PeerPollerConfig cfg = ResolvePeerPollerConfig();
  EXPECT_EQ(cfg.min_pollers, 3);
  EXPECT_EQ(cfg.max_pollers, 17);
}

TEST_F(PeerPollerConfigTest, MaxIsRaisedToMinWhenInverted) {
  ::setenv("UMBP_PEER_MIN_POLLERS", "12", 1);
  ::setenv("UMBP_PEER_MAX_POLLERS", "4", 1);
  const PeerPollerConfig cfg = ResolvePeerPollerConfig();
  EXPECT_EQ(cfg.min_pollers, 12);
  EXPECT_EQ(cfg.max_pollers, 12) << "gRPC requires max >= min";
}

TEST_F(PeerPollerConfigTest, InvalidValuesFallBackToDefaults) {
  const PeerPollerConfig def = ResolvePeerPollerConfig();
  ResetEnvWarnStateForTesting();
  // 0 is below min_allowed=1; "abc" does not parse.  Neither may produce a
  // pool that cannot serve, so both must land back on the default.
  ::setenv("UMBP_PEER_MIN_POLLERS", "0", 1);
  ::setenv("UMBP_PEER_MAX_POLLERS", "abc", 1);
  const PeerPollerConfig cfg = ResolvePeerPollerConfig();
  EXPECT_EQ(cfg.min_pollers, def.min_pollers);
  EXPECT_EQ(cfg.max_pollers, def.max_pollers);
}

}  // namespace
}  // namespace mori::umbp
