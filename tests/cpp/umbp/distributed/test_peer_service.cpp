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

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/distributed/peer/peer_service.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/local/block_index/local_block_index.h"
#include "umbp/local/storage/local_storage_manager.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {
namespace {

constexpr size_t kStagingSize = 4096;
constexpr uint16_t kBasePort = 50200;
constexpr int kNumReadSlots = 4;
constexpr int kNumWriteSlots = 4;
constexpr int kLeaseTimeoutS = 2;

static uint16_t AllocPort() {
  static std::atomic<uint16_t> next{kBasePort};
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

    UMBPConfig cfg;
    cfg.dram.capacity_bytes = 1 << 20;
    cfg.ssd.enabled = true;
    cfg.ssd.storage_dir = ssd_dir_.string();
    cfg.ssd.capacity_bytes = 1 << 20;
    storage_ = std::make_unique<LocalStorageManager>(cfg, &index_);

    PoolClientConfig pc_cfg;
    pc_cfg.master_config.master_address = "localhost:9999";
    pc_cfg.master_config.node_id = "test_node";
    coordinator_ = std::make_unique<PoolClient>(std::move(pc_cfg));

    port_ = AllocPort();

    server_ = std::make_unique<PeerServiceServer>(
        staging_buffer_, kStagingSize, ssd_staging_mem_desc_, *storage_, index_, *coordinator_,
        kNumReadSlots, kNumWriteSlots, kLeaseTimeoutS);
    server_->Start(port_);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto channel = grpc::CreateChannel("localhost:" + std::to_string(port_),
                                       grpc::InsecureChannelCredentials());
    stub_ = ::umbp::UMBPPeer::NewStub(channel);
  }

  void TearDown() override {
    server_->Stop();
    server_.reset();
    coordinator_.reset();
    storage_.reset();
    std::free(staging_buffer_);
    std::filesystem::remove_all(ssd_dir_);
  }

  void WriteTestDataToSsd(const std::string& key, const std::string& data) {
    storage_->Write(key, data.data(), data.size(), StorageTier::LOCAL_SSD);
    index_.Insert(key, {StorageTier::LOCAL_SSD, 0, data.size()});
  }

  void* staging_buffer_ = nullptr;
  std::filesystem::path ssd_dir_;
  std::vector<uint8_t> ssd_staging_mem_desc_;
  uint16_t port_ = 0;
  LocalBlockIndex index_;
  std::unique_ptr<LocalStorageManager> storage_;
  std::unique_ptr<PoolClient> coordinator_;
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

// --- AllocateWriteSlot ---

TEST_F(PeerServiceSlotTest, AllocateWriteSlotSuccess) {
  ::umbp::AllocateWriteSlotRequest req;
  req.set_size(64);
  ::umbp::AllocateWriteSlotResponse resp;
  grpc::ClientContext ctx;

  auto status = stub_->AllocateWriteSlot(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  ASSERT_TRUE(resp.success());
  EXPECT_GT(resp.lease_id(), 0u);
  EXPECT_GT(resp.lease_ttl_ms(), 0u);
  EXPECT_LT(resp.staging_offset(), kStagingSize / 2);
}

TEST_F(PeerServiceSlotTest, AllocateWriteSlotTooLarge) {
  ::umbp::AllocateWriteSlotRequest req;
  req.set_size(kStagingSize);  // exceeds slot size
  ::umbp::AllocateWriteSlotResponse resp;
  grpc::ClientContext ctx;

  auto status = stub_->AllocateWriteSlot(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(resp.success());
}

TEST_F(PeerServiceSlotTest, AllocateWriteSlotZeroSize) {
  ::umbp::AllocateWriteSlotRequest req;
  req.set_size(0);
  ::umbp::AllocateWriteSlotResponse resp;
  grpc::ClientContext ctx;

  auto status = stub_->AllocateWriteSlot(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(resp.success());
}

TEST_F(PeerServiceSlotTest, AllocateWriteSlotExhaustAll) {
  for (int i = 0; i < kNumWriteSlots; ++i) {
    ::umbp::AllocateWriteSlotRequest req;
    req.set_size(32);
    ::umbp::AllocateWriteSlotResponse resp;
    grpc::ClientContext ctx;
    auto status = stub_->AllocateWriteSlot(&ctx, req, &resp);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(resp.success()) << "Slot " << i << " should succeed";
  }

  ::umbp::AllocateWriteSlotRequest req;
  req.set_size(32);
  ::umbp::AllocateWriteSlotResponse resp;
  grpc::ClientContext ctx;
  auto status = stub_->AllocateWriteSlot(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(resp.success()) << "Should fail when all slots exhausted";
}

TEST_F(PeerServiceSlotTest, AllocateWriteSlotTtlReclaim) {
  for (int i = 0; i < kNumWriteSlots; ++i) {
    ::umbp::AllocateWriteSlotRequest req;
    req.set_size(32);
    ::umbp::AllocateWriteSlotResponse resp;
    grpc::ClientContext ctx;
    stub_->AllocateWriteSlot(&ctx, req, &resp);
  }

  std::this_thread::sleep_for(std::chrono::seconds(kLeaseTimeoutS + 1));

  ::umbp::AllocateWriteSlotRequest req;
  req.set_size(32);
  ::umbp::AllocateWriteSlotResponse resp;
  grpc::ClientContext ctx;
  auto status = stub_->AllocateWriteSlot(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_TRUE(resp.success()) << "Should succeed after TTL reclaim";
  EXPECT_GT(server_->Metrics().expired_reclaims.load(), 0u);
}

// --- PrepareSsdRead ---

TEST_F(PeerServiceSlotTest, PrepareSsdReadSuccess) {
  const std::string data = "read me from ssd";
  WriteTestDataToSsd("block_read", data);

  ::umbp::PrepareSsdReadRequest req;
  req.set_key("block_read");
  req.set_ssd_location_id("0:block_read");
  req.set_size(data.size());
  ::umbp::PrepareSsdReadResponse resp;
  grpc::ClientContext ctx;

  auto status = stub_->PrepareSsdRead(&ctx, req, &resp);
  ASSERT_TRUE(status.ok()) << status.error_message();
  ASSERT_TRUE(resp.success());
  EXPECT_GE(resp.staging_offset(), kStagingSize / 2);
  EXPECT_GT(resp.lease_id(), 0u);
  EXPECT_GT(resp.lease_ttl_ms(), 0u);

  std::string loaded(static_cast<const char*>(staging_buffer_) + resp.staging_offset(),
                     data.size());
  EXPECT_EQ(loaded, data);
}

TEST_F(PeerServiceSlotTest, PrepareSsdReadNotFound) {
  ::umbp::PrepareSsdReadRequest req;
  req.set_key("nonexistent");
  req.set_ssd_location_id("nonexistent");
  req.set_size(64);
  ::umbp::PrepareSsdReadResponse resp;
  grpc::ClientContext ctx;

  auto status = stub_->PrepareSsdRead(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(resp.success());
}

TEST_F(PeerServiceSlotTest, PrepareSsdReadTooLarge) {
  ::umbp::PrepareSsdReadRequest req;
  req.set_key("anything");
  req.set_ssd_location_id("anything");
  req.set_size(kStagingSize);
  ::umbp::PrepareSsdReadResponse resp;
  grpc::ClientContext ctx;

  auto status = stub_->PrepareSsdRead(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(resp.success());
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
    req.set_ssd_location_id("0:" + key);
    req.set_size(6);
    ::umbp::PrepareSsdReadResponse resp;
    grpc::ClientContext ctx;
    auto status = stub_->PrepareSsdRead(&ctx, req, &resp);
    ASSERT_TRUE(status.ok());
    ASSERT_TRUE(resp.success()) << "Slot " << i << " should succeed";
  }

  WriteTestDataToSsd("block_extra", "extra");
  ::umbp::PrepareSsdReadRequest req;
  req.set_key("block_extra");
  req.set_ssd_location_id("0:block_extra");
  req.set_size(5);
  ::umbp::PrepareSsdReadResponse resp;
  grpc::ClientContext ctx;
  auto status = stub_->PrepareSsdRead(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(resp.success()) << "Should fail when all read slots exhausted";
}

// --- ReleaseSsdLease ---

TEST_F(PeerServiceSlotTest, ReleaseSsdLeaseSuccess) {
  const std::string data = "release test";
  WriteTestDataToSsd("block_rel", data);

  uint64_t lease_id;
  {
    ::umbp::PrepareSsdReadRequest req;
    req.set_key("block_rel");
    req.set_ssd_location_id("0:block_rel");
    req.set_size(data.size());
    ::umbp::PrepareSsdReadResponse resp;
    grpc::ClientContext ctx;
    stub_->PrepareSsdRead(&ctx, req, &resp);
    ASSERT_TRUE(resp.success());
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

// --- CommitSsdWrite with lease_id ---

TEST_F(PeerServiceSlotTest, CommitSsdWriteInvalidLeaseId) {
  ::umbp::CommitSsdWriteRequest req;
  req.set_key("block_bad_lease");
  req.set_staging_offset(0);
  req.set_size(10);
  req.set_store_index(0);
  req.set_lease_id(9999);
  ::umbp::CommitSsdWriteResponse resp;
  grpc::ClientContext ctx;

  auto status = stub_->CommitSsdWrite(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(resp.success());
  EXPECT_GT(server_->Metrics().invalid_lease_rejects.load(), 0u);
}

TEST_F(PeerServiceSlotTest, CommitSsdWriteBadStoreIndex) {
  ::umbp::AllocateWriteSlotRequest alloc_req;
  alloc_req.set_size(32);
  ::umbp::AllocateWriteSlotResponse alloc_resp;
  grpc::ClientContext alloc_ctx;
  stub_->AllocateWriteSlot(&alloc_ctx, alloc_req, &alloc_resp);
  ASSERT_TRUE(alloc_resp.success());

  ::umbp::CommitSsdWriteRequest req;
  req.set_key("block_bad_store");
  req.set_staging_offset(alloc_resp.staging_offset());
  req.set_size(32);
  req.set_store_index(99);
  req.set_lease_id(alloc_resp.lease_id());
  ::umbp::CommitSsdWriteResponse resp;
  grpc::ClientContext ctx;

  auto status = stub_->CommitSsdWrite(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(resp.success());
}

TEST_F(PeerServiceSlotTest, CommitSsdWriteSizeTooLarge) {
  ::umbp::AllocateWriteSlotRequest alloc_req;
  alloc_req.set_size(32);
  ::umbp::AllocateWriteSlotResponse alloc_resp;
  grpc::ClientContext alloc_ctx;
  stub_->AllocateWriteSlot(&alloc_ctx, alloc_req, &alloc_resp);
  ASSERT_TRUE(alloc_resp.success());

  ::umbp::CommitSsdWriteRequest req;
  req.set_key("block_size_check");
  req.set_staging_offset(alloc_resp.staging_offset());
  req.set_size(64);  // larger than allocated 32
  req.set_store_index(0);
  req.set_lease_id(alloc_resp.lease_id());
  ::umbp::CommitSsdWriteResponse resp;
  grpc::ClientContext ctx;

  auto status = stub_->CommitSsdWrite(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(resp.success());
}

TEST_F(PeerServiceSlotTest, CommitSsdWriteOffsetMismatch) {
  ::umbp::AllocateWriteSlotRequest alloc_req;
  alloc_req.set_size(32);
  ::umbp::AllocateWriteSlotResponse alloc_resp;
  grpc::ClientContext alloc_ctx;
  stub_->AllocateWriteSlot(&alloc_ctx, alloc_req, &alloc_resp);
  ASSERT_TRUE(alloc_resp.success());

  ::umbp::CommitSsdWriteRequest req;
  req.set_key("block_offset_check");
  req.set_staging_offset(alloc_resp.staging_offset() + 1);  // wrong offset
  req.set_size(32);
  req.set_store_index(0);
  req.set_lease_id(alloc_resp.lease_id());
  ::umbp::CommitSsdWriteResponse resp;
  grpc::ClientContext ctx;

  auto status = stub_->CommitSsdWrite(&ctx, req, &resp);
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
    req.set_ssd_location_id("0:" + key);
    req.set_size(16);
    ::umbp::PrepareSsdReadResponse resp;
    grpc::ClientContext ctx;
    stub_->PrepareSsdRead(&ctx, req, &resp);
    ASSERT_TRUE(resp.success());
    offsets.push_back(resp.staging_offset());
  }

  for (size_t i = 0; i < offsets.size(); ++i) {
    for (size_t j = i + 1; j < offsets.size(); ++j) {
      EXPECT_NE(offsets[i], offsets[j]) << "Slots " << i << " and " << j << " overlap";
    }
  }
}

TEST_F(PeerServiceSlotTest, MultipleWriteSlotsDifferentOffsets) {
  std::vector<uint64_t> offsets;
  for (int i = 0; i < kNumWriteSlots; ++i) {
    ::umbp::AllocateWriteSlotRequest req;
    req.set_size(32);
    ::umbp::AllocateWriteSlotResponse resp;
    grpc::ClientContext ctx;
    stub_->AllocateWriteSlot(&ctx, req, &resp);
    ASSERT_TRUE(resp.success());
    offsets.push_back(resp.staging_offset());
  }

  for (size_t i = 0; i < offsets.size(); ++i) {
    for (size_t j = i + 1; j < offsets.size(); ++j) {
      EXPECT_NE(offsets[i], offsets[j]) << "Write slots " << i << " and " << j << " overlap";
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
    req.set_ssd_location_id("0:" + key);
    req.set_size(10);
    ::umbp::PrepareSsdReadResponse resp;
    grpc::ClientContext ctx;
    stub_->PrepareSsdRead(&ctx, req, &resp);
    ASSERT_TRUE(resp.success());
  }

  std::this_thread::sleep_for(std::chrono::seconds(kLeaseTimeoutS + 1));

  const std::string key = "ttl_0";
  ::umbp::PrepareSsdReadRequest req;
  req.set_key(key);
  req.set_ssd_location_id("0:" + key);
  req.set_size(10);
  ::umbp::PrepareSsdReadResponse resp;
  grpc::ClientContext ctx;
  auto status = stub_->PrepareSsdRead(&ctx, req, &resp);
  ASSERT_TRUE(status.ok());
  EXPECT_TRUE(resp.success()) << "Should succeed after TTL reclaim";
}

}  // namespace
}  // namespace mori::umbp
