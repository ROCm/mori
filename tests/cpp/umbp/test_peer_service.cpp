#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "umbp/peer_service.h"
#include "umbp_peer.grpc.pb.h"

namespace mori::umbp {
namespace {

constexpr size_t kStagingSize = 4096;
constexpr size_t kSsdCapacity = 1 << 20;  // 1 MB
constexpr uint64_t kStagingBaseOffset = 0;
constexpr uint16_t kBasePort = 50200;

static uint16_t AllocPort() {
  static std::atomic<uint16_t> next{kBasePort};
  return next.fetch_add(1);
}

class PeerServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    staging_buffer_ = std::malloc(kStagingSize);
    ASSERT_NE(staging_buffer_, nullptr);
    std::memset(staging_buffer_, 0, kStagingSize);

    ssd_dir_ = std::filesystem::temp_directory_path() / ("umbp_test_ssd_" + std::to_string(getpid())
               + "_" + std::to_string(AllocPort()));
    std::filesystem::create_directories(ssd_dir_);

    engine_desc_ = {0x01, 0x02, 0x03, 0x04};
    dram_memory_desc_ = {0xA0, 0xB0, 0xC0};

    port_ = AllocPort();

    server_ = std::make_unique<PeerServiceServer>(staging_buffer_, kStagingSize, engine_desc_,
                                                  dram_memory_desc_, ssd_dir_.string(),
                                                  kSsdCapacity, kStagingBaseOffset);
    server_->Start(port_);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto channel = grpc::CreateChannel("localhost:" + std::to_string(port_),
                                       grpc::InsecureChannelCredentials());
    stub_ = ::umbp::UMBPPeer::NewStub(channel);
  }

  void TearDown() override {
    server_->Stop();
    server_.reset();
    std::free(staging_buffer_);
    std::filesystem::remove_all(ssd_dir_);
  }

  void* staging_buffer_ = nullptr;
  std::filesystem::path ssd_dir_;
  std::vector<uint8_t> engine_desc_;
  std::vector<uint8_t> dram_memory_desc_;
  uint16_t port_ = 0;
  std::unique_ptr<PeerServiceServer> server_;
  std::unique_ptr<::umbp::UMBPPeer::Stub> stub_;
};

TEST_F(PeerServiceTest, GetPeerInfo) {
  ::umbp::GetPeerInfoRequest request;
  ::umbp::GetPeerInfoResponse response;
  grpc::ClientContext context;

  auto status = stub_->GetPeerInfo(&context, request, &response);
  ASSERT_TRUE(status.ok()) << status.error_message();

  EXPECT_EQ(response.engine_desc(),
            std::string(engine_desc_.begin(), engine_desc_.end()));
  EXPECT_EQ(response.dram_memory_desc(),
            std::string(dram_memory_desc_.begin(), dram_memory_desc_.end()));
  EXPECT_EQ(response.ssd_capacity(), kSsdCapacity);
  EXPECT_EQ(response.ssd_available(), kSsdCapacity);
  EXPECT_EQ(response.staging_base_offset(), kStagingBaseOffset);
}

TEST_F(PeerServiceTest, CommitSsdWriteSuccess) {
  const std::string test_data = "hello, ssd world!";
  std::memcpy(staging_buffer_, test_data.data(), test_data.size());

  ::umbp::CommitSsdWriteRequest request;
  request.set_key("block_1");
  request.set_staging_offset(0);
  request.set_size(test_data.size());

  ::umbp::CommitSsdWriteResponse response;
  grpc::ClientContext context;

  auto status = stub_->CommitSsdWrite(&context, request, &response);
  ASSERT_TRUE(status.ok()) << status.error_message();
  EXPECT_TRUE(response.success());
  EXPECT_EQ(response.ssd_location_id(), "block_1.bin");
}

TEST_F(PeerServiceTest, CommitSsdWriteFileContents) {
  const std::string test_data = "verify file contents";
  std::memcpy(staging_buffer_, test_data.data(), test_data.size());

  ::umbp::CommitSsdWriteRequest request;
  request.set_key("block_verify");
  request.set_staging_offset(0);
  request.set_size(test_data.size());

  ::umbp::CommitSsdWriteResponse response;
  grpc::ClientContext context;

  auto status = stub_->CommitSsdWrite(&context, request, &response);
  ASSERT_TRUE(status.ok()) << status.error_message();
  ASSERT_TRUE(response.success());

  std::ifstream file(ssd_dir_ / "block_verify.bin", std::ios::binary);
  ASSERT_TRUE(file.is_open());
  std::string file_contents((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  EXPECT_EQ(file_contents, test_data);
}

TEST_F(PeerServiceTest, PrepareSsdReadSuccess) {
  const std::string test_data = "read me from ssd";
  std::memcpy(staging_buffer_, test_data.data(), test_data.size());

  {
    ::umbp::CommitSsdWriteRequest req;
    req.set_key("block_read");
    req.set_staging_offset(0);
    req.set_size(test_data.size());
    ::umbp::CommitSsdWriteResponse resp;
    grpc::ClientContext ctx;
    auto s = stub_->CommitSsdWrite(&ctx, req, &resp);
    ASSERT_TRUE(s.ok());
    ASSERT_TRUE(resp.success());
  }

  std::memset(staging_buffer_, 0, kStagingSize);

  {
    ::umbp::PrepareSsdReadRequest req;
    req.set_key("block_read");
    req.set_ssd_location_id("block_read.bin");
    req.set_size(test_data.size());
    ::umbp::PrepareSsdReadResponse resp;
    grpc::ClientContext ctx;
    auto s = stub_->PrepareSsdRead(&ctx, req, &resp);
    ASSERT_TRUE(s.ok()) << s.error_message();
    ASSERT_TRUE(resp.success());
    EXPECT_EQ(resp.staging_offset(), kStagingBaseOffset);
  }

  std::string loaded(static_cast<const char*>(staging_buffer_) + kStagingBaseOffset,
                     test_data.size());
  EXPECT_EQ(loaded, test_data);
}

TEST_F(PeerServiceTest, PrepareSsdReadNotFound) {
  ::umbp::PrepareSsdReadRequest request;
  request.set_key("nonexistent");
  request.set_ssd_location_id("nonexistent.bin");
  request.set_size(64);

  ::umbp::PrepareSsdReadResponse response;
  grpc::ClientContext context;

  auto status = stub_->PrepareSsdRead(&context, request, &response);
  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(response.success());
}

TEST_F(PeerServiceTest, CommitSsdWriteOverCapacity) {
  auto server_small = std::make_unique<PeerServiceServer>(
      staging_buffer_, kStagingSize, engine_desc_, dram_memory_desc_,
      ssd_dir_.string(), 16, kStagingBaseOffset);
  uint16_t small_port = AllocPort();
  server_small->Start(small_port);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  auto channel = grpc::CreateChannel("localhost:" + std::to_string(small_port),
                                     grpc::InsecureChannelCredentials());
  auto small_stub = ::umbp::UMBPPeer::NewStub(channel);

  const std::string data(32, 'x');
  std::memcpy(staging_buffer_, data.data(), data.size());

  ::umbp::CommitSsdWriteRequest request;
  request.set_key("too_big");
  request.set_staging_offset(0);
  request.set_size(data.size());

  ::umbp::CommitSsdWriteResponse response;
  grpc::ClientContext context;

  auto status = small_stub->CommitSsdWrite(&context, request, &response);
  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(response.success());

  server_small->Stop();
}

}  // namespace
}  // namespace mori::umbp
