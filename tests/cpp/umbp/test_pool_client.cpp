#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "umbp/master_server.h"
#include "umbp/pool_client.h"
#include "umbp/route_put_strategy.h"

namespace mori::umbp {
namespace {

constexpr uint16_t kMasterBasePort = 50300;

static uint16_t AllocPort() {
  static std::atomic<uint16_t> next{kMasterBasePort};
  return next.fetch_add(1);
}

class LocalOnlyPutStrategy : public RoutePutStrategy {
  std::string local_node_id_;
  TierType tier_;

 public:
  LocalOnlyPutStrategy(std::string node_id, TierType tier)
      : local_node_id_(std::move(node_id)), tier_(tier) {}

  std::optional<RoutePutResult> Select(
      const std::vector<ClientRecord>& alive_clients,
      uint64_t block_size) override {
    for (const auto& c : alive_clients) {
      if (c.node_id == local_node_id_) {
        return RoutePutResult{c.node_id, c.node_address, tier_};
      }
    }
    return std::nullopt;
  }
};

class PoolClientDramTest : public ::testing::Test {
 protected:
  void SetUp() override {
    node_id_ = "test-node-dram";
    port_ = AllocPort();
    master_addr_ = "localhost:" + std::to_string(port_);

    dram_buffer_.resize(1024 * 1024);

    MasterServerConfig master_config;
    master_config.listen_address = "0.0.0.0:" + std::to_string(port_);
    master_config.registry_config.heartbeat_ttl = std::chrono::seconds(30);
    master_config.put_strategy =
        std::make_unique<LocalOnlyPutStrategy>(node_id_, TierType::DRAM);

    master_ = std::make_unique<MasterServer>(std::move(master_config));
    master_thread_ = std::thread([this] { master_->Run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    PoolClientConfig client_config;
    client_config.master_config.master_address = master_addr_;
    client_config.master_config.node_id = node_id_;
    client_config.master_config.node_address = "localhost";
    client_config.master_config.auto_heartbeat = false;
    client_config.exportable_dram_buffer = dram_buffer_.data();
    client_config.exportable_dram_buffer_size = dram_buffer_.size();
    client_config.tier_capacities[TierType::DRAM] = {
        dram_buffer_.size(), dram_buffer_.size()};

    client_ = std::make_unique<PoolClient>(std::move(client_config));
    ASSERT_TRUE(client_->Init());
  }

  void TearDown() override {
    if (client_) {
      client_->Shutdown();
      client_.reset();
    }
    if (master_) {
      master_->Shutdown();
    }
    if (master_thread_.joinable()) {
      master_thread_.join();
    }
  }

  std::string node_id_;
  uint16_t port_;
  std::string master_addr_;
  std::vector<char> dram_buffer_;
  std::unique_ptr<MasterServer> master_;
  std::thread master_thread_;
  std::unique_ptr<PoolClient> client_;
};

TEST_F(PoolClientDramTest, PutGetRemove) {
  const std::string key = "dram-block-1";
  const std::string data = "hello UMBP DRAM pool!";
  std::vector<char> src(data.begin(), data.end());
  std::vector<char> dst(src.size(), 0);

  ASSERT_TRUE(client_->Put(key, src.data(), src.size()));
  ASSERT_TRUE(client_->Get(key, dst.data(), dst.size()));
  EXPECT_EQ(src, dst);
  ASSERT_TRUE(client_->Remove(key));
}

TEST_F(PoolClientDramTest, MultiplePuts) {
  for (int i = 0; i < 5; ++i) {
    std::string key = "dram-multi-" + std::to_string(i);
    std::string data = "data-" + std::to_string(i);
    std::vector<char> src(data.begin(), data.end());

    ASSERT_TRUE(client_->Put(key, src.data(), src.size()));

    std::vector<char> dst(src.size(), 0);
    ASSERT_TRUE(client_->Get(key, dst.data(), dst.size()));
    EXPECT_EQ(src, dst);

    ASSERT_TRUE(client_->Remove(key));
  }
}

class PoolClientSsdTest : public ::testing::Test {
 protected:
  void SetUp() override {
    node_id_ = "test-node-ssd";
    port_ = AllocPort();
    master_addr_ = "localhost:" + std::to_string(port_);

    ssd_dir_ = std::filesystem::temp_directory_path() /
               ("umbp_test_ssd_pool_" + std::to_string(getpid()) + "_" +
                std::to_string(port_));
    std::filesystem::create_directories(ssd_dir_);

    MasterServerConfig master_config;
    master_config.listen_address = "0.0.0.0:" + std::to_string(port_);
    master_config.registry_config.heartbeat_ttl = std::chrono::seconds(30);
    master_config.put_strategy =
        std::make_unique<LocalOnlyPutStrategy>(node_id_, TierType::SSD);

    master_ = std::make_unique<MasterServer>(std::move(master_config));
    master_thread_ = std::thread([this] { master_->Run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    PoolClientConfig client_config;
    client_config.master_config.master_address = master_addr_;
    client_config.master_config.node_id = node_id_;
    client_config.master_config.node_address = "localhost";
    client_config.master_config.auto_heartbeat = false;
    client_config.exportable_ssd_dir = ssd_dir_.string();
    client_config.exportable_ssd_capacity = 10 * 1024 * 1024;
    client_config.tier_capacities[TierType::SSD] = {
        10ULL * 1024 * 1024, 10ULL * 1024 * 1024};

    client_ = std::make_unique<PoolClient>(std::move(client_config));
    ASSERT_TRUE(client_->Init());
  }

  void TearDown() override {
    if (client_) {
      client_->Shutdown();
      client_.reset();
    }
    if (master_) {
      master_->Shutdown();
    }
    if (master_thread_.joinable()) {
      master_thread_.join();
    }
    std::filesystem::remove_all(ssd_dir_);
  }

  std::string node_id_;
  uint16_t port_;
  std::string master_addr_;
  std::filesystem::path ssd_dir_;
  std::unique_ptr<MasterServer> master_;
  std::thread master_thread_;
  std::unique_ptr<PoolClient> client_;
};

TEST_F(PoolClientSsdTest, PutGetRemove) {
  const std::string key = "ssd-block-1";
  const std::string data = "hello UMBP SSD pool!";
  std::vector<char> src(data.begin(), data.end());
  std::vector<char> dst(src.size(), 0);

  ASSERT_TRUE(client_->Put(key, src.data(), src.size()));

  auto file_path = ssd_dir_ / (key + ".bin");
  EXPECT_TRUE(std::filesystem::exists(file_path));

  ASSERT_TRUE(client_->Get(key, dst.data(), dst.size()));
  EXPECT_EQ(src, dst);

  ASSERT_TRUE(client_->Remove(key));
}

TEST_F(PoolClientSsdTest, MultiplePuts) {
  for (int i = 0; i < 5; ++i) {
    std::string key = "ssd-multi-" + std::to_string(i);
    std::string data = "ssd-data-" + std::to_string(i) + "-payload";
    std::vector<char> src(data.begin(), data.end());

    ASSERT_TRUE(client_->Put(key, src.data(), src.size()));

    std::vector<char> dst(src.size(), 0);
    ASSERT_TRUE(client_->Get(key, dst.data(), dst.size()));
    EXPECT_EQ(src, dst);

    ASSERT_TRUE(client_->Remove(key));
  }
}

TEST_F(PoolClientSsdTest, RemoveNonexistentKey) {
  EXPECT_FALSE(client_->Remove("nonexistent-key"));
}

class PoolClientLifecycleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    port_ = AllocPort();
    master_addr_ = "localhost:" + std::to_string(port_);

    MasterServerConfig master_config;
    master_config.listen_address = "0.0.0.0:" + std::to_string(port_);
    master_config.registry_config.heartbeat_ttl = std::chrono::seconds(30);

    master_ = std::make_unique<MasterServer>(std::move(master_config));
    master_thread_ = std::thread([this] { master_->Run(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
  }

  void TearDown() override {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (master_) {
      master_->Shutdown();
    }
    if (master_thread_.joinable()) {
      master_thread_.join();
    }
  }

  uint16_t port_;
  std::string master_addr_;
  std::unique_ptr<MasterServer> master_;
  std::thread master_thread_;
};

TEST_F(PoolClientLifecycleTest, InitShutdown) {
  PoolClientConfig config;
  config.master_config.master_address = master_addr_;
  config.master_config.node_id = "lifecycle-node";
  config.master_config.node_address = "localhost";
  config.master_config.auto_heartbeat = false;

  PoolClient client(std::move(config));
  EXPECT_FALSE(client.IsInitialized());

  ASSERT_TRUE(client.Init());
  EXPECT_TRUE(client.IsInitialized());

  client.Shutdown();
  EXPECT_FALSE(client.IsInitialized());
}

TEST_F(PoolClientLifecycleTest, DoubleInitIdempotent) {
  PoolClientConfig config;
  config.master_config.master_address = master_addr_;
  config.master_config.node_id = "double-init-node";
  config.master_config.node_address = "localhost";
  config.master_config.auto_heartbeat = false;

  PoolClient client(std::move(config));
  ASSERT_TRUE(client.Init());
  ASSERT_TRUE(client.Init());

  client.Shutdown();
}

}  // namespace
}  // namespace mori::umbp
