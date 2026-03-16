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
      if (c.node_id == local_node_id_ &&
          c.tier_capacities.count(tier_) > 0) {
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
    client_config.dram_buffers.push_back(
        {dram_buffer_.data(), dram_buffer_.size()});
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
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (master_) {
      master_->Shutdown();
    }
    if (master_thread_.joinable()) {
      master_thread_.join();
    }
    master_.reset();
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
    client_config.ssd_stores.push_back({ssd_dir_.string(), 10 * 1024 * 1024});
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
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (master_) {
      master_->Shutdown();
    }
    if (master_thread_.joinable()) {
      master_thread_.join();
    }
    master_.reset();
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

class PoolClientMultiSsdTest : public ::testing::Test {
 protected:
  void SetUp() override {
    node_id_ = "test-node-multi-ssd";
    port_ = AllocPort();
    master_addr_ = "localhost:" + std::to_string(port_);

    std::string base = std::filesystem::temp_directory_path().string() +
                       "/umbp_test_multi_ssd_" + std::to_string(getpid()) + "_" +
                       std::to_string(port_);
    ssd_dir_0_ = base + "_0";
    ssd_dir_1_ = base + "_1";
    std::filesystem::create_directories(ssd_dir_0_);
    std::filesystem::create_directories(ssd_dir_1_);

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
    client_config.ssd_stores.push_back({ssd_dir_0_, 1024});
    client_config.ssd_stores.push_back({ssd_dir_1_, 1024});
    client_config.tier_capacities[TierType::SSD] = {2048, 2048};

    client_ = std::make_unique<PoolClient>(std::move(client_config));
    ASSERT_TRUE(client_->Init());
  }

  void TearDown() override {
    if (client_) {
      client_->Shutdown();
      client_.reset();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (master_) {
      master_->Shutdown();
    }
    if (master_thread_.joinable()) {
      master_thread_.join();
    }
    master_.reset();
    std::filesystem::remove_all(ssd_dir_0_);
    std::filesystem::remove_all(ssd_dir_1_);
  }

  std::string node_id_;
  uint16_t port_;
  std::string master_addr_;
  std::string ssd_dir_0_;
  std::string ssd_dir_1_;
  std::unique_ptr<MasterServer> master_;
  std::thread master_thread_;
  std::unique_ptr<PoolClient> client_;
};

TEST_F(PoolClientMultiSsdTest, DistributesAcrossStores) {
  std::vector<std::string> keys;
  for (int i = 0; i < 6; ++i) {
    std::string key = "multi-ssd-" + std::to_string(i);
    std::string data = "data-" + std::to_string(i) + "-payload";
    std::vector<char> src(data.begin(), data.end());

    ASSERT_TRUE(client_->Put(key, src.data(), src.size()))
        << "Put failed for key=" << key;
    keys.push_back(key);

    std::vector<char> dst(src.size(), 0);
    ASSERT_TRUE(client_->Get(key, dst.data(), dst.size()));
    EXPECT_EQ(src, dst);
  }

  int files_in_0 = 0, files_in_1 = 0;
  for (const auto& entry : std::filesystem::directory_iterator(ssd_dir_0_)) {
    if (entry.path().extension() == ".bin") ++files_in_0;
  }
  for (const auto& entry : std::filesystem::directory_iterator(ssd_dir_1_)) {
    if (entry.path().extension() == ".bin") ++files_in_1;
  }
  EXPECT_EQ(files_in_0 + files_in_1, 6);

  for (const auto& key : keys) {
    ASSERT_TRUE(client_->Remove(key));
  }
}

class PoolClientSsdFullTest : public ::testing::Test {
 protected:
  void SetUp() override {
    node_id_ = "test-node-ssd-full";
    port_ = AllocPort();
    master_addr_ = "localhost:" + std::to_string(port_);

    std::string base = std::filesystem::temp_directory_path().string() +
                       "/umbp_test_ssd_full_" + std::to_string(getpid()) + "_" +
                       std::to_string(port_);
    ssd_dir_0_ = base + "_0";
    ssd_dir_1_ = base + "_1";
    std::filesystem::create_directories(ssd_dir_0_);
    std::filesystem::create_directories(ssd_dir_1_);

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
    client_config.ssd_stores.push_back({ssd_dir_0_, 100});
    client_config.ssd_stores.push_back({ssd_dir_1_, 100});
    client_config.tier_capacities[TierType::SSD] = {200, 200};

    client_ = std::make_unique<PoolClient>(std::move(client_config));
    ASSERT_TRUE(client_->Init());
  }

  void TearDown() override {
    if (client_) {
      client_->Shutdown();
      client_.reset();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (master_) {
      master_->Shutdown();
    }
    if (master_thread_.joinable()) {
      master_thread_.join();
    }
    master_.reset();
    std::filesystem::remove_all(ssd_dir_0_);
    std::filesystem::remove_all(ssd_dir_1_);
  }

  std::string node_id_;
  uint16_t port_;
  std::string master_addr_;
  std::string ssd_dir_0_;
  std::string ssd_dir_1_;
  std::unique_ptr<MasterServer> master_;
  std::thread master_thread_;
  std::unique_ptr<PoolClient> client_;
};

TEST_F(PoolClientSsdFullTest, OverflowToSecondStore) {
  std::string data_80(80, 'A');
  ASSERT_TRUE(client_->Put("fill-0", data_80.data(), data_80.size()));

  std::string data_60(60, 'B');
  ASSERT_TRUE(client_->Put("fill-1", data_60.data(), data_60.size()));

  std::vector<char> dst(80, 0);
  ASSERT_TRUE(client_->Get("fill-0", dst.data(), dst.size()));
  EXPECT_EQ(std::string(dst.begin(), dst.end()), data_80);

  dst.resize(60);
  std::fill(dst.begin(), dst.end(), 0);
  ASSERT_TRUE(client_->Get("fill-1", dst.data(), dst.size()));
  EXPECT_EQ(std::string(dst.begin(), dst.end()), data_60);

  ASSERT_TRUE(client_->Remove("fill-0"));
  ASSERT_TRUE(client_->Remove("fill-1"));
}

TEST_F(PoolClientSsdFullTest, BothStoresFullReturnsFalse) {
  std::string data_90(90, 'X');
  ASSERT_TRUE(client_->Put("big-0", data_90.data(), data_90.size()));
  ASSERT_TRUE(client_->Put("big-1", data_90.data(), data_90.size()));

  std::string data_30(30, 'Y');
  EXPECT_FALSE(client_->Put("too-much", data_30.data(), data_30.size()));

  ASSERT_TRUE(client_->Remove("big-0"));
  ASSERT_TRUE(client_->Remove("big-1"));
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
    if (master_) {
      master_->Shutdown();
    }
    if (master_thread_.joinable()) {
      master_thread_.join();
    }
    master_.reset();
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

// ---------------------------------------------------------------------------
// Multi-DRAM buffer test
// ---------------------------------------------------------------------------
class PoolClientMultiDramTest : public ::testing::Test {
 protected:
  void SetUp() override {
    node_id_ = "test-node-multi-dram";
    port_ = AllocPort();
    master_addr_ = "localhost:" + std::to_string(port_);

    // Two small DRAM buffers (128 bytes each)
    dram_buf_0_ = std::make_unique<char[]>(128);
    dram_buf_1_ = std::make_unique<char[]>(128);
    std::memset(dram_buf_0_.get(), 0, 128);
    std::memset(dram_buf_1_.get(), 0, 128);

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
    client_config.dram_buffers.push_back({dram_buf_0_.get(), 128});
    client_config.dram_buffers.push_back({dram_buf_1_.get(), 128});
    client_config.tier_capacities[TierType::DRAM] = {256, 256};

    client_ = std::make_unique<PoolClient>(std::move(client_config));
    ASSERT_TRUE(client_->Init());
  }

  void TearDown() override {
    if (client_) {
      client_->Shutdown();
      client_.reset();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (master_) {
      master_->Shutdown();
    }
    if (master_thread_.joinable()) {
      master_thread_.join();
    }
    master_.reset();
  }

  std::string node_id_;
  uint16_t port_;
  std::string master_addr_;
  std::unique_ptr<char[]> dram_buf_0_;
  std::unique_ptr<char[]> dram_buf_1_;
  std::unique_ptr<MasterServer> master_;
  std::thread master_thread_;
  std::unique_ptr<PoolClient> client_;
};

TEST_F(PoolClientMultiDramTest, PutGetAcrossBuffers) {
  // Fill buffer 0 (128 bytes) then overflow to buffer 1
  std::string data_a(64, 'A');
  std::string data_b(64, 'B');
  std::string data_c(64, 'C');

  // First two Puts should go to buffer 0
  ASSERT_TRUE(client_->Put("dram-multi-a", data_a.data(), data_a.size()));
  ASSERT_TRUE(client_->Put("dram-multi-b", data_b.data(), data_b.size()));

  // Buffer 0 is now full (128 bytes used). Third Put should go to buffer 1
  ASSERT_TRUE(client_->Put("dram-multi-c", data_c.data(), data_c.size()));

  // Verify all Get correctly
  std::vector<char> dst(64, 0);

  ASSERT_TRUE(client_->Get("dram-multi-a", dst.data(), dst.size()));
  EXPECT_EQ(std::string(dst.begin(), dst.end()), data_a);

  ASSERT_TRUE(client_->Get("dram-multi-b", dst.data(), dst.size()));
  EXPECT_EQ(std::string(dst.begin(), dst.end()), data_b);

  ASSERT_TRUE(client_->Get("dram-multi-c", dst.data(), dst.size()));
  EXPECT_EQ(std::string(dst.begin(), dst.end()), data_c);

  // Remove all and verify capacity recovered
  ASSERT_TRUE(client_->Remove("dram-multi-a"));
  ASSERT_TRUE(client_->Remove("dram-multi-b"));
  ASSERT_TRUE(client_->Remove("dram-multi-c"));

  // Should be able to Put again after removal
  ASSERT_TRUE(client_->Put("dram-multi-d", data_a.data(), data_a.size()));
  ASSERT_TRUE(client_->Get("dram-multi-d", dst.data(), dst.size()));
  EXPECT_EQ(std::string(dst.begin(), dst.end()), data_a);
  client_->Remove("dram-multi-d");
}

TEST_F(PoolClientMultiDramTest, AllBuffersFull) {
  // Fill both buffers completely (128 + 128 = 256 bytes)
  std::string big(128, 'X');
  ASSERT_TRUE(client_->Put("fill-0", big.data(), big.size()));
  ASSERT_TRUE(client_->Put("fill-1", big.data(), big.size()));

  // Third Put should fail - no space
  std::string small(1, 'Y');
  EXPECT_FALSE(client_->Put("overflow", small.data(), small.size()));

  // Remove one, now space available
  ASSERT_TRUE(client_->Remove("fill-0"));
  ASSERT_TRUE(client_->Put("reuse", small.data(), small.size()));
  client_->Remove("reuse");
  client_->Remove("fill-1");
}

}  // namespace
}  // namespace mori::umbp
