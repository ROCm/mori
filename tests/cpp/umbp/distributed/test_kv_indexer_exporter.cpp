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
#include <google/protobuf/util/json_util.h>
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>

#include "umbp/distributed/indexer/kv_indexer_exporter.h"
#include "umbp/distributed/peer/backend/page_backend.h"
#include "umbp/distributed/pool_client.h"

namespace mori::umbp {
namespace {
namespace pb = ::kv_indexer::v1;
using namespace std::chrono_literals;

KvIndexerExporter::Config Config(const std::string& endpoint, uint32_t components = 1) {
  KvIndexerExporter::Config config;
  config.set_endpoint(endpoint);
  config.set_worker_id("umbp-test-worker");
  config.set_worker_address("http://127.0.0.1:30000");
  auto* spec = config.mutable_cache_spec();
  spec->set_version(1);
  spec->set_components(components);
  spec->set_full_tier_mask(6);
  spec->set_swa_tier_mask(6);
  spec->set_mamba_tier_mask(6);
  spec->set_swa_window_tokens(32);
  config.set_rpc_timeout_ms(1000);
  return config;
}

indexer::BlockMapping* Block(KvIndexerExporter::Config& config, int64_t hash,
                             std::vector<std::string> keys) {
  auto* block = config.add_blocks();
  block->set_hash(hash);
  block->set_block_size(16);
  auto* component = block->add_components();
  component->set_component(1);
  for (auto& key : keys) component->add_keys(std::move(key));
  return block;
}

std::unique_ptr<PageBackend> Backend() {
  PageBackend::TierConfig config;
  config.buffer_sizes = {1024 * 32};
  config.buffer_descs = {{}};  // Metadata-only allocator; no transfer registrations.
  return std::make_unique<PageBackend>(TierType::DRAM, 1024, config, 5s, 0ms);
}

void Commit(PageBackend& backend, const std::string& key) {
  auto slot = backend.Allocate(key, 1024);
  ASSERT_EQ(slot.outcome, AllocateOutcome::kSuccessAllocated);
  uint64_t bytes = 0;
  ASSERT_TRUE(backend.Commit(slot.slot_id, key, bytes));
}

// Captures the actual wire request. The real Rust backend is tested separately
// below; this receiver makes failures after commit and deadlines reproducible.
class Receiver : public pb::KVIndexer::Service {
 public:
  grpc::Status ApplyExternalKvBatch(grpc::ServerContext* context,
                                    const pb::ApplyExternalKvBatchRequest* request,
                                    pb::ApplyExternalKvBatchResponse*) override {
    std::unique_lock<std::mutex> lock(mutex);
    requests.push_back(*request);
    for (const auto& action : request->actions()) {
      auto& tier = state[action.tier()];
      if (action.type() == pb::ACTION_CLEAR_ALL_AT_TIER) tier.clear();
      for (int i = 0; i < action.hashes_size(); ++i) {
        if (action.type() == pb::ACTION_REVOKE) tier.erase(action.hashes(i));
        if (action.type() == pb::ACTION_REPORT) tier[action.hashes(i)] = action.component_masks(i);
      }
    }
    const bool fail = fail_next_report && request->actions(0).type() == pb::ACTION_REPORT;
    if (fail) fail_next_report = false;
    const bool hang = hang_next;
    hang_next = false;
    lock.unlock();
    if (hang) {
      while (!context->IsCancelled()) std::this_thread::sleep_for(1ms);
      return {grpc::StatusCode::CANCELLED, "test deadline"};
    }
    if (fail) return {grpc::StatusCode::UNAVAILABLE, "test failure after commit"};
    return grpc::Status::OK;
  }
  uint32_t Mask(int64_t hash, pb::TierType tier = pb::TIER_DRAM) {
    std::lock_guard<std::mutex> lock(mutex);
    auto found = state[tier].find(hash);
    return found == state[tier].end() ? 0 : found->second;
  }
  std::mutex mutex;
  std::map<int, std::map<int64_t, uint32_t>> state;
  std::vector<pb::ApplyExternalKvBatchRequest> requests;
  bool fail_next_report = false;
  bool hang_next = false;
};

class ExporterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&receiver);
    server = builder.BuildAndStart();
    ASSERT_NE(server, nullptr);
    endpoint = "127.0.0.1:" + std::to_string(port);
  }
  void TearDown() override { server->Shutdown(); }
  Receiver receiver;
  std::unique_ptr<grpc::Server> server;
  std::string endpoint;
};

TEST_F(ExporterTest, CommittedComponentsEvictionAndHeartbeatOutbox) {
  auto first = Backend(), second = Backend();
  auto config = Config(endpoint, 7);
  auto* block = Block(config, -9007199254740993LL, {"full-rank0", "full-rank1"});
  for (const auto& [bit, key] : std::map<uint32_t, std::string>{{2, "swa"}, {4, "mamba"}}) {
    auto* component = block->add_components();
    component->set_component(bit);
    component->add_keys(key);
  }
  KvIndexerExporter exporter(config, [&] {
    auto all = first->SnapshotOwnedKeys();
    const auto other = second->SnapshotOwnedKeys();
    all.insert(all.end(), other.begin(), other.end());
    return all;
  });
  auto slot = first->Allocate("full-rank0", 1024);
  ASSERT_EQ(slot.outcome, AllocateOutcome::kSuccessAllocated);
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(receiver.Mask(block->hash()), 0u);
  uint64_t bytes;
  ASSERT_TRUE(first->Commit(slot.slot_id, "full-rank0", bytes));
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(receiver.Mask(block->hash()), 0u);
  Commit(*second, "full-rank1");
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(receiver.Mask(block->hash()), 1u);
  Commit(*first, "swa");
  Commit(*second, "mamba");
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(receiver.Mask(block->hash()), 7u);
  ASSERT_GT(first->Evict({"swa"})[0].bytes_freed, 0u);
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(receiver.Mask(block->hash()), 5u);  // REPORT replaces, not ORs.
  Commit(*first, "full-rank1");                 // Two physical copies still mean one component.
  ASSERT_GT(second->Evict({"full-rank1"})[0].bytes_freed, 0u);
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(receiver.Mask(block->hash()), 5u);
  first->ClearLocal();
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(receiver.Mask(block->hash()), 0u);
  EXPECT_TRUE(first->IsClearFullSyncPending());
  // Exporting must not drain events belonging to MasterClient's heartbeat.
  auto events = second->DrainPendingEvents();
  ASSERT_EQ(events.size(), 3u);
  EXPECT_EQ(events.back().kind, KvEvent::Kind::REMOVE);
  first->ClearFullSyncAcked();  // Normally acknowledged by the master heartbeat.
  Commit(*first, "full-rank0");
  Commit(*first, "full-rank1");
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(receiver.Mask(block->hash()), 5u);
  ASSERT_TRUE(exporter.Stop().ok());
  EXPECT_EQ(receiver.Mask(block->hash()), 0u);
}

TEST_F(ExporterTest, FailureAfterCommitResamplesInsteadOfReplayingOldReport) {
  auto backend = Backend();
  auto config = Config(endpoint);
  Block(config, 1, {"a"});
  KvIndexerExporter exporter(config, [&] { return backend->SnapshotOwnedKeys(); });
  Commit(*backend, "a");
  receiver.fail_next_report = true;
  EXPECT_EQ(exporter.SyncOnce().error_code(), grpc::StatusCode::UNAVAILABLE);
  EXPECT_EQ(receiver.Mask(1), 1u);  // Server committed before the reply failed.
  backend->ClearLocal();
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(receiver.Mask(1), 0u);
}

TEST_F(ExporterTest, PeriodicFullSyncRestoresLostStateAndPreservesHbm) {
  auto config = Config(endpoint);
  config.set_full_sync_interval_ms(1);
  Block(config, 1, {"a"});
  KvIndexerExporter exporter(
      config, [] { return std::vector<KvEvent>{{KvEvent::Kind::ADD, "a", TierType::DRAM}}; });
  ASSERT_TRUE(exporter.SyncOnce().ok());
  {
    std::lock_guard<std::mutex> lock(receiver.mutex);
    receiver.state.clear();  // Indexer restart.
    receiver.state[pb::TIER_HBM][99] = 1;
  }
  std::this_thread::sleep_for(2ms);
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(receiver.Mask(1), 1u);
  EXPECT_EQ(receiver.Mask(99, pb::TIER_HBM), 1u);
  ASSERT_TRUE(exporter.Stop().ok());
  EXPECT_EQ(receiver.Mask(99, pb::TIER_HBM), 1u);
}

TEST_F(ExporterTest, BatchesRespectLimitsAndCarryCanonicalMetadata) {
  auto config = Config(endpoint);
  for (int i = 0; i < 600; ++i) {
    auto* block = Block(config, i - 300, {"a"});
    if (i) block->set_parent_hash(i - 301);
  }
  KvIndexerExporter exporter(
      config, [] { return std::vector<KvEvent>{{KvEvent::Kind::ADD, "a", TierType::DRAM}}; });
  ASSERT_TRUE(exporter.SyncOnce().ok());
  std::lock_guard<std::mutex> lock(receiver.mutex);
  ASSERT_EQ(receiver.requests.size(), 4u);  // clear + 256 + 256 + 88.
  uint64_t previous = 0;
  size_t reported = 0;
  for (const auto& request : receiver.requests) {
    EXPECT_GT(request.seq(), previous);
    previous = request.seq();
    EXPECT_EQ(request.worker_id(), config.worker_id());
    EXPECT_EQ(request.worker_address(), config.worker_address());
    EXPECT_EQ(request.cache_spec().SerializeAsString(), config.cache_spec().SerializeAsString());
    EXPECT_LE(request.actions_size(), 256);
    for (const auto& action : request.actions()) {
      EXPECT_EQ(action.tier(), pb::TIER_DRAM);
      if (action.type() != pb::ACTION_REPORT) continue;
      ++reported;
      EXPECT_EQ(action.hashes_size(), 1);
      EXPECT_EQ(action.component_masks(0), 1u);
      EXPECT_EQ(action.block_sizes(0), 16u);
      EXPECT_EQ(action.has_parent_block_hash(), action.hashes(0) != -300);
      if (action.has_parent_block_hash())
        EXPECT_EQ(action.parent_block_hash(), action.hashes(0) - 1);
    }
  }
  EXPECT_EQ(reported, 600u);
}

TEST_F(ExporterTest, SnapshotExceptionWithdrawsStalePlacements) {
  auto config = Config(endpoint);
  Block(config, 1, {"a"});
  bool fail = false;
  KvIndexerExporter exporter(config, [&] {
    if (fail) throw std::runtime_error("snapshot unavailable");
    return std::vector<KvEvent>{{KvEvent::Kind::ADD, "a", TierType::DRAM},
                                {KvEvent::Kind::ADD, "ssd-only", TierType::SSD}};
  });
  ASSERT_TRUE(exporter.SyncOnce().ok());
  fail = true;
  EXPECT_EQ(exporter.SyncOnce().error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(receiver.Mask(1), 0u);
}

TEST_F(ExporterTest, SsdAndHbmDoNotCountAsLocalDram) {
  auto config = Config(endpoint);
  Block(config, 1, {"a"});
  KvIndexerExporter exporter(config, [] {
    return std::vector<KvEvent>{{KvEvent::Kind::ADD, "a", TierType::SSD},
                                {KvEvent::Kind::ADD, "a", TierType::HBM},
                                {KvEvent::Kind::REMOVE, "a", TierType::DRAM}};
  });
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(receiver.Mask(1), 0u);
}

TEST_F(ExporterTest, DeadlineAndStopAreBoundedAndIdempotent) {
  auto config = Config(endpoint);
  config.set_rpc_timeout_ms(30);
  config.set_poll_interval_ms(60000);
  receiver.hang_next = true;
  KvIndexerExporter exporter(config, [] { return std::vector<KvEvent>{}; });
  const auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(exporter.SyncOnce().error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
  exporter.Start();
  ASSERT_TRUE(exporter.Stop().ok());
  ASSERT_TRUE(exporter.Stop().ok());
  EXPECT_LT(std::chrono::steady_clock::now() - start, 2s);
}

class ManifestFile {
 public:
  ManifestFile() {
    char name[] = "/tmp/mori-indexer-test-XXXXXX";
    const int fd = mkstemp(name);
    if (fd < 0) throw std::runtime_error("mkstemp failed");
    close(fd);
    path = name;
  }
  ~ManifestFile() { std::filesystem::remove(path); }
  void Write(const KvIndexerExporter::Config& config) {
    std::string json;
    ASSERT_TRUE(google::protobuf::util::MessageToJsonString(config, &json).ok());
    {
      std::ofstream stream(path + ".tmp");
      stream << json;
    }
    std::filesystem::rename(path + ".tmp", path);
  }
  std::string path;
};

TEST_F(ExporterTest, ManifestReloadAndInvalidIdentityFailClosed) {
  auto config = Config(endpoint);
  config.set_poll_interval_ms(60000);
  Block(config, 9007199254740993LL, {"a"});
  ManifestFile file;
  file.Write(config);
  EXPECT_EQ(KvIndexerExporter::LoadConfig(file.path).blocks(0).hash(), 9007199254740993LL);
  KvIndexerExporter exporter(
      config, [] { return std::vector<KvEvent>{{KvEvent::Kind::ADD, "a", TierType::DRAM}}; });
  exporter.Start(file.path);
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(receiver.Mask(9007199254740993LL), 1u);
  config.mutable_blocks(0)->mutable_components(0)->set_keys(0, "missing");
  file.Write(config);
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(receiver.Mask(9007199254740993LL), 0u);
  config.mutable_blocks(0)->set_block_size(32);
  file.Write(config);
  EXPECT_EQ(exporter.SyncOnce().error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  config.mutable_blocks(0)->set_block_size(16);
  config.mutable_blocks(0)->mutable_components(0)->set_keys(0, "a");
  file.Write(config);
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(receiver.Mask(9007199254740993LL), 1u);
  config.set_worker_id("different-worker");
  file.Write(config);
  EXPECT_EQ(exporter.SyncOnce().error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(receiver.Mask(9007199254740993LL), 0u);
}

TEST(KvIndexerConfig, RejectsIncompleteOrCyclicMappings) {
  auto config = Config("localhost:50051", 3);
  Block(config, 1, {"a"});
  EXPECT_THROW(KvIndexerExporter::Validate(config), std::invalid_argument);
  config.mutable_cache_spec()->set_components(1);
  EXPECT_NO_THROW(KvIndexerExporter::Validate(config));
  config.mutable_blocks(0)->set_parent_hash(2);
  EXPECT_THROW(KvIndexerExporter::Validate(config), std::invalid_argument);
  Block(config, 2, {"b"})->set_parent_hash(1);
  EXPECT_THROW(KvIndexerExporter::Validate(config), std::invalid_argument);
  config.mutable_blocks(0)->clear_parent_hash();
  EXPECT_NO_THROW(KvIndexerExporter::Validate(config));
  Block(config, 2, {"duplicate"});
  EXPECT_THROW(KvIndexerExporter::Validate(config), std::invalid_argument);
}

class ExporterEnvironment {
 public:
  explicit ExporterEnvironment(const std::string& path) {
    if (const char* old = std::getenv("UMBP_KV_INDEXER_CONFIG")) {
      had_value_ = true;
      old_ = old;
    }
    setenv("UMBP_KV_INDEXER_CONFIG", path.c_str(), 1);
  }
  ~ExporterEnvironment() {
    if (had_value_)
      setenv("UMBP_KV_INDEXER_CONFIG", old_.c_str(), 1);
    else
      unsetenv("UMBP_KV_INDEXER_CONFIG");
  }

 private:
  bool had_value_ = false;
  std::string old_;
};

TEST_F(ExporterTest, PoolClientOptInPutGetClearAndShutdown) {
  auto config = Config(endpoint);
  config.set_poll_interval_ms(5);
  Block(config, 1, {"object"});
  ManifestFile file;
  file.Write(config);
  ExporterEnvironment env(file.path);
  PoolClientConfig pool_config;
  pool_config.master_config.node_id = "indexer-local-test";
  pool_config.dram_page_size = 4096;
  pool_config.dram.buffer_sizes = {16 * 4096};
  PoolClient pool(pool_config);
  ASSERT_TRUE(pool.Init());
  auto wait_mask = [&](uint32_t expected) {
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    while (receiver.Mask(1) != expected && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(5ms);
    EXPECT_EQ(receiver.Mask(1), expected);
  };
  std::vector<char> input(4096, 'k'), output(4096);
  ASSERT_TRUE(pool.Put("object", input.data(), input.size()));
  wait_mask(1);
  ASSERT_TRUE(pool.Get("object", output.data(), output.size()));
  EXPECT_EQ(input, output);
  ASSERT_TRUE(pool.Clear());
  wait_mask(0);
  ASSERT_TRUE(pool.Put("object", input.data(), input.size()));
  wait_mask(1);
  pool.Shutdown();
  EXPECT_EQ(receiver.Mask(1), 0u);
  EXPECT_FALSE(pool.IsInitialized());
  // Reinitializing the same PoolClient must create a fresh exporter/registry.
  ASSERT_TRUE(pool.Init());
  ASSERT_TRUE(pool.Put("object", input.data(), input.size()));
  wait_mask(1);
  pool.Shutdown();
  EXPECT_EQ(receiver.Mask(1), 0u);
}

TEST_F(ExporterTest, PoolClientInvalidConfigurationFailsInitialization) {
  ManifestFile file;  // Empty file is not a valid exporter config.
  ExporterEnvironment env(file.path);
  PoolClientConfig config;
  config.master_config.node_id = "indexer-invalid-test";
  config.dram_page_size = 4096;
  config.dram.buffer_sizes = {16 * 4096};
  PoolClient pool(config);
  EXPECT_FALSE(pool.Init());
  EXPECT_FALSE(pool.IsInitialized());
  std::lock_guard<std::mutex> lock(receiver.mutex);
  EXPECT_TRUE(receiver.requests.empty());
}

TEST_F(ExporterTest, PoolClientDisabledHasNoExporterTraffic) {
  ExporterEnvironment env("");
  PoolClientConfig config;
  config.master_config.node_id = "indexer-disabled-test";
  config.dram_page_size = 4096;
  config.dram.buffer_sizes = {16 * 4096};
  PoolClient pool(config);
  ASSERT_TRUE(pool.Init());
  std::vector<char> data(4096, 'k');
  ASSERT_TRUE(pool.Put("object", data.data(), data.size()));
  pool.Shutdown();
  std::lock_guard<std::mutex> lock(receiver.mutex);
  EXPECT_TRUE(receiver.requests.empty());
}

// Optional interoperability test. Run against an isolated, unmodified pinned
// sgl-kv-indexer server; never point this at a production Indexer.
TEST(KvIndexerInterop, RealServerComponentPrefixEvictionAndRecovery) {
  const char* endpoint = std::getenv("UMBP_TEST_KV_INDEXER_ENDPOINT");
  if (!endpoint || !*endpoint) GTEST_SKIP() << "set UMBP_TEST_KV_INDEXER_ENDPOINT";
  auto config = Config(endpoint, 7);
  config.set_worker_id("mori-interop-" + std::to_string(getpid()));
  for (int i = 1; i <= 3; ++i) {
    auto* block = Block(config, -i, {"full" + std::to_string(i)});
    if (i > 1) block->set_parent_hash(-(i - 1));
    for (uint32_t bit : {2u, 4u}) {
      auto* component = block->add_components();
      component->set_component(bit);
      component->add_keys(std::to_string(bit) + "-" + std::to_string(i));
    }
  }
  auto backend = Backend();
  KvIndexerExporter exporter(config, [&] { return backend->SnapshotOwnedKeys(); });
  auto stub =
      pb::KVIndexer::NewStub(grpc::CreateChannel(endpoint, grpc::InsecureChannelCredentials()));
  auto prefix = [&] {
    pb::MatchExternalKvPrefixRequest request;
    for (int64_t h : {-1, -2, -3}) request.add_hashes(h);
    pb::MatchExternalKvPrefixResponse response;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 2s);
    auto status = stub->MatchExternalKvPrefix(&context, request, &response);
    EXPECT_TRUE(status.ok()) << status.error_message();
    for (const auto& match : response.matches()) {
      if (match.worker_id() == config.worker_id()) {
        EXPECT_EQ(match.worker_address(), config.worker_address());
        return match.matched_prefix_blocks();
      }
    }
    return 0u;
  };
  for (int i = 1; i <= 3; ++i) Commit(*backend, "full" + std::to_string(i));
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(prefix(), 0u);  // FULL alone is not sufficient for this worker.
  Commit(*backend, "2-2");
  Commit(*backend, "2-3");
  Commit(*backend, "4-3");
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(prefix(), 3u);  // SWA trailing 32 tokens; MAMBA boundary only.
  ASSERT_GT(backend->Evict({"2-2"})[0].bytes_freed, 0u);
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(prefix(), 0u);
  Commit(*backend, "2-2");
  ASSERT_TRUE(exporter.SyncOnce(true).ok());
  EXPECT_EQ(prefix(), 3u);
  ASSERT_GT(backend->Evict({"full2"})[0].bytes_freed, 0u);
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(prefix(), 0u);
  Commit(*backend, "full2");
  ASSERT_TRUE(exporter.SyncOnce().ok());
  EXPECT_EQ(prefix(), 3u);
  ASSERT_TRUE(exporter.Stop().ok());
  EXPECT_EQ(prefix(), 0u);
}
}  // namespace
}  // namespace mori::umbp
