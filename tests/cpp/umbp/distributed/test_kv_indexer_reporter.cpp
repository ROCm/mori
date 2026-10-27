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
#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <thread>

#include "umbp/distributed/indexer/kv_indexer_reporter.h"
#include "umbp/distributed/pool_client.h"
#include "umbp/local/host_mem_allocator.h"
#include "umbp/standalone/standalone_server.h"
#include "umbp/umbp_client.h"

namespace mori::umbp {
namespace {
namespace pb = ::kv_indexer::v1;
using namespace std::chrono_literals;

class Receiver : public pb::KVIndexer::Service {
 public:
  grpc::Status ApplyExternalKvBatch(grpc::ServerContext* context,
                                    const pb::ApplyExternalKvBatchRequest* request,
                                    pb::ApplyExternalKvBatchResponse*) override {
    const int fault = mode.load();
    if (fault == 3) mode = 1;  // One successful batch, then apply-and-fail subsequent batches.
    {
      std::lock_guard<std::mutex> lock(mutex);
      requests.push_back(*request);
      if (fault == 1) ++failures;
      for (const auto& action : request->actions()) {
        if (action.type() == pb::ACTION_CLEAR_ALL_AT_TIER) masks.clear();
        for (int i = 0; i < action.hashes_size(); ++i) {
          if (action.type() == pb::ACTION_REPORT)
            masks[action.hashes(i)] = action.component_masks(i);
          else
            masks.erase(action.hashes(i));
        }
      }
    }
    changed.notify_all();
    if (fault == 2) {
      while (!context->IsCancelled()) std::this_thread::sleep_for(1ms);
    }
    return fault == 1 ? grpc::Status(grpc::StatusCode::UNAVAILABLE, "injected") : grpc::Status::OK;
  }
  bool WaitFor(const std::map<int64_t, uint32_t>& expected) {
    std::unique_lock<std::mutex> lock(mutex);
    return changed.wait_for(lock, 3s, [&] { return masks == expected; });
  }
  bool WaitForFailure() {
    std::unique_lock<std::mutex> lock(mutex);
    return changed.wait_for(lock, 3s, [&] { return failures > 0; });
  }
  std::mutex mutex;
  std::condition_variable changed;
  std::map<int64_t, uint32_t> masks;
  std::atomic<int> mode{0};
  size_t failures = 0;
  std::vector<pb::ApplyExternalKvBatchRequest> requests;
};

class AutomaticKvIndexer : public ::testing::Test {
 protected:
  static PoolClientConfig PoolConfig(size_t pages = 16) {
    PoolClientConfig cfg;
    cfg.master_config.node_id = "indexer-test";
    cfg.dram_page_size = 4096;
    cfg.dram.buffer_sizes = {pages * 4096};
    return cfg;
  }
  void SetUp() override {
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&receiver);
    server = builder.BuildAndStart();
    ASSERT_NE(server, nullptr);
    config = {"127.0.0.1:" + std::to_string(port), "worker", "http://worker:30000", 1, 0, 50};
  }
  void TearDown() override { server->Shutdown(); }
  Receiver receiver;
  std::unique_ptr<grpc::Server> server;
  KvIndexerConfig config;
};

TEST_F(AutomaticKvIndexer, BatchesRespectWireLimitsAndPreserveCanonicalMetadata) {
  PoolClient pool(PoolConfig(2));
  ASSERT_TRUE(pool.Init());
  pool.ConfigureKvIndexer(config);
  std::map<int64_t, uint32_t> expected;
  const int64_t root = -9007199254740993LL;
  for (int i = 0; i < 300; ++i) {
    pool.RegisterKvBlock(
        {root + i, i ? std::optional<int64_t>(root + i - 1) : std::nullopt, 16, {{1, {"shared"}}}});
    expected[root + i] = 1;
  }
  ASSERT_TRUE(pool.Put("shared", "data", 4));
  ASSERT_TRUE(receiver.WaitFor(expected));
  pool.Shutdown();
  size_t reported = 0, batches = 0;
  std::lock_guard<std::mutex> lock(receiver.mutex);
  for (const auto& request : receiver.requests) {
    EXPECT_EQ(request.worker_id(), config.worker_id);
    EXPECT_EQ(request.worker_address(), config.worker_address);
    EXPECT_EQ(request.cache_spec().full_tier_mask(), 6u);
    EXPECT_LE(request.actions_size(), 256);
    bool has_report = false;
    for (const auto& action : request.actions()) {
      EXPECT_EQ(action.tier(), pb::TIER_DRAM);
      if (action.type() != pb::ACTION_REPORT) continue;
      has_report = true;
      EXPECT_EQ(action.hashes(0), root + reported);
      EXPECT_EQ(action.has_parent_block_hash(), reported != 0);
      if (reported) EXPECT_EQ(action.parent_block_hash(), root + reported - 1);
      EXPECT_EQ(action.block_sizes(0), 16u);
      EXPECT_EQ(action.component_masks(0), 1u);
      ++reported;
    }
    batches += has_report;
  }
  EXPECT_EQ(reported, 300u);
  EXPECT_EQ(batches, 2u);
}

TEST_F(AutomaticKvIndexer, CommittedComponentsEvictionClearAndRecovery) {
  PoolClient pool(PoolConfig());
  ASSERT_TRUE(pool.Init());
  EXPECT_THROW(pool.RegisterKvBlock({1, {}, 16, {{1, {"k"}}}}), std::logic_error);
  config.components = 7;
  config.swa_window_tokens = 32;
  pool.ConfigureKvIndexer(config);
  EXPECT_THROW(pool.ConfigureKvIndexer(config), std::logic_error);
  auto* backend = pool.Backends().Get(TierType::DRAM);
  backend->SetEventPublishing(true);
  const KvBlockMetadata block{
      -9007199254740993LL, {}, 16, {{1, {"k", "v"}}, {2, {"swa"}}, {4, {"mamba"}}}};
  pool.RegisterKvBlock(block);
  EXPECT_THROW(pool.RegisterKvBlock({1, {}, 16, {{1, {"k"}}}}), std::invalid_argument);
  EXPECT_THROW(pool.RegisterKvBlock({block.hash, 42, 16, block.component_keys}),
               std::invalid_argument);
  pool.RegisterKvBlock({10, 11, 16, block.component_keys});
  EXPECT_THROW(pool.RegisterKvBlock({11, 10, 16, block.component_keys}), std::invalid_argument);
  pool.RegisterKvBlock({10, {}, 0, {}});
  pool.RegisterKvBlock({11, {}, 0, {}});
  const std::string data(128, 'x');
  ASSERT_TRUE(pool.Put("k", data.data(), data.size()));
  auto pending = backend->BatchAllocate({{"v", data.size()}});
  ASSERT_EQ(pending[0].outcome, AllocateOutcome::kSuccessAllocated);
  EXPECT_FALSE(backend->BatchCommit({{pending[0].slot_id + 1000, "v"}})[0].success);
  ASSERT_TRUE(pool.FlushKvIndexer());
  EXPECT_TRUE(receiver.WaitFor({}));  // Neither pending nor failed commits advertise FULL.
  ASSERT_TRUE(backend->BatchCommit({{pending[0].slot_id, "v"}})[0].success);
  ASSERT_TRUE(receiver.WaitFor({{block.hash, 1}}));  // No flush: actual commit wakes the worker.
  ASSERT_EQ(pool.BatchPut({"swa", "mamba"}, {data.data(), data.data()}, {128, 128}),
            (std::vector<bool>{true, true}));
  ASSERT_TRUE(receiver.WaitFor({{block.hash, 7}}));
  ASSERT_GT(backend->Evict({"swa"})[0].bytes_freed, 0u);
  ASSERT_TRUE(receiver.WaitFor({{block.hash, 5}}));
  std::string copy(128, '\0');
  ASSERT_TRUE(pool.Get("k", copy.data(), copy.size()));
  EXPECT_EQ(copy, data);
  auto held = backend->BatchResolve({"k"}, false, false, ResolveReader::kLocal);
  EXPECT_EQ(backend->Evict({"k"})[0].bytes_freed, 0u);  // An active read protects it.
  ASSERT_TRUE(pool.FlushKvIndexer());
  EXPECT_TRUE(receiver.WaitFor({{block.hash, 5}}));
  held.clear();
  EXPECT_FALSE(backend->DrainPendingEvents().empty());  // Indexer leaves master outbox intact.
  ASSERT_TRUE(pool.Clear());
  ASSERT_TRUE(receiver.WaitFor({}));
  ASSERT_TRUE(pool.Put("k", data.data(), data.size()));
  ASSERT_TRUE(pool.Put("v", data.data(), data.size()));
  ASSERT_TRUE(receiver.WaitFor({{block.hash, 1}}));
  pool.RegisterKvBlock({block.hash, {}, 0, {}});  // Metadata removal withdraws existing residency.
  ASSERT_TRUE(receiver.WaitFor({}));

  receiver.mode = 1;  // Apply then fail: recovery must resample, not replay the stale report.
  pool.RegisterKvBlock(block);
  EXPECT_FALSE(pool.FlushKvIndexer());
  ASSERT_GT(backend->Evict({"v"})[0].bytes_freed, 0u);
  ASSERT_TRUE(pool.Put("unregistered", data.data(), data.size()));
  receiver.mode = 0;
  ASSERT_TRUE(receiver.WaitFor({}));
  ASSERT_TRUE(pool.FlushKvIndexer());
  ASSERT_TRUE(pool.Put("v", data.data(), data.size()));
  ASSERT_TRUE(receiver.WaitFor({{block.hash, 1}}));
  receiver.mode = 2;
  pool.RegisterKvBlock({block.hash, {}, 0, {}});
  EXPECT_FALSE(pool.FlushKvIndexer());  // A stalled RPC is bounded by the configured deadline.
  receiver.mode = 0;
  pool.Shutdown();
  EXPECT_TRUE(receiver.WaitFor({}));
}

TEST_F(AutomaticKvIndexer, LocalWatermarkEvictionWithdrawsWithoutCallerEvents) {
  auto cfg = PoolConfig(8);
  cfg.local_evict_high_watermark = 0.75;
  cfg.local_evict_low_watermark = 0.5;
  PoolClient pool(cfg);
  ASSERT_TRUE(pool.Init());
  pool.ConfigureKvIndexer(config);
  const std::string data(4096, 'x');
  for (int64_t i = 0; i < 7; ++i) {
    const auto key = std::to_string(i);
    pool.RegisterKvBlock({i, {}, 16, {{1, {key}}}});
    ASSERT_TRUE(pool.Put(key, data.data(), data.size()));
  }
  const auto deadline = std::chrono::steady_clock::now() + 3s;
  auto* backend = pool.Backends().Get(TierType::DRAM);
  while (backend->SnapshotOwnedKeys().size() == 7 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(10ms);
  std::map<int64_t, uint32_t> expected;
  for (const auto& event : backend->SnapshotOwnedKeys()) expected[std::stoll(event.key)] = 1;
  ASSERT_FALSE(expected.empty());
  ASSERT_LT(expected.size(), 7u);
  EXPECT_TRUE(receiver.WaitFor(expected));
}

TEST_F(AutomaticKvIndexer, PartialBatchFailureResamplesAfterEviction) {
  PoolClient pool(PoolConfig(2));
  ASSERT_TRUE(pool.Init());
  pool.ConfigureKvIndexer(config);
  std::map<int64_t, uint32_t> expected;
  for (int64_t hash = 0; hash < 300; ++hash) {
    pool.RegisterKvBlock({hash, {}, 16, {{1, {"shared"}}}});
    expected[hash] = 1;
  }
  ASSERT_TRUE(pool.FlushKvIndexer());  // Finish startup before injecting a mid-report failure.
  receiver.mode = 3;
  ASSERT_TRUE(pool.Put("shared", "data", 4));
  ASSERT_TRUE(receiver.WaitForFailure());
  ASSERT_GT(pool.Backends().Get(TierType::DRAM)->Evict({"shared"})[0].bytes_freed, 0u);
  receiver.mode = 0;
  ASSERT_TRUE(pool.FlushKvIndexer());
  EXPECT_TRUE(receiver.WaitFor({}));  // Neither successful nor ambiguous batches may be replayed.
  ASSERT_TRUE(pool.Put("shared", "data", 4));
  EXPECT_TRUE(receiver.WaitFor(expected));
}

TEST_F(AutomaticKvIndexer, StandaloneForwardsMetadataToTheStorageOwner) {
  const std::string address = "unix:///tmp/umbp_indexer_" + std::to_string(getpid()) + ".sock";
  UMBPConfig cfg;
  cfg.dram.capacity_bytes = 1 << 20;
  cfg.ssd.enabled = false;
  cfg.standalone_process = UMBPStandaloneProcessConfig{};
  cfg.standalone_process->address = address;
  standalone::StandaloneServer owner(cfg, address);
  ASSERT_TRUE(owner.Start());
  config.metadata_provider = [](const auto&) {
    return std::vector<KvBlockMetadata>{{42, {}, 16, {{1, {"key"}}}}};
  };
  cfg.kv_indexer = config;
  auto client = CreateUMBPClient(cfg);
  EXPECT_THROW(client->RegisterKvBlock({43, {}, 16, {{2, {"key"}}}}), std::invalid_argument);
  HostMemAllocator allocator;
  HostBufferOptions opts;
  opts.backing = HostBufferBacking::kAnonymousShm;
  auto buffer = allocator.Alloc(4096, opts);
  ASSERT_TRUE(buffer.valid());
  const auto ptr = reinterpret_cast<uintptr_t>(buffer.ptr);
  ASSERT_TRUE(client->RegisterMemory(ptr, buffer.mapped_size));
  std::memset(buffer.ptr, 0x5a, 128);
  ASSERT_TRUE(client->Put("key", ptr, 128));
  ASSERT_TRUE(receiver.WaitFor({{42, 1}}));
  ASSERT_TRUE(client->Get("key", ptr + 128, 128));
  EXPECT_EQ(std::memcmp(buffer.ptr, static_cast<char*>(buffer.ptr) + 128, 128), 0);
  ASSERT_TRUE(client->Clear());
  ASSERT_TRUE(receiver.WaitFor({}));
  ASSERT_TRUE(client->Put("key", ptr, 128));
  ASSERT_TRUE(receiver.WaitFor({{42, 1}}));
  client->DeregisterMemory(ptr);
  client->Close();
  EXPECT_TRUE(receiver.WaitFor({{42, 1}}));  // Forwarder close must not clear the shared owner.
  EXPECT_THROW(client->RegisterKvBlock({42, {}, 0, {}}), std::logic_error);
  allocator.Free(buffer);
  owner.Shutdown();
  EXPECT_TRUE(receiver.WaitFor({}));
}

TEST_F(AutomaticKvIndexer, ClientConfigControlsAutomaticMetadataAndReporting) {
  UMBPConfig cfg;
  cfg.dram.capacity_bytes = 1 << 20;
  cfg.ssd.enabled = false;
  cfg.page_size = 4096;
  const std::string data = "value";
  const auto ptr = reinterpret_cast<uintptr_t>(data.data());
  auto client = CreateUMBPClient(cfg);
  ASSERT_TRUE(client->Put("key", ptr, data.size()));
  ASSERT_TRUE(client->Flush());
  {
    std::lock_guard<std::mutex> lock(receiver.mutex);
    EXPECT_TRUE(receiver.requests.empty());
  }
  client->Close();

  config.metadata_provider = [](const auto& keys) {
    EXPECT_EQ(keys, (std::vector<std::string>{"key"}));
    return std::vector<KvBlockMetadata>{{42, {}, 16, {{1, {"key"}}}}};
  };
  cfg.kv_indexer = config;
  client = CreateUMBPClient(cfg);
  ASSERT_TRUE(client->Put("key", ptr, data.size()));
  EXPECT_TRUE(receiver.WaitFor({{42, 1}}));
  client->Close();
  EXPECT_TRUE(receiver.WaitFor({}));
}

TEST_F(AutomaticKvIndexer, WriteNotificationsDoNotBypassRetryBackoff) {
  receiver.mode = 1;
  std::atomic<bool> resident{true};
  KvIndexerReporter reporter(config, [&] {
    return resident ? std::unordered_set<std::string>{"key"} : std::unordered_set<std::string>{};
  });
  reporter.Register({42, {}, 16, {{1, {"key"}}}});
  ASSERT_TRUE(receiver.WaitForFailure());
  for (int i = 0; i < 20; ++i) {
    reporter.Notify();
    std::this_thread::sleep_for(5ms);
  }
  {
    std::lock_guard<std::mutex> lock(receiver.mutex);
    EXPECT_EQ(receiver.failures, 1u);
  }
  resident = false;
  receiver.mode = 0;
  reporter.Notify();
  ASSERT_TRUE(receiver.WaitFor({}));
  resident = true;
  reporter.Notify();
  EXPECT_TRUE(receiver.WaitFor({{42, 1}}));
}
}  // namespace
}  // namespace mori::umbp
