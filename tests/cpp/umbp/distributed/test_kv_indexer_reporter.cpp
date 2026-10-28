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

#include "umbp/distributed/indexer/kv_indexer_reporter.h"
#include "umbp/distributed/pool_client.h"

namespace mori::umbp {
namespace {
namespace pb = ::kv_indexer::v1;

class Receiver : public pb::KVIndexer::Service {
 public:
  grpc::Status ApplyExternalKvBatch(grpc::ServerContext*,
                                    const pb::ApplyExternalKvBatchRequest* request,
                                    pb::ApplyExternalKvBatchResponse*) override {
    std::lock_guard<std::mutex> lock(mutex);
    requests.push_back(*request);
    for (const auto& action : request->actions()) {
      if (action.type() == pb::ACTION_CLEAR_ALL_AT_TIER) masks.clear();
      for (int i = 0; i < action.hashes_size(); ++i) {
        if (action.type() == pb::ACTION_REPORT)
          masks[action.hashes(i)] = action.component_masks(i);
        else
          masks.erase(action.hashes(i));
      }
    }
    changed.notify_all();
    return fail ? grpc::Status(grpc::StatusCode::UNAVAILABLE, "applied then failed")
                : grpc::Status::OK;
  }
  bool WaitFor(const std::map<int64_t, uint32_t>& expected) {
    std::unique_lock<std::mutex> lock(mutex);
    return changed.wait_for(lock, std::chrono::seconds(3), [&] { return masks == expected; });
  }
  std::mutex mutex;
  std::condition_variable changed;
  std::map<int64_t, uint32_t> masks;
  std::atomic<bool> fail{false};
  std::vector<pb::ApplyExternalKvBatchRequest> requests;
};

class AutomaticKvIndexer : public ::testing::Test {
 protected:
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

TEST_F(AutomaticKvIndexer, SharedPoolReportsCommittedComponentsAndWithdraws) {
  config.worker_id.clear();
  config.worker_address.clear();
  config.dram_pool_id = "shared";
  config.components = 7;
  config.swa_window_tokens = 32;
  PoolClientConfig cfg;
  cfg.master_config.node_id = "indexer-test";
  cfg.dram_page_size = 4096;
  cfg.dram.buffer_sizes = {16 * 4096};
  PoolClient pool(cfg);
  ASSERT_TRUE(pool.Init());
  pool.ConfigureKvIndexer(config);
  EXPECT_NO_THROW(pool.ConfigureKvIndexer(config));
  auto conflict = config;
  conflict.dram_pool_id = "other";
  EXPECT_THROW(pool.ConfigureKvIndexer(conflict), std::logic_error);
  const KvBlockMetadata block{42, {}, 16, {{1, {"k", "v"}}, {2, {"swa"}}, {4, {"mamba"}}}};
  pool.RegisterKvBlock(block);
  EXPECT_NO_THROW(pool.RegisterKvBlock(block));
  EXPECT_THROW(pool.RegisterKvBlock({42, {}, 32, block.component_keys}), std::invalid_argument);
  EXPECT_THROW(pool.RegisterKvBlock({43, 43, 16, block.component_keys}), std::invalid_argument);
  ASSERT_TRUE(pool.Put("k", "data", 4));
  ASSERT_TRUE(pool.FlushKvIndexer());
  ASSERT_TRUE(receiver.WaitFor({}));  // FULL needs both committed keys.
  ASSERT_TRUE(pool.Put("v", "data", 4));
  ASSERT_TRUE(receiver.WaitFor({{42, 1}}));
  ASSERT_EQ(pool.BatchPut({"swa", "mamba"}, {"data", "data"}, {4, 4}),
            (std::vector<bool>{true, true}));
  ASSERT_TRUE(receiver.WaitFor({{42, 7}}));
  ASSERT_GT(pool.Backends().Get(TierType::DRAM)->Evict({"k"})[0].bytes_freed, 0u);
  ASSERT_TRUE(receiver.WaitFor({{42, 6}}));  // Members may supply FULL from HBM.
  ASSERT_TRUE(pool.Clear());
  ASSERT_TRUE(receiver.WaitFor({}));
  ASSERT_TRUE(pool.Put("k", "data", 4));
  ASSERT_TRUE(pool.Put("v", "data", 4));
  ASSERT_TRUE(receiver.WaitFor({{42, 1}}));
  pool.Shutdown();
  ASSERT_TRUE(receiver.WaitFor({}));
  for (const auto& request : receiver.requests) {
    EXPECT_EQ(request.dram_pool_id(), "shared");
    EXPECT_TRUE(request.worker_id().empty());
    for (const auto& action : request.actions()) EXPECT_EQ(action.tier(), pb::TIER_DRAM);
  }
}

TEST_F(AutomaticKvIndexer, RetryWithdrawsAmbiguousReportsAndRepairDoesNotClear) {
  std::atomic<bool> resident{true};
  std::atomic<int> lookups{0};
  KvIndexerReporter reporter(config, [&](const auto&) {
    ++lookups;
    return resident.load();
  });
  ASSERT_TRUE(reporter.Sync());
  receiver.fail = true;
  reporter.Register({42, {}, 16, {{1, {"key"}}}});
  EXPECT_FALSE(reporter.Sync());
  ASSERT_TRUE(receiver.WaitFor({{42, 1}}));
  resident = false;
  receiver.fail = false;
  reporter.Notify({"key"});
  ASSERT_TRUE(reporter.Sync());
  ASSERT_TRUE(receiver.WaitFor({}));
  resident = true;
  reporter.Notify({"key"});
  ASSERT_TRUE(reporter.Sync());
  ASSERT_TRUE(receiver.WaitFor({{42, 1}}));
  lookups = 0;
  reporter.Notify({"unrelated"});
  ASSERT_TRUE(reporter.Sync());
  EXPECT_EQ(lookups, 0);  // A delta does not scan unrelated history.
  {
    std::lock_guard<std::mutex> lock(receiver.mutex);
    receiver.requests.clear();
  }
  ASSERT_TRUE(reporter.Sync(true));
  std::lock_guard<std::mutex> lock(receiver.mutex);
  ASSERT_FALSE(receiver.requests.empty());
  for (const auto& request : receiver.requests)
    for (const auto& action : request.actions())
      EXPECT_NE(action.type(), pb::ACTION_CLEAR_ALL_AT_TIER);
}
}  // namespace
}  // namespace mori::umbp
