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

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <thread>

#include "umbp/distributed/indexer/kv_indexer_reporter.h"

namespace mori::umbp {
namespace {
namespace pb = ::kv_indexer::v1;
using namespace std::chrono_literals;

class Receiver : public pb::KVIndexer::Service {
 public:
  grpc::Status ApplyExternalKvBatch(grpc::ServerContext* context,
                                    const pb::ApplyExternalKvBatchRequest* request,
                                    pb::ApplyExternalKvBatchResponse*) override {
    {
      std::lock_guard<std::mutex> lock(mutex);
      requests.push_back(*request);
    }
    if (hang.load()) {
      while (!context->IsCancelled()) std::this_thread::sleep_for(1ms);
    }
    return fail.load() ? grpc::Status(grpc::StatusCode::UNAVAILABLE, "injected") : grpc::Status::OK;
  }
  std::mutex mutex;
  std::vector<pb::ApplyExternalKvBatchRequest> requests;
  std::atomic<bool> fail{false}, hang{false};
};

class ReporterTest : public ::testing::Test {
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

TEST_F(ReporterTest, ReportsCanonicalChainAndOnlyMutatesDram) {
  KvIndexerReporter reporter(endpoint, "worker", "http://worker:30000", 7, 32);
  const int64_t root = -9007199254740993LL;
  reporter.Report({root}, std::nullopt, 16, 7);
  reporter.Report({42, 43}, root, 16, 3);
  reporter.Revoke({42});
  reporter.Clear();
  std::lock_guard<std::mutex> lock(receiver.mutex);
  ASSERT_EQ(receiver.requests.size(), 4u);
  for (size_t i = 0; i < receiver.requests.size(); ++i) {
    const auto& request = receiver.requests[i];
    EXPECT_EQ(request.worker_id(), "worker");
    EXPECT_EQ(request.worker_address(), "http://worker:30000");
    EXPECT_EQ(request.seq(), i + 1);
    EXPECT_EQ(request.cache_spec().components(), 7u);
    EXPECT_EQ(request.cache_spec().swa_window_tokens(), 32u);
    EXPECT_EQ(request.cache_spec().full_tier_mask(), 6u);
    ASSERT_EQ(request.actions_size(), 1);
    EXPECT_EQ(request.actions(0).tier(), pb::TIER_DRAM);
  }
  const auto& root_action = receiver.requests[0].actions(0);
  EXPECT_FALSE(root_action.has_parent_block_hash());
  EXPECT_EQ(root_action.hashes(0), root);
  const auto& chain = receiver.requests[1].actions(0);
  EXPECT_EQ(chain.type(), pb::ACTION_REPORT);
  EXPECT_EQ(chain.parent_block_hash(), root);
  ASSERT_EQ(chain.hashes_size(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(chain.hashes(i), 42 + i);
    EXPECT_EQ(chain.component_masks(i), 3u);
    EXPECT_EQ(chain.block_sizes(i), 16u);
  }
  EXPECT_EQ(receiver.requests[2].actions(0).type(), pb::ACTION_REVOKE);
  EXPECT_EQ(receiver.requests[2].actions(0).hashes(0), 42);
  EXPECT_EQ(receiver.requests[3].actions(0).type(), pb::ACTION_CLEAR_ALL_AT_TIER);
}

TEST_F(ReporterTest, RejectsInvalidInputsAndIgnoresEmptyBatches) {
  EXPECT_THROW(KvIndexerReporter(endpoint, "worker", "address", 3), std::invalid_argument);
  KvIndexerReporter reporter(endpoint, "worker", "address");
  EXPECT_THROW(reporter.Report({1}, std::nullopt, 0), std::invalid_argument);
  EXPECT_THROW(reporter.Report({1}, std::nullopt, 16, 2), std::invalid_argument);
  EXPECT_THROW(reporter.Report({1}, std::nullopt, 16, 3), std::invalid_argument);
  EXPECT_THROW(reporter.Revoke(std::vector<int64_t>(16385)), std::invalid_argument);
  EXPECT_THROW(reporter.Report(std::vector<int64_t>(16385), std::nullopt, 16),
               std::invalid_argument);
  reporter.Report({}, std::nullopt, 16);
  reporter.Revoke({});
  std::lock_guard<std::mutex> lock(receiver.mutex);
  EXPECT_TRUE(receiver.requests.empty());
}

TEST_F(ReporterTest, SurfacesRpcErrorsAndDeadlineWithoutReplaying) {
  KvIndexerReporter reporter(endpoint, "worker", "address", 1, 0, 30);
  receiver.fail = true;
  EXPECT_THROW(reporter.Report({1}, std::nullopt, 16), std::runtime_error);
  receiver.fail = false;
  receiver.hang = true;
  const auto start = std::chrono::steady_clock::now();
  EXPECT_THROW(reporter.Revoke({1}), std::runtime_error);
  EXPECT_LT(std::chrono::steady_clock::now() - start, 2s);
  receiver.hang = false;
  reporter.Clear();
  std::lock_guard<std::mutex> lock(receiver.mutex);
  ASSERT_EQ(receiver.requests.size(), 3u);  // No retries or implicit clear.
  EXPECT_EQ(receiver.requests.back().actions(0).type(), pb::ACTION_CLEAR_ALL_AT_TIER);
}

TEST(KvIndexerInterop, HybridPrefixComponentReplacementAndRevoke) {
  const char* endpoint = std::getenv("UMBP_TEST_KV_INDEXER_ENDPOINT");
  if (!endpoint || !*endpoint) GTEST_SKIP() << "set UMBP_TEST_KV_INDEXER_ENDPOINT";
  auto stub =
      pb::KVIndexer::NewStub(grpc::CreateChannel(endpoint, grpc::InsecureChannelCredentials()));
  const std::string worker = "mori-reporter-interop";
  KvIndexerReporter reporter(endpoint, worker, "http://worker:30000", 7, 32);
  auto prefix = [&] {
    pb::MatchExternalKvPrefixRequest request;
    for (int64_t hash : {1, 2, 3}) request.add_hashes(hash);
    pb::MatchExternalKvPrefixResponse response;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 2s);
    auto status = stub->MatchExternalKvPrefix(&context, request, &response);
    EXPECT_TRUE(status.ok()) << status.error_message();
    for (const auto& match : response.matches()) {
      if (match.worker_id() == worker) {
        EXPECT_EQ(match.worker_address(), "http://worker:30000");
        return match.matched_prefix_blocks();
      }
    }
    return 0u;
  };
  reporter.Clear();
  reporter.Report({1, 2, 3}, std::nullopt, 16);
  EXPECT_EQ(prefix(), 0u);
  reporter.Report({2}, 1, 16, 3);  // FULL + SWA.
  reporter.Report({3}, 2, 16, 7);  // FULL + SWA + MAMBA at boundary.
  EXPECT_EQ(prefix(), 3u);
  reporter.Report({2}, 1, 16, 1);  // Replace mask after SWA eviction.
  EXPECT_EQ(prefix(), 0u);
  reporter.Report({2}, 1, 16, 3);
  EXPECT_EQ(prefix(), 3u);
  reporter.Revoke({2});
  EXPECT_EQ(prefix(), 0u);
  reporter.Report({2}, 1, 16, 3);
  EXPECT_EQ(prefix(), 3u);
  reporter.Clear();
  EXPECT_EQ(prefix(), 0u);
}
}  // namespace
}  // namespace mori::umbp
