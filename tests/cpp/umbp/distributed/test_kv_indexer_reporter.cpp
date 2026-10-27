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

#include <atomic>
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
    const int fault = mode.load();
    if (fault == 2) {
      while (!context->IsCancelled()) std::this_thread::sleep_for(1ms);
    }
    return fault == 1 ? grpc::Status(grpc::StatusCode::UNAVAILABLE, "injected") : grpc::Status::OK;
  }
  std::mutex mutex;
  std::atomic<int> mode{0};
  std::vector<pb::ApplyExternalKvBatchRequest> requests;
};

TEST(KvIndexerReporter, WireContractValidationAndRpcErrors) {
  Receiver receiver;
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
  builder.RegisterService(&receiver);
  auto server = builder.BuildAndStart();
  ASSERT_NE(server, nullptr);
  const auto endpoint = "127.0.0.1:" + std::to_string(port);
  KvIndexerReporter reporter(endpoint, "worker", "http://worker:30000", 7, 32, 30);
  const int64_t root = -9007199254740993LL;
  reporter.Report({root}, std::nullopt, 16, 7);
  reporter.Report({42, 43}, root, 16, 3);
  reporter.Revoke({42});
  reporter.Clear();
  reporter.Report({}, std::nullopt, 16);
  reporter.Revoke({});
  EXPECT_THROW(KvIndexerReporter(endpoint, "worker", "address", 3), std::invalid_argument);
  EXPECT_THROW(reporter.Report({1}, {}, 0), std::invalid_argument);
  EXPECT_THROW(reporter.Report({1}, {}, 16, 2), std::invalid_argument);
  EXPECT_THROW(reporter.Report({1}, {}, 16, 9), std::invalid_argument);
  EXPECT_THROW(reporter.Report(std::vector<int64_t>(16385), {}, 16), std::invalid_argument);
  EXPECT_THROW(reporter.Revoke(std::vector<int64_t>(16385)), std::invalid_argument);
  receiver.mode = 1;
  EXPECT_THROW(reporter.Report({1}, {}, 16), std::runtime_error);
  receiver.mode = 2;
  const auto start = std::chrono::steady_clock::now();
  EXPECT_THROW(reporter.Revoke({1}), std::runtime_error);
  EXPECT_LT(std::chrono::steady_clock::now() - start, 2s);
  receiver.mode = 0;
  reporter.Clear();
  server->Shutdown();
  server->Wait();
  const auto& requests = receiver.requests;
  ASSERT_EQ(requests.size(), 7u);  // No retries, implicit clear, or empty reports.
  for (size_t i = 0; i < requests.size(); ++i) {
    EXPECT_EQ(requests[i].worker_id(), "worker");
    EXPECT_EQ(requests[i].worker_address(), "http://worker:30000");
    EXPECT_EQ(requests[i].seq(), i + 1);
    EXPECT_EQ(requests[i].cache_spec().components(), 7u);
    EXPECT_EQ(requests[i].cache_spec().swa_window_tokens(), 32u);
    ASSERT_EQ(requests[i].actions_size(), 1);
    EXPECT_EQ(requests[i].actions(0).tier(), pb::TIER_DRAM);
  }
  EXPECT_FALSE(requests[0].actions(0).has_parent_block_hash());
  EXPECT_EQ(requests[0].actions(0).hashes(0), root);
  const auto& chain = requests[1].actions(0);
  EXPECT_EQ(chain.type(), pb::ACTION_REPORT);
  EXPECT_EQ(chain.parent_block_hash(), root);
  ASSERT_EQ(chain.hashes_size(), 2);
  ASSERT_EQ(chain.component_masks_size(), 2);
  ASSERT_EQ(chain.block_sizes_size(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(chain.hashes(i), 42 + i);
    EXPECT_EQ(chain.component_masks(i), 3u);
    EXPECT_EQ(chain.block_sizes(i), 16u);
  }
  EXPECT_EQ(requests[2].actions(0).type(), pb::ACTION_REVOKE);
  EXPECT_EQ(requests[2].actions(0).hashes(0), 42);
  EXPECT_EQ(requests[3].actions(0).type(), pb::ACTION_CLEAR_ALL_AT_TIER);
  EXPECT_EQ(requests[6].actions(0).type(), pb::ACTION_CLEAR_ALL_AT_TIER);
}
}  // namespace
}  // namespace mori::umbp
