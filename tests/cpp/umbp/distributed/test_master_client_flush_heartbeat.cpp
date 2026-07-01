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

// Tests for MasterClient::FlushHeartbeat():
//
//  1. FlushHeartbeat() wakes the sleeping heartbeat thread and fires a
//     heartbeat immediately, well before the configured interval elapses.
//
//  2. FlushHeartbeat() called before StartHeartbeat() is a safe no-op —
//     no crash, no heartbeat sent.
//
//  3. FlushHeartbeat() called while a heartbeat RPC is in-flight sets
//     flush_requested_, so the next loop iteration fires immediately once
//     the in-flight RPC completes (rather than sleeping the full interval).

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <ctime>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "umbp.grpc.pb.h"
#include "umbp/distributed/master/master_client.h"

namespace mori::umbp {
namespace {

static uint16_t AllocPort() {
  static std::atomic<uint16_t> next{0};
  if (next.load() == 0) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    next.store(static_cast<uint16_t>(55500 + (std::rand() % 1500)));
  }
  return next.fetch_add(7);
}

// --------------------------------------------------------------------------
// Fake master: records heartbeat arrival times; optionally blocks each
// Heartbeat RPC until ReleaseAll() is called.
// --------------------------------------------------------------------------
class RecordingMasterService final : public ::umbp::UMBPMaster::Service {
 public:
  explicit RecordingMasterService(int interval_ms, bool block_heartbeats = false)
      : interval_ms_(interval_ms), block_heartbeats_(block_heartbeats) {}

  grpc::Status RegisterClient(grpc::ServerContext*, const ::umbp::RegisterClientRequest*,
                              ::umbp::RegisterClientResponse* resp) override {
    resp->set_heartbeat_interval_ms(interval_ms_);
    return grpc::Status::OK;
  }

  grpc::Status Heartbeat(grpc::ServerContext* ctx, const ::umbp::HeartbeatRequest*,
                         ::umbp::HeartbeatResponse* resp) override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      ++count_;
      last_time_ = std::chrono::steady_clock::now();
      entered_cv_.notify_all();
    }
    if (block_heartbeats_) {
      std::unique_lock<std::mutex> lock(mu_);
      // Poll cancel flag at 25 ms so server shutdown isn't stuck.
      release_cv_.wait_for(lock, std::chrono::milliseconds(25),
                           [&] { return released_ || ctx->IsCancelled(); });
      while (!released_ && !ctx->IsCancelled()) {
        release_cv_.wait_for(lock, std::chrono::milliseconds(25),
                             [&] { return released_ || ctx->IsCancelled(); });
      }
    }
    resp->set_acked_seq(0);
    return grpc::Status::OK;
  }

  grpc::Status UnregisterClient(grpc::ServerContext*, const ::umbp::UnregisterClientRequest*,
                                ::umbp::UnregisterClientResponse*) override {
    return grpc::Status::OK;
  }

  void ReleaseAll() {
    std::lock_guard<std::mutex> lock(mu_);
    released_ = true;
    release_cv_.notify_all();
  }

  // Block until at least `target` heartbeats have been received, or `timeout` elapses.
  bool WaitForCount(int target, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mu_);
    return entered_cv_.wait_for(lock, timeout, [&] { return count_ >= target; });
  }

  // Block until a heartbeat that arrived strictly after `since` is recorded.
  bool WaitForHeartbeatAfter(std::chrono::steady_clock::time_point since,
                             std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mu_);
    return entered_cv_.wait_for(lock, timeout, [&] { return count_ > 0 && last_time_ > since; });
  }

  int Count() {
    std::lock_guard<std::mutex> lock(mu_);
    return count_;
  }

  std::chrono::steady_clock::time_point LastTime() {
    std::lock_guard<std::mutex> lock(mu_);
    return last_time_;
  }

 private:
  const int interval_ms_;
  const bool block_heartbeats_;

  std::mutex mu_;
  std::condition_variable entered_cv_;
  std::condition_variable release_cv_;
  int count_ = 0;
  std::chrono::steady_clock::time_point last_time_;
  bool released_ = false;
};

// --------------------------------------------------------------------------
// Test fixture
// --------------------------------------------------------------------------
class FlushHeartbeatTest : public ::testing::Test {
 protected:
  void BuildServer(int interval_ms, bool block = false) {
    service_ = std::make_unique<RecordingMasterService>(interval_ms, block);
    uint16_t port = AllocPort();
    address_ = "127.0.0.1:" + std::to_string(port);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address_, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);
  }

  std::unique_ptr<MasterClient> MakeRegisteredClient() {
    UMBPMasterClientConfig cfg;
    cfg.node_id = "flush-hb-test-node";
    cfg.node_address = "127.0.0.1";
    cfg.master_address = address_;
    auto client = std::make_unique<MasterClient>(cfg);
    std::map<TierType, TierCapacity> caps;
    caps[TierType::DRAM] = {1u << 20, 1u << 20};
    auto status = client->RegisterSelf(caps);
    EXPECT_TRUE(status.ok()) << "RegisterSelf failed: " << status.error_message();
    return client;
  }

  void TearDown() override {
    if (service_) service_->ReleaseAll();
    if (server_) {
      server_->Shutdown(std::chrono::system_clock::now() + std::chrono::milliseconds(500));
      server_->Wait();
    }
  }

  std::string address_;
  std::unique_ptr<RecordingMasterService> service_;
  std::unique_ptr<grpc::Server> server_;
};

// --------------------------------------------------------------------------
// Test 1: FlushHeartbeat wakes the sleeping heartbeat thread immediately.
//
// With a 10-second interval the first heartbeat would not arrive for ~10s.
// FlushHeartbeat() must deliver it within 500 ms.
// --------------------------------------------------------------------------
TEST_F(FlushHeartbeatTest, WakesHeartbeatThreadImmediately) {
  constexpr int kLongIntervalMs = 10'000;
  constexpr int kFlushDeadlineMs = 500;

  ASSERT_NO_FATAL_FAILURE(BuildServer(kLongIntervalMs));
  auto client = MakeRegisteredClient();

  client->StartHeartbeat();
  ASSERT_EQ(service_->Count(), 0) << "Unexpected heartbeat before FlushHeartbeat()";

  auto t0 = std::chrono::steady_clock::now();
  client->FlushHeartbeat();

  bool fired = service_->WaitForHeartbeatAfter(t0, std::chrono::milliseconds(kFlushDeadlineMs));
  ASSERT_TRUE(fired) << "Heartbeat did not arrive within " << kFlushDeadlineMs
                     << " ms after FlushHeartbeat()";

  auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(service_->LastTime() - t0).count();
  EXPECT_LT(elapsed_ms, kFlushDeadlineMs)
      << "Heartbeat arrived after " << elapsed_ms << " ms (budget " << kFlushDeadlineMs << " ms)";
}

// --------------------------------------------------------------------------
// Test 2: FlushHeartbeat() before StartHeartbeat() is a safe no-op.
// --------------------------------------------------------------------------
TEST_F(FlushHeartbeatTest, NoOpBeforeStart) {
  ASSERT_NO_FATAL_FAILURE(BuildServer(5000));

  // Case A: before RegisterSelf and StartHeartbeat.
  {
    UMBPMasterClientConfig cfg;
    cfg.node_id = "flush-noop-node";
    cfg.node_address = "127.0.0.1";
    cfg.master_address = address_;
    MasterClient client(cfg);
    EXPECT_NO_FATAL_FAILURE(client.FlushHeartbeat());
  }

  // Case B: after RegisterSelf but before StartHeartbeat.
  {
    auto client = MakeRegisteredClient();
    EXPECT_NO_FATAL_FAILURE(client->FlushHeartbeat());
    // Destructor runs without starting the heartbeat thread — should be clean.
  }

  EXPECT_EQ(service_->Count(), 0) << "No heartbeat should have been sent";
}

// --------------------------------------------------------------------------
// Test 3: FlushHeartbeat() called while a heartbeat RPC is in-flight.
//
// flush_requested_ persists past the in-flight send; the next wait_for
// iteration sees it as true and fires immediately — gap from RPC release
// to second heartbeat must be well under the 10-second interval.
// --------------------------------------------------------------------------
TEST_F(FlushHeartbeatTest, FlushWhileRPCInFlightFiresNextTickImmediately) {
  constexpr int kLongIntervalMs = 10'000;
  constexpr int kDeadlineMs = 1000;

  ASSERT_NO_FATAL_FAILURE(BuildServer(kLongIntervalMs, /*block=*/true));
  auto client = MakeRegisteredClient();
  client->StartHeartbeat();

  // Kick the first heartbeat and wait for it to enter (and block in) the RPC.
  client->FlushHeartbeat();
  ASSERT_TRUE(service_->WaitForCount(1, std::chrono::milliseconds(500)))
      << "First heartbeat never reached the server";

  // While the RPC is still blocked, request a second flush.
  client->FlushHeartbeat();

  // Release the blocked RPC and measure how quickly the second heartbeat arrives.
  auto t_release = std::chrono::steady_clock::now();
  service_->ReleaseAll();

  bool got_second = service_->WaitForCount(2, std::chrono::milliseconds(kDeadlineMs));
  ASSERT_TRUE(got_second) << "Second heartbeat did not arrive within " << kDeadlineMs
                          << " ms after releasing the blocked RPC";

  auto gap_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(service_->LastTime() - t_release)
          .count();
  EXPECT_LT(gap_ms, kDeadlineMs)
      << "Second heartbeat arrived " << gap_ms << " ms after release (budget " << kDeadlineMs
      << " ms); flush_requested_ may not have been preserved across the in-flight RPC";
}

}  // namespace
}  // namespace mori::umbp
