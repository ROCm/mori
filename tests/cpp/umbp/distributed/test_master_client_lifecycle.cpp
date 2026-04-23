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

// Tests for the destructor wall-time bound added by
// distributed-known-issues.md #7.  A fake UMBPMaster service that
// answers RegisterClient quickly but blocks Heartbeat / UnregisterClient
// stands in for an unreachable master.  ~MasterClient must return
// within (Heartbeat 3 s + UnregisterClient 3 s + slack) thanks to the
// per-RPC ClientContext deadlines on the shutdown path.

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "umbp.grpc.pb.h"
#include "umbp/distributed/config.h"
#include "umbp/distributed/master/master_client.h"

namespace mori::umbp {
namespace {

constexpr int kHeartbeatIntervalMs = 50;  // fast loop so heartbeat enters quickly
constexpr int kDestructorBudgetMs = 7000;

static uint16_t AllocPort() {
  static std::atomic<uint16_t> next{0};
  if (next.load() == 0) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    next.store(static_cast<uint16_t>(54000 + (std::rand() % 1500)));
  }
  return next.fetch_add(7);
}

// Fake master service: RegisterClient returns immediately so the client
// can proceed to start its heartbeat loop.  Heartbeat and UnregisterClient
// park on a CV until ReleaseAll() is called from TearDown — this both
// (a) lets the client-side deadlines be the only termination path during
// the test body, and (b) lets server_->Shutdown() complete promptly in
// TearDown without waiting for an uninterruptable sleep_for to finish.
class BlackholeMasterService final : public ::umbp::UMBPMaster::Service {
 public:
  explicit BlackholeMasterService(int heartbeat_interval_ms)
      : heartbeat_interval_ms_(heartbeat_interval_ms) {}

  grpc::Status RegisterClient(grpc::ServerContext*, const ::umbp::RegisterClientRequest*,
                              ::umbp::RegisterClientResponse* response) override {
    response->set_heartbeat_interval_ms(heartbeat_interval_ms_);
    return grpc::Status::OK;
  }

  grpc::Status Heartbeat(grpc::ServerContext* ctx, const ::umbp::HeartbeatRequest*,
                         ::umbp::HeartbeatResponse*) override {
    heartbeat_entered_.store(true);
    BlockUntilReleasedOrCancelled(ctx);
    return grpc::Status::OK;
  }

  grpc::Status UnregisterClient(grpc::ServerContext* ctx, const ::umbp::UnregisterClientRequest*,
                                ::umbp::UnregisterClientResponse*) override {
    unregister_entered_.store(true);
    BlockUntilReleasedOrCancelled(ctx);
    return grpc::Status::OK;
  }

  void ReleaseAll() {
    std::lock_guard<std::mutex> lock(mu_);
    released_ = true;
    cv_.notify_all();
  }

  bool HeartbeatEntered() const { return heartbeat_entered_.load(); }
  bool UnregisterEntered() const { return unregister_entered_.load(); }

 private:
  // Park the handler until ReleaseAll() flips the flag or gRPC cancels
  // the context (Server::Shutdown does this).  Polls the cancel flag at
  // 25 ms granularity since gRPC offers no portable wait-for-cancel.
  void BlockUntilReleasedOrCancelled(grpc::ServerContext* ctx) {
    std::unique_lock<std::mutex> lock(mu_);
    while (!released_ && !ctx->IsCancelled()) {
      cv_.wait_for(lock, std::chrono::milliseconds(25));
    }
  }

  int heartbeat_interval_ms_;
  std::atomic<bool> heartbeat_entered_{false};
  std::atomic<bool> unregister_entered_{false};
  std::mutex mu_;
  std::condition_variable cv_;
  bool released_ = false;
};

class MasterClientLifecycleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    port_ = AllocPort();
    address_ = "127.0.0.1:" + std::to_string(port_);
    service_ = std::make_unique<BlackholeMasterService>(kHeartbeatIntervalMs);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address_, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    ASSERT_NE(server_, nullptr);
  }

  void TearDown() override {
    if (service_) service_->ReleaseAll();
    if (server_) {
      server_->Shutdown(std::chrono::system_clock::now() + std::chrono::milliseconds(500));
      server_->Wait();
    }
  }

  uint16_t port_ = 0;
  std::string address_;
  std::unique_ptr<BlackholeMasterService> service_;
  std::unique_ptr<grpc::Server> server_;
};

// Master accepts RegisterClient then black-holes Heartbeat and
// UnregisterClient.  ~MasterClient must finish within the documented
// 6 s budget (3 s Heartbeat + 3 s Unregister) plus a small slack.
TEST_F(MasterClientLifecycleTest, DestructorBoundedWhenMasterUnresponsive) {
  UMBPMasterClientConfig cfg;
  cfg.node_id = "lifecycle-test-node";
  cfg.node_address = "127.0.0.1";
  cfg.master_address = address_;
  cfg.auto_heartbeat = true;

  auto client = std::make_unique<MasterClient>(cfg);
  std::map<TierType, TierCapacity> caps;
  caps[TierType::DRAM] = {1 << 20, 1 << 20};
  auto status = client->RegisterSelf(caps);
  ASSERT_TRUE(status.ok()) << "RegisterSelf failed: " << status.error_message();

  // Wait until the heartbeat thread is actually blocked inside the RPC,
  // otherwise StopHeartbeat()'s notify_one() would unblock it cleanly
  // before the deadline path is exercised.  Up to 1 s with short polls.
  for (int i = 0; i < 200 && !service_->HeartbeatEntered(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_TRUE(service_->HeartbeatEntered())
      << "Heartbeat never entered the blocked handler; cannot validate the "
         "destructor's Heartbeat-deadline path.";

  const auto t0 = std::chrono::steady_clock::now();
  client.reset();  // triggers ~MasterClient
  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0)
          .count();

  EXPECT_LE(elapsed_ms, kDestructorBudgetMs)
      << "Destructor exceeded budget: took " << elapsed_ms << " ms (budget " << kDestructorBudgetMs
      << " ms).  Heartbeat or UnregisterClient deadline "
         "may have regressed.";
  EXPECT_TRUE(service_->UnregisterEntered());
}

}  // namespace
}  // namespace mori::umbp
